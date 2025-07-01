use crate::audio_engine::gpu_structures::{DelayBuffer, InstanceBuffer, GPU_WINDOW_SIZE};
use ash::vk::{AccessFlags, Buffer, BufferCreateInfo, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceSize, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, SpecializationMapEntry, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use crevice::std430::{Std430, Vec3};
use std::array::from_ref;
use std::mem::transmute;
use std::rc::Rc;
use vk_mem::{Alloc, AllocationCreateInfo, Allocator, MemoryUsage};
use crate::audio_engine::buffer_initializer::BufferInitializer;
use crate::audio_engine::read_file_words;
use crate::audio_engine::signal_processor::SignalProcessorConstants;

pub(crate) struct DelayModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue,

    delay_buffer: Buffer,
}

impl DelayModule {
    pub unsafe fn new(
        allocator: Rc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        queue: (Queue, u32),
        descriptor_pool: DescriptorPool,
        instance_buffer: Buffer,
        fft_starting_buffer: Buffer,
        constants: SignalProcessorConstants,
    ) -> Self {
        let (mut delay_buffer, delay_buffer_memory) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(DelayBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = AllocationCreateInfo {
                usage: MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        initializer.clear_buffer(&device, queue.0.clone(), &mut delay_buffer, DelayBuffer::max_size() as DeviceSize);

        let (descriptor_set, descriptor_set_layout) = {
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),
            ];

            let set_layout_info = DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            let set_layout = device
                .create_descriptor_set_layout(&set_layout_info, None)
                .expect("Failed to create descriptor set layout");

            // Allocate one set per stage
            let set_info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(from_ref(&set_layout));

            let set = device
                .allocate_descriptor_sets(&set_info)
                .expect("Failed to allocate descriptor sets")[0];

            let buffer_infos = [
                DescriptorBufferInfo::default()
                    .buffer(instance_buffer)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(delay_buffer)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(fft_starting_buffer)
                    .range(WHOLE_SIZE),
            ];

            // Write the descriptor sets
            let writes = [
                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[0])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[1])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(2)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[2])),
            ];

            device.update_descriptor_sets(&writes, &[]);
            (set, set_layout)
        };

        let (pipeline, pipeline_layout) = {
            let push_constant_range = PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .size(16);

            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_set_layout))
                .push_constant_ranges(from_ref(&push_constant_range));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/delay.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let specialization_entries =
                SignalProcessorConstants::get_entries(&[0, 1, 3, 4, 5]); // select these constants

            let specialization_info = SpecializationInfo::default()
                .map_entries(&specialization_entries)
                .data(constants.to_slice());

            let stage_info = PipelineShaderStageCreateInfo::default()
                .stage(ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");

            let pipeline_info = ComputePipelineCreateInfo::default()
                .layout(layout)
                .stage(stage_info);

            let pipeline = device
                .create_compute_pipelines(PipelineCache::null(), from_ref(&pipeline_info), None)
                .expect("Failed to create pipeline")[0];

            (pipeline, layout)
        };

        Self {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set,
            descriptor_set_layout,
            queue: queue.0,

            delay_buffer
        }
    }

    pub(crate) unsafe fn apply_delay(&mut self, command_buffer: &mut CommandBuffer, frame_counter: u32, camera_delta: Vec3, instance_amt: usize) {
        self.device.cmd_bind_descriptor_sets(
            *command_buffer,
            PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            from_ref(&self.descriptor_set),
            &[]
        );

        self.device.cmd_bind_pipeline(
            *command_buffer,
            PipelineBindPoint::COMPUTE,
            self.pipeline
        );

        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 0, camera_delta.as_bytes());
        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 12, frame_counter.as_bytes());

        let workgroups = (GPU_WINDOW_SIZE as u32 / 64, instance_amt as u32);
        self.device.cmd_dispatch(*command_buffer, workgroups.0, workgroups.1, 1);

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
            PipelineStageFlags::COMPUTE_SHADER, // ...before executing any transfers from now on
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        );
    }

    pub(super) fn delay_buffer(&self) -> Buffer {
        self.delay_buffer
    }
}
