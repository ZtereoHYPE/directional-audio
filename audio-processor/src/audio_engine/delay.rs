use crate::audio_engine::gpu_constants::{GPU_WINDOW_SIZE, MAX_DELAY_FRAMES, MAX_SOURCES};
use crate::audio_engine::{InstanceBufferData};
use crate::scene::source::FRAME_SIZE;
use crate::util::AsBytes;
use crate::vulkan::buffer::{InlineBufferData, VulkanBuffer};
use crate::vulkan::buffer_initializer::{BufferInitializer, InitMode};
use ash::vk::{AccessFlags, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SpecializationInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use glam::{Mat3, Mat3A, Vec2, Vec3};
use std::array::from_ref;
use std::rc::Rc;
use vk_mem::Allocator;
use crate::audio_engine::fft::FftBufferData;
use crate::vulkan::read_spirv_words;
use crate::vulkan::spec_constants::SpecConstantList;

pub(crate) struct DelayModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue,

    pub(super) delay_buffer: VulkanBuffer<DelayBufferData>,
}

impl DelayModule {
    pub unsafe fn new(
        allocator: Rc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        queue: (Queue, u32),
        descriptor_pool: DescriptorPool,
        instance_buffer: &VulkanBuffer<InstanceBufferData>,
        fft_starting_buffer: &VulkanBuffer<FftBufferData>,
    ) -> Self {
        let constants = SpecConstantList::new()
            .append(GPU_WINDOW_SIZE as u32) // window size
            .append(FRAME_SIZE as u32) // frame size
            .append((MAX_DELAY_FRAMES * FRAME_SIZE) as u32) // delay buffer size
            .append(0_u32) // pipelined frames
            .append(44100.0_f32) // sampling rate
            .build();
        
        let mut delay_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        initializer.init_buffer(&mut delay_buffer, InitMode::Zeroed, queue, &device);

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
                    .buffer(instance_buffer.handle())
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(delay_buffer.handle())
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(fft_starting_buffer.handle())
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
                .size(64);

            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_set_layout))
                .push_constant_ranges(from_ref(&push_constant_range));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_spirv_words("target/shaders/delay.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let specialization_info = SpecializationInfo::default()
                .map_entries(&constants.0)
                .data(&constants.1);

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

    pub(crate) unsafe fn apply_delay(&mut self, command_buffer: &mut CommandBuffer, frame_counter: usize, camera_delta: Vec3, camera_rotation: Mat3, instance_amt: usize) {
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

        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 0, Mat3A::from(camera_rotation).as_bytes());
        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 48, camera_delta.as_bytes());
        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 60, (frame_counter as u32).as_bytes());

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
}

#[repr(C)]
pub(crate) struct DelayBufferData {
    frames: [[[Vec2; FRAME_SIZE]; MAX_DELAY_FRAMES]; MAX_SOURCES]
}

impl InlineBufferData for DelayBufferData {}
