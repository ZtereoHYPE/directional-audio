use crate::audio_engine::ray_tracer::rays::SourceBufferData;
use crate::audio_engine::{read_file_words, InstanceBufferData};
use crate::util::{workgroup_div, AsBytes};
use crate::vulkan::buffer::{VulkanBuffer};
use ash::vk::{AccessFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use glam::{Mat3, Mat3A, Vec3, Vec3A};
use std::array::from_ref;

pub(crate) const SPHERE_POINTS: usize = 1024;

pub(super) struct DebugRayModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue
}

impl DebugRayModule {
    pub(super) fn new(
        device: Device,
        descriptor_pool: DescriptorPool,
        sources_buffer: &VulkanBuffer<SourceBufferData>, // this is NOT the right buffer man
        instance_buffer: &VulkanBuffer<InstanceBufferData>,
        queue: Queue
    ) -> Self {
        let (descriptor_set, descriptor_set_layout) = unsafe {
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
                    .buffer(sources_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(instance_buffer.handle())
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
            ];

            device.update_descriptor_sets(&writes, &[]);
            (set, set_layout)
        };

        let (pipeline, pipeline_layout) = unsafe {
            let push_constant_range = PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .size(12);

            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_set_layout))
                .push_constant_ranges(from_ref(&push_constant_range));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/rt_debug.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

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
            queue,
        }
    }

    pub(super) unsafe fn copy_sources(&mut self, command_buffer: &mut CommandBuffer, source_amt: u32, origin: Vec3) {
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

        self.device.cmd_push_constants(
            *command_buffer,
            self.pipeline_layout,
            ShaderStageFlags::COMPUTE,
            0,
            Vec3A::from(origin).as_bytes()
        );

        self.device.cmd_dispatch(*command_buffer, workgroup_div(source_amt, 64), 1, 1);

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any shader write caches
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
