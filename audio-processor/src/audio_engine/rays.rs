use crate::audio_engine::gpu_constants::{MAX_SOURCES, SPHERE_POINTS};
use crate::scene::mesh::bvh::BvhBufferData;
use crate::scene::mesh::TriangleBufferData;
use crate::scene::source::AudioSource;
use crate::scene::Scene;
use crate::util::AsBytes;
use crate::vulkan::buffer::{BufferData, InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use crate::vulkan::buffer_initializer::{BufferInitializer, InitMode};
use ash::vk::{AccessFlags, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SpecializationInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use glam::{Vec3, Vec3A, Vec4};
use std::array::from_ref;
use std::rc::Rc;
use vk_mem::Allocator;
use crate::audio_engine::{RayTracerConstants, RtOutputBufferData};
use crate::vulkan::misc::read_spirv_words;

pub(super) struct RayModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    pub(super) sources_buffer: VulkanBuffer<SourceBufferData>,
    local_sources_buffer: LocalVulkanBuffer<SourceBufferData>,
    bvh_buffer: VulkanBuffer<BvhBufferData>,
    triangle_buffer: VulkanBuffer<TriangleBufferData>,
    ray_buffer: VulkanBuffer<RayBufferData>,
    pub(super) local_ray_buffer: LocalVulkanBuffer<RayBufferData>,
    pub(super) output_buffer: VulkanBuffer<RtOutputBufferData>,
    queue: Queue
}

impl RayModule {
    pub(super) fn new(
        allocator: Rc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        descriptor_pool: DescriptorPool,
        queue: (Queue, u32),
        scene: &Scene,
        constants: RayTracerConstants
    ) -> Self {
        let sources_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let local_sources_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC,
            allocator.clone()
        );

        let ray_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            allocator.clone()
        );

        let local_ray_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST,
            allocator.clone()
        );
        
        let mut bvh_buffer = VulkanBuffer::new_dynamic(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            scene.mesh.bvh.clone(),
            allocator.clone()
        );

        let mut triangle_buffer = VulkanBuffer::new_dynamic(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            scene.mesh.triangles.clone(),
            allocator.clone()
        );

        unsafe {
            initializer.init_buffer(&mut bvh_buffer, InitMode::Populated(Box::new(scene.mesh.bvh.clone())), queue, &device);
            initializer.init_buffer(&mut triangle_buffer, InitMode::Populated(Box::new(scene.mesh.triangles.clone())), queue, &device);
        }
        
        let output_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            allocator.clone()
        );

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
                DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),
                DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),
                DescriptorSetLayoutBinding::default()
                    .binding(4)
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
                    .buffer(bvh_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(triangle_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(output_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(ray_buffer.handle())
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
                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(3)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[3])),
                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(4)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[4])),
            ];

            device.update_descriptor_sets(&writes, &[]);
            (set, set_layout)
        };

        let (pipeline, pipeline_layout) = unsafe {
            let push_constant_range = PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .size(16);

            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_set_layout))
                .push_constant_ranges(from_ref(&push_constant_range));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_spirv_words("target/shaders/raytracing.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let specialization_entries =
                RayTracerConstants::get_entries(&[0, 1, 2, 3]); // select these constants

            let specialization_info = SpecializationInfo::default()
                .map_entries(&specialization_entries)
                .data(constants.to_slice());

            let stage_info = PipelineShaderStageCreateInfo::default()
                .stage(ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .specialization_info(&specialization_info)
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
            sources_buffer,
            local_sources_buffer,
            bvh_buffer,
            triangle_buffer,
            ray_buffer,
            local_ray_buffer,
            output_buffer,
            queue: queue.0,
        }
    }

    pub(super) unsafe fn stage_sources(&mut self, command_buffer: &mut CommandBuffer, sources: &Vec<AudioSource>) {
        // Stage the new coordinates
        self.local_sources_buffer.buffer_data().copy_coordinates(sources);
        self.local_sources_buffer.invalidate();

        // Copy from staging buffer
        self.device.cmd_copy_buffer(
            *command_buffer,
            self.local_sources_buffer.handle(),
            self.sources_buffer.handle(),
            SourceBufferData::region()
        );

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::TRANSFER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::TRANSFER, 
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        );
    }

    pub(super) unsafe fn shoot_rays(&mut self, command_buffer: &mut CommandBuffer, source_amt: u32, origin: Vec3, store_rays: bool) {
        if store_rays {
            self.device.cmd_fill_buffer(*command_buffer, self.ray_buffer.handle(), 0, WHOLE_SIZE, 0);
        }
        
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
            origin.as_bytes()
        );

        self.device.cmd_push_constants(
            *command_buffer,
            self.pipeline_layout,
            ShaderStageFlags::COMPUTE,
            12,
            (store_rays as u32).as_bytes()
        );

        self.device.cmd_dispatch(*command_buffer, SPHERE_POINTS as u32 / 64, source_amt, 1);

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

    pub(super) unsafe fn copy_ray_buffer(&mut self, command_buffer: &mut CommandBuffer) {
        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any shader write caches
            .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any transfer read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::COMPUTE_SHADER, // wait for all compute commands so far...
            PipelineStageFlags::TRANSFER, // ...before executing any transfer from now on
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        );

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.ray_buffer.handle(),
            self.local_ray_buffer.handle(),
            RayBufferData::region()
        );
    }
}

pub struct RayBufferData {
    pub rays: [Vec4; SPHERE_POINTS * 4 * MAX_SOURCES]
}

impl InlineBufferData for RayBufferData {}


pub(super) struct SourceBufferData {
    sources: [Vec3A; MAX_SOURCES]
}

impl SourceBufferData {
    pub(crate) fn copy_coordinates(&mut self, sources: &Vec<AudioSource>) {
        for (idx, source) in sources.iter().enumerate() {
            self.sources[idx] = source.coordinates.into();
        }
    }
}

impl InlineBufferData for SourceBufferData {}
