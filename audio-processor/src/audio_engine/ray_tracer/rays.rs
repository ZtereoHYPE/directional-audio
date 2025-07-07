use crate::audio_engine::{read_file_words, GpuData};
use ash::vk::{AccessFlags, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceSize, MemoryBarrier, MemoryPropertyFlags, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use crevice::std430::{Std430, Vec3, Vec4};
use std::array::from_ref;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use vk_mem::{Alloc, AllocationCreateFlags, Allocator};
use crate::audio_engine::buffer_initializer::BufferInitializer;
use crate::audio_engine::gpu_structures::{FftBuffer, FromMemoryMap, OutputBuffer, SourcesBuffer, MAX_SOURCES};
use crate::audio_engine::ray_tracer::RayTracerConstants;
use crate::scene::Scene;
use crate::scene::source::AudioSource;

pub(crate) const SPHERE_POINTS: usize = 1024;

pub(super) struct RayModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    sources_buffer: Buffer,
    staging_sources_buffer: Buffer,
    staging_sources_map: *mut u8,
    bvh_buffer: Buffer,
    triangle_buffer: Buffer,
    rt_output_buffer: Buffer,
    ray_buffer: Buffer,
    staging_ray_buffer: Buffer,
    staging_ray_map: *mut u8,
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
        let (sources_buffer, sources_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(SourcesBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (staging_sources_buffer, staging_sources_mem, staging_sources_map) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(SourcesBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
                ..Default::default()
            };

            let (buffer, mut memory) = allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer");

            let map = allocator
                .map_memory(&mut memory)
                .expect("Failed to map memory");

            (buffer, memory, map)
        };

        let (ray_buffer, ray_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(RayBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (staging_ray_buffer, staging_ray_mem, staging_ray_map) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(RayBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
                required_flags: MemoryPropertyFlags::HOST_COHERENT, // TODO: remove, very slow
                ..Default::default()
            };

            let (buffer, mut memory) = allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer");

            let map = allocator
                .map_memory(&mut memory)
                .expect("Failed to map memory");

            (buffer, memory, map)
        };

        let (mut bvh_buffer, bvh_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(scene.mesh.bvh.size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (mut triangle_buffer, triangle_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(scene.mesh.triangles.size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (rt_output_buffer, rt_output_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(OutputBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

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
                    .buffer(sources_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(bvh_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(triangle_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(rt_output_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(ray_buffer)
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

            let code_words = read_file_words("target/shaders/raytracing.comp.spv");

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
            staging_sources_buffer,
            staging_sources_map,
            bvh_buffer,
            triangle_buffer,
            rt_output_buffer,
            ray_buffer,
            staging_ray_buffer,
            staging_ray_map,
            queue: queue.0,
        }
    }

    pub(super) unsafe fn stage_sources(&mut self, command_buffer: &mut CommandBuffer, sources: &Vec<AudioSource>) {
        // Stage the new coordinates
        SourcesBuffer::from_memory_map(self.staging_sources_map).copy_coordinates(sources);

        // todo: invalidate the allocation

        // Copy from staging buffer
        let region = BufferCopy::default().size(SourcesBuffer::max_size() as DeviceSize);
        self.device.cmd_copy_buffer(*command_buffer, self.staging_sources_buffer, self.sources_buffer, from_ref(&region));
    }

    pub(super) unsafe fn shoot_rays(&mut self, command_buffer: &mut CommandBuffer, source_amt: u32, origin: Vec3, store_rays: bool) {
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

    /// Warning: The buffer returned by this does not contain the data yet!
    /// The command buffer has to be submitted and fenced on first!
    pub(super) unsafe fn copy_ray_buffer(&mut self, command_buffer: &mut CommandBuffer) -> ManuallyDrop<Box<RayBuffer>> {
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

        let region = BufferCopy::default()
            .size(RayBuffer::max_size() as _);

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.ray_buffer,
            self.staging_ray_buffer,
            from_ref(&region)
        );

        RayBuffer::from_memory_map(self.staging_ray_map)
    }

    pub(super) fn output_buffer(&self) -> Buffer {
        self.rt_output_buffer
    }

    pub(super) unsafe fn local_ray_buffer(&self) -> Box<RayBuffer> {
        RayBuffer::from_memory_map(self.staging_ray_map).to_local_copy()
    }

    pub(super) fn sources_buffer(&self) -> Buffer {
        self.sources_buffer
    }
}

pub struct RayBuffer {
    pub rays: [Vec4; SPHERE_POINTS * 4 * MAX_SOURCES]
}

impl RayBuffer {
    pub(crate) fn max_size() -> usize {
        size_of::<Self>()
    }
}

impl FromMemoryMap for RayBuffer {}
