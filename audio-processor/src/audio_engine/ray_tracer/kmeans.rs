use crate::audio_engine::gpu_constants::{MAX_SOURCES, SPHERE_POINTS};
use crate::audio_engine::ray_tracer::{RayTracerConstants, RtOutputBufferData};
use crate::audio_engine::{read_file_words, InstanceBufferData};
use crate::util::{workgroup_div, AsBytes};
use crate::vulkan::buffer::{BufferData, BufferOps, InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use crate::vulkan::buffer_initializer::{BufferInitializer, InitMode};
use ash::vk::{AccessFlags, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SpecializationInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use glam::{vec3a, Vec3A};
use std::array::from_ref;
use std::rc::Rc;
use vk_mem::Allocator;

const ITERATIONS: usize = 6;
pub(super) const CENTROIDS: usize = 8;

pub(super) struct KMeansModule {
    device: Device,
    
    closest_pipeline: Pipeline,
    closest_pipeline_layout: PipelineLayout,
    closest_descriptor_set: DescriptorSet,
    closest_descriptor_set_layout: DescriptorSetLayout,
    
    sum_pipeline: Pipeline,
    sum_pipeline_layout: PipelineLayout,
    sum_descriptor_set: DescriptorSet,
    sum_descriptor_set_layout: DescriptorSetLayout,

    neighbours_buffer: VulkanBuffer<NeighboursBufferData>,
    centroids_buffer: VulkanBuffer<CentroidBufferData>,
    pub(super) instance_buffer: VulkanBuffer<InstanceBufferData>,
    pub(super) local_instance_buffer: LocalVulkanBuffer<InstanceBufferData>,

    queue: Queue
}

impl KMeansModule {
    pub(super) fn new(
        allocator: Rc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        descriptor_pool: DescriptorPool,
        rt_output_buffer: &VulkanBuffer<RtOutputBufferData>,
        queue: (Queue, u32),
        constants: RayTracerConstants
    ) -> Self {
        let neighbours_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let mut centroids_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let init_data = Box::new(CentroidBufferData::from_initial());
        initializer.init_buffer(
            &mut centroids_buffer,
            InitMode::Populated(init_data),
            queue,
            &device
        );

        let instance_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let local_instance_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let (closest_descriptor_set, closest_descriptor_set_layout) = unsafe {
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
                    .buffer(rt_output_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(centroids_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(neighbours_buffer.handle())
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

        let (closest_pipeline, closest_pipeline_layout) = unsafe {
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&closest_descriptor_set_layout));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/kmeans_closest.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let specialization_entries =
                RayTracerConstants::get_entries(&[0, 1, 4]); // select these constants

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

        // todo: find a way to reduce the amount of descriptor set-related code
        let (sum_descriptor_set, sum_descriptor_set_layout) = unsafe {
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
                    .buffer(rt_output_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(centroids_buffer.handle())
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(neighbours_buffer.handle())
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
            ];

            device.update_descriptor_sets(&writes, &[]);
            (set, set_layout)
        };

        let (sum_pipeline, sum_pipeline_layout) = unsafe {
            let push_constant_range = PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .size(4);

            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&sum_descriptor_set_layout))
                .push_constant_ranges(from_ref(&push_constant_range));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/kmeans_sum.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let specialization_entries =
                RayTracerConstants::get_entries(&[0, 1, 4]); // select these constants

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
            closest_pipeline,
            closest_pipeline_layout,
            closest_descriptor_set,
            closest_descriptor_set_layout,
            sum_pipeline,
            sum_pipeline_layout,
            sum_descriptor_set,
            sum_descriptor_set_layout,
            neighbours_buffer,
            centroids_buffer,
            instance_buffer,
            local_instance_buffer,
            queue: queue.0,
        }
    }

    // do the kmeans
    pub(super) unsafe fn cluster_rays(&mut self, command_buffer: &mut CommandBuffer, source_amt: u32) {
        for iteration in 0..ITERATIONS {
            // Step 1: Find the closest centroid of each point
            self.device.cmd_bind_descriptor_sets(
                *command_buffer,
                PipelineBindPoint::COMPUTE,
                self.closest_pipeline_layout,
                0,
                from_ref(&self.closest_descriptor_set),
                &[]
            );

            self.device.cmd_bind_pipeline(
                *command_buffer,
                PipelineBindPoint::COMPUTE,
                self.closest_pipeline
            );

            self.device.cmd_dispatch(*command_buffer, SPHERE_POINTS as u32 / 64, source_amt as u32, 1);

            let memory_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
                .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

            self.device.cmd_pipeline_barrier(
                *command_buffer,
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute from now on
                DependencyFlags::empty(),
                from_ref(&memory_barrier),
                &[],
                &[]
            );

            // Step 2: Update the centroids
            self.device.cmd_bind_descriptor_sets(
                *command_buffer,
                PipelineBindPoint::COMPUTE,
                self.sum_pipeline_layout,
                0,
                from_ref(&self.sum_descriptor_set),
                &[]
            );

            self.device.cmd_bind_pipeline(
                *command_buffer,
                PipelineBindPoint::COMPUTE,
                self.sum_pipeline
            );

            let last = (iteration == (ITERATIONS-1)) as u32;
            self.device.cmd_push_constants(*command_buffer, self.sum_pipeline_layout, ShaderStageFlags::COMPUTE, 0, last.as_bytes());

            self.device.cmd_dispatch(*command_buffer, workgroup_div(source_amt, 64), 1, 1);

            self.device.cmd_pipeline_barrier(
                *command_buffer,
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute from now on
                DependencyFlags::empty(),
                from_ref(&memory_barrier),
                &[],
                &[]
            );
        }
    }

    pub(super) unsafe fn copy_instance_buffer(&mut self, command_buffer: &mut CommandBuffer) {
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
            self.instance_buffer.handle(),
            self.local_instance_buffer.handle(),
            InstanceBufferData::region()
        );
    }

    fn initial_centroids() -> [Vec3A; CENTROIDS] {
        [
            vec3a(1.0, 0.0, 0.0),
            vec3a(-1.0, 0.0, 0.0),
            vec3a(0.0, 1.0, 0.0),
            vec3a(0.0, -1.0, 0.0),
            vec3a(0.0, 0.0, 1.0),
            vec3a(0.0, 0.0, -1.0),
            vec3a(2.0, 0.0, 2.0),
            vec3a(-2.0, 0.0, 2.0),
        ]
    }
}

/// Buffer used to contain the centroid locations across iterations of the kmeans algorithm
pub(super) struct CentroidBufferData {
    centroids: [[Vec3A; CENTROIDS]; MAX_SOURCES],
}

impl InlineBufferData for CentroidBufferData {}

impl CentroidBufferData {
    pub(crate) fn from_initial() -> Self {
        Self {
            centroids: [KMeansModule::initial_centroids(); MAX_SOURCES]
        }
    }
}


/// Buffer used to contain the neighbouring cluster to each point
pub(super) struct NeighboursBufferData {
    neighbours: [u32; SPHERE_POINTS * MAX_SOURCES]
}

impl InlineBufferData for NeighboursBufferData {}

impl NeighboursBufferData {
    pub(crate) fn max_size() -> usize {
        size_of::<Self>()
    }
}