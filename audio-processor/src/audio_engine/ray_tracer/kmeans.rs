use crate::audio_engine::gpu_structures::MAX_SOURCES;
use crate::audio_engine::ray_tracer::rays::SPHERE_POINTS;
use crate::audio_engine::{read_file_words, GpuData};
use ash::vk::{AccessFlags, Buffer, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use crevice::std430::{Std430, Vec3};
use std::array::from_ref;

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
    
    queue: Queue
}

impl KMeansModule {
    pub(super) fn new(
        device: Device,
        descriptor_pool: DescriptorPool,
        rt_output_buffer: Buffer,
        centroids_buffer: Buffer,
        neighbours_buffer: Buffer,
        instance_buffer: Buffer,
        queue: Queue
    ) -> Self {
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
                    .buffer(rt_output_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(centroids_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(neighbours_buffer)
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
                    .buffer(rt_output_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(centroids_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(neighbours_buffer)
                    .range(WHOLE_SIZE),
                DescriptorBufferInfo::default()
                    .buffer(instance_buffer)
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
            closest_pipeline,
            closest_pipeline_layout,
            closest_descriptor_set,
            closest_descriptor_set_layout,
            sum_pipeline,
            sum_pipeline_layout,
            sum_descriptor_set,
            sum_descriptor_set_layout,
            queue,
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

            self.device.cmd_dispatch(*command_buffer, SPHERE_POINTS as u32 / 64, source_amt, 1);

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

            // todo: make sure all dispatches have the minimum required to cover everything (they round up)
            //       maybe could be extracted to a method
            let x_workgroups = (source_amt + 63) / 64; // rounds up
            self.device.cmd_dispatch(*command_buffer, x_workgroups, 1, 1);

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

    pub(crate) fn initial_centroids() -> [Vec3; CENTROIDS] {
        [
            Vec3 {x: 1.0, y: 0.0, z: 0.0},
            Vec3 {x: -1.0, y: 0.0, z: 0.0},
            Vec3 {x: 0.0, y: 1.0, z: 0.0},
            Vec3 {x: 0.0, y: -1.0, z: 0.0},
            Vec3 {x: 0.0, y: 0.0, z: 1.0},
            Vec3 {x: 0.0, y: 0.0, z: -1.0},
            Vec3 {x: 2.0, y: 0.0, z: 2.0},
            Vec3 {x: -2.0, y: 0.0, z: 2.0},
        ]
    }
}

/// Buffer used to contain the centroid locations across iterations of the kmeans algorithm
pub(super) struct CentroidBuffer {
    centroids: [[Vec3; CENTROIDS]; MAX_SOURCES],
}

impl GpuData for CentroidBuffer {
    unsafe fn serialize(&self, dst: *mut u8) {
        std::ptr::copy_nonoverlapping(
            (self as *const CentroidBuffer).cast(),
            dst,
            self.size()
        );
    }

    fn size(&self) -> usize {
        Self::max_size()
    }
}

impl CentroidBuffer {
    pub(crate) fn from_initial_centroids(initial_centroids: [Vec3; CENTROIDS]) -> Self {
        Self {
            centroids: [initial_centroids; MAX_SOURCES]
        }
    }

    pub(crate) fn max_size() -> usize {
        size_of::<Self>() // the vec3s are aligned like vec4s
    }
}

/// Buffer used to contain the neighbouring cluster to each point
pub(super) struct NeighboursBuffer();
impl NeighboursBuffer {
    pub(crate) fn max_size() -> usize {
        SPHERE_POINTS * MAX_SOURCES * size_of::<u32>()
    }
}