use std::cmp::PartialEq;
use std::collections::HashMap;
use std::rc::Rc;
use std::slice::from_ref;
use std::time::Instant;
use ash::Device;
use ash::vk::{AccessFlags, Buffer, BufferUsageFlags, CommandBuffer, DependencyFlags, MemoryBarrier, PipelineStageFlags};
use bytemuck::Zeroable;
use glam::{dvec3, vec3, DVec2, DVec3, Vec3};
use vk_mem::Allocator;
use approx_dbscan::approximate_dbscan;
use approx_dbscan::clusterable::{Clusterable, Point};
use crate::audio_engine::{AudioInstance, InstanceBufferData};
use crate::audio_engine::gpu_constants::{MAX_INSTANCES, MAX_SOURCES};
use crate::audio_engine::ray_tracer::{Output, RtOutputBufferData};
use crate::audio_engine::ray_tracer::cluster::ClusterModulePhase::{Clear, ClusteredData, LocalRtData};
use crate::audio_engine::ray_tracer::debug::SPHERE_POINTS;
use crate::vulkan::buffer::{InlineBufferData, LocalVulkanBuffer, VulkanBuffer};

enum ClusterModulePhase {
    Clear,
    LocalRtData,
    ClusteredData,
}

// todo pub(super)
pub(crate) struct ClusterModule {
    device: Device,
    local_rt_output_buffer: LocalVulkanBuffer<RtOutputBufferData>,
    pub local_instance_buffer: LocalVulkanBuffer<InstanceBufferData>,
    rt_output_handle: Buffer,

    phase: ClusterModulePhase,
    last_clusters: Vec<AudioInstance>
}

impl ClusterModule {
    pub(super) fn new(
        allocator: Rc<Allocator>,
        device: Device,
        rt_output_buffer: &VulkanBuffer<RtOutputBufferData>,
    ) -> Self {
        let local_rt_output_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST,
            allocator.clone()
        );

        let local_instance_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC,
            allocator.clone()
        );

        Self {
            device,
            local_rt_output_buffer,
            local_instance_buffer,
            rt_output_handle: rt_output_buffer.handle(),
            phase: Clear,
            last_clusters: vec![],
        }
    }

    pub(super) fn copy_rt_output(&mut self, command_buffer: &mut CommandBuffer) {
        match self.phase {
            Clear => {}
            _ => panic!("The cluster module is not in the right phase")
        }

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any shader read caches

        unsafe {
            self.device.cmd_pipeline_barrier(
                *command_buffer,
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::TRANSFER, // ...before executing any transfer from now on
                DependencyFlags::empty(),
                from_ref(&memory_barrier),
                &[],
                &[]
            );

            self.device.cmd_copy_buffer(
                *command_buffer,
                self.rt_output_handle,
                self.local_rt_output_buffer.handle(),
                RtOutputBufferData::region()
            )
        }

        self.phase = LocalRtData;
    }

    // todo: move this off-thread (this class will kinda act as a monitor / abstraction to handle the thread stuff)
    //  you can have an is_done() method that checks for the phase, the state can be atomic or behind a mutex
    pub(super) unsafe fn cluster(&mut self) -> usize {
        match self.phase {
            LocalRtData => {}
            _ => panic!("The cluster module is not in the right phase")
        }

        let initial_time = Instant::now();

        let buffer_data = self.local_rt_output_buffer.buffer_data();

        // Split the different sources' hits, filtering to only the ones that reached the source
        let source_clusters = buffer_data.outputs
            .as_chunks_unchecked::<SPHERE_POINTS>() // split the various sources
            .iter()
            .map(|chunk|
                chunk.clone()
                    .into_iter()
                    .filter(|ray| ray.found_source) // filter each chunk to only valid outputs
                    .map(|o| AudioInstance {
                        direction: o.direction,
                        distance: o.additional_distance,
                        index: o.source,
                    })
                    .collect::<Vec<_>>() // collect each chunk into a vector of audio instances
            )
            .enumerate()
            .filter(|(_, src)| !src.is_empty()) // remove empty sources
            .map(|(idx, src)| (idx, approximate_dbscan(src, 2.5, 0.1, 3))) // cluster them
            .collect::<Vec<_>>();

        let cluster_time = Instant::now();

        // todo: this is very easily parallelizable
        let mut total_instances = vec![];
        for (source_idx, source) in source_clusters {
            #[derive(Zeroable)]
            struct InstanceData {
                direction: DVec3,
                distance: f64,
                cluster_size: usize,
            }

            // todo: influence in some way the strength based on the relative amount of instances in the cluster
            for cluster in source.iter().skip(1) {
                let cluster_size = cluster.len() as f32;
                let mut avg_direction = Vec3::ZERO;
                let mut avg_distance = 0.0;

                for source in cluster {
                    avg_direction += source.direction / cluster_size;
                    avg_distance += source.distance / cluster_size;
                }

                total_instances.push(AudioInstance {
                    direction: avg_direction,
                    distance: avg_distance,
                    index: source_idx as u32,
                })
            }
        }

        // todo: handle this gracefully
        if total_instances.len() > MAX_INSTANCES {
            panic!("There are way too many clusters!")
        }

        self.local_instance_buffer.buffer_data().copy_instances(&total_instances);
        self.local_instance_buffer.flush();

        self.phase = ClusteredData;
        self.last_clusters = total_instances;

        println!("Clustering time: {:?}, averaging time: {:?}", cluster_time - initial_time, Instant::now() - cluster_time);
        self.last_clusters.len()
    }

    pub(crate) unsafe fn upload_to_buffer(&mut self, command_buffer: &mut CommandBuffer, buffer: &mut VulkanBuffer<InstanceBufferData>) {
        match self.phase {
            ClusteredData => {}
            _ => panic!("The cluster module is not in the right phase")
        }

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.local_instance_buffer.handle(),
            buffer.handle(),
            InstanceBufferData::region()
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

        self.phase = Clear;
    }
    
    pub(super) fn get_clusters_debug(&self) -> &Vec<AudioInstance> {
        &self.last_clusters
    }
}


impl Clusterable<3> for AudioInstance {
    fn distance(&self, other: &Self) -> f64 {
        self.direction.distance(other.direction) as f64
    }

    fn point(&self) -> Point<3> {
        [
            self.direction.x as f64,
            self.direction.y as f64,
            self.direction.z as f64,
        ]
    }

    fn nth(&self, idx: usize) -> f64 {
        match idx {
            0 => self.direction.x as f64,
            1 => self.direction.y as f64,
            2 => self.direction.z as f64,
            _ => unreachable!()
        }
    }
}

fn dvec3_to_vec3(dvec: DVec3) -> Vec3 {
    Vec3 {
        x: dvec.x as f32,
        y: dvec.y as f32,
        z: dvec.z as f32,
    }
}
