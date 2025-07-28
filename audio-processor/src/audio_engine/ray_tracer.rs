#![allow(unsafe_op_in_unsafe_fn)]

use crate::audio_engine::gpu_constants::{MAX_SOURCES, SPHERE_POINTS};
use crate::audio_engine::ray_tracer::debug::DebugRayModule;
use crate::audio_engine::ray_tracer::rays::{RayBufferData, RayModule};
use crate::audio_engine::{AudioInstance, InstanceBufferData};
use crate::scene::mesh::bvh::MAX_BVH_DEPTH;
use crate::scene::Scene;
use crate::vulkan::buffer::{InlineBufferData, VulkanBuffer};
use crate::vulkan::buffer_initializer::BufferInitializer;
use ash::prelude::VkResult;
use ash::vk::{CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, Fence, FenceCreateInfo, PhysicalDevice, Queue, SpecializationMapEntry, SubmitInfo};
use ash::{Device, Instance};
use glam::{Mat3, Vec3};
use std::array::from_ref;
use std::mem::transmute;
use std::rc::Rc;
use vk_mem::{Alloc, Allocator, AllocatorCreateInfo};
use crate::audio_engine::ray_tracer::cluster::ClusterModule;

pub(crate) mod rays;
pub(crate) mod cluster;
mod debug;

#[repr(C)]
#[derive(Copy, Clone)]
struct RayTracerConstants {
    point_amount: u32,
    source_amount: u32,
    max_bvh_depth: u32,
    max_bounces: u32,
    kmeans_centroids: u32,
}

impl RayTracerConstants {
    const SIZE: usize = size_of::<RayTracerConstants>();

    /// warning: this assumes that all fields are 4 in size
    fn get_entries(entries: &[u32]) -> Vec<SpecializationMapEntry> {
        entries
            .into_iter()
            .map(|&idx| SpecializationMapEntry::default()
                .constant_id(idx)
                .offset(4 * idx)
                .size(4)
            )
            .collect()
    }

    unsafe fn to_slice(&self) -> &[u8; Self::SIZE] {
        transmute(self)
    }
}

pub struct RtDebugData {
    pub origin: Vec3,
    pub rays: Box<RayBufferData>,
    pub instances: Vec<AudioInstance>
}

pub(crate) struct RayTracer {
    device: Device,
    buffer_allocator: Rc<Allocator>,
    async_queue: (Queue, u32),
    command_buffer: CommandBuffer,
    fence: Fence,

    ray_module: RayModule,
    pub cluster_module: ClusterModule,
    pub debug_module: DebugRayModule,

    last_rt_pos: Vec3,
}

impl RayTracer {
    pub(super) fn new(
        scene: &Scene,
        instance: &Instance,
        gpu: &PhysicalDevice,
        device: Device,
        async_queue: (Queue, u32),
        buffer_initializer: &mut BufferInitializer
    ) -> Self {
        let constants = RayTracerConstants {
            point_amount: SPHERE_POINTS as u32,
            source_amount: MAX_SOURCES as u32,
            max_bvh_depth: MAX_BVH_DEPTH as u32,
            max_bounces: 4,
            kmeans_centroids: 8,
        };

        let buffer_allocator = unsafe {
            let allocator_create_info = AllocatorCreateInfo::new(
                instance,
                &device,
                *gpu
            );

            Rc::new(
                Allocator::new(allocator_create_info)
                    .expect("Failed to create memory allocator")
            )
        };

        let async_command_pool = unsafe {
            let mut pool_create_info = CommandPoolCreateInfo::default()
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(async_queue.1);

            device
                .create_command_pool(&pool_create_info, None)
                .expect("Failed to create command pool")
        };

        let command_buffer = unsafe {
            let mut command_buffer_info = CommandBufferAllocateInfo::default()
                .command_pool(async_command_pool)
                .command_buffer_count(1)
                .level(CommandBufferLevel::PRIMARY);

            device
                .allocate_command_buffers(&command_buffer_info)
                .expect("Failed to allocate command buffers")[0]
        };

        let descriptor_pool = unsafe {
            let pool_sizes = [
                DescriptorPoolSize::default()
                    .ty(DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(4),
                DescriptorPoolSize::default()
                    .ty(DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(11),
            ];

            let pool_info = DescriptorPoolCreateInfo::default()
                .max_sets(32)
                .pool_sizes(&pool_sizes);

            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let fence = unsafe {
            device
                .create_fence(&FenceCreateInfo::default(), None)
                .expect("failed to create fence")
        };

        let ray_module = RayModule::new(
            buffer_allocator.clone(),
            buffer_initializer,
            device.clone(),
            descriptor_pool,
            async_queue.clone(),
            scene,
            constants
        );

        let cluster_module = ClusterModule::new(
            buffer_allocator.clone(),
            device.clone(),
            &ray_module.output_buffer
        );

        let debug_module = DebugRayModule::new(
            buffer_allocator.clone(),
            device.clone(),
            descriptor_pool,
            &ray_module.sources_buffer,
            async_queue.0
        );

        let last_rt_pos = scene.listener.location;

        Self {
            device,
            buffer_allocator,
            async_queue,
            command_buffer,
            fence,

            ray_module,
            cluster_module,
            debug_module,

            last_rt_pos,
        }
    }

    pub(super) unsafe fn trace_rays(&mut self, scene: &Scene, store_rays: bool) -> VkResult<()> {
        let last_rt_pos = scene.listener.location;
        self.device.reset_fences(from_ref(&self.fence))?;

        // Begin the command buffer
        self.device.reset_command_buffer(self.command_buffer, CommandBufferResetFlags::empty())?;

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(self.command_buffer, &begin_info)?;

        // Stage the sources
        self.ray_module.stage_sources(&mut self.command_buffer, &scene.sources);

        // Trace the rays
        self.ray_module.shoot_rays(&mut self.command_buffer, MAX_SOURCES as u32, last_rt_pos, store_rays);

        // Copy the result of shooting rays locally
        if store_rays {
            self.ray_module.copy_ray_buffer(&mut self.command_buffer);
        }

        // Copy the results of shooting rays to the cluster module
        self.cluster_module.copy_rt_output(&mut self.command_buffer);

        // Submit the command buffer
        self.device.end_command_buffer(self.command_buffer)?;

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&self.command_buffer));

        self.device.queue_submit(self.async_queue.0, &[submit_info], self.fence)?;

        // Wait for it to execute
        self.device.wait_for_fences(from_ref(&self.fence), true, u64::MAX)?;

        self.last_rt_pos = last_rt_pos; // update it only once it's fully done

        Ok(())
    }

    pub(super) unsafe fn cluster_rays(&mut self) -> usize {
        self.cluster_module.cluster()
    }

    pub(super) unsafe fn download_debug_data(&mut self) -> VkResult<RtDebugData> {
        Ok(RtDebugData {
            origin: self.last_rt_pos,
            rays: self.ray_module.local_ray_buffer.buffer_data().to_local_copy(),
            instances: self.cluster_module.get_clusters_debug().clone()
        })
    }

    pub(super) unsafe fn copy_sources_debug(&mut self, scene: &Scene) {
        let rt_pos = scene.listener.location;

        self.device
            .reset_fences(from_ref(&self.fence))
            .expect("Failed to reset raytracer fence!");

        // Begin the command buffer
        self.device
            .reset_command_buffer(self.command_buffer, CommandBufferResetFlags::empty())
            .expect("Failed to reset command buffer");

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device
            .begin_command_buffer(self.command_buffer, &begin_info)
            .expect("Failed to begin command buffer recording");

        // Stage the new coordinates
        self.ray_module.stage_sources(&mut self.command_buffer, &scene.sources);

        // Trace the rays
        self.debug_module.copy_sources(&mut self.command_buffer, MAX_SOURCES as u32, rt_pos);

        // Submit the command buffer
        self.device
            .end_command_buffer(self.command_buffer)
            .expect("Failed to end command buffer!");

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&self.command_buffer));

        self.device
            .queue_submit(self.async_queue.0, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");

        // Wait for it to execute
        self.device
            .wait_for_fences(from_ref(&self.fence), true, u64::MAX)
            .expect("Failed to wait for fence!");

        self.last_rt_pos = rt_pos; // update it only once it's fully done
    }

    pub(super) fn last_rt_pos(&self) -> Vec3 {
        self.last_rt_pos
    }
}

/// Ray Tracing output buffer
#[repr(align(16))]
#[derive(Clone)]
pub(crate) struct Output {
    direction: Vec3,
    additional_distance: f32,
    bounces: u32,
    source: u32,
    found_source: bool,
}

pub(crate) struct RtOutputBufferData {
    outputs: [Output; MAX_SOURCES * SPHERE_POINTS]
}

impl InlineBufferData for RtOutputBufferData {}