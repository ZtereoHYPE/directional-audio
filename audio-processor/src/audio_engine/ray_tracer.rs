#![allow(unsafe_op_in_unsafe_fn)]

use crate::audio_engine::buffer_initializer::BufferInitializer;
use crate::audio_engine::gpu_structures::{InstanceBuffer, OutputBuffer, SourcesBuffer, MAX_SOURCES};
use crate::audio_engine::ray_tracer::kmeans::{CentroidBuffer, KMeansModule, NeighboursBuffer};
use crate::audio_engine::ray_tracer::rays::RayModule;
use crate::audio_engine::GpuData;
use crate::scene::Scene;
use ash::vk::{Buffer, BufferCreateInfo, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, Fence, FenceCreateInfo, PhysicalDevice, Queue, SharingMode, SubmitInfo};
use ash::{Device, Instance};
use std::array::from_ref;
use std::rc::Rc;
use vk_mem::{Alloc, Allocator, AllocatorCreateInfo};

pub(crate) mod kmeans;
pub(crate) mod rays;

pub(crate) struct RayTracer {
    scene: Scene,
    device: Device,
    buffer_allocator: Rc<Allocator>,
    async_queue: (Queue, u32),
    command_buffer: CommandBuffer,
    instance_buffer: Buffer,
    fence: Fence,

    ray_module: RayModule,
    kmeans_module: KMeansModule,
}

impl RayTracer {
    pub(super) fn new(
        scene: Scene,
        instance: &Instance,
        gpu: &PhysicalDevice,
        device: Device,
        async_queue: (Queue, u32),
        buffer_uploader: &mut BufferInitializer
    ) -> Self {
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

        let (sources_buffer, sources_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(SourcesBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (bvh_buffer, bvh_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(scene.mesh.bvh.size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (triangle_buffer, triangle_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(scene.mesh.triangles.size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (rt_output_buffer, rt_output_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(OutputBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (neighbours_buffer, neighbours_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(NeighboursBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let (mut centroid_buffer, centroid_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(CentroidBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };
        unsafe {
            let buffer_data = CentroidBuffer::from_initial_centroids(KMeansModule::initial_centroids());
            buffer_uploader.upload_buffer_onetime(&device, async_queue.0, buffer_data, &mut centroid_buffer);
        }

        let (instance_buffer, centroid_buffer_mem) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(InstanceBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER) // todo: remove TRANSFER_SRC
                .queue_family_indices(from_ref(&async_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let fence = unsafe {
            device
                .create_fence(&FenceCreateInfo::default(), None)
                .expect("failed to create fence")
        };

        let ray_module = RayModule::new(
            device.clone(),
            descriptor_pool,
            sources_buffer,
            bvh_buffer,
            triangle_buffer,
            rt_output_buffer,
            async_queue.0
        );

        let kmeans_module = KMeansModule::new(
            device.clone(),
            descriptor_pool,
            rt_output_buffer,
            centroid_buffer,
            neighbours_buffer,
            instance_buffer,
            async_queue.0,
        );

        Self {
            scene,
            device,
            buffer_allocator,
            async_queue,
            command_buffer,
            instance_buffer,
            fence,

            ray_module,
            kmeans_module,
        }
    }

    pub(super) unsafe fn trace_rays(&mut self) {
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

        // Trace the rays
        self.ray_module.shoot_rays(&mut self.command_buffer, MAX_SOURCES as u32, self.scene.listener_pos());

        // Cluster the rays with kmeans
        self.kmeans_module.cluster_rays(&mut self.command_buffer, MAX_SOURCES as u32);

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
    }

    pub(super) fn get_instance_buffer(&self) -> Buffer {
        self.instance_buffer
    }
}