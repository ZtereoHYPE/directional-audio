#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused)]

use crate::audio_engine::GpuData;
use ash::vk::{Buffer, BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, DependencyFlags, Extent3D, Fence, FenceCreateInfo, Image, ImageAspectFlags, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange, PhysicalDevice, PipelineStageFlags, Queue, SharingMode, SubmitInfo, QUEUE_FAMILY_IGNORED, WHOLE_SIZE};
use ash::{Device, Instance};
use std::array::from_ref;
use std::u64::MAX;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateInfo};

const STAGING_BUFFER_SIZE: u64 = 256 * 1024 * 1024; // 128 MB

// Util object to one-time upload data to a buffer using a staging buffer
pub(crate) struct BufferUploader {
    allocator: Allocator,
    staging_buffer: Buffer,
    staging_memory: Allocation,
    staging_map: *mut u8,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    fence: Fence,
}

impl BufferUploader {
    // todo: make this work for different queues! (inspect queue transfers)
    pub(crate) fn new(instance: &Instance, device: &Device, gpu: &PhysicalDevice, compute_queue_idx: u32) -> Self {
        let allocator = {
            let allocator_create_info = AllocatorCreateInfo::new(
                instance,
                device,
                *gpu
            );

            unsafe {
                Allocator::new(allocator_create_info)
                    .expect("Failed to create memory allocator")
            }
        };

        let command_pool = {
            let pool_create_info = CommandPoolCreateInfo::default()
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_queue_idx);

            unsafe {
                device
                    .create_command_pool(&pool_create_info, None)
                    .expect("Failed to create command pool")
            }
        };

        let command_buffer = {
            let command_buffer_info = CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(1)
                .level(CommandBufferLevel::PRIMARY);

            unsafe{
                device
                    .allocate_command_buffers(&command_buffer_info)
                    .expect("Failed to allocate command buffers")[0]
            }
        };

        let (staging_buffer, staging_memory, staging_map) = {
            let buffer_info = BufferCreateInfo::default()
                .size(STAGING_BUFFER_SIZE)
                .usage(BufferUsageFlags::TRANSFER_SRC)
                .queue_family_indices(from_ref(&compute_queue_idx))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::Auto,
                flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            // Create a UBO per stage
            unsafe {
                let (buffer, mut memory) = allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer");

                let map = allocator
                    .map_memory(&mut memory)
                    .expect("Failed to map memory");

                (buffer, memory, map)
            }
        };

        let fence = unsafe {
            device
                .create_fence(&FenceCreateInfo::default(), None)
                .expect("failed to create fence")
        };

        Self {
            allocator,
            staging_buffer,
            staging_memory,
            staging_map,
            command_pool,
            command_buffer,
            fence
        }
    }

    pub(crate) unsafe fn upload_buffer_onetime<T: GpuData>(&mut self, device: &Device, queue: Queue, src: T, dst: &mut Buffer) {
        let size = src.size() as u64;

        self.prepare_onetime(device, src);

        // perform copy
        let region = BufferCopy::default().size(size);
        device.cmd_copy_buffer(self.command_buffer, self.staging_buffer, *dst, from_ref(&region));

        // submit
        device.end_command_buffer(self.command_buffer);

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&self.command_buffer));

        device
            .queue_submit(queue, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");

        // wait for fence
        device
            .wait_for_fences(from_ref(&self.fence), true, MAX)
            .expect("Failed to wait for fences");
    }

    // warning: this also transitions the image layout
    pub(crate) unsafe fn upload_image_onetime<T: GpuData>(&mut self, device: &Device, queue: Queue, src: T, dst: &mut Image, dst_layout: ImageLayout, dst_extent: Extent3D) {
        let size = src.size() as u64;

        self.prepare_onetime(device, src);

        // transition the image's layout to the required one
        let subresource = ImageSubresourceRange::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1);

        let mut image_barrier = ImageMemoryBarrier::default()
            .old_layout(ImageLayout::UNDEFINED)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .image(*dst)
            .subresource_range(subresource);

        device.cmd_pipeline_barrier(
            self.command_buffer,
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
            DependencyFlags::empty(),
            &[],
            &[],
            from_ref(&image_barrier)
        );

        // perform copy
        let subresource = ImageSubresourceLayers::default()
            .aspect_mask(ImageAspectFlags::COLOR)
            .layer_count(1);

        let region = BufferImageCopy::default()
            .image_extent(dst_extent)
            .image_subresource(subresource);

        device.cmd_copy_buffer_to_image(self.command_buffer, self.staging_buffer, *dst, ImageLayout::TRANSFER_DST_OPTIMAL, from_ref(&region));

        image_barrier = image_barrier
            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(dst_layout);

        device.cmd_pipeline_barrier(
            self.command_buffer,
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::BOTTOM_OF_PIPE,
            DependencyFlags::empty(),
            &[],
            &[],
            from_ref(&image_barrier)
        );

        // submit
        device
            .end_command_buffer(self.command_buffer)
            .expect("Failed to end command buffer");

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&self.command_buffer));

        device
            .queue_submit(queue, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");

        // wait for fence
        device
            .wait_for_fences(from_ref(&self.fence), true, MAX)
            .expect("Failed to wait for fence");
    }

    unsafe fn prepare_onetime<T: GpuData>(&self, device: &Device, src: T) {
        let size = src.size() as u64;

        if size > STAGING_BUFFER_SIZE {
            todo!("Uploading data bigger than the staging buffer isn't supported yet");
        }

        device
            .reset_fences(from_ref(&self.fence))
            .expect("Failed to reset fences");

        // copy data to the staging buffer
        src.serialize(self.staging_map);
        self.allocator
            .flush_allocation(&self.staging_memory, 0, WHOLE_SIZE)
            .expect("Failed to flush allocation");

        // reset and begin command buffer
        device
            .reset_command_buffer(self.command_buffer, CommandBufferResetFlags::RELEASE_RESOURCES) // todo: release resources?
            .expect("Failed to reset command buffer");

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(self.command_buffer, &begin_info)
            .expect("Failed to begin command buffer recording");
    }
    
    // todo: buffer clearer (vkcmdclearbuffer)
}