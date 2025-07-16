#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused)]

use crate::vulkan::buffer::{BufferData, BufferOps, VulkanBuffer};
use ash::vk::{Buffer, BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, DependencyFlags, DeviceSize, Extent3D, Fence, FenceCreateInfo, Image, ImageAspectFlags, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange, PhysicalDevice, PipelineStageFlags, Queue, SharingMode, SubmitInfo, QUEUE_FAMILY_IGNORED, WHOLE_SIZE};
use ash::{Device, Instance};
use std::array::from_ref;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateInfo};

pub(crate) enum InitMode<T: BufferData> {
    Zeroed,
    Populated(Box<T>)
}

const STAGING_BUFFER_SIZE: usize = 256 * 1024 * 1024; // 256 MB

// Util object to one-time upload data to a buffer using a staging buffer
pub(crate) struct BufferInitializer {
    allocator: Allocator,
    staging_buffer: Buffer,
    staging_memory: Allocation,
    staging_map: *mut u8,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    queue_family: u32,
    fence: Fence,
}

impl BufferInitializer {
    // todo: make this work for different queues! (inspect queue transfers)
    pub(crate) fn new(instance: &Instance, device: &Device, gpu: &PhysicalDevice, queue_family: u32) -> Self { // todo: queue_idx -> array (have a map  u32 -> pool  of the different supported)
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
                .queue_family_index(queue_family);

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

            unsafe {
                device
                    .allocate_command_buffers(&command_buffer_info)
                    .expect("Failed to allocate command buffers")[0]
            }
        };

        let (staging_buffer, staging_memory, staging_map) = {
            let buffer_info = BufferCreateInfo::default()
                .size(STAGING_BUFFER_SIZE as DeviceSize)
                .usage(BufferUsageFlags::TRANSFER_SRC)
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
            queue_family,
            fence
        }
    }

    pub(crate) fn init_buffer<T: BufferData>(&mut self, buffer: &mut VulkanBuffer<T>, mode: InitMode<T>, queue: (Queue, u32), device: &Device) {
        // Check if the queue matches
        if queue.1 != self.queue_family {
            panic!("Currently, only queue family {} is supported! {} was requested.", self.queue_family, queue.1)
        }

        if let InitMode::Zeroed = mode {
            unsafe { self.clear_buffer(device, queue.0, &mut buffer.handle(), buffer.size()); }
            return;
        }

        let InitMode::Populated(data) = mode else { unreachable!() };

        unsafe {
            let region = BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: data.size() as _,
            };

            // stage data and begin command
            self.stage_data(device, data);
            self.begin_command(device);

            // perform copy
            device.cmd_copy_buffer(self.command_buffer, self.staging_buffer, buffer.handle(), from_ref(&region));

            // submit
            self.end_command(device, queue.0);
        }
    }

    // warning: this also transitions the image layout
    pub(crate) unsafe fn init_image<T: BufferData>(&mut self, device: &Device, queue: Queue, src: Box<T>, dst: &mut Image, dst_layout: ImageLayout, dst_extent: Extent3D) {
        let size = src.size() as u64;

        // stage data and begin command
        self.stage_data(device, src);
        self.begin_command(device);

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
        self.end_command(device, queue);
    }

    pub(crate) unsafe fn clear_buffer(&mut self, device: &Device, queue: Queue, buffer: &mut Buffer, size: DeviceSize) {
        self.begin_command(device);

        device.cmd_fill_buffer(self.command_buffer, *buffer, 0, size, 0);

        self.end_command(device, queue);
    }

    // todo: remove this when proper pipelining is implemented
    pub(crate) unsafe fn copy_buffer(&mut self, device: &Device, queue: Queue, src: &mut Buffer, dst: &mut Buffer, size: DeviceSize) {
        self.begin_command(device);
        
        let region = BufferCopy::default().size(size);
        
        device.cmd_copy_buffer(self.command_buffer, *src, *dst, from_ref(&region));

        self.end_command(device, queue);
    }

    unsafe fn begin_command(&mut self, device: &Device) {
        device
            .reset_fences(from_ref(&self.fence))
            .expect("Failed to reset fences");

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

    unsafe fn end_command(&mut self, device: &Device, queue: Queue) {
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
            .wait_for_fences(from_ref(&self.fence), true, u64::MAX)
            .expect("Failed to wait for fence");
    }

    unsafe fn stage_data<T: BufferData>(&self, device: &Device, data: Box<T>) {
        if size_of::<T>() > STAGING_BUFFER_SIZE {
            todo!("Uploading data bigger than the staging buffer isn't supported yet");
        }

        data.serialize(self.staging_map);

        self.allocator
            .flush_allocation(&self.staging_memory, 0, WHOLE_SIZE)
            .expect("Failed to flush allocation");
    }
}
