#![allow(unused)]
#![allow(unsafe_op_in_unsafe_fn)]


use core::ffi;
use std::array::from_ref;
use std::borrow::Cow;
use std::collections::HashMap;
use std::process::Command;
use std::u64::MAX;
use std::{ffi::c_char, fs::File};
use std::io::Read;
use std::num::NonZero;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ash::ext::debug_utils;
use ash::vk::{AccessFlags, Buffer, BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, ComputePipelineCreateInfo, DebugUtilsMessengerEXT, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceCreateInfo, DeviceQueueCreateInfo, ExtendsRayTracingPipelineCreateInfoKHR, Extent3D, Fence, FenceCreateFlags, FenceCreateInfo, Image, ImageAspectFlags, ImageLayout, ImageMemoryBarrier, ImageSubresource, ImageSubresourceLayers, ImageSubresourceRange, ImageView, MappedMemoryRange, MemoryAllocateInfo, MemoryBarrier, MemoryPropertyFlags, PhysicalDevice, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo, PipelineStageFlags, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SubmitInfo, WriteDescriptorSet, QUEUE_FAMILY_IGNORED, WHOLE_SIZE};
use ash::{vk::{self, ApplicationInfo, InstanceCreateInfo}, Entry, Instance, Device};
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, Allocator, AllocatorCreateInfo};

use crate::audio::Frame;




//  TODO: MOVE THIS TO VULKAN.RS
// RENAME VULKAN TO VULKAN_MODULES



const STAGING_BUFFER_SIZE: u64 = 128 * 1024 * 1024; // 128 MB

pub(crate) struct DescriptorRequirement {
    pub ttype: DescriptorType,
    pub amount: u32
}

pub(crate) trait VulkanModule {
    fn descriptors() -> Vec<DescriptorRequirement>;
}

pub(crate) trait GpuData {
    unsafe fn serialize(&self, dst: *mut u8);
    unsafe fn deserialize(src: *const u8) -> Box<Self>;
    fn size(&self) -> usize;
}

// Util object to one-time upload data to a buffer using a staging buffer
pub(crate) struct BufferUploader {
    allocator: Allocator,
    staging_buffer: Buffer,
    staging_memory: Allocation,
    staging_map: *mut u8,
    command_buffer: CommandBuffer,
    fence: Fence,
}

impl BufferUploader {
    fn new(instance: &Instance, device: &Device, gpu: &PhysicalDevice, compute_queue_idx: u32, command_buffer: CommandBuffer) -> Self {
        let allocator = {
            let allocator_create_info = AllocatorCreateInfo::new(
                instance, 
                device,
                *gpu
            );

            unsafe {
                vk_mem::Allocator::new(allocator_create_info)
                    .expect("Failed to create memory allocator")
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
            command_buffer,
            fence
        }
    }

    // todo: after the refactor make these methods mutate the buffer uploader!
    pub(crate) unsafe fn upload_buffer_onetime<T: GpuData>(&self, device: &Device, queue: Queue, src: T, dst: &mut Buffer) {
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
        device.wait_for_fences(from_ref(&self.fence), true, MAX);
    }

    // warning: this also transitions the image layout
    pub(crate) unsafe fn upload_image_onetime<T: GpuData>(&self, device: &Device, queue: Queue, src: T, dst: &mut Image, dst_layout: ImageLayout, dst_extent: Extent3D) {
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
        device.end_command_buffer(self.command_buffer);

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&self.command_buffer));

        device
            .queue_submit(queue, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");

        // wait for fence
        device.wait_for_fences(from_ref(&self.fence), true, MAX);
    }

    unsafe fn prepare_onetime<T: GpuData>(&self, device: &Device, src: T) {
        let size = src.size() as u64;

        if size > STAGING_BUFFER_SIZE {
            todo!("Uploading data bigger than the staging buffer isn't supported yet");
        }

        device.reset_fences(from_ref(&self.fence));

        // copy data to the staging buffer
        src.serialize(self.staging_map);
        self.allocator.flush_allocation(&self.staging_memory, 0, WHOLE_SIZE);

        // reset and begin command buffer
        device
            .reset_command_buffer(self.command_buffer, CommandBufferResetFlags::RELEASE_RESOURCES) // todo: releaase resources?
            .expect("Failed to reset command buffer"); 

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(self.command_buffer, &begin_info)
            .expect("Failed to begin command buffer recording");
    }
}

pub struct VulkanBuilder {
    descriptors: HashMap<DescriptorType, u32>
}

impl VulkanBuilder {
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new()
        }
    }

    pub fn register_module<T: VulkanModule>(mut self) -> Self {
        let module_requirements = T::descriptors();

        for req in module_requirements {
            let prev = self.descriptors.get(&req.ttype).or(Some(&0)).unwrap();
            self.descriptors.insert(req.ttype, prev + req.amount);
        }

        return self;
    }

    pub unsafe fn build(self) -> VulkanContext {
        VulkanContext::new(self.descriptors)
    }
}

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub debug_callback: DebugUtilsMessengerEXT,

    pub gpu: PhysicalDevice,
    pub device: Device,
    pub compute_queue: (Queue, u32),

    pub command_pool: CommandPool,
    pub command_buffer: CommandBuffer,

    pub descriptor_pool: DescriptorPool,
    pub buffer_uploader: BufferUploader
}

impl VulkanContext {
    unsafe fn new(descriptors: HashMap<DescriptorType, u32>) -> Self {
        let entry = Entry::load().expect("Could not load vulkan library");

        let instance = {
            let layers_names_raw: [*const c_char; 1] = [c"VK_LAYER_KHRONOS_validation"] // c"VK_LAYER_LUNARG_api_dump"
                .map(|raw_name| raw_name.as_ptr());

            let extension_names_raw: [*const c_char; 1] = [c"VK_EXT_debug_utils"]
                .map(|raw_name| raw_name.as_ptr());

            let application_info = ApplicationInfo::default()
                .api_version(vk::make_api_version(0, 1, 3, 0))
                .application_name(c"Audio Processor")
                .engine_name(c"No Engine");
            
            let instance_info = InstanceCreateInfo::default()
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw)
                .application_info(&application_info);

            entry
                .create_instance(&instance_info, None)
                .expect("Failed to create vulkan instance")
        };

        let debug_callback = {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(Self::debug_callback));

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);

            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        // todo: better logic for selecting device and queue
        let (gpu, queue_family_index) = {
            let gpus = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices");

            gpus
                .iter()
                .flat_map(|gpu| {
                    instance
                        .get_physical_device_queue_family_properties(*gpu)
                        .iter()
                        .filter(|info| info.queue_flags.contains(vk::QueueFlags::COMPUTE))
                        .enumerate()
                        .map(|(index, info)| (*gpu, index as u32))
                        .collect::<Vec<_>>()
                })
                .next()
                .expect("Couldn't find suitable device.")
        };

        let device = {
            let queue_info = DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .queue_priorities(&[1.0]);

            let device_create_info = DeviceCreateInfo::default()
                .queue_create_infos(from_ref(&queue_info));
        
            instance
                .create_device(gpu, &device_create_info, None)
                .expect("Failed to create device!")
        };

        let compute_queue = device.get_device_queue(queue_family_index, 0);

        let command_pool = {
            let pool_create_info = CommandPoolCreateInfo::default()
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            device
                .create_command_pool(&pool_create_info, None)
                .expect("Failed to create command pool")
        };

        let (setup_command_buffer, command_buffer) = {
            let command_buffer_info = CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(2)
                .level(CommandBufferLevel::PRIMARY);

            let buffers = device
                .allocate_command_buffers(&command_buffer_info)
                .expect("Failed to allocate command buffers");

            (buffers[0], buffers[1])
        };

        let descriptor_pool = {
            let mut pool_sizes = vec![];

            for (ttype, size) in descriptors {
                pool_sizes.push(
                    DescriptorPoolSize::default()
                        .ty(ttype)
                        .descriptor_count(size)
                );
            }

            let pool_info = DescriptorPoolCreateInfo::default()
                .max_sets(32) // todo: get this info from modules
                .pool_sizes(&pool_sizes[..]);

            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let buffer_uploader = BufferUploader::new(&instance, &device, &gpu, queue_family_index, setup_command_buffer);

        Self {
            entry,
            instance,
            debug_callback,
            gpu,
            device,
            compute_queue: (compute_queue, queue_family_index),
            command_pool,
            command_buffer,
            descriptor_pool,
            buffer_uploader
        }
    }
    
    unsafe extern "system" fn debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}