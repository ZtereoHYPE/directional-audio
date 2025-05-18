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
use ash::vk::{AccessFlags, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, ComputePipelineCreateInfo, DebugUtilsMessengerEXT, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceCreateInfo, DeviceQueueCreateInfo, Fence, FenceCreateInfo, MappedMemoryRange, MemoryAllocateInfo, MemoryBarrier, MemoryPropertyFlags, PhysicalDevice, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo, PipelineStageFlags, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SubmitInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::{vk::{self, ApplicationInfo, InstanceCreateInfo}, Entry, Instance, Device};
use vk_mem::{Alloc, AllocationCreateFlags, AllocationCreateInfo, AllocatorCreateInfo};

use crate::audio::Frame;

pub(crate) struct DescriptorRequirement {
    pub ttype: DescriptorType,
    pub amount: u32
}

pub(crate) trait VulkanModule {
    fn descriptors() -> Vec<DescriptorRequirement>;
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
    pub setup_command_buffer: CommandBuffer,
    pub command_buffer: CommandBuffer,

    pub descriptor_pool: DescriptorPool,
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

        Self {
            entry,
            instance,
            debug_callback,
            gpu,
            device,
            compute_queue: (compute_queue, queue_family_index),
            command_pool,
            setup_command_buffer,
            command_buffer,
            descriptor_pool,
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