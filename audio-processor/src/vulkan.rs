#![allow(unused)]
#![allow(unsafe_op_in_unsafe_fn)]


use core::ffi;
use std::array::from_ref;
use std::borrow::Cow;
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


pub struct VulkanEngine {
    entry: Entry,
    instance: Instance,
    debug_callback: DebugUtilsMessengerEXT,

    gpu: PhysicalDevice,
    device: Device,
    compute_queue: Queue,
    
    command_pool: CommandPool,
    setup_command_buffer: CommandBuffer,
    command_buffer: CommandBuffer,

    buffer_allocator: vk_mem::Allocator,
    cpu_buffer: Buffer,
    cpu_buffer_memory: vk_mem::Allocation,
    cpu_buffer_map: *mut u8,
    gpu_buffer: Buffer,
    gpu_buffer_memory: vk_mem::Allocation,

    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<DescriptorSet>,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,

    readback_fence: Fence,

}

impl VulkanEngine {
    pub unsafe fn new() -> Self {
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

        let buffer_allocator = {
            let allocator_create_info = AllocatorCreateInfo::new(
                &instance, 
                &device,
                gpu
            );

            vk_mem::Allocator::new(allocator_create_info)
                .expect("Failed to create memory allocator")
        };

        let (cpu_buffer, mut cpu_buffer_memory, mut cpu_buffer_map) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<Frame>() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST)
                .queue_family_indices(from_ref(&queue_family_index))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                preferred_flags: MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_CACHED,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
                ..Default::default()
            };

            let (buffer, mut memory) = buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer");

            let map = buffer_allocator
                .map_memory(&mut memory)
                .expect("Failed to map memory");

            (buffer, memory, map)
        };

        let (gpu_buffer, gpu_buffer_memory) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<Frame>() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue_family_index))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let descriptor_pool = {
            let pool_sizes = [
                DescriptorPoolSize::default()
                    .descriptor_count(1)
                    .ty(DescriptorType::STORAGE_BUFFER)
            ];

            let pool_info = DescriptorPoolCreateInfo::default()
                .max_sets(1) // for now this pool can only allocate 1 set
                .pool_sizes(&pool_sizes);

            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let (descriptor_set, descriptor_layout) = {
            // Create layout
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE)
            ];

            let set_layout_info = DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            let set_layout = device
                .create_descriptor_set_layout(&set_layout_info, None)
                .expect("Failed to create descriptor set layout");

            // Allocate set
            let info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(from_ref(&set_layout));
            
            let set = device
                .allocate_descriptor_sets(&info)
                .expect("Failed to allocate descriptor sets")[0];


            // Write set
            let buffer_info = DescriptorBufferInfo::default()
                .buffer(gpu_buffer)
                .range(WHOLE_SIZE);
            
            let write = WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(from_ref(&buffer_info));

            device.update_descriptor_sets(&[write], &[]);

            (set, set_layout)
        };

        let (pipeline, pipeline_layout) = {
            let code_words = read_file_words("target/shaders/shader.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_layout));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let stage_info = PipelineShaderStageCreateInfo::default()
                .stage(ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");

            let pipeline_info = ComputePipelineCreateInfo::default()
                .layout(layout)
                .stage(stage_info);

            let pipeline = device
                .create_compute_pipelines(
                    PipelineCache::null(), 
                    from_ref(&pipeline_info), 
                    None
                )
                .expect("Failed to create pipeline")[0];

            (pipeline, layout)
        };

        let readback_fence = device
            .create_fence(&FenceCreateInfo::default(), None)
            .expect("Failed to create fence");

        Self {
            entry,
            instance,
            debug_callback,
            gpu,
            device,
            compute_queue,
            command_pool,
            setup_command_buffer,
            command_buffer,
            buffer_allocator,
            cpu_buffer,
            cpu_buffer_memory,
            cpu_buffer_map,
            gpu_buffer,
            gpu_buffer_memory,
            descriptor_pool,
            descriptor_sets: vec![descriptor_set],
            pipeline,
            pipeline_layout,
            readback_fence,
        }
    }

    pub unsafe fn process_frame(&mut self, frame: &Frame) -> Frame {
        // copy frame to cpu buffer
        // todo: there must be a safer way to do things..
        let mut align = ash::util::Align::new(
            self.cpu_buffer_map.cast(), 
            align_of::<Frame>() as u64, 
            size_of::<Frame>() as u64
        );
        
        align.copy_from_slice(frame);
        self.buffer_allocator.flush_allocation(&self.cpu_buffer_memory, 0, vk::WHOLE_SIZE);


        // dispatch compute
        {
            let begin_info = CommandBufferBeginInfo::default()
                .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("Failed to begin command buffer recording");

            let regions = [ BufferCopy::default().size(size_of::<Frame>() as _) ]; // todo: this could not be right size
            
            self.device.cmd_bind_pipeline(
                self.command_buffer, 
                PipelineBindPoint::COMPUTE, 
                self.pipeline
            );

            self.device.cmd_bind_descriptor_sets(
                self.command_buffer, 
                PipelineBindPoint::COMPUTE, 
                self.pipeline_layout, 
                0, 
                &self.descriptor_sets[..], 
                &[]
            );
            
            self.device.cmd_copy_buffer(
                self.command_buffer, 
                self.cpu_buffer, 
                self.gpu_buffer, 
                &regions
            );

            let memory_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::TRANSFER_WRITE) // flush any transfer write caches
                .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

            self.device.cmd_pipeline_barrier(
                self.command_buffer, 
                PipelineStageFlags::TRANSFER, // wait for all transfer commands so far...
                PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute from now on
                DependencyFlags::empty(), 
                //from_ref(&memory_barrier), 
                &[],
                &[], 
                &[]
            );

            self.device.cmd_dispatch(
                self.command_buffer, 
                1024, 1, 1
            );

            let memory_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::SHADER_WRITE) // flush any compute write write caches
                .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any transfer read caches

            self.device.cmd_pipeline_barrier(
                self.command_buffer, 
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::TRANSFER, // ...before executing any transfer from now on
                DependencyFlags::empty(), 
                //from_ref(&memory_barrier), 
                &[],
                &[], 
                &[]
            );

            self.device.cmd_copy_buffer(
                self.command_buffer, 
                self.gpu_buffer, 
                self.cpu_buffer, 
                &regions
            );

            self.device.end_command_buffer(self.command_buffer);

            // todo: much more synchronization here
            let submit_info = SubmitInfo::default()
                .command_buffers(from_ref(&self.command_buffer));

            self.device
                .queue_submit(self.compute_queue, &[submit_info], self.readback_fence)
                .expect("Failed to submit command buffer");
        }

        self.device.wait_for_fences(from_ref(&self.readback_fence), true, MAX);

        // readback result
        self.buffer_allocator.invalidate_allocation(&self.cpu_buffer_memory, 0, vk::WHOLE_SIZE);

        let mut output_frame = *(self.cpu_buffer_map as *const Frame); // todo: AAAAAaaaaaAaAAA

        output_frame
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

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        // Free the buffers
        unsafe {
            self.buffer_allocator.unmap_memory(&mut self.cpu_buffer_memory);
            self.buffer_allocator.free_memory(&mut self.cpu_buffer_memory);
            self.buffer_allocator.free_memory(&mut self.gpu_buffer_memory);

            // todo: more cleanup
        }
    }
}

fn read_file_words(path: impl AsRef<Path>) -> Vec<u32> {
    let path = path.as_ref();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let mut file = File::open(&path).unwrap();

    ash::util::read_spv(&mut file).unwrap()
}
