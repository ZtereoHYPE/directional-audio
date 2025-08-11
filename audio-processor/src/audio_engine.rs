#[allow(unsafe_op_in_unsafe_fn)]

use crate::audio_engine::gpu_constants::{MAX_INSTANCES, SLIDING_WINDOW_FRAME_AMT};
use crate::scene::source::{AudioSource, Frame, FRAME_SIZE};
use crate::scene::Scene;
use crate::vulkan::buffer::{InlineBufferData, VulkanBuffer};
use crate::vulkan::buffer_initializer::{BufferInitializer, InitMode};
use ash::ext::debug_utils;
use ash::vk::{ApplicationInfo, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, DeviceCreateInfo, DeviceQueueCreateInfo, DeviceSize, Fence, FenceCreateInfo, InstanceCreateInfo, PhysicalDeviceFeatures2, PhysicalDeviceShaderAtomicFloatFeaturesEXT, Queue, SpecializationMapEntry, SubmitInfo};
use ash::{vk, vk::{DebugUtilsMessengerEXT, PhysicalDevice}, Device, Entry, Instance};
use bytemuck::Zeroable;
use glam::{Mat3, Vec2, Vec3};
use std::array::from_ref;
use std::borrow::Cow;
use std::ffi::{c_char, CStr};
use std::{fs::File, path::Path};
use std::f32::consts::PI;
use std::mem::{transmute, zeroed};
use std::ptr::copy_nonoverlapping;
use std::rc::Rc;
use ash::prelude::VkResult;
use vk_mem::{Allocator, AllocatorCreateInfo};
use crate::audio_engine::cluster::ClusterModule;
use crate::audio_engine::debug::{DebugRayModule};
use crate::audio_engine::delay::DelayModule;
use crate::audio_engine::fft::FftModule;
use crate::audio_engine::gpu_constants::{GPU_WINDOW_SIZE, MAX_DELAY_FRAMES, MAX_SOURCES, SPHERE_POINTS};
use crate::audio_engine::hrtf::HrtfModule;
use crate::audio_engine::rays::{RayBufferData, RayModule};
use crate::audio_engine::transfer::{DownloadBufferData, TransferModule};
use crate::scene::listener::AudioListener;
use crate::scene::mesh::bvh::MAX_BVH_DEPTH;
use crate::VisualizationData;
use crate::vulkan::debug_callback;

pub(crate) mod gpu_constants;
pub mod rays;
mod debug;
mod cluster;
mod transfer;
mod delay;
mod hrtf;
pub(crate) mod fft;

/// Represents a single audio frame on the GPU, containing complex values to enable FFT
pub(crate) type GpuFrame = [Vec2; FRAME_SIZE];

/// Represents the window used for partitioned convolution
pub(crate) type GpuWindow = [GpuFrame; SLIDING_WINDOW_FRAME_AMT]; // represents a sliding window of audio frames

pub struct AudioEngine {
    scene: Scene,

    entry: Entry,
    instance: Instance,
    debug_callback: DebugUtilsMessengerEXT,
    gpu: PhysicalDevice,
    device: Device,
    
    buffer_initializer: BufferInitializer,
    buffer_allocator: Rc<Allocator>,

    queue: (Queue, u32),

    delay_module: DelayModule,
    fft_module: FftModule,
    hrtf_module: HrtfModule,
    transfer_module: TransferModule,
    ray_module: RayModule,
    cluster_module: ClusterModule,
    debug_module: DebugRayModule,

    last_rt_pos: Vec3,
    frame_counter: usize,
    instance_buffer: VulkanBuffer<InstanceBufferData>,

    command_buffer: CommandBuffer,
    fence: Fence
}

impl AudioEngine {
    pub(crate) unsafe fn new(scene: Scene) -> Self { unsafe {
        let entry = Entry::load().expect("Could not load audio_engine library");

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
                .expect("Failed to create audio_engine instance")
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
                .pfn_user_callback(Some(debug_callback));

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);

            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        // todo: better logic for selecting device and queue(s)
        // igpu detection could happen here to better adapt things
        let (gpu, compute_queue_idx, device) = {
            let gpus = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices");

            let (gpu, idx) = gpus
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
                .expect("Couldn't find suitable device.");

            let device_extensions: [*const c_char; 1] = [c"VK_EXT_shader_atomic_float"]
                .map(|raw_name| raw_name.as_ptr());
            
            let mut atomic_floats_feature = PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
                .shader_buffer_float32_atomics(true);
            
            let mut gpu_features = PhysicalDeviceFeatures2::default().push_next(&mut atomic_floats_feature);
            instance.get_physical_device_features2(gpu, &mut gpu_features);

            let queue_info = DeviceQueueCreateInfo::default()
                .queue_family_index(idx)
                .queue_priorities(&[1.0]);

            let device_create_info = DeviceCreateInfo::default()
                .queue_create_infos(from_ref(&queue_info))
                .enabled_extension_names(&device_extensions[..])
                .push_next(&mut gpu_features);

            let device = instance
                .create_device(gpu, &device_create_info, None)
                .expect("Failed to create device!");

            (gpu, idx, device)
        };

        // todo: better detection and selection of the various queues
        let compute_queue = device.get_device_queue(compute_queue_idx, 0);

        let mut buffer_initializer = BufferInitializer::new(
            &instance,
            &device,
            &gpu,
            compute_queue_idx
        );

        let buffer_allocator = unsafe {
            let allocator_create_info = AllocatorCreateInfo::new(
                &instance,
                &device,
                gpu
            );

            Rc::new(
                Allocator::new(allocator_create_info)
                    .expect("Failed to create memory allocator")
            )
        };

        let command_pool = unsafe {
            let mut pool_create_info = CommandPoolCreateInfo::default()
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_queue_idx);

            device
                .create_command_pool(&pool_create_info, None)
                .expect("Failed to create command pool")
        };

        let command_buffer = unsafe {
            let mut command_buffer_info = CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
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
                    .descriptor_count(4 + 4),
                DescriptorPoolSize::default()
                    .ty(DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(11),
                DescriptorPoolSize::default()
                    .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(2),
            ];

            let pool_info = DescriptorPoolCreateInfo::default()
                .max_sets(32)
                .pool_sizes(&pool_sizes);

            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let mut instance_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            buffer_allocator.clone()
        );

        // Pre-populate the instance buffer until the first RT data comes in
        let mut data = Box::new(InstanceBufferData::from_scene_data(&scene));
        buffer_initializer.init_buffer(&mut instance_buffer, InitMode::Populated(data), (compute_queue, compute_queue_idx), &device);

        let fence = unsafe {
            device
                .create_fence(&FenceCreateInfo::default(), None)
                .expect("failed to create fence")
        };

        let fft_module = FftModule::new(
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            (compute_queue, compute_queue_idx),
            descriptor_pool,
        );

        let delay_module = DelayModule::new(
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            (compute_queue, compute_queue_idx),
            descriptor_pool,
            &instance_buffer,
            fft_module.starting_buffer(),
        );

        let hrtf_module = HrtfModule::new(
            scene.listener.filter.clone(),
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            (compute_queue, compute_queue_idx),
            descriptor_pool,
            &instance_buffer,
            fft_module.starting_buffer(), // todo: this is hardcoded for the size
        );

        let transfer_module = TransferModule::new(
            buffer_allocator.clone(),
            device.clone(),
            (compute_queue, compute_queue_idx),
            &delay_module.delay_buffer,
            &hrtf_module.output_buffer
        );

        let ray_module = RayModule::new(
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            descriptor_pool,
            (compute_queue, compute_queue_idx),
            &scene,
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
            compute_queue
        );

        Self {
            scene,
            entry,
            instance,
            debug_callback,
            gpu,
            device,
            queue: (compute_queue, compute_queue_idx),
            delay_module,
            fft_module,
            hrtf_module,
            transfer_module,
            ray_module,
            cluster_module,
            buffer_initializer,
            buffer_allocator,
            debug_module,
            last_rt_pos: Vec3::ZERO,
            frame_counter: 0,
            instance_buffer,
            command_buffer,
            fence
        }
    }}

    pub(crate) fn get_next_frames(&mut self, copy_debug_info: bool) -> (Frame, Frame, Option<VisualizationData>) { unsafe {
        self.trace_rays(copy_debug_info);
        let instance_amt = self.cluster_module.cluster();
        self.buffer_initializer.onetime_action(
            &self.device,
            self.queue.0,
            |cmd| self.cluster_module.upload_to_buffer(cmd, &mut self.instance_buffer)
        );

        // let instance_amt = self.scene.sources.len();
        // self.copy_sources_debug();
        // // todo: have 2 instance buffers
        // self.buffer_initializer.onetime_action(
        //     &self.device,
        //     self.queue.0,
        //     |cmd| {
        //         self.device.cmd_copy_buffer(
        //             *cmd,
        //             self.debug_module.instance_buffer.handle(),
        //             self.instance_buffer.handle(),
        //             InstanceBufferData::region()
        //         )
        //     }
        // );

        let ray_debug_data = if copy_debug_info {
            let mut debug_data = VisualizationData {
                last_rt_origin: self.last_rt_pos,
                rays: self.ray_module.local_ray_buffer.buffer_data().to_local_copy(),
                instances: self.cluster_module.get_clusters_debug().clone()
            };

            // debug_data.instances = self.scene.sources
            //     .iter()
            //     .map(|s| AudioInstance {
            //         direction: s.coordinates,
            //         distance: 0.0,
            //         index: 0,
            //         attenuation: 1.0,
            //     })
            //     .collect();

            Some(debug_data)
        } else {
            None
        };


        let (left, right) = self.process_frames(instance_amt as u32);

        (
            gpu_to_frame(&left),
            gpu_to_frame(&right),
            ray_debug_data
        )
    }}

    // from signal processor
    pub unsafe fn process_frames(&mut self, instance_amt: u32) -> (GpuFrame, GpuFrame) {
        // transfer data to right buffer
        self.transfer_module
            .upload_new_frames(&mut self.command_buffer, &mut self.scene.sources, self.frame_counter)
            .expect("Failed to upload frames to the GPU!");

        // move the delayed windows to the fft buffer
        let camera_delta = self.scene.listener.location - self.last_rt_pos;
        self.delay_module.apply_delay(&mut self.command_buffer, self.frame_counter, camera_delta, self.scene.listener.rotation, MAX_SOURCES);

        // perform fourier transform
        self.fft_module.gpu_fourier_transform(&mut self.command_buffer, 0, false, MAX_SOURCES);

        // wipe the output buffer
        self.hrtf_module.wipe_output(&mut self.command_buffer);

        // perform HRTF dsp
        self.hrtf_module.apply_hrtf(&mut self.command_buffer, instance_amt);

        // transfer data back
        let (left_window, right_window) = self.transfer_module
            .download_windows(&mut self.command_buffer)
            .expect("Failed to download frames from GPU!");

        self.frame_counter += 1;

        let left = FftModule::local_fourier_transform(window_to_vec(left_window), true);
        let right = FftModule::local_fourier_transform(window_to_vec(right_window), true);
        let (start, end) = DownloadBufferData::last_frame_range();

        (
            GpuFrame::try_from(&left[start..end]).unwrap(),
            GpuFrame::try_from(&right[start..end]).unwrap(),
        )
    }

    // from ray tracer
    pub(super) unsafe fn trace_rays(&mut self, store_rays: bool) -> VkResult<()> {
        let last_rt_pos = self.scene.listener.location;
        self.device.reset_fences(from_ref(&self.fence))?;

        // Begin the command buffer
        self.device.reset_command_buffer(self.command_buffer, CommandBufferResetFlags::empty())?;

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(self.command_buffer, &begin_info)?;

        // Stage the sources
        self.ray_module.stage_sources(&mut self.command_buffer, &self.scene.sources);

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

        self.device.queue_submit(self.queue.0, &[submit_info], self.fence)?;

        // Wait for it to execute
        self.device.wait_for_fences(from_ref(&self.fence), true, u64::MAX)?;

        self.last_rt_pos = last_rt_pos; // update it only once it's fully done

        Ok(())
    }

    pub(super) unsafe fn copy_sources_debug(&mut self) {
        let rt_pos = self.scene.listener.location;

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
        self.ray_module.stage_sources(&mut self.command_buffer, &self.scene.sources);

        // Trace the rays
        self.debug_module.copy_sources(&mut self.command_buffer, MAX_SOURCES as u32, rt_pos);

        // Submit the command buffer
        self.device
            .end_command_buffer(self.command_buffer)
            .expect("Failed to end command buffer!");

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&self.command_buffer));

        self.device
            .queue_submit(self.queue.0, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");

        // Wait for it to execute
        self.device
            .wait_for_fences(from_ref(&self.fence), true, u64::MAX)
            .expect("Failed to wait for fence!");

        self.last_rt_pos = rt_pos; // update it only once it's fully done
    }

    pub(crate) fn update_listener(&mut self, new_location: Vec3, new_rotation: Mat3) {
        self.scene.listener.location = new_location;
        self.scene.listener.rotation = new_rotation;
    }

    pub(crate) fn update_sources(&mut self, new_locations: Vec<(usize, Vec3)>) {
        for (idx, location) in new_locations {
            self.scene.sources[idx].coordinates = location;
        }
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        // todo: Cleanup!

    }
}

pub(crate) fn frame_to_gpu(frame: &Frame) -> GpuFrame {
    // todo: avoid initialization here
    let mut samples = [Vec2::ZERO; FRAME_SIZE];

    for (idx, value) in frame.iter().enumerate() {
        samples[idx].x = *value;
    }

    samples
}

pub(crate) fn gpu_to_frame(input: &GpuFrame) -> Frame {
    let mut frame: Frame = [0.0; FRAME_SIZE];

    for (idx, value) in input.iter().enumerate() {
        frame[idx] = value.x;
    }

    frame
}

pub(crate) unsafe fn window_to_vec(window: Box<GpuWindow>) -> Vec<Vec2> {
    // transmute the pointer without performing a copy; arrays in rust are guaranteed to be sequential
    let flat_window = transmute::<Box<GpuWindow>, Box<[Vec2; GPU_WINDOW_SIZE]>>(window);
    (flat_window as Box<[_]>).into_vec() // turn the box into a vector
}

#[repr(align(16))]
#[derive(Copy, Clone, Zeroable)]
pub struct AudioInstance {
    pub direction: Vec3,
    pub distance: f32,
    pub attenuation: f32,
    pub index: u32,
}

pub struct InstanceBufferData {
    pub instances: [AudioInstance; MAX_INSTANCES],
}

impl InlineBufferData for InstanceBufferData {}

impl InstanceBufferData {
    pub(crate) fn from_scene_data(scene: &Scene) -> Self {
        let mut instance = Self {
            instances: [AudioInstance::zeroed(); MAX_INSTANCES]
        };

        for (idx, source) in scene.sources.iter().enumerate() {
            instance.instances[idx] = (AudioInstance {
                direction: source.coordinates,
                distance: source.coordinates.length(),
                index: idx as u32,
                attenuation: 0.0,
            });
        }

        instance
    }

    pub(crate) fn copy_instances(&mut self, instances: &Vec<AudioInstance>) {
        assert!(instances.len() <= MAX_INSTANCES);

        for (idx, val) in instances.iter().enumerate() {
            self.instances[idx] = *val;
        }
    }
}
