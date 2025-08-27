use crate::audio_engine::cluster::ClusterModule;
use crate::audio_engine::debug::DebugRayModule;
use crate::audio_engine::delay::DelayModule;
use crate::audio_engine::fft::FftModule;
use crate::audio_engine::gpu_constants::{GPU_WINDOW_SIZE, MAX_SOURCES};
#[allow(unsafe_op_in_unsafe_fn)]

use crate::audio_engine::gpu_constants::{MAX_INSTANCES, SLIDING_WINDOW_FRAME_AMT};
use crate::audio_engine::hrtf::HrtfModule;
use crate::audio_engine::rays::RayModule;
use crate::scene::source::{Frame, FRAME_SIZE};
use crate::scene::Scene;
use crate::vulkan::buffer::{InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use crate::vulkan::buffer_initializer::{BufferInitializer, InitMode};
use crate::vulkan::{debug_callback, get_device_score};
use crate::VisualizationData;
use ash::ext::debug_utils;
use ash::prelude::VkResult;
use ash::vk::{ApplicationInfo, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorType, DeviceCreateInfo, DeviceQueueCreateInfo, Fence, FenceCreateFlags, FenceCreateInfo, InstanceCreateInfo, PhysicalDeviceFeatures, PhysicalDeviceFeatures2, PhysicalDeviceShaderAtomicFloatFeaturesEXT, PhysicalDeviceTimelineSemaphoreFeatures, PhysicalDeviceType, PipelineStageFlags, Queue, Semaphore, SemaphoreCreateInfo, SemaphoreSignalInfo, SemaphoreType, SemaphoreTypeCreateInfo, SemaphoreWaitInfo, SubmitInfo, TimelineSemaphoreSubmitInfo};
use ash::{vk, vk::{DebugUtilsMessengerEXT, PhysicalDevice}, Device, Entry, Instance};
use bytemuck::Zeroable;
use glam::{Mat3, Vec2, Vec3};
use std::array::from_ref;
use std::cell::OnceCell;
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::mem::transmute;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::fence;
use std::time::Instant;
use itertools::izip;
use vk_mem::{Allocator, AllocatorCreateInfo};
use crate::audio_engine::transfer::{DownloadBufferData, TransferModule};
use crate::util::complex::from;
use crate::vulkan::in_flight::{InFlight, InFlightCounter};
use crate::vulkan::queue::{select_queues, VulkanQueue};
use crate::vulkan::timeline::{PipelineStage, TimelineTracker};

pub(crate) mod gpu_constants;
pub(crate) mod fft;
pub mod rays;
mod debug;
mod cluster;
mod transfer;
mod delay;
mod hrtf;

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

    compute_queue: VulkanQueue,
    async_queue: VulkanQueue,
    transfer_queue: VulkanQueue,

    buffer_initializer: BufferInitializer,
    buffer_allocator: Arc<Allocator>,

    delay_module: DelayModule,
    fft_module: FftModule,
    hrtf_module: HrtfModule,
    transfer_module: TransferModule,
    ray_module: RayModule,
    cluster_module: ClusterModule,
    debug_module: DebugRayModule,

    instance_buffer: VulkanBuffer<InstanceBufferData>,

    audio_command_buffers: InFlight<(CommandBuffer, CommandBuffer, CommandBuffer)>,
    audio_fences: InFlight<(Fence, Fence)>,
    audio_timeline: TimelineTracker<AudioSyncStage>,
    audio_counter: InFlightCounter,

    rt_command_buffers: (CommandBuffer, CommandBuffer),
    rt_fence: Fence,
    rt_timeline: TimelineTracker<RtSyncStage>,
    rt_counter: InFlightCounter,

    frame_counter: usize,
}

impl AudioEngine {
    pub(crate) unsafe fn new(scene: Scene, frames_in_flight: usize) -> Self { unsafe {
        let entry = Entry::load().expect("Could not load audio_engine library");

        let instance = {
            let layers = entry.enumerate_instance_layer_properties();
            println!("layers: {:?}", layers);
            let layers_names_raw: [*const c_char; _] = [c"VK_LAYER_KHRONOS_validation", ] // c"VK_LAYER_LUNARG_api_dump"
                .map(|raw_name: &CStr| raw_name.as_ptr());

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

        let (gpu, device, queue_selection) = {
            let gpus = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices");

            let mut gpu = gpus
                .iter()
                .filter(|&&gpu| select_queues(&instance, gpu).is_some())
                .filter_map(|&gpu| get_device_score(&instance, gpu).map(|s| (gpu, s)))
                .max_by(|(_, left), (_, right)| right.cmp(left))
                .expect("Could not find a suitable GPU!").0;

            let queue_selection = select_queues(&instance, gpu).unwrap();
            let queue_create_infos = queue_selection.get_queue_create_infos();

            let mut timeline_features = PhysicalDeviceTimelineSemaphoreFeatures::default()
                .timeline_semaphore(true);

            let device_create_info = DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_infos)
                .push_next(&mut timeline_features);

            let device = instance
                .create_device(gpu, &device_create_info, None)
                .expect("Failed to create device!");

            (gpu, device, queue_selection)
        };

        queue_selection.set_global_families();

        let (compute_queue, async_queue, transfer_queue) = queue_selection.get_queues(&device);

        let mut buffer_initializer = BufferInitializer::new(
            &instance,
            &device,
            &gpu,
            compute_queue.family
        );

        let buffer_allocator = unsafe {
            let allocator_create_info = AllocatorCreateInfo::new(
                &instance,
                &device,
                gpu
            );

            Arc::new(
                Allocator::new(allocator_create_info)
                    .expect("Failed to create memory allocator")
            )
        };

        let (compute_pool, async_pool, transfer_pool) = unsafe {
            let mut pool_info = CommandPoolCreateInfo::default()
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_queue.family);

            let compute_p = device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create command pool");

            pool_info = pool_info.queue_family_index(async_queue.family);
            let async_p = device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create command pool");

            pool_info = pool_info.queue_family_index(transfer_queue.family);
            let transfer_p = device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create command pool");

            (compute_p, async_p, transfer_p)
        };

        let (audio_command_buffers, rt_command_buffers) = unsafe {
            let mut cb_info = CommandBufferAllocateInfo::default()
                .command_buffer_count(frames_in_flight as u32)
                .level(CommandBufferLevel::PRIMARY);

            cb_info = cb_info.command_pool(transfer_pool);
            let upload_cbs = device
                    .allocate_command_buffers(&cb_info)
                    .expect("Failed to allocate command buffers");

            let download_cbs = device
                    .allocate_command_buffers(&cb_info)
                    .expect("Failed to allocate command buffers");

            cb_info = cb_info.command_pool(compute_pool);
            let dsp_cbs = device
                    .allocate_command_buffers(&cb_info)
                    .expect("Failed to allocate command buffers");

            cb_info = cb_info
                .command_pool(async_pool)
                .command_buffer_count(2);
            let rt_cbs = device
                .allocate_command_buffers(&cb_info)
                .expect("Failed to allocate command buffers");

            let audio_cbs = InFlight::from(izip!(upload_cbs, dsp_cbs, download_cbs).collect::<Vec<_>>());

            (audio_cbs, (rt_cbs[0], rt_cbs[1]))
        };

        let descriptor_pool = unsafe {
            // todo: double-check that these sizes are OK
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
        buffer_initializer.init_buffer(&mut instance_buffer, InitMode::Populated(data), compute_queue, &device);

        let (audio_semaphores, rt_semaphore) = unsafe {
            let mut semaphore_type = SemaphoreTypeCreateInfo::default()
                .initial_value(0) // above or equal to the finished value for both
                .semaphore_type(SemaphoreType::TIMELINE);

            let semaphore_info = SemaphoreCreateInfo::default().push_next(&mut semaphore_type);

            (
                InFlight::create(
                    frames_in_flight,
                    |_| device.create_semaphore(&semaphore_info, None).expect("Failed to create semaphore")
                ),
                device.create_semaphore(&semaphore_info, None).expect("Failed to create semaphore")
            )
        };

        let (audio_fences, rt_fences) = {
            let fence_info = FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED);

            let audio = InFlight::create(
                frames_in_flight,
                |_| (
                    device.create_fence(&fence_info, None).expect("Failed to create audio fence"),
                    device.create_fence(&fence_info, None).expect("Failed to create audio fence")
                )
            );

            let rt = device.create_fence(&fence_info, None).expect("Failed to create rt fence");

            (audio, rt)
        };

        // let audio_deps = [
        //     (AudioSyncStage::Upload,     vec![(1, AudioSyncStage::Upload)]),
        //     (AudioSyncStage::Compute,    vec![(0, AudioSyncStage::Upload), (1, AudioSyncStage::Compute)]),
        //     (AudioSyncStage::Download,   vec![(0, AudioSyncStage::Compute), (1, AudioSyncStage::Submit)]),
        //     // (AudioSyncStage::Download,   vec![(0, AudioSyncStage::Compute)]),
        //     (AudioSyncStage::Submit,     vec![(0, AudioSyncStage::Download)]),
        // ].into_iter().collect();

        // TODO MASSIVE: figure out why pipelining causes crackling sound
        let audio_deps_seq = [
            (AudioSyncStage::Upload,     vec![(1, AudioSyncStage::Submit)]),
            (AudioSyncStage::Compute,    vec![(0, AudioSyncStage::Upload)]),
            (AudioSyncStage::Download,   vec![(0, AudioSyncStage::Compute)]),
            (AudioSyncStage::Submit,     vec![(0, AudioSyncStage::Download)]),
        ].into_iter().collect();

        let audio_counter = InFlightCounter::new(frames_in_flight);
        let audio_timeline = TimelineTracker::new(audio_semaphores, audio_deps_seq);

        let rt_deps = [
            (RtSyncStage::RayTrace, vec![]),
            (RtSyncStage::Cluster,  vec![(0, RtSyncStage::RayTrace)]),
            (RtSyncStage::Upload,   vec![(0, RtSyncStage::Cluster)]),
        ].into_iter().collect();

        let rt_timeline = TimelineTracker::new(InFlight::from(vec![rt_semaphore]), rt_deps);
        let rt_counter = InFlightCounter::new(1);

        let fft_module = FftModule::new(
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            descriptor_pool,
        );

        let delay_module = DelayModule::new(
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            compute_queue,
            descriptor_pool,
            &instance_buffer,
            fft_module.starting_buffer(),
        );

        let hrtf_module = HrtfModule::new(
            scene.listener.filter.clone(),
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            compute_queue,
            descriptor_pool,
            frames_in_flight,
            &instance_buffer,
            fft_module.ending_buffer(),
        );

        let transfer_module = TransferModule::new(
            buffer_allocator.clone(),
            device.clone(),
            compute_queue,
            frames_in_flight,
            &delay_module.delay_buffer,
            &hrtf_module.output_buffers,
        );

        let ray_module = RayModule::new(
            buffer_allocator.clone(),
            &mut buffer_initializer,
            device.clone(),
            descriptor_pool,
            compute_queue,
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
            compute_queue.handle
        );

        Self {
            scene,

            entry,
            instance,
            debug_callback,
            gpu,
            device,
            compute_queue,
            async_queue,
            transfer_queue,

            delay_module,
            fft_module,
            hrtf_module,
            transfer_module,
            ray_module,
            cluster_module,
            buffer_initializer,
            buffer_allocator,
            debug_module,

            instance_buffer,

            audio_command_buffers,
            audio_fences,
            audio_timeline,
            audio_counter,

            rt_command_buffers,
            rt_fence: rt_fences,
            rt_timeline,
            rt_counter,

            frame_counter: 0,
        }
    }}

    pub(crate) fn request_frame<F, D>(&mut self, frame_consumer: F, debug_consumer: Option<D>)
    where
        F: Fn(Frame, Frame) + Send + 'static,
        D: Fn(VisualizationData)
    { unsafe {
        let request_debug_data = debug_consumer.is_some();

        // Request an update to the instance buffer, if possible
         let update_ready = self
            .update_instances(request_debug_data)
            .expect("Failed to calculate new instances!");

        // Gather and submit the debug data
        if request_debug_data {
            let vis_data = VisualizationData {
                last_rt_origin: self.ray_module.last_rt_origin,
                rays: self.ray_module.local_ray_buffer.buffer_data().to_local_copy(),
                instances: self.cluster_module.last_clusters.clone()
            };

            debug_consumer.unwrap()(vis_data);
        }
        
        // Schedule the next audio frame
        self.request_audio(frame_consumer, update_ready);

        self.frame_counter += 1;
    }}

    // from signal processor
    unsafe fn request_audio(&mut self, consumer: impl Fn(Frame, Frame) + Send + 'static, update_instances: bool) -> VkResult<()> {
        let start_time = Instant::now();

        self.cluster_module.update_instances(); // we know that these won't change until the compute is done
        let instance_amt = self.cluster_module.instance_amt() as u32;

        let timeline = &mut self.audio_timeline;
        let counter = self.audio_counter;

        // Wait until the command buffers are done executing
        let fences = self.audio_fences[counter];
        self.device.wait_for_fences(&[fences.0, fences.1], true, u64::MAX)?;
        self.device.reset_fences(&[fences.0, fences.1])?;

        let fence_time = Instant::now();

        // Prepare the command buffers
        let (mut upload_buffer, mut compute_buffer, mut download_buffer) = self.audio_command_buffers[self.audio_counter];

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT); // todo: this might not be ideal

        self.device.reset_command_buffer(upload_buffer, CommandBufferResetFlags::empty())?;
        self.device.reset_command_buffer(compute_buffer, CommandBufferResetFlags::empty())?;
        self.device.reset_command_buffer(download_buffer, CommandBufferResetFlags::empty())?;
        self.device.begin_command_buffer(upload_buffer, &begin_info)?;
        self.device.begin_command_buffer(compute_buffer, &begin_info)?;
        self.device.begin_command_buffer(download_buffer, &begin_info)?;

        // transfer data to right buffer
        self.transfer_module.upload_new_frames(&mut upload_buffer, &mut self.scene.sources, self.frame_counter)?;

        // update the instances
        if update_instances {
            println!("sample number {} is updating instances", self.frame_counter * FRAME_SIZE);
            // the compute buffer ensures that the next compute calculations waits until this is done
            // before executing, therefore avoiding bad reads
            self.device.cmd_copy_buffer(
                compute_buffer,
                self.cluster_module.instance_buffer.handle(),
                self.instance_buffer.handle(),
                InstanceBufferData::region()
            );
        }

        // move the delayed windows to the fft buffer
        // todo: get rid of this
        self.delay_module.apply_delay(&mut compute_buffer, self.frame_counter, self.scene.listener.location, MAX_SOURCES);

        // perform fourier transform
        self.fft_module.gpu_fourier_transform(&mut compute_buffer, 0, false, MAX_SOURCES);

        // perform HRTF dsp
        self.hrtf_module.apply_hrtf(&mut compute_buffer, counter, instance_amt, self.scene.listener.rotation, self.scene.listener.location);

        // transfer data back
        self.transfer_module.download_windows(&mut download_buffer, counter);

        self.device.end_command_buffer(upload_buffer)?;
        self.device.end_command_buffer(compute_buffer)?;
        self.device.end_command_buffer(download_buffer)?;

        let (upload_wait_semaphore, upload_wait_values) = timeline.get_wait_info(counter, AudioSyncStage::Upload);
        let (upload_signal_semaphore, upload_signal_values) = timeline.get_signal_info(counter, AudioSyncStage::Upload);
        let upload_stages = vec![PipelineStageFlags::BOTTOM_OF_PIPE; upload_wait_semaphore.len()];
        let mut timeline_info_upload = TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&upload_wait_values)
            .signal_semaphore_values(&upload_signal_values);
        let submit_info_upload = SubmitInfo::default()
            .wait_semaphores(&upload_wait_semaphore)
            .signal_semaphores(&upload_signal_semaphore)
            .wait_dst_stage_mask(&upload_stages)
            .command_buffers(from_ref(&upload_buffer))
            .push_next(&mut timeline_info_upload);

        let (compute_wait_semaphore, compute_wait_values) = timeline.get_wait_info(counter, AudioSyncStage::Compute);
        let (compute_signal_semaphore, compute_signal_values) = timeline.get_signal_info(counter, AudioSyncStage::Compute);
        let compute_stages = vec![PipelineStageFlags::BOTTOM_OF_PIPE; compute_wait_semaphore.len()];
        let mut timeline_info_compute = TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&compute_wait_values)
            .signal_semaphore_values(&compute_signal_values);
        let submit_info_compute = SubmitInfo::default()
            .wait_semaphores(&compute_wait_semaphore)
            .signal_semaphores(&compute_signal_semaphore)
            .wait_dst_stage_mask(&compute_stages)
            .command_buffers(from_ref(&compute_buffer))
            .push_next(&mut timeline_info_compute);

        let (download_wait_semaphore, download_wait_values) = timeline.get_wait_info(counter, AudioSyncStage::Download);
        let (download_signal_semaphore, download_signal_values) = timeline.get_signal_info(counter, AudioSyncStage::Download);
        let download_stages = vec![PipelineStageFlags::BOTTOM_OF_PIPE; download_wait_semaphore.len()];
        let mut timeline_info_download = TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&download_wait_values)
            .signal_semaphore_values(&download_signal_values);
        let submit_info_download = SubmitInfo::default()
            .wait_semaphores(&download_wait_semaphore)
            .signal_semaphores(&download_signal_semaphore)
            .wait_dst_stage_mask(&download_stages)
            .command_buffers(from_ref(&download_buffer))
            .push_next(&mut timeline_info_download);

        println!("SUBMITTING AUDIO");
        self.device.queue_submit(self.transfer_queue.handle, &[submit_info_upload, submit_info_download], fences.0)?;
        self.device.queue_submit(self.compute_queue.handle, &[submit_info_compute], fences.1)?;

        let gpu_record_time = Instant::now();

        // Schedule a submission of this audio frame data back to the consumer
        let device = self.device.clone();
        let thread_timeline = timeline.clone();
        self.transfer_module.schedule_submit_task(Box::new(move |
            download_buffer: &mut LocalVulkanBuffer<DownloadBufferData>
        | unsafe {
            let thread_initial_time = Instant::now();

            let (wait_semaphores, wait_values) = thread_timeline.get_wait_info(counter, AudioSyncStage::Submit);
            let wait_info = SemaphoreWaitInfo::default()
                .semaphores(&wait_semaphores)
                .values(&wait_values);

            // Wait until this job is ready to be submitted
            device
                .wait_semaphores(&wait_info, u64::MAX)
                .expect("Failed to wait for audio sync");

            let execution_time = Instant::now();

            // Get the data from the local buffer
            download_buffer.invalidate();
            let (left_window, right_window) = download_buffer.buffer_data().get_windows();

            // Signal that the audio sync is finished, and the next download can start
            let (signal_semaphores, signal_values) = thread_timeline.get_signal_info(counter, AudioSyncStage::Submit);
            let signal_info = SemaphoreSignalInfo::default()
                .semaphore(signal_semaphores[0])
                .value(signal_values[0]);
            device.signal_semaphore(&signal_info);

            let copy_time = Instant::now();

            // Transform the data into frames
            let left = FftModule::local_fourier_transform(window_to_vec(left_window), true);
            let right = FftModule::local_fourier_transform(window_to_vec(right_window), true);

            // todo: AVOID COPIES, perhaps make a single function that does all of this
            let (start, end) = DownloadBufferData::last_frame_range();
            let left = gpu_to_frame(&GpuFrame::try_from(&left[start..end]).unwrap());
            let right = gpu_to_frame(&GpuFrame::try_from(&right[start..end]).unwrap());

            let misha_time = Instant::now();

            // Submit to consumer
            consumer(left, right);

            println!("Submit thread time elapsed: wait {:?}; copy+signal {:?}; misha {:?}; submit {:?}", execution_time - thread_initial_time, copy_time - execution_time, misha_time - copy_time, misha_time.elapsed())
        }));

        timeline.advance_frame(counter);
        self.audio_counter = self.audio_counter.next();

        println!("Main thread time elapsed: fence {:?}; record {:?}; end {:?}", fence_time - start_time, gpu_record_time - fence_time, gpu_record_time.elapsed());

        Ok(())
    }

    // todo: currently this updates the only instance buffer: might cause bad reads!
    unsafe fn update_instances(&mut self, store_rays: bool) -> VkResult<bool> {
        if store_rays {
            self.ray_module.copy_next_debug = true;
        }

        let timeline = &mut self.rt_timeline;
        let counter = self.rt_counter;
        let fence = self.rt_fence;
        
        // If the previous instance update is still happening, simply return
        if !self.device.get_fence_status(fence)? {
            return Ok(false);
        }

        self.device.reset_fences(&[fence])?;

        // Reset the command buffer
        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT); // todo: this might not be ideal

        self.device.reset_command_buffer(self.rt_command_buffers.0, CommandBufferResetFlags::empty())?;
        self.device.reset_command_buffer(self.rt_command_buffers.1, CommandBufferResetFlags::empty())?;
        self.device.begin_command_buffer(self.rt_command_buffers.0, &begin_info)?;
        self.device.begin_command_buffer(self.rt_command_buffers.1, &begin_info)?;

        // Upload the sources
        self.ray_module.upload_sources(&mut self.rt_command_buffers.0, &self.scene.sources);

        // Trace the rays
        self.ray_module.shoot_rays(&mut self.rt_command_buffers.0, MAX_SOURCES as u32, self.scene.listener.location, store_rays);

        // Copy the result of shooting rays locally
        self.ray_module.copy_ray_buffer(&mut self.rt_command_buffers.0);

        // Copy the results of shooting rays to the cluster module
        self.cluster_module.copy_rt_output(&mut self.rt_command_buffers.0);

        // Cluster the result
        self.cluster_module.cluster_hits_async(timeline.clone(), counter);

        // Upload the cluster to the instance buffer
        // todo: move the buffers to be managed by the cluster module? figure out a better way perhaps
        self.cluster_module.upload_to_buffer(&mut self.rt_command_buffers.1);

        // Submit the command buffer
        let (mut wait_semaphore_0, mut wait_value_0) = timeline.get_wait_info(counter, RtSyncStage::RayTrace);
        let (signal_semaphore_0, signal_value_0) = timeline.get_signal_info(counter, RtSyncStage::RayTrace);
        // Add a dependency to the compute stage of the next audio timeline, to avoid changing the current values until that's done
        wait_semaphore_0.push(self.audio_timeline.semaphores[self.audio_counter]);
        wait_value_0.push(self.audio_timeline.completed_stage_val(0, self.audio_counter, AudioSyncStage::Compute));
        let mut timeline_info_0 = TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_value_0)
            .signal_semaphore_values(&signal_value_0);
        let submit_info_0 = SubmitInfo::default()
            .wait_semaphores(&wait_semaphore_0)
            .signal_semaphores(&signal_semaphore_0)
            .command_buffers(from_ref(&self.rt_command_buffers.0))
            .wait_dst_stage_mask(&[PipelineStageFlags::BOTTOM_OF_PIPE; 1])
            .push_next(&mut timeline_info_0);

        let (wait_semaphore_1, wait_value_1) = timeline.get_wait_info(counter, RtSyncStage::Upload);
        let (signal_semaphore_1, signal_value_1) = timeline.get_signal_info(counter, RtSyncStage::Upload);
        let mut timeline_info_1 = TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_value_1)
            .signal_semaphore_values(&signal_value_1);
        let submit_info_1 = SubmitInfo::default()
            .wait_semaphores(&wait_semaphore_1)
            .signal_semaphores(&signal_semaphore_1)
            .command_buffers(from_ref(&self.rt_command_buffers.1))
            .wait_dst_stage_mask(&[PipelineStageFlags::BOTTOM_OF_PIPE; 1])
            .push_next(&mut timeline_info_1);

        let (wait_semaphore_mid, wait_value_mid) = timeline.get_wait_info(counter, RtSyncStage::Cluster);
        let (signal_semaphore_mid, signal_value_mid) = timeline.get_signal_info(counter, RtSyncStage::Cluster);

        self.device.end_command_buffer(self.rt_command_buffers.0)?;
        self.device.end_command_buffer(self.rt_command_buffers.1)?;

        println!("SUBMITTING RT");
        self.device.queue_submit(self.async_queue.handle, &[submit_info_0, submit_info_1], fence)?;

        timeline.advance_frame(counter);
        self.rt_counter = self.rt_counter.next();

        Ok(true)
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

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
enum RtSyncStage {
    RayTrace = 0,
    Cluster = 1,
    Upload = 2,
}
impl PipelineStage for RtSyncStage {
    fn val(&self) -> i64 { *self as i64 }
    fn num_stages() -> i64 { 3 }
    fn last() -> Self { Self::Upload }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
enum AudioSyncStage {
    Upload = 0,
    Compute = 1,
    Download = 2,
    Submit = 3,
}
impl PipelineStage for AudioSyncStage {
    fn val(&self) -> i64 { *self as i64 }
    fn num_stages() -> i64 { 4 }
    fn last() -> Self { Self::Submit }
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

pub(crate) unsafe fn window_to_vec(window: Box<GpuWindow>) -> Vec<Vec2> { unsafe {
    // transmute the pointer without performing a copy; arrays in rust are guaranteed to be sequential
    let flat_window = transmute::<Box<GpuWindow>, Box<[Vec2; GPU_WINDOW_SIZE]>>(window);
    (flat_window as Box<[_]>).into_vec() // turn the box into a vector
}}


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

        // todo: replace with a memcopy perhaps?
        for (idx, val) in instances.iter().enumerate() {
            self.instances[idx] = *val;
        }
    }
}
