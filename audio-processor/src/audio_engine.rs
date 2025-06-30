#[allow(unsafe_op_in_unsafe_fn)]

use crate::audio_engine::buffer_initializer::BufferInitializer;
use crate::audio_engine::gpu_structures::{GpuFrame, GpuWindow, InstanceBuffer};
use crate::audio_engine::ray_tracer::RayTracer;
use crate::audio_engine::signal_processor::SignalProcessor;
use crate::scene::source::{Frame, FRAME_SIZE};
use crate::scene::Scene;
use ash::ext::debug_utils;
use ash::vk::{ApplicationInfo, Buffer, DeviceCreateInfo, DeviceQueueCreateInfo, DeviceSize, InstanceCreateInfo, PhysicalDeviceFeatures2, PhysicalDeviceShaderAtomicFloatFeaturesEXT, Queue};
use ash::{vk, vk::{DebugUtilsMessengerEXT, PhysicalDevice}, Device, Entry, Instance};
use crevice::std430::{Mat3, Vec2, Vec3};
use std::array::from_ref;
use std::borrow::Cow;
use std::ffi::{c_char, CStr};
use std::{fs::File, path::Path};
use vk_mem::Allocation;

pub(crate) mod signal_processor;
pub(crate) mod gpu_structures;
pub(crate) mod ray_tracer;
mod buffer_initializer;

struct GpuBuffer {
    buffer: Buffer,
    buffer_memory: Allocation,
    size: usize,
}

pub(crate) trait GpuData {
    unsafe fn serialize(&self, dst: *mut u8);
    fn size(&self) -> usize;
}

pub struct AudioEngine {
    scene: Scene,

    entry: Entry,
    instance: Instance,
    debug_callback: DebugUtilsMessengerEXT,
    gpu: PhysicalDevice,
    device: Device,

    compute_queue: Queue,

    buffer_initializer: BufferInitializer,
    signal_processor: SignalProcessor,
    ray_tracer: RayTracer,
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
        let (gpu, queue_family_index, device) = {
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
        let compute_queue = device.get_device_queue(queue_family_index, 0);

        let mut buffer_initializer = BufferInitializer::new(
            &instance,
            &device,
            &gpu,
            queue_family_index
        );

        let signal_processor = SignalProcessor::new(
            &scene,
            &instance,
            &gpu,
            device.clone(),
            (compute_queue, queue_family_index),
            (compute_queue, queue_family_index),
            &mut buffer_initializer,
        );

        let ray_tracer = RayTracer::new(
            &scene,
            &instance,
            &gpu,
            device.clone(),
            (compute_queue, queue_family_index),
            &mut buffer_initializer,
        );

        Self {
            scene,
            entry,
            instance,
            debug_callback,
            gpu,
            device,
            compute_queue,
            buffer_initializer,
            signal_processor,
            ray_tracer,
        }
    }}

    pub(crate) fn process_frames(&mut self) -> (Frame, Frame) {
        unsafe {
            // self.ray_tracer.trace_rays(&self.scene);
            self.ray_tracer.copy_sources_debug(&self.scene);

            let mut src_audio_instances = self.ray_tracer.get_instance_buffer();
            let mut dst_audio_instances = self.signal_processor.get_instance_buffer();

            self.buffer_initializer.copy_buffer(
                &self.device, 
                self.compute_queue, 
                &mut src_audio_instances, 
                &mut dst_audio_instances, 
                InstanceBuffer::max_size() as DeviceSize
            );

            let (left, right) = self.signal_processor.process_frames(
                &self.scene.listener,
                &mut self.scene.sources,
                self.ray_tracer.get_last_rt_pos()
            );

            (
                gpu_to_frame(&left),
                gpu_to_frame(&right)
            )
        }
    }

    // pub(crate) fn get_fft(&mut self) -> (GpuWindow, GpuWindow) {
    //     unsafe {
    //         // self.ray_tracer.trace_rays(&self.scene);
    //         self.ray_tracer.copy_sources_debug(&self.scene);
    // 
    //         let mut src_audio_instances = self.ray_tracer.get_instance_buffer();
    //         let mut dst_audio_instances = self.signal_processor.get_instance_buffer();
    // 
    //         self.buffer_initializer.copy_buffer(
    //             &self.device,
    //             self.compute_queue,
    //             &mut src_audio_instances,
    //             &mut dst_audio_instances,
    //             InstanceBuffer::max_size() as DeviceSize
    //         );
    // 
    //         let last_rt_pos = self.ray_tracer.get_last_rt_pos();
    //         self.signal_processor.get_fft(&mut self.scene, last_rt_pos)
    //     }
    // }

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

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 { unsafe {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}}

// Util function for submodules
fn read_file_words(path: impl AsRef<Path>) -> Vec<u32> {
    let path = path.as_ref();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let mut file = File::open(&path).unwrap();

    ash::util::read_spv(&mut file).unwrap()
}

pub(crate) fn frame_to_gpu(frame: &Frame) -> GpuFrame {
    // todo: avoid initialization here
    let mut samples = [Vec2{x: 0.0, y: 0.0}; FRAME_SIZE];

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
