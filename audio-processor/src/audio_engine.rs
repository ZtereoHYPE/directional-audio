#[allow(unsafe_op_in_unsafe_fn)]

use crate::audio_engine::buffer_uploader::BufferUploader;
use crate::audio_engine::gpu_structures::{GpuFrame, GPU_WINDOW_SIZE};
use crate::audio_engine::ray_tracer::RayTracer;
use crate::audio_engine::signal_processor::SignalProcessor;
use crate::scene::hrtf_filter::{HrtfFilter, HrtfOptions};
use crate::scene::{Frame, FRAME_SIZE};
use ash::ext::debug_utils;
use ash::vk::{ApplicationInfo, Buffer, DeviceCreateInfo, DeviceQueueCreateInfo, InstanceCreateInfo};
use ash::{vk, vk::{DebugUtilsMessengerEXT, PhysicalDevice}, Device, Entry, Instance};
use crevice::std430::Vec2;
use std::array::from_ref;
use std::borrow::Cow;
use std::f32::consts::PI;
use std::ffi::{c_char, CStr};
use std::{fs::File, path::Path};
use vk_mem::Allocation;

pub(crate) mod signal_processor;
pub(crate) mod gpu_structures;
mod ray_tracer;
mod buffer_uploader;

struct GpuBuffer {
    buffer: Buffer,
    buffer_memory: Allocation,
    size: usize,
}

pub(crate) trait GpuData {
    unsafe fn serialize(&self, dst: *mut u8);
    unsafe fn deserialize(src: *const u8) -> Box<Self>;
    fn size(&self) -> usize;
}

pub struct AudioEngine {
    entry: Entry,
    instance: Instance,
    debug_callback: DebugUtilsMessengerEXT,
    gpu: PhysicalDevice,
    device: Device,

    buffer_uploader: BufferUploader,
    signal_processor: SignalProcessor,
    ray_tracer: RayTracer,
}

impl AudioEngine {
    pub(crate) unsafe fn new() -> Self {
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
                .pfn_user_callback(Some(Self::debug_callback));

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);

            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        // todo: better logic for selecting device and queue(s)
        // igpu detection could happen here to better adapt things
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

        // todo: better detection and selection of the various queues
        let compute_queue = device.get_device_queue(queue_family_index, 0);

        let filter_options = HrtfOptions {
            azimuth_samples: 180, // one every 2 deg
            elevation_samples: 90, // one every 2 deg
            elevation_max: PI, // full sphere was captured
            elevation_min: 0.0, // "
            sampling_rate: 44100.0
        };

        let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa", GPU_WINDOW_SIZE); //todo: explore with lower size...

        let mut buffer_uploader = BufferUploader::new(
            &instance,
            &device,
            &gpu,
            queue_family_index
        );

        let signal_processor = SignalProcessor::new(
            &instance,
            &gpu,
            device.clone(),
            (compute_queue, queue_family_index),
            (compute_queue, queue_family_index),
            &mut buffer_uploader,
            filter
        );

        let ray_tracer = RayTracer::new();

        Self {
            entry,
            instance,
            debug_callback,
            gpu,
            device,
            buffer_uploader,
            signal_processor,
            ray_tracer,
        }
    }

    pub(crate) fn process_frames(&mut self, frames: Vec<Frame>) -> (Frame, Frame) {
        unsafe {
            let gpu_frames = frames.iter().map(|f| frame_to_gpu(f)).collect();
            let (left, right) = self.signal_processor.process_frames(gpu_frames);

            (
                gpu_to_frame(&left),
                gpu_to_frame(&right)
            )
        }
    }

    pub(crate) fn process_frames_frequency(&mut self, frames: Vec<Frame>) -> (Vec<Vec2>, Vec<Vec2>) {
        unsafe {
            let gpu_frames = frames.iter().map(|f| frame_to_gpu(f)).collect();
            self.signal_processor.process_frames_frequency(gpu_frames)
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
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        // todo: Cleanup!
    }
}

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
