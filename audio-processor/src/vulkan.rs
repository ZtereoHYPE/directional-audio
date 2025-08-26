use ash::{vk, Instance};
use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::path::Path;
use ash::vk::PhysicalDevice;

pub(crate) mod buffer;
pub(crate) mod buffer_initializer;
pub(crate) mod spec_constants;
pub(crate) mod queue;
pub(crate) mod in_flight;
pub(crate) mod timeline;

const USE_CPU: bool = false;

pub(crate) unsafe extern "system" fn debug_callback(
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
pub(crate) fn read_spirv_words(path: impl AsRef<Path>) -> Vec<u32> {
    let path = path.as_ref();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let mut file = File::open(&path).unwrap();

    ash::util::read_spv(&mut file).unwrap()
}

pub(crate) fn get_device_score(instance: &Instance, gpu: PhysicalDevice) -> Option<i32> {
    // Query timelineSemaphore support
    let mut timeline_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut timeline_features);

    unsafe { instance.get_physical_device_features2(gpu, &mut features2) };

    if timeline_features.timeline_semaphore == vk::FALSE {
        return None; // device doesn't support timeline semaphores
    }

    let device_properties = unsafe {instance.get_physical_device_properties(gpu)};
    match device_properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => Some(4),
        vk::PhysicalDeviceType::INTEGRATED_GPU => Some(3),
        vk::PhysicalDeviceType::VIRTUAL_GPU => Some(2),
        vk::PhysicalDeviceType::CPU => if USE_CPU {Some(5)} else {Some(0)},
        _ => Some(1),
    }
}
