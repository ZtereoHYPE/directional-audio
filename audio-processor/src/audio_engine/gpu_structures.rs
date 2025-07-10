use std::alloc::{alloc_zeroed, Layout};
use crate::audio_engine::ray_tracer::rays::SPHERE_POINTS;
use crate::audio_engine::signal_processor::transfer::copy_to_box;
use crate::audio_engine::DynamicBufferData;
use crate::scene::source::{AudioSource, FRAME_SIZE};
use crate::scene::Scene;
use crate::util::vec3;
use crevice::std430::{AsStd430, Vec2, Vec3, Vec4};
use std::mem::ManuallyDrop;
use std::slice;
use std::slice::from_ref;
use ash::vk::{BufferCopy, DeviceSize};
use crate::vulkan::buffer::BufferData;


#[derive(AsStd430)]
pub(crate) struct FftUboData {
    pub(crate) split_size: u32,
    pub(crate) radix_stride: u32,
    pub(crate) angle_direction_factor: f32,
    pub(crate) angle_spin_factor: f32,
    pub(crate) normalization_factor: f32,
}
impl BufferData for FftUboData {}


pub(crate) struct FftBufferData {
    pub windows: [GpuWindow; MAX_INSTANCES]
}
impl BufferData for FftBufferData {}


pub(crate) const MAX_SOURCES: usize = 1;

/// Represents a single audio frame on the GPU
pub(crate) type GpuFrame = [Vec2; FRAME_SIZE];

/// The upload buffer contains the stream data that gets uploaded to the GPU every frame.
#[repr(C)]
pub(crate) struct UploadBufferData {
   pub frames: [GpuFrame; MAX_SOURCES]
}

impl BufferData for UploadBufferData {}

impl UploadBufferData {
    pub(crate) fn new() -> Self {
        Self {
            frames: [[Vec2 { x: 0.0, y: 0.0 }; FRAME_SIZE]; MAX_SOURCES]
        }
    }
}


pub(crate) const SLIDING_WINDOW_FRAME_AMT: usize = 2;
pub(crate) const GPU_WINDOW_SIZE: usize = FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT;

pub(crate) type GpuWindow = [GpuFrame; SLIDING_WINDOW_FRAME_AMT]; // represents a sliding window of audio frames

// todo: rename to something more descriptive
pub(crate) struct DownloadBufferData {
    pub windows: [GpuWindow; 2]
}

impl BufferData for DownloadBufferData {}

impl DownloadBufferData {
    pub(crate) unsafe fn get_windows(&self) -> (Box<GpuWindow>, Box<GpuWindow>) { unsafe {
        let left = copy_to_box(&self.windows[0] as *const GpuWindow);
        let right = copy_to_box(&self.windows[1] as *const GpuWindow);

        (left, right)
    }}

    pub(crate) const fn last_frame_range() -> (usize, usize) {
        (
            FRAME_SIZE * (SLIDING_WINDOW_FRAME_AMT - 1),
            FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT,
        )
    }
}


/// Instances represent "audio streams" and their locations.
/// This allows a single stream to have an HRTF applied from multiple locations.
/// The stream is represented as an index within the Stream Buffer.
pub(crate) const MAX_INSTANCES: usize = 64; // todo: currently limited by the synchronization issues

#[derive(AsStd430, Clone)]
#[repr(align(16))]
#[derive(Copy)]
pub struct AudioInstance {
    pub direction: Vec3,
    distance: f32,
    index: u32,
}

pub struct InstanceBufferData {
    pub instances: [AudioInstance; MAX_INSTANCES],
}

impl BufferData for InstanceBufferData {}

impl InstanceBufferData {
    pub(crate) fn from_scene_data(scene: &Scene) -> Self {
        let mut instance = Self {
            instances: [AudioInstance {direction: vec3::ZERO, distance: 0.0, index: 0}; MAX_INSTANCES]
        };

        for (idx, source) in scene.sources.iter().enumerate() {
            instance.instances[idx] = (AudioInstance {
                direction: source.coordinates,
                distance: vec3::len(source.coordinates),
                index: idx as u32,
            });
        }

        instance
    }
}


/// Delay buffer responsible for holding the delayed audio. This only gets uploaded
pub(crate) const MAX_DELAY_FRAMES: usize = 430;  // = ~5s @ 44100Hz, rounded to %WINDOW_SIZE

pub(crate) struct DelayBufferData {
    frames: [[[Vec2; FRAME_SIZE]; MAX_DELAY_FRAMES]; MAX_SOURCES]
}

impl BufferData for DelayBufferData {}


///////////////////////////////////////////
/////////////// RAY TRACING ///////////////
///////////////////////////////////////////
/// Ray Tracing output buffer
#[derive(AsStd430)]
#[repr(align(16))]
pub(crate) struct Output {
    direction: Vec3,
    total_distance: f32,
    bounces: u32,
    source: u32,
}

pub(crate) struct RtOutputBufferData {
    outputs: [Output; MAX_SOURCES * SPHERE_POINTS]
}

impl BufferData for RtOutputBufferData {}
