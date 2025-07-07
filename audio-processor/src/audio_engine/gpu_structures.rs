use std::alloc::{alloc_zeroed, Layout};
use crate::audio_engine::ray_tracer::rays::SPHERE_POINTS;
use crate::audio_engine::signal_processor::transfer::copy_to_box;
use crate::audio_engine::GpuData;
use crate::scene::source::{AudioSource, FRAME_SIZE};
use crate::scene::Scene;
use crate::util::vec3;
use crevice::std430::{AsStd430, Vec2, Vec3, Vec4};
use std::mem::ManuallyDrop;
use std::slice;

pub(crate) trait FromMemoryMap : Sized {
    unsafe fn from_memory_map(pointer: *mut u8) -> ManuallyDrop<Box<Self>> { unsafe {
        if !pointer.cast::<Self>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        ManuallyDrop::new(Box::from_raw(pointer.cast()))
    }}

    unsafe fn to_local_copy(&self) -> Box<Self> { unsafe {
        // Allocate the required space
        let layout = Layout::new::<Self>();
        let dst = alloc_zeroed(layout) as *mut Self;
        let src = self as *const Self;

        // Copy the memory value
        std::ptr::copy_nonoverlapping(src, dst, 1);

        // Wrap in a box
        Box::from_raw(dst)
    }}
}

#[derive(AsStd430)]
pub(crate) struct FftUbo {
    pub(crate) split_size: u32,
    pub(crate) radix_stride: u32,
    pub(crate) angle_direction_factor: f32,
    pub(crate) angle_spin_factor: f32,
    pub(crate) normalization_factor: f32,
}

// todo: have buffers take ownership of (and even create) their own stuff!
//       using the newtype pattern?


pub(crate) struct FftBuffer {
    pub windows: [GpuWindow; MAX_INSTANCES]
}

impl FftBuffer {
    pub(crate) fn max_size() -> usize {
        MAX_INSTANCES * GPU_WINDOW_SIZE * size_of::<Vec2>()
    }
}


pub(crate) const MAX_SOURCES: usize = 1;

/// Represents a single audio frame on the GPU
pub(crate) type GpuFrame = [Vec2; FRAME_SIZE];

/// The upload buffer contains the stream data that gets uploaded to the GPU every frame.
#[repr(C)]
pub(crate) struct UploadBuffer {
   pub frames: [GpuFrame; MAX_SOURCES]
}

impl GpuData for UploadBuffer {
    unsafe fn serialize(&self, dst: *mut u8) { unsafe {
        std::ptr::copy_nonoverlapping(
            (self as *const UploadBuffer).cast(),
            dst,
            size_of::<UploadBuffer>()
        );
    }}

    fn size(&self) -> usize {
        Self::max_size()
    }
}

impl UploadBuffer {
    pub(crate) fn new() -> Self {
        Self {
            frames: [[Vec2 { x: 0.0, y: 0.0 }; FRAME_SIZE]; MAX_SOURCES]
        }
    }

    pub(crate) unsafe fn from_memory_map(pointer: *mut u8) -> ManuallyDrop<Box<Self>> { unsafe {
        if !pointer.cast::<Self>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        ManuallyDrop::new(Box::from_raw(pointer.cast()))
    }}

    pub(crate) fn max_size() -> usize {
        size_of::<Self>()
    }
}


pub(crate) const SLIDING_WINDOW_FRAME_AMT: usize = 2;
pub(crate) const GPU_WINDOW_SIZE: usize = FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT;

pub(crate) type GpuWindow = [GpuFrame; SLIDING_WINDOW_FRAME_AMT]; // represents a sliding window of audio frames

/// CPU-mapped buffer responsible for downloading the two sliding windows from the GPU
pub(crate) struct DownloadBuffer {
    pub windows: [GpuWindow; 2]
}

impl DownloadBuffer {
    pub(crate) unsafe fn from_memory_map(pointer: *mut u8) -> ManuallyDrop<Box<Self>> { unsafe {
        if !pointer.cast::<Self>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        ManuallyDrop::new(Box::from_raw(pointer.cast()))
    }}

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

    pub(crate) fn max_size() -> usize {
        size_of::<Self>()
    }
}


/// Instances represent "audio streams" and their locations.
/// This allows a single stream to have an HRTF applied from multiple locations.
/// The stream is represented as an index within the Stream Buffer.
pub(crate) const MAX_INSTANCES: usize = 64; // todo: currently limited by the synchronization issues

#[derive(AsStd430, Clone)]
#[repr(align(16))]
pub struct AudioInstance {
    pub direction: Vec3,
    distance: f32,
    index: u32,
}

// todo: get rid of the variable size here, these buffers should only have fixed-size and not be reference types (ie. directly mappable)
pub struct InstanceBuffer {
    pub instances: Vec<AudioInstance>,
}

impl GpuData for InstanceBuffer {
    unsafe fn serialize(&self, dst: *mut u8) { unsafe {
        std::ptr::copy_nonoverlapping(
            (&self.instances[..] as *const [AudioInstance]).cast(),
            dst,
            size_of::<AudioInstance>() * self.instances.len()
        );
    }}

    fn size(&self) -> usize {
        self.instances.len() * 32
    }
}

impl FromMemoryMap for InstanceBuffer {
    unsafe fn from_memory_map(pointer: *mut u8) -> ManuallyDrop<Box<Self>> { unsafe {
        if !pointer.cast::<*mut AudioInstance>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        let buffer = InstanceBuffer {
            instances: slice::from_raw_parts(pointer.cast(), MAX_INSTANCES).to_vec()
        };

        ManuallyDrop::new(buffer.into())
    }}
}

impl InstanceBuffer {
    pub(crate) fn from_scene_data(scene: &Scene) -> Self {
        let mut instance = Self {
            instances: vec![]
        };

        for (idx, source) in scene.sources.iter().enumerate() {
            instance.instances.push(AudioInstance {
                direction: source.coordinates,
                distance: vec3::len(source.coordinates),
                index: idx as u32,
            });
        }
        
        instance
    }

    pub(crate) fn truncate(&mut self, instance_amt: usize) {
        assert!(self.instances.len() >= instance_amt);
        self.instances.truncate(instance_amt);
    }

    pub(crate) fn max_size() -> usize {
        32 * MAX_INSTANCES
    }
}


/// Delay buffer responsible for holding the delayed audio. This only gets uploaded
pub(crate) const MAX_DELAY_FRAMES: usize = 430;  // = ~5s @ 44100Hz, rounded to %WINDOW_SIZE

pub(crate) struct DelayBuffer();

impl DelayBuffer {
    pub(crate) fn max_size() -> usize {
        MAX_SOURCES * FRAME_SIZE * MAX_DELAY_FRAMES * size_of::<Vec2>()
    }
}


///////////////////////////////////////////
/////////////// RAY TRACING ///////////////
///////////////////////////////////////////
pub(crate) struct SourcesBuffer {
    sources: [Vec3; MAX_SOURCES]
}

impl SourcesBuffer {
    pub(crate) unsafe fn from_memory_map(pointer: *mut u8) -> ManuallyDrop<Box<Self>> { unsafe {
        if !pointer.cast::<Self>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        ManuallyDrop::new(Box::from_raw(pointer.cast()))
    }}

    pub(crate) unsafe fn copy_coordinates(&mut self, sources: &Vec<AudioSource>) {
        for (idx, source) in sources.iter().enumerate() {
            self.sources[idx] = source.coordinates;
        }
    }

    pub(crate) fn max_size() -> usize {
        size_of::<SourcesBuffer>()
    }
}

/// Ray Tracing output buffer
#[derive(AsStd430)]
#[repr(align(16))]
pub(crate) struct Output {
    direction: Vec3,
    total_distance: f32,
    bounces: u32,
    source: u32,
}

pub(crate) struct OutputBuffer();
impl OutputBuffer {
    pub(crate) fn max_size() -> usize {
        MAX_SOURCES * SPHERE_POINTS * 32
    }
}
