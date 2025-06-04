use crate::audio_engine::GpuData;
use crate::scene::FRAME_SIZE;
use crevice::std430::{AsStd430, Vec2, Vec4};
use std::mem::ManuallyDrop;

#[derive(AsStd430)]
pub(crate) struct FftUbo {
    pub(crate) split_size: u32,
    pub(crate) radix_stride: u32,
    pub(crate) angle_direction_factor: f32,
    pub(crate) angle_spin_factor: f32,
    pub(crate) normalization_factor: f32,
}

#[repr(C)]
pub(crate) struct FftConstants {
    pub(crate) radix: i32,
    pub(crate) frame_size: i32
}


/// The stream buffer contains the stream data that gets uploaded to the GPU every frame.
/// This includes a sliding window for each frame stream for partitioned convolution.

pub(crate) const MAX_STREAMS: usize = 1; // todo: experiment with more!
pub(crate) const SLIDING_WINDOW_FRAME_AMT: usize = 2;
pub(crate) const GPU_WINDOW_SIZE: usize = FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT;

pub(crate) type GpuFrame = [Vec2; FRAME_SIZE];  // Represents a single audio frame
pub(crate) type GpuWindow = [GpuFrame; SLIDING_WINDOW_FRAME_AMT]; // represents a sliding window of audio frames

#[repr(C)]
pub(crate) struct StreamBuffer {
   pub windows: [GpuWindow; MAX_STREAMS]
}

impl GpuData for StreamBuffer {
    unsafe fn serialize(&self, dst: *mut u8) {
        std::ptr::copy_nonoverlapping(
            (self as *const StreamBuffer).cast(),
            dst,
            size_of::<StreamBuffer>()
        );
    }

    unsafe fn deserialize(_: *const u8) -> Box<Self> {
        panic!("The Stream Buffer should only be uploaded!")
    }

    fn size(&self) -> usize {
        Self::size()
    }
}

impl StreamBuffer {
    pub(crate) fn new() -> Self {
        Self {
            windows: [[[Vec2 { x: 0.0, y: 0.0}; FRAME_SIZE]; SLIDING_WINDOW_FRAME_AMT]; MAX_STREAMS]
        }
    }

    pub(crate) unsafe fn from_memory_map(pointer: *mut u8) -> ManuallyDrop<Box<Self>> {
        if !pointer.cast::<Self>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        ManuallyDrop::new(Box::from_raw(pointer.cast()))
    }
    
    // todo: improve by allowing to insert individual frames at an index to not have to allocate a vector
    pub(crate) fn insert_frames(&mut self, frames: Vec<GpuFrame>) {
        for (idx, window) in self.windows.iter_mut().enumerate() {
            window.copy_within(1..SLIDING_WINDOW_FRAME_AMT, 0); // slide the elements 1-4 to 0-3
            window[SLIDING_WINDOW_FRAME_AMT - 1] = frames[idx];
        }
    }

    pub(crate) const fn last_frame_range() -> (usize, usize) {
        (
            FRAME_SIZE * (SLIDING_WINDOW_FRAME_AMT - 1),
            FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT,
        )
    }

    pub(crate) fn size() -> usize {
        size_of::<Self>()
    }
}

/// Virtual Sources represent "audio streams" and their locations.
/// This allows a single stream to have an HRTF applied from multiple locations.
/// The stream is represented as an index within the Stream Buffer.
pub(crate) const MAX_VIRTUAL_SOURCES: usize = 512;

pub(crate) struct VirtualSources {
    sources: Vec<Vec4>,
}

impl GpuData for VirtualSources {
    unsafe fn serialize(&self, dst: *mut u8) {
        std::ptr::copy_nonoverlapping(
            (&self.sources[..] as *const [Vec4]).cast(),
            dst,
            size_of::<Vec4>() * self.sources.len()
        );
    }

    unsafe fn deserialize(_: *const u8) -> Box<Self> {
        panic!("UBOs should only be uploaded!")
    }

    fn size(&self) -> usize {
        self.sources.len() * size_of::<Vec4>()
    }
}

impl VirtualSources {
    pub(crate) fn new() -> Self {
        Self {
            sources: Vec::new()
        }
    }

    pub(crate) fn push_source(&mut self, x: f32, y: f32, z: f32, audio_stream_idx: u32) {
        self.sources.push(Vec4 {
            x, y, z,
            w: f32::from_bits(audio_stream_idx)
        })
    }

    pub(crate) fn size() -> usize {
        size_of::<Vec4>() * MAX_VIRTUAL_SOURCES
    }
}