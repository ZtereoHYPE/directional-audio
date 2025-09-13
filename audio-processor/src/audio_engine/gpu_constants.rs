use crate::scene::source::FRAME_SIZE;

/// Maximum concurrent sources allowed to upload new audio frames to the GPU
pub(crate) const MAX_SOURCES: usize = 8;

/// Maximum concurrent sources allowed to upload new audio frames to the GPU
pub(crate) const MAX_BOUNCES: usize = 6;

pub(crate) const SLIDING_WINDOW_FRAME_AMT: usize = 2;
pub(crate) const GPU_WINDOW_SIZE: usize = FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT;

/// The maximum total amount of audio "instances". An instance 
pub(crate) const MAX_INSTANCES: usize = 512;

/// Delay buffer responsible for holding the delayed audio. This only gets uploaded
pub(crate) const MAX_DELAY_FRAMES: usize = 430;  // = ~5s @ 44100Hz, rounded to %WINDOW_SIZE

pub(crate) const SPHERE_POINTS: usize = 4096;
