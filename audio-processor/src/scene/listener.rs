use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::scene::listener::hrtf_filter::{HrtfFilter, HrtfOptions};
use crevice::std430::{Mat3, Vec2, Vec3};
use std::f32::consts::PI;

pub mod hrtf_filter;

pub(crate) struct AudioListener {
    pub(crate) rotation: Mat3,
    pub(crate) location: Vec3,
    pub(crate) filter: HrtfFilter
}

impl AudioListener {
    pub(crate) fn new(location: Vec3, rotation: Mat3, filter: HrtfFilter) -> Self {
        Self {
            location,
            rotation,
            filter
        }
    }
}
