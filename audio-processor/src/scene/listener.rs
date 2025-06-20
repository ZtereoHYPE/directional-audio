use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::scene::listener::hrtf_filter::{HrtfFilter, HrtfOptions};
use crevice::std430::Vec3;
use std::f32::consts::PI;

pub mod hrtf_filter;

pub(crate) struct AudioListener {
    pub(crate) location: Vec3,
    pub(crate) filter: HrtfFilter
}

impl AudioListener {
    pub(crate) fn new(location: Vec3, filter: &str, filter_options: HrtfOptions) -> Self {
        let filter = HrtfFilter::new(filter_options, filter, GPU_WINDOW_SIZE);

        Self {
            location,
            filter
        }
    }
}
