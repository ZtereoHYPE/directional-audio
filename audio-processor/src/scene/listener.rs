use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::scene::listener::hrtf_filter::{HrtfFilter, HrtfOptions};
use crevice::std430::Vec3;
use std::f32::consts::PI;

pub(crate) mod hrtf_filter;

pub(crate) struct AudioListener {
    pub(crate) location: Vec3,
    pub(crate) filter: HrtfFilter
}

impl AudioListener {
    pub(crate) fn new(location: Vec3) -> Self {
        let filter_options = HrtfOptions {
            azimuth_samples: 90, // one every 2 deg
            elevation_samples: 45, // one every 2 deg
            elevation_max: 0.0, // full sphere was captured
            elevation_min: PI, // "
            sampling_rate: 44100.0
        };

        let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa", GPU_WINDOW_SIZE);

        Self {
            location,
            filter
        }
    }
}
