use crate::scene::listener::hrtf_filter::HrtfFilter;
use glam::{Mat3, Vec3};

pub mod hrtf_filter;

pub(crate) struct AudioListener {
    pub(crate) rotation: Mat3,
    pub(crate) prev_rotation: Mat3,
    pub(crate) location: Vec3,
    pub(crate) filter: HrtfFilter
}

impl AudioListener {
    pub(crate) fn new(location: Vec3, rotation: Mat3, filter: HrtfFilter) -> Self {
        Self {
            location,
            rotation,
            prev_rotation: Mat3::IDENTITY,
            filter
        }
    }
}
