mod file;
mod frequency;

use crate::audio_engine::gpu_structures::GpuFrame;
use crevice::std430::Vec3;
use rand::Rng;

pub(crate) const FRAME_SIZE: usize = 512;
pub type Frame = [f32; FRAME_SIZE];

pub trait AudioProvider {
    /// Returns true if new data was available, false if not.
    fn next_frame(&mut self, frame: &mut GpuFrame) -> bool;
}

pub(crate) struct AudioSource {
    pub coordinates: Vec3,
    pub provider: Box<dyn AudioProvider>
}

impl AudioSource {
    pub(crate) fn new(provider: Box<dyn AudioProvider>, coordinates: Vec3) -> Self {
        Self {
            coordinates,
            provider
        }
    }

    pub(crate) fn update_location(&mut self, coordinates: Vec3) {
        self.coordinates = coordinates;
    }
}
