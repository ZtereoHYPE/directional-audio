use crate::audio_engine::gpu_structures::GpuFrame;
use crate::scene::source::AudioProvider;
use std::f32::consts::PI;

pub struct FrequencyAudioProvider {
    frequency: f32,
    phase: f32,
}

impl FrequencyAudioProvider {
    pub(crate) fn new(frequency: f32) -> Self {
        Self {
            frequency,
            phase: 0.0
        }
    }
}

impl AudioProvider for FrequencyAudioProvider {
    fn next_frame(&mut self, frame: &mut GpuFrame) -> bool {
        for idx in 0..frame.len() {
            frame[idx].x = self.phase.sin() * 100.0;
            frame[idx].y = 0.0;
            self.phase += 2.0 * PI * (self.frequency / 44100.0);
        }

        true
    }
}
