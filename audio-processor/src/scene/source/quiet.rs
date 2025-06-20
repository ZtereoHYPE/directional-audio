use crate::audio_engine::gpu_structures::GpuFrame;
use crate::scene::source::AudioProvider;

pub struct SilentAudioProfider;

impl AudioProvider for SilentAudioProfider {
    fn next_frame(&mut self, frame: &mut GpuFrame) -> bool {
        for idx in 0..frame.len() {
            frame[idx].x = 0.0;
            frame[idx].y = 0.0;
        }

        true
    }
}
