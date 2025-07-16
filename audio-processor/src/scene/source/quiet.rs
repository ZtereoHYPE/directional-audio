use crate::audio_engine::GpuFrame;
use crate::scene::source::AudioProvider;

pub struct SilentAudioProvider;

impl AudioProvider for SilentAudioProvider {
    fn next_frame(&mut self, frame: &mut GpuFrame) -> bool {
        for idx in 0..frame.len() {
            frame[idx].x = 0.0;
            frame[idx].y = 0.0;
        }

        true
    }
}
