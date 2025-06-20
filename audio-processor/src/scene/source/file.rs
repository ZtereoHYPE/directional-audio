use crate::audio_engine::gpu_structures::GpuFrame;
use crate::scene::source::{AudioProvider, Frame, FRAME_SIZE};

pub struct FileAudioProvider {
    frames: Vec<Frame>,
    loop_audio: bool,
    current_idx: usize
}

impl FileAudioProvider {
    pub fn new(name: &str, loop_audio: bool) -> Self {
        let mut reader = hound::WavReader::open(name)
            .expect("Failed to open wav file!");

        println!("bps {}, format {:?}", reader.spec().bits_per_sample, reader.spec().sample_format);
        let channels = reader.spec().channels as usize;

        let frames = reader
            .samples::<i16>()
            .map(|s| s.unwrap())
            .step_by(channels)
            .collect::<Vec<_>>()
            .chunks_exact(FRAME_SIZE)
            .map(|chunk|
                chunk
                    .iter()
                    .map(|&s| f32::from(s))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            )
            .collect::<Vec<Frame>>();

        Self {
            frames,
            loop_audio,
            current_idx: 0
        }
    }
}

impl AudioProvider for FileAudioProvider {
    fn next_frame(&mut self, frame: &mut GpuFrame) -> bool {
        if self.current_idx < self.frames.len() {
            let file_frame = self.frames[self.current_idx];
            self.current_idx += 1;

            for idx in 0..frame.len() {
                frame[idx].x = file_frame[idx];
                frame[idx].y = 0.0;
            }
            
            return true;
        } else if self.loop_audio {
            self.current_idx = 0;
            return self.next_frame(frame);
        }
        
        false
    }
}
