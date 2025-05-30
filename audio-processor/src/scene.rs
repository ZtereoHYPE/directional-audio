// this module will be responsible for the scene sampling and splitting and stuff
// as well as its interface

use std::f32::consts::PI;

use rand::Rng;

pub mod hrtf_filter;
mod source;
mod listener;

// todo: move to source

pub(crate) const FRAME_SIZE: usize = 512;
pub(crate) const FRAME_AMT: usize = 1;

pub type Frame = [f32; FRAME_SIZE];
pub struct AudioProvider {}
impl AudioProvider {
    pub fn random_frame(amt: usize) -> Frame {
        let mut rng = rand::rng();

        let mut buffer = [0.0; FRAME_SIZE];
        let mut phases: Vec<f32> = vec![0.0; amt];
        let amplitudes: Vec<f32> = (0..amt).map(|_| rng.random_range(0.3..2.0)).collect();
        let frequencies: Vec<f32> = (0..amt).map(|_| rng.random_range(0.05..1.0)).collect();

        for value in buffer.iter_mut() {
            for idx in 0..4 {
                *value += phases[idx].sin() * amplitudes[idx];
                phases[idx] += PI * frequencies[idx];
            }
        }
        
        buffer
    }

    pub fn from_file(name: &str) -> Vec<Frame> {
        let mut reader = hound::WavReader::open(name)
            .expect("Failed to open wav file!");

        println!("bps {}, format {:?}", reader.spec().bits_per_sample, reader.spec().sample_format);
        let channels = reader.spec().channels as usize;
        
        reader
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
            .collect::<Vec<Frame>>()
    }
}