// this module will be responsible for the audio sampling and splitting and stuff
// as well as its interface

use std::f32::consts::{PI, TAU};

use rand::Rng;
pub mod hrtf;


pub(crate) const FRAME_SIZE: usize = 512;
pub(crate) const FRAME_AMT: usize = 1;

pub type Frame = [f32; FRAME_SIZE];
pub struct AudioProvider {}
impl AudioProvider {
    // todo: remember, the frame needs to be A SLICE
    pub fn next_frame() -> Frame {
        let mut rng = rand::rng();

        let amt = 4;
        
        let mut buffer = [0.0; FRAME_SIZE];
        let mut phases: Vec<f32> = vec![0.0; 4];
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
}