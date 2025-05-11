// this module will be responsible for the audio sampling and splitting and stuff
// as well as its interface

pub type Frame = [u32; 65536];
pub struct AudioProvider {}

impl AudioProvider {
    // todo: remember, the frame needs to be A SLICE
    pub fn next_frame() -> Frame {
        [1; 65536]
    }
}