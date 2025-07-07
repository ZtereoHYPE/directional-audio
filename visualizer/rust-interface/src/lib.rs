use godot::prelude::*;
use audio_processor::util::vec3;
use crevice::std430::Vec3;

mod audio_listener;
mod audio_source;
mod audio_mesh;
mod visualization;

struct AudioVisualization;

#[gdextension]
unsafe impl ExtensionLibrary for AudioVisualization {}

pub fn to_vec(vec: Vector3) -> Vec3 {
    vec3::from(vec.x, vec.y, vec.z)
}