use glam::Vec3;
use godot::prelude::*;

mod audio_listener;
mod audio_source;
mod audio_mesh;
mod visualization;

struct AudioVisualization;

#[gdextension]
unsafe impl ExtensionLibrary for AudioVisualization {}

pub fn to_vec(vec: Vector3) -> Vec3 {
    Vec3::new(vec.x, vec.y, vec.z)
}