use crate::scene::listener::hrtf_filter::HrtfFilter;
use crate::scene::listener::AudioListener;
use crate::scene::mesh::{SceneMesh, Triangle};
use crate::scene::source::AudioSource;
use bytemuck::Zeroable;
use glam::{Mat3, Vec3};

pub mod source;
pub mod listener;
pub mod mesh;

pub struct Scene {
    pub(crate) mesh: SceneMesh,
    pub(crate) sources: Vec<AudioSource>,
    pub(crate) listener: AudioListener,
}

impl Scene {
    pub fn new(sources: Vec<AudioSource>, triangles: Vec<Triangle>, hrtf_filter: HrtfFilter) -> Self {
        Self {
            mesh: SceneMesh::from_triangles(triangles),
            sources,
            listener: AudioListener::new(Vec3::zeroed(), Mat3::zeroed(), hrtf_filter)
        }
    }

    pub fn set_listener_location(&mut self, location: Vec3) {
        self.listener.location = location;
    }
}
