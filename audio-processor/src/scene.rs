use crate::scene::listener::AudioListener;
use crate::scene::mesh::{SceneMesh, Triangle};
use crate::scene::source::AudioSource;
use crate::util::vec3;
use crevice::std430::Vec3;
use crate::scene::listener::hrtf_filter::HrtfOptions;

pub mod source;
pub mod listener;
pub mod mesh;
pub(crate) mod bvh;

pub struct Scene {
    pub(crate) mesh: SceneMesh,
    pub(crate) sources: Vec<AudioSource>,
    pub(crate) listener: AudioListener,
}

impl Scene {
    // todo: simplify this a bunch to take in a filter and SceneMesh should be gone and the scene itself should hold the triangles and the BVH probably
    pub fn new(sources: Vec<AudioSource>, triangles: Vec<Triangle>, hrtf_filter: &str, hrtf_options: HrtfOptions) -> Self {
        Self {
            mesh: SceneMesh::from_triangles(triangles),
            sources,
            listener: AudioListener::new(vec3::from(0.0, 0.0, 0.0), hrtf_filter, hrtf_options)
        }
    }

    pub fn set_listener_location(&mut self, location: Vec3) {
        self.listener.location = location;
    }
}
