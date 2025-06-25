use crevice::internal::bytemuck::Zeroable;
use crevice::std430::Mat3;
use crate::scene::listener::AudioListener;
use crate::scene::mesh::{SceneMesh, Triangle};
use crate::scene::source::AudioSource;
use crate::util::{complex, vec3};
use crevice::std430::Vec3;
use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::scene::listener::hrtf_filter::{HrtfFilter, HrtfOptions};

pub mod source;
pub mod listener;
pub mod mesh;

pub struct Scene {
    pub(crate) mesh: SceneMesh,
    pub(crate) sources: Vec<AudioSource>,
    pub(crate) listener: AudioListener,
}

impl Scene {
    // todo: simplify this a bunch to take in a filter and SceneMesh should be gone and the scene itself should hold the triangles and the BVH probably
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
