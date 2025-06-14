use crate::scene::listener::AudioListener;
use crate::scene::mesh::{SceneMesh, Triangle};
use crate::scene::source::AudioSource;
use crate::util::vec3;
use crevice::std430::Vec3;

pub(crate) mod source;
pub(crate) mod listener;
pub(crate) mod mesh;
pub(crate) mod bvh;

pub(crate) struct Scene {
    pub(crate) mesh: SceneMesh,
    pub(crate) sources: Vec<AudioSource>,
    pub(crate) listener: AudioListener,
}

impl Scene {
    fn new(sources: Vec<AudioSource>, triangles: Vec<Triangle>) -> Self {
        Self {
            mesh: SceneMesh::from_triangles(triangles),
            sources,
            listener: AudioListener::new(vec3::from(0.0, 0.0, 0.0))
        }
    }

    pub fn set_listener_location(&mut self, location: Vec3) {
        self.listener.location = location;
    }
}
