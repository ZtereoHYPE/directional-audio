use crate::scene::listener::AudioListener;
use crate::scene::mesh::{SceneMesh, Triangle};
use crate::scene::source::AudioSource;
use crate::util::vec3;
use crevice::std430::Vec3;
use std::sync::Mutex;

pub(crate) mod source;
pub(crate) mod listener;
pub(crate) mod mesh;
pub(crate) mod bvh;

pub(crate) struct Scene {
    pub mesh: SceneMesh,
    pub sources: Mutex<Vec<AudioSource>>,
    pub listener: AudioListener,
}

impl Scene {
    pub(crate) fn new(sources: Vec<AudioSource>) -> Self {
        let triangle = Triangle {
            vertices: [
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                }
            ]
        };

        Self {
            mesh: SceneMesh::from_triangles(vec![triangle]),
            sources: Mutex::from(sources),
            listener: AudioListener::new(vec3::from(0.0, 0.0, 0.0))
        }
    }

    pub(crate) fn get_listener_location(&self) -> Vec3 {
        *self.listener.location.lock().unwrap()
    }

    pub(crate) fn set_listener_location(&mut self, location: Vec3) {
        *self.listener.location.lock().unwrap() = location;
    }
}
