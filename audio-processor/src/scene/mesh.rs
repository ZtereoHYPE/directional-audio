use crate::audio_engine::DynamicBufferData;
use crate::scene::mesh::bvh::{BvhBuffer, BvhBuilder};
use crate::util::{vec3, Axis};
use glam::{Vec3, Vec3A};

pub(crate) mod bvh;

#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub vertices: [Vec3A; 3],
}

impl Triangle {

    pub(crate) fn min_bound(&self) -> Vec3 {
        Vec3A::min(Vec3A::min(self.vertices[0], self.vertices[1]), self.vertices[2]).into()
    }

    pub(crate) fn max_bound(&self) -> Vec3 {
        Vec3A::max(Vec3A::max(self.vertices[0], self.vertices[1]), self.vertices[2]).into()
    }

    pub(crate) fn axis_min(&self, axis: Axis) -> f32 {
        let mut min: f32 = vec3::axis(self.vertices[0].into(), axis);
        for vtx in self.vertices.iter().skip(1) {
            min = f32::min(min, vec3::axis(Vec3::from(*vtx), axis));
        }
        min
    }

    pub(crate) fn axis_max(&self, axis: Axis) -> f32 {
        let mut max: f32 = vec3::axis(self.vertices[0].into(), axis);
        for vtx in self.vertices.iter().skip(1) {
            max = f32::max(max, vec3::axis(Vec3::from(*vtx), axis));
        }
        max
    }
}

#[derive(Clone)]
pub(crate) struct TriangleBuffer(Vec<Triangle>);

impl DynamicBufferData for TriangleBuffer {
    unsafe fn serialize(&self, dst: *mut u8) { unsafe {
        if self.0.is_empty() {
            return;
        }

        std::ptr::copy_nonoverlapping(
            (&self.0[..] as *const [Triangle]).cast(),
            dst,
            self.size()
        );
    }}

    fn size(&self) -> usize {
        48 * self.0.len()
    }
}

pub(crate) struct SceneMesh {
    pub(crate) bvh: BvhBuffer,
    pub(crate) triangles: TriangleBuffer,
}

impl SceneMesh {
    pub(super) fn from_triangles(mut triangles: Vec<Triangle>) -> Self {
        let bvh = BvhBuilder::new(&mut triangles).build();

        Self {
            bvh,
            triangles: TriangleBuffer(triangles)
        }
    }
}
