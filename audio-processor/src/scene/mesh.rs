use crate::audio_engine::GpuData;
use crate::scene::bvh::{BvhBuffer, BvhBuilder};
use crate::util::{vec3, Axis};
use crevice::std430::Vec3;

#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub vertices: [Vec3; 3],
}

impl Triangle {
    pub(crate) fn min_bound(&self) -> Vec3 {
        vec3::min(vec3::min(self.vertices[0], self.vertices[1]), self.vertices[2])
    }

    pub(crate) fn max_bound(&self) -> Vec3 {
        vec3::max(vec3::max(self.vertices[0], self.vertices[1]), self.vertices[2])
    }

    pub(crate) fn axis_min(&self, axis: Axis) -> f32 {
        let mut min: f32 = vec3::axis(self.vertices[0], axis);
        for vtx in self.vertices.iter().skip(1) {
            min = f32::min(min, vec3::axis(*vtx, axis));
        }
        min
    }

    pub(crate) fn axis_max(&self, axis: Axis) -> f32 {
        let mut max: f32 = vec3::axis(self.vertices[0], axis);
        for vtx in self.vertices.iter().skip(1) {
            max = f32::max(max, vec3::axis(*vtx, axis));
        }
        max
    }
}

#[derive(Clone)]
pub(crate) struct TriangleBuffer(Vec<Triangle>);

impl GpuData for TriangleBuffer {
    unsafe fn serialize(&self, dst: *mut u8) {
        if self.0.is_empty() {
            return;
        }

        std::ptr::copy_nonoverlapping(
            (&self.0[..] as *const [Triangle]).cast(),
            dst,
            self.size()
        );
    }

    fn size(&self) -> usize {
        size_of::<Triangle>() * self.0.len() // should not be 0 to avoid crashes
    }
}

pub(crate) struct SceneMesh {
    pub(crate) bvh: BvhBuffer,
    pub(crate) triangles: TriangleBuffer,
}

impl SceneMesh {
    pub(super) fn from_triangles(mut triangles: Vec<Triangle>) -> Self {
        let bvh_nodes = BvhBuilder::new(&mut triangles).build();

        Self {
            bvh: BvhBuffer(bvh_nodes),
            triangles: TriangleBuffer(triangles)
        }
    }
}
