use crate::scene::mesh::bvh::{BvhBufferData, BvhBuilder};
use crate::util::{vec3, Axis};
use crate::vulkan::buffer::BufferData;
use glam::{Vec3, Vec3A};

pub(crate) mod bvh;

#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub vertices: [Vec3A; 3],
    pub absorption: f32,
}

impl Triangle {
    pub fn new(vertices: [Vec3A; 3], absorption: f32) -> Self {
        Self {
            vertices,
            absorption
        }
    }

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
pub(crate) struct TriangleBufferData(Vec<Triangle>);

impl BufferData for TriangleBufferData {
    unsafe fn serialize(&self, dst: *mut u8) { unsafe {
        if self.0.is_empty() {
            return;
        }

        for (idx, triangle) in self.0.iter().enumerate() {
            std::ptr::copy_nonoverlapping(
                (triangle as *const Triangle).cast(),
                dst.offset((idx * 64) as isize),
                size_of::<Triangle>()
            );
        }
    }}

    fn size(&self) -> usize {
        64 * self.0.len()
    }
}

pub(crate) struct SceneMesh {
    pub(crate) bvh: BvhBufferData,
    pub(crate) triangles: TriangleBufferData,
}

impl SceneMesh {
    pub(super) fn from_triangles(mut triangles: Vec<Triangle>) -> Self {
        let bvh = BvhBuilder::new(&mut triangles).build();

        Self {
            bvh,
            triangles: TriangleBufferData(triangles)
        }
    }
}
