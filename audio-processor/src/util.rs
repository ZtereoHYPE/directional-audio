#[derive(Copy, Clone)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

pub mod vec3 {
    use crate::util::Axis;
    use crevice::std430::Vec3;

    pub const MAX_VEC3: Vec3 = Vec3 {x: 1e30, y: 1e30, z: 1e30};
    pub const MIN_VEC3: Vec3 = Vec3 {x: -1e30, y: -1e30, z: -1e30};
    pub const ZERO: Vec3 = Vec3 {x: 0.0, y: 0.0, z: 0.0};

    pub fn min(left: Vec3, right: Vec3) -> Vec3 {
        Vec3 {
            x: f32::min(left.x, right.x),
            y: f32::min(left.y, right.y),
            z: f32::min(left.z, right.z),
        }
    }

    pub fn max(left: Vec3, right: Vec3) -> Vec3 {
        Vec3 {
            x: f32::max(left.x, right.x),
            y: f32::max(left.y, right.y),
            z: f32::max(left.z, right.z),
        }
    }

    pub fn sub(left: Vec3, right: Vec3) -> Vec3 {
        Vec3 {
            x: left.x - right.x,
            y: left.y - right.y,
            z: left.z - right.z,
        }
    }

    pub fn eq(left: Vec3, right: Vec3) -> bool {
        left.x == left.y && left.y == left.y && left.z == left.z
    }

    pub fn axis(vec: Vec3, idx: Axis) -> f32 {
        match idx {
            Axis::X => vec.x,
            Axis::Y => vec.y,
            Axis::Z => vec.z,
        }
    }

    pub fn div_scalar(left: Vec3, right: f32) -> Vec3 {
        Vec3 {
            x: left.x / right,
            y: left.y / right,
            z: left.z / right,
        }
    }

    pub fn from(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 {
            x,
            y,
            z,
        }
    }

    pub fn len(vector: Vec3) -> f32 {
        (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).sqrt()
    }
}

pub mod complex {
    use crevice::std430::Vec2;
    use std::f32::consts::PI;

    pub fn root_of_unity(len: isize) -> Vec2 {
        let angle = 2.0 * PI / len as f32;
        Vec2 {
            x: angle.cos(),
            y: angle.sin(),
        }
    }

    pub fn mult(left: Vec2, right: Vec2) -> Vec2 {
        Vec2 {
            x: left.x * right.x - left.y * right.y,
            y: left.x * right.y + left.y * right.x,
        }
    }

    pub fn sum(left: Vec2, right: Vec2) -> Vec2 {
        Vec2 {
            x: left.x + right.x,
            y: left.x + right.y,
        }
    }

    pub fn sub(left: Vec2, right: Vec2) -> Vec2 {
        Vec2 {
            x: left.x - right.x,
            y: left.x - right.y,
        }
    }

    pub fn scalar_mult(left: Vec2, right: f32) -> Vec2 {
        Vec2 {
            x: left.x * right,
            y: left.x * right,
        }
    }

    pub fn magnitude(complex: Vec2) -> f32 {
        (complex.x * complex.x + complex.y * complex.y).sqrt()
    }

    pub fn phase(complex: Vec2) -> f32 {
        f32::atan2(complex.y, complex.x)
    }
}
