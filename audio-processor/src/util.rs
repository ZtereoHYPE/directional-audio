use bytemuck::Zeroable;
use core::slice;
use glam::{Mat3, Vec3};

#[derive(Copy, Clone)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

pub mod vec3 {
    use crate::util::Axis;
    use glam::{Vec2, Vec3};

    pub(crate) fn axis(vec: Vec3, idx: Axis) -> f32 {
        match idx {
            Axis::X => vec.x,
            Axis::Y => vec.y,
            Axis::Z => vec.z,
        }
    }

    /// elevation is 0..PI where 0 is the top and PI is the bottom
    /// azimuth is -PI..PI
    pub fn from_unit_polar(azimuth: f32, elevation: f32) -> Vec3 {
        Vec3 {
            x: elevation.sin() * azimuth.cos(),
            y: elevation.cos(),
            z: elevation.sin() * -azimuth.sin(),
        }
    }

    pub fn to_unit_polar(cartesian: Vec3) -> Vec2 {
        let radius = cartesian.length();

        Vec2 {
            x: f32::atan2(-cartesian.z, cartesian.x), // azimuth
            y: f32::acos(cartesian.y / radius), // elevation
        }
    }
}

// newtype pattern for types
// struct GodotCoords([f32; 3]);
// struct UnitPolarCoords([f32; 2]);
// struct Complex(f32, f32);

pub mod complex {
    use glam::{Vec2, Vec4};
    use std::f32::consts::PI;

    pub fn from(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

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

    pub fn phase(complex: Vec2) -> f32 {
        f32::atan2(complex.y, complex.x)
    }

    // todo: perform log interpolation
    pub fn to_linear_polar(cartesian: Vec2) -> Vec4 {
        // done in f64 to avoid as much precision loss as possible
        let x = cartesian.x as f64;
        let y = cartesian.y as f64;
        let mag = f64::sqrt(x * x + y * y);

        Vec4::new(
            mag as f32,
            (x / mag) as f32,
            (y / mag) as f32,
            0.0
        )
    }

    pub fn from_linear_polar(polar: Vec4) -> Vec2 {
        let x = polar.y;
        let y = polar.z;
        let len = (x * x + y * y).sqrt();
        let normalized = Vec2 { x: x / len, y: y / len };

        Vec2 {
            x: normalized.x * polar.x,
            y: normalized.y * polar.x,
        }
    }
}

pub fn rotation_matrix(pitch: f32, yaw: f32) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(yaw.cos(), 0.0, yaw.sin() * pitch.cos()),
        Vec3::new(0.0, pitch.cos(), -pitch.sin()),
        Vec3::new(-yaw.sin(), pitch.sin(), yaw.cos() * pitch.cos())
    )
}

pub fn workgroup_div(instances: u32, warp_size: u32) -> u32 {
    ((instances + warp_size - 1) / warp_size)
}

pub trait AsBytes: Sized {
    unsafe fn as_bytes(&self) -> &[u8] { unsafe {
        slice::from_raw_parts(
            (self as *const Self) as *const u8,
            size_of::<Self>(),
        )
    }}
}

// Automatically implement AsBytes for all Sized structs
impl<T: Sized> AsBytes for T {}
