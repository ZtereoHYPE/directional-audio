use crevice::internal::bytemuck::Zeroable;
use crevice::std140::Vec3;
use crevice::std430::Mat3;

#[derive(Copy, Clone)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

pub mod vec3 {
    use std::f32::consts::PI;
    use crate::util::{vec3, Axis};
    use crevice::std430::{Vec2, Vec3};

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
        left.x == right.x && left.y == right.y && left.z == right.z
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
        let radius = len(cartesian);

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
    use crevice::std430::{Vec2, Vec4};
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

    pub fn sum(left: Vec2, right: Vec2) -> Vec2 {
        Vec2 {
            x: left.x + right.x,
            y: left.y + right.y,
        }
    }

    pub fn sub(left: Vec2, right: Vec2) -> Vec2 {
        Vec2 {
            x: left.x - right.x,
            y: left.y - right.y,
        }
    }

    pub fn scalar_mult(left: Vec2, right: f32) -> Vec2 {
        Vec2 {
            x: left.x * right,
            y: left.y * right,
        }
    }

    pub fn amplitude(complex: Vec2) -> f32 {
        (complex.x * complex.x + complex.y * complex.y).sqrt()
    }

    pub fn phase(complex: Vec2) -> f32 {
        f32::atan2(complex.y, complex.x)
    }

    // todo: perform log interpolation
    // todo: perhaps find even accurater way
    pub fn to_linear_polar(cartesian: Vec2) -> Vec4 {
        // done in f64 to avoid as much precision loss as possible
        let x = cartesian.x as f64;
        let y = cartesian.y as f64;
        let mag = f64::sqrt(x * x + y * y);

        Vec4 {
            x: mag as f32,
            y: (x / mag) as f32,
            z: (y / mag) as f32,
            w: 0.0 // because RGB is much less supported than RGBA
        }
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

#[cfg(test)]
mod vec3_tests {
    use crevice::std430::{Vec2, Vec3};
    use super::*;

    fn assert_vec3_eq(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < 1e-6, "x: {} != {}", a.x, b.x);
        assert!((a.y - b.y).abs() < 1e-6, "y: {} != {}", a.y, b.y);
        assert!((a.z - b.z).abs() < 1e-6, "z: {} != {}", a.z, b.z);
    }

    #[test]
    fn test_vec3_min_max() {
        let a = Vec3 { x: 1.0, y: 5.0, z: -2.0 };
        let b = Vec3 { x: 2.0, y: 3.0, z: 0.0 };
        let min = vec3::min(a, b);
        let max = vec3::max(a, b);
        assert_vec3_eq(min, Vec3 { x: 1.0, y: 3.0, z: -2.0 });
        assert_vec3_eq(max, Vec3 { x: 2.0, y: 5.0, z: 0.0 });
    }

    #[test]
    fn test_vec3_sub() {
        let a = Vec3 { x: 3.0, y: 2.0, z: 1.0 };
        let b = Vec3 { x: 1.0, y: 1.0, z: 1.0 };
        let result = vec3::sub(a, b);
        assert_vec3_eq(result, Vec3 { x: 2.0, y: 1.0, z: 0.0 });
    }

    #[test]
    fn test_vec3_eq() {
        let a = Vec3 { x: 1.0, y: 1.0, z: 1.0 };
        let b = Vec3 { x: 1.0, y: 1.0, z: 1.0 };
        let c = Vec3 { x: 1.0, y: 2.0, z: 1.0 };
        let d = Vec3 { x: 2.0, y: 1.0, z: 1.0 };
        let e = Vec3 { x: 1.0, y: 1.0, z: 2.0 };

        assert!(vec3::eq(a, b));
        assert!(vec3::eq(a, a));
        assert!(vec3::eq(c, c));
        assert!(!vec3::eq(a, c), "Vectors with different y should not be equal");
        assert!(!vec3::eq(a, d), "Vectors with different x should not be equal");
        assert!(!vec3::eq(a, e), "Vectors with different z should not be equal");
    }

    #[test]
    fn test_vec3_axis() {
        let v = Vec3 { x: 7.0, y: 8.0, z: 9.0 };
        assert_eq!(vec3::axis(v, Axis::X), 7.0);
        assert_eq!(vec3::axis(v, Axis::Y), 8.0);
        assert_eq!(vec3::axis(v, Axis::Z), 9.0);
    }

    #[test]
    fn test_vec3_div_scalar() {
        let v = Vec3 { x: 6.0, y: 3.0, z: 9.0 };
        let result = vec3::div_scalar(v, 3.0);
        assert_vec3_eq(result, Vec3 { x: 2.0, y: 1.0, z: 3.0 });
    }

    #[test]
    fn test_vec3_from_and_len() {
        let v = vec3::from(3.0, 4.0, 0.0);
        assert_vec3_eq(v, Vec3 { x: 3.0, y: 4.0, z: 0.0 });
        assert!((vec3::len(v) - 5.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod complex_tests {
    use std::f32::consts::PI;
    use crevice::std430::{Vec2, Vec4};
    use crate::util::complex;

    #[test]
    fn test_sum() {
        let a = Vec2 { x: 1.0, y: 2.0 };
        let b = Vec2 { x: 3.0, y: 4.0 };
        let result = complex::sum(a, b);
        assert!((result.x - 4.0).abs() < 1e-6);
        assert!((result.y - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_sub() {
        let a = Vec2 { x: 54.0, y: 6.0 };
        let b = Vec2 { x: 2.0, y: -43.0 };
        let result = complex::sub(a, b);
        assert!((result.x - 52.0).abs() < 1e-6);
        assert!((result.y - 49.0).abs() < 1e-6);
    }

    #[test]
    fn test_mult() {
        let a = Vec2 { x: 1.0, y: 2.0 };
        let b = Vec2 { x: 3.0, y: 4.0 };
        let result = complex::mult(a, b);
        assert!((result.x - (-5.0)).abs() < 1e-6);
        assert!((result.y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_mult() {
        let a = Vec2 { x: 1.0, y: 2.0 };
        let scalar = 3.0;
        let result = complex::scalar_mult(a, scalar);
        assert!((result.x - 3.0).abs() < 1e-6);
        assert!((result.y - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude() {
        let a = Vec2 { x: 3.0, y: 4.0 };
        let result = complex::amplitude(a);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase() {
        let a = Vec2 { x: 1.0, y: 1.0 };
        let result = complex::phase(a);
        assert!((result - (std::f32::consts::FRAC_PI_4)).abs() < 1e-6, "Expected phase to be π/4, got {}", result);

        let b = Vec2 { x: -1.0, y: 1.0 };
        let result = complex::phase(b);
        assert!((result - (3.0 * std::f32::consts::FRAC_PI_4)).abs() < 1e-6, "Expected phase to be 3π/4, got {}", result);
    }

    #[test]
    fn test_root_of_unity() {
        let len = 8;
        let result = complex::root_of_unity(len as isize);
        let expected_x = (2.0 * PI / len as f32).cos();
        let expected_y = (2.0 * PI / len as f32).sin();
        assert!((result.x - expected_x).abs() < 1e-6, "Expected x to be {}, got {}", expected_x, result.x);
        assert!((result.y - expected_y).abs() < 1e-6, "Expected y to be {}, got {}", expected_y, result.y);

        let len = -8;
        let result = complex::root_of_unity(len as isize);
        let expected_x = (2.0 * PI / len as f32).cos();
        let expected_y = (2.0 * PI / len as f32).sin();
        assert!((result.x - expected_x).abs() < 1e-6, "Expected x to be {}, got {}", expected_x, result.x);
        assert!((result.y - expected_y).abs() < 1e-6, "Expected y to be {}, got {}", expected_y, result.y);
    }

    #[test]
    fn test_linear_polar_interpolation() {
        let vec1 = complex::from(10.0 * (5.0 * PI / 6.0).cos(), 10.0 * (5.0 * PI / 6.0).sin());
        let vec2 = complex::from(5.0 * (0.0_f32).cos(), 5.0 * (0.0_f32).sin());
        let polar1 = complex::to_linear_polar(vec1); // amplitude 10, phase π/3
        let polar2 = complex::to_linear_polar(vec2); // amplitude 5, phase 0

        // 25/75 linear interpolation
        let amplitude = (10.0 * 0.25 + 5.0 * 0.75);
        let phase = 5.0 * PI / 6.0 * 0.25 + 0.0;
        let expected = complex::from(amplitude * phase.cos(), amplitude * phase.sin());
        let result = complex::from_linear_polar(
            Vec4 {
                x: (polar1.x * 0.25 + polar2.x * 0.75),
                y: (polar1.y * 0.25 + polar2.y * 0.75),
                z: (polar1.z * 0.25 + polar2.z * 0.75),
                w: 0.0,
            }
        );

        let simple_lerp = complex::sum(complex::scalar_mult(vec1, 0.25), complex::scalar_mult(vec2, 0.75));

        println!("expected {} {}, linear_polar {} {}, straight up lerp {} {}",
            amplitude, phase,
            complex::amplitude(result), complex::phase(result),
            complex::amplitude(simple_lerp), complex::phase(simple_lerp)
        );
    }
}

pub fn rotation_matrix(pitch: f32, yaw: f32) -> Mat3 {
    let mut mat = Mat3::zeroed();

    mat.x = vec3::from(yaw.cos(), 0.0, yaw.sin() * pitch.cos());
    mat.y = vec3::from(0.0, pitch.cos(), -pitch.sin());
    mat.z = vec3::from(-yaw.sin(), pitch.sin(), yaw.cos() * pitch.cos());

    mat
}
