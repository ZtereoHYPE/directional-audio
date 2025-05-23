// For now, we only support Gauss-Legendre distribution of points on the sphere
// todo: find a better way to load raw samples from the HRTF, maybe support lebedev  

use std::f32::consts::PI;

use crevice::{std430::Vec2};
use sofar::reader::{Filter, OpenOptions};

use crate::vulkan::{engine::GpuData, fft::{cpu_fft, root_of_unity}};

pub struct HrtfOptions {
    pub elevation_samples: u32,
    pub azimuth_samples: u32,
    pub elevation_max: f32, // in radians
    pub elevation_min: f32, // in radians
    pub sampling_rate: f32,
}

pub struct HrtfFilter {
    pub options: HrtfOptions,
    pub filter_len: usize,
    pub left: HrtfFilterChannel, 
    pub right: HrtfFilterChannel,
}

impl HrtfFilter {
    pub fn new(options: HrtfOptions, file: &str, pad_length: usize) -> Self {
        let sofa = OpenOptions::new()
            .sample_rate(options.sampling_rate)
            .open(file)
            .unwrap();

        let filter_len = sofa.filter_len();
        println!("Loading HRTF Filter of length {}", filter_len);

        // Sample with Gauss-Lagemdre distribution
        let azimuths: Vec<f32> = {
            let sample_distance = 2.0 * PI / options.azimuth_samples as f32;
            
            (0..options.azimuth_samples)
                .map(|e| (e as f32) * sample_distance)
                .collect::<Vec<_>>()
        };

        let elevations: Vec<f32> = {
            let elevation_range = (options.elevation_max - options.elevation_min) as f32;
            let sample_distance = (options.elevation_samples as f32) / elevation_range;
            
            (0..options.elevation_samples)
                .map(|e| (e as f32) * sample_distance + options.elevation_min)
                .collect::<Vec<_>>()
        };
        
        // allocate the space except for the inntermost vector which will be set later
        let mut left_data = vec![vec![vec![]; azimuths.len()]; elevations.len()];
        let mut right_data = vec![vec![vec![]; azimuths.len()]; elevations.len()];

        let mut filter = Filter::new(filter_len);
        for (e_idx, &elevation) in elevations.iter().enumerate() {
            for (a_idx, &azimuth) in azimuths.iter().enumerate() {
                let (x, y, z) = polar_to_cartesian(azimuth, elevation);
                sofa.filter_nointerp(x, y, z, &mut filter);

                let left_transformed = fourier_transform(&filter.left, pad_length);
                let right_transformed = fourier_transform(&filter.right, pad_length);

                left_data[e_idx][a_idx] = left_transformed;
                right_data[e_idx][a_idx] = right_transformed;
            }
        }

        Self {
            options, //todo: replace with inline fields from options because elevation max/min aren't needed and .options.x is annoying
            filter_len: pad_length,
            left: HrtfFilterChannel { data: left_data },
            right: HrtfFilterChannel { data: right_data },
        }
    }
}

pub struct HrtfFilterChannel {
    data: Vec<Vec<Vec<Vec2>>> // azimuth<altitude<frequency<dampening>>>
}

impl GpuData for HrtfFilterChannel {
    unsafe fn serialize(&self, mut dst: *mut u8) {
        for azimuth in &self.data {
            for elevation in azimuth {
                let len = elevation.len() * size_of::<f32>();
                let src = elevation.as_ptr().cast();

                unsafe {
                    std::ptr::copy(src, dst, len);
                    dst = dst.offset(len as isize);
                }
            }
        }
    }

    unsafe fn deserialize(src: *const u8) -> Box<Self> {
        panic!("HRTF filters should not be deserialized from the gpu!")
    }

    fn size(&self) -> usize {
        // todo: this assumes they all have the same length... find a way to enforce that?
        let azimuths = self.data.len();
        let elevations = self.data[0].len(); 
        let filter_len = self.data[0][0].len();

        azimuths * elevations * filter_len * size_of::<Vec2>()
    }
}

fn fourier_transform(filter: &Box<[f32]>, pad_length: usize) -> Vec<Vec2> {
    assert!(pad_length > filter.len(), "Padded length must be equal or larger than the filter's length!");

    let mut vec: Vec<Vec2> = filter.iter().map(|tap| Vec2 {x: *tap, y: 0.0}).collect();

    while vec.len() < pad_length {
        vec.push(Vec2 {x: 0.0, y: 0.0});
    }

    cpu_fft(vec, root_of_unity(pad_length as isize))
}

fn polar_to_cartesian(azimuth: f32, elevation: f32) -> (f32, f32, f32) {
    (
        elevation.cos() * azimuth.cos(),
        elevation.sin(),
        elevation.cos() * azimuth.sin(),
    )
}
