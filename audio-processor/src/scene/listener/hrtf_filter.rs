use crate::audio_engine::signal_processor::fft::FftModule;
use crate::audio_engine::GpuData;
use crevice::std430::{Vec2, Vec3};
use crevice::std430::Vec4;
use sofar::reader::{Filter, OpenOptions};
use std::f32::consts::PI;
use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::util::{complex, vec3};

#[derive(Clone)]
pub struct HrtfOptions {
    pub elevation_samples: u32,
    pub azimuth_samples: u32,
    pub elevation_top: f32, // in radians
    pub elevation_bottom: f32, // in radians
    pub sampling_rate: f32,
}

#[derive(Clone)]
pub struct HrtfFilter {
    pub options: HrtfOptions,
    pub filter_len: usize,
    pub left: HrtfFilterChannel, 
    pub right: HrtfFilterChannel,
}

impl HrtfFilter {
    pub fn new(options: HrtfOptions, file: &str) -> Self {
        let pad_length = GPU_WINDOW_SIZE;
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
                .map(|e| (e as f32) * sample_distance) // -PI..PI
                .map(|e| if e > PI { e - 2.0 * PI } else { e })
                .collect::<Vec<_>>()
        };

        let elevations: Vec<f32> = {
            let elevation_range = (options.elevation_top - options.elevation_bottom).abs();
            let sample_distance = elevation_range / ((options.elevation_samples - 1) as f32);
            
            (0..options.elevation_samples)
                .map(|e| (e as f32) * sample_distance + options.elevation_top) // 0..PI
                .collect::<Vec<_>>()
        };
        
        // allocate the space except for the inntermost vector which will be set later
        let mut left_data = vec![vec![vec![]; elevations.len()]; azimuths.len()];
        let mut right_data = vec![vec![vec![]; elevations.len()]; azimuths.len()];

        let mut filter = Filter::new(filter_len);
        for (a_idx, &azimuth) in azimuths.iter().enumerate() {
            for (e_idx, &elevation) in elevations.iter().enumerate() {
                let unit_coordinates = vec3::from_unit_polar(azimuth, elevation);
                let Vec3 {x, y, z} = godot_to_hrtf_space(unit_coordinates);
                
                sofa.filter_nointerp(x, y, z, &mut filter);

                let left_transformed = transform_filter(&filter.left, pad_length);
                let right_transformed = transform_filter(&filter.right, pad_length);
                
                left_data[a_idx][e_idx] = left_transformed;
                right_data[a_idx][e_idx] = right_transformed;
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

#[derive(Clone)]
pub struct HrtfFilterChannel {
    data: Vec<Vec<Vec<Vec4>>> // azimuth<altitude<frequency<dampening>>>
}

impl GpuData for HrtfFilterChannel {
    unsafe fn serialize(&self, mut dst: *mut u8) {
        for azimuth in &self.data {
            for elevation in azimuth {
                let len = elevation.len() * size_of::<Vec4>();
                let src = elevation.as_ptr().cast();

                unsafe {
                    std::ptr::copy(src, dst, len);
                    dst = dst.offset(len as isize);
                }
            }
        }
    }

    fn size(&self) -> usize {
        // todo: this assumes they all have the same length... find a way to enforce that?
        let azimuths = self.data.len();
        let elevations = self.data[0].len(); 
        let filter_len = self.data[0][0].len();

        azimuths * elevations * filter_len * size_of::<Vec4>()
    }
}

fn transform_filter(filter: &Box<[f32]>, pad_length: usize) -> Vec<Vec4> {
    assert!(pad_length > filter.len(), "Padded length must be equal or larger than the filter's length!");

    let mut vec: Vec<Vec2> = filter.iter().map(|tap| Vec2 {x: *tap, y: 0.0}).collect();

    while vec.len() < pad_length {
        vec.push(Vec2 {x: 0.0, y: 0.0});
    }

    let fourier = FftModule::local_fourier_transform(vec, false);

    // The filter must also be converted to polar coordinates in order to enable interpolation
    fourier
        .iter()
        .map(|val| complex::to_linear_polar(*val))
        .collect()
}

fn godot_to_hrtf_space(vec: Vec3) -> Vec3 {
    Vec3 {
        x: -vec.z,
        y: -vec.x,
        z: vec.y, // todo: maybe negative?
    }
}
