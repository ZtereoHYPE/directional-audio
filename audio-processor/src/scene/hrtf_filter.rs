// For now, we only support Gauss-Legendre distribution of points on the sphere
// todo: find a better way to load raw samples from the HRTF, maybe support lebedev  

use std::f32::consts::PI;

use crate::audio_engine::signal_processor::fft::FftModule;
use crate::audio_engine::GpuData;
use crevice::std430::Vec4;
use crevice::std430::Vec2;
use sofar::reader::{Filter, OpenOptions, Sofar};

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
    sofa: Sofar
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
                let (x, y, z) = polar_to_cartesian_3d(azimuth, elevation);
                sofa.filter_nointerp(x, y, z, &mut filter);
        
                let left_transformed = transform_filter(&filter.left, pad_length);
                let right_transformed = transform_filter(&filter.right, pad_length);
        
                left_data[e_idx][a_idx] = left_transformed;
                right_data[e_idx][a_idx] = right_transformed;
            }
        }

        Self {
            options, //todo: replace with inline fields from options because elevation max/min aren't needed and .options.x is annoying
            filter_len: pad_length,
            left: HrtfFilterChannel { data: left_data },
            right: HrtfFilterChannel { data: right_data },
            sofa
        }
    }

    // pub fn for_angle(&self, azimuth: f32, elevation: f32) -> (Vec<Vec2>, Vec<Vec2>) {
    //     let mut filter = Filter::new(self.sofa.filter_len());
    //
    //     let (x, y, z) = polar_to_cartesian_3d(azimuth, elevation);
    //     self.sofa.filter(x, y, z, &mut filter);
    //
    //     let left_transformed = transform_filter(&filter.left, GPU_WINDOW_SIZE);
    //     let right_transformed = transform_filter(&filter.right, GPU_WINDOW_SIZE);
    //
    //     (left_transformed, right_transformed)
    // }
}

pub struct HrtfFilterChannel {
    data: Vec<Vec<Vec<Vec4>>> // azimuth<altitude<frequency<dampening>>>
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

    unsafe fn deserialize(_: *const u8) -> Box<Self> {
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
        .map(|val| cartesian_to_linear_polar(*val))
        .collect()
}

fn polar_to_cartesian_3d(azimuth: f32, elevation: f32) -> (f32, f32, f32) {
    (
        elevation.cos() * azimuth.cos(),
        (elevation - PI / 2.0).sin(),
        elevation.cos() * azimuth.sin(),
    )
}

// todo: perform log interpolation
fn cartesian_to_linear_polar(cartesian: Vec2) -> Vec4 {
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
