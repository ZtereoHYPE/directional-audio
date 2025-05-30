mod audio_engine;
mod scene;

use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::prelude::{BitMapBackend, IntoDrawingArea};
use plotters::series::LineSeries;
use plotters::style::full_palette::RED;
use plotters::style::{BLUE, WHITE};
use scene::FRAME_SIZE;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_engine::gpu_structures::{GpuWindow, StreamBuffer, GPU_WINDOW_SIZE};
    use crate::audio_engine::signal_processor::fft::{complex, FftModule};
    use crate::audio_engine::{frame_to_gpu, gpu_to_frame, AudioEngine};
    use crate::scene::hrtf_filter::{HrtfFilter, HrtfOptions};
    use crate::scene::{AudioProvider, Frame};
    use crevice::std430::Vec2;
    use std::f32::consts::PI;
    use std::mem::transmute;
    // pub unsafe fn alloc_empty_buffer() -> Box<FftBuffer> {
    //     let layout = std::alloc::Layout::new::<FftBuffer>();
    //     let ptr = std::alloc::alloc_zeroed(layout) as *mut FftBuffer;
    //
    //     Box::from_raw(ptr)
    // }

    pub fn plot_data(frame: &Vec<f32>, fft: &Vec<f32>, ifft: &Vec<f32>, name: &str) {
        let root = BitMapBackend::new(name, (1280, 720)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                "soundwave",
                ("sans-serif", 40),
            )
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Right, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(
                0..FRAME_SIZE,
                -10.0..10.0,
            ).unwrap();

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .x_labels(30)
            .max_light_lines(4)
            .y_desc("amplitude")
            .draw()
            .unwrap();

        chart.draw_series(LineSeries::new(
            frame.iter().map(|&f| f as f64).enumerate(),
            &BLUE,
        )).unwrap();

        chart.draw_series(LineSeries::new(
            fft.iter().map(|&f| f as f64).enumerate(),
            &RED,
        )).unwrap();

        // chart.draw_series(LineSeries::new(
        //     ifft.iter().map(|&f| f as f64).enumerate(),
        //     &GREEN,
        // )).unwrap();

        root.present().expect("Unable to write result to file, pr");
    }

    // #[ignore]
    #[test]
    fn audio_test() {
        let sampels = AudioProvider::from_file("./datasources/sample-15s.wav");

        let mut left: Vec<Frame> = vec![];
        let mut right: Vec<Frame> = vec![];

        unsafe {
            let mut engine = AudioEngine::new();

            println!("started rendering");
            for sample in sampels {
                let (left_frame, right_frame) = engine.process_frames(vec![sample]);
                
                

                left.push(left_frame);
                right.push(right_frame);
            }

            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::create("output.wav", spec).unwrap();

            for (&l, r) in left.iter().zip(right) {
                for (&sl, sr) in l.iter().zip(r) {
                    writer.write_sample(sl / 10000.0);
                    writer.write_sample(sr / 10000.0);
                }
            }

            writer.finalize().unwrap();
        }
    }

    #[ignore]
    #[test]
    fn audio_test_local() {
        let sampels = AudioProvider::from_file("./datasources/sample-15s.wav");
    
        let filter_options = HrtfOptions {
            azimuth_samples: 180, // one every 2 deg
            elevation_samples: 90, // one every 2 deg
            elevation_max: PI, // full sphere was captured
            elevation_min: 0.0, // "
            sampling_rate: 44100.0
        };
    
        let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa", GPU_WINDOW_SIZE);
    
        let mut left: Vec<Frame> = vec![];
        let mut right: Vec<Frame> = vec![];
    
        let mut stream_buffer = StreamBuffer::new();
    
        fn window_to_vec(window: Box<GpuWindow>) -> Vec<Vec2> {
            unsafe {
                // transmute the pointer without performing a copy; arrays in rust are guaranteed to be sequential
                let flat_window = transmute::<Box<GpuWindow>, Box<[Vec2; GPU_WINDOW_SIZE]>>(window);
                (flat_window as Box<[_]>).into_vec() // turn the box into a vector
            }
        }
    
        for sample in sampels {
            stream_buffer.insert_frames(vec![frame_to_gpu(&sample).into()]);
    
            let fft_frame = FftModule::local_fourier_transform(window_to_vec(Box::from(stream_buffer.windows[0])), false);
    
            let (filter_left, filter_right) = filter.for_angle(0.0, 0.5);
    
            let multiplied_left: Vec<_> = fft_frame
                .iter()
                .enumerate()
                .map(|(idx, &s)| complex::mult(s, filter_left[idx]))
                .collect();
            
            let multiplied_right: Vec<_> = fft_frame
                .iter()
                .enumerate()
                .map(|(idx, &s)| complex::mult(s, filter_right[idx]))
                .collect();
            
            let ifft_left = FftModule::local_fourier_transform(multiplied_left, true);
            let ifft_right = FftModule::local_fourier_transform(multiplied_right, true);
            
            let (start, end) = StreamBuffer::last_frame_range();
    
            left.push(gpu_to_frame(&ifft_left[start..end].try_into().unwrap()));
            right.push(gpu_to_frame(&ifft_right[start..end].try_into().unwrap()));
        }
    
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
    
        let mut writer = hound::WavWriter::create("output_local.wav", spec).unwrap();
    
        for (&l, r) in left.iter().zip(right) {
            for (&sl, sr) in l.iter().zip(r) {
                writer.write_sample(sl / 100000.0).unwrap();
                writer.write_sample(sr / 100000.0).unwrap();
            }
        }
    
        writer.finalize().unwrap();
    }
}
