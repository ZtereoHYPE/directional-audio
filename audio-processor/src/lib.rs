#![allow(unused)]

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
    use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
    use crate::audio_engine::signal_processor::fft::complex;
    use crate::audio_engine::AudioEngine;
    use crate::scene::hrtf_filter::{HrtfFilter, HrtfOptions};
    use crate::scene::{AudioProvider, Frame};
    use std::f32::consts::PI;
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

    pub fn plot_histogram(data: &Vec<f32>, data2: &Vec<f32>, name: &str) {
        let root = BitMapBackend::new(name, (1280, 720)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                "frequency",
                ("sans-serif", 40),
            )
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(
                0..GPU_WINDOW_SIZE,
                0..2,
            ).unwrap();

        chart
            .configure_mesh()
            .draw()
            .unwrap();

        // chart.draw_series(Histogram::vertical(&chart)
        //     .style(ShapeStyle::from(BLUE).filled())
        //     .data(data.iter().map(|x| (*x * 10000.0) as i32).enumerate())
        // ).unwrap();

        chart.draw_series(LineSeries::new(
            data.iter().map(|&f| (f) as i32).enumerate(),
            &BLUE,
        )).unwrap();

        chart.draw_series(LineSeries::new(
            data2.iter().map(|&f| (f) as i32).enumerate(),
            &RED,
        )).unwrap();

        root.present().expect("Unable to write result to file, pr");
    }

    #[ignore]
    #[test]
    fn audio_test() {
        let samples = AudioProvider::from_file("./datasources/sample-15s.wav");

        let mut left: Vec<Frame> = vec![];
        let mut right: Vec<Frame> = vec![];

        unsafe {
            let mut engine = AudioEngine::new();

            println!("started rendering");
            for sample in samples {
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
                    writer.write_sample(sl / 100000.0).unwrap();
                    writer.write_sample(sr / 100000.0).unwrap();
                }
            }

            writer.finalize().unwrap();
        }
    }

    // #[ignore]
    #[test]
    fn freq_test() {
        let frame = AudioProvider::from_file("./datasources/sample-15s.wav")[15];
        // let frame = AudioProvider::frequency(5000.0);

        let filter_options = HrtfOptions {
            azimuth_samples: 0, // one every 2 deg
            elevation_samples: 1, // one every 2 deg
            elevation_max: PI, // full sphere was captured
            elevation_min: 0.0, // "
            sampling_rate: 44100.0
        };

        let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa", GPU_WINDOW_SIZE);

        unsafe {
            let mut engine = AudioEngine::new();

            println!("started rendering");
            engine.process_frames_frequency(vec![frame]);
            let (left_window, _) = engine.process_frames_frequency(vec![frame]);
            println!("finished rendering");

            let left_amplitude: Vec<_> = left_window.iter().map(|s| complex::magnitude(*s)).collect();
            // let right_amplitude = left_window.iter().map(|s| complex::magnitude(*s)).collect();

            let left_amplitude_2: Vec<_> = filter.for_angle(2.0 * PI * 0.39857, PI * 0.928747).0.iter().map(|s| complex::magnitude(*s)).collect();

            let mse = left_amplitude.iter().zip(left_amplitude_2.clone()).fold(0.0, |acc, (&l, r)| acc + (l - r) * (l - r));
            println!("Mean square error: {}", mse / GPU_WINDOW_SIZE as f32);

            plot_histogram(&left_amplitude, &left_amplitude_2,"./frequency.png");
        }
    }

    // #[ignore]
    // #[test]
    // fn audio_test_local() {
    //     let sampels = AudioProvider::from_file("./datasources/sample-15s.wav");
    // 
    //     let filter_options = HrtfOptions {
    //         azimuth_samples: 180, // one every 2 deg
    //         elevation_samples: 90, // one every 2 deg
    //         elevation_max: PI, // full sphere was captured
    //         elevation_min: 0.0, // "
    //         sampling_rate: 44100.0
    //     };
    // 
    //     let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa", GPU_WINDOW_SIZE);
    // 
    //     let mut left: Vec<Frame> = vec![];
    //     let mut right: Vec<Frame> = vec![];
    // 
    //     let mut stream_buffer = StreamBuffer::new();
    // 
    //     fn window_to_vec(window: Box<GpuWindow>) -> Vec<Vec2> {
    //         unsafe {
    //             // transmute the pointer without performing a copy; arrays in rust are guaranteed to be sequential
    //             let flat_window = transmute::<Box<GpuWindow>, Box<[Vec2; GPU_WINDOW_SIZE]>>(window);
    //             (flat_window as Box<[_]>).into_vec() // turn the box into a vector
    //         }
    //     }
    // 
    //     for sample in sampels {
    //         stream_buffer.insert_frames(vec![frame_to_gpu(&sample).into()]);
    // 
    //         let fft_frame = FftModule::local_fourier_transform(window_to_vec(Box::from(stream_buffer.windows[0])), false);
    // 
    //         let (filter_left, filter_right) = filter.for_angle(0.0, 0.5);
    // 
    //         let multiplied_left: Vec<_> = fft_frame
    //             .iter()
    //             .enumerate()
    //             .map(|(idx, &s)| complex::mult(s, filter_left[idx]))
    //             .collect();
    //         
    //         let multiplied_right: Vec<_> = fft_frame
    //             .iter()
    //             .enumerate()
    //             .map(|(idx, &s)| complex::mult(s, filter_right[idx]))
    //             .collect();
    //         
    //         let ifft_left = FftModule::local_fourier_transform(multiplied_left, true);
    //         let ifft_right = FftModule::local_fourier_transform(multiplied_right, true);
    //         
    //         let (start, end) = StreamBuffer::last_frame_range();
    // 
    //         left.push(gpu_to_frame(&ifft_left[start..end].try_into().unwrap()));
    //         right.push(gpu_to_frame(&ifft_right[start..end].try_into().unwrap()));
    //     }
    // 
    //     let spec = hound::WavSpec {
    //         channels: 2,
    //         sample_rate: 44100,
    //         bits_per_sample: 32,
    //         sample_format: hound::SampleFormat::Float,
    //     };
    // 
    //     let mut writer = hound::WavWriter::create("output_local.wav", spec).unwrap();
    // 
    //     for (&l, r) in left.iter().zip(right) {
    //         for (&sl, sr) in l.iter().zip(r) {
    //             writer.write_sample(sl / 100000.0).unwrap();
    //             writer.write_sample(sr / 100000.0).unwrap();
    //         }
    //     }
    // 
    //     writer.finalize().unwrap();
    // }
}
