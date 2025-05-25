mod vulkan;
mod audio;

use audio::FRAME_SIZE;
use crevice::std430::Vec2;
use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::prelude::{BitMapBackend, IntoDrawingArea};
use plotters::series::LineSeries;
use plotters::style::full_palette::RED;
use plotters::style::{BLUE, GREEN, WHITE};

#[cfg(test)]
mod tests {
    use crate::audio::{AudioProvider, FRAME_AMT};
    use crate::vulkan::AudioEngine;
    use crate::vulkan::signal_processor::fft::FftModule;
    use crate::vulkan::signal_processor::FftBuffer;
    use super::*;

    pub unsafe fn alloc_empty_buffer() -> Box<FftBuffer> {
        let layout = std::alloc::Layout::new::<FftBuffer>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut FftBuffer;

        Box::from_raw(ptr)
    }

    pub fn plot_data(frame: &Vec<Vec2>, fft: &Vec<Vec2>, ifft: &Vec<Vec2>, name: &str) {
        let root = BitMapBackend::new(name, (1280, 720)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                "Averge temperature in Salt Lake City, UT",
                ("sans-serif", 40),
            )
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Right, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(
                0..FRAME_SIZE,
                -2.0..2.0,
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
            frame.iter().enumerate().map(|(x, y)| (x, (*y).x as f64)),
            &BLUE,
        )).unwrap();

        chart.draw_series(LineSeries::new(
            fft.iter().enumerate().map(|(x, y)| (x, (*y).x as f64)),
            &RED,
        )).unwrap();

        chart.draw_series(LineSeries::new(
            ifft.iter().enumerate().map(|(x, y)| (x, (*y).x as f64 )),
            &GREEN,
        )).unwrap();

        root.present().expect("Unable to write result to file, pr");
    }

    #[test]
    fn test() {
        unsafe {
            let mut engine = AudioEngine::new();

            let mut buffer: Box<FftBuffer> = alloc_empty_buffer();
            for idx in 0..FRAME_AMT {
                buffer.frames[idx] = FftModule::frame_to_fft(&AudioProvider::random_frame(4));
            }

            let (left, right) = engine.process_frames(buffer.clone());

            plot_data(&(*buffer).frames[0].into(), &(*left).into(), &(*right).into(), "./roundtrip.png");
        }
    }

    #[test]
    fn fft_test() {
        unsafe {
            let mut engine = AudioEngine::new();
            let frame = FftModule::frame_to_fft(&AudioProvider::random_frame(16));

            let fft = engine.fft_gpu(Box::from(frame.clone()));
            
            let ifft = FftModule::local_fourier_transform((*fft).into(), true);

            plot_data(&frame.into(), &(*fft).into(), &ifft, "./fft.png");
        }
    }
    
    
    #[test]
    fn audio_test() {
        let sampels = AudioProvider::from_file("./datasources/sample-15s.wav");
    }
}
