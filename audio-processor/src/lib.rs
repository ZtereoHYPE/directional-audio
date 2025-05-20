mod vulkan;
mod audio;

use std::time::Instant;
use audio::{Frame, FRAME_SIZE};
use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea};
use plotters::series::LineSeries;
use plotters::style::full_palette::RED;
use plotters::style::{BLUE, GREEN, WHITE};
use vulkan::engine::VulkanBuilder;
use vulkan::fft::{FftModule};

fn plot_data(frame: &Frame, gpu_fft: &Frame, name: &str) {
    let root = BitMapBackend::new(name, (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(
            "Fourier transform",
            ("sans-serif", 40),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            0..FRAME_SIZE,
            -4.0..4.0,
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
        frame.iter().enumerate().map(|(x, y)| (x, *y as f64)),
        &BLUE,
    )).unwrap();

    chart.draw_series(LineSeries::new(
        gpu_fft.iter().enumerate().map(|(x, y)| (x, *y as f64)),
        &RED,
    )).unwrap();

    root.present().expect("Unable to write result to file, pr");
}

#[cfg(test)]
mod tests {
    use crevice::std140::Vec2;

    use crate::{audio::{AudioProvider, FRAME_AMT}, vulkan::fft::{alloc_empty_buffer, FftBuffer, FftFrame}};
    use super::*;

    const EPSILON: f32 = 0.00001;

    #[test]
    fn it_works() {
        unsafe {
            let mut buffer: Box<FftBuffer> = alloc_empty_buffer();

            for idx in 0..FRAME_AMT {
                buffer.frames[idx] = FftModule::frame_to_fft(&AudioProvider::next_frame());
            }

            let mut engine = VulkanBuilder::new()
                .register_module::<FftModule>()
                .register_module::<FftModule>()
                .build();
        
            let mut fft = FftModule::new(&mut engine, false);
            let mut ifft = FftModule::new(&mut engine, true);

            let start = Instant::now();
            let computed_fft = fft.process_buffer(&engine, &buffer);
            println!("time: {:?}", Instant::now() - start);

            let computed_ifft = ifft.process_buffer(&engine, &computed_fft);

            for idx in 0..FRAME_AMT {
                let before = FftModule::fft_to_frame(&buffer.frames[idx]);
                let after = FftModule::fft_to_frame(&computed_ifft.frames[idx]);

                for (s_before, s_after) in before.iter().zip(after) {
                    let diff = (s_after - s_before).abs();
                    assert!(diff < EPSILON);
                }
            }
        }
    }
}
