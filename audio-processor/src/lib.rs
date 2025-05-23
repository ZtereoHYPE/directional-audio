mod vulkan;
mod audio;

use std::time::Instant;
use audio::{Frame, FRAME_SIZE};
use crevice::std430::Vec2;
use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea};
use plotters::series::LineSeries;
use plotters::style::full_palette::RED;
use plotters::style::{BLUE, GREEN, WHITE};
use vulkan::fft::{magnitude, FftModule};

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::{audio::{hrtf::{HrtfFilter, HrtfOptions}, AudioProvider, FRAME_AMT}, vulkan::{engine::VulkanBuilder, fft::{cpu_fft, root_of_unity, FftBuffer, FftFrame}, hrtf::HrtfModule}};
    use super::*;

    const EPSILON: f32 = 0.0005;

    pub unsafe fn alloc_empty_buffer() -> Box<FftBuffer> {
        let layout = std::alloc::Layout::new::<FftBuffer>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut FftBuffer;

        Box::from_raw(ptr)
    }

    fn plot_data(frame: &Vec<Vec2>, fft: &Vec<Vec2>, ifft: &Vec<Vec2>) {
        let root = BitMapBackend::new("./test.png", (1280, 720)).into_drawing_area();
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
    fn hrtf_test() {
        unsafe {
            let filter_options = HrtfOptions {
                //elevation_samples: 90, // one every 2 deg
                //azimuth_samples: 180, // one every 2 deg
                azimuth_samples: 4,
                elevation_samples: 4,
                elevation_max: PI, // full sphere was captured
                elevation_min: 0.0, // "
                sampling_rate: 48000.0
            };
            
            let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa", FRAME_SIZE);

            let mut buffer: Box<FftBuffer> = alloc_empty_buffer();

            for idx in 0..FRAME_AMT {
                buffer.frames[idx] = FftModule::frame_to_fft(&AudioProvider::next_frame());
            }

            let mut engine = VulkanBuilder::new()
                .register_module::<FftModule>()
                .register_module::<FftModule>()
                .register_module::<HrtfModule>()
                .build();
        
            let mut fft = FftModule::new(&mut engine, false);
            let mut ifft = FftModule::new(&mut engine, false);

            let computed_fft = fft.process_buffer(&engine, &buffer);
            
            let hrtf = HrtfModule::new(&mut engine, filter, &computed_fft);
            let computed_hrtf = hrtf.apply(&mut engine);

            ifft.process_buffer(&mut engine, &Box::new(FftBuffer {
                frames: [*computed_hrtf.0]
            }));

            let result = cpu_fft(computed_hrtf.0.samples.into(), root_of_unity(-(FRAME_SIZE as isize)));

            // todo: compare before and after in the frequency domain
            plot_data(&buffer.frames[0].samples.into(), &(*computed_hrtf.0).samples.into(), &result);

            //for idx in 0..FRAME_AMT {
            //    let before = FftModule::fft_to_frame(&buffer.frames[idx]);
            //    let after = FftModule::fft_to_frame(&computed_ifft.frames[idx]);

            //    for (s_before, s_after) in before.iter().zip(after) {
            //        let diff = (s_after - s_before).abs();
            //        assert!(diff < EPSILON);
            //    }
            //}
        }
    }

    //#[test]
    //fn gpu_fft_test() {
    //    unsafe {
    //        //let mut engine = VulkanBuilder::new()
    //        //    .register_module::<FftModule>()
    //        //    .register_module::<HrtfModule>()
    //        //    .build();

    //        //let mut fft = FftModule::new(&mut engine, false);
    //        //let mut hrtf = FftModule::new(&mut engine, true);
            
    //        //let computed_fft = fft.process_buffer(&engine, &buffer);
    //        //let computed_ifft = fft.process_buffer(&mut engine);

    //        //for idx in 0..FRAME_AMT {
    //        //    let before = FftModule::fft_to_frame(&buffer.frames[idx]);
    //        //    let after = FftModule::fft_to_frame(&computed_ifft.frames[idx]);

    //        //    for (s_before, s_after) in before.iter().zip(after) {
    //        //        let diff = (s_after - s_before).abs();
    //        //        assert!(diff < EPSILON);
    //        //    }
    //        //}
    //    }
    //}

    #[test]
    fn cpu_fft_test() {
        let vector = Vec::from(FftModule::frame_to_fft(&AudioProvider::next_frame()).samples);
        let fft = cpu_fft(vector.clone(), root_of_unity(FRAME_SIZE as isize));
        let ifft = cpu_fft(fft.clone(), root_of_unity(-(FRAME_SIZE as isize)));

        // plot_data(&vector, &fft, &ifft);

        for (&s_before, s_after) in vector.iter().zip(ifft) {
            let diff = (s_after.x / FRAME_SIZE as f32 - s_before.x).abs();
            assert!(diff < EPSILON);
        }
    }
}
