#![allow(unused)]

mod audio_engine;
pub mod scene;
pub mod util;

use crate::audio_engine::AudioEngine;
use crate::scene::source::file::FileAudioProvider;
use crate::scene::source::{AudioSource, Frame, FRAME_SIZE};
use crate::scene::Scene;
use crate::util::vec3;
use crevice::std430::{Mat3, Vec3};
use plotters::prelude::*;
use std::error::Error;
use std::mem::take;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

enum Message {
    Terminate,
    UpdateListener(Vec3, Mat3),
    UpdateSources(Vec<(usize, Vec3)>)
}

pub struct AudioEngineMonitor {
    msg_tx: Sender<Message>,
    frame_rx: Receiver<(Frame, Frame)>,
    vulkan_thread: JoinHandle<()>
}

impl AudioEngineMonitor {
    pub fn start(mut scene: Scene) -> Self {
        let (msg_tx, msg_rx) = mpsc::channel();
        let (frame_tx, frame_rx) = mpsc::channel();

        let vulkan_thread = thread::spawn(move || {
            let engine = unsafe {
                AudioEngine::new(scene)
            };

            Self::vulkan_thread_job(engine, msg_rx, frame_tx);
        });

        Self {
            msg_tx,
            frame_rx,
            vulkan_thread
        }
    }

    pub fn update_listener(&self, pos: Vec3, rot: Mat3) {
        self.msg_tx.send(Message::UpdateListener(pos, rot)).unwrap()
    }

    pub fn update_sources(&self, sources: Vec<(usize, Vec3)>) {
        self.msg_tx.send(Message::UpdateSources(sources)).unwrap()
    }

    pub fn get_frames(&self, max: usize) -> Vec<(Frame, Frame)> {
        self.frame_rx.try_iter().take(max).collect()
    }

    pub fn terminate(self) {
        self.msg_tx.send(Message::Terminate);
        self.vulkan_thread.join().unwrap()
    }

    fn vulkan_thread_job(mut engine: AudioEngine, msg_rx: Receiver<Message>, frame_tx: Sender<(Frame, Frame)>) {
        loop {
            match msg_rx.try_recv() {
                Ok(Message::UpdateListener(pos, rot)) => engine.update_listener(pos, rot),
                Ok(Message::UpdateSources(sources)) => engine.update_sources(sources),
                Ok(Message::Terminate) => break,
                Err(TryRecvError::Empty) => {},
                Err(e) => {
                    println!("Error receiving message: {}", e);
                }
            }

            let now = Instant::now();
            // todo: handle this better (the channel might shut down before and return a SendError
            frame_tx.send(engine.process_frames()).unwrap();

            let ideal_time = (FRAME_SIZE * 1000 / 44100);
            let difference = Duration::from_millis(ideal_time as u64) - now.elapsed();
            thread::sleep(difference);
        }
    }
}


mod test {
    use std::f32::consts::PI;
    use crevice::std430::Vec2;
    use plotters::backend::BitMapBackend;
    use plotters::chart::{ChartBuilder, LabelAreaPosition};
    use plotters::series::LineSeries;
    use plotters::drawing::IntoDrawingArea;
    use plotters::style::{BLUE, GREEN, RED, WHITE};
    use rand::Rng;
    use crate::audio_engine::AudioEngine;
    use crate::audio_engine::gpu_structures::{GpuWindow, GPU_WINDOW_SIZE};
    use crate::audio_engine::signal_processor::fft::FftModule;
    use crate::scene::listener::hrtf_filter::{HrtfFilter, HrtfOptions};
    use crate::scene::mesh::Triangle;
    use crate::scene::Scene;
    use crate::scene::source::{AudioSource, FRAME_SIZE};
    use crate::scene::source::frequency::FrequencyAudioProvider;
    use crate::util::{complex, vec3};

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
                0..GPU_WINDOW_SIZE,
                -10.0..256.0,
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
            &RED,
        )).unwrap();

        chart.draw_series(LineSeries::new(
            fft.iter().map(|&f| f as f64).enumerate(),
            &GREEN,
        )).unwrap();

        chart.draw_series(LineSeries::new(
            ifft.iter().map(|&f| f as f64).enumerate(),
            &BLUE,
        )).unwrap();

        root.present().expect("Unable to write result to file, pr");
    }

    #[test]
    fn test_fft() {
        let sources = vec![
            AudioSource::new(Box::new(FrequencyAudioProvider::new(500.0)), vec3::from(1.0, 0.0, 0.0))
        ];
        let filter_options = HrtfOptions {
            azimuth_samples: 1, // one every 2 deg
            elevation_samples: 1, // one every 2 deg
            elevation_top: 0.0, // full sphere was captured
            elevation_bottom: PI, // "
            sampling_rate: 44100.0
        };

        let triangles = vec![
            Triangle {
                vertices: [
                    vec3::from(1.0, 0.0, 0.0),
                    vec3::from(1.0, 1.0, 0.0),
                    vec3::from(1.0, 1.0, 1.0)
                ]
            }
        ];

        let filter = HrtfFilter::new(filter_options, "datasources/HRIR_FULL2DEG.sofa");
        let scene = Scene::new(sources, triangles, filter);

        let mut engine = unsafe {
            AudioEngine::new(scene)
        };

        // fill the buffer
        // engine.get_fft();
        // engine.get_fft();

        // let (pre_fft, post_fft) = engine.get_fft();
        // let data = pre_fft.into_iter().flatten().collect::<Vec<_>>();
        // let gpu_fft = post_fft.into_iter().flatten().collect::<Vec<_>>();
        // let cpu_fft = FftModule::local_fourier_transform(data.clone(), false);
        //
        // plot_data(
        //     &data.iter().map(|v| v.x * 20.0 + 20.0).collect(),
        //     &gpu_fft.iter().map(|v| complex::magnitude(*v)).collect(),
        //     &cpu_fft.iter().map(|v| complex::magnitude(*v)).collect(),
        //     "test.png"
        // );
    }

    fn make_window() -> Vec<Vec2> {
        let mut frame = [Vec2 {x: 0.0, y: 0.0}; GPU_WINDOW_SIZE];
        let mut phase: f32 = 0.0;

        for idx in 0..frame.len() {
            frame[idx].x = phase.sin();
            frame[idx].y = 0.0;
            phase += 2.0 * PI * (500.0 / 44100.0);
        }

        frame.into()
    }
}