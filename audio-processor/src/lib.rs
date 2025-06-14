#![allow(unused)]

mod audio_engine;
pub mod scene;
mod util;

use crate::audio_engine::AudioEngine;
use crate::scene::source::file::FileAudioProvider;
use crate::scene::source::{AudioSource, Frame};
use crate::scene::Scene;
use crate::util::vec3;
use crevice::std430::Vec3;
use plotters::prelude::IntoDrawingArea;
use std::error::Error;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::thread::JoinHandle;

enum Message {
    Terminate,
    UpdateListener(Vec3),
    UpdateSources(Vec<(usize, Vec3)>)
}

struct AudioEngineMonitor {
    msg_tx: Sender<Message>,
    frame_rx: Receiver<(Frame, Frame)>,
    vulkan_thread: JoinHandle<()>
}

impl AudioEngineMonitor {
    pub fn start(mut scene: Scene) -> Self {
        let (msg_tx, msg_rx) = mpsc::channel();
        let (frame_tx, frame_rx) = mpsc::channel();

        scene.sources.push(AudioSource::new(Box::from(FileAudioProvider::new("mrow")), vec3::from(0.0,0.0,0.0)));

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

    pub fn update_listener(&self, listener: Vec3) {
        self.msg_tx.send(Message::UpdateListener(listener)).unwrap()
    }

    pub fn update_sources(&self, sources: Vec<(usize, Vec3)>) {
        self.msg_tx.send(Message::UpdateSources(sources)).unwrap()
    }

    pub fn get_next_frame(&self) -> Option<(Frame, Frame)> {
        self.frame_rx.try_recv().ok()
    }

    pub fn terminate(self) {
        self.msg_tx.send(Message::Terminate);
        self.vulkan_thread.join().unwrap()
    }

    fn vulkan_thread_job(mut engine: AudioEngine, msg_rx: Receiver<Message>, frame_tx: Sender<(Frame, Frame)>) {
        loop {
            match msg_rx.try_recv() {
                Ok(Message::UpdateListener(listener)) => engine.update_listener(listener),
                Ok(Message::UpdateSources(sources)) => engine.update_sources(sources),
                Ok(Message::Terminate) | Err(_) => break,
            }

            frame_tx.send(engine.process_frames()).unwrap()
        }
    }
}
