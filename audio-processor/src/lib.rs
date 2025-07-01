#![allow(unused)]

mod audio_engine;
pub mod scene;
pub mod util;

use crate::audio_engine::AudioEngine;
use crate::scene::source::Frame;
use crate::scene::Scene;
use crevice::internal::bytemuck::Zeroable;
use crevice::std430::{Mat3, Vec3};
use std::error::Error;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

enum Message {
    Terminate,
    UpdateListener(Vec3, Mat3),
    UpdateSources(Vec<(usize, Vec3)>)
}

struct StateUpdates {
    terminate: bool,
    listener: Option<(Vec3, Mat3)>,
    sources: Option<Vec<(usize, Vec3)>>,
}

pub struct AudioEngineMonitor {
    state: Arc<Mutex<StateUpdates>>,
    frame_rx: Receiver<(Frame, Frame)>,
    vulkan_thread: JoinHandle<()>
}

impl AudioEngineMonitor {
    pub fn start(mut scene: Scene, max_ahead: usize) -> Self {
        // Create sync structures
        let (frame_tx, frame_rx) = mpsc::sync_channel(max_ahead); // render max 10 frames ahead
        let state = Arc::new(Mutex::new(StateUpdates {
            terminate: false,
            listener: (scene.listener.location, scene.listener.rotation).into(),
            sources: Some(scene.sources
                .iter()
                .enumerate()
                .map(|(idx, src)| (idx, src.coordinates))
                .collect())
        }));

        // Start the thread
        let thread_state = state.clone();
        let vulkan_thread = thread::spawn(move || {
            let engine = unsafe {
                AudioEngine::new(scene)
            };

            Self::vulkan_thread_job(engine, thread_state, frame_tx);
        });

        Self {
            state,
            frame_rx,
            vulkan_thread
        }
    }

    pub fn update_listener(&self, pos: Vec3, rot: Mat3) {
        self.state.lock().unwrap().listener = Some((pos, rot));
    }

    pub fn update_sources(&self, sources: Vec<(usize, Vec3)>) {
        self.state.lock().unwrap().sources = Some(sources);
    }

    pub fn get_frames(&self, max: usize) -> Vec<(Frame, Frame)> {
        self.frame_rx.try_iter().take(max).collect()
    }

    pub fn terminate(self) {
        self.state.lock().unwrap().terminate = true;
        self.vulkan_thread.join().unwrap()
    }

    fn vulkan_thread_job(mut engine: AudioEngine, state: Arc<Mutex<StateUpdates>>, frame_tx: SyncSender<(Frame, Frame)>) {
        loop {
            {
                // Check for any updates to listener or sources
                let mut locked_state = state.lock().unwrap();
                if locked_state.terminate {
                    break;
                }

                if let Some((pos, dir)) = locked_state.listener {
                    engine.update_listener(pos, dir);
                    locked_state.listener = None;
                }

                if let Some(sources) = &locked_state.sources {
                    engine.update_sources(sources.clone());
                    locked_state.sources = None;
                }
            }

            if let Err(send_err) = frame_tx.send(engine.process_frames())
                && !state.lock().unwrap().terminate {
                panic!("An error occurred while appending new frames!");
            }
        }
    }
}
