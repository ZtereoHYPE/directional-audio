#![allow(unused)]
extern crate core;

mod audio_engine;
pub mod scene;
pub mod util;
mod vulkan;

use crate::audio_engine::rays::RayBufferData;
use crate::audio_engine::{AudioEngine, AudioInstance};
use crate::scene::source::Frame;
use crate::scene::Scene;
use glam::{Mat3, Vec3};
use std::error::Error;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

pub struct VisualizationData {
    pub last_rt_origin: Vec3,
    pub rays: Box<RayBufferData>,
    pub instances: Vec<AudioInstance>
}

enum EngineAction {
    Terminate,
    RequestVisData,
    Pause,
    Play
}

struct StateUpdates {
    listener: Option<(Vec3, Mat3)>,
    sources: Option<Vec<(usize, Vec3)>>,
}

pub struct AudioEngineMonitor {
    state: Arc<Mutex<StateUpdates>>,
    frame_rx: Receiver<(Frame, Frame)>,
    debug_rx: Receiver<VisualizationData>,
    action_tx: Sender<EngineAction>,
    vulkan_thread: JoinHandle<()>
}

impl AudioEngineMonitor {
    pub fn start(mut scene: Scene, max_ahead: usize) -> Self {
        // Create sync structures
        let (frame_tx, frame_rx) = mpsc::sync_channel(max_ahead); // render max 10 frames ahead
        let (debug_tx, debug_rx) = mpsc::channel();
        let (action_tx, action_rx) = mpsc::channel();

        let state = Arc::new(Mutex::new(StateUpdates {
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
                AudioEngine::new(scene, 2)
            };

            Self::vulkan_thread_job(engine, thread_state, frame_tx, debug_tx, action_rx);
        });

        Self {
            state,
            frame_rx,
            debug_rx,
            action_tx,
            vulkan_thread
        }
    }

    pub fn update_listener(&self, pos: Vec3, rot: Mat3) {
        self.state.lock().unwrap().listener = Some((pos, rot));
    }

    pub fn update_sources(&self, sources: Vec<(usize, Vec3)>) {
        self.state.lock().unwrap().sources = Some(sources);
    }

    pub fn set_play_state(&self, playing: bool) {
        if playing {
            self.action_tx.send(EngineAction::Play).unwrap();
            self.vulkan_thread.thread().unpark();
        } else {
            self.action_tx.send(EngineAction::Pause).unwrap();
        }
    }

    pub fn request_debug(&self) {
        self.action_tx.send(EngineAction::RequestVisData).unwrap();
    }

    pub fn get_frames(&self, max: usize) -> Vec<(Frame, Frame)> {
        self.frame_rx.try_iter().take(max).collect()
    }

    pub fn get_debug_data(&self) -> Option<VisualizationData> {
        self.debug_rx.try_recv().ok()
    }

    pub fn terminate(self) {
        self.action_tx.send(EngineAction::Play).unwrap();
        self.action_tx.send(EngineAction::Terminate).unwrap();
        self.vulkan_thread.thread().unpark();
        self.vulkan_thread.join().unwrap()
    }

    fn vulkan_thread_job(
        mut engine: AudioEngine,
        scene_state: Arc<Mutex<StateUpdates>>,
        frame_tx: SyncSender<(Frame, Frame)>,
        debug_tx: Sender<VisualizationData>,
        action_rx: Receiver<EngineAction>
    ) {
        loop {
            let mut debug_data = false;
            match action_rx.try_recv() {
                Ok(EngineAction::Terminate)                         => break,
                Ok(EngineAction::Pause)                             => Self::park_vulkan_thread(&action_rx),
                Ok(EngineAction::RequestVisData)                    => debug_data = true,
                Err(TryRecvError::Empty) | Ok(EngineAction::Play)   => {},
                Err(e)                                  => println!("Error receiving message: {}", e)
            }

            // Check for any updates to listener or sources
            {
                let mut locked_state = scene_state.lock().unwrap();
                if let Some((pos, dir)) = locked_state.listener {
                    engine.update_listener(pos, dir);
                    locked_state.listener = None;
                }

                if let Some(sources) = &locked_state.sources {
                    engine.update_sources(sources.clone());
                    locked_state.sources = None;
                }
            }

            let debug_callback = if debug_data {
                Some(|vis_data| debug_tx.send(vis_data).unwrap())
            } else {
                None
            };

            let sender_copy = frame_tx.clone();
            engine.request_frame(
                move |left, right| {
                    println!("Received frames!");
                    sender_copy.send((left, right)).unwrap()
                },
                debug_callback
            );
        }
    }

    fn park_vulkan_thread(action_rx: &Receiver<EngineAction>) {
        loop {
            thread::park();
        
            'consume_messages: loop {
                match action_rx.try_recv() {
                    Ok(EngineAction::Play) => return,
                    Err(TryRecvError::Empty) => break 'consume_messages,
                    Ok(_) | Err(_) => {}
                }
            }
        }
    }
}
