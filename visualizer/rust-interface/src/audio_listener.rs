use crate::audio_mesh::{AudioMeshNode, AUDIO_MESH_GROUP};
use crate::audio_source::{AudioSourceNode, AUDIO_SOURCE_GROUP};
use crate::to_vec;
use crate::visualization::GodotVisualizationData;
use audio_processor;
use audio_processor::scene::listener::hrtf_filter::{HrtfFilter, HrtfOptions};
use audio_processor::scene::source::FRAME_SIZE;
use audio_processor::scene::Scene;
use audio_processor::util::rotation_matrix;
use audio_processor::AudioEngineMonitor;
use godot::classes::audio_stream_generator::AudioStreamGeneratorMixRate;
use godot::classes::AudioStreamGenerator;
use godot::classes::AudioStreamGeneratorPlayback;
use godot::classes::{AudioStreamPlayer, ProjectSettings};
use godot::obj::NewAlloc;
use godot::prelude::*;
use std::f32::consts::PI;
use std::path::Path;

const SAMPLE_RATE: f64 = 44100.0;

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct AudioListenerNode {
    #[export]
    enabled: bool,
    #[export(file = "*.sofa")]
    hrtf_filter: GString,
    #[export]
    export_audio: bool,
    #[export]
    max_frames_ahead: u32,

    volume: f32,

    accumulated_samples: Vec<Vector2>,
    allowed_samples: f64,

    last_position: Vector3,
    last_rotation: Vector3,

    audio_stream_player: Gd<AudioStreamPlayer>, // warning: never gets free'd
    audio_engine: Option<AudioEngineMonitor>,

    base: Base<Node3D>
}

#[godot_api]
impl INode3D for AudioListenerNode {
    fn init(base: Base<Node3D>) -> Self {
        // Create an audio stream player
        let mut audio_stream_player = AudioStreamPlayer::new_alloc();

        // Create a stream generator with 44.1kHz sampling rate
        let mut generator = AudioStreamGenerator::new_gd();
        generator.set_mix_rate_mode(AudioStreamGeneratorMixRate::CUSTOM);
        generator.set_mix_rate(SAMPLE_RATE as f32);
        audio_stream_player.set_stream(&generator);

        Self {
            enabled: true,
            hrtf_filter: GString::from(""),
            export_audio: false,
            max_frames_ahead: 5,
            volume: 0.75,
            accumulated_samples: vec![],
            allowed_samples: 5.0,
            last_position: Vector3::ZERO,
            last_rotation: Vector3::ZERO,
            audio_stream_player,
            audio_engine: None,
            base
        }
    }

    // Gets called every frame
    fn process(&mut self, delta: f64) {
        if self.audio_engine.is_none() {
            return;
        }

        let engine = self.audio_engine.as_ref().unwrap();
        self.allowed_samples += delta * SAMPLE_RATE / FRAME_SIZE as f64;

        // Push new frames to playback
        if let Some(playback) = &self.audio_stream_player.get_stream_playback() {
            let mut playback = playback.clone().cast::<AudioStreamGeneratorPlayback>();

            let max_frames = usize::min(playback.get_frames_available() as usize / FRAME_SIZE, self.allowed_samples as usize);
            self.allowed_samples -= max_frames as f64;
            
            let frames = engine.get_frames(max_frames)
                .iter()
                .flat_map(|(l, r)| l.iter().zip(r))
                .map(|(&l, &r)| Vector2::new(l * self.volume, r * self.volume))
                .collect::<PackedVector2Array>();

            playback.push_buffer(&frames);

            if self.export_audio {
                self.accumulated_samples.extend_from_slice(frames.as_slice())
            }
        }

        // todo: rename from Debug to visualization
        if let Some(debug_data) = engine.get_debug_data() {
            let mut vis_data = GodotVisualizationData::new_gd();
            vis_data.bind_mut().set_data(debug_data);
            self.signals().visualization_data_received().emit(&vis_data);
        }
    }

    // gets called every "tick" (physics frame)
    fn physics_process(&mut self, _: f64) {
        if self.audio_engine.is_none() {
            return;
        }

        let engine = self.audio_engine.as_ref().unwrap();

        // Update listener location
        let position = self.base().get_global_position();
        let rotation = self.base().get_global_rotation();
        if position != self.last_position || rotation != self.last_rotation {
            self.last_position = position;
            self.last_rotation = rotation;

            engine.update_listener(to_vec(position), rotation_matrix(rotation.x, rotation.y));
        }

        engine.request_debug(); // for now, only request debug once per tick. Later do once per frame.
    }

    fn enter_tree(&mut self) {
        let player = self.audio_stream_player.clone();
        self.base_mut().add_child(&player);
    }

    fn exit_tree(&mut self) {
        if !self.enabled {
            return;
        }

        if self.export_audio {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };

            let mut writer = hound::WavWriter::create("exported.wav", spec).unwrap();

            for vec in &self.accumulated_samples {
                writer.write_sample((vec.x * 32_768.0) as i16).unwrap();
                writer.write_sample((vec.y * 32_768.0) as i16).unwrap();
            }

            writer.finalize().unwrap();
        }
    }

    // Gets called after the node structure has "stabilized"
    fn ready(&mut self) {
        if !self.enabled {
            return;
        }

        self.allowed_samples = self.max_frames_ahead as f64;

        let hrtf_path = ProjectSettings::singleton().globalize_path(&self.hrtf_filter).to_string();
        if !Path::new(&hrtf_path).is_file() {
            godot_warn!("The listener has no HRTF filter set!");
            return;
        }

        // Collect all the mesh triangles
        let triangles = self.base()
            .get_tree()
            .unwrap()
            .get_nodes_in_group(AUDIO_MESH_GROUP)
            .iter_shared()
            .filter_map(|n| n.try_cast::<AudioMeshNode>().ok())
            .flat_map(|n| n.bind().get_mesh_triangles())
            .collect::<Vec<_>>();

        // Collect all the audio sources
        let audio_sources = self.base()
            .get_tree()
            .unwrap()
            .get_nodes_in_group(AUDIO_SOURCE_GROUP)
            .iter_shared()
            .filter_map(|n| n.try_cast::<AudioSourceNode>().ok())
            .map(|n| n.bind().get_audio_source())
            .collect::<Vec<_>>();

        let filter_options = HrtfOptions {
            azimuth_samples: 90, // one every 2 deg
            elevation_samples: 45, // one every 2 deg
            elevation_top: 0.0, // full sphere was captured
            elevation_bottom: PI, // "
            sampling_rate: 44100.0
        };

        godot_print!("Creating filter!");
        let filter = HrtfFilter::new(filter_options, &hrtf_path);

        godot_print!("Creating scene!");
        let scene = Scene::new(
            audio_sources,
            triangles,
            filter
        );

        godot_print!("Creating engine!");
        self.audio_engine = Some(AudioEngineMonitor::start(scene, self.max_frames_ahead as usize));

        // Start the playback
        self.audio_stream_player.play();
    }
}

#[godot_api]
impl AudioListenerNode {
    #[func]
    fn on_volume_change(&mut self, volume: f32) {
        self.volume = volume / 100.0;
    }

    #[func]
    fn set_play_state(&mut self, state: bool) {
        if let Some(engine) = &self.audio_engine {
            engine.set_play_state(state);
        }
    }

    #[signal]
    fn visualization_data_received(data: Gd<GodotVisualizationData>);
}
