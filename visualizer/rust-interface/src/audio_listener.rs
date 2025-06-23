use std::f32::consts::PI;
use std::path::Path;
use godot::classes::{audio_stream_interactive, AudioServer, AudioStreamPlayer, AudioStreamPlayer3D, IAudioStreamPlayer3D, ProjectSettings};
use godot::classes::audio_stream_generator::AudioStreamGeneratorMixRate;
use godot::classes::AudioStream;
use godot::classes::AudioStreamGenerator;
use godot::classes::AudioStreamGeneratorPlayback;
use godot::classes::AudioStreamPlayback;
use godot::obj::NewAlloc;
use godot::prelude::*;
use godot::classes::Sprite2D;
use godot::classes::IAudioStream;
use godot::private::callbacks::to_string;
use audio_processor;
use audio_processor::AudioEngineMonitor;
use audio_processor::scene::listener::hrtf_filter::{cartesian_to_polar, HrtfOptions};
use audio_processor::scene::mesh::Triangle;
use audio_processor::scene::Scene;
use audio_processor::scene::source::{AudioSource, FRAME_SIZE};
use audio_processor::util::vec3;
use crate::audio_mesh::{AudioMeshNode, AUDIO_MESH_GROUP};
use crate::audio_source::{AudioSourceNode, AUDIO_SOURCE_GROUP};
use crate::to_lib_coords;

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct AudioListenerNode {
    #[export(file = "*.sofa")]
    hrtf_filter: GString,

    #[export]
    export_audio: bool,
    accumulated_samples: Vec<Vector2>,

    last_position: Vector3,
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
        generator.set_mix_rate(44100.0);
        audio_stream_player.set_stream(&generator);

        Self {
            hrtf_filter: GString::from(""),
            export_audio: false,
            accumulated_samples: vec![],
            audio_stream_player,
            last_position: Vector3::ZERO,
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

        // Push new frames to playback
        if let Some(playback) = &self.audio_stream_player.get_stream_playback() {
            let mut playback = playback.clone().cast::<AudioStreamGeneratorPlayback>();

            let max_frames = playback.get_frames_available() as usize / FRAME_SIZE;
            let frames = engine.get_frames(max_frames)
                .iter()
                .flat_map(|(l, r)| l.iter().zip(r))
                .map(|(&l, &r)| Vector2::new(l, r))
                .collect::<PackedVector2Array>();

            playback.push_buffer(&frames);

            if self.export_audio {
                self.accumulated_samples.extend_from_slice(frames.as_slice())
            }
        }
    }

    // gets called every "tick" (physics frame)
    fn physics_process(&mut self, delta: f64) {
        if self.audio_engine.is_none() {
            return;
        }
        let engine = self.audio_engine.as_ref().unwrap();

        // Update listener location
        let position = self.base().get_global_position();
        if position != self.last_position {
            self.last_position = position;
            engine.update_listener(to_lib_coords(position));
        }

        // todo: process the audio sources' new positions

    }

    fn enter_tree(&mut self) {
        let player = self.audio_stream_player.clone();
        self.base_mut().add_child(&player);
    }

    fn exit_tree(&mut self) {
        if self.export_audio {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create("exported.wav", spec).unwrap();

            for vec in &self.accumulated_samples {
                writer.write_sample(vec.x).unwrap();
                writer.write_sample(vec.y).unwrap();
            }

            writer.finalize().unwrap();
        }
    }

    // Gets called after the node structure has "stabilized"
    fn ready(&mut self) {
        // self.audio_stream_player.play();
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
            elevation_max: 0.0, // full sphere was captured
            elevation_min: PI, // "
            sampling_rate: 44100.0
        };

        {
            let relative_pos = vec3::sub(audio_sources[0].location(), to_lib_coords(self.base().get_global_position()));
            print!("relatively speaking, {} {} {}", relative_pos.x, relative_pos.y, relative_pos.z);
            let polar = cartesian_to_polar(relative_pos);
            println!("; in polar: {:?}", polar);
        }

        godot_print!("Creating scene!");
        let scene = Scene::new(
            audio_sources,
            triangles,
            &hrtf_path,
            filter_options
        );

        godot_print!("Creating engine!");
        self.audio_engine = Some(AudioEngineMonitor::start(scene));

        // Start the playback
        self.audio_stream_player.play();
    }
}
