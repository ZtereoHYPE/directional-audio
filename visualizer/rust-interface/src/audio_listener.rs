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
use audio_processor::scene::listener::hrtf_filter::HrtfOptions;
use audio_processor::scene::mesh::Triangle;
use audio_processor::scene::Scene;
use audio_processor::scene::source::{AudioSource, FRAME_SIZE};
use audio_processor::util::vec3;
use crate::audio_mesh::{AudioMeshNode, AUDIO_MESH_GROUP};
use crate::audio_source::{AudioSourceNode, AUDIO_SOURCE_GROUP};
//#[derive(GodotClass)]
//#[class(base=AudioListener3D)]
//struct DirectionalAudioListener {
//    speed: f64,
//    angular_speed: f64,

//    base: Base<AudioStream>
//}


//#[godot_api]
//impl IAudioStream for DirectionalAudioStream {
//    fn init(base: Base<Sprite2D>) -> Self {
//        godot_print!("Hello, sdfsdfsdsdfsdf!"); // Prints to the Godot console

//        let success = audio_processor::run_test();
//        godot_print!("{}", success);
        
//        Self {
//            speed: 400.0,
//            angular_speed: std::f64::consts::PI,
//            base,
//        }
//    }

//    //// physics_process gives us the deltas
//    //fn physics_process(&mut self, delta: f64) {
//    //    // In GDScript, this would be: 
//    //    // rotation += angular_speed * delta
        
//    //    let radians = (self.angular_speed * delta) as f32;
//    //    self.base_mut().rotate(radians);

//    //    let rotation = self.base().get_rotation();
//    //    let velocity = Vector2::UP.rotated(rotation) * self.speed as f32;
//    //    self.base_mut().translate(velocity * delta as f32);
//    //}
//}

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct AudioListenerNode {
    #[export(file = "*.sofa")]
    hrtf_filter: GString,

    last_location: Vector3,
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
            audio_stream_player,
            last_location: Vector3::ZERO,
            audio_engine: None,
            base
        }
    }

    // Gets called every frame
    fn process(&mut self, delta: f64) {
        if self.audio_engine.is_none() {
            // godot_warn!("An audio listener has no engine instance!");
            return;
        }
        let engine = self.audio_engine.as_ref().unwrap();

        // Update listener location
        let location = self.base().get_position();
        if location != self.last_location {
            self.last_location = location;
            engine.update_listener(vec3::from(location.x, location.y, location.z));
        }

        // Push new frames to playback
        if let Some(playback) = &self.audio_stream_player.get_stream_playback() {
            let mut playback = playback.clone().cast::<AudioStreamGeneratorPlayback>();

            let max_frames = playback.get_frames_available() as usize / FRAME_SIZE;
            let frames = engine.get_frames(max_frames)
                .iter()
                .flat_map(|(l, r)| l.iter().zip(r))
                .map(|(&l, &r)| Vector2::new(l, r))  // todo: it might be more efficient to avoid these allocations
                .collect::<PackedVector2Array>();

            godot_print!("Adding {} samples to the audio buffer...", frames.len());

            playback.push_buffer(&frames);
        }
    }

    // gets called every "tick" (physics frame)
    fn physics_process(&mut self, delta: f64) {
        // todo: process the audio sources' new location
    }

    fn enter_tree(&mut self) {
        let player = self.audio_stream_player.clone();
        self.base_mut().add_child(&player);
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
            azimuth_samples: 1, // one every 2 deg // todo: fix these
            elevation_samples: 1, // one every 2 deg
            elevation_max: 0.0, // full sphere was captured
            elevation_min: PI, // "
            sampling_rate: 44100.0
        };

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
