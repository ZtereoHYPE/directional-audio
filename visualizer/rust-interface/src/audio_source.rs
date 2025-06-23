use std::path::Path;
use godot::classes::{CsgCombiner3D, FileAccess, ICsgCombiner3D, ProjectSettings};
use godot::obj::{Base, Gd, WithBaseField, WithUserSignals};
use godot::prelude::*;
use audio_processor::scene::mesh::Triangle;
use audio_processor::scene::source::{AudioProvider, AudioSource};
use audio_processor::scene::source::file::FileAudioProvider;
use audio_processor::scene::source::frequency::FrequencyAudioProvider;
use audio_processor::scene::source::quiet::SilentAudioProfider;
use audio_processor::util::vec3;
use crate::audio_mesh::AudioMeshNode;
use crate::to_lib_coords;

pub const AUDIO_SOURCE_GROUP: &str = "AudioSources";

#[derive(GodotClass)]
#[class(init, base=Node3D)]
pub struct AudioSourceNode {
    #[export(file = "*.wav")]
    audio_file: GString,
    #[export]
    loop_audio: bool,

    base: Base<Node3D>
}

#[godot_api]
impl INode3D for AudioSourceNode {
    fn enter_tree(&mut self) {
        self.base_mut().add_to_group(AUDIO_SOURCE_GROUP);
    }
}

impl AudioSourceNode {
    pub fn get_audio_source(&self) -> AudioSource {
        let pos = self.base().get_global_position();
        let source_path = ProjectSettings::singleton().globalize_path(&self.audio_file).to_string();

        let provider: Box<dyn AudioProvider + Send> =
            if !Path::new(&source_path).is_file() {
                godot_warn!("An audio source does not have its source file set! It will play a silent stream");
                Box::from(SilentAudioProfider)
            } else {
                Box::from(FileAudioProvider::new(&source_path, self.loop_audio))
            };

        AudioSource::new(
            provider,
            to_lib_coords(pos)
        )
    }
}

