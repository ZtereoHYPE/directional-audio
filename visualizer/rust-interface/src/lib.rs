use godot::prelude::*;

mod audio_listener;
mod audio_source;
mod audio_mesh;

struct AudioVisualization;

#[gdextension]
unsafe impl ExtensionLibrary for AudioVisualization {}
