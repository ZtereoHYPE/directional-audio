use godot::prelude::*;

mod player;
mod audio;
struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}
