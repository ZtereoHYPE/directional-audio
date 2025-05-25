use godot::classes::audio_stream_interactive;
use godot::classes::AudioStream;
use godot::classes::AudioStreamGenerator;
use godot::classes::AudioStreamGeneratorPlayback;
use godot::classes::AudioStreamPlayback;
use godot::obj::NewAlloc;
use godot::prelude::*;
use godot::classes::Sprite2D;
use godot::classes::IAudioStream;

use audio_processor;

use crate::player;

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
struct DirectionalAudioListener {
    audio_stream_player: Gd<AudioStreamPlayer>, // warning: never gets free'd
    playback: Option<Gd<AudioStreamGeneratorPlayback>>,
    base: Base<Node3D>
}

#[godot_api]
impl INode3D for DirectionalAudioListener {
    fn init(base: Base<Node3D>) -> Self {
        // Create an audio stream player
        let mut audio_stream_player = AudioStreamPlayer::new_alloc();

        // Set its stream to a stream generator
        audio_stream_player.set_stream(&AudioStreamGenerator::new_gd());

        // Get its playback
        let playback = audio_stream_player
            .get_stream_playback()
            .and_then(|p| p.try_cast::<AudioStreamGeneratorPlayback>().ok());

        Self {
            audio_stream_player,
            playback,
            base
        }
    }

    // Gets called after the node structure has "stabilized"
    fn ready(&mut self) {
        self.audio_stream_player.play();
    }

    // Gets called every frame
    fn process(&mut self, delta: f64) {

        // If we have a playback...
        if let Some(playback) = &self.playback {

            // this is how many samples of audio can be pushed
            let available_samples = playback.get_frames_available();
            
            // if they are larger than a vulkan frame, then call that and push it
            // at 44.1kHz, 512 samples is 11.6ms of time, and can hold frequencies down to 90Hz, without partitioned convolution
            // todo: a problem with this is pacing: the rate at which samples are requested from vulkan will depend on the framerate. We want it to be independent and faster
            // the library needs to be aware of the sample rate to be able to measure the passage of time!
        }

    }

    // gets called every "tick" (physics frame)
    fn physics_process(&mut self, delta: f64) {
        // todo: update relative position and velocity of every audio source
    }
}

// todo: define:
// - directional audio emitter class
// - audio listener class -> extends AudioListener3D
