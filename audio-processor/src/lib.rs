mod vulkan;
mod audio;

use std::time::Instant;

pub fn run_test() -> bool {
    unsafe {
        let mut engine = vulkan::VulkanEngine::new();

        let frame = audio::AudioProvider::next_frame();

        let past = Instant::now();
        let processed_frame = engine.process_frame(&frame);
        let future = Instant::now();
        println!("{:?}", future - past);

        for value in processed_frame {
            if value != 256 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn it_works() {
        unsafe {
            let mut engine = vulkan::VulkanEngine::new();

            let frame = audio::AudioProvider::next_frame();

            let past = Instant::now();
            let processed_frame = engine.process_frame(&frame);
            let future = Instant::now();
            println!("{:?}", future - past);

            for value in processed_frame {
                assert_eq!(value, 256);
            }
        }
    }
}
