use std::{fs::File, path::Path};

use ash::vk::{Buffer, ImageView};
use engine::VulkanContext;

pub mod engine;
pub mod fft;
pub mod hrtf;

// Util function for submodules
fn read_file_words(path: impl AsRef<Path>) -> Vec<u32> {
    let path = path.as_ref();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let mut file = File::open(&path).unwrap();

    ash::util::read_spv(&mut file).unwrap()
}
