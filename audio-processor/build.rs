use std::process::Command;

fn main() {
    // todo: make build cross platform by using shaderc crate
    Command::new("make")
        .args(&["shaders"])
        .status()
        .expect("Failed to compile shaders!");
}
