use std::process::Command;

fn main() {
    // todo: make build cross platform by using shaderc crate
    let status = Command::new("make")
        .args(&["shaders"])
        .status()
        .expect("Failed to compile shaders!");

    assert!(status.success());
}
