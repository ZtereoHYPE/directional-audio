#[compute]
#version 450

#include "./imports/cubemap.glsl"

// one x per instance
layout(local_size_x = 64) in;

layout(binding = 0) readonly uniform Scene {
    vec3 camera_position;
    uint instance_amount;
};

layout(binding = 1, std430) readonly restrict buffer AudioData {
    vec2 prev_audio_strength; // Left and Right strength
    vec2 audio_strength; // Left and Right strength
    vec3 instances[]; // todo: include other instance data to influence visualzation?
};

layout(binding = 2, r32f) writeonly uniform imageCube heightmap_tex;

layout(binding = 3, r32f) writeonly uniform imageCube ripple_tex;

float get_intensity(vec2 strength, vec3 direction) {
    float factor = direction.x * 0.5 + 0.5;
    return mix(strength.x, strength.y, factor);
}

void main() {
    if (gl_GlobalInvocationID.x >= instance_amount)
        return;

    vec3 instance = instances[gl_GlobalInvocationID.x];

    vec3 direction = normalize(instance - camera_position);
    float intensity = get_intensity(audio_strength, direction);
    float prev_intensity = get_intensity(prev_audio_strength, direction);

    ivec3 tex_coords = direction_to_cubemap(direction);

    // Add dot to heightmap, will get blurred later
    imageStore(heightmap_tex, tex_coords, vec4(1));

    // Store the intensity delta as what will cause ripples (current tex)
    imageStore(ripple_tex, tex_coords, vec4(intensity - prev_intensity));
}