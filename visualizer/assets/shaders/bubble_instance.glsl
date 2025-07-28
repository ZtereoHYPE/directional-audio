#[compute]
#version 450

#define PI 3.1415

const ivec2 TEX_SIZE = ivec2(64, 32);

// one x per instance
layout(local_size_x = 64) in;

layout(binding = 0, std430) readonly restrict buffer AudioData {
    vec2 audioStrength; // Left and Right strength
    vec3 instances[]; // todo: include other instance data to influence visualzation?
};

// todo: look for a better format, we don't need all this precision do we
layout(binding = 1, rgba32f) writeonly uniform image2D bubble_height;

layout(binding = 2) readonly uniform Camera {
    vec3 camera_position;
};

float get_intensity(vec3 direction) {
    float factor = direction.x * 0.5 + 0.5;
    return mix(audioStrength.x, audioStrength.y, factor);
}

vec2 sphere_uv(vec3 direction) {
    return vec2(
        0.5 + atan(direction.z, direction.x) / (2 * PI),
        0.5 + asin(direction.y) / PI
    );
}


void main() {
    vec3 instance = instances[gl_GlobalInvocationID.x];

    vec3 direction = normalize(instance - camera_position);
    float intensity = get_intensity(direction);

    vec2 uv_coords = sphere_uv(direction);
    ivec2 tex_coords = ivec2(round(uv_coords * TEX_SIZE));

    imageStore(bubble_height, tex_coords, vec4(intensity, 0.0, 0.0, 1.0));
}