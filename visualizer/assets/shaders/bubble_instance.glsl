#[compute]
#version 450

#include "./imports/cubemap.glsl"

// one x per instance
layout(local_size_x = 8, local_size_y = 8) in;

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
    // todo: use a different horizontal axys depending on the camera rotation
    float factor = direction.x * 0.5 + 0.5;
    return mix(strength.x, strength.y, factor);
}

void main() {
    ivec3 tex_coords = ivec3(gl_GlobalInvocationID.xyz);

    if (gl_GlobalInvocationID.x >= FACE_RESOLUTION || gl_GlobalInvocationID.y >= FACE_RESOLUTION)
        return;

    vec3 local_direction = cubemap_to_direction(tex_coords);

    // todo: optimize using workgroups
    float height = 0.0;
    for (int idx = 0; idx < instance_amount; idx++) {
        vec3 instance_dir = -normalize(instances[idx] - camera_position);
        float dist = distance(local_direction, instance_dir);

        // Spawn a ripple if we are on the tip pixel
        if (dist <= 1.0 / FACE_RESOLUTION)
            imageStore(ripple_tex, tex_coords, vec4(0.5));

        // Calculate the height added from the instance
        const float STDEV_2 = 0.008;
        float value = exp(-(dist * dist / STDEV_2));
        height += value;
    }

    imageStore(heightmap_tex, tex_coords, vec4(height));
}
