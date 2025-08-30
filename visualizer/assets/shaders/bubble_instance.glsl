#[compute]
#version 450

#include "./imports/cubemap.glsl"

// one x per instance
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) readonly uniform Scene {
    vec3 camera_position;
    uint instance_amount;
};

struct Instance {
    vec3 position;
    float cluster_size;
    float loudness;
    float prev_loudness;
};

layout(binding = 1, std430) readonly restrict buffer AudioData {
    Instance instances[];
};

layout(binding = 2, r32f) writeonly uniform imageCube heightmap_tex;
layout(binding = 3, r32f) writeonly uniform imageCube ripple_tex;

void main() {
    ivec3 tex_coords = ivec3(gl_GlobalInvocationID.xyz);

    if (gl_GlobalInvocationID.x >= FACE_RESOLUTION || gl_GlobalInvocationID.y >= FACE_RESOLUTION)
        return;

    vec3 local_direction = cubemap_to_direction(tex_coords);

    // todo: optimize using workgroups
    float height = 0.0;
    float ripple_height = 0.0;
    for (int idx = 0; idx < instance_amount; idx++) {
        Instance instance = instances[idx];
        vec3 instance_dir = -normalize(instance.position - camera_position);
        float dist = distance(local_direction, instance_dir);

        // Vary the width based on how much is clustered
        float width = float(clamp(1, 1, 40)) / 40.0;
        float stddev_2 = 0.0055 + width * 0.015;

        // Calculate the height added from the instance
        float value = exp(-(dist * dist / stddev_2));
        height += value * (instance.loudness * 0.2 + 1.0);

        // If we're at the tip of the bell curve, add a ripple
        if (value > 0.98)
            ripple_height += max(instance.loudness - instance.prev_loudness, 0.0);
    }

    imageStore(heightmap_tex, tex_coords, vec4(height));

    if (ripple_height > 0.0)
        imageStore(ripple_tex, tex_coords, vec4(ripple_height));
}
