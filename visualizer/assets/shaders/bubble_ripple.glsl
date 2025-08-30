#[compute]
#version 450

#include "./imports/cubemap.glsl"

const float vel_dampening_factor = 1.0;
const float height_dampening_factor = 0.9;

// time_step < cell_dist / speed
const float speed = 1.0;
const float cell_dist = 1.0;
const float time_step = 0.5;

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) uniform imageCube velocity_ripple_tex;
layout(binding = 1, r32f) readonly uniform imageCube curr_ripple_tex;
layout(binding = 2, r32f) writeonly uniform imageCube next_ripple_tex;

void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    if (coords.x >= FACE_RESOLUTION || coords.y >= FACE_RESOLUTION)
        return;

    float up =      imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(0,  1))).x;
    float down =    imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(0, -1))).x;
    float left =    imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(-1, 0))).x;
    float right =   imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(1,  0))).x;
    float curr =    imageLoad(curr_ripple_tex, coords).x;
    float vel =     imageLoad(velocity_ripple_tex, coords).x;

    float f = speed * speed * (up + down + left + right - 4 * curr) / (cell_dist * cell_dist);
    float new_vel = vel + f * time_step;
    float new_height = curr + new_vel * time_step;

    imageStore(velocity_ripple_tex, coords, vec4(new_vel));
    imageStore(next_ripple_tex, coords, vec4(new_height * height_dampening_factor));
}
