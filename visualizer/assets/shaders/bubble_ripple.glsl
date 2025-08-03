#[compute]
#version 450

#include "./imports/cubemap.glsl"

const float dampening_factor = 0.8;

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) readonly uniform imageCube prev_ripple_tex;
layout(binding = 1, r32f) readonly uniform imageCube curr_ripple_tex;
layout(binding = 2, r32f) writeonly uniform imageCube next_ripple_tex;

void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    if (coords.x >= FACE_RESOLUTION || coords.y >= FACE_RESOLUTION)
        return;
//
//    float prev =    imageLoad(prev_ripple_tex, coords).x;
//    float up =      imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(0, 1))).x;
//    float down =    imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(0,-1))).x;
//    float right =   imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(1, 0))).x;
//    float left =    imageLoad(curr_ripple_tex, cubemap_offset_sample(coords, ivec2(-1,0))).x;
//
//    // next[center] = (current[up] + current[down] + current[left] + current[right]) / 4 - previous[center]
//    float new_height = ((min(up, 1) + min(down,1) + min(left,1) + min(right,1)) / 4 - prev) * dampening_factor;
//    imageStore(next_ripple_tex, coords, vec4(new_height));

    for (int i = 0; i < FACE_RESOLUTION / 2; i++) {
        for (int j = 0; j < FACE_RESOLUTION / 2; j++) {
//            imageStore(next_ripple_tex, ivec3(i, j, 4), vec4(0.5));
            imageStore(next_ripple_tex, cubemap_offset_sample(ivec3(i, j, 4), ivec2(0, 1)), vec4(1));
        }
    }
}