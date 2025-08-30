#[compute]
#version 450

#include "./imports/cubemap.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) uniform imageCube heightmap_tex;
layout(binding = 1, r32f) readonly uniform imageCube ripple_tex;

void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    float ripple = clamp(imageLoad(ripple_tex, coords), 0.0, 0.2).x;
    float height = imageLoad(heightmap_tex, coords).x;
    float sum = height + mix(ripple, 0, height);

    imageStore(heightmap_tex, coords, vec4(sum));
}