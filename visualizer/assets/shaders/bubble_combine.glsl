#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) uniform imageCube heightmap_tex;
layout(binding = 1, r32f) readonly uniform imageCube ripple_tex;

void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    vec4 sum = imageLoad(heightmap_tex, coords) + imageLoad(ripple_tex, coords);
    imageStore(heightmap_tex, coords, vec4(sum));
}