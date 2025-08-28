#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, r32f) readonly uniform imageCube from_tex;
layout(binding = 1, r32f) writeonly uniform imageCube to_tex;

void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    vec4 value = imageLoad(from_tex, coords);
    imageStore(to_tex, coords, value);
}
