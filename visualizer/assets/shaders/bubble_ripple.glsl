//#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// todo: look for a better format, we don't need all this precision
layout(binding = 0, rgba32f) uniform image2D previous_bubble_height;
layout(binding = 1, rgba32f) uniform image2D next_bubble_height;

void main() {
    uint instance_idx = gl_GlobalInvocationID.x;
    vec3 instance = instances[instance_idx];

    // todo: implement ripple effect to smooth out the instancing
}