#define PI 3.1415
const int FACE_RESOLUTION = 128;

ivec3 direction_to_cubemap(vec3 direction) {
    vec3 abs_dir = abs(direction);
    int face_idx;
    vec2 tex_coord;

    if (abs_dir.x > abs_dir.y && abs_dir.x > abs_dir.z) {
        face_idx = direction.x > 0.0 ? 0 : 1; // Positive X or Negative X
        int flip = direction.x > 0.0 ? -1 : 1;
        tex_coord = vec2(flip * direction.z / abs_dir.x, -direction.y / abs_dir.x);
    } else if (abs_dir.y > abs_dir.z) {
        face_idx = direction.y > 0.0 ? 2 : 3; // Positive Y or Negative Y
        int flip = direction.y > 0.0 ? 1 : -1;
        tex_coord = vec2(direction.x / abs_dir.y, flip * direction.z / abs_dir.y);
    } else {
        face_idx = direction.z > 0.0 ? 4 : 5; // Positive Z or Negative Z
        int flip = direction.z > 0.0 ? 1 : -1;
        tex_coord = vec2(flip * direction.x / abs_dir.z, -direction.y / abs_dir.z);
    }

    tex_coord = (tex_coord * 0.5 + 0.5) * FACE_RESOLUTION;

    return ivec3(ivec2(round(tex_coord)), face_idx);
}

vec3 cubemap_to_direction(ivec3 cubemap) {
    ivec2 texel = cubemap.xy;
    int face = cubemap.z;

    // convert texel coordinate to [-1, 1] range, center-aligned
    vec2 uv = (vec2(texel) + 0.5) / FACE_RESOLUTION;
    uv = uv * 2.0 - 1.0;

    vec3 direction;

    // map from face and uv to 3D direction
    if (face == 0) {         // +X
        direction = vec3(1.0, -uv.y, -uv.x);
    } else if (face == 1) {  // -X
        direction = vec3(-1.0, -uv.y, uv.x);
    } else if (face == 2) {  // +Y
        direction = vec3(uv.x, 1.0, uv.y);
    } else if (face == 3) {  // -Y
        direction = vec3(uv.x, -1.0, -uv.y);
    } else if (face == 4) {  // +Z
        direction = vec3(uv.x, -uv.y, 1.0);
    } else if (face == 5) {  // -Z
        direction = vec3(-uv.x, -uv.y, -1.0);
    }

    return normalize(direction);
}

const int wraps[] = {
    2, 3, 4, 5, // +X
    2, 3, 5, 4, // -X
    5, 4, 1, 0, // +Y
    4, 5, 0, 1, // -Y
    2, 3, 1, 0, // +Z
    2, 3, 0, 1  // -Z
};

// max offset is of 1 in any direction, should NOT be called at corners, they require special casing
ivec3 cubemap_offset_sample(ivec3 origin, ivec2 offset) {
    uint bottom = uint(origin.y == FACE_RESOLUTION - 1);
    uint top = uint(origin.y == 0);
    uint left = uint(origin.x == 0);
    uint right = uint(origin.x == FACE_RESOLUTION - 1);

//    return origin + ivec3(offset, 0);

    // Early exit if we're not near a border
    if (top + bottom + left + right == 0) {
        return origin + ivec3(offset, 0);
    }

    vec2 pos = vec2(right * 1 + left * -1, top * -1 + bottom * 1);

    // If we are crossing a border, translate to right face of the cubemap
    if (dot(pos, vec2(offset)) > 0.9) {
        uint to = bottom * 1 + right * 2 + left * 3;
        bool flip = ((origin.z == 2) || (origin.z == 3)) && (bool(left) || bool(right)); // when to flip the coordinates
        bool invert_x = ((origin.z == 2) && bool(top)) || ((origin.z == 3) && bool(bottom)); // when to invert x coordinate (X = RES - X)
        bool invert_y = invert_x || ((origin.z == 2) && bool(right)) || ((origin.z == 3) && bool(left)); // when to invert y coordinate (Y = RES - Y)

        ivec2 coords = (origin.xy + offset + ivec2(FACE_RESOLUTION)) % FACE_RESOLUTION; // wrap around
        coords = flip ? coords.yx : coords.xy;
        coords = ivec2(
            invert_x ? FACE_RESOLUTION - coords.x : coords.x,
            invert_y ? FACE_RESOLUTION - coords.y : coords.y
        );

        return ivec3(coords, wraps[origin.z * 4 + to]);
    } else {
        return origin + ivec3(offset, 0);
    }
}
