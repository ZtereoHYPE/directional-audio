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

    // map from face && uv to 3D direction
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

ivec3 cubemap_offset_sample(ivec3 origin, ivec2 offset) {
    uint face = origin.z;
    float x = origin.x;
    float y = origin.y;

    uint F = FACE_RESOLUTION - 1;
    bool bottom = (y == F) && (offset.y == 1);
    bool top = (y == 0) && (offset.y == -1);
    bool left = (x == 0) && (offset.x == -1);
    bool right = (x == F) && (offset.x == 1);

    // If we're not crossing a border, early exit
    if (!(top || bottom || left || right))
        return origin + ivec3(offset, 0);

    if (face == 0) {
        if      (top)       return ivec3(F, F-x, 2);
        else if (bottom)    return ivec3(F, x, 3);
        else if (left)      return ivec3(F, y, 4);
        else                return ivec3(0, y, 5);
    } else if (face == 1) {
        if      (top)       return ivec3(0, x, 2);
        else if (bottom)    return ivec3(0, F-x, 3);
        else if (left)      return ivec3(F, y, 5);
        else                return ivec3(0, y, 4);
    } else if (face == 2) {
        if      (top)       return ivec3(F-x, 0, 5);
        else if (bottom)    return ivec3(x, 0, 4);
        else if (left)      return ivec3(y, 0, 1);
        else                return ivec3(F-y, 0, 0);
    } else if (face == 3) {
        if      (top)       return ivec3(x, F, 4);
        else if (bottom)    return ivec3(F-x, F, 5);
        else if (left)      return ivec3(F-y, F, 1);
        else                return ivec3(y, F, 0);
    } else if (face == 4) {
        if      (top)       return ivec3(x, F, 2);
        else if (bottom)    return ivec3(x, 0, 3);
        else if (left)      return ivec3(F, y, 1);
        else                return ivec3(0, y, 0);
    } else if (face == 5) {
        if      (top)       return ivec3(F-x, 0, 2);
        else if (bottom)    return ivec3(F-x, F, 3);
        else if (left)      return ivec3(F, y, 0);
        else                return ivec3(0, y, 1);
    }
}
