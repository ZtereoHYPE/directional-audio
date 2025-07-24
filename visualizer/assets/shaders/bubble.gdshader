shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

uniform sampler2D heightmap;
const float NORMAL_SAMPLING = 0.025;


// Caluclates and applies TBN matrix
// todo: understand https://computergraphics.stackexchange.com/questions/5498/compute-sphere-tangent-for-normal-mapping
vec3 normal_mapping(vec3 vector, vec3 normal) {
	vec3 up = vec3(0,1,0);
	vec3 T = cross(up, normal);
	vec3 B = cross(normal, T);

	mat3 TBN = mat3(T, B, normal);
	return TBN * vector;
}


vec4 get_bump_map(vec2 uv, vec3 old_normal) {
	// sample the texture to calculate the tangent-space normal
	const vec2 offset = vec2(NORMAL_SAMPLING, 0);
    float s01 = texture(heightmap, uv - offset.xy).x;
    float s21 = texture(heightmap, uv + offset.xy).x;
    float s10 = texture(heightmap, uv - offset.yx).x;
    float s12 = texture(heightmap, uv + offset.yx).x;
    vec3 dxdz = vec3(NORMAL_SAMPLING*2.0, 0, s21-s01);
    vec3 dydz = vec3(0, NORMAL_SAMPLING*2.0, s10-s12);

	vec3 new_normal = normalize(cross(dxdz, dydz));

	// rotate the vector by the sphere's normal
	new_normal = normal_mapping(new_normal, old_normal);

	//return old_normal;
	return vec4(new_normal, texture(heightmap, uv).x);
}

void vertex() {
	VERTEX *= texture(heightmap, UV).x;
}

void fragment() {
	vec4 bump = get_bump_map(UV, NORMAL);

	vec3 color = mix(vec3(0,0.5,1), vec3(1,0,0), bump.w*bump.w);
	NORMAL = bump.xyz;
	// Called for every pixel the material is visible on.
	METALLIC = 0.0;
	ROUGHNESS = 0.01;
	ALBEDO = color;
	RIM = 0.2;
	//ALPHA = 0.7;
}

void light() {}
