shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

uniform samplerCube heightmap : filter_nearest;

vec3 get_face_normal(vec3 vertex) {
	vec3 dx = dFdx(vertex);
	vec3 dy = dFdy(vertex);

	return normalize(cross(dy, dx));
}

void vertex() {
	//VERTEX += texture(heightmap, NORMAL + vec3(0.0001, 0.0001, 0.0001)).x * NORMAL;
}

void fragment() {
	NORMAL = get_face_normal(VERTEX);
	vec3 world_normal = (INV_VIEW_MATRIX * vec4(NORMAL.xyz, 0.0)).xyz;

	//vec3 color = mix(vec3(0,0.5,1), vec3(1,0,0), bump.w*bump.w);
	ALBEDO = texture(heightmap, world_normal).xyz;
	METALLIC = 0.0;
	ROUGHNESS = 0.01;
	RIM = 0.2;
	//ALPHA = 0.7;
}

void light() {}
