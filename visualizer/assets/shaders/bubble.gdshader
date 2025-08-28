shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

//uniform samplerCube heightmap : filter_nearest;
uniform samplerCube heightmap : filter_nearest;
uniform vec3 center;

varying vec3 world_normal;

void vertex() {
	VERTEX += texture(heightmap, NORMAL + vec3(0.0001, 0.0001, 0.0001)).x * NORMAL * 0.5;
	NORMAL = normalize((MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz - center);
	world_normal = NORMAL;
}

void fragment() {
	//NORMAL = get_face_normal(VERTEX);
	//NORMAL = sphere_normal;

	//vec3 color = mix(vec3(0,0.5,1), vec3(1,0,0), bump.w*bump.w);
	ALBEDO = texture(heightmap, world_normal).xyz;
	METALLIC = 0.0;
	ROUGHNESS = 0.01;
	RIM = 0.2;
	//ALPHA = 0.7;
}

void light() {}
