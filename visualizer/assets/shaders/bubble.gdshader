shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

uniform samplerCube heightmap : filter_linear;
//uniform samplerCube heightmap : filter_nearest;
uniform vec3 center;

varying vec3 sphere_direction;
varying vec2 height;

float shlick(float u) {
	float m = 1.0 - u;
	float m2 = m * m;
	return m2 * m2 * m;
}

vec2 GetGradient(vec2 intPos, float t) {
    // Uncomment for calculated rand
    float rand = fract(sin(dot(intPos, vec2(12.9898, 78.233))) * 43758.5453);

    // Rotate gradient: random starting rotation, random rotation rate
    float angle = 6.283185 * rand + 4.0 * t * rand;
    return vec2(cos(angle), sin(angle));
}

// Flow noise by LukeRissacher on Shadertoy
// https://www.shadertoy.com/view/MtcGRl
float Pseudo3dNoise(vec3 pos) {
    vec2 i = floor(pos.xy);
    vec2 f = pos.xy - i;
    vec2 blend = f * f * (3.0 - 2.0 * f);
    float noiseVal =
        mix(
            mix(
                dot(GetGradient(i + vec2(0, 0), pos.z), f - vec2(0, 0)),
                dot(GetGradient(i + vec2(1, 0), pos.z), f - vec2(1, 0)),
                blend.x),
            mix(
                dot(GetGradient(i + vec2(0, 1), pos.z), f - vec2(0, 1)),
                dot(GetGradient(i + vec2(1, 1), pos.z), f - vec2(1, 1)),
                blend.x),
        blend.y
    );
    return noiseVal / 0.7; // normalize to about [-1..1]
}

vec3 rotate_y(vec3 vector, float angle) {
	mat3 rotation = mat3(
		vec3(cos(angle),  0, sin(angle)),
		vec3(0,           1, 0),
		vec3(-sin(angle), 0, cos(angle))
	);

	return rotation * vector;
}

float fractal_noise(vec3 pos) {
	pos = vec3(0, sin(TIME / 5.0) , 0) + rotate_y(pos, TIME / 20.0) * 2.0;
	float octave_0 = Pseudo3dNoise(pos);
	float octave_1 = Pseudo3dNoise(pos * 5.0) * 0.1 * (octave_0 * 0.5 + 0.25);
	return octave_0 + octave_1;
}

vec2 vertex_height(vec3 sphere_dir) {
	float volume_height = texture(heightmap, sphere_dir).x * 0.5;
	float bubbleness = (fractal_noise(sphere_dir) * 0.5 + 0.5) * 0.1;

	return vec2(bubbleness + volume_height, volume_height); // both are needed
}

vec3 get_normal(vec3 dir, float epsilon) {
	// Sample along tangent and bitangent
	// todo: gets imprecise around Y axis
	vec3 tangent = cross(dir, vec3(0, 1, 0));
	vec3 bitangent = cross(dir, tangent) * epsilon;
	tangent *= epsilon;

	// note: this is slightly imprecise wrt epsilon: SIN(THETA) ≈ THETA for small THETA
	vec3 s01 = vertex_height(normalize(dir - tangent)).x * normalize(dir - tangent);
	vec3 s21 = vertex_height(normalize(dir + tangent)).x * normalize(dir + tangent);
	vec3 s10 = vertex_height(normalize(dir - bitangent)).x * normalize(dir - bitangent);
	vec3 s12 = vertex_height(normalize(dir + bitangent)).x * normalize(dir + bitangent);

	// Get derivatives from those positions
	epsilon = sin(epsilon) * 2.0;
	vec3 dx = (s21 - s01) / epsilon;
	vec3 dy = (s12 - s10) / epsilon;

	// Get normal from those derivatives
	return normalize(-cross(dy, dx) + dir);
}

void vertex() {
	sphere_direction = normalize((MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz - center);
	height = vertex_height(sphere_direction);

	VERTEX += height.x * sphere_direction;
	NORMAL = get_normal(sphere_direction, 0.001);
}

void fragment() {
	vec3 color = mix(vec3(0.8, 0.8, 1), vec3(1,0,0), height.y * 2.0);
	ALBEDO = mix(vec3(1.0) - NORMAL * 0.3, color, height.y * 2.0);
	//ALBEDO = NORMAL;
	METALLIC = 0.0;
	ROUGHNESS = 0.00;
	RIM = 0.2;

	float rim = abs(dot(VIEW, NORMAL));
	float fresnel = clamp(shlick(rim), 0.0, 1.0);
	ALPHA = mix(fresnel, 1.0, 0.03);
	ALPHA += min(0.5 * sqrt(ALPHA) + height.y * 2.0, 0.9);
}

//void light() {}
