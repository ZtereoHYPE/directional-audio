shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

uniform samplerCube heightmap : filter_linear;
uniform vec3 center;

varying vec3 sphere_direction;
varying vec2 height;

float shlick_glass(float u) {
    const float R0 = (-0.45 / 2.45) * (-0.45 / 2.45);
	float m = 1.0 - u;
	float m2 = m * m;
	return R0 + (1.0 - R0) * m2 * m2 * m;
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
	float octave_1 = Pseudo3dNoise(pos * 5.0) * 0.1 * (octave_0 * 0.5 + 0.5);
	return octave_0 + octave_1;
}

vec2 vertex_height(vec3 sphere_dir) {
	float height_value = texture(heightmap, sphere_dir).x * 0.5;
	float bubble_noise = (fractal_noise(sphere_dir) * 0.5 + 0.5) * 0.1;

	return vec2(bubble_noise + height_value, height_value); // both are needed
}

vec3 get_normal(vec3 dir, float epsilon) {
	// Sample along tangent and bitangent
	vec3 tangent = cross(dir, vec3(0, 1, 0));
	vec3 bitangent = cross(dir, tangent) * epsilon;
	tangent *= epsilon;

	vec3 s01 = vertex_height(normalize(dir - tangent)).x * normalize(dir - tangent);
	vec3 s21 = vertex_height(normalize(dir + tangent)).x * normalize(dir + tangent);
	vec3 s10 = vertex_height(normalize(dir - bitangent)).x * normalize(dir - bitangent);
	vec3 s12 = vertex_height(normalize(dir + bitangent)).x * normalize(dir + bitangent);

	// Get derivatives from those positions
	epsilon = sin(epsilon) * 2.0;
	vec3 dx = (s21 - s01) / epsilon;
	vec3 dy = (s12 - s10) / epsilon;

	// Get normal from those derivatives
	return normalize(cross(dx, dy) + dir);
}

void vertex() {
	sphere_direction = normalize(VERTEX);
	height = vertex_height(sphere_direction);

	VERTEX += height.x * sphere_direction;
	NORMAL = get_normal(sphere_direction, 0.001);
}

vec4 get_height_color(vec3 normal) {
	float factor = height.y * 2.0;

	vec3 colors[] = {vec3(0.8, 0.8, 1), vec3(0.6, 1.0, 0.6), vec3(1.0, 0.3, 0.1)};
	vec3 color = mix(colors[0], colors[1], smoothstep(0.0, 0.5, factor));
	color = mix(color, colors[2], smoothstep(0.5, 1.0, factor));

	color = mix(vec3(1.0) - normal * 0.3, color, factor * 2.0);
	return vec4(color, min(factor * 2.0, 1.0));
}

void fragment() {
	vec4 color = get_height_color(NORMAL);
	ALBEDO = color.xyz;
	METALLIC = 0.0;
	ROUGHNESS = 0.00;
	RIM = 0.2;

	float rim = abs(dot(VIEW, NORMAL));
	float fresnel = clamp(shlick_glass(rim), 0.0, 1.0);
	ALPHA = mix(fresnel, 1.0, 0.03);
	ALPHA = min(ALPHA + 0.3 * sqrt(ALPHA) + color.a, 0.95);
}

//void light() {}
