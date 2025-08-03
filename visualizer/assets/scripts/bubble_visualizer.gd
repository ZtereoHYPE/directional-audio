extends MeshInstance3D

const MAX_INSTANCE_AMOUNT := 64;
const BUFFER_SIZE := 16 + 16 * MAX_INSTANCE_AMOUNT;

const FACE_RESOLUTION := 64;

@export
var subdivision_amt := 64;

var instance_amount := 0;

# RENDERING RESOURCES:
var rd: RenderingDevice;

# textures + buffers
var heightmap_tex_main_handle: RID;
var heightmap_tex: RID;
var ripple_tex: Array[RID];
var instance_buffer: RID;
var scene_buffer: RID;

# descriptors + pipelines
var instance_descriptor_set: RID;
var ripple_descriptor_set: RID;
var blur_descriptor_set: RID;
var instance_pipeline: RID;
var ripple_pipeline: RID;
var h_blur_pipeline: RID;
var v_blur_pipeline: RID;

var coord_faces: Array[Callable] = [
	func top(x: float, y: float) -> Vector3:
		return Vector3(x, 1, -y),
	
	func bottom(x: float, y: float) -> Vector3:
		return Vector3(x, -1, y),
	
	func pos_z(x: float, y: float) -> Vector3:
		return Vector3(x, y, 1),
	
	func neg_z(x: float, y: float) -> Vector3:
		return Vector3(-x, y, -1),
	
	func pos_x(x: float, y: float) -> Vector3:
		return Vector3(1, y, -x),
	
	func neg_x(x: float, y: float) -> Vector3:
		return Vector3(-1, y, x),
]

func map_to_sphere(v: Vector3) -> Vector3:
	var x2 := v.x * v.x;
	var y2 := v.y * v.y;
	var z2 := v.z * v.z;
	
	return Vector3(
		v.x * sqrt(1.0 - y2/2.0 - z2/2.0 + y2*z2/3.0),
		v.y * sqrt(1.0 - z2/2.0 - x2/2.0 + z2*x2/3.0),
		v.z * sqrt(1.0 - x2/2.0 - y2/2.0 + x2*y2/3.0),
	)

func create_box_sphere() -> ArrayMesh:
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var indices := PackedInt32Array()
	
	var subdivisions: Array[float] = []
	for i in range(subdivision_amt + 1):
		subdivisions.append(-1 + (float(i) / float(subdivision_amt)) * 2)
	
	# Generate the vertices
	for face_idx in range(6):
		for y in subdivisions:
			for x in subdivisions:
				var coords: Vector3 = coord_faces[face_idx].call(x, y);
				vertices.append(map_to_sphere(coords))
				normals.append(coords.normalized())
	
	# Generate the indices
	var row := subdivision_amt + 1
	for face_idx in range(6):
		var base := face_idx * row * row
		for y in range(row - 1):
			for x in range(row - 1):
				var offset := y * row + x
				indices.append(base + offset + row)
				indices.append(base + offset + 1)
				indices.append(base + offset)
				indices.append(base + offset + row)
				indices.append(base + offset + row + 1)
				indices.append(base + offset + 1)
	
	# Initialize the arrays.
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_INDEX] = indices
	
	# Create the Mesh.
	var arr_mesh := ArrayMesh.new()
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	
	return arr_mesh;


func create_textures() -> void:
	var main_rd := RenderingServer.get_rendering_device();
	var fmt := RDTextureFormat.new();
	fmt.width = FACE_RESOLUTION;
	fmt.height = FACE_RESOLUTION;
	fmt.texture_type = RenderingDevice.TEXTURE_TYPE_CUBE;
	fmt.format = RenderingDevice.DATA_FORMAT_R32_SFLOAT;
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT | RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	fmt.array_layers = 6;
	
	# Create a heightmap cubemap
	heightmap_tex_main_handle = main_rd.texture_create(fmt, RDTextureView.new()) # make the texture on the main device
	main_rd.texture_clear(heightmap_tex_main_handle, Color(0.0,0.0,0.0), 0, 1, 0, 6);
	
	# Get a handle for the heightmap for the different rendering device
	heightmap_tex = rd.texture_create_from_extension( 
		fmt.texture_type,
		fmt.format,
		fmt.samples,
		fmt.usage_bits,
		main_rd.get_driver_resource(RenderingDevice.DRIVER_RESOURCE_TEXTURE, heightmap_tex_main_handle, 0),
		fmt.width,
		fmt.height,
		fmt.depth,
		fmt.array_layers
	)
	
	# Create ripple texture
	ripple_tex = [
		rd.texture_create(fmt, RDTextureView.new()),
		rd.texture_create(fmt, RDTextureView.new()),
		rd.texture_create(fmt, RDTextureView.new())
	]
	
	rd.texture_clear(ripple_tex[0], Color(0.0,0.0,0.0), 0, 1, 0, 6);
	rd.texture_clear(ripple_tex[1], Color(0.0,0.0,0.0), 0, 1, 0, 6);
	rd.texture_clear(ripple_tex[2], Color(0.0,0.0,0.0), 0, 1, 0, 6);



func create_instance_resources() -> void:
	# Load GLSL shader
	var shader_file := load("res://assets/shaders/bubble_instance.glsl") as RDShaderFile;
	var shader_spirv := shader_file.get_spirv();
	var shader := rd.shader_create_from_spirv(shader_spirv);
	
	# Create buffers
	instance_buffer = rd.storage_buffer_create(BUFFER_SIZE);
	scene_buffer = rd.uniform_buffer_create(16);
	
	# create descriptor set
	var scene_uniform := RDUniform.new() # todo: this can use push constants
	scene_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER;
	scene_uniform.binding = 0;
	scene_uniform.add_id(scene_buffer);
	
	var buffer_uniform := RDUniform.new();
	buffer_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	buffer_uniform.binding = 1;
	buffer_uniform.add_id(instance_buffer);
	
	var heightmap_uniform := RDUniform.new()
	heightmap_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	heightmap_uniform.binding = 2;
	heightmap_uniform.add_id(heightmap_tex)
	
	var ripple_uniform := RDUniform.new()
	ripple_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	ripple_uniform.binding = 3;
	ripple_uniform.add_id(ripple_tex[1])
	
	instance_descriptor_set = rd.uniform_set_create([scene_uniform, buffer_uniform, heightmap_uniform, ripple_uniform], shader, 0);
	
	# Create a compute pipeline
	instance_pipeline = rd.compute_pipeline_create(shader);


func create_ripple_resources() -> void:
	# Load GLSL shader
	var shader_file := load("res://assets/shaders/bubble_ripple.glsl") as RDShaderFile;
	var shader_spirv := shader_file.get_spirv();
	var shader := rd.shader_create_from_spirv(shader_spirv);
	
	# create descriptor set
	var ripple_tex_prev_uniform := RDUniform.new();
	ripple_tex_prev_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	ripple_tex_prev_uniform.binding = 0;
	ripple_tex_prev_uniform.add_id(ripple_tex[0]);
	
	var ripple_tex_curr_uniform := RDUniform.new();
	ripple_tex_curr_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	ripple_tex_curr_uniform.binding = 1;
	ripple_tex_curr_uniform.add_id(ripple_tex[1]);
	
	var ripple_tex_next_uniform := RDUniform.new(); 
	ripple_tex_next_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	ripple_tex_next_uniform.binding = 2;
	ripple_tex_next_uniform.add_id(ripple_tex[2]);
	
	ripple_descriptor_set = rd.uniform_set_create([ripple_tex_prev_uniform, ripple_tex_curr_uniform, ripple_tex_next_uniform], shader, 0);
	
	# Create a compute pipeline
	ripple_pipeline = rd.compute_pipeline_create(shader);


func create_blur_resources() -> void:
	# Load GLSL shaders
	var shader_file := load("res://assets/shaders/bubble_add.glsl") as RDShaderFile;
	var shader_spirv := shader_file.get_spirv();
	var h_shader := rd.shader_create_from_spirv(shader_spirv);
	
	#shader_file = load("res://assets/shaders/bubble_blur_v.glsl") as RDShaderFile;
	#shader_spirv = shader_file.get_spirv();
	#var v_shader := rd.shader_create_from_spirv(shader_spirv);
	
	# create descriptor set
	var heightmap_uniform := RDUniform.new();
	heightmap_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	heightmap_uniform.binding = 0;
	heightmap_uniform.add_id(heightmap_tex);
	
	var ripple_tex_uniform := RDUniform.new();
	ripple_tex_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	ripple_tex_uniform.binding = 1;
	ripple_tex_uniform.add_id(ripple_tex[2]);
	
	blur_descriptor_set = rd.uniform_set_create([heightmap_uniform, ripple_tex_uniform], h_shader, 0);
	
	# Create a compute pipeline
	h_blur_pipeline = rd.compute_pipeline_create(h_shader);
	#v_blur_pipeline = rd.compute_pipeline_create(v_shader);


func _ready() -> void:
	mesh = create_box_sphere();
	
	# todo: this might get updated to push constants later
	rd = RenderingServer.create_local_rendering_device();

	create_textures();
	create_instance_resources();
	create_ripple_resources();
	create_blur_resources();

	# set the current shader's texture parameter to the right texture!
	var drawing_texture := TextureCubemapRD.new();
	drawing_texture.texture_rd_rid = heightmap_tex_main_handle;
	
	var material := ShaderMaterial.new()
	material.shader = load("res://assets/shaders/bubble.gdshader") as Shader
	material.set_shader_parameter("heightmap", drawing_texture) # todo: find a way to disable texture repeat!
	mesh.surface_set_material(0, material)


func _process(_delta: float) -> void:
	# Lock the rotation to 0 to prevent the head rotation from messing up the locations
	global_rotation = Vector3(0, 0, 0);
	global_position = Vector3(5, 1.5, 1);
	
	# Update scene uniform
	rd.buffer_update(scene_buffer, 0, 12, PackedVector3Array([($".." as Node3D).global_position]).to_byte_array());
	
	# Clear heightmap to prepare for new data
	rd.texture_clear(heightmap_tex, Color(0.0,0.0,0.0), 0, 1, 0, 6);
	
	# Shift the ripple textures
	for face in range(6):
		rd.texture_copy(ripple_tex[1], ripple_tex[0], Vector3.ZERO, Vector3.ZERO, Vector3(FACE_RESOLUTION, FACE_RESOLUTION, 1), 0, 0, face, face);
		rd.texture_copy(ripple_tex[2], ripple_tex[1], Vector3.ZERO, Vector3.ZERO, Vector3(FACE_RESOLUTION, FACE_RESOLUTION, 1), 0, 0, face, face);
	
	# Dispatch compute shaders
	var compute_list := rd.compute_list_begin();
	if instance_amount != 0:
		rd.compute_list_bind_compute_pipeline(compute_list, instance_pipeline);
		rd.compute_list_bind_uniform_set(compute_list, instance_descriptor_set, 0);
		@warning_ignore("integer_division")
		rd.compute_list_dispatch(compute_list, (instance_amount + 63) / 64, 1, 1);
	
	rd.compute_list_bind_compute_pipeline(compute_list, ripple_pipeline);
	rd.compute_list_bind_uniform_set(compute_list, ripple_descriptor_set, 0);
	rd.compute_list_dispatch(compute_list, FACE_RESOLUTION / 8, FACE_RESOLUTION / 8, 6);
	
	rd.compute_list_bind_uniform_set(compute_list, blur_descriptor_set, 0);
	rd.compute_list_bind_compute_pipeline(compute_list, h_blur_pipeline);
	rd.compute_list_dispatch(compute_list, FACE_RESOLUTION / 8, FACE_RESOLUTION / 8, 6);
	#rd.compute_list_bind_compute_pipeline(compute_list, v_blur_pipeline);
	#rd.compute_list_dispatch(compute_list, TEXTURE_WIDTH / 16, TEXTURE_HEIGHT / 16, 1);
		
	rd.compute_list_end();
	rd.submit();
	rd.sync();


func _exit_tree() -> void:
	pass; # todo: cleanups
	#rd.free_rid(instance_buffer);
	#rd.free_rid(scene_buffer);
	#rd.free_rid(compute_texture);
	#rd.free_rid(instance_descriptor_set);
	#rd.free_rid(instance_pipeline);


func update_visualization_data(instances: PackedVector4Array, volumes: Vector2, prev_volumes: Vector2) -> void:
	instance_amount = instances.size();
	rd.buffer_update(scene_buffer, 12, 4, PackedInt32Array([instance_amount]).to_byte_array());
	rd.buffer_update(instance_buffer, 0, 16, PackedVector2Array([volumes, prev_volumes]).to_byte_array());
	var instance_bytes := instances.to_byte_array();
	rd.buffer_update(instance_buffer, 16, instance_bytes.size(), instance_bytes);


func _on_audio_listener_node_visualization_data_received(data: GodotVisualizationData) -> void:
	update_visualization_data(data.instances, Vector2.ONE, Vector2.ZERO);
