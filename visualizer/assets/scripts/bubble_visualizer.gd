extends MeshInstance3D

# update the uniforms directly when receiving new data (?)
# else save here the 2 data needed

const INSTANCE_AMOUNT := 64;
const BUFFER_SIZE := 8 + 16 * INSTANCE_AMOUNT;

const TEXTURE_WIDTH := 64;
const TEXTURE_HEIGHT := 32;

var rd: RenderingDevice;
var buffer: RID;
var main_texture: RID;
var compute_texture: RID;
var descriptor_set: RID;
var pipeline: RID;

func _ready() -> void:
	rd = RenderingServer.create_local_rendering_device();
	var main_rd := RenderingServer.get_rendering_device();
	
	# Load GLSL shader
	var shader_file := load("res://assets/shaders/bubble_instance.glsl") as RDShaderFile;
	var shader_spirv := shader_file.get_spirv();
	var shader := rd.shader_create_from_spirv(shader_spirv);
	
	# Create descriptors
	buffer = rd.storage_buffer_create(BUFFER_SIZE);
	
	var fmt := RDTextureFormat.new()
	fmt.width = TEXTURE_WIDTH
	fmt.height = TEXTURE_HEIGHT
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT | RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	
	main_texture = main_rd.texture_create(fmt, RDTextureView.new()) # make the texture on the main device
	compute_texture = rd.texture_create_from_extension( # convert to a texture in the compute device
		RenderingDevice.TEXTURE_TYPE_2D,
		fmt.format,
		fmt.samples,
		fmt.usage_bits,
		main_rd.get_driver_resource(RenderingDevice.DRIVER_RESOURCE_TEXTURE, main_texture, 0),
		fmt.width,
		fmt.height,
		fmt.depth,
		fmt.array_layers
	)
	
	# create descriptor set
	var buffer_uniform := RDUniform.new();
	buffer_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER;
	buffer_uniform.binding = 0;
	buffer_uniform.add_id(buffer);
	
	var texture_uniform := RDUniform.new()
	texture_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE;
	texture_uniform.binding = 1;
	texture_uniform.add_id(compute_texture)
	
	descriptor_set = rd.uniform_set_create([buffer_uniform, texture_uniform], shader, 0);
	
	# Create a compute pipeline
	pipeline = rd.compute_pipeline_create(shader);
	
	# set the current shader's texture parameter to the right texture!
	var drawing_texture := Texture2DRD.new();
	drawing_texture.texture_rd_rid = main_texture;
	
	var material := get_active_material(0) as ShaderMaterial;
	material.set_shader_parameter("heightmap", drawing_texture);
	(($TextureViewer as MeshInstance3D).get_active_material(0) as StandardMaterial3D).albedo_texture = drawing_texture;


func _exit_tree() -> void:
	rd.free_rid(buffer);
	rd.free_rid(compute_texture);
	rd.free_rid(descriptor_set);
	rd.free_rid(pipeline);


func update_texture(instances: PackedVector3Array, volumes: Vector2) -> void:
	rd.buffer_update(buffer, 0, 8, PackedVector2Array([volumes]).to_byte_array());
	rd.buffer_update(buffer, 8, BUFFER_SIZE - 8, instances.to_byte_array());
	
	rd.texture_clear(compute_texture, Color(0.5, 0.5, 0.5), 0, 1, 0, 1); # temporary, the ripple will handle clearing (?)
	
	var compute_list := rd.compute_list_begin();
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline);
	rd.compute_list_bind_uniform_set(compute_list, descriptor_set, 0);
	@warning_ignore("integer_division")
	rd.compute_list_dispatch(compute_list, INSTANCE_AMOUNT / 64, 1, 1);
	rd.compute_list_end();
	
	rd.submit();
	rd.sync();


func _on_audio_listener_node_visualization_data_received(data: GodotVisualizationData) -> void:
	update_texture.call_deferred(data.instances, Vector2.ONE);
	pass;
