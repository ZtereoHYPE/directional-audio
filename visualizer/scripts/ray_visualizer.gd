extends MeshInstance3D

const BOUNCE_AMOUNT: int = 4;

func _ready() -> void:
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
	global_position = Vector3.ZERO;


func _on_audio_listener_node_visualization_data_received(data: GodotVisualizationData) -> void:
	mesh = build_ray_mesh(data.ray_origin, data.rays);


func build_ray_mesh(origin: Vector3, rays: PackedVector4Array) -> ImmediateMesh:
	var new_mesh := ImmediateMesh.new();
	
	var material := StandardMaterial3D.new();
	material.albedo_color = Color.RED;
	
	var index := 0;
	var total_segments := 0;
	while (index < rays.size() / BOUNCE_AMOUNT):
		var ray_index: int = index * 4;
		
		# If the ray gets nowhere, skip this one
		if rays[ray_index].w < 0.5:
			index += 1;
			continue;
		
		# Start a new line
		new_mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP, material);
		new_mesh.surface_add_vertex(origin);
		
		# For every bounce, if it exists then extend the line
		for bounce_index: int in range(BOUNCE_AMOUNT):
			var pos := rays[ray_index + bounce_index];
			if pos.w < 0.5:
				break; # if the bounce doesn't exist stop the iteration
			
			total_segments += 1;
			new_mesh.surface_add_vertex(Vector3(pos.x, pos.y, pos.z));
			
		index += 4;
		new_mesh.surface_end();
	
	return new_mesh;
