extends MeshInstance3D

const BOUNCE_AMOUNT: int = 4;

@export
var color_map: Dictionary[float, Color] = {
	1.0: Color(0.9, 0, 0),
	2.0: Color(0, 0.3, 0.9),
}

func _ready() -> void:
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
	global_position = Vector3.ZERO;


func _on_audio_listener_node_visualization_data_received(data: GodotVisualizationData) -> void:
	mesh = build_ray_mesh(data.ray_origin, data.rays);


func build_ray_mesh(origin: Vector3, rays: PackedVector4Array) -> ImmediateMesh:
	var new_mesh := ImmediateMesh.new();
	var material := StandardMaterial3D.new();
	material.vertex_color_use_as_albedo = true;
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_VERTEX;
	new_mesh.surface_begin(Mesh.PRIMITIVE_LINES, material);
	
	var index := 0;
	while (index < rays.size() / BOUNCE_AMOUNT):
		var vertices: Array[Vector4] = [];
		var ray_index: int = index * 4;
		
		# If the ray gets nowhere, skip this one
		if rays[ray_index].w < 0.5:
			index += 1;
			continue;
		
		# For every bounce, if it exists then extend the line
		var source_reached := false;
		for bounce_index in range(BOUNCE_AMOUNT):
			var pos := rays[ray_index + bounce_index];
			if pos.w < 0.5:
				break; # if the bounce doesn't exist stop the iteration
			elif pos.w == 2.0:
				source_reached = true;
			
			vertices.push_back(pos);
		
		if source_reached:
			var previous := Vector3(origin.x, origin.y, origin.z);
			for vtx in vertices:
				new_mesh.surface_set_color(color_map[vtx.w])
				new_mesh.surface_add_vertex(previous);
				previous = Vector3(vtx.x, vtx.y, vtx.z);
				new_mesh.surface_add_vertex(previous);
		 
		index += 1;
	
	new_mesh.surface_end();
	return new_mesh;
