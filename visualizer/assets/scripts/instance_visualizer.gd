extends MultiMeshInstance3D

@onready
var instance_mesh := $Instance.mesh as SphereMesh;

func _ready() -> void:
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
	global_position = Vector3(0,0,0);
	
	multimesh = MultiMesh.new();
	multimesh.transform_format = MultiMesh.TRANSFORM_3D;
	multimesh.mesh = instance_mesh;
	multimesh.instance_count = 1024 * 64;


func _on_audio_listener_node_visualization_data_received(data: GodotVisualizationData) -> void:
	build_instances(data.instances);


func build_instances(instances: PackedVector3Array) -> void:
	multimesh.visible_instance_count = instances.size();
	
	for idx in range(instances.size()):
		var offset := instances[idx];
		multimesh.set_instance_transform(idx, Transform3D().translated(offset))
