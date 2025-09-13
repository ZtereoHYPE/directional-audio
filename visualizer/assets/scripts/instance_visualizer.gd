extends MultiMeshInstance3D

@onready
var instance_mesh := ($Instance.mesh as SphereMesh);
var enabled := false;

func _ready() -> void:
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;
	global_position = Vector3(0,0,0);
	
	multimesh = MultiMesh.new();
	multimesh.transform_format = MultiMesh.TRANSFORM_3D;
	multimesh.mesh = instance_mesh;
	multimesh.instance_count = 1024 * 64;


func _process(delta: float) -> void:
	if !enabled:
		multimesh.visible_instance_count = 0;


func _on_audio_listener_node_visualization_data_received(data: GodotVisualizationData) -> void:
	multimesh.visible_instance_count = data.instance_amount;
	
	var offsets := data.get_instance_coordinates();
	for idx in range(offsets.size()):
		multimesh.set_instance_transform(idx, Transform3D().translated(offsets[idx]))
		#multimesh.set_instance_color()


func _on_instance_checkbox_toggled(toggled_on: bool) -> void:
	enabled = toggled_on;
