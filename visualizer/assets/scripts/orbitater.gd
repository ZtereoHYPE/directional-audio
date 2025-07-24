extends Node3D


func _on_person_selector_item_selected(index: int) -> void:
	if index == 1:
		$OrbitCamera.make_current()
