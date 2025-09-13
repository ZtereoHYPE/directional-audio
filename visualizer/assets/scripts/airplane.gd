extends AudioSourceNode

@onready
var path_follow := $".." as PathFollow3D

func _physics_process(delta: float) -> void:
	path_follow.progress += 10 * delta;
