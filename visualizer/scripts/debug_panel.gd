extends PanelContainer

func _ready() -> void:
	visible = false

func _on_debug_panel_checkbox_toggled(toggled_on: bool) -> void:
	visible = toggled_on
