use audio_processor::DebugData;
use godot::builtin::{PackedVector3Array, PackedVector4Array, Vector3, Vector4};
use godot::prelude::GodotClass;

#[derive(GodotClass)]
#[class(init)]
pub struct GodotVisualizationData {
    #[var]
    rays: PackedVector4Array,
    #[var]
    instances: PackedVector3Array // for now they're just coordinates, might get upgraded later
}

// todo: there are a LOT of copies going on, find a way to avoid them as much as possible
impl GodotVisualizationData {
    pub fn set_data(&mut self, data: DebugData) {
        let ray_hits = data.rt.rays.rays.map(|ray| Vector4::new(ray.x, ray.y, ray.z, ray.w));
        let instance_locations: Vec<Vector3> = data.rt.instances.instances
            .iter()
            .map(|i| Vector3::new(i.direction.x, i.direction.y, i.direction.z))
            .collect();

        self.rays = PackedVector4Array::from(ray_hits);
        self.instances = PackedVector3Array::from(instance_locations);
    }
}
