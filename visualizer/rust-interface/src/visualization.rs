use audio_processor::VisualizationData;
use godot::builtin::{PackedVector3Array, PackedVector4Array, Vector3, Vector4};
use godot::prelude::GodotClass;

#[derive(GodotClass)]
#[class(init)]
pub struct GodotVisualizationData {
    #[var]
    ray_origin: Vector3,
    #[var]
    rays: PackedVector4Array,
    #[var]
    instances: PackedVector4Array // for now they're just coordinates, might get upgraded later
}

// todo: there are a LOT of copies going on, find a way to avoid them as much as possible
impl GodotVisualizationData {
    pub fn set_data(&mut self, data: VisualizationData) {
        let ray_hits = data.rays.rays.map(|ray| Vector4::new(ray.x, ray.y, ray.z, ray.w));
        let instance_locations = data.instances
            .iter()
            .map(|i| Vector4::new(i.direction.x, i.direction.y, i.direction.z, 0.0))
            .collect::<Vec<Vector4>>();

        self.ray_origin = Vector3::new(data.last_rt_origin.x, data.last_rt_origin.y, data.last_rt_origin.z);
        self.rays = PackedVector4Array::from(ray_hits);
        self.instances = PackedVector4Array::from(instance_locations);
    }
}
