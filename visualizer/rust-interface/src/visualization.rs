use audio_processor::{Loudness, VisualizationData};
use godot::builtin::{PackedByteArray, PackedVector3Array, PackedVector4Array, Vector3, Vector4};
use godot::obj::Gd;
use godot::prelude::{godot_api, GodotClass};
use crate::audio_listener::AudioListenerNode;

#[repr(align(16))]
struct GodotInstance {
    position: Vector3,
    cluster_size: u32,
    loudness: f32,
    prev_loudness: f32,
}

#[derive(GodotClass)]
#[class(init)]
pub struct GodotVisualizationData {
    #[var]
    ray_origin: Vector3,
    #[var]
    rays: PackedVector4Array,
    #[var]
    instance_amount: u32,

    instances: Vec<(Vector3, u32)>,
}

// todo: there are a LOT of copies going on, find a way to avoid them as much as possible
impl GodotVisualizationData {
    pub fn set_data(&mut self, data: VisualizationData) {
        let ray_hits = data.rays.rays.map(|ray| Vector4::new(ray.x, ray.y, ray.z, ray.w));

        self.ray_origin = Vector3::new(data.last_rt_origin.x, data.last_rt_origin.y, data.last_rt_origin.z);
        self.rays = PackedVector4Array::from(ray_hits);
        self.instance_amount = data.instances.len() as u32;
        self.instances = data.instances
            .into_iter()
            .map(|i| (Vector3::new(i.direction.x, i.direction.y, i.direction.z), i.cluster_size))
            .collect()
    }

}

#[godot_api]
impl GodotVisualizationData {
    #[func]
    pub fn get_instance_buffer(&self, listener: Gd<AudioListenerNode>) -> PackedByteArray { unsafe {
        let loudness = listener.bind().loudness_history
            .iter()
            .take(5)
            .fold(Loudness::empty(), |l, r| l + *r);

        let prev_loudness = listener.bind().loudness_history
            .iter()
            .skip(3)
            .take(5)
            .fold(Loudness::empty(), |l, r| l + *r);

        // println!("loudness of 0: {}", loudness.0[0]);

        self.instances
            .iter()
            .enumerate()
            .map(|(idx, (pos, cluster))| GodotInstance {
                position: *pos,
                cluster_size: *cluster,
                loudness: loudness.0[idx] / 5.0,
                prev_loudness: prev_loudness.0[idx] / 5.0,
            })
            .collect::<Vec<_>>()
            .as_slice()
            .align_to::<u8>().1
            .into()
    }}

    #[func]
    pub fn get_instance_coordinates(&self) -> PackedVector3Array {
        self.instances
            .iter()
            .map(|(pos, _)| *pos)
            .collect()
    }
}
