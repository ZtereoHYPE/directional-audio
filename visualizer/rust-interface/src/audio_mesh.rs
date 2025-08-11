use crate::to_vec;
use audio_processor::scene::mesh::Triangle;
use godot::classes::mesh::ArrayType;
use godot::classes::{IStaticBody3D, Mesh, MeshInstance3D, StaticBody3D};
use godot::obj::{Base, WithBaseField};
use godot::prelude::*;

pub const AUDIO_MESH_GROUP: &str = "AudioMeshes";
pub const MESH_ABSORPTION_COEFFICIENT: f32 = 0.1; // todo: make this configurable from godot!

#[derive(GodotClass)]
#[class(init, base=StaticBody3D)]
pub struct AudioMeshNode {
    base: Base<StaticBody3D>
}

#[godot_api]
impl IStaticBody3D for AudioMeshNode {
    fn enter_tree(&mut self) {
        self.base_mut().add_to_group(AUDIO_MESH_GROUP);
    }
}

impl AudioMeshNode {
    pub fn get_mesh_triangles(&self) -> Vec<Triangle> {
        let mut triangles = vec![];
        Self::get_mesh_triangles_rec(self.to_gd().upcast(), &mut triangles);
        triangles
    }

    // Traverse all MeshInstance3D children and get their meshes
    fn get_mesh_triangles_rec(node: Gd<Node>, triangles: &mut Vec<Triangle>) {
        if node.is_class("MeshInstance3D") {
            // push something to vector
            let mesh_node = node.clone().cast::<MeshInstance3D>();
            let mesh = mesh_node.get_mesh();

            if let Some(mesh) = mesh {
                let offset = mesh_node.get_global_position();
                triangles.append(&mut Self::extract_mesh_triangles(mesh, offset))
            }
        }

        for child in node.get_children().iter_shared() {
            Self::get_mesh_triangles_rec(child, triangles);
        }
    }

    fn extract_mesh_triangles(mesh: Gd<Mesh>, offset: Vector3) -> Vec<Triangle> {
        let mut triangles = vec![];
        for surf_idx in 0..mesh.get_surface_count() {
            let arrays = mesh.surface_get_arrays(surf_idx);
            let has_indices = !arrays.at(ArrayType::INDEX.ord() as usize).is_nil();

            if has_indices {
                let vertices: PackedVector3Array = arrays.at(ArrayType::VERTEX.ord() as usize).to();
                let indices: PackedInt32Array = arrays.at(ArrayType::INDEX.ord() as usize).to();

                let mut mesh_tris = (0..indices.len())
                    .map(|idx| indices.get(idx).unwrap() as usize)
                    .map(|indice| vertices.get(indice).unwrap())
                    .map(|vertex| to_vec(vertex + offset).into())
                    .collect::<Vec<_>>()
                    .chunks_exact(3)
                    .map(|vertices| Triangle::new(vertices.try_into().unwrap(), MESH_ABSORPTION_COEFFICIENT))
                    .collect::<Vec<_>>();

                triangles.append(&mut mesh_tris);

            } else {
                let vertices: PackedVector3Array = arrays.at(ArrayType::VERTEX.ord() as usize).to();

                let mut mesh_tris = (0..vertices.len())
                    .map(|idx| vertices.get(idx).unwrap())
                    .map(|vertex| to_vec(vertex + offset).into())
                    .collect::<Vec<_>>()
                    .chunks_exact(3)
                    .map(|vertices| Triangle::new(vertices.try_into().unwrap(), MESH_ABSORPTION_COEFFICIENT))
                    .collect::<Vec<_>>();

                triangles.append(&mut mesh_tris);
            }
        }

        triangles
    }
}
