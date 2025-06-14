use crate::audio_engine::GpuData;
use crate::scene::mesh::Triangle;
use crate::util::vec3::{MAX_VEC3, MIN_VEC3, ZERO};
use crate::util::{vec3, Axis};
use crevice::std430::{AsStd430, Vec3};

const MAX_BVH_DEPTH: usize = 64;
const BVH_SPLIT_ATTEMPTS: usize = 8;

#[derive(Clone)]
pub(crate) struct BvhBuffer(pub Vec<BvhNode>);
impl GpuData for BvhBuffer {
    unsafe fn serialize(&self, dst: *mut u8) {
        std::ptr::copy_nonoverlapping(
            (&self.0[..] as *const [BvhNode]).cast(),
            dst,
            self.size()
        );
    }
    
    fn size(&self) -> usize { 
        size_of::<BvhNode>() * self.0.len()
    }
}


#[derive(Clone, AsStd430)]
pub(crate) struct BvhNode {
    min: Vec3,
    idx: u32,
    max: Vec3,
    amt: u32,
}

impl Default for BvhNode {
    fn default() -> Self {
        Self {
            min: ZERO,
            idx: u32::default(),
            max: ZERO,
            amt: u32::default()
        }
    }
}

impl BvhNode {
    fn init(&mut self, triangles: &Vec<Triangle>, indices: &[u32], offset: u32) {
        // Expand the node's bounds
        for &idx in indices {
            self.expand(&triangles[idx as usize])
        }

        // Populate the various fields
        self.idx = offset;                  // offset into the buffer
        self.amt = indices.len() as u32;    // amt size => leaf
    }

    fn expand(&mut self, triangle: &Triangle) {
        self.min = vec3::min(self.min, triangle.min_bound());
        self.max = vec3::max(self.max, triangle.max_bound());
    }

    fn area(&self) -> f32 {
        if vec3::eq(self.max, MIN_VEC3) || vec3::eq(self.min, MAX_VEC3) {
            return 0.0;
        }

        let len = vec3::sub(self.max, self.min);
        2.0 * (len.x * len.y + len.z * len.y + len.x * len.z)
    }
}

 
pub(crate) struct BvhBuilder<'a> {
    bvh_list: Vec<BvhNode>,
    triangles: &'a mut Vec<Triangle>,
}

impl<'a> BvhBuilder<'a> {
    pub(crate) fn new(triangles: &'a mut Vec<Triangle>) -> Self {
        Self {
            bvh_list: Vec::new(),
            triangles
        }
    }

    /// Builder method to finally the BVH tree.
    /// The nodes are stored linearly in a vector which is returned, and the triangles
    /// are reordered using an index array during the construction of the BVH to avoid
    /// excessive memory writes.
    /// Motion blur is applied as an additional step.
    pub(crate) fn build(mut self) -> Vec<BvhNode> {
        // Create triangle index list with linear sequence (0,1,2...)
        let mut indices: Vec<_> = (0..self.triangles.len() as u32).collect();

        // Push the initial node
        self.bvh_list.push(BvhNode::default());

        // Build the BVH
        self.recursive_build(0, &mut indices[..], 0, 0, 1e30);

        // Apply the resulting index ordering to the triangles
        Self::apply_ordering(&mut self.triangles, &indices);

        self.bvh_list
    }

    /// Builds the BVH by recursively performing the following steps:
    /// - Initialize the node as leaf
    /// - Find the best split location and split the indices
    /// - If no improvements are yielded over the parent node return. This node is now a leaf.
    /// - Else, change the node to be a parent node, push the two children, and recurse on them.
    fn recursive_build(&mut self, node_idx: usize, indices: &mut [u32], depth: usize, offset: usize, parent_cost: f32) {
        // Initialize node with all children
        self.bvh_list[node_idx].init(self.triangles, indices, offset as u32);

        if depth >= MAX_BVH_DEPTH || indices.len() <= 1 {
            return;
        }

        // Find the best split location, return if it gets more expensive
        let (split_axis, split_pos) = self.find_best_split(node_idx, indices);
        let cost = self.split_cost(indices, split_axis, split_pos);
        if cost >= parent_cost {
            return;
        }

        // Perform the split
        let mut left_idx = 0;
        let mut right_idx = indices.len() - 1;
        while left_idx <= right_idx {
            let tri = self.triangles[indices[left_idx] as usize];
            let center = (tri.axis_min(split_axis) + tri.axis_max(split_axis)) / 2.0;

            if (center < split_pos) {
                left_idx += 1;
            } else {
                // This ends up ordering the index array
                indices.swap(left_idx, right_idx);
                right_idx -= 1;
            }
        }

        // Avoid creating empty nodes
        if ((left_idx == 0) || (right_idx == indices.len() - 1)) {
            return;
        }

        // Add the children indices and recurse down
        let left_node_idx = self.bvh_list.len();
        let right_node_idx = left_node_idx + 1;

        self.bvh_list[node_idx].idx = left_node_idx as u32; // child index
        self.bvh_list[node_idx].amt = 0;           // not a leaf

        self.bvh_list.push(BvhNode::default());
        self.bvh_list.push(BvhNode::default());

        let (left, right) = indices.split_at_mut(left_idx);

        self.recursive_build(left_node_idx,  left,  depth + 1,       offset,            cost);
        self.recursive_build(right_node_idx, right, depth + 1, offset + left_idx, cost);
    }

    /// Finds the best location and axis to perform a split by attempting to split the
    /// volume SPLIT_ATTEMPTS times, and minimising a cost function.
    fn find_best_split(&self, node_idx: usize, indices: &[u32]) -> (Axis, f32) {
        // Get the node's AABB information
        let node = self.bvh_list[node_idx].clone();
        let start_pos = node.min;
        let dimensions = vec3::div_scalar(vec3::sub(node.max, node.min), (BVH_SPLIT_ATTEMPTS + 1) as f32);

        // Use the SAH to find the best split position by trying to split at SPLIT_ATTEMPT uniform intervals
        let mut best_axis = None;
        let mut best_pos = 0.0;
        let mut best_cost = 1e30;
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            // Try SPLIT_ATTEMPTS splits and pick the best
            for attempt in 0..BVH_SPLIT_ATTEMPTS {
                let pos = vec3::axis(start_pos, axis) + vec3::axis(dimensions, axis) * attempt as f32;
                let cost = self.split_cost(indices, axis, pos);

                if (cost < best_cost) {
                    best_axis = Some(axis);
                    best_pos = pos;
                    best_cost = cost;
                }
            }
        }

        if let Some(axis) = best_axis {
            (axis, best_pos)
        } else {
            panic!("No good split was found! This should never happen.");
        }
    }

    /// The Surface Area Heuristic estimates the "cost" of a split, to be minimized.
    /// 
    /// The reasoning behind it is that the factor that a BVH wants to minimize is 
    /// the amount of intersection checks, and a larger area is more likely to be 
    /// intersected. 
    /// This means that a very small area with very few triangles has a low cost,
    /// but a large area with a lot of triangles might end up costing a lot if hit,
    /// and since it's very likely to be hit its cost will be high:
    /// 
    /// Cost = TriangleAmountLeft * AreaLeft + TriangleAmountRight * AreaRight
    fn split_cost(&self, indices: &[u32], axis: Axis, location: f32) -> f32 {
        let mut left_amount: f32 = 0.0;
        let mut right_amount: f32 = 0.0;
        let mut node_left = BvhNode::default();
        let mut node_right = BvhNode::default();

        for &idx in indices {
            let tri = &self.triangles[idx as usize];
            let center = (tri.axis_min(axis) + tri.axis_max(axis)) / 2.0;

            if (center < location) {
                node_left.expand(tri);
                left_amount += 1.0;
            } else {
                node_right.expand(tri);
                right_amount += 1.0;
            }
        }

        left_amount * node_left.area() + right_amount * node_right.area()
    }

    fn apply_ordering(items: &mut Vec<Triangle>, ordering: &Vec<u32>) {
        let size = items.len();

        let mut sorted = Vec::with_capacity(size);
        for idx in 0..size {
            sorted.push(items[ordering[idx] as usize]);
        }

        *items = sorted;
    }
}
