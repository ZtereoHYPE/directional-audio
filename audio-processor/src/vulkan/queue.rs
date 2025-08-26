use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use ash::{Device, Instance};
use ash::vk::{DeviceQueueCreateInfo, PhysicalDevice, Queue, QueueFamilyProperties, QueueFlags};
use once_cell::sync::OnceCell;

static QUEUE_FAMILIES: OnceCell<Vec<u32>> = OnceCell::new();

#[derive(Copy, Clone)]
pub(crate) struct VulkanQueue {
    pub(crate) handle: Queue,
    pub(crate) family: u32,
}

pub(crate) struct QueueSelection {
    transfer_family: u32,
    compute_family: u32,
    async_comp_family: u32,
}

impl QueueSelection {
    pub(crate) fn get_queue_create_infos(&'_ self) -> Vec<DeviceQueueCreateInfo<'_>> {
        let mut map: HashMap<u32, Vec<f32>> = HashMap::new();

        map.entry(self.compute_family).or_default().push(1.0);
        map.entry(self.transfer_family).or_default().push(0.5);
        map.entry(self.async_comp_family).or_default().push(0.0);

        map .into_iter()
            .map(|(family_index, priorities_vec)| {
                let boxed: Box<[f32]> = priorities_vec.into_boxed_slice();
                let slice_ref: &'_ [f32] = Box::leak(boxed); // very minor memory leak (12 bytes)

                DeviceQueueCreateInfo::default()
                    .queue_family_index(family_index)
                    .queue_priorities(slice_ref)
            })
            .collect()
    }

    pub(crate) unsafe fn get_queues(&self, device: &Device) -> (VulkanQueue, VulkanQueue, VulkanQueue) {
        // Accumulate how many queues we've already taken per family.
        let mut counts: HashMap<_, _> = [
            (self.compute_family, 0usize),
            (self.async_comp_family, 0),
            (self.transfer_family, 0),
        ]
            .into_iter()
            .collect();

        let mut take_queue = |family| {
            let idx = *counts.get(&family).unwrap() as u32;
            *counts.get_mut(&family).unwrap() += 1;
            device.get_device_queue(family, idx)
        };

        let compute_queue = take_queue(self.compute_family);
        let transfer_queue = take_queue(self.transfer_family);
        let async_queue = take_queue(self.async_comp_family);

        (
            VulkanQueue { handle: compute_queue, family: self.compute_family },
            VulkanQueue { handle: async_queue,   family: self.async_comp_family },
            VulkanQueue { handle: transfer_queue, family: self.transfer_family },
        )
    }

    /// ONLY CALL ONCE
    pub(crate) unsafe fn set_global_families(&self) {
        if QUEUE_FAMILIES.get().is_some() {
            panic!("Cannot set global queue families more than once!")
        }
        
        let families = HashSet::from([self.compute_family, self.async_comp_family, self.transfer_family])
            .into_iter()
            .collect();
        
        QUEUE_FAMILIES.set(families).expect("Failed to set global queue families")
    }

    pub(crate) fn get_global_families() -> Vec<u32> {
        match QUEUE_FAMILIES.get() {
            None => panic!("Failed to retrieve global queue families"),
            Some(families) => families.clone()
        }
    }
}

pub(crate) unsafe fn select_queues(instance: &Instance, gpu: PhysicalDevice) -> Option<QueueSelection> {
    let family_properties = instance.get_physical_device_queue_family_properties(gpu);

    // Get the best-fit families, ensuring they all exist
    let compute_family = best_index(&family_properties, rank_main_compute_aptness)?;
    let transfer_family = best_index(&family_properties, rank_transfer_aptness)?;
    let async_comp_family = best_index(&family_properties, rank_async_compute_aptness)?;

    // Ensure that there are enough queues for everyone
    let mut map = HashMap::new();
    *map.entry(compute_family).or_insert(0) += 1;
    *map.entry(transfer_family).or_insert(0) += 1;
    *map.entry(async_comp_family).or_insert(0) += 1;

    for (idx, amt) in map.iter() {
        if family_properties[*idx as usize].queue_count < *amt {
            return None;
        }
    }

    Some(QueueSelection {
        transfer_family,
        compute_family,
        async_comp_family,
    })
}

fn best_index<T, O: Ord>(items: &[T], mut score_fn: fn(&T) -> Option<O>) -> Option<u32> {
    items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| score_fn(item).map(|s| (i, s)))
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i as u32)
}

/// Transfer queues usually only support transfer ops, and there's a couple of them
fn rank_transfer_aptness(family: &QueueFamilyProperties) -> Option<i32> {
    let supports_transfer = family.queue_flags.contains(QueueFlags::TRANSFER); // dealbreaker
    let supports_many_queues = family.queue_count > 1; // +1
    let supports_compute = family.queue_flags.contains(QueueFlags::COMPUTE); // -1
    let supports_graphics = family.queue_flags.contains(QueueFlags::GRAPHICS); // -2

    if (!supports_transfer) {
        return None;
    }

    let mut score = 0;
    if supports_graphics {score -= 2};
    if supports_compute {score -= 1};
    if supports_many_queues {score += 1};

    Some(score)
}

/// Async compute queues usually only support compute + transfer, and there's quite a few of them
fn rank_async_compute_aptness(family: &QueueFamilyProperties) -> Option<i32> {
    let supports_compute = family.queue_flags.contains(QueueFlags::COMPUTE); // dealbreaker
    let supports_many_queues = family.queue_count > 1; // +2
    let supports_transfer = family.queue_flags.contains(QueueFlags::TRANSFER); // +1
    let supports_graphics = family.queue_flags.contains(QueueFlags::GRAPHICS); // -2

    if (!supports_compute) {
        return None;
    }

    let mut score = 0;
    if supports_many_queues {score += 2};
    if supports_transfer {score += 1};
    if supports_graphics {score -= 2};

    Some(score)
}

// todo: might be clever to defer this to an async compute queue (with higher priority)
//  to avoid other GUI programs stealing the GPU resources when running
/// Highest priority queue, we choose the "main queue", there's usually only one
fn rank_main_compute_aptness(family: &QueueFamilyProperties) -> Option<i32> {
    let supports_compute = family.queue_flags.contains(QueueFlags::COMPUTE); // dealbreaker
    let supports_graphics = family.queue_flags.contains(QueueFlags::GRAPHICS); // +2
    let supports_transfer = family.queue_flags.contains(QueueFlags::TRANSFER); // +1
    let supports_many_queues = family.queue_count > 1; // -2

    if (!supports_compute) {
        return None;
    }

    let mut score = 0;
    if supports_graphics {score += 2};
    if supports_transfer {score += 1};
    if supports_many_queues {score -= 2};

    Some(score)
}
