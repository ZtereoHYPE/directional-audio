use std::cmp::PartialEq;
use std::collections::HashMap;
use std::hash::Hash;
use ash::vk::Semaphore;
use crate::vulkan::in_flight::{InFlight, InFlightCounter};

pub(crate) trait PipelineStage: Eq + Hash + Copy + Clone {
    fn val(&self) -> i64;
    fn num_stages() -> i64;
    fn last() -> Self;
}

/// TimelineTracker manages timeline semaphore values for N frames in flight
#[derive(Clone)]
pub(crate) struct TimelineTracker<T: PipelineStage> {
    frames: InFlight<u64>,
    semaphores: InFlight<Semaphore>,
    dependencies: HashMap<T, Vec<(u8, T)>>,
}



impl<T: PipelineStage> TimelineTracker<T> {
    pub fn new(semaphores: InFlight<Semaphore>, dependencies: HashMap<T, Vec<(u8, T)>>) -> Self {
        assert_eq!(dependencies.len(), T::num_stages() as usize);

        Self {
            frames: InFlight::from(vec![0; semaphores.0.len()]),
            semaphores,
            dependencies,
        }
    }

    pub fn completed_stage_val(&self, frames_prior: u8, stage_counter: InFlightCounter, stage: T) -> u64 {
        // If we are depending on a previous frame, then we want to get a semaphore value that has already
        // been scheduled, as not doing that would result in a deadlock.
        let in_flight = self.frames.0.len() as i64;
        let frame = self.frames[stage_counter] as i64 + (-(frames_prior as i64 + (in_flight - 1)) / in_flight);

        i64::max(frame * T::num_stages() + stage.val() + 1, 0) as u64
    }

    /// Call once per frame to allocate timeline values for the current frame
    pub fn advance_frame(&mut self, counter: InFlightCounter) {
        self.frames[counter] += 1;
    }

    /// Get wait dependencies for a given frame/stage
    pub fn get_wait_info(&self, counter: InFlightCounter, stage: T) -> (Vec<Semaphore>, Vec<u64>) {
        let mut values = vec![];
        let mut semaphores = vec![];

        for (frames_back, stage) in &self.dependencies[&stage] {
            let mut back_counter = counter;
            for _ in (0u8..*frames_back) { back_counter = back_counter.prev()}; // go back n frames

            values.push(self.completed_stage_val(*frames_back, back_counter, *stage));
            semaphores.push(self.semaphores[back_counter]);
        }

        assert_eq!(values.len(), semaphores.len());

        (semaphores, values)
    }

    pub fn get_signal_info(&self, counter: InFlightCounter, stage: T) -> (Vec<Semaphore>, Vec<u64>) {
        let next_val = self.completed_stage_val(0, counter, stage);

        (vec![self.semaphores[counter]], vec![next_val])
    }
}
