use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::scene::FRAME_AMT;
use ash::vk::{AccessFlags, CommandBuffer, DependencyFlags, DescriptorSet, DescriptorSetLayout, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineLayout, PipelineStageFlags, Queue};
use ash::Device;
use std::array::from_ref;

pub(crate) const MAX_DELAY_MS: u32 = 5000;

pub struct HrtfModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue,
}

impl HrtfModule {
    pub fn new(
        device: Device,
        pipeline: Pipeline,
        pipeline_layout: PipelineLayout,
        descriptor_set: DescriptorSet,
        descriptor_set_layout: DescriptorSetLayout,
        queue: Queue,
    ) -> Self {
        Self {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set,
            descriptor_set_layout,
            queue,
        }
    }

    pub unsafe fn apply_hrtf(&mut self, command_buffer: &mut CommandBuffer) {
        self.device.cmd_bind_descriptor_sets(
            *command_buffer,
            PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            from_ref(&self.descriptor_set),
            &[]
        );

        self.device.cmd_bind_pipeline(
            *command_buffer,
            PipelineBindPoint::COMPUTE,
            self.pipeline
        );

        let workgroups = (GPU_WINDOW_SIZE as u32 / 64 / 2, FRAME_AMT as u32);

        self.device.cmd_dispatch(*command_buffer, workgroups.0, workgroups.1, 1);

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any shader read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
            PipelineStageFlags::TRANSFER, // ...before executing any transfers from now on
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        );
    }
}
