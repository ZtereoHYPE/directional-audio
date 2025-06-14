use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use ash::vk::{AccessFlags, CommandBuffer, DependencyFlags, DescriptorSet, DescriptorSetLayout, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineLayout, PipelineStageFlags, Queue, ShaderStageFlags};
use ash::Device;
use crevice::std430::{Std430, Vec3};
use std::array::from_ref;

pub(crate) struct DelayModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue,
}

impl DelayModule {
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

    pub(crate) unsafe fn apply_delay(&mut self, command_buffer: &mut CommandBuffer, frame_counter: u32, camera_delta: Vec3, instance_amt: usize) {
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

        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 0, camera_delta.as_bytes());
        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 12, frame_counter.as_bytes());

        let workgroups = (GPU_WINDOW_SIZE as u32 / 64, instance_amt as u32);
        self.device.cmd_dispatch(*command_buffer, workgroups.0, workgroups.1, 1);

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
            PipelineStageFlags::COMPUTE_SHADER, // ...before executing any transfers from now on
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        );
    }
}
