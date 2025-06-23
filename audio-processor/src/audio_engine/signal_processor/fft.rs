use crate::audio_engine::gpu_structures::GPU_WINDOW_SIZE;
use crate::util::complex;
use crate::util::complex::{root_of_unity, scalar_mult};
use ash::vk::{AccessFlags, CommandBuffer, DependencyFlags, DescriptorSet, DescriptorSetLayout, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineLayout, PipelineStageFlags, Queue};
use ash::Device;
use crevice::std430::Vec2;
use std::array::from_ref;
// todo: maybe use newtype pattern to refer to buffers?

pub(crate) const RADIX_AMT: usize = 3;
pub(crate) const RADICES: [u32; RADIX_AMT] = [8, 4, 2];

// The FFT algorithm for the GPU is divided into log(N) stages performing
// butterfly operations on the data and shifting data around.
// For performance reasons, each butterfly operation is performed, if possible,
// on batches larger than 2 items at a time. This batch size is called the radix,
// and allows greatly reduced amount of stages performed on the data.
#[derive(Debug)]
pub(crate) struct FftStage {
    // The radix used for the stage of the FFT calculation.
    pub(crate) radix: u32,

    // How large the current "subarray" of data being processed is.
    pub(crate) split_size: u32,

    // The stride between data in a given shader invocation (= input_size / radix)
    pub(crate) stride: u32
}

pub(crate) struct FftModule {
    device: Device,
    pipelines: [Pipeline; RADIX_AMT], // todo: potentially move away from fixed-size arrays?
    pipeline_layout: PipelineLayout,
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue,
    stages: Vec<FftStage>,
}

impl FftModule {
    pub(crate) fn new(
        device: Device,
        pipelines: [Pipeline; RADIX_AMT],
        pipeline_layout: PipelineLayout,
        descriptor_sets: Vec<DescriptorSet>,
        descriptor_set_layout: DescriptorSetLayout,
        queue: Queue,
        stages: Vec<FftStage>,
    ) -> Self {
        Self {
            device,
            pipelines,
            pipeline_layout,
            descriptor_sets,
            descriptor_set_layout,
            queue,
            stages,
        }
    }

    pub unsafe fn gpu_fourier_transform(&mut self, command_buffer: &mut CommandBuffer, initial_buffer: u32, inverse: bool, instance_amt: usize) -> u32 {
        for (idx, stage) in self.stages.iter().enumerate() {
            let workgroups = (GPU_WINDOW_SIZE as u32 / (stage.radix * 32), instance_amt as u32);

            // todo: Push constants to select the right buffer! -> change the shader as well
            self.device.cmd_bind_descriptor_sets(
                *command_buffer,
                PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                from_ref(&self.descriptor_sets[idx]),
                &[]
            );
            self.device.cmd_bind_pipeline(
                *command_buffer,
                PipelineBindPoint::COMPUTE,
                self.pipelines[Self::stage_pipeline(stage.radix)]
            );

            self.device.cmd_dispatch(
                *command_buffer,
                workgroups.0, workgroups.1, 1
            );

            let memory_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
                .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

            self.device.cmd_pipeline_barrier(
                *command_buffer,
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute dispatches from now on
                DependencyFlags::empty(),
                from_ref(&memory_barrier),
                &[],
                &[]
            );
        }

        (self.stages.len() % 2) as u32 // this is the index of the buffer where the result will be
    }

    pub(super) fn fft_stages(input_size: usize) -> Vec<FftStage> {
        let mut stages = vec![];

        // the first stage covers the whole array
        let mut split_size = input_size as u32;

        // while we haven't "recursed down" to the base case
        while split_size > 1 {
            let largest_compatible_radix = RADICES
                .into_iter()
                .find(|radix| split_size % *radix == 0)
                .expect("Failed to find a radix that could divide array size. Are you sure it's a power of 2?");

            stages.push(FftStage {
                radix: largest_compatible_radix,
                split_size: (input_size as u32) / split_size,
                stride: (input_size as u32) / largest_compatible_radix,
            });

            split_size /= largest_compatible_radix;
        }

        stages
    }

    // todo: could be a bit more elegant
    fn stage_pipeline(radix: u32) -> usize {
        for (idx, rdx) in RADICES.iter().enumerate() {
            if radix == *rdx {
                return idx;
            }
        }
        panic!("invalid radix");
    }

    pub(crate) fn local_fourier_transform(buffer: Vec<Vec2>, inverse: bool) -> Vec<Vec2> {
        let len = buffer.len();
        if !len.is_power_of_two() {
            panic!("This function can only be called on buffers whose length is a power of two")
        }

        let w = if inverse {
            root_of_unity(len as isize)
        } else {
            root_of_unity(-(len as isize))
        };

        let mut result = Self::cpu_fft(buffer, w);

        if inverse {
            let normalization = 1.0 / len as f32;
            result.iter_mut().for_each(|v| *v = scalar_mult(*v, normalization));
        }

        result
    }

    // todo: implement a more performant version, perhaps in-place?
    fn cpu_fft(mut buffer: Vec<Vec2>, w: Vec2) -> Vec<Vec2> {
        let len = buffer.len();
        if len == 1 {
            return buffer;
        }

        let left = buffer.iter().step_by(2).cloned().collect();
        let right = buffer.iter().skip(1).step_by(2).cloned().collect();

        let left = FftModule::cpu_fft(left, complex::mult(w, w));
        let right = FftModule::cpu_fft(right, complex::mult(w, w));

        let half = len / 2;
        let mut x = Vec2 {x: 1.0, y: 0.0};

        for idx in 0..half {
            let multiplied_right = complex::mult(x, right[idx]);
            buffer[idx       ] = complex::sum(left[idx], multiplied_right);
            buffer[idx + half] = complex::sub(left[idx], multiplied_right);
            x = complex::mult(x, w);
        }

        buffer
    }
}
