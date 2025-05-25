use crate::audio::{Frame, FRAME_AMT, FRAME_SIZE};
use crate::vulkan::signal_processor::fft::complex::{root_of_unity, scalar_mult};
use crate::vulkan::signal_processor::{FftFrame, FftStage, RADICES, RADIX_AMT};
use ash::vk::{AccessFlags, CommandBuffer, DependencyFlags, DescriptorSet, DescriptorSetLayout, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineLayout, PipelineStageFlags, Queue};
use ash::Device;
use crevice::std430::Vec2;
use std::array::from_ref;
// todo: maybe use newtype patter to refer to buffers?

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
    pub fn new(
        device: Device,
        pipelines: [Pipeline; RADIX_AMT],
        pipeline_layout: PipelineLayout,
        descriptor_sets: Vec<DescriptorSet>,
        descriptor_set_layout: DescriptorSetLayout,
        queue: Queue,
        stages: Vec<FftStage>,
    ) -> FftModule {
        FftModule {
            device,
            pipelines,
            pipeline_layout,
            descriptor_sets,
            descriptor_set_layout,
            queue,
            stages,
        }
    }

    pub unsafe fn gpu_fourier_transform(&mut self, command_buffer: &mut CommandBuffer, initial_buffer: u32, inverse: bool) -> u32 {
        for (idx, stage) in self.stages.iter().enumerate() {
            let workgroups = (FRAME_SIZE as u32 / (stage.radix * 32), FRAME_AMT as u32);

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

    // todo: look into making this a const fn
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

    pub fn frame_to_fft(frame: &Frame) -> FftFrame {
        // todo: avoid initialization here
        let mut samples = [Vec2{x: 0.0, y: 0.0}; FRAME_SIZE];

        for (idx, value) in frame.iter().enumerate() {
            samples[idx].x = *value;
        }

        samples
    }

    pub fn fft_to_frame(input: &FftFrame) -> Frame {
        let mut frame: Frame = [0.0; FRAME_SIZE];

        for (idx, value) in input.iter().enumerate() {
            frame[idx] = value.x;
        }

        frame
    }
    
    pub(crate) fn local_fourier_transform(buffer: Vec<Vec2>, inverse: bool) -> Vec<Vec2> {
        let len = buffer.len();
        if (len & (len - 1)) != 0 {
            panic!("This function can only be called on buffers whose length is a power of two")
        }

        let w = if inverse {
            root_of_unity(-(len as isize))
        } else {
            root_of_unity(len as isize)
        };

        let mut result = Self::cpu_fft(buffer, w);

        if inverse {
            let normalization = 1.0 / len as f32;
            result.iter_mut().for_each(|v| *v = scalar_mult(*v, normalization));
        }

        result
    }

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

        (0..half).for_each(|idx| {
            buffer[idx       ] = complex::sum(left[idx], complex::mult(x, right[idx]));
            buffer[idx + half] = complex::sub(left[idx], complex::mult(x, right[idx]));
            x = complex::mult(x, w);
        });

        buffer
    }
}

mod complex {
    use crevice::std430::Vec2;
    use std::f32::consts::PI;

    pub fn root_of_unity(len: isize) -> Vec2 {
        let angle = 2.0 * PI / len as f32;
        Vec2 {
            x: angle.cos(),
            y: angle.sin(),
        }
    }

    pub fn mult(left: Vec2, right: Vec2) -> Vec2 {
        Vec2 {
            x: left.x * right.x - left.y * right.y,
            y: left.x * right.y + left.y * right.x,
        }
    }

    pub fn sum(mut left: Vec2, right: Vec2) -> Vec2 {
        left.x += right.x;
        left.y += right.y;
        left
    }

    pub fn sub(mut left: Vec2, right: Vec2) -> Vec2 {
        left.x -= right.x;
        left.y -= right.y;
        left
    }

    pub fn scalar_mult(mut left: Vec2, right: f32) -> Vec2 {
        left.x *= right;
        left.y *= right;
        left
    }

    pub fn magnitude(complex: Vec2) -> f32 {
        (complex.x * complex.x + complex.y * complex.y).sqrt()
    }
}

#[cfg(test)]
mod test {
    use crate::audio::{AudioProvider, FRAME_SIZE};
    use crate::vulkan::signal_processor::fft::FftModule;

    const EPSILON: f32 = 0.0005;

    #[test]
    fn cpu_fft_test() {
        let vector = Vec::from(FftModule::frame_to_fft(&AudioProvider::random_frame(32)));
        let fft = FftModule::local_fourier_transform(vector.clone(), false);
        let ifft = FftModule::local_fourier_transform(fft.clone(), true);
        
        for (&s_before, s_after) in vector.iter().zip(ifft) {
            let diff = (s_after.x - s_before.x).abs();
            assert!(diff < EPSILON);
        }
    }
}