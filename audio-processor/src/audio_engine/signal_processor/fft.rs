use crate::audio_engine::gpu_structures::{FftBuffer, FftConstants, FftUbo, GPU_WINDOW_SIZE};
use crate::util::complex;
use crate::util::complex::{root_of_unity, scalar_mult};
use ash::vk::{AccessFlags, Buffer, BufferCreateInfo, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, SpecializationMapEntry, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use crevice::std430::Vec2;
use std::array::from_ref;
use std::f32::consts::PI;
use std::mem::transmute;
use std::rc::Rc;
use vk_mem::{Alloc, AllocationCreateFlags, Allocator};
use crate::audio_engine::buffer_initializer::BufferInitializer;
use crate::audio_engine::read_file_words;
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
    buffers: [Buffer; 2],
    queue: Queue,
    stages: Vec<FftStage>,
}

impl FftModule {
    pub(crate) fn new(
        allocator: Rc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        queue: (Queue, u32),
        descriptor_pool: DescriptorPool,
        stages: Vec<FftStage>,
    ) -> Self {
        let (fft_ubos, fft_ubo_memories) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftUbo>() as u64)
                .usage(BufferUsageFlags::UNIFORM_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let mut buffers = vec![];
            let mut memories = vec![];

            // Create a UBO per stage
            for stage in &stages {
                let (buffer, mut memory) = allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer");

                let map = allocator
                    .map_memory(&mut memory)
                    .expect("Failed to map memory");

                let inverse = false;

                // Populate the UBO
                let direction: f32 = if !inverse { -1.0 } else { 1.0 };
                let normalization: f32 = if !inverse { 1.0 } else { 1.0 / stage.radix as f32};
                let data = FftUbo {
                    split_size: stage.split_size,
                    radix_stride: stage.stride,
                    angle_direction_factor: direction,
                    angle_spin_factor: direction * (PI / (stage.split_size as f32)),
                    normalization_factor: normalization,
                };

                std::ptr::write(map.cast(), data);

                allocator.unmap_memory(&mut memory);
                buffers.push(buffer);
                memories.push(memory);
            }

            (buffers, memories)
        };

        let ((fft_gpu_buf_1, fft_gpu_mem_1), (fft_gpu_buf_2, fft_gpu_mem_2)) = unsafe {
            let buffer_info = BufferCreateInfo::default()
                .size(FftBuffer::max_size() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            (
                allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer"),

                allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer"),
            )
        };

        let (descriptor_sets, descriptor_set_layout) = unsafe {
            // Create layout, which is the same across every stage
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE)
            ];

            let set_layout_info = DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            let set_layouts = vec![
                device
                    .create_descriptor_set_layout(&set_layout_info, None)
                    .expect("Failed to create descriptor set layout")
                ; stages.len()
            ];

            // Allocate one set per stage
            let set_info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts[..]);

            let sets = device
                .allocate_descriptor_sets(&set_info)
                .expect("Failed to allocate descriptor sets");

            // Create buffer information
            let ubo_infos = fft_ubos
                .iter()
                .map(|ubo| {
                    DescriptorBufferInfo::default()
                        .buffer(*ubo)
                        .range(WHOLE_SIZE)
                })
                .collect::<Vec<_>>();

            let ssbo_infos = (
                DescriptorBufferInfo::default()
                    .buffer(fft_gpu_buf_1)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(fft_gpu_buf_2)
                    .range(WHOLE_SIZE),
            );

            // Write the descriptor sets
            let mut writes = vec![];
            for (idx, set) in sets.iter().enumerate() {
                writes.extend_from_slice(&[
                    WriteDescriptorSet::default()
                        .dst_set(*set)
                        .descriptor_count(1)
                        .dst_binding(0)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(from_ref(&ubo_infos[idx])),

                    WriteDescriptorSet::default()
                        .dst_set(*set)
                        .descriptor_count(1)
                        .dst_binding(1 + (idx % 2) as u32) // 1, 2, 1, ...
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(from_ref(&ssbo_infos.0)),

                    WriteDescriptorSet::default()
                        .dst_set(*set)
                        .descriptor_count(1)
                        .dst_binding(1 + ((idx + 1) % 2) as u32) // 2, 1, 2, ...
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(from_ref(&ssbo_infos.1))
                ]);
            }

            device.update_descriptor_sets(&writes[..], &[]);
            (sets, set_layouts[0])
        };

        let (pipelines, pipeline_layout) = unsafe {
            // The layout is the same for all pipelines
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_set_layout));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");


            // The SPIR-V is the same for all pipelines
            let code_words = read_file_words("target/shaders/fft.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");


            // There is a specialization constant with a different value for each pipeline
            let specialization_entries = [
                SpecializationMapEntry::default()
                    .constant_id(0)
                    .offset(0)
                    .size(size_of::<i32>()),

                SpecializationMapEntry::default()
                    .constant_id(1)
                    .offset(4)
                    .size(size_of::<i32>()),
            ];

            let constant_data = RADICES
                .iter()
                .map(|radix| {
                    FftConstants {
                        radix: *radix as i32,
                        frame_size: GPU_WINDOW_SIZE as i32
                    }
                })
                .collect::<Vec<_>>();

            let specialization_infos: [_; RADIX_AMT] = constant_data
                .iter()
                .map(|datum| {
                    SpecializationInfo::default()
                        .map_entries(&specialization_entries)
                        .data(transmute::<_, &[u8; 8]>(datum))
                })
                .collect::<Vec<_>>()
                .try_into().unwrap();

            let pipeline_infos: [_; RADIX_AMT] = (0..RADIX_AMT)
                .map(|idx| {
                    let stage_info = PipelineShaderStageCreateInfo::default()
                        .stage(ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .specialization_info(&specialization_infos[idx])
                        .name(c"main");

                    ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(stage_info)
                })
                .collect::<Vec<_>>()
                .try_into().unwrap();

            // Create pipelines
            let pipelines: [Pipeline; RADIX_AMT] = device
                .create_compute_pipelines(PipelineCache::null(), &pipeline_infos, None)
                .expect("Failed to create pipeline")
                .try_into().unwrap();

            (pipelines, layout)
        };

        Self {
            device,
            pipelines,
            pipeline_layout,
            descriptor_sets,
            descriptor_set_layout,
            buffers: [fft_gpu_buf_1, fft_gpu_buf_2],
            queue: queue.0,
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

    pub fn starting_buffer(&self) -> Buffer {
        self.buffers[0]
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
