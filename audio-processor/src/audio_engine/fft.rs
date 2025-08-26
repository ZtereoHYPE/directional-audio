use crate::audio_engine::gpu_constants::{GPU_WINDOW_SIZE, MAX_INSTANCES};
use crate::audio_engine::GpuWindow;
use crate::util::complex::root_of_unity;
use crate::util::{complex, AsBytes};
use crate::vulkan::buffer::{InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use crate::vulkan::buffer_initializer::BufferInitializer;
use crate::vulkan::read_spirv_words;
use crate::vulkan::spec_constants::SpecConstantList;
use ash::vk::{AccessFlags, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, ShaderModuleCreateInfo, ShaderStageFlags, SpecializationInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use glam::Vec2;
use std::array::from_ref;
use std::f32::consts::PI;
use std::rc::Rc;
use std::sync::Arc;
use vk_mem::Allocator;
use crate::vulkan::queue::VulkanQueue;

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

#[repr(C)]
struct FftPushConstants {
    input_buffer: u32,
    output_buffer: u32,
    split_size: u32,
    angle_direction_factor: f32,
    angle_spin_factor: f32,
    normalization_factor: f32,
}

pub(crate) struct FftModule {
    device: Device,
    pipelines: [Pipeline; RADIX_AMT],
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    buffers: [VulkanBuffer<FftBufferData>; 2],
    pub debug_buffer: LocalVulkanBuffer<FftBufferData>,
    stages: Vec<FftStage>,
}

impl FftModule {
    pub(super) fn new(
        allocator: Arc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        descriptor_pool: DescriptorPool,
    ) -> Self {
        let constants = SpecConstantList::new()
            .append(GPU_WINDOW_SIZE as u32);
        
        let stages = Self::fft_stages(GPU_WINDOW_SIZE);
        let fft_buffer_0 = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let fft_buffer_1 = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let debug_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        let (descriptor_set, descriptor_set_layout) = unsafe {
            // Create layout, which is the same across every stage
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_count(2)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),
            ];

            let set_layout_info = DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            let set_layout = device
                .create_descriptor_set_layout(&set_layout_info, None)
                .expect("Failed to create descriptor set layout");

            // Allocate one set per stage
            let set_info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(from_ref(&set_layout));

            let set = device
                .allocate_descriptor_sets(&set_info)
                .expect("Failed to allocate descriptor sets")[0];

            let ssbo_infos = [
                DescriptorBufferInfo::default()
                    .buffer(fft_buffer_0.handle())
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(fft_buffer_1.handle())
                    .range(WHOLE_SIZE),
            ];

            // Write the descriptor set
            let mut write = WriteDescriptorSet::default()
                .dst_set(set)
                .descriptor_count(2)
                .dst_binding(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&ssbo_infos[..]);

            device.update_descriptor_sets(from_ref(&write), &[]);
            (set, set_layout)
        };

        let (pipelines, pipeline_layout) = unsafe {
            let push_constant_range = PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .size(size_of::<FftPushConstants>() as _);

            // The layout is the same for all pipelines
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_set_layout))
                .push_constant_ranges(from_ref(&push_constant_range));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            // The SPIR-V is the same for all pipelines
            let code_words = read_spirv_words("target/shaders/fft.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            // For each radix create a new set of spec constants
            let constant_data = RADICES
                .iter()
                .map(|&radix| constants.clone().append(radix).build())
                .collect::<Vec<_>>();

            let specialization_infos: [_; RADIX_AMT] = constant_data
                .iter()
                .map(|(entries, data)| {
                    SpecializationInfo::default()
                        .map_entries(&entries)
                        .data(&data)
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
            descriptor_set,
            descriptor_set_layout,
            buffers: [fft_buffer_0, fft_buffer_1],
            debug_buffer,
            stages,
        }
    }

    pub unsafe fn gpu_fourier_transform(&mut self, command_buffer: &mut CommandBuffer, initial_buffer: u32, inverse: bool, instance_amt: usize) -> u32 {
        self.device.cmd_bind_descriptor_sets(
            *command_buffer,
            PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            from_ref(&self.descriptor_set),
            &[]
        );

        for (idx, stage) in self.stages.iter().enumerate() {
            let workgroups = (GPU_WINDOW_SIZE as u32 / (stage.radix * 32), instance_amt as u32);

            let direction: f32 = if !inverse { -1.0 } else { 1.0 };
            let normalization: f32 = if !inverse { 1.0 } else { 1.0 / stage.radix as f32};
            let push_constants = FftPushConstants {
                input_buffer: (idx % 2) as u32,
                output_buffer: ((idx + 1) % 2) as u32,
                split_size: stage.split_size,
                angle_direction_factor: direction,
                angle_spin_factor: direction * (PI / (stage.split_size as f32)),
                normalization_factor: normalization,
            };

            self.device.cmd_push_constants(
                *command_buffer,
                self.pipeline_layout,
                ShaderStageFlags::COMPUTE,
                0,
                push_constants.as_bytes()
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
                from_ref(&memory_barrier), // todo: use individual buffer barriers (based on stage) and measure!
                &[],
                &[]
            );
        }

        // todo: only if debug enabled, this is quite expensive!
        // todo: not hardcoded to the specific buffer!
        self.device.cmd_copy_buffer(
            *command_buffer,
            self.buffers[0].handle(),
            self.debug_buffer.handle(),
            FftBufferData::region()
        );

        (self.stages.len() % 2) as u32 // this is the index of the buffer where the result will be
    }

    pub fn starting_buffer(&self) -> &VulkanBuffer<FftBufferData> {
        &self.buffers[0]
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
            result.iter_mut().for_each(|v| *v *= normalization);
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

        let next_w = complex::mult(w, w);
        let left = FftModule::cpu_fft(left, next_w);
        let right = FftModule::cpu_fft(right, next_w);

        let half = len / 2;
        let mut x = Vec2 {x: 1.0, y: 0.0};

        for idx in 0..half {
            let multiplied_right = complex::mult(x, right[idx]);
            buffer[idx       ] = left[idx] + multiplied_right;
            buffer[idx + half] = left[idx] - multiplied_right;
            x = complex::mult(x, w);
        }

        buffer
    }
}

#[repr(C)]
pub(crate) struct FftBufferData {
    pub windows: [GpuWindow; MAX_INSTANCES]
}
impl InlineBufferData for FftBufferData {}
