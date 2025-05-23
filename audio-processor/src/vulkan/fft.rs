#![allow(unsafe_op_in_unsafe_fn)]

use std::{array::from_ref, f32::consts::PI, mem::transmute, num, ops::Add, sync::Arc, u64::MAX};

use ash::{util::Align, vk::{AccessFlags, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, CommandBufferBeginInfo, CommandBufferUsageFlags, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, Fence, FenceCreateInfo, MemoryBarrier, MemoryPropertyFlags, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet, WHOLE_SIZE}};
use crevice::std430::{self, AsStd430, Vec2, Vec3};
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, AllocatorCreateInfo};

use crate::{audio::{Frame, FRAME_AMT, FRAME_SIZE}, vulkan::engine::DescriptorRequirement};

use super::{engine::{VulkanContext, VulkanModule}, read_file_words};


const RADIX_AMT: usize = 3;
const RADICES: [u32; RADIX_AMT] = [8, 4, 2];

// The FFT algorithm for the GPU is divided into log(N) stages performing 
// butterfly operations on the data and shifting data around.
// For performance reasons, each butterfly operation is performed, if possible,
// on batches larger than 2 items at a time. This batch size is called the radix,
// and allows greatly reduced amount of stages performed on the data.
#[derive(Debug)]
struct FftStage {
    // The radix used for the stage of the FFT calculation.
    radix: u32,

    // How large the current "subarray" of data being processed is.
    split_size: u32,

    // The stride between data in a given shader invocation (= input_size / radix)
    stride: u32
}

// todo: implement GpuData for these structs and use that trait instead

#[derive(AsStd430)]
struct FftUbo {
    split_size: u32,
    radix_stride: u32,
    angle_direction_factor: f32,
    angle_spin_factor: f32,
    normalization_factor: f32,
}

#[repr(C)]
struct FftConstants {
    radix: i32,
    frame_size: i32
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct FftFrame {
    pub(crate) samples: [Vec2; FRAME_SIZE] // todo: this is not just for fft, should be GPU frame or smth
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FftBuffer {
    pub frames: [FftFrame; FRAME_AMT]
}

pub struct FftModule {
    buffer_allocator: vk_mem::Allocator,

    cpu_buffer: Buffer,
    cpu_buffer_memory: vk_mem::Allocation,
    cpu_buffer_map: *mut u8,

    gpu_buffers: [Buffer; 2],
    gpu_buffers_memory: [vk_mem::Allocation; 2],

    // todo: switch to push constants
    ubos: Vec<Buffer>,
    ubo_memories: Vec<vk_mem::Allocation>,

    descriptor_sets: Vec<DescriptorSet>,
    descriptor_layout: DescriptorSetLayout,

    pipelines: [Pipeline; RADIX_AMT],
    pipeline_layout: PipelineLayout,

    readback_fence: Fence
}

impl VulkanModule for FftModule {
    fn descriptors() -> Vec<super::engine::DescriptorRequirement> {
        let stages = Self::fft_stages(FRAME_SIZE);

        vec![
            DescriptorRequirement {
                ttype: DescriptorType::UNIFORM_BUFFER,
                amount: stages.len() as u32
            },
            DescriptorRequirement {
                ttype: DescriptorType::STORAGE_BUFFER,
                amount: 1 + 2 * stages.len() as u32 // cpu + 2 gpu * stages
            },
        ]
    }
}

impl FftModule {
    pub unsafe fn new(ctx: &mut VulkanContext, inverse: bool) -> Self {
        let stages = FftModule::fft_stages(FRAME_SIZE);

        let buffer_allocator = {
            let allocator_create_info = AllocatorCreateInfo::new(
                &ctx.instance, 
                &ctx.device,
                ctx.gpu
            );

            vk_mem::Allocator::new(allocator_create_info)
                .expect("Failed to create memory allocator")
        };

        let (cpu_buffer, cpu_buffer_memory, cpu_buffer_map) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftBuffer>() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST)
                .queue_family_indices(from_ref(&ctx.compute_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                preferred_flags: MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_CACHED,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
                ..Default::default()
            };

            let (buffer, mut memory) = buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer");

            let map = buffer_allocator
                .map_memory(&mut memory)
                .expect("Failed to map memory");

            (buffer, memory, map)
        };

        let (ubos, ubo_memories) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftUbo>() as u64)
                .usage(BufferUsageFlags::UNIFORM_BUFFER)
                .queue_family_indices(from_ref(&ctx.compute_queue.1))
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
                let (buffer, mut memory) = buffer_allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer");
    
                let map = buffer_allocator
                    .map_memory(&mut memory)
                    .expect("Failed to map memory");

                // Populate the UBO
                let direction: f32 = if inverse { -1.0 } else { 1.0 };
                let normalization: f32 = if inverse { 1.0  } else { 1.0 / stage.radix as f32}; // todo: i flipped these, make sure that was a right decision
                let data = FftUbo {
                    split_size: stage.split_size,
                    radix_stride: stage.stride,
                    angle_direction_factor: direction,
                    angle_spin_factor: direction * (PI / (stage.split_size as f32)),
                    normalization_factor: normalization,
                };

                std::ptr::write(map.cast(), data);

                buffer_allocator.unmap_memory(&mut memory);
                buffers.push(buffer);
                memories.push(memory);
            }

            (buffers, memories)
        };

        let ((gpu_buf_1, gpu_mem_1), (gpu_buf_2, gpu_mem_2)) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftBuffer>() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&ctx.compute_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            (
                buffer_allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer"),

                buffer_allocator
                    .create_buffer(&buffer_info, &allocation_info)
                    .expect("Failed to create buffer"),
            )
        };

        let (descriptor_sets, descriptor_layout) = {
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
                ctx.device
                    .create_descriptor_set_layout(&set_layout_info, None)
                    .expect("Failed to create descriptor set layout")
                ; stages.len()
            ];

            // Allocate one set per stage
            let set_info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(ctx.descriptor_pool)
                .set_layouts(&set_layouts[..]);
            
            let sets = ctx.device
                .allocate_descriptor_sets(&set_info)
                .expect("Failed to allocate descriptor sets");

            // Create buffer information
            let ubo_infos = ubos
                .iter()
                .map(|ubo| {
                    DescriptorBufferInfo::default()
                    .buffer(*ubo)
                    .range(WHOLE_SIZE)
                })
                .collect::<Vec<_>>();

            let ssbo_infos = (
                DescriptorBufferInfo::default()
                    .buffer(gpu_buf_1)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(gpu_buf_2)
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

            ctx.device.update_descriptor_sets(&writes[..], &[]);
            (sets, set_layouts[0])
        };

        let (pipelines, pipeline_layout) = {
            // The layout is the same for all pipelines
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_layout));

            let layout = ctx.device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");


            // The SPIR-V is the same for all pipelines
            let code_words = read_file_words("target/shaders/fft.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = ctx.device
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
                        frame_size: FRAME_SIZE as i32
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
            let pipelines: [Pipeline; RADIX_AMT] = ctx.device
                .create_compute_pipelines(PipelineCache::null(), &pipeline_infos, None)
                .expect("Failed to create pipeline")
                .try_into().unwrap();
            
            (pipelines, layout)
        };

        let readback_fence = ctx.device
            .create_fence(&FenceCreateInfo::default(), None)
            .expect("failed to create fence");

        Self {
            buffer_allocator,

            cpu_buffer,
            cpu_buffer_memory,
            cpu_buffer_map,

            gpu_buffers: [gpu_buf_1, gpu_buf_2],
            gpu_buffers_memory: [gpu_mem_1, gpu_mem_2],

            ubos,
            ubo_memories,

            descriptor_sets,
            descriptor_layout,

            pipelines,
            pipeline_layout,

            readback_fence
        }
    }

    // todo: potentially switch buffer to ref only with .as_ref()
    pub unsafe fn process_buffer(&mut self, ctx: &VulkanContext, buffer: &Box<FftBuffer>) -> Buffer {
        // copy frame to cpu buffer
        copy_from_box(buffer, self.cpu_buffer_map.cast::<FftBuffer>());

        self.buffer_allocator.flush_allocation(&self.cpu_buffer_memory, 0, WHOLE_SIZE);

        let stages = FftModule::fft_stages(FRAME_SIZE);
        
        // dispatch compute
        {
            let begin_info = CommandBufferBeginInfo::default()
                .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            ctx.device
                .begin_command_buffer(ctx.command_buffer, &begin_info)
                .expect("Failed to begin command buffer recording");

            let regions = [ BufferCopy::default().size(size_of::<FftBuffer>() as _) ]; // todo: this could not be right size
            
            ctx.device.cmd_copy_buffer(
                ctx.command_buffer, 
                self.cpu_buffer, 
                self.gpu_buffers[0], 
                &regions
            );

            let memory_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::TRANSFER_WRITE) // flush any transfer write caches
                .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

            ctx.device.cmd_pipeline_barrier(
                ctx.command_buffer, 
                PipelineStageFlags::TRANSFER, // wait for all transfer commands so far...
                PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute from now on
                DependencyFlags::empty(), 
                from_ref(&memory_barrier), 
                //&[],
                &[], 
                &[]
            );

            // todo: an analytical way of finding the size would be nice for constness reasons
            for (idx, stage) in stages.iter().enumerate() { 
                let workgroups = (FRAME_SIZE as u32 / (stage.radix * 32), FRAME_AMT as u32);

                ctx.device.cmd_bind_descriptor_sets(
                    ctx.command_buffer, 
                    PipelineBindPoint::COMPUTE, 
                    self.pipeline_layout, 
                    0, 
                    from_ref(&self.descriptor_sets[idx]), 
                    &[]
                );

                ctx.device.cmd_bind_pipeline(
                    ctx.command_buffer, 
                    PipelineBindPoint::COMPUTE, 
                    self.pipelines[FftModule::stage_pipeline(stage.radix)]
                );

                ctx.device.cmd_dispatch(
                    ctx.command_buffer, 
                    workgroups.0, workgroups.1, 1
                );

                let memory_barrier = MemoryBarrier::default()
                    .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
                    .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

                ctx.device.cmd_pipeline_barrier(
                    ctx.command_buffer, 
                    PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                    PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute dispatches from now on
                    DependencyFlags::empty(), 
                    from_ref(&memory_barrier), 
                    //&[],
                    &[], 
                    &[]
                );
            }

            let memory_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::SHADER_WRITE) // flush any compute write write caches
                .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any transfer read caches

            ctx.device.cmd_pipeline_barrier(
                ctx.command_buffer, 
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::TRANSFER, // ...before executing any transfer from now on
                DependencyFlags::empty(), 
                from_ref(&memory_barrier), 
                //&[],
                &[], 
                &[]
            );

            ctx.device.cmd_copy_buffer(
                ctx.command_buffer, 
                self.gpu_buffers[stages.len() % 2],  // 1 stage is buffer 2 (index 1), 2 stages is buffer 1 (index 0), ...
                self.cpu_buffer, 
                &regions
            );

            ctx.device.end_command_buffer(ctx.command_buffer);

            // todo: much more synchronization here
            let submit_info = SubmitInfo::default()
                .command_buffers(from_ref(&ctx.command_buffer));

            ctx.device
                .queue_submit(ctx.compute_queue.0, &[submit_info], self.readback_fence)
                .expect("Failed to submit command buffer");
        }

        ctx.device.wait_for_fences(from_ref(&self.readback_fence), true, MAX);

        // readback result
        self.buffer_allocator.invalidate_allocation(&self.cpu_buffer_memory, 0, WHOLE_SIZE);

        //copy_to_box(self.cpu_buffer_map as *const FftBuffer)

        self.gpu_buffers[stages.len() % 2]
    }

    // todo: look into making this a const fn
    fn fft_stages(input_size: usize) -> Vec<FftStage> {
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

        FftFrame { samples }
    }

    pub fn fft_to_frame(input: &FftFrame) -> Frame {
        let mut frame: Frame = [0.0; FRAME_SIZE];

        for (idx, value) in input.samples.iter().enumerate() {
            frame[idx] = value.x;
        }

        frame
    }

    pub unsafe fn free_buf(&mut self, buffer: Buffer) {
        let stages = FftModule::fft_stages(FRAME_SIZE);
        self.buffer_allocator.destroy_buffer(buffer, &mut self.gpu_buffers_memory[stages.len() % 2]);
    }
}

impl Drop for FftModule {
    fn drop(&mut self) {
        unsafe {
            self.buffer_allocator.unmap_memory(&mut self.cpu_buffer_memory);
            self.buffer_allocator.destroy_buffer(self.cpu_buffer, &mut self.cpu_buffer_memory);
    
            for (alloc, buf) in self.gpu_buffers_memory.iter_mut().zip(self.gpu_buffers) {
                self.buffer_allocator.destroy_buffer(buf, alloc);
            }
    
            for (alloc, buf) in self.ubo_memories.iter_mut().zip(self.ubos.clone()) {
                self.buffer_allocator.destroy_buffer(buf, alloc);
            }
        }
    }
}

pub(crate) unsafe fn copy_to_box<T>(mem: *const T) -> Box<T> {
    // Allocate the required space
    let layout = std::alloc::Layout::new::<T>();
    let ptr = std::alloc::alloc_zeroed(layout) as *mut T;

    // Copy the memory value
    std::ptr::copy(mem, ptr, 1);

    // Wrap in a box
    Box::from_raw(ptr)
}

pub(crate) fn cpu_fft(mut buffer: Vec<Vec2>, w: Vec2) -> Vec<Vec2> {
    let len = buffer.len();
    if len == 1 {
        return buffer;
    }

    let left = buffer.iter().step_by(2).cloned().collect();
    let right = buffer.iter().skip(1).step_by(2).cloned().collect();

    let left = cpu_fft(left, complex_mult(w, w));
    let right = cpu_fft(right, complex_mult(w, w));

    let half = len / 2;
    let mut x = Vec2 {x: 1.0, y: 0.0};

    (0..half).for_each(|idx| {
        buffer[idx       ] = complex_sum(left[idx], complex_mult(x, right[idx]));
        buffer[idx + half] = complex_sub(left[idx], complex_mult(x, right[idx]));
        x = complex_mult(x, w);
    });

    buffer
}

pub(crate) fn root_of_unity(len: isize) -> Vec2 {
    let angle = 2.0 * PI / len as f32;
    Vec2 {
        x: angle.cos(),
        y: angle.sin(),
    }
}

fn complex_mult(left: Vec2, right: Vec2) -> Vec2 {
    Vec2 {
        x: left.x * right.x - left.y * right.y,
        y: left.x * right.y + left.y * right.x,
    }
}

fn complex_sum(mut left: Vec2, right: Vec2) -> Vec2 {
    left.x += right.x;
    left.y += right.y;
    left
}

fn complex_sub(mut left: Vec2, right: Vec2) -> Vec2 {
    left.x -= right.x;
    left.y -= right.y;
    left
}

pub(crate) fn magnitude(complex: Vec2) -> f32 {
    (complex.x * complex.x + complex.y * complex.y).sqrt()
}

unsafe fn copy_from_box<T>(src: &Box<T>, dst: *mut T) {
    std::ptr::copy(src.as_ref(), dst, 1);
}

