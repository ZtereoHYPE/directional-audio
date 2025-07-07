#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused)]

pub(crate) mod fft;
mod hrtf;
pub mod transfer;
mod delay;

use crate::audio_engine::buffer_initializer::BufferInitializer;
use crate::audio_engine::gpu_structures::{DelayBuffer, DownloadBuffer, FftBuffer, FftUbo, GpuFrame, GpuWindow, InstanceBuffer, UploadBuffer, GPU_WINDOW_SIZE, MAX_DELAY_FRAMES, MAX_SOURCES};
use crate::audio_engine::signal_processor::delay::DelayModule;
use crate::audio_engine::signal_processor::fft::{FftModule, RADICES, RADIX_AMT};
use crate::audio_engine::signal_processor::hrtf::HrtfModule;
use crate::audio_engine::signal_processor::transfer::TransferModule;
use crate::audio_engine::{read_file_words, GpuData};
use crate::scene::Scene;
use crate::util::vec3;
use ash::vk::{Buffer, BufferCreateInfo, BufferUsageFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferLevel, CommandPoolCreateFlags, CommandPoolCreateInfo, ComputePipelineCreateInfo, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceSize, Extent3D, Fence, FenceCreateInfo, Filter, Format, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, MemoryPropertyFlags, PhysicalDevice, Pipeline, PipelineCache, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PushConstantRange, Queue, SampleCountFlags, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, SpecializationMapEntry, WriteDescriptorSet, WHOLE_SIZE};
use ash::{Device, Instance};
use crevice::std430::{AsStd430, Vec2, Vec3};
use std::f32::consts::PI;
use std::mem::transmute;
use std::rc::Rc;
use std::slice::from_ref;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateInfo};
use crate::scene::listener::AudioListener;
use crate::scene::source::{AudioSource, FRAME_SIZE};

#[repr(C)]
#[derive(Copy, Clone)]
struct SignalProcessorConstants {
    window_size: u32,
    frame_size: u32,
    filter_size: u32,
    delay_buffer_size: u32,
    pipelined_frames: u32,
    sampling_rate: f32,
    min_elevation: f32,
    max_elevation: f32
}

impl SignalProcessorConstants {
    const SIZE: usize = size_of::<SignalProcessorConstants>();

    // warning: this assumes that all fields are 4 in size
    fn get_entries(entries: &[u32]) -> Vec<SpecializationMapEntry> {
        entries
            .into_iter()
            .map(|&idx| SpecializationMapEntry::default()
                .constant_id(idx)
                .offset(4 * idx)
                .size(4)
            )
            .collect()
    }

    unsafe fn to_slice(&self) -> &[u8; Self::SIZE] {
        transmute(self)
    }
}

pub struct SignalProcessor {
    device: Device,
    buffer_allocator: Rc<Allocator>,

    transfer_queue: (Queue, u32),
    compute_queue: (Queue, u32),
    compute_command_buffer: CommandBuffer,
    transfer_command_buffer: CommandBuffer,

    instance_buffer: Buffer,
    instance_buffer_memory: Allocation,

    delay_module: DelayModule,
    fft_module: FftModule,
    hrtf_module: HrtfModule,
    transfer_module: TransferModule,

    counter: usize,
}

impl SignalProcessor {
    pub unsafe fn new(
        scene: &Scene,
        instance: &Instance,
        gpu: &PhysicalDevice,
        device: Device,
        transfer_queue: (Queue, u32),
        compute_queue: (Queue, u32),
        buffer_initializer: &mut BufferInitializer
    ) -> Self {
        let stages = FftModule::fft_stages(GPU_WINDOW_SIZE);
        let constants = SignalProcessorConstants {
            window_size: GPU_WINDOW_SIZE as u32,
            frame_size: FRAME_SIZE as u32,
            filter_size: GPU_WINDOW_SIZE as u32,
            delay_buffer_size: (MAX_DELAY_FRAMES * FRAME_SIZE) as u32,
            pipelined_frames: 0, // todo: do not hardcode these
            sampling_rate: 44100.0,
            min_elevation: PI,
            max_elevation: 0.0
        };

        let buffer_allocator = {
            let allocator_create_info = AllocatorCreateInfo::new(
                instance,
                &device,
                *gpu
            );

            Rc::new(
                Allocator::new(allocator_create_info)
                    .expect("Failed to create memory allocator")
            )
        };

        let (compute_command_pool, transfer_command_pool) = {
            let mut pool_create_info = CommandPoolCreateInfo::default()
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_queue.1);

            let compute = device
                .create_command_pool(&pool_create_info, None)
                .expect("Failed to create command pool");

            pool_create_info = pool_create_info.queue_family_index(transfer_queue.1);

            let transfer = device
                .create_command_pool(&pool_create_info, None)
                .expect("Failed to create command pool");

            (compute, transfer)
        };

        let (compute_command_buffer, transfer_command_buffer) = {
            let mut command_buffer_info = CommandBufferAllocateInfo::default()
                .command_pool(compute_command_pool)
                .command_buffer_count(1)
                .level(CommandBufferLevel::PRIMARY);

            let compute = device
                .allocate_command_buffers(&command_buffer_info)
                .expect("Failed to allocate command buffers")[0];

            command_buffer_info = command_buffer_info.command_pool(transfer_command_pool);

            let transfer = device
                .allocate_command_buffers(&command_buffer_info)
                .expect("Failed to allocate command buffers")[0];

            (compute, transfer)
        };

        let descriptor_pool = {
            let pool_sizes = [
                DescriptorPoolSize::default()
                    .ty(DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(4),

                DescriptorPoolSize::default()
                    .ty(DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(11),

                DescriptorPoolSize::default()
                    .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(2),
            ];

            let pool_info = DescriptorPoolCreateInfo::default()
                .max_sets(32)
                .pool_sizes(&pool_sizes);

            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let (mut instance_buffer, instance_buffer_memory) = {
            let buffer_info = BufferCreateInfo::default()
                .size(InstanceBuffer::max_size() as u64)
                .usage(BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST)
                .queue_family_indices(from_ref(&compute_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            let (buffer, mut memory) = buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer");

            (buffer, memory)
        };

        // Populate the instance buffer
        let mut data = InstanceBuffer::from_scene_data(scene);
        buffer_initializer.upload_buffer_onetime(&device, compute_queue.0.clone(), data, &mut instance_buffer);

        let fft_module = FftModule::new(
            buffer_allocator.clone(),
            buffer_initializer,
            device.clone(),
            compute_queue.clone(),
            descriptor_pool,
            stages,
            constants
        );

        let delay_module = DelayModule::new(
            buffer_allocator.clone(),
            buffer_initializer,
            device.clone(),
            compute_queue.clone(),
            descriptor_pool,
            instance_buffer,
            fft_module.starting_buffer(),
            constants,
        );

        let hrtf_module = HrtfModule::new(
            scene.listener.filter.clone(),
            buffer_allocator.clone(),
            buffer_initializer,
            device.clone(),
            compute_queue.clone(),
            descriptor_pool,
            instance_buffer,
            fft_module.starting_buffer(), // todo: this is hardcoded for the size
            constants
        );

        let transfer_module = TransferModule::new(
            buffer_allocator.clone(),
            device.clone(),
            transfer_queue,
            delay_module.delay_buffer(),
            hrtf_module.output_buffer()
        );

        Self {
            device,
            buffer_allocator,

            transfer_queue,
            compute_queue,
            compute_command_buffer,
            transfer_command_buffer,

            instance_buffer,
            instance_buffer_memory,

            delay_module,
            fft_module,
            hrtf_module,
            transfer_module,

            counter: 0
        }
    }

    pub unsafe fn process_frames(&mut self, listener: &AudioListener, sources: &mut Vec<AudioSource>, instance_amt: u32, last_rt_pos: Vec3) -> (GpuFrame, GpuFrame) {
        // transfer data to right buffer
        self.transfer_module
            .upload_new_frames(&mut self.compute_command_buffer, sources, self.counter)
            .expect("Failed to upload frames to the GPU!");

        // move the delayed windows to the fft buffer
        let camera_delta = vec3::sub(listener.location, last_rt_pos);
        self.delay_module.apply_delay(&mut self.compute_command_buffer, self.counter as u32, camera_delta, MAX_SOURCES);

        // perform fourier transform
        self.fft_module.gpu_fourier_transform(&mut self.compute_command_buffer, 0, false, MAX_SOURCES);

        // wipe the output buffer
        self.hrtf_module.wipe_output(&mut self.compute_command_buffer);

        // perform HRTF dsp
        self.hrtf_module.apply_hrtf(&mut self.compute_command_buffer, instance_amt);

        // transfer data back
        let (left_window, right_window) = self.transfer_module
            .download_windows(&mut self.compute_command_buffer)
            .expect("Failed to download frames from GPU!");

        self.counter += 1;

        let left = FftModule::local_fourier_transform(window_to_vec(left_window), true);
        let right = FftModule::local_fourier_transform(window_to_vec(right_window), true);
        let (start, end) = DownloadBuffer::last_frame_range();

        (
            GpuFrame::try_from(&left[start..end]).unwrap(),
            GpuFrame::try_from(&right[start..end]).unwrap(),
        )
    }

    pub(super) fn instance_buffer(&self) -> Buffer {
        self.instance_buffer
    }
}

impl Drop for SignalProcessor {
    fn drop(&mut self) {
        // todo: better cleanup procedure once this has stabilized
        unsafe {
            // for (alloc, buf) in self.fft_gpu_buffers_memory.iter_mut().zip(self.fft_gpu_buffers) {
            //     self.buffer_allocator.destroy_buffer(buf, alloc);
            // }
            //
            // for (alloc, buf) in self.fft_ubo_memories.iter_mut().zip(self.fft_ubos.clone()) {
            //     self.buffer_allocator.destroy_buffer(buf, alloc);
            // }
            //
            // self.buffer_allocator.destroy_buffer(self.instance_buffer, &mut self.instance_buffer_memory);
            // self.buffer_allocator.destroy_buffer(self.hrtf_output, &mut self.hrtf_output_memory);
            // self.buffer_allocator.destroy_image(self.hrtfs[0], &mut self.hrtf_memories[0]);
            // self.buffer_allocator.destroy_image(self.hrtfs[1], &mut self.hrtf_memories[1]);
        }
    }
}

pub(crate) unsafe fn window_to_vec(window: Box<GpuWindow>) -> Vec<Vec2> {
    // transmute the pointer without performing a copy; arrays in rust are guaranteed to be sequential
    let flat_window = transmute::<Box<GpuWindow>, Box<[Vec2; GPU_WINDOW_SIZE]>>(window);
    (flat_window as Box<[_]>).into_vec() // turn the box into a vector
}
