#![allow(unsafe_op_in_unsafe_fn)]

pub(crate) mod fft;
mod hrtf;
mod transfer;

use crate::audio::hrtf_filter::HrtfFilter;
use crate::audio::{FRAME_AMT, FRAME_SIZE};
use crate::vulkan::buffer_uploader::BufferUploader;
use crate::vulkan::signal_processor::fft::FftModule;
use crate::vulkan::signal_processor::hrtf::HrtfModule;
use crate::vulkan::signal_processor::transfer::TransferModule;
use crate::vulkan::{read_file_words, GpuData};
use ash::vk::{Buffer, BufferCreateInfo, BufferUsageFlags, BufferViewCreateInfo, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo, ComputePipelineCreateInfo, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, Extent3D, Fence, FenceCreateInfo, Filter, Format, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, MemoryPropertyFlags, PhysicalDevice, Pipeline, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, Queue, SampleCountFlags, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::{Device, Instance};
use crevice::std430::{AsStd430, Vec2, Vec4};
use std::array::from_ref;
use std::f32::consts::PI;
use std::intrinsics::transmute;
use std::rc::Rc;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateInfo};

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

// #[repr(C)]
// #[derive(Copy, Clone, Debug)]
// pub struct  {
pub(crate) type FftFrame = [Vec2; FRAME_SIZE]; // todo: this is not just for fft, should be GPU frame or smth
// }

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FftBuffer {
    pub frames: [FftFrame; FRAME_AMT]
}

struct HrtfUbo {
    metadata: Vec4,
}

impl GpuData for HrtfUbo {
    unsafe fn serialize(&self, dst: *mut u8) {
        std::ptr::copy((&self.metadata as *const Vec4).cast(), dst, size_of::<Vec4>());
    }

    unsafe fn deserialize(_: *const u8) -> Box<Self> {
        todo!("UBOs should not be deserialized!")
    }

    fn size(&self) -> usize {
        size_of::<Vec4>()
    }
}

pub struct SignalProcessor {
    device: Device,

    buffer_allocator: Rc<Allocator>,
    transfer_queue: (Queue, u32),
    compute_queue: (Queue, u32),
    compute_command_buffer: CommandBuffer,
    transfer_command_buffer: CommandBuffer,
    fence: Fence,

    // For FFT
    fft_gpu_buffers: [Buffer; 2],
    fft_gpu_buffers_memory: [Allocation; 2],
    // todo: switch to push constants
    fft_ubos: Vec<Buffer>,
    fft_ubo_memories: Vec<Allocation>,

    // for HRTF
    hrtf_ubo: Buffer,
    hrtf_ubo_memory: Allocation,
    hrtf_output: Buffer,
    hrtf_output_memory: Allocation,
    hrtfs: [Image; 2],
    hrtf_memories: [Allocation; 2],
    hrtf_views: [ImageView; 2],
    hrtf_sampler: Sampler,

    fft_module: FftModule,
    hrtf_module: HrtfModule,
    transfer_module: TransferModule,
}

// todo: better inspect mutability
impl SignalProcessor {
    pub unsafe fn new(
        instance: &Instance,
        gpu: &PhysicalDevice,
        device: Device,
        transfer_queue: (Queue, u32),
        compute_queue: (Queue, u32),
        buffer_uploader: &mut BufferUploader,
        filter: HrtfFilter
    ) -> Self {
        let stages = FftModule::fft_stages(FRAME_SIZE);

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
                    .descriptor_count(3),

                DescriptorPoolSize::default()
                    .ty(DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(8),

                DescriptorPoolSize::default()
                    .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(2),

                DescriptorPoolSize::default()
                    .ty(DescriptorType::STORAGE_TEXEL_BUFFER)
                    .descriptor_count(1),
            ];

            let pool_info = DescriptorPoolCreateInfo::default()
                .max_sets(32)
                .pool_sizes(&pool_sizes);

            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let (cpu_buffer, cpu_buffer_memory, cpu_buffer_map) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftBuffer>() as u64 * 2)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST)
                .queue_family_indices(from_ref(&transfer_queue.1))
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

        let (fft_ubos, fft_ubo_memories) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftUbo>() as u64)
                .usage(BufferUsageFlags::UNIFORM_BUFFER)
                .queue_family_indices(from_ref(&compute_queue.1))
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

                let inverse = false;

                // Populate the UBO
                let direction: f32 = if inverse { -1.0 } else { 1.0 };
                let normalization: f32 = if !inverse { 1.0  } else { 1.0 / stage.radix as f32}; // todo: i flipped these, make sure that was a right decision
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

        let ((fft_gpu_buf_1, fft_gpu_mem_1), (fft_gpu_buf_2, fft_gpu_mem_2)) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftBuffer>() as u64)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&compute_queue.1))
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

        let (fft_descriptor_sets, fft_descriptor_layout) = {
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

        let (fft_pipelines, fft_pipeline_layout) = {
            // The layout is the same for all pipelines
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&fft_descriptor_layout));

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
            let pipelines: [Pipeline; RADIX_AMT] = device
                .create_compute_pipelines(PipelineCache::null(), &pipeline_infos, None)
                .expect("Failed to create pipeline")
                .try_into().unwrap();

            (pipelines, layout)
        };

        let (mut hrtf_ubo, hrtf_ubo_memory) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<HrtfUbo>() as u64)
                .usage(BufferUsageFlags::STORAGE_TEXEL_BUFFER | BufferUsageFlags::TRANSFER_DST)
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

        // Populate the UBO
        let data = HrtfUbo {
            metadata: Vec4 {x:1.0, y:0.0, z:0.0, w: f32::from_bits(0)}
        };
        buffer_uploader.upload_buffer_onetime(&device, compute_queue.0.clone(), data, &mut hrtf_ubo);

        let (hrtf_output, hrtf_output_memory) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftFrame>() as u64 * 2)
                .usage(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER)
                .queue_family_indices(from_ref(&compute_queue.1))
                .sharing_mode(SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            buffer_allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let ((mut hrtf_left, hrtf_left_mem, hrtf_left_view), (mut hrtf_right, hrtf_right_mem, hrtf_right_view)) = {
            let image_info = ImageCreateInfo::default()
                .image_type(ImageType::TYPE_3D)
                .format(Format::R32_SFLOAT) // supported by 96.92% of devices
                .samples(SampleCountFlags::TYPE_1)
                .tiling(ImageTiling::OPTIMAL)
                .mip_levels(1)
                .array_layers(1)
                .extent(Extent3D {width: filter.options.azimuth_samples, height: filter.options.elevation_samples, depth: filter.filter_len as u32})
                .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED)
                .sharing_mode(SharingMode::EXCLUSIVE)
                .initial_layout(ImageLayout::UNDEFINED);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            println!("w{} h{} d{}", filter.options.azimuth_samples, filter.options.azimuth_samples, filter.filter_len);

            let (left, left_mem) = buffer_allocator
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create hrtf image");

            let (right, right_mem) = buffer_allocator
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create hrtf image");

            let subresource = ImageSubresourceRange::default()
                .aspect_mask(ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);

            let mut view_info = ImageViewCreateInfo::default()
                .image(left)
                .view_type(ImageViewType::TYPE_3D)
                .format(Format::R32_SFLOAT)
                .subresource_range(subresource);

            let left_view = device
                .create_image_view(&view_info, None)
                .expect("Failed to create image view");

            view_info = view_info.image(right);

            let right_view = device
                .create_image_view(&view_info, None)
                .expect("Failed to create image view");

            ((left, left_mem, left_view), (right, right_mem, right_view))
        };

        // Upload the HRTF data to the images
        let extent = Extent3D {
            width: filter.options.azimuth_samples,
            height: filter.options.elevation_samples,
            depth: filter.filter_len as u32,
        };

        buffer_uploader.upload_image_onetime(
            &device,
            compute_queue.0.clone(),
            filter.left,
            &mut hrtf_left,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            extent
        );

        buffer_uploader.upload_image_onetime(
            &device,
            compute_queue.0.clone(),
            filter.right,
            &mut hrtf_right,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            extent
        );

        let hrtf_sampler = {
            let sampler_info = SamplerCreateInfo::default()
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .compare_enable(false)
                .unnormalized_coordinates(false);

            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create sampler")
        };

        let (hrtf_descriptor_set, hrtf_descriptor_layout) = {
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_TEXEL_BUFFER) // todo: investigate UNIFORM_TEXEL_BUFFER
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
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(4)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(5)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
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

            let buffer_view_info = BufferViewCreateInfo::default()
                .buffer(hrtf_ubo)
                .format(Format::R32G32B32A32_SFLOAT)
                .range(WHOLE_SIZE);

            let buffer_view = device
                .create_buffer_view(&buffer_view_info, None)
                .expect("Failed to create buffer view");

            let buffer_infos = [
                DescriptorBufferInfo::default()
                    .buffer(fft_gpu_buf_2) // TODO: this is hardcoded for frame size 512!
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(hrtf_output)
                    .offset(0)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(hrtf_output)
                    .offset(size_of::<FftFrame>() as _)
                    .range(WHOLE_SIZE),
            ];

            let sampler_infos = [
                DescriptorImageInfo::default()
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hrtf_left_view)
                    .sampler(hrtf_sampler),

                DescriptorImageInfo::default()
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hrtf_right_view)
                    .sampler(hrtf_sampler),
            ];

            // Write the descriptor sets
            let writes = [
                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(0)
                    .descriptor_type(DescriptorType::STORAGE_TEXEL_BUFFER)
                    .texel_buffer_view(from_ref(&buffer_view)),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[0])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(2)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[1])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(3)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[2])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(4)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(from_ref(&sampler_infos[0])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(5)
                    .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(from_ref(&sampler_infos[1])),
            ];

            device.update_descriptor_sets(&writes, &[]);
            (set, set_layout)
        };

        let (hrtf_pipeline, hrtf_pipeline_layout) = {
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&hrtf_descriptor_layout));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/hrtf.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let stage_info = PipelineShaderStageCreateInfo::default()
                .stage(ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");

            let pipeline_info = ComputePipelineCreateInfo::default()
                .layout(layout)
                .stage(stage_info);

            let pipeline = device
                .create_compute_pipelines(PipelineCache::null(), from_ref(&pipeline_info), None)
                .expect("Failed to create pipeline")[0];

            (pipeline, layout)
        };

        let transfer_fence = device
            .create_fence(&FenceCreateInfo::default(), None)
            .expect("failed to create fence");

        let fence = device
            .create_fence(&FenceCreateInfo::default(), None)
            .expect("failed to create fence");

        let fft_module = FftModule::new(
            device.clone(),
            fft_pipelines,
            fft_pipeline_layout,
            fft_descriptor_sets,
            fft_descriptor_layout,
            compute_queue.0.clone(),
            stages
        );

        let hrtf_module = HrtfModule::new(
            device.clone(),
            hrtf_pipeline,
            hrtf_pipeline_layout,
            hrtf_descriptor_set,
            hrtf_descriptor_layout,
            compute_queue.0.clone(),
        );

        let transfer_module = TransferModule::new(
            buffer_allocator.clone(),
            device.clone(),
            transfer_queue.0.clone(),
            cpu_buffer,
            cpu_buffer_memory,
            cpu_buffer_map,
            transfer_fence
        );

        Self {
            device, // refcell

            buffer_allocator,
            transfer_queue, // problem
            compute_queue, // problem
            compute_command_buffer,
            transfer_command_buffer,
            fence,

            // cpu_buffer,
            // cpu_buffer_memory,
            // cpu_buffer_map,

            fft_gpu_buffers: [fft_gpu_buf_1, fft_gpu_buf_2],
            fft_gpu_buffers_memory: [fft_gpu_mem_1, fft_gpu_mem_2],
            fft_ubos,
            fft_ubo_memories,
            // fft_descriptor_sets,
            // fft_descriptor_layout,
            // fft_pipelines,
            // fft_pipeline_layout,

            hrtf_ubo,
            hrtf_ubo_memory,
            hrtf_output,
            hrtf_output_memory,
            hrtfs: [hrtf_left, hrtf_right],
            hrtf_memories: [hrtf_left_mem, hrtf_right_mem],
            hrtf_views: [hrtf_left_view, hrtf_right_view],
            hrtf_sampler,
            // hrtf_descriptor_set,
            // hrtf_descriptor_layout,
            // hrtf_pipeline,
            // hrtf_pipeline_layout,

            fft_module,
            hrtf_module,
            transfer_module,
        }
    }

    // todo: might switch to optional to remove a lot of these expects, in transfer.rs too.
    pub unsafe fn process_frame(&mut self, frame: Box<FftBuffer>) -> (Box<FftFrame>, Box<FftFrame>) {
        // transfer data to right buffer
        self.transfer_module.transfer_to_gpu(&mut self.compute_command_buffer, frame, &self.fft_gpu_buffers[0]);

        // perform fourier transform
        self.fft_module.gpu_fourier_transform(&mut self.compute_command_buffer, 0, false);

        // perform HRTF dsp
        self.hrtf_module.apply_hrtf(&mut self.compute_command_buffer);

        // transfer data back
        self.transfer_module.transfer_from_gpu(&mut self.compute_command_buffer, &self.hrtf_output)
    }

    pub unsafe fn gpu_fft(&mut self, frame: Box<FftFrame>) -> Box<FftFrame> {
        self.transfer_module.transfer_to_gpu(&mut self.compute_command_buffer, frame, &self.fft_gpu_buffers[0]);

        // todo: have much more variable size in all of these! require a size parameter, and use push constants to indicate it
        self.fft_module.gpu_fourier_transform(&mut self.compute_command_buffer, 0, false);

        let (left, _) = self.transfer_module.transfer_from_gpu(&mut self.compute_command_buffer, &self.fft_gpu_buffers[1]);

        left
    }
}

impl Drop for SignalProcessor {
    fn drop(&mut self) {
        // todo: better cleanup procedure once this has stabilized
        unsafe {
            for (alloc, buf) in self.fft_gpu_buffers_memory.iter_mut().zip(self.fft_gpu_buffers) {
                self.buffer_allocator.destroy_buffer(buf, alloc);
            }
            
            self.buffer_allocator.destroy_buffer(self.hrtf_ubo, &mut self.hrtf_ubo_memory);
            self.buffer_allocator.destroy_image(self.hrtfs[0], &mut self.hrtf_memories[0]);
            self.buffer_allocator.destroy_image(self.hrtfs[1], &mut self.hrtf_memories[1]);
        }
    }
}
