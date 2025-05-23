#![allow(unsafe_op_in_unsafe_fn)]

use std::{array::from_ref, u64::MAX};

use ash::vk::{AccessFlags, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, BufferViewCreateInfo, CommandBufferBeginInfo, CommandBufferResetFlags, CommandBufferUsageFlags, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, Extent3D, Fence, FenceCreateInfo, Filter, Format, Image, ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageSubresource, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, MemoryBarrier, MemoryPropertyFlags, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, SampleCountFlags, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet, WHOLE_SIZE};
use crevice::std430::Vec4;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateInfo};

use crate::{audio::{hrtf::HrtfFilter, FRAME_AMT}, vulkan::engine::DescriptorRequirement, FRAME_SIZE};

use super::{engine::{GpuData, VulkanContext, VulkanModule}, fft::{copy_to_box, FftFrame}, read_file_words};

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

pub struct HrtfModule {
    buffer_allocator: Allocator,

    ubo: Buffer,
    ubo_memory: Allocation,

    outputs: [Buffer; 2],
    output_memories: [Allocation; 2],

    hrtfs: [Image; 2],
    hrtf_memories: [Allocation; 2],
    hrtf_views: [ImageView; 2],

    cpu_buffer: Buffer, 
    cpu_buffer_memory: Allocation, 
    cpu_buffer_map: *mut u8,

    sampler: Sampler,

    descriptor_set: DescriptorSet,
    descriptor_layout: DescriptorSetLayout,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    readback_fence: Fence
}

impl VulkanModule for HrtfModule {
    fn descriptors() -> Vec<DescriptorRequirement> {
        vec![
            DescriptorRequirement {
                ttype: DescriptorType::STORAGE_TEXEL_BUFFER,
                amount: 1
            },
            DescriptorRequirement {
                ttype: DescriptorType::STORAGE_BUFFER,
                amount: 3
            },
            DescriptorRequirement {
                ttype: DescriptorType::SAMPLED_IMAGE,
                amount: 2
            },
        ]
    }
}

impl HrtfModule {
    pub unsafe fn new(ctx: &mut VulkanContext, filter: HrtfFilter, frames_buf: &Buffer) -> Self {
        let buffer_allocator = {
            let allocator_create_info = AllocatorCreateInfo::new(
                &ctx.instance, 
                &ctx.device,
                ctx.gpu
            );

            vk_mem::Allocator::new(allocator_create_info)
                .expect("Failed to create memory allocator")
        };

        let (mut ubo, ubo_memory) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<HrtfUbo>() as u64)
                .usage(BufferUsageFlags::STORAGE_TEXEL_BUFFER | BufferUsageFlags::TRANSFER_DST)
                .queue_family_indices(from_ref(&ctx.compute_queue.1))
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
        ctx.buffer_uploader.upload_buffer_onetime(&ctx.device, ctx.compute_queue.0.clone(), data, &mut ubo);

        let ((output_left, output_left_mem), (output_right, output_right_mem)) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftFrame>() as u64)
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

            let left_view = ctx.device
                .create_image_view(&view_info, None)
                .expect("Failed to create image view");

            view_info.image(right);

            let right_view = ctx.device
                .create_image_view(&view_info, None)
                .expect("Failed to create image view");

            ((left, left_mem, left_view), (right, right_mem, right_view))
        };

        let (cpu_buffer, cpu_buffer_memory, cpu_buffer_map) = {
            let buffer_info = BufferCreateInfo::default()
                .size(size_of::<FftFrame>() as u64 * 2)
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
        
        // Upload the HRTF data to the images
        let extent = Extent3D {
            width: filter.options.azimuth_samples,
            height: filter.options.elevation_samples,
            depth: filter.filter_len as u32,
        };
        ctx.buffer_uploader.upload_image_onetime(&ctx.device, ctx.compute_queue.0.clone(), filter.left, &mut hrtf_left, ImageLayout::SHADER_READ_ONLY_OPTIMAL, extent);
        ctx.buffer_uploader.upload_image_onetime(&ctx.device, ctx.compute_queue.0.clone(), filter.right, &mut hrtf_right, ImageLayout::SHADER_READ_ONLY_OPTIMAL, extent);

        let sampler = {
            let sampler_info = SamplerCreateInfo::default()
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .address_mode_u(SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(SamplerAddressMode::CLAMP_TO_BORDER)
                .anisotropy_enable(false)
                .compare_enable(false)
                .unnormalized_coordinates(false);

            ctx.device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create sampler")
        };

        let (descriptor_set, descriptor_layout) = {
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
                    .descriptor_type(DescriptorType::SAMPLED_IMAGE)
                    .stage_flags(ShaderStageFlags::COMPUTE),

                DescriptorSetLayoutBinding::default()
                    .binding(5)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::SAMPLED_IMAGE)
                    .stage_flags(ShaderStageFlags::COMPUTE),
            ];

            let set_layout_info = DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            let set_layout = ctx.device
                .create_descriptor_set_layout(&set_layout_info, None)
                .expect("Failed to create descriptor set layout");

            // Allocate one set per stage
            let set_info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(ctx.descriptor_pool)
                .set_layouts(from_ref(&set_layout));
            
            let set = ctx.device
                .allocate_descriptor_sets(&set_info)
                .expect("Failed to allocate descriptor sets")[0];

            let buffer_view_info = BufferViewCreateInfo::default()
                .buffer(ubo)
                .format(Format::R32G32B32A32_SFLOAT)
                .range(WHOLE_SIZE);

            let buffer_view = ctx.device
                .create_buffer_view(&buffer_view_info, None)
                .expect("Failed to create buffer view");

            let buffer_infos = [
                DescriptorBufferInfo::default()
                    .buffer(*frames_buf)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(output_left)
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(output_right)
                    .range(WHOLE_SIZE),
            ];

            let sampler_infos = [
                DescriptorImageInfo::default()
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hrtf_left_view)
                    .sampler(sampler),

                DescriptorImageInfo::default()
                    .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hrtf_right_view)
                    .sampler(sampler),
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
                    .descriptor_type(DescriptorType::SAMPLED_IMAGE)
                    .image_info(from_ref(&sampler_infos[0])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(5)
                    .descriptor_type(DescriptorType::SAMPLED_IMAGE)
                    .image_info(from_ref(&sampler_infos[1])),
            ];

            ctx.device.update_descriptor_sets(&writes, &[]);
            (set, set_layout)
        };

        let (pipeline, pipeline_layout) = {
            let layout_info = PipelineLayoutCreateInfo::default()
                .set_layouts(from_ref(&descriptor_layout));

            let layout = ctx.device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/hrtf.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = ctx.device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let stage_info = PipelineShaderStageCreateInfo::default()
                        .stage(ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(c"main");
    
            let pipeline_info = ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(stage_info);

            let pipeline = ctx.device
                .create_compute_pipelines(PipelineCache::null(), from_ref(&pipeline_info), None)
                .expect("Failed to create pipeline")[0];
            
            (pipeline, layout)
        };

        let readback_fence = ctx.device
            .create_fence(&FenceCreateInfo::default(), None)
            .expect("failed to create fence");

        Self {
            buffer_allocator,
            ubo,
            ubo_memory,
            outputs: [output_left, output_right],
            output_memories: [output_left_mem, output_right_mem],
            hrtfs: [hrtf_left, hrtf_right],
            hrtf_memories: [hrtf_left_mem, hrtf_right_mem],
            hrtf_views: [hrtf_left_view, hrtf_right_view],
            cpu_buffer,
            cpu_buffer_memory,
            cpu_buffer_map,
            sampler,
            descriptor_set,
            descriptor_layout,
            pipeline,
            pipeline_layout,
            readback_fence
        }
    }

    pub unsafe fn apply(&self, ctx: &mut VulkanContext) -> (Box<FftFrame>, Box<FftFrame>) {
        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        ctx.device
            .reset_command_buffer(ctx.command_buffer, CommandBufferResetFlags::empty())
            .expect("Failed to reset command buffer");

        ctx.device
            .begin_command_buffer(ctx.command_buffer, &begin_info)
            .expect("Failed to begin command buffer recording");

        ctx.device.cmd_bind_descriptor_sets(
            ctx.command_buffer, 
            PipelineBindPoint::COMPUTE, 
            self.pipeline_layout, 
            0, 
            from_ref(&self.descriptor_set), 
            &[]
        );

        ctx.device.cmd_bind_pipeline(
            ctx.command_buffer, 
            PipelineBindPoint::COMPUTE, 
            self.pipeline
        );

        let workgroups = (FRAME_SIZE as u32 / 64, FRAME_AMT as u32);

        ctx.device.cmd_dispatch(ctx.command_buffer, workgroups.0, workgroups.1, 1);

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any shader read caches

        ctx.device.cmd_pipeline_barrier(
            ctx.command_buffer, 
            PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
            PipelineStageFlags::TRANSFER, // ...before executing any transfers from now on
            DependencyFlags::empty(), 
            from_ref(&memory_barrier), 
            &[], 
            &[]
        );

        let region = BufferCopy::default()
            .size(size_of::<FftFrame>() as _);

        ctx.device.cmd_copy_buffer(
            ctx.command_buffer, 
            self.outputs[0],
            self.cpu_buffer, 
            from_ref(&region)
        );

        let region = BufferCopy::default()
            .size(size_of::<FftFrame>() as _)
            .dst_offset(size_of::<FftFrame>() as u64);

        ctx.device.cmd_copy_buffer(
            ctx.command_buffer,
            self.outputs[1],
            self.cpu_buffer,
            from_ref(&region)
        );

        ctx.device.end_command_buffer(ctx.command_buffer);
        
        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(&ctx.command_buffer));

        ctx.device
            .queue_submit(ctx.compute_queue.0, &[submit_info], self.readback_fence)
            .expect("Failed to submit command buffer");

        ctx.device.wait_for_fences(from_ref(&self.readback_fence), true, MAX);

        // readback result
        self.buffer_allocator.invalidate_allocation(&self.cpu_buffer_memory, 0, WHOLE_SIZE);

        let left = copy_to_box(self.cpu_buffer_map as *const FftFrame);
        let right = copy_to_box(self.cpu_buffer_map.offset(size_of::<FftFrame>() as isize) as *const FftFrame);

        (left, right)
    }
}

impl Drop for HrtfModule {
    fn drop(&mut self) {
        unsafe {
            self.buffer_allocator.destroy_buffer(self.ubo, &mut self.ubo_memory);
            self.buffer_allocator.destroy_buffer(self.outputs[0], &mut self.output_memories[0]);
            self.buffer_allocator.destroy_buffer(self.outputs[1], &mut self.output_memories[1]);
            self.buffer_allocator.destroy_image(self.hrtfs[0], &mut self.hrtf_memories[0]);
            self.buffer_allocator.destroy_image(self.hrtfs[1], &mut self.hrtf_memories[1]);
        }
    }
}
