use crate::audio_engine::gpu_constants::{GPU_WINDOW_SIZE, MAX_INSTANCES};
use crate::audio_engine::signal_processor::fft::FftBufferData;
use crate::audio_engine::signal_processor::transfer::DownloadBufferData;
use crate::audio_engine::signal_processor::SignalProcessorConstants;
use crate::audio_engine::{read_file_words, GpuWindow, InstanceBufferData};
use crate::scene::listener::hrtf_filter::HrtfFilter;
use crate::util::{workgroup_div, AsBytes};
use crate::vulkan::buffer::{VulkanBuffer};
use crate::vulkan::buffer_initializer::{BufferInitializer, InitMode};
use ash::vk::{AccessFlags, BufferUsageFlags, CommandBuffer, ComputePipelineCreateInfo, DependencyFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, Extent3D, Filter, Format, ImageAspectFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageViewCreateInfo, ImageViewType, MemoryBarrier, Pipeline, PipelineBindPoint, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PushConstantRange, Queue, SampleCountFlags, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SpecializationInfo, WriteDescriptorSet, WHOLE_SIZE};
use ash::Device;
use std::array::from_ref;
use std::rc::Rc;
use vk_mem::{Alloc, Allocator};

// todo: rename to DspModule because it performs more than just HRTF (attenuation)
pub struct HrtfModule {
    device: Device,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    descriptor_set: DescriptorSet,
    descriptor_set_layout: DescriptorSetLayout,
    queue: Queue,

    pub(super) output_buffer: VulkanBuffer<DownloadBufferData>,
}

impl HrtfModule {
    pub(super) unsafe fn new(
        filter: HrtfFilter,
        allocator: Rc<Allocator>,
        initializer: &mut BufferInitializer,
        device: Device,
        queue: (Queue, u32),
        descriptor_pool: DescriptorPool,
        instance_buffer: &VulkanBuffer<InstanceBufferData>,
        fft_ending_buffer: &VulkanBuffer<FftBufferData>,
        constants: SignalProcessorConstants,
    ) -> Self {
        let mut output_buffer = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
            allocator.clone()
        );

        initializer.init_buffer(&mut output_buffer, InitMode::Zeroed, queue, &device);

        let ((mut hrtf_left, hrtf_left_mem, hrtf_left_view), (mut hrtf_right, hrtf_right_mem, hrtf_right_view)) = {
            let image_info = ImageCreateInfo::default()
                .image_type(ImageType::TYPE_3D)
                .format(Format::R32G32B32A32_SFLOAT) // supported by 96.92% of devices
                .samples(SampleCountFlags::TYPE_1)
                .tiling(ImageTiling::OPTIMAL)
                .mip_levels(1)
                .array_layers(1)
                .extent(Extent3D {width: filter.filter_len as u32, height: filter.options.elevation_samples, depth: filter.options.azimuth_samples})
                .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED)
                .sharing_mode(SharingMode::EXCLUSIVE)
                .initial_layout(ImageLayout::UNDEFINED);

            let allocation_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            let (left, left_mem) = allocator
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create hrtf image");

            let (right, right_mem) = allocator
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create hrtf image");

            let subresource = ImageSubresourceRange::default()
                .aspect_mask(ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);

            let mut view_info = ImageViewCreateInfo::default()
                .image(left)
                .view_type(ImageViewType::TYPE_3D)
                .format(Format::R32G32B32A32_SFLOAT)
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
            width: filter.filter_len as u32,
            height: filter.options.elevation_samples,
            depth: filter.options.azimuth_samples,
        };

        initializer.init_image(
            &device,
            queue.0.clone(),
            Box::new(filter.left),
            &mut hrtf_left,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            extent
        );

        initializer.init_image(
            &device,
            queue.0.clone(),
            Box::new(filter.right),
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

        let (descriptor_set, descriptor_set_layout) = {
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
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

            let buffer_infos = [
                DescriptorBufferInfo::default()
                    .buffer(instance_buffer.handle())
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(fft_ending_buffer.handle()) // TODO: this is hardcoded for window size 2048!
                    .range(WHOLE_SIZE),

                DescriptorBufferInfo::default()
                    .buffer(output_buffer.handle())
                    .offset(0)
                    .range(size_of::<GpuWindow>() as _),

                DescriptorBufferInfo::default()
                    .buffer(output_buffer.handle())
                    .offset(size_of::<GpuWindow>() as _)
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
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[0])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[1])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(2)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[2])),

                WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_count(1)
                    .dst_binding(3)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(from_ref(&buffer_infos[3])),

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

        let (pipeline, pipeline_layout) = {
            let push_constant_range = PushConstantRange::default()
                .stage_flags(ShaderStageFlags::COMPUTE)
                .size(4);

            let layout_info = PipelineLayoutCreateInfo::default()
                .push_constant_ranges(from_ref(&push_constant_range))
                .set_layouts(from_ref(&descriptor_set_layout));

            let layout = device
                .create_pipeline_layout(&layout_info, None)
                .expect("Failed to create pipeline layout");

            let code_words = read_file_words("target/shaders/hrtf.comp.spv");

            let shader_module_info = ShaderModuleCreateInfo::default()
                .code(&code_words[..]);

            let shader_module = device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module");

            let specialization_entries =
                SignalProcessorConstants::get_entries(&[0, 1, 2, 5, 6, 7]); // select these constants

            let specialization_info = SpecializationInfo::default()
                .map_entries(&specialization_entries)
                .data(constants.to_slice());

            let stage_info = PipelineShaderStageCreateInfo::default()
                .stage(ShaderStageFlags::COMPUTE)
                .specialization_info(&specialization_info)
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

        Self {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set,
            descriptor_set_layout,
            queue: queue.0,

            output_buffer
        }
    }

    pub(super) unsafe fn apply_hrtf(&mut self, command_buffer: &mut CommandBuffer, instance_amt: u32) {
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

        if (MAX_INSTANCES > 64) {
            panic!("Currently maximum 64 sources are supported!");
        }

        let workgroups = (GPU_WINDOW_SIZE as u32 / 2 + 1, workgroup_div(instance_amt, 64));

        self.device.cmd_push_constants(*command_buffer, self.pipeline_layout, ShaderStageFlags::COMPUTE, 0, instance_amt.as_bytes());

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

    pub(super) unsafe fn wipe_output(&mut self, command_buffer: &mut CommandBuffer) {
        self.device.cmd_fill_buffer(*command_buffer, self.output_buffer.handle(), 0, size_of::<DownloadBufferData>() as _, 0);
    }
}
