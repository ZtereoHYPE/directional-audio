use crate::audio_engine::gpu_constants::{MAX_DELAY_FRAMES, MAX_SOURCES, SLIDING_WINDOW_FRAME_AMT};
use crate::audio_engine::signal_processor::delay::DelayBufferData;
use crate::audio_engine::{GpuFrame, GpuWindow};
use crate::scene::source::{AudioSource, FRAME_SIZE};
use crate::vulkan::buffer::{InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use ash::prelude::VkResult;
use ash::vk::{AccessFlags, Buffer, BufferCopy, BufferUsageFlags, CommandBuffer, CommandBufferBeginInfo, CommandBufferResetFlags, CommandBufferUsageFlags, DependencyFlags, DeviceSize, Fence, FenceCreateInfo, MemoryBarrier, PipelineStageFlags, Queue, SubmitInfo};
use ash::Device;
use glam::Vec2;
use std::array::from_ref;
use std::error::Error;
use std::rc::Rc;
use vk_mem::Allocator;

pub struct TransferModule {
    allocator: Rc<Allocator>,
    device: Device,
    queue: Queue,
    fence: Fence,

    upload_buffer: LocalVulkanBuffer<UploadBufferData>,
    download_buffer: LocalVulkanBuffer<DownloadBufferData>,

    input_buffer_handle: Buffer, // do NOT free these, they are not this module's responsibility
    output_buffer_handle: Buffer,
}

impl TransferModule {
    pub unsafe fn new(
        allocator: Rc<Allocator>,
        device: Device,
        queue: (Queue, u32),
        input_buffer: &VulkanBuffer<DelayBufferData>,
        output_buffer: &VulkanBuffer<DownloadBufferData>,
    ) -> TransferModule {
        let upload_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC,
            allocator.clone()
        );
        
        let download_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_DST,
            allocator.clone()
        );

        let fence = device
            .create_fence(&FenceCreateInfo::default(), None)
            .expect("failed to create fence");

        TransferModule {
            allocator,
            device,
            queue: queue.0,
            upload_buffer,
            download_buffer,
            fence,
            input_buffer_handle: input_buffer.handle(),
            output_buffer_handle: output_buffer.handle(),
        }
    }

    pub unsafe fn upload_new_frames(&mut self, command_buffer: &mut CommandBuffer, sources: &mut Vec<AudioSource>, frame_counter: usize) -> VkResult<()> {
        let mut regions = vec![];
        let mut clear_frames: Vec<DeviceSize> = vec![];
        
        // let mut sources = &mut scene.sources;
        let delay_buffer_offset = (frame_counter % MAX_DELAY_FRAMES) * FRAME_SIZE * size_of::<Vec2>() ; // start at 0

        // copy frame to cpu buffer, keeping track of which regions are actually updated
        let mut stream_buffer = self.upload_buffer.buffer_data();
        for idx in 0..MAX_SOURCES {
            let destination = idx * (MAX_DELAY_FRAMES * FRAME_SIZE * size_of::<Vec2>()) + delay_buffer_offset;
            let mut has_data = false;
            if (idx < sources.len()) {
                has_data = sources[idx].provider.next_frame(&mut stream_buffer.frames[idx]);
            }

            if (has_data) {
                regions.push(
                    BufferCopy::default()
                        .size((FRAME_SIZE * size_of::<Vec2>()) as DeviceSize)
                        .src_offset((idx * FRAME_SIZE * size_of::<Vec2>()) as DeviceSize)
                        .dst_offset(destination as DeviceSize)
                );
            } else {
                clear_frames.push(destination as DeviceSize)
            }
        }

        self.upload_buffer.flush();

        self.device.reset_command_buffer(*command_buffer, CommandBufferResetFlags::empty())?;

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(*command_buffer, &begin_info)?;

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.upload_buffer.handle(),
            self.input_buffer_handle,
            &regions[..]
        );

        for offset in clear_frames {
            self.device.cmd_fill_buffer(*command_buffer, self.input_buffer_handle, offset, (FRAME_SIZE * size_of::<Vec2>()) as DeviceSize, 0)
        }

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::TRANSFER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::TRANSFER, // wait for all transfer commands so far...
            PipelineStageFlags::COMPUTE_SHADER, // ...before executing any compute from now on
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        );

        Ok(())
    }

    pub unsafe fn download_windows(&mut self, command_buffer: &mut CommandBuffer) -> VkResult<(Box<GpuWindow>, Box<GpuWindow>)> {
        self.device.reset_fences(from_ref(&self.fence))?;

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.output_buffer_handle,
            self.download_buffer.handle(),
            DownloadBufferData::region()
        );
    
        self.device.end_command_buffer(*command_buffer)?;

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(command_buffer));
    
        self.device.queue_submit(self.queue, &[submit_info], self.fence)?;
        self.device.wait_for_fences(from_ref(&self.fence), true, u64::MAX)?;
        self.download_buffer.invalidate();

        Ok(self.download_buffer.buffer_data().get_windows())
    }
}

impl Drop for TransferModule {
    fn drop(&mut self) {
        // free the other objects
    }
}

pub(crate) unsafe fn copy_to_box<T>(mem: *const T) -> Box<T> {
    // Null Check
    assert!(!mem.is_null(), "Input pointer must not be null");

    // Alignment Check
    assert_eq!(mem as usize % align_of::<T>(), 0, "Input pointer must be properly aligned");

    // Allocate the required space
    let layout = std::alloc::Layout::new::<T>();
    let ptr = std::alloc::alloc_zeroed(layout) as *mut T;

    // Copy the memory value
    std::ptr::copy(mem, ptr, 1);

    // Wrap in a box
    Box::from_raw(ptr)
}

pub(crate) unsafe fn copy_from_box<T>(src: &Box<T>, dst: *mut T) {
    std::ptr::copy(src.as_ref(), dst, 1);
}

/// The upload buffer contains the stream data that gets uploaded to the GPU every frame.
#[repr(C)]
pub(crate) struct UploadBufferData {
    pub frames: [GpuFrame; MAX_SOURCES]
}
impl InlineBufferData for UploadBufferData {}

#[repr(C)]
pub(crate) struct DownloadBufferData {
    pub windows: [GpuWindow; 2]
}

impl InlineBufferData for DownloadBufferData {}

impl DownloadBufferData {
    pub(crate) unsafe fn get_windows(&self) -> (Box<GpuWindow>, Box<GpuWindow>) { unsafe {
        let left = copy_to_box(&self.windows[0] as *const GpuWindow);
        let right = copy_to_box(&self.windows[1] as *const GpuWindow);

        (left, right)
    }}

    pub(crate) const fn last_frame_range() -> (usize, usize) {
        (
            FRAME_SIZE * (SLIDING_WINDOW_FRAME_AMT - 1),
            FRAME_SIZE * SLIDING_WINDOW_FRAME_AMT,
        )
    }
}