use crate::audio_engine::delay::DelayBufferData;
use crate::audio_engine::gpu_constants::{GPU_WINDOW_SIZE, MAX_DELAY_FRAMES, MAX_SOURCES, SLIDING_WINDOW_FRAME_AMT};
use crate::scene::source::{AudioSource, Frame, FRAME_SIZE};
use crate::vulkan::buffer::{InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use ash::prelude::VkResult;
use ash::vk::{AccessFlags, Buffer, BufferCopy, BufferUsageFlags, CommandBuffer, CommandBufferBeginInfo, CommandBufferResetFlags, CommandBufferUsageFlags, DependencyFlags, DeviceSize, Fence, FenceCreateInfo, MemoryBarrier, PipelineStageFlags, Queue, Semaphore, SemaphoreSignalInfo, SemaphoreWaitInfo, SubmitInfo};
use ash::Device;
use glam::Vec2;
use std::array::from_ref;
use std::error::Error;
use std::intrinsics::transmute;
use std::rc::Rc;
use std::sync::{mpsc, Arc};
use std::sync::mpsc::Sender;
use std::thread;
use std::thread::JoinHandle;
use vk_mem::Allocator;
use crate::audio_engine::{AudioSyncStage, GpuFrame, GpuWindow, RtSyncStage};
use crate::audio_engine::fft::FftModule;
use crate::audio_engine::rays::RtOutputBufferData;
use crate::VisualizationData;
use crate::vulkan::in_flight::{InFlight, InFlightCounter};
use crate::vulkan::queue::VulkanQueue;

type Task = Box<dyn Fn(&mut LocalVulkanBuffer<DownloadBufferData>) + Send>;

pub struct TransferModule {
    device: Device,
    queue: VulkanQueue,

    upload_buffer: LocalVulkanBuffer<UploadBufferData>,
    download_buffer_handle: Buffer,

    input_buffer_handle: Buffer,
    output_buffer_handles: InFlight<Buffer>,

    submit_thread: JoinHandle<()>,
    submit_queue: Sender<Task>,
}

impl TransferModule {
    pub unsafe fn new(
        allocator: Arc<Allocator>,
        device: Device,
        queue: VulkanQueue,
        frames_in_flight: usize,
        input_buffer: &VulkanBuffer<DelayBufferData>,
        output_buffer: &InFlight<VulkanBuffer<DownloadBufferData>>,
    ) -> TransferModule {
        let upload_buffer = LocalVulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC,
            allocator.clone()
        );
        
        let (submit_queue, submit_receiver) = mpsc::channel();
        let (handle_sender, handle_receiver) = mpsc::channel();

        let submit_thread = thread::spawn(move || {
            let mut download_buffer: LocalVulkanBuffer<DownloadBufferData> =
                LocalVulkanBuffer::new_inline(BufferUsageFlags::TRANSFER_DST, allocator.clone());
            
            handle_sender.send(download_buffer.handle());

            loop {
                let task: Task = submit_receiver.recv().unwrap();
                task(&mut download_buffer);
            }
        });

        let download_buffer_handle = handle_receiver.recv().unwrap();

        let output_buffer_handles = InFlight::create(
            frames_in_flight,
            |idx| output_buffer.0[idx].handle()
        );

        TransferModule {
            device,
            queue,
            upload_buffer,
            download_buffer_handle,
            input_buffer_handle: input_buffer.handle(),
            output_buffer_handles,
            submit_thread,
            submit_queue
        }
    }

    pub(super) unsafe fn upload_new_frames(&mut self, command_buffer: &mut CommandBuffer, sources: &mut Vec<AudioSource>, frame_counter: usize) -> VkResult<()> {
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

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.upload_buffer.handle(),
            self.input_buffer_handle,
            &regions[..]
        );

        for offset in clear_frames {
            self.device.cmd_fill_buffer(*command_buffer, self.input_buffer_handle, offset, (FRAME_SIZE * size_of::<Vec2>()) as DeviceSize, 0)
        }

        Ok(())
    }

    pub(super) unsafe fn download_windows(&mut self, command_buffer: &mut CommandBuffer, counter: InFlightCounter) {
        self.device.cmd_copy_buffer(
            *command_buffer,
            self.output_buffer_handles[counter],
            self.download_buffer_handle,
            DownloadBufferData::region()
        );
    }

    pub(super) fn schedule_submit_task(&mut self, task: Task) {
        self.submit_queue.send(task);
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