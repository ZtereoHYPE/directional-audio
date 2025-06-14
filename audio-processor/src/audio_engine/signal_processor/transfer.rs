use crate::audio_engine::gpu_structures::{DownloadBuffer, GpuWindow, UploadBuffer, MAX_DELAY_FRAMES, MAX_SOURCES};
use crate::audio_engine::GpuData;
use crate::scene::source::FRAME_SIZE;
use crate::scene::Scene;
use ash::vk::{AccessFlags, Buffer, BufferCopy, CommandBuffer, CommandBufferBeginInfo, CommandBufferResetFlags, CommandBufferUsageFlags, DependencyFlags, DeviceSize, Fence, MemoryBarrier, PipelineStageFlags, Queue, SubmitInfo, WHOLE_SIZE};
use ash::Device;
use crevice::std430::Vec2;
use std::array::from_ref;
use std::rc::Rc;
use std::u64::MAX;
use vk_mem::{Allocation, Allocator};

pub struct TransferModule {
    allocator: Rc<Allocator>,
    device: Device,
    queue: Queue,

    cpu_buffers: [Buffer; 2],
    cpu_buffer_memories: [Allocation; 2],
    cpu_buffer_maps: [*mut u8; 2],

    fence: Fence,
}

impl TransferModule {
    pub unsafe fn new(
        allocator: Rc<Allocator>,
        device: Device,
        queue: Queue,
        cpu_buffers: [Buffer; 2],
        cpu_buffer_memories: [Allocation; 2],
        cpu_buffer_maps: [*mut u8; 2],
        fence: Fence,
    ) -> TransferModule {
        TransferModule {
            allocator,
            device,
            queue,
            cpu_buffers,
            cpu_buffer_memories,
            cpu_buffer_maps,
            fence,
        }
    }

    // todo: might switch to optional to remove a lot of these expects, in other modules too.
    pub unsafe fn upload_new_frames(&mut self, command_buffer: &mut CommandBuffer, scene: &mut Scene, dst: &Buffer, frame_counter: usize) {
        let mut regions = vec![];
        let mut clear_frames: Vec<DeviceSize> = vec![];
        
        let mut sources = &mut scene.sources;
        let delay_buffer_offset = ((frame_counter + MAX_DELAY_FRAMES - 1) % MAX_DELAY_FRAMES) * FRAME_SIZE * size_of::<Vec2>();

        // copy frame to cpu buffer, keeping track of which regions are actually updated
        let mut stream_buffer = UploadBuffer::from_memory_map(self.cpu_buffer_maps[0]);
        for idx in 0..MAX_SOURCES {
            let destination = idx * MAX_DELAY_FRAMES * FRAME_SIZE * size_of::<Vec2>() + delay_buffer_offset;
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

        self.allocator
            .flush_allocation(&self.cpu_buffer_memories[0], 0, WHOLE_SIZE)
            .expect("Failed to flush the cpu buffer allocation");

        self.device
            .reset_command_buffer(*command_buffer, CommandBufferResetFlags::empty())
            .expect("Failed to reset command buffer");

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device
            .begin_command_buffer(*command_buffer, &begin_info)
            .expect("Failed to begin command buffer recording");

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.cpu_buffers[0],
            *dst,
            &regions[..]
        );

        for offset in clear_frames {
            self.device.cmd_fill_buffer(*command_buffer, *dst, offset, (FRAME_SIZE * size_of::<Vec2>()) as DeviceSize, 0)
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
    }

    pub unsafe fn download_upload_buf(&mut self, command_buffer: &mut CommandBuffer, src: &Buffer) -> Box<UploadBuffer> {
        self.device
            .reset_fences(from_ref(&self.fence))
            .expect("Failed to reset fence");

        let region = BufferCopy::default()
            .size(UploadBuffer::max_size() as _);

        self.device.cmd_copy_buffer(
            *command_buffer,
            *src,
            self.cpu_buffers[0],
            from_ref(&region)
        );

        self.device
            .end_command_buffer(*command_buffer)
            .expect("Failed to end command buffer!");

        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(command_buffer));

        self.device
            .queue_submit(self.queue, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");

        self.device
            .wait_for_fences(from_ref(&self.fence), true, MAX)
            .expect("Failed to wait for fence!");

        self.allocator
            .invalidate_allocation(&self.cpu_buffer_memories[0], 0, WHOLE_SIZE)
            .expect("Failed to invalidate allocation");

        copy_to_box(self.cpu_buffer_maps[0] as *const UploadBuffer)
    }

    // todo: this could be made a bit more efficient if only the relevant part of the frame is copied. This would involve perfoming the FFT here.
    pub unsafe fn download_windows(&mut self, command_buffer: &mut CommandBuffer, src: &Buffer) -> (Box<GpuWindow>, Box<GpuWindow>) {
        self.device
            .reset_fences(from_ref(&self.fence))
            .expect("Failed to reset fence");
    
        let region = BufferCopy::default()
            .size(size_of::<GpuWindow>() as u64 * 2);
    
        self.device.cmd_copy_buffer(
            *command_buffer,
            *src,
            self.cpu_buffers[1],
            from_ref(&region)
        );
    
        self.device
            .end_command_buffer(*command_buffer)
            .expect("Failed to end command buffer!");
    
        let submit_info = SubmitInfo::default()
            .command_buffers(from_ref(command_buffer));
    
        self.device
            .queue_submit(self.queue, &[submit_info], self.fence)
            .expect("Failed to submit command buffer");
    
        self.device
            .wait_for_fences(from_ref(&self.fence), true, u64::MAX)
            .expect("Failed to wait for fence!");
    
        self.allocator
            .invalidate_allocation(&self.cpu_buffer_memories[1], 0, WHOLE_SIZE)
            .expect("Failed to invalidate allocation");

        DownloadBuffer::from_memory_map(self.cpu_buffer_maps[1])
            .get_windows()
    }
}

impl Drop for TransferModule {
    fn drop(&mut self) {
        unsafe {
            self.allocator.destroy_buffer(self.cpu_buffers[0], &mut self.cpu_buffer_memories[0]);
            self.allocator.destroy_buffer(self.cpu_buffers[1], &mut self.cpu_buffer_memories[1]);
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

pub(crate) unsafe fn copy_from_box<T>(src: &Box<T>, dst: *mut T) {
    std::ptr::copy(src.as_ref(), dst, 1);
}
