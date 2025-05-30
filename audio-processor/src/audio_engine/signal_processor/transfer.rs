use crate::audio_engine::gpu_structures::{GpuFrame, GpuWindow, StreamBuffer};
use crate::audio_engine::GpuData;
use ash::vk::{AccessFlags, Buffer, BufferCopy, CommandBuffer, CommandBufferBeginInfo, CommandBufferResetFlags, CommandBufferUsageFlags, DependencyFlags, Fence, MemoryBarrier, PipelineStageFlags, Queue, SubmitInfo, WHOLE_SIZE};
use ash::Device;
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
    pub unsafe fn upload_new_frames(&mut self, command_buffer: &mut CommandBuffer, frames: Vec<GpuFrame>, dst: &Buffer) {
        // copy frame to cpu buffer
        let mut stream_buffer = StreamBuffer::from_memory_map(self.cpu_buffer_maps[0]);
        stream_buffer.insert_frames(frames.clone());

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

        let region = BufferCopy::default()
            .size(StreamBuffer::size() as _);

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.cpu_buffers[0],
            *dst,
            from_ref(&region)
        );

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

    pub unsafe fn download_stream_buf(&mut self, command_buffer: &mut CommandBuffer, src: &Buffer) -> Box<StreamBuffer> {
        self.device
            .reset_fences(from_ref(&self.fence))
            .expect("Failed to reset fence");

        let region = BufferCopy::default()
            .size((size_of::<GpuWindow>() * 2) as _);

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
            .wait_for_fences(from_ref(&self.fence), true, MAX)
            .expect("Failed to wait for fence!");

        self.allocator
            .invalidate_allocation(&self.cpu_buffer_memories[1], 0, WHOLE_SIZE)
            .expect("Failed to invalidate allocation");

        copy_to_box(self.cpu_buffer_maps[1] as *const StreamBuffer)
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
            .wait_for_fences(from_ref(&self.fence), true, MAX)
            .expect("Failed to wait for fence!");
    
        self.allocator
            .invalidate_allocation(&self.cpu_buffer_memories[1], 0, WHOLE_SIZE)
            .expect("Failed to invalidate allocation");
    
        let left = copy_to_box(self.cpu_buffer_maps[1] as *const GpuWindow);
        let right = copy_to_box(self.cpu_buffer_maps[1].offset(size_of::<GpuWindow>() as isize) as *const GpuWindow);
    
        (left, right)
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
