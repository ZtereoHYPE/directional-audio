use std::array::from_ref;
use std::rc::Rc;
use std::u64::MAX;
use ash::Device;
use ash::vk::{AccessFlags, Buffer, BufferCopy, CommandBuffer, CommandBufferBeginInfo, CommandBufferResetFlags, CommandBufferUsageFlags, DependencyFlags, Fence, MemoryBarrier, PipelineStageFlags, Queue, SubmitInfo, WHOLE_SIZE};
use vk_mem::{Allocation, Allocator};
use crate::vulkan::signal_processor::{FftBuffer, FftFrame};

pub struct TransferModule {
    allocator: Rc<Allocator>,
    device: Device,
    queue: Queue,

    cpu_buffer: Buffer,
    cpu_buffer_memory: Allocation,
    cpu_buffer_map: *mut u8,

    fence: Fence,
}

impl TransferModule {
    pub unsafe fn new(
        allocator: Rc<Allocator>,
        device: Device,
        queue: Queue,
        cpu_buffer: Buffer,
        cpu_buffer_memory: Allocation,
        cpu_buffer_map: *mut u8,
        fence: Fence,
    ) -> TransferModule {
        TransferModule {
            allocator,
            device,
            queue,
            cpu_buffer,
            cpu_buffer_memory,
            cpu_buffer_map,
            fence,
        }
    }

    pub unsafe fn transfer_to_gpu<T>(&mut self, command_buffer: &mut CommandBuffer, src: Box<T>, dst: &Buffer) {
        // copy frame to cpu buffer
        Self::copy_from_box(&src, self.cpu_buffer_map.cast::<T>());
        self.allocator.flush_allocation(&self.cpu_buffer_memory, 0, WHOLE_SIZE);

        self.device
            .reset_command_buffer(*command_buffer, CommandBufferResetFlags::empty())
            .expect("Failed to reset command buffer");

        let begin_info = CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device
            .begin_command_buffer(*command_buffer, &begin_info)
            .expect("Failed to begin command buffer recording");

        let regions = [ BufferCopy::default().size(size_of::<T>() as _) ]; // todo: this could not be right size

        self.device.cmd_copy_buffer(
            *command_buffer,
            self.cpu_buffer,
            *dst,
            &regions
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

    pub unsafe fn transfer_from_gpu(&mut self, command_buffer: &mut CommandBuffer, src: &Buffer) -> (Box<FftFrame>, Box<FftFrame>) {
        let region = BufferCopy::default()
            .size(size_of::<FftFrame>() as u64 * 2);

        self.device.cmd_copy_buffer(
            *command_buffer,
            *src,
            self.cpu_buffer,
            from_ref(&region)
        );

        // let region = BufferCopy::default()
        //     .size(size_of::<FftFrame>() as _)
        //     .dst_offset(size_of::<FftFrame>() as u64);
        // 
        // self.device.cmd_copy_buffer(
        //     *command_buffer,
        //     *right,
        //     self.cpu_buffer,
        //     from_ref(&region)
        // );

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
            .invalidate_allocation(&self.cpu_buffer_memory, 0, WHOLE_SIZE)
            .expect("Failed to invalidate allocation");

        let left = Self::copy_to_box(self.cpu_buffer_map as *const FftFrame);
        let right = Self::copy_to_box(self.cpu_buffer_map.offset(size_of::<FftFrame>() as isize) as *const FftFrame);

        (left, right)
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
}