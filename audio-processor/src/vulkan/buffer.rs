use ash::vk;
use ash::vk::{BufferCopy, BufferCreateInfo, BufferUsageFlags, DeviceSize, SharingMode, WHOLE_SIZE};
use std::alloc::{alloc_zeroed, Layout};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::slice::from_ref;
use vk_mem::{Alloc, AllocationCreateFlags, Allocator};

pub(crate) trait InlineBufferData: BufferData {
    const REGION: BufferCopy = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: size_of::<Self>() as _,
    };

    fn region() -> &'static[BufferCopy] {
        from_ref(&Self::REGION)
    }

    fn from_memory_map(pointer: *mut u8) -> *mut UnsafeCell<Self> {
        if pointer.is_null() {
            panic!("The given pointer is null!");
        }

        if !pointer.cast::<Self>().is_aligned() {
            panic!("The given pointer is not properly aligned!");
        }

        pointer.cast()
    }

    unsafe fn to_local_copy(&self) -> Box<Self> {
        let layout = Layout::new::<Self>();
        let src = self as *const Self;

        unsafe {
            // Allocate the required space
            let dst = alloc_zeroed(layout) as *mut Self;

            // Copy the memory value
            std::ptr::copy_nonoverlapping(src, dst, 1);

            // Wrap in a box
            Box::from_raw(dst)
        }
    }
}

pub(crate) trait BufferData: Sized {
    unsafe fn serialize(&self, dst: *mut u8);
    fn size(&self) -> usize;
}

/// Implement the buffer data functions for all inline data
impl<T: InlineBufferData> BufferData for T {
    unsafe fn serialize(&self, dst: *mut u8) { unsafe {
        std::ptr::copy_nonoverlapping(
            (self as *const T).cast(),
            dst,
            self.size()
        );
    }}

    fn size(&self) -> usize {
        size_of::<T>()
    }
}


pub(crate) trait BufferOps<T: BufferData>  {
    fn base(&self) -> &BufferBase<T>;
    fn base_mut(&mut self) -> &mut BufferBase<T>;

    // other methods, with default implementations
    fn handle(&self) -> vk::Buffer {
        self.base().handle
    }

    fn size(&self) -> DeviceSize {
        size_of::<T>() as DeviceSize
    }
}

pub(crate) struct BufferBase<T: BufferData> {
    handle: vk::Buffer,
    memory: vk_mem::Allocation,
    allocator: Rc<Allocator>,
    _marker: PhantomData<T> // While this struct doesn't directly own a T, it still "contains" it (through the buffer)
}

pub(crate) struct VulkanBuffer<T: BufferData> {
    base: BufferBase<T>,
}

impl<T: BufferData> VulkanBuffer<T> {
    pub(crate) fn new_dynamic(usage: BufferUsageFlags, data: T, allocator: Rc<Allocator>) -> Self {
        Self::new(usage, data.size(), allocator)
    }

    fn new(usage: BufferUsageFlags, size: usize, allocator: Rc<Allocator>) -> Self {
        let buffer_info = BufferCreateInfo::default()
            .size(size as DeviceSize)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (buffer, memory) = unsafe {
            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        Self {
            base: BufferBase::<T> {
                handle: buffer,
                memory,
                allocator,
                _marker: PhantomData
            }
        }
    }
}

impl<T: BufferData> VulkanBuffer<T> {
    pub(crate) fn new_inline(usage: BufferUsageFlags, allocator: Rc<Allocator>) -> Self {
        Self::new(usage, size_of::<T>(), allocator)
    }
}

impl<T: BufferData> BufferOps<T> for VulkanBuffer<T> {
    fn base(&self) -> &BufferBase<T> { &self.base }
    fn base_mut(&mut self) -> &mut BufferBase<T> { &mut self.base }
}

impl<T: BufferData> Drop for VulkanBuffer<T> {
    fn drop(&mut self) {
        let allocator = self.base().allocator.clone();
        unsafe {
            allocator.destroy_buffer(self.handle(), &mut self.base_mut().memory);
        }
    }
}



/// # Safety:
/// To avoid UB caused by interior mutability, UnsafeCell<T> is used to store the pointer
/// to the locally-mapped memory.
pub(crate) struct LocalVulkanBuffer<T: InlineBufferData> {
    base: BufferBase<T>,
    map: *mut UnsafeCell<T>,
}

impl<T: InlineBufferData> LocalVulkanBuffer<T> {
    pub(crate) fn new_inline(usage: BufferUsageFlags, allocator: Rc<Allocator>) -> Self {
        let buffer_info = BufferCreateInfo::default()
            .size(size_of::<T>() as DeviceSize)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (handle, mut memory) = unsafe {
            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        let map_pointer = unsafe {
            allocator
                .map_memory(&mut memory)
                .expect("Failed to map memory")
        };

        let map = T::from_memory_map(map_pointer);

        Self {
            base: BufferBase {
                handle,
                memory,
                allocator,
                _marker: PhantomData
            },
            map,
        }
    }

    pub(crate) fn buffer_data(&mut self) -> &mut T {
        unsafe { &mut *(*self.map).get() }
    }

    /// Call after writing to local memory
    pub(crate) fn flush(&mut self) {
        self.base.allocator
            .flush_allocation(&self.base.memory, 0, WHOLE_SIZE)
            .expect("Failed to flush allocation")
    }

    /// Call before reading local memory
    pub(crate) fn invalidate(&mut self) {
        self.base.allocator
            .invalidate_allocation(&self.base.memory, 0, WHOLE_SIZE)
            .expect("Failed to invalidate allocation")
    }
}

impl<T: InlineBufferData> BufferOps<T> for LocalVulkanBuffer<T> {
    fn base(&self) -> &BufferBase<T> { &self.base }
    fn base_mut(&mut self) -> &mut BufferBase<T> { &mut self.base }
}

impl<T: InlineBufferData> Drop for LocalVulkanBuffer<T> {
    fn drop(&mut self) {
        let allocator = self.base().allocator.clone();
        unsafe {
            allocator.unmap_memory(&mut self.base_mut().memory);
            allocator.destroy_buffer(self.handle(), &mut self.base_mut().memory);
        }
    }
}
