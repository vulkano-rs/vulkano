//! GPU-visible memory allocation.
use std::mem;
use std::os::raw::c_void;
use std::slice;

use vk;

pub use self::device_memory::CpuAccess;
pub use self::device_memory::DeviceMemory;
pub use self::device_memory::MappedDeviceMemory;

mod device_memory;

#[derive(Debug, Copy, Clone)]
pub struct MemoryRequirements {
    pub size: usize,
    pub alignment: usize,
    pub memory_type_bits: u32,
}

#[doc(hidden)]
impl From<vk::MemoryRequirements> for MemoryRequirements {
    #[inline]
    fn from(reqs: vk::MemoryRequirements) -> MemoryRequirements {
        MemoryRequirements {
            size: reqs.size as usize,
            alignment: reqs.alignment as usize,
            memory_type_bits: reqs.memoryTypeBits,
        }
    }
}

/// Trait for types of data that can be mapped.
// TODO: move to `buffer` module
pub unsafe trait Content {
    /// Builds a pointer to this type from a raw pointer.
    fn ref_from_ptr<'a>(ptr: *mut c_void, size: usize) -> Option<*mut Self>;

    /// Returns true if the size is suitable to store a type like this.
    fn is_size_suitable(usize) -> bool;

    /// Returns the size of an individual element.
    fn indiv_size() -> usize;
}

unsafe impl<T> Content for T {
    #[inline]
    fn ref_from_ptr<'a>(ptr: *mut c_void, size: usize) -> Option<*mut T> {
        if size < mem::size_of::<T>() {
            return None;
        }

        Some(ptr as *mut T)
    }

    #[inline]
    fn is_size_suitable(size: usize) -> bool {
        size == mem::size_of::<T>()
    }

    #[inline]
    fn indiv_size() -> usize {
        mem::size_of::<T>()
    }
}

unsafe impl<T> Content for [T] {
    #[inline]
    fn ref_from_ptr<'a>(ptr: *mut c_void, size: usize) -> Option<*mut [T]> {
        let ptr = ptr as *mut T;
        let size = size / mem::size_of::<T>();
        Some(unsafe { slice::from_raw_parts_mut(&mut *ptr, size) as *mut [T] })
    }

    #[inline]
    fn is_size_suitable(size: usize) -> bool {
        size % mem::size_of::<T>() == 0
    }

    #[inline]
    fn indiv_size() -> usize {
        mem::size_of::<T>()
    }
}

/*
TODO: do this when it's possible
unsafe impl Content for .. {}
impl<'a, T> !Content for &'a T {}
impl<'a, T> !Content for &'a mut T {}
impl<T> !Content for *const T {}
impl<T> !Content for *mut T {}
impl<T> !Content for Box<T> {}
impl<T> !Content for UnsafeCell<T> {}

*/
