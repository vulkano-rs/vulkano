// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Device memory allocation and memory pools.
//!
//! By default, memory allocation is automatically handled by the vulkano library when you create
//! a buffer or an image. But if you want more control, you have the possibility to costumize the
//! memory allocation strategy.
//!
//! # Memory types and heaps
//!
//! A physical device is composed of one or more **memory heaps**. A memory heap is a pool of
//! memory that can be allocated.
//!
//! ```
//! // Enumerating memory heaps.
//! # let physical_device: vulkano::instance::PhysicalDevice = return;
//! for heap in physical_device.memory_heaps() {
//!     println!("Heap #{:?} has a capacity of {:?} bytes", heap.id(), heap.size());
//! }
//! ```
//!
//! However you can't allocate directly from a memory heap. A memory heap is shared amongst one or
//! multiple **memory types**, which you can allocate memory from. Each memory type has different
//! characteristics.
//!
//! A memory type may or may not be visible to the host. In other words, it may or may not be
//! directly writable by the CPU. A memory type may or may not be device-local. A device-local
//! memory type has a much quicker access time from the GPU than a non-device-local type. Note
//! that non-device-local memory types are still accessible by the device, they are just slower.
//!
//! ```
//! // Enumerating memory types.
//! # let physical_device: vulkano::instance::PhysicalDevice = return;
//! for ty in physical_device.memory_types() {
//!     println!("Memory type belongs to heap #{:?}", ty.heap().id());
//!     println!("Host-accessible: {:?}", ty.is_host_visible());
//!     println!("Device-local: {:?}", ty.is_device_local());
//! }
//! ```
//!
//! Memory types are order from "best" to "worse". In other words, the implementation prefers that
//! you use the memory types that are earlier in the list. This means that selecting a memory type
//! should always be done by enumerating them and taking the first one that matches our criterias.
//!
//! ## In practice
//!
//! In practice, desktop machines usually have two memory heaps: one that represents the RAM of
//! the CPU, and one that represents the RAM of the GPU. The CPU's RAM is host-accessible but not
//! device-local, while the GPU's RAM is not host-accessible but is device-local.
//!
//! Mobile machines usually have a single memory heap that is "equally local" to both the CPU and
//! the GPU. It is both host-accessible and device-local.
//!
//! # Allocating memory and memory pools
//!
//! Allocating memory can be done by calling `DeviceMemory::alloc()`.
//!
//! Here is an example:
//!
//! ```
//! use vulkano::memory::DeviceMemory;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! // Taking the first memory type for the sake of this example.
//! let ty = device.physical_device().memory_types().next().unwrap();
//!
//! let alloc = DeviceMemory::alloc(device.clone(), ty, 1024).expect("Failed to allocate memory");
//!
//! // The memory is automatically free'd when `alloc` is destroyed.
//! ```
//!
//! However allocating and freeing memory is very slow (up to several hundred milliseconds
//! sometimes). Instead you are strongly encouraged to use a memory pool. A memory pool is not
//! a Vulkan concept but a vulkano concept.
//!
//! A memory pool is any object that implements the `MemoryPool` trait. You can implement that
//! trait on your own structure and then use it when you create buffers and images so that they
//! get memory from that pool. By default if you don't specify any pool when creating a buffer or
//! an image, an instance of `StdMemoryPool` that is shared by the `Device` object is used.

use std::mem;
use std::os::raw::c_void;
use std::slice;

use buffer::sys::UnsafeBuffer;
use image::sys::UnsafeImage;
use vk;

pub use self::device_memory::CpuAccess;
pub use self::device_memory::DeviceMemory;
pub use self::device_memory::DeviceMemoryAllocError;
pub use self::device_memory::MappedDeviceMemory;
pub use self::pool::MemoryPool;

mod device_memory;
pub mod pool;

/// Represents requirements expressed by the Vulkan implementation when it comes to binding memory
/// to a resource.
#[derive(Debug, Copy, Clone)]
pub struct MemoryRequirements {
    /// Number of bytes of memory required.
    pub size: usize,

    /// Alignment of the requirement buffer. The base memory address must be a multiple
    /// of this value.
    pub alignment: usize,

    /// Indicates which memory types can be used. Each bit that is set to 1 means that the memory
    /// type whose index is the same as the position of the bit can be used.
    pub memory_type_bits: u32,

    /// True if the implementation prefers to use dedicated allocations (in other words, allocate
    /// a whole block of memory dedicated to this resource alone). If the
    /// `khr_get_memory_requirements2` extension isn't enabled, then this will be false.
    ///
    /// > **Note**: As its name says, using a dedicated allocation is an optimization and not a
    /// > requirement.
    pub prefer_dedicated: bool,
}

impl MemoryRequirements {
    #[inline]
    pub(crate) fn from_vulkan_reqs(reqs: vk::MemoryRequirements) -> MemoryRequirements {
        MemoryRequirements {
            size: reqs.size as usize,
            alignment: reqs.alignment as usize,
            memory_type_bits: reqs.memoryTypeBits,
            prefer_dedicated: false,
        }
    }
}

/// Indicates whether we want to allocate memory for a specific resource, or in a generic way.
///
/// Using dedicated allocations can yield faster performances, but requires the
/// `VK_KHR_dedicated_allocation` extension to be enabled on the device.
///
/// If a dedicated allocation is performed, it must only be bound to any resource other than the
/// one that was passed with the enumeration.
#[derive(Debug, Copy, Clone)]
pub enum DedicatedAlloc<'a> {
    /// Generic allocation.
    None,
    /// Allocation dedicated to a buffer.
    Buffer(&'a UnsafeBuffer),
    /// Allocation dedicated to an image.
    Image(&'a UnsafeImage),
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
