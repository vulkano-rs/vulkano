//! GPU-visible memory allocation and management.
//! 
//! When you create a buffer or a texture with Vulkan, you have to bind it to a chunk of allocated
//! memory. To do so, you have to pass a type that implements the `MemorySource` trait.
//! 
//! There are several implementations of the trait, ie. several things that you can pass to the
//! constructors of buffers and textures:
//! 
//! - `&Arc<Device>`, which will simply allocate a new chunk of memory every time (easy but not
//!   very efficient).
//! - `MemorySource`, which is the same as `&Arc<Device>` except that it will use the
//!   already-allocated block.
//! - ... needs more ...
//! 
//! # Synchronization
//! 
//! In Vulkan, it's the job of the programmer to enforce memory safety. In other words, the
//! programmer must take care that two chunks of memory are not read and written simultaneously.
//! 
//! In this library, this is enforced by the implementation of `MemorySource` or
//! `MemorySourceChunk`.
//! 
//! There are two mechanisms in Vulkan that can provide synchronization: fences and semaphores.
//! Fences provide synchronization between the CPU and the GPU, and semaphores provide
//! synchronization between multiple queues of the GPU. See the `sync` module for more info.
//!
//! # Sparse resources
//! 
//! **Not yet implemented**.
//! 
//! Instead of creating a buffer or an image with a single chunk of memory, you also have the
//! possibility to create resources with *sparse memory*.
//! 
//! For example you can bind the first half of the buffer to a memory chunk, and the second half of
//! the buffer to another memory chunk.
//! 
//! There is a hierarchy of three features related to sparse resources:
//! 
//!  - The `sparseBinding` feature allows you to use sparse resources.
//!  - The `sparseResidency` feature is a superset of `sparseBinding` and allows you to leave some
//!    parts of the resource unbinded before using it.
//!  - The `sparseResidencyAliased` feature is a superset of `sparseResidency` and allows you to
//!    bind the same memory chunk to multiple different resources at once.
//!
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::raw::c_void;
use std::slice;

pub use self::device_memory::DeviceMemory;
pub use self::device_memory::MappedDeviceMemory;
pub use self::single::DeviceLocal;
pub use self::single::DeviceLocalChunk;
pub use self::single::HostVisible;
pub use self::single::HostVisibleChunk;

mod device_memory;
mod single;

/// Trait for memory objects that can be accessed from the CPU.
pub unsafe trait CpuAccessible<'a, T: ?Sized> {
    /// The object that provides the access.
    type Read: Deref<Target = T>;

    /// Gives a read access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, blocks until it is available.
    // TODO: what happens if timeout is reached? a panic?
    fn read(&'a self, timeout_ns: u64) -> Self::Read;

    /// Tries to give a read access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, returns `None`.
    fn try_read(&'a self) -> Option<Self::Read>;
}

/// Trait for memory objects that can be mutably accessed from the CPU.
pub unsafe trait CpuWriteAccessible<'a, T: ?Sized>: CpuAccessible<'a, T> {
    /// The object that provides the access.
    type Write: DerefMut<Target = T>;

    /// Gives a write access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, blocks until it is available.
    // TODO: what happens if timeout is reached? a panic?
    fn write(&'a self, timeout_ns: u64) -> Self::Write;

    /// Tries to give a write access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, returns `None`.
    fn try_write(&'a self) -> Option<Self::Write>;
}

pub enum ChunkProperties<'a> {
    Regular {
        memory: &'a DeviceMemory,
        offset: usize,
        size: usize,
    },

    Sparse,     // TODO: unimplemented
}

/// Trait for types of data that can be mapped.
pub unsafe trait Content {
    /// Builds a pointer to this type from a raw pointer.
    fn ref_from_ptr<'a>(ptr: *mut c_void, size: usize) -> Option<*mut Self>;

    /// Returns true if the size is suitable to store a type like this.
    fn is_size_suitable(usize) -> bool;
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
