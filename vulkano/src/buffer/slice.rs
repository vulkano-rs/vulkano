// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use buffer::traits::BufferAccess;
use buffer::traits::BufferInner;
use buffer::traits::TypedBufferAccess;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use sync::AccessError;

/// A subpart of a buffer.
///
/// This object doesn't correspond to any Vulkan object. It exists for API convenience.
///
/// # Example
///
/// Creating a slice:
///
/// ```ignore       // FIXME: unignore
/// use vulkano::buffer::BufferSlice;
/// # let buffer: std::sync::Arc<vulkano::buffer::DeviceLocalBuffer<[u8]>> = return;
/// let _slice = BufferSlice::from(&buffer);
/// ```
///
/// Selecting a slice of a buffer that contains `[T]`:
///
/// ```ignore       // FIXME: unignore
/// use vulkano::buffer::BufferSlice;
/// # let buffer: std::sync::Arc<vulkano::buffer::DeviceLocalBuffer<[u8]>> = return;
/// let _slice = BufferSlice::from(&buffer).slice(12 .. 14).unwrap();
/// ```
///
pub struct BufferSlice<T: ?Sized, B> {
    marker: PhantomData<T>,
    resource: B,
    offset: usize,
    size: usize,
}

// We need to implement `Clone` manually, otherwise the derive adds a `T: Clone` requirement.
impl<T: ?Sized, B> Clone for BufferSlice<T, B>
    where B: Clone
{
    #[inline]
    fn clone(&self) -> Self {
        BufferSlice {
            marker: PhantomData,
            resource: self.resource.clone(),
            offset: self.offset,
            size: self.size,
        }
    }
}

impl<T: ?Sized, B> BufferSlice<T, B> {
    #[inline]
    pub fn from_typed_buffer_access(r: B) -> BufferSlice<T, B>
        where B: TypedBufferAccess<Content = T>
    {
        let size = r.size();

        BufferSlice {
            marker: PhantomData,
            resource: r,
            offset: 0,
            size: size,
        }
    }

    /// Returns the buffer that this slice belongs to.
    pub fn buffer(&self) -> &B {
        &self.resource
    }

    /// Returns the offset of that slice within the buffer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the size of that slice in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Builds a slice that contains an element from inside the buffer.
    ///
    /// This method builds an object that represents a slice of the buffer. No actual operation
    /// is performed.
    ///
    /// # Example
    ///
    /// TODO
    ///
    /// # Safety
    ///
    /// The object whose reference is passed to the closure is uninitialized. Therefore you
    /// **must not** access the content of the object.
    ///
    /// You **must** return a reference to an element from the parameter. The closure **must not**
    /// panic.
    #[inline]
    pub unsafe fn slice_custom<F, R: ?Sized>(self, f: F) -> BufferSlice<R, B>
        where F: for<'r> FnOnce(&'r T) -> &'r R // TODO: bounds on R
    {
        let data: &T = mem::zeroed();
        let result = f(data);
        let size = mem::size_of_val(result);
        let result = result as *const R as *const () as usize;

        assert!(result <= self.size());
        assert!(result + size <= self.size());

        BufferSlice {
            marker: PhantomData,
            resource: self.resource,
            offset: self.offset + result,
            size: size,
        }
    }
}

impl<T, B> BufferSlice<[T], B> {
    /// Returns the number of elements in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.size() % mem::size_of::<T>(), 0);
        self.size() / mem::size_of::<T>()
    }

    /// Reduces the slice to just one element of the array.
    ///
    /// Returns `None` if out of range.
    #[inline]
    pub fn index(self, index: usize) -> Option<BufferSlice<T, B>> {
        if index >= self.len() {
            return None;
        }

        Some(BufferSlice {
                 marker: PhantomData,
                 resource: self.resource,
                 offset: self.offset + index * mem::size_of::<T>(),
                 size: mem::size_of::<T>(),
             })
    }

    /// Reduces the slice to just a range of the array.
    ///
    /// Returns `None` if out of range.
    #[inline]
    pub fn slice(self, range: Range<usize>) -> Option<BufferSlice<[T], B>> {
        if range.end > self.len() {
            return None;
        }

        Some(BufferSlice {
                 marker: PhantomData,
                 resource: self.resource,
                 offset: self.offset + range.start * mem::size_of::<T>(),
                 size: (range.end - range.start) * mem::size_of::<T>(),
             })
    }
}

unsafe impl<T: ?Sized, B> BufferAccess for BufferSlice<T, B>
    where B: BufferAccess
{
    #[inline]
    fn inner(&self) -> BufferInner {
        let inner = self.resource.inner();
        BufferInner {
            buffer: inner.buffer,
            offset: inner.offset + self.offset,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn conflicts_buffer(&self, other: &BufferAccess) -> bool {
        self.resource.conflicts_buffer(other)
    }

    #[inline]
    fn conflicts_image(&self, other: &ImageAccess) -> bool {
        self.resource.conflicts_image(other)
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.resource.conflict_key()
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> Result<(), AccessError> {
        self.resource.try_gpu_lock(exclusive_access, queue)
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        self.resource.increase_gpu_lock()
    }

    #[inline]
    unsafe fn unlock(&self) {
        self.resource.unlock()
    }
}

unsafe impl<T: ?Sized, B> TypedBufferAccess for BufferSlice<T, B>
    where B: BufferAccess
{
    type Content = T;
}

unsafe impl<T: ?Sized, B> DeviceOwned for BufferSlice<T, B>
    where B: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.resource.device()
    }
}

impl<T, B> From<BufferSlice<T, B>> for BufferSlice<[T], B> {
    #[inline]
    fn from(r: BufferSlice<T, B>) -> BufferSlice<[T], B> {
        BufferSlice {
            marker: PhantomData,
            resource: r.resource,
            offset: r.offset,
            size: r.size,
        }
    }
}

/// Takes a `BufferSlice` that points to a struct, and returns a `BufferSlice` that points to
/// a specific field of that struct.
#[macro_export]
macro_rules! buffer_slice_field {
    ($slice:expr, $field:ident) => (
        // TODO: add #[allow(unsafe_code)] when that's allowed
        unsafe { $slice.slice_custom(|s| &s.$field) }
    )
}
