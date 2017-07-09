// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;

use buffer::BufferSlice;
use buffer::sys::UnsafeBuffer;
use device::DeviceOwned;
use device::Queue;
use image::ImageAccess;
use memory::Content;
use sync::AccessError;

use SafeDeref;
use VulkanObject;

/// Trait for objects that represent a way for the GPU to have access to a buffer or a slice of a
/// buffer.
///
/// See also `TypedBufferAccess`.
pub unsafe trait BufferAccess: DeviceOwned {
    /// Returns the inner information about this buffer.
    fn inner(&self) -> BufferInner;

    /// Returns the size of the buffer in bytes.
    fn size(&self) -> usize;

    /// Returns the length of the buffer in number of elements.
    ///
    /// This method can only be called for buffers whose type is known to be an array.
    #[inline]
    fn len(&self) -> usize
        where Self: TypedBufferAccess,
              Self::Content: Content
    {
        self.size() / <Self::Content as Content>::indiv_size()
    }

    /// Builds a `BufferSlice` object holding the buffer by reference.
    #[inline]
    fn as_buffer_slice(&self) -> BufferSlice<Self::Content, &Self>
        where Self: Sized + TypedBufferAccess
    {
        BufferSlice::from_typed_buffer_access(self)
    }

    /// Builds a `BufferSlice` object holding part of the buffer by reference.
    ///
    /// This method can only be called for buffers whose type is known to be an array.
    ///
    /// This method can be used when you want to perform an operation on some part of the buffer
    /// and not on the whole buffer.
    ///
    /// Returns `None` if out of range.
    #[inline]
    fn slice<T>(&self, range: Range<usize>) -> Option<BufferSlice<[T], &Self>>
        where Self: Sized + TypedBufferAccess<Content = [T]>
    {
        BufferSlice::slice(self.as_buffer_slice(), range)
    }

    /// Builds a `BufferSlice` object holding the buffer by value.
    #[inline]
    fn into_buffer_slice(self) -> BufferSlice<Self::Content, Self>
        where Self: Sized + TypedBufferAccess
    {
        BufferSlice::from_typed_buffer_access(self)
    }

    /// Builds a `BufferSlice` object holding part of the buffer by reference.
    ///
    /// This method can only be called for buffers whose type is known to be an array.
    ///
    /// This method can be used when you want to perform an operation on a specific element of the
    /// buffer and not on the whole buffer.
    ///
    /// Returns `None` if out of range.
    #[inline]
    fn index<T>(&self, index: usize) -> Option<BufferSlice<[T], &Self>>
        where Self: Sized + TypedBufferAccess<Content = [T]>
    {
        self.slice(index .. (index + 1))
    }

    /// Returns true if an access to `self` (as defined by `self_offset` and `self_size`)
    /// potentially overlaps the same memory as an access to `other` (as defined by `other_offset`
    /// and `other_size`).
    ///
    /// If this function returns `false`, this means that we are allowed to access the offset/size
    /// of `self` at the same time as the offset/size of `other` without causing a data race.
    ///
    /// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
    /// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize, other: &BufferAccess,
                        other_offset: usize, other_size: usize)
                        -> bool {
        // TODO: should we really provide a default implementation?

        debug_assert!(self_size <= self.size());

        if self.inner().buffer.internal_object() != other.inner().buffer.internal_object() {
            return false;
        }

        true
    }

    /// Returns true if an access to `self` (as defined by `self_offset` and `self_size`)
    /// potentially overlaps the same memory as an access to `other` (as defined by
    /// `other_first_layer`, `other_num_layers`, `other_first_mipmap` and `other_num_mipmaps`).
    ///
    /// If this function returns `false`, this means that we are allowed to access the offset/size
    /// of `self` at the same time as the offset/size of `other` without causing a data race.
    ///
    /// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
    /// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
    fn conflicts_image(&self, self_offset: usize, self_size: usize, other: &ImageAccess,
                       other_first_layer: u32, other_num_layers: u32, other_first_mipmap: u32,
                       other_num_mipmaps: u32)
                       -> bool {
        let other_key = other.conflict_key(other_first_layer,
                                           other_num_layers,
                                           other_first_mipmap,
                                           other_num_mipmaps);
        self.conflict_key(self_offset, self_size) == other_key
    }

    /// Returns a key that uniquely identifies the range given by offset/size.
    ///
    /// Two ranges that potentially overlap in memory should return the same key.
    ///
    /// The key is shared amongst all buffers and images, which means that you can make several
    /// different buffer objects share the same memory, or make some buffer objects share memory
    /// with images, as long as they return the same key.
    ///
    /// Since it is possible to accidentally return the same key for memory ranges that don't
    /// overlap, the `conflicts_buffer` or `conflicts_image` function should always be called to
    /// verify whether they actually overlap.
    fn conflict_key(&self, self_offset: usize, self_size: usize) -> u64 {
        // FIXME: remove implementation
        unimplemented!()
    }

    /// Shortcut for `conflicts_buffer` that compares the whole buffer to another.
    #[inline]
    fn conflicts_buffer_all(&self, other: &BufferAccess) -> bool {
        self.conflicts_buffer(0, self.size(), other, 0, other.size())
    }

    /// Shortcut for `conflicts_image` that compares the whole buffer to a whole image.
    #[inline]
    fn conflicts_image_all(&self, other: &ImageAccess) -> bool {
        self.conflicts_image(0,
                             self.size(),
                             other,
                             0,
                             other.dimensions().array_layers(),
                             0,
                             other.mipmap_levels())
    }

    /// Shortcut for `conflict_key` that grabs the key of the whole buffer.
    #[inline]
    fn conflict_key_all(&self) -> u64 {
        self.conflict_key(0, self.size())
    }

    /// Locks the resource for usage on the GPU. Returns an error if the lock can't be acquired.
    ///
    /// This function exists to prevent the user from causing a data race by reading and writing
    /// to the same resource at the same time.
    ///
    /// If you call this function, you should call `unlock()` once the resource is no longer in use
    /// by the GPU. The implementation is not expected to automatically perform any unlocking and
    /// can rely on the fact that `unlock()` is going to be called.
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> Result<(), AccessError>;

    /// Locks the resource for usage on the GPU. Supposes that the resource is already locked, and
    /// simply increases the lock by one.
    ///
    /// Must only be called after `try_gpu_lock()` succeeded.
    ///
    /// If you call this function, you should call `unlock()` once the resource is no longer in use
    /// by the GPU. The implementation is not expected to automatically perform any unlocking and
    /// can rely on the fact that `unlock()` is going to be called.
    unsafe fn increase_gpu_lock(&self);

    /// Unlocks the resource previously acquired with `try_gpu_lock` or `increase_gpu_lock`.
    ///
    /// # Safety
    ///
    /// Must only be called once per previous lock.
    unsafe fn unlock(&self);
}

/// Inner information about a buffer.
#[derive(Copy, Clone, Debug)]
pub struct BufferInner<'a> {
    /// The underlying buffer object.
    pub buffer: &'a UnsafeBuffer,
    /// The offset in bytes from the start of the underlying buffer object to the start of the
    /// buffer we're describing.
    pub offset: usize,
}

unsafe impl<T> BufferAccess for T
    where T: SafeDeref,
          T::Target: BufferAccess
{
    #[inline]
    fn inner(&self) -> BufferInner {
        (**self).inner()
    }

    #[inline]
    fn size(&self) -> usize {
        (**self).size()
    }

    #[inline]
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize, other: &BufferAccess,
                        other_offset: usize, other_size: usize)
                        -> bool {
        (**self).conflicts_buffer(self_offset, self_size, other, other_offset, other_size)
    }

    #[inline]
    fn conflict_key(&self, self_offset: usize, self_size: usize) -> u64 {
        (**self).conflict_key(self_offset, self_size)
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> Result<(), AccessError> {
        (**self).try_gpu_lock(exclusive_access, queue)
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        (**self).increase_gpu_lock()
    }

    #[inline]
    unsafe fn unlock(&self) {
        (**self).unlock()
    }
}

/// Extension trait for `BufferAccess`. Indicates the type of the content of the buffer.
pub unsafe trait TypedBufferAccess: BufferAccess {
    /// The type of the content.
    type Content: ?Sized;
}

unsafe impl<T> TypedBufferAccess for T
    where T: SafeDeref,
          T::Target: TypedBufferAccess
{
    type Content = <T::Target as TypedBufferAccess>::Content;
}
