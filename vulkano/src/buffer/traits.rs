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
use device::Queue;
use image::Image;
use memory::Content;

use SafeDeref;
use VulkanObject;

/// Trait for objects that represent either a buffer or a slice of a buffer.
pub unsafe trait Buffer {
    /// Returns the inner information about this buffer.
    fn inner(&self) -> BufferInner;

    /// Returns the size of the buffer in bytes.
    #[inline]
    fn size(&self) -> usize {
        self.inner().buffer.size()
    }

    /// Returns the length of the buffer in number of elements.
    ///
    /// This method can only be called for buffers whose type is known to be an array.
    #[inline]
    fn len(&self) -> usize where Self: TypedBuffer, Self::Content: Content {
        self.size() / <Self::Content as Content>::indiv_size()
    }

    /// Builds a `BufferSlice` object holding the buffer by reference.
    #[inline]
    fn as_buffer_slice(&self) -> BufferSlice<Self::Content, &Self>
        where Self: Sized + TypedBuffer
    {
        BufferSlice::from(self)
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
        where Self: Sized + TypedBuffer<Content = [T]>,
              T: 'static
    {
        BufferSlice::slice(self.as_buffer_slice(), range)
    }

    /// Builds a `BufferSlice` object holding the buffer by value.
    #[inline]
    fn into_buffer_slice(self) -> BufferSlice<Self::Content, Self>
        where Self: Sized + TypedBuffer
    {
        BufferSlice::from(self)
    }

    /// Returns true if an access to `self` (as defined by `self_offset` and `self_size`)
    /// potentially overlaps the same memory as an access to `other` (as defined by `other_offset`
    /// and `other_size`).
    ///
    /// If this function returns `false`, this means that we are allowed to access the offset/size
    /// of `self` at the same time as the offset/size of `other` without causing a data race.
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize,
                        other: &Buffer, other_offset: usize, other_size: usize)
                        -> bool
    {
        // TODO: should we really provide a default implementation?

        debug_assert!(self_size <= self.size());

        if self.inner().buffer.internal_object() != other.inner().buffer.internal_object() {
            return false;
        }

        let self_offset = self_offset + self.inner().offset;
        let other_offset = other_offset + other.inner().offset;

        if self_offset < other_offset && self_offset + self_size <= other_offset {
            return false;
        }

        if other_offset < self_offset && other_offset + other_size <= self_offset {
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
    fn conflicts_image(&self, self_offset: usize, self_size: usize, other: &Image,
                       other_first_layer: u32, other_num_layers: u32, other_first_mipmap: u32,
                       other_num_mipmaps: u32) -> bool
    {
        let other_key = other.conflict_key(other_first_layer, other_num_layers, other_first_mipmap,
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

    /// Returns true if the buffer can be given access on the given queue.
    ///
    /// This function implementation should remember that it has been called and return `false` if
    /// it gets called a second time.
    ///
    /// The only way to know that the GPU has stopped accessing a queue is when the buffer object
    /// gets destroyed. Therefore you are encouraged to use temporary objects or handles (similar
    /// to a lock) in order to represent a GPU access.
    fn gpu_access(&self, exclusive_access: bool, queue: &Queue) -> bool;
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

unsafe impl<T> Buffer for T where T: SafeDeref, T::Target: Buffer {
    #[inline]
    fn inner(&self) -> BufferInner {
        (**self).inner()
    }

    #[inline]
    fn size(&self) -> usize {
        (**self).size()
    }

    #[inline]
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize,
                        other: &Buffer, other_offset: usize, other_size: usize) -> bool
    {
        (**self).conflicts_buffer(self_offset, self_size, other, other_offset, other_size)
    }

    #[inline]
    fn conflict_key(&self, self_offset: usize, self_size: usize) -> u64 {
        (**self).conflict_key(self_offset, self_size)
    }

    #[inline]
    fn gpu_access(&self, exclusive_access: bool, queue: &Queue) -> bool {
        (**self).gpu_access(exclusive_access, queue)
    }
}

pub unsafe trait TypedBuffer: Buffer {
    type Content: ?Sized + 'static;
}

unsafe impl<T> TypedBuffer for T where T: SafeDeref, T::Target: TypedBuffer {
    type Content = <T::Target as TypedBuffer>::Content;
}
