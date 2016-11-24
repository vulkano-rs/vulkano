// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::sync::Arc;

use buffer::BufferSlice;
use buffer::sys::UnsafeBuffer;
use image::Image;
use memory::Content;

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

    /// Returns true if an access to `self` (as defined by `self_offset`, `self_size` and
    /// `self_write`) shouldn't execute at the same time as an access to `other` (as defined by
    /// `other_offset`, `other_size` and `other_write`).
    ///
    /// Returns false if they can be executed simultaneously.
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize, self_write: bool,
                        other: &Buffer, other_offset: usize, other_size: usize, other_write: bool)
                        -> bool
    {
        // TODO: should we really provide a default implementation?

        debug_assert!(self_size <= self.size());

        if self.inner().buffer.internal_object() != other.inner().buffer.internal_object() {
            return false;
        }

        if !self_write && !other_write {
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

    /// Returns true if an access to `self` shouldn't execute at the same time as an access to
    /// `other`.
    ///
    /// Returns false if they can be executed simultaneously.
    fn conflicts_image(&self, self_offset: usize, self_size: usize, self_write: bool, other: &Image,
                       other_first_layer: u32, other_num_layers: u32, other_first_mipmap: u32,
                       other_num_mipmaps: u32, other_write: bool) -> bool
    {
        // TODO: should we really provide a default implementation?
        false
    }

    /// Two resources that conflict with each other should return the same key.
    fn conflict_key(&self, self_offset: usize, self_size: usize, self_write: bool) -> u64 {
        // FIXME: remove implementation
        unimplemented!()
    }
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

unsafe impl<'a, B: ?Sized> Buffer for &'a B where B: Buffer + 'a {
    #[inline]
    fn inner(&self) -> BufferInner {
        (**self).inner()
    }

    #[inline]
    fn size(&self) -> usize {
        (**self).size()
    }

    #[inline]
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize, self_write: bool,
                        other: &Buffer, other_offset: usize, other_size: usize, other_write: bool)
                        -> bool
    {
        (**self).conflicts_buffer(self_offset, self_size, self_write, other, other_offset,
                                  other_size, other_write)
    }

    #[inline]
    fn conflict_key(&self, self_offset: usize, self_size: usize, self_write: bool) -> u64 {
        (**self).conflict_key(self_offset, self_size, self_write)
    }
}

unsafe impl<B: ?Sized> Buffer for Arc<B> where B: Buffer {
    #[inline]
    fn inner(&self) -> BufferInner {
        (**self).inner()
    }

    #[inline]
    fn size(&self) -> usize {
        (**self).size()
    }

    #[inline]
    fn conflicts_buffer(&self, self_offset: usize, self_size: usize, self_write: bool,
                        other: &Buffer, other_offset: usize, other_size: usize, other_write: bool)
                        -> bool
    {
        (**self).conflicts_buffer(self_offset, self_size, self_write, other, other_offset,
                                  other_size, other_write)
    }

    #[inline]
    fn conflict_key(&self, self_offset: usize, self_size: usize, self_write: bool) -> u64 {
        (**self).conflict_key(self_offset, self_size, self_write)
    }
}

pub unsafe trait TypedBuffer: Buffer {
    type Content: ?Sized + 'static;
}

unsafe impl<B: ?Sized> TypedBuffer for Arc<B> where B: TypedBuffer {
    type Content = B::Content;
}

unsafe impl<'a, B: ?Sized + 'a> TypedBuffer for &'a B where B: TypedBuffer {
    type Content = B::Content;
}
