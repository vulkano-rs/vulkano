// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Location in memory that contains data.
//!
//! All buffers are guaranteed to be accessible from the GPU.
//!
//! # High-level wrappers
//!
//! The low level implementation of a buffer is `UnsafeBuffer`. However, the vulkano library
//! provides high-level wrappers around that type that are specialized depending on the way you
//! are going to use it:
//!
//! - `CpuAccessBuffer` designates a buffer located in RAM and whose content can be written by
//!   your application.
//! - `ImmutableBuffer` designates a buffer in video memory and whose content can only be
//!   written once.
//! 
//! If are a beginner, you are strongly encouraged to use one of these wrappers.
//!
//! # Buffers usage
//!
//! Buffers have the following usages:
//!
//! - Can contain arbitrary data that can be copied from/to other buffers and images.
//! - Can be read and modified from a shader.
//! - Can be used as a source of vertices and indices.
//! - Can be used as a source of list of models for draw indirect commands.
//!
//! The correct `Usage` flags have to be passed when you create a buffer. Trying to use a buffer
//! in a way that wasn't specified when you created it will result in an error.
//!
//! Accessing a buffer from a shader can be done in the following ways:
//!
//! - As a uniform buffer. Uniform buffers are read-only.
//! - As a storage buffer. Storage buffers can be read and written.
//! - As a uniform texel buffer. Contrary to a uniform buffer, the data is interpreted by the
//!   GPU and can be for example normalized.
//! - As a storage texel buffer. Additionnally, some data formats can be modified with atomic
//!   operations.
//!
//! Using uniform/storage texel buffers requires creating a *buffer view*. See the `view` module
//! for how to create a buffer view.
//!
//! # Data uploads
//!
//! One of the major usages of buffers is to upload data to other buffers and to images. When you
//! create a buffer or an image in non-host-visible memory, one of the only way to upload data to
//! it is write the data in a temporary buffer and ask the device to copy from the temporary buffer
//! to the real buffer.
//!
//! TODO: add example of this
//!
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

pub use self::device_local::DeviceLocalBuffer;
pub use self::sys::Usage;
pub use self::traits::Buffer;
pub use self::traits::TypedBuffer;
pub use self::view::BufferView;

pub mod cpu_access;
pub mod device_local;
pub mod immutable;
pub mod sys;
pub mod traits;
pub mod view;

/// A subpart of a buffer.
///
/// This object doesn't correspond to any Vulkan object. It exists for the programmer's
/// convenience.
///
/// # Example
///
/// TODO: example
///
#[derive(Clone)]
pub struct BufferSlice<'a, T: ?Sized, B: 'a> {
    marker: PhantomData<T>,
    resource: &'a Arc<B>,
    offset: usize,
    size: usize,
}

impl<'a, T: ?Sized, B: 'a> BufferSlice<'a, T, B> {
    /// Returns the buffer that this slice belongs to.
    pub fn buffer(&self) -> &'a Arc<B> {
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
}

impl<'a, T, B: 'a> BufferSlice<'a, [T], B> {
    /// Returns the number of elements in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
}

impl<'a, T: ?Sized, B: 'a> From<&'a Arc<B>> for BufferSlice<'a, T, B>
    where B: TypedBuffer<Content = T>, T: 'static
{
    #[inline]
    fn from(r: &'a Arc<B>) -> BufferSlice<'a, T, B> {
        BufferSlice {
            marker: PhantomData,
            resource: r,
            offset: 0,
            size: r.size(),
        }
    }
}

impl<'a, T, B: 'a> From<BufferSlice<'a, T, B>> for BufferSlice<'a, [T], B>
    where T: 'static
{
    #[inline]
    fn from(r: BufferSlice<'a, T, B>) -> BufferSlice<'a, [T], B> {
        BufferSlice {
            marker: PhantomData,
            resource: r.resource,
            offset: r.offset,
            size: r.size,
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO: restore these tests
    /*use std::mem;

    use buffer::Usage;
    use buffer::Buffer;
    use buffer::BufferView;
    use buffer::BufferViewCreationError;
    use memory::DeviceLocal;

    #[test]
    fn create() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = Buffer::<[i8; 16], _>::new(&device, &Usage::all(), DeviceLocal, &queue).unwrap();
    }

    #[test]
    fn array_len() {
        let (device, queue) = gfx_dev_and_queue!();

        let b = Buffer::<[i16], _>::array(&device, 12, &Usage::all(),
                                          DeviceLocal, &queue).unwrap();
        assert_eq!(b.len(), 12);
        assert_eq!(b.size(), 12 * mem::size_of::<i16>());
    }*/
}
