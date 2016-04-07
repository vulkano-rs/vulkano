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
//! The `Buffer` struct has two template parameters:
//!
//! - `T` is the type of data that is contained in the buffer. It can be a struct
//!   (eg. `Foo`), an array (eg. `[u16; 1024]` or `[Foo; 1024]`), or an unsized array (eg. `[u16]`).
//!
//! - `M` is the object that provides memory and handles synchronization for the buffer.
//!   If the `CpuAccessible` and/or `CpuWriteAccessible` traits are implemented on `M`, then you
//!   can access the buffer's content from your program.
//!
//!
//! # Strong typing
//! 
//! All buffers take a template parameter that indicates their content.
//! 
//! # Memory
//! 
//! Creating a buffer requires passing an object that will be used by this library to provide
//! memory to the buffer.
//! 
//! All accesses to the memory are done through the `Buffer` object.
//! 
//! TODO: proof read this section
//!
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

pub use self::sys::Usage;
pub use self::traits::Buffer;
pub use self::traits::TypedBuffer;
pub use self::view::BufferView;

pub mod cpu_access;
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
