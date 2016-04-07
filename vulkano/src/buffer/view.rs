// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! View of a buffer, in order to use it as a uniform texel buffer or storage texel buffer.
//! 
//! In order to use a buffer as a uniform texel buffer or a storage texel buffer, you have to
//! create a `BufferView`, which indicates which format the data is in.
//! 
//! In order to create a view from a buffer, the buffer must have been created with either the
//! `uniform_texel_buffer` or the `storage_texel_buffer` usage.
//!
//! # Example
//!
//! ```no_run
//! # use std::sync::Arc;
//! use vulkano::buffer::immutable::ImmutableBuffer;
//! use vulkano::buffer::sys::Usage;
//! use vulkano::buffer::BufferView;
//! use vulkano::format;
//!
//! # let device: Arc<vulkano::device::Device> = unsafe { std::mem::uninitialized() };
//! # let queue: Arc<vulkano::device::Queue> = unsafe { std::mem::uninitialized() };
//! let usage = Usage {
//!     storage_texel_buffer: true,
//!     .. Usage::none()
//! };
//!
//! let buffer = ImmutableBuffer::<[u32]>::array(&device, 128, &usage,
//!                                              Some(queue.family())).unwrap();
//! let _view = BufferView::new(&buffer, format::R32Uint).unwrap();
//! ```

use std::marker::PhantomData;
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use format::FormatDesc;
use format::StrongStorage;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
pub struct BufferView<F, B> where B: Buffer {
    view: vk::BufferView,
    buffer: Arc<B>,
    marker: PhantomData<F>,
    atomic_accesses: bool,
}

impl<F, B> BufferView<F, B> where B: Buffer {
    /// Builds a new buffer view.
    #[inline]
    pub fn new<'a, S>(buffer: S, format: F)
                      -> Result<Arc<BufferView<F, B>>, BufferViewCreationError>
        where S: Into<BufferSlice<'a, [F::Pixel], B>>, B: 'static, F: StrongStorage + 'static
    {
        unsafe {
            BufferView::unchecked(buffer, format)
        }
    }

    /// Builds a new buffer view without checking that the format is correct.
    pub unsafe fn unchecked<'a, S, T: ?Sized>(buffer: S, format: F)
                                              -> Result<Arc<BufferView<F, B>>,
                                                        BufferViewCreationError>
        where S: Into<BufferSlice<'a, T, B>>, B: 'static, T: 'static, F: FormatDesc + 'static
    {
        let buffer = buffer.into();
        let device = buffer.resource.inner_buffer().device();
        let format = format.format();

        // TODO: check minTexelBufferOffsetAlignment

        if !buffer.buffer().inner_buffer().usage_uniform_texel_buffer() &&
           !buffer.buffer().inner_buffer().usage_storage_texel_buffer()
        {
            return Err(BufferViewCreationError::WrongBufferUsage);
        }

        let format_props = {
            let vk_i = device.instance().pointers();
            let mut output = mem::uninitialized();
            vk_i.GetPhysicalDeviceFormatProperties(device.physical_device().internal_object(),
                                                   format as u32, &mut output);
            output.bufferFeatures
        };

        if buffer.buffer().inner_buffer().usage_uniform_texel_buffer() {
            if (format_props & vk::FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) == 0 {
                return Err(BufferViewCreationError::UnsupportedFormat);
            }
        }

        if buffer.buffer().inner_buffer().usage_storage_texel_buffer() {
            if (format_props & vk::FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) == 0 {
                return Err(BufferViewCreationError::UnsupportedFormat);
            }
        }

        let infos = vk::BufferViewCreateInfo {
            sType: vk::STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved,
            buffer: buffer.resource.inner_buffer().internal_object(),
            format: format as u32,
            offset: buffer.offset as u64,
            range: buffer.size as u64,
        };

        let view = {
            let vk = device.pointers();
            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateBufferView(device.internal_object(), &infos,
                                                  ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(BufferView {
            view: view,
            buffer: buffer.resource.clone(),
            marker: PhantomData,
            atomic_accesses: (format_props &
                              vk::FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) != 0,
        }))
    }

    /// Returns true if the buffer view can be used as a uniform texel buffer.
    #[inline]
    pub fn uniform_texel_buffer(&self) -> bool {
        self.buffer.inner_buffer().usage_uniform_texel_buffer()
    }

    /// Returns true if the buffer view can be used as a storage texel buffer.
    #[inline]
    pub fn storage_texel_buffer(&self) -> bool {
        self.buffer.inner_buffer().usage_storage_texel_buffer()
    }

    /// Returns true if the buffer view can be used as a storage texel buffer with atomic accesses.
    #[inline]
    pub fn storage_texel_buffer_atomic(&self) -> bool {
        self.atomic_accesses && self.storage_texel_buffer()
    }
}

unsafe impl<F, B> VulkanObject for BufferView<F, B> where B: Buffer {
    type Object = vk::BufferView;

    #[inline]
    fn internal_object(&self) -> vk::BufferView {
        self.view
    }
}

impl<F, B> Drop for BufferView<F, B> where B: Buffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.buffer.inner_buffer().device().pointers();
            vk.DestroyBufferView(self.buffer.inner_buffer().device().internal_object(), self.view,
                                 ptr::null());
        }
    }
}

/// Error that can happen when creating a buffer view.
#[derive(Debug, Copy, Clone)]
pub enum BufferViewCreationError {
    /// Out of memory.
    OomError(OomError),

    /// The buffer was not creating with one of the `storage_texel_buffer` or
    /// `uniform_texel_buffer` usages.
    WrongBufferUsage,

    /// The requested format is not supported for this usage.
    UnsupportedFormat,
}

impl error::Error for BufferViewCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            BufferViewCreationError::OomError(_) => "out of memory when creating buffer view",
            BufferViewCreationError::WrongBufferUsage => "the buffer is missing correct usage \
                                                          flags",
            BufferViewCreationError::UnsupportedFormat => "the requested format is not supported \
                                                           for this usage",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            BufferViewCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for BufferViewCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for BufferViewCreationError {
    #[inline]
    fn from(err: OomError) -> BufferViewCreationError {
        BufferViewCreationError::OomError(err)
    }
}

impl From<Error> for BufferViewCreationError {
    #[inline]
    fn from(err: Error) -> BufferViewCreationError {
        OomError::from(err).into()
    }
}

#[cfg(test)]
mod tests {
    use buffer::BufferView;
    use buffer::sys::Usage;
    use buffer::view::BufferViewCreationError;
    use buffer::immutable::ImmutableBuffer;
    use format;

    #[test]
    fn create_uniform() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, queue) = gfx_dev_and_queue!();

        let usage = Usage {
            uniform_texel_buffer: true,
            .. Usage::none()
        };

        let buffer = ImmutableBuffer::<[[u8; 4]]>::array(&device, 128, &usage,
                                                         Some(queue.family())).unwrap();
        let view = BufferView::new(&buffer, format::R8G8B8A8Unorm).unwrap();

        assert!(view.uniform_texel_buffer());
    }

    #[test]
    fn create_storage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, queue) = gfx_dev_and_queue!();

        let usage = Usage {
            storage_texel_buffer: true,
            .. Usage::none()
        };

        let buffer = ImmutableBuffer::<[[u8; 4]]>::array(&device, 128, &usage,
                                                         Some(queue.family())).unwrap();
        let view = BufferView::new(&buffer, format::R8G8B8A8Unorm).unwrap();

        assert!(view.storage_texel_buffer());
    }

    #[test]
    fn create_storage_atomic() {
        // `VK_FORMAT_R32_UINT` guaranteed to be a supported format for atomics
        let (device, queue) = gfx_dev_and_queue!();

        let usage = Usage {
            storage_texel_buffer: true,
            .. Usage::none()
        };

        let buffer = ImmutableBuffer::<[u32]>::array(&device, 128, &usage,
                                                     Some(queue.family())).unwrap();
        let view = BufferView::new(&buffer, format::R32Uint).unwrap();

        assert!(view.storage_texel_buffer());
        assert!(view.storage_texel_buffer_atomic());
    }

    #[test]
    fn wrong_usage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = ImmutableBuffer::<[[u8; 4]]>::array(&device, 128, &Usage::none(),
                                                         Some(queue.family())).unwrap();

        match BufferView::new(&buffer, format::R8G8B8A8Unorm) {
            Err(BufferViewCreationError::WrongBufferUsage) => (),
            _ => panic!()
        }
    }

    #[test]
    fn unsupported_format() {
        let (device, queue) = gfx_dev_and_queue!();

        let usage = Usage {
            uniform_texel_buffer: true,
            storage_texel_buffer: true,
            .. Usage::none()
        };

        let buffer = ImmutableBuffer::<[[f64; 4]]>::array(&device, 128, &usage,
                                                          Some(queue.family())).unwrap();

        // TODO: what if R64G64B64A64Sfloat is supported?
        match BufferView::new(&buffer, format::R64G64B64A64Sfloat) {
            Err(BufferViewCreationError::UnsupportedFormat) => (),
            _ => panic!()
        }
    }
}
