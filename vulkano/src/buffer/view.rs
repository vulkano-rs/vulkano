// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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
//! ```
//! # use std::sync::Arc;
//! use vulkano::buffer::immutable::ImmutableBuffer;
//! use vulkano::buffer::BufferUsage;
//! use vulkano::buffer::BufferView;
//! use vulkano::format::Format;
//!
//! # let device: Arc<vulkano::device::Device> = return;
//! # let queue: Arc<vulkano::device::Queue> = return;
//! let usage = BufferUsage {
//!     storage_texel_buffer: true,
//!     .. BufferUsage::none()
//! };
//!
//! let (buffer, _future) = ImmutableBuffer::<[u32]>::from_iter((0..128).map(|n| n), usage,
//!                                                             queue.clone()).unwrap();
//! let _view = BufferView::new(buffer, Format::R32Uint).unwrap();
//! ```

use crate::buffer::BufferAccess;
use crate::buffer::BufferInner;
use crate::buffer::TypedBufferAccess;
use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::format::Format;
use crate::format::Pixel;
use crate::Error;
use crate::OomError;
use crate::SafeDeref;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
pub struct BufferView<B>
where
    B: BufferAccess,
{
    view: ash::vk::BufferView,
    buffer: B,
    atomic_accesses: bool,
}

impl<B> BufferView<B>
where
    B: BufferAccess,
{
    /// Builds a new buffer view.
    #[inline]
    pub fn new<Px>(buffer: B, format: Format) -> Result<BufferView<B>, BufferViewCreationError>
    where
        B: TypedBufferAccess<Content = [Px]>,
        Px: Pixel,
    {
        unsafe { BufferView::unchecked(buffer, format) }
    }

    /// Builds a new buffer view without checking that the format is correct.
    pub unsafe fn unchecked(
        org_buffer: B,
        format: Format,
    ) -> Result<BufferView<B>, BufferViewCreationError>
    where
        B: BufferAccess,
    {
        let (view, format_props) = {
            let size = org_buffer.size();
            let BufferInner { buffer, offset } = org_buffer.inner();

            let device = buffer.device();

            if (offset
                % device
                    .physical_device()
                    .properties()
                    .min_texel_buffer_offset_alignment
                    .unwrap())
                != 0
            {
                return Err(BufferViewCreationError::WrongBufferAlignment);
            }

            if !buffer.usage().uniform_texel_buffer && !buffer.usage().storage_texel_buffer {
                return Err(BufferViewCreationError::WrongBufferUsage);
            }

            {
                let nb = size
                    / format
                        .size()
                        .expect("Can't use a compressed format for buffer views");
                let l = device
                    .physical_device()
                    .properties()
                    .max_texel_buffer_elements
                    .unwrap();
                if nb as u32 > l {
                    return Err(BufferViewCreationError::MaxTexelBufferElementsExceeded);
                }
            }

            let format_props = {
                let fns_i = device.instance().fns();
                let mut output = MaybeUninit::uninit();
                fns_i.v1_0.get_physical_device_format_properties(
                    device.physical_device().internal_object(),
                    format.into(),
                    output.as_mut_ptr(),
                );
                output.assume_init().buffer_features
            };

            if buffer.usage().uniform_texel_buffer {
                if (format_props & ash::vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER).is_empty() {
                    return Err(BufferViewCreationError::UnsupportedFormat);
                }
            }

            if buffer.usage().storage_texel_buffer {
                if (format_props & ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER).is_empty() {
                    return Err(BufferViewCreationError::UnsupportedFormat);
                }
            }

            let infos = ash::vk::BufferViewCreateInfo {
                flags: ash::vk::BufferViewCreateFlags::empty(),
                buffer: buffer.internal_object(),
                format: format.into(),
                offset,
                range: size,
                ..Default::default()
            };

            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_buffer_view(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            (output.assume_init(), format_props)
        };

        Ok(BufferView {
            view,
            buffer: org_buffer,
            atomic_accesses: !(format_props
                & ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER_ATOMIC)
                .is_empty(),
        })
    }

    /// Returns the buffer associated to this view.
    #[inline]
    pub fn buffer(&self) -> &B {
        &self.buffer
    }

    /// Returns true if the buffer view can be used as a uniform texel buffer.
    #[inline]
    pub fn uniform_texel_buffer(&self) -> bool {
        self.buffer.inner().buffer.usage().uniform_texel_buffer
    }

    /// Returns true if the buffer view can be used as a storage texel buffer.
    #[inline]
    pub fn storage_texel_buffer(&self) -> bool {
        self.buffer.inner().buffer.usage().storage_texel_buffer
    }

    /// Returns true if the buffer view can be used as a storage texel buffer with atomic accesses.
    #[inline]
    pub fn storage_texel_buffer_atomic(&self) -> bool {
        self.atomic_accesses && self.storage_texel_buffer()
    }
}

unsafe impl<B> VulkanObject for BufferView<B>
where
    B: BufferAccess,
{
    type Object = ash::vk::BufferView;

    #[inline]
    fn internal_object(&self) -> ash::vk::BufferView {
        self.view
    }
}

unsafe impl<B> DeviceOwned for BufferView<B>
where
    B: BufferAccess,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

impl<B> fmt::Debug for BufferView<B>
where
    B: BufferAccess + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("BufferView")
            .field("raw", &self.view)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl<B> Drop for BufferView<B>
where
    B: BufferAccess,
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.buffer.inner().buffer.device().fns();
            fns.v1_0.destroy_buffer_view(
                self.buffer.inner().buffer.device().internal_object(),
                self.view,
                ptr::null(),
            );
        }
    }
}

pub unsafe trait BufferViewRef {
    type BufferAccess: BufferAccess;

    fn view(&self) -> &BufferView<Self::BufferAccess>;
}

unsafe impl<B> BufferViewRef for BufferView<B>
where
    B: BufferAccess,
{
    type BufferAccess = B;

    #[inline]
    fn view(&self) -> &BufferView<B> {
        self
    }
}

unsafe impl<T, B> BufferViewRef for T
where
    T: SafeDeref<Target = BufferView<B>>,
    B: BufferAccess,
{
    type BufferAccess = B;

    #[inline]
    fn view(&self) -> &BufferView<B> {
        &**self
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

    /// The offset within the buffer is not a multiple of the `min_texel_buffer_offset_alignment`
    /// limit.
    WrongBufferAlignment,

    /// The requested format is not supported for this usage.
    UnsupportedFormat,

    /// The maximum number of elements in the buffer view has been exceeded.
    MaxTexelBufferElementsExceeded,
}

impl error::Error for BufferViewCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            BufferViewCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for BufferViewCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                BufferViewCreationError::OomError(_) => "out of memory when creating buffer view",
                BufferViewCreationError::WrongBufferUsage => {
                    "the buffer is missing correct usage flags"
                }
                BufferViewCreationError::WrongBufferAlignment => {
                    "the offset within the buffer is not a multiple of the
                 `min_texel_buffer_offset_alignment` limit"
                }
                BufferViewCreationError::UnsupportedFormat => {
                    "the requested format is not supported for this usage"
                }
                BufferViewCreationError::MaxTexelBufferElementsExceeded => {
                    "the maximum number of texel elements is exceeded"
                }
            }
        )
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
    use crate::buffer::immutable::ImmutableBuffer;
    use crate::buffer::view::BufferViewCreationError;
    use crate::buffer::BufferUsage;
    use crate::buffer::BufferView;
    use crate::format::Format;

    #[test]
    fn create_uniform() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, queue) = gfx_dev_and_queue!();

        let usage = BufferUsage {
            uniform_texel_buffer: true,
            ..BufferUsage::none()
        };

        let (buffer, _) =
            ImmutableBuffer::<[[u8; 4]]>::from_iter((0..128).map(|_| [0; 4]), usage, queue.clone())
                .unwrap();
        let view = BufferView::new(buffer, Format::R8G8B8A8Unorm).unwrap();

        assert!(view.uniform_texel_buffer());
    }

    #[test]
    fn create_storage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, queue) = gfx_dev_and_queue!();

        let usage = BufferUsage {
            storage_texel_buffer: true,
            ..BufferUsage::none()
        };

        let (buffer, _) =
            ImmutableBuffer::<[[u8; 4]]>::from_iter((0..128).map(|_| [0; 4]), usage, queue.clone())
                .unwrap();
        let view = BufferView::new(buffer, Format::R8G8B8A8Unorm).unwrap();

        assert!(view.storage_texel_buffer());
    }

    #[test]
    fn create_storage_atomic() {
        // `VK_FORMAT_R32_UINT` guaranteed to be a supported format for atomics
        let (device, queue) = gfx_dev_and_queue!();

        let usage = BufferUsage {
            storage_texel_buffer: true,
            ..BufferUsage::none()
        };

        let (buffer, _) =
            ImmutableBuffer::<[u32]>::from_iter((0..128).map(|_| 0), usage, queue.clone()).unwrap();
        let view = BufferView::new(buffer, Format::R32Uint).unwrap();

        assert!(view.storage_texel_buffer());
        assert!(view.storage_texel_buffer_atomic());
    }

    #[test]
    fn wrong_usage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, queue) = gfx_dev_and_queue!();

        let (buffer, _) = ImmutableBuffer::<[[u8; 4]]>::from_iter(
            (0..128).map(|_| [0; 4]),
            BufferUsage::none(),
            queue.clone(),
        )
        .unwrap();

        match BufferView::new(buffer, Format::R8G8B8A8Unorm) {
            Err(BufferViewCreationError::WrongBufferUsage) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn unsupported_format() {
        let (device, queue) = gfx_dev_and_queue!();

        let usage = BufferUsage {
            uniform_texel_buffer: true,
            storage_texel_buffer: true,
            ..BufferUsage::none()
        };

        let (buffer, _) = ImmutableBuffer::<[[f64; 4]]>::from_iter(
            (0..128).map(|_| [0.0; 4]),
            usage,
            queue.clone(),
        )
        .unwrap();

        // TODO: what if R64G64B64A64Sfloat is supported?
        match BufferView::new(buffer, Format::R64G64B64A64Sfloat) {
            Err(BufferViewCreationError::UnsupportedFormat) => (),
            _ => panic!(),
        }
    }
}
