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
//! use vulkano::buffer::view::{BufferView, BufferViewCreateInfo};
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
//! let _view = BufferView::new(
//!     buffer,
//!     BufferViewCreateInfo {
//!         format: Some(Format::R32_UINT),
//!         ..Default::default()
//!     },
//! ).unwrap();
//! ```

use crate::buffer::{BufferAccess, BufferAccessObject, BufferInner};
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::format::Format;
use crate::format::FormatFeatures;
use crate::DeviceSize;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use crate::{check_errors, Version};
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
#[derive(Debug)]
pub struct BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    handle: ash::vk::BufferView,
    buffer: Arc<B>,

    format: Option<Format>,
    format_features: FormatFeatures,
}

impl<B> BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    /// Creates a new `BufferView`.
    pub fn new(
        buffer: Arc<B>,
        create_info: BufferViewCreateInfo,
    ) -> Result<Arc<BufferView<B>>, BufferViewCreationError> {
        let BufferViewCreateInfo { format, _ne: _ } = create_info;

        let device = buffer.device();
        let properties = device.physical_device().properties();
        let range = buffer.size();
        let BufferInner {
            buffer: inner_buffer,
            offset,
        } = buffer.inner();

        // No VUID, but seems sensible?
        let format = format.unwrap();

        // VUID-VkBufferViewCreateInfo-buffer-00932
        if !(inner_buffer.usage().uniform_texel_buffer || inner_buffer.usage().storage_texel_buffer)
        {
            return Err(BufferViewCreationError::BufferMissingUsage);
        }

        let format_features = device
            .physical_device()
            .format_properties(format)
            .buffer_features;

        // VUID-VkBufferViewCreateInfo-buffer-00933
        if inner_buffer.usage().uniform_texel_buffer && !format_features.uniform_texel_buffer {
            return Err(BufferViewCreationError::UnsupportedFormat);
        }

        // VUID-VkBufferViewCreateInfo-buffer-00934
        if inner_buffer.usage().storage_texel_buffer && !format_features.storage_texel_buffer {
            return Err(BufferViewCreationError::UnsupportedFormat);
        }

        let block_size = format.block_size().unwrap();
        let texels_per_block = format.texels_per_block();

        // VUID-VkBufferViewCreateInfo-range-00929
        if range % block_size != 0 {
            return Err(BufferViewCreationError::RangeNotAligned {
                range,
                required_alignment: block_size,
            });
        }

        // VUID-VkBufferViewCreateInfo-range-00930
        if ((range / block_size) * texels_per_block as DeviceSize) as u32
            > properties.max_texel_buffer_elements
        {
            return Err(BufferViewCreationError::MaxTexelBufferElementsExceeded);
        }

        if device.api_version() >= Version::V1_3 || device.enabled_features().texel_buffer_alignment
        {
            let element_size = if block_size % 3 == 0 {
                block_size / 3
            } else {
                block_size
            };

            if inner_buffer.usage().storage_texel_buffer {
                let mut required_alignment = properties
                    .storage_texel_buffer_offset_alignment_bytes
                    .unwrap();

                if properties
                    .storage_texel_buffer_offset_single_texel_alignment
                    .unwrap()
                {
                    required_alignment = required_alignment.min(element_size);
                }

                // VUID-VkBufferViewCreateInfo-buffer-02750
                if offset % required_alignment != 0 {
                    return Err(BufferViewCreationError::OffsetNotAligned {
                        offset,
                        required_alignment,
                    });
                }
            }

            if inner_buffer.usage().uniform_texel_buffer {
                let mut required_alignment = properties
                    .uniform_texel_buffer_offset_alignment_bytes
                    .unwrap();

                if properties
                    .uniform_texel_buffer_offset_single_texel_alignment
                    .unwrap()
                {
                    required_alignment = required_alignment.min(element_size);
                }

                // VUID-VkBufferViewCreateInfo-buffer-02751
                if offset % required_alignment != 0 {
                    return Err(BufferViewCreationError::OffsetNotAligned {
                        offset,
                        required_alignment,
                    });
                }
            }
        } else {
            let required_alignment = properties.min_texel_buffer_offset_alignment;

            // VUID-VkBufferViewCreateInfo-offset-02749
            if offset % required_alignment != 0 {
                return Err(BufferViewCreationError::OffsetNotAligned {
                    offset,
                    required_alignment,
                });
            }
        }

        let create_info = ash::vk::BufferViewCreateInfo {
            flags: ash::vk::BufferViewCreateFlags::empty(),
            buffer: inner_buffer.internal_object(),
            format: format.into(),
            offset,
            range,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_buffer_view(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(BufferView {
            handle,
            buffer,

            format: Some(format),
            format_features,
        }))
    }

    /// Returns the buffer associated to this view.
    #[inline]
    pub fn buffer(&self) -> &Arc<B> {
        &self.buffer
    }
}

impl<B> Drop for BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.buffer.inner().buffer.device().fns();
            fns.v1_0.destroy_buffer_view(
                self.buffer.inner().buffer.device().internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl<B> VulkanObject for BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    type Object = ash::vk::BufferView;

    #[inline]
    fn internal_object(&self) -> ash::vk::BufferView {
        self.handle
    }
}

unsafe impl<B> DeviceOwned for BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

impl<B> PartialEq for BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl<B> Eq for BufferView<B> where B: BufferAccess + ?Sized {}

impl<B> Hash for BufferView<B>
where
    B: BufferAccess + ?Sized,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Parameters to create a new `BufferView`.
#[derive(Clone, Debug)]
pub struct BufferViewCreateInfo {
    /// The format of the buffer view.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    pub _ne: crate::NonExhaustive,
}

impl Default for BufferViewCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            format: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Error that can happen when creating a buffer view.
#[derive(Debug, Copy, Clone)]
pub enum BufferViewCreationError {
    /// Out of memory.
    OomError(OomError),

    /// The buffer was not created with one of the `storage_texel_buffer` or
    /// `uniform_texel_buffer` usages.
    BufferMissingUsage,

    /// The offset within the buffer is not a multiple of the required alignment.
    OffsetNotAligned {
        offset: DeviceSize,
        required_alignment: DeviceSize,
    },

    /// The range within the buffer is not a multiple of the required alignment.
    RangeNotAligned {
        range: DeviceSize,
        required_alignment: DeviceSize,
    },

    /// The requested format is not supported for this usage.
    UnsupportedFormat,

    /// The `max_texel_buffer_elements` limit has been exceeded.
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
        match *self {
            BufferViewCreationError::OomError(_) => write!(
                fmt,
                "out of memory when creating buffer view",
            ),
            BufferViewCreationError::BufferMissingUsage => write!(
                fmt,
                "the buffer was not created with one of the `storage_texel_buffer` or `uniform_texel_buffer` usages",
            ),
            BufferViewCreationError::OffsetNotAligned { .. } => write!(
                fmt,
                "the offset within the buffer is not a multiple of the required alignment",
            ),
            BufferViewCreationError::RangeNotAligned { .. } => write!(
                fmt,
                "the range within the buffer is not a multiple of the required alignment",
            ),
            BufferViewCreationError::UnsupportedFormat => write!(
                fmt,
                "the requested format is not supported for this usage",
            ),
            BufferViewCreationError::MaxTexelBufferElementsExceeded => write!(
                fmt,
                "the `max_texel_buffer_elements` limit has been exceeded",
            ),
        }
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

pub unsafe trait BufferViewAbstract:
    VulkanObject<Object = ash::vk::BufferView> + DeviceOwned + Send + Sync
{
    /// Returns the wrapped buffer that this buffer view was created from.
    fn buffer(&self) -> Arc<dyn BufferAccess>;

    /// Returns the format of the buffer view.
    fn format(&self) -> Option<Format>;

    /// Returns the features supported by the buffer view's format.
    fn format_features(&self) -> &FormatFeatures;
}

unsafe impl<B> BufferViewAbstract for BufferView<B>
where
    B: BufferAccess + ?Sized + 'static,
    Arc<B>: BufferAccessObject,
{
    #[inline]
    fn buffer(&self) -> Arc<dyn BufferAccess> {
        self.buffer.as_buffer_access_object()
    }

    #[inline]
    fn format(&self) -> Option<Format> {
        self.format
    }

    #[inline]
    fn format_features(&self) -> &FormatFeatures {
        &self.format_features
    }
}

impl PartialEq for dyn BufferViewAbstract {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.internal_object() == other.internal_object() && self.device() == other.device()
    }
}

impl Eq for dyn BufferViewAbstract {}

impl Hash for dyn BufferViewAbstract {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_object().hash(state);
        self.device().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::immutable::ImmutableBuffer;
    use crate::buffer::view::{BufferView, BufferViewCreateInfo, BufferViewCreationError};
    use crate::buffer::BufferUsage;
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
        let view = BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Some(Format::R8G8B8A8_UNORM),
                ..Default::default()
            },
        )
        .unwrap();
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
        BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Some(Format::R8G8B8A8_UNORM),
                ..Default::default()
            },
        )
        .unwrap();
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
        BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Some(Format::R32_UINT),
                ..Default::default()
            },
        )
        .unwrap();
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

        match BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Some(Format::R8G8B8A8_UNORM),
                ..Default::default()
            },
        ) {
            Err(BufferViewCreationError::BufferMissingUsage) => (),
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

        // TODO: what if R64G64B64A64_SFLOAT is supported?
        match BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Some(Format::R64G64B64A64_SFLOAT),
                ..Default::default()
            },
        ) {
            Err(BufferViewCreationError::UnsupportedFormat) => (),
            _ => panic!(),
        }
    }
}
