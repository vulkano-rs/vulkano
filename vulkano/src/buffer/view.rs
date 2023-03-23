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
//! # Examples
//!
//! ```
//! # use std::sync::Arc;
//! use vulkano::buffer::{Buffer, BufferAllocateInfo, BufferUsage};
//! use vulkano::buffer::view::{BufferView, BufferViewCreateInfo};
//! use vulkano::format::Format;
//!
//! # let queue: Arc<vulkano::device::Queue> = return;
//! # let memory_allocator: vulkano::memory::allocator::StandardMemoryAllocator = return;
//! let buffer = Buffer::new_slice::<u32>(
//!     &memory_allocator,
//!     BufferAllocateInfo {
//!         buffer_usage: BufferUsage::STORAGE_TEXEL_BUFFER,
//!         ..Default::default()
//!     },
//!     128,
//! )
//! .unwrap();
//!
//! let view = BufferView::new(
//!     buffer,
//!     BufferViewCreateInfo {
//!         format: Some(Format::R32_UINT),
//!         ..Default::default()
//!     },
//! )
//! .unwrap();
//! ```

use super::{BufferUsage, Subbuffer};
use crate::{
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    macros::impl_id_counter,
    memory::{is_aligned, DeviceAlignment},
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Range,
    ptr,
    sync::Arc,
};

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
#[derive(Debug)]
pub struct BufferView {
    handle: ash::vk::BufferView,
    subbuffer: Subbuffer<[u8]>,
    id: NonZeroU64,

    format: Option<Format>,
    format_features: FormatFeatures,
    range: Range<DeviceSize>,
}

impl BufferView {
    /// Creates a new `BufferView`.
    #[inline]
    pub fn new(
        subbuffer: Subbuffer<impl ?Sized>,
        create_info: BufferViewCreateInfo,
    ) -> Result<Arc<BufferView>, BufferViewCreationError> {
        Self::new_inner(subbuffer.into_bytes(), create_info)
    }

    fn new_inner(
        subbuffer: Subbuffer<[u8]>,
        create_info: BufferViewCreateInfo,
    ) -> Result<Arc<BufferView>, BufferViewCreationError> {
        let BufferViewCreateInfo { format, _ne: _ } = create_info;

        let buffer = subbuffer.buffer();
        let device = buffer.device();
        let properties = device.physical_device().properties();
        let size = subbuffer.size();
        let offset = subbuffer.offset();

        // No VUID, but seems sensible?
        let format = format.unwrap();

        // VUID-VkBufferViewCreateInfo-format-parameter
        format.validate_device(device)?;

        // VUID-VkBufferViewCreateInfo-buffer-00932
        if !buffer
            .usage()
            .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER)
        {
            return Err(BufferViewCreationError::BufferMissingUsage);
        }

        // Use unchecked, because all validation has been done above.
        let format_features = unsafe {
            device
                .physical_device()
                .format_properties_unchecked(format)
                .buffer_features
        };

        // VUID-VkBufferViewCreateInfo-buffer-00933
        if buffer.usage().intersects(BufferUsage::UNIFORM_TEXEL_BUFFER)
            && !format_features.intersects(FormatFeatures::UNIFORM_TEXEL_BUFFER)
        {
            return Err(BufferViewCreationError::UnsupportedFormat);
        }

        // VUID-VkBufferViewCreateInfo-buffer-00934
        if buffer.usage().intersects(BufferUsage::STORAGE_TEXEL_BUFFER)
            && !format_features.intersects(FormatFeatures::STORAGE_TEXEL_BUFFER)
        {
            return Err(BufferViewCreationError::UnsupportedFormat);
        }

        let block_size = format.block_size().unwrap();
        let texels_per_block = format.texels_per_block();

        // VUID-VkBufferViewCreateInfo-range-00929
        if size % block_size != 0 {
            return Err(BufferViewCreationError::RangeNotAligned {
                range: size,
                required_alignment: block_size,
            });
        }

        // VUID-VkBufferViewCreateInfo-range-00930
        if ((size / block_size) * texels_per_block as DeviceSize) as u32
            > properties.max_texel_buffer_elements
        {
            return Err(BufferViewCreationError::MaxTexelBufferElementsExceeded);
        }

        if device.api_version() >= Version::V1_3 || device.enabled_features().texel_buffer_alignment
        {
            let element_size = DeviceAlignment::new(if block_size % 3 == 0 {
                block_size / 3
            } else {
                block_size
            })
            .unwrap();

            if buffer.usage().intersects(BufferUsage::STORAGE_TEXEL_BUFFER) {
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
                if !is_aligned(offset, required_alignment) {
                    return Err(BufferViewCreationError::OffsetNotAligned {
                        offset,
                        required_alignment,
                    });
                }
            }

            if buffer.usage().intersects(BufferUsage::UNIFORM_TEXEL_BUFFER) {
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
                if !is_aligned(offset, required_alignment) {
                    return Err(BufferViewCreationError::OffsetNotAligned {
                        offset,
                        required_alignment,
                    });
                }
            }
        } else {
            let required_alignment = properties.min_texel_buffer_offset_alignment;

            // VUID-VkBufferViewCreateInfo-offset-02749
            if !is_aligned(offset, required_alignment) {
                return Err(BufferViewCreationError::OffsetNotAligned {
                    offset,
                    required_alignment,
                });
            }
        }

        let create_info = ash::vk::BufferViewCreateInfo {
            flags: ash::vk::BufferViewCreateFlags::empty(),
            buffer: buffer.handle(),
            format: format.into(),
            offset,
            range: size,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_buffer_view)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(BufferView {
            handle,
            subbuffer,
            id: Self::next_id(),
            format: Some(format),
            format_features,
            range: 0..size,
        }))
    }

    /// Returns the buffer associated to this view.
    #[inline]
    pub fn buffer(&self) -> &Subbuffer<[u8]> {
        &self.subbuffer
    }

    /// Returns the format of this view.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.format
    }

    /// Returns the features supported by this view’s format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns the byte range of the wrapped buffer that this view exposes.
    #[inline]
    pub fn range(&self) -> Range<DeviceSize> {
        self.range.clone()
    }
}

impl Drop for BufferView {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.subbuffer.device().fns();
            (fns.v1_0.destroy_buffer_view)(
                self.subbuffer.device().handle(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl VulkanObject for BufferView {
    type Handle = ash::vk::BufferView;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for BufferView {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.subbuffer.device()
    }
}

impl_id_counter!(BufferView);

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

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The buffer was not created with one of the `storage_texel_buffer` or
    /// `uniform_texel_buffer` usages.
    BufferMissingUsage,

    /// The offset within the buffer is not a multiple of the required alignment.
    OffsetNotAligned {
        offset: DeviceSize,
        required_alignment: DeviceAlignment,
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

impl Error for BufferViewCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            BufferViewCreationError::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for BufferViewCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "out of memory when creating buffer view"),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::BufferMissingUsage => write!(
                f,
                "the buffer was not created with one of the `storage_texel_buffer` or \
                `uniform_texel_buffer` usages",
            ),
            Self::OffsetNotAligned { .. } => write!(
                f,
                "the offset within the buffer is not a multiple of the required alignment",
            ),
            Self::RangeNotAligned { .. } => write!(
                f,
                "the range within the buffer is not a multiple of the required alignment",
            ),
            Self::UnsupportedFormat => {
                write!(f, "the requested format is not supported for this usage")
            }
            Self::MaxTexelBufferElementsExceeded => {
                write!(f, "the `max_texel_buffer_elements` limit has been exceeded")
            }
        }
    }
}

impl From<OomError> for BufferViewCreationError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<VulkanError> for BufferViewCreationError {
    fn from(err: VulkanError) -> Self {
        OomError::from(err).into()
    }
}

impl From<RequirementNotMet> for BufferViewCreationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferView, BufferViewCreateInfo, BufferViewCreationError};
    use crate::{
        buffer::{Buffer, BufferAllocateInfo, BufferUsage},
        format::Format,
        memory::allocator::{MemoryUsage, StandardMemoryAllocator},
    };

    #[test]
    fn create_uniform() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer = Buffer::new_slice::<[u8; 4]>(
            &memory_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_TEXEL_BUFFER,
                memory_usage: MemoryUsage::GpuOnly,
                ..Default::default()
            },
            128,
        )
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
    fn create_storage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer = Buffer::new_slice::<[u8; 4]>(
            &memory_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::STORAGE_TEXEL_BUFFER,
                memory_usage: MemoryUsage::GpuOnly,
                ..Default::default()
            },
            128,
        )
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
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer = Buffer::new_slice::<u32>(
            &memory_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::STORAGE_TEXEL_BUFFER,
                memory_usage: MemoryUsage::GpuOnly,
                ..Default::default()
            },
            128,
        )
        .unwrap();
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
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer = Buffer::new_slice::<[u8; 4]>(
            &memory_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::TRANSFER_DST, // Dummy value
                memory_usage: MemoryUsage::GpuOnly,
                ..Default::default()
            },
            128,
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
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = StandardMemoryAllocator::new_default(device);

        let buffer = Buffer::new_slice::<[f64; 4]>(
            &memory_allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER,
                memory_usage: MemoryUsage::GpuOnly,
                ..Default::default()
            },
            128,
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
