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
//! use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
//! use vulkano::buffer::view::{BufferView, BufferViewCreateInfo};
//! use vulkano::format::Format;
//! use vulkano::memory::allocator::AllocationCreateInfo;
//!
//! # let queue: Arc<vulkano::device::Queue> = return;
//! # let memory_allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
//! let buffer = Buffer::new_slice::<u32>(
//!     memory_allocator.clone(),
//!     BufferCreateInfo {
//!         usage: BufferUsage::STORAGE_TEXEL_BUFFER,
//!         ..Default::default()
//!     },
//!     AllocationCreateInfo::default(),
//!     128,
//! )
//! .unwrap();
//!
//! let view = BufferView::new(
//!     buffer,
//!     BufferViewCreateInfo {
//!         format: Format::R32_UINT,
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
    DeviceSize, Validated, ValidationError, Version, VulkanError, VulkanObject,
};
use ash::vk;
use std::{mem::MaybeUninit, num::NonZeroU64, ops::Range, ptr, sync::Arc};

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
#[derive(Debug)]
pub struct BufferView {
    handle: vk::BufferView,
    subbuffer: Subbuffer<[u8]>,
    id: NonZeroU64,

    format: Format,
    format_features: FormatFeatures,
    range: Range<DeviceSize>,
}

impl BufferView {
    /// Creates a new `BufferView`.
    #[inline]
    pub fn new(
        subbuffer: Subbuffer<impl ?Sized>,
        create_info: BufferViewCreateInfo,
    ) -> Result<Arc<BufferView>, Validated<VulkanError>> {
        let subbuffer = subbuffer.into_bytes();
        Self::validate_new(&subbuffer, &create_info)?;

        Ok(unsafe { Self::new_unchecked(subbuffer, create_info) }?)
    }

    fn validate_new(
        subbuffer: &Subbuffer<[u8]>,
        create_info: &BufferViewCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        let device = subbuffer.device();

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        let &BufferViewCreateInfo { format, _ne: _ } = create_info;

        let buffer = subbuffer.buffer();
        let properties = device.physical_device().properties();

        let format_properties =
            unsafe { device.physical_device().format_properties_unchecked(format) };
        let format_features = format_properties.buffer_features;

        if !buffer
            .usage()
            .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "subbuffer".into(),
                problem: "was not created with the `BufferUsage::UNIFORM_TEXEL_BUFFER` \
                    or `BufferUsage::STORAGE_TEXEL_BUFFER` usage"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-buffer-00932"],
                ..Default::default()
            }));
        }

        if buffer.usage().intersects(BufferUsage::UNIFORM_TEXEL_BUFFER)
            && !format_features.intersects(FormatFeatures::UNIFORM_TEXEL_BUFFER)
        {
            return Err(Box::new(ValidationError {
                problem: "`subbuffer` was created with the `BufferUsage::UNIFORM_TEXEL_BUFFER` \
                    usage, but the format features of `create_info.format` do not include \
                    `FormatFeatures::UNIFORM_TEXEL_BUFFER`"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-buffer-00933"],
                ..Default::default()
            }));
        }

        if buffer.usage().intersects(BufferUsage::STORAGE_TEXEL_BUFFER)
            && !format_features.intersects(FormatFeatures::STORAGE_TEXEL_BUFFER)
        {
            return Err(Box::new(ValidationError {
                problem: "`subbuffer` was created with the `BufferUsage::STORAGE_TEXEL_BUFFER` \
                    usage, but the format features of `create_info.format` do not include \
                    `FormatFeatures::STORAGE_TEXEL_BUFFER`"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-buffer-00934"],
                ..Default::default()
            }));
        }

        let block_size = format.block_size();
        let texels_per_block = format.texels_per_block();

        if subbuffer.size() % block_size != 0 {
            return Err(Box::new(ValidationError {
                problem: "`subbuffer.size()` is not a multiple of \
                    `create_info.format.block_size()`"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-range-00929"],
                ..Default::default()
            }));
        }

        if ((subbuffer.size() / block_size) * texels_per_block as DeviceSize) as u32
            > properties.max_texel_buffer_elements
        {
            return Err(Box::new(ValidationError {
                problem: "`subbuffer.size() / create_info.format.block_size() * \
                    create_info.format.texels_per_block()` is greater than the \
                    `max_texel_buffer_elements` limit"
                    .into(),
                vuids: &["VUID-VkBufferViewCreateInfo-range-00930"],
                ..Default::default()
            }));
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
                if properties
                    .storage_texel_buffer_offset_single_texel_alignment
                    .unwrap()
                {
                    if !is_aligned(
                        subbuffer.offset(),
                        properties
                            .storage_texel_buffer_offset_alignment_bytes
                            .unwrap()
                            .min(element_size),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`subbuffer` was created with the \
                                `BufferUsage::STORAGE_TEXEL_BUFFER` usage, and the \
                                `storage_texel_buffer_offset_single_texel_alignment` \
                                property is `true`, but \
                                `subbuffer.offset()` is not a multiple of the \
                                minimum of `create_info.format.block_size()` and the \
                                `storage_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02750"],
                            ..Default::default()
                        }));
                    }
                } else {
                    if !is_aligned(
                        subbuffer.offset(),
                        properties
                            .storage_texel_buffer_offset_alignment_bytes
                            .unwrap(),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`subbuffer` was created with the \
                                `BufferUsage::STORAGE_TEXEL_BUFFER` usage, and the \
                                `storage_texel_buffer_offset_single_texel_alignment` \
                                property is `false`, but \
                                `subbuffer.offset()` is not a multiple of the \
                                `storage_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02750"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if buffer.usage().intersects(BufferUsage::UNIFORM_TEXEL_BUFFER) {
                if properties
                    .uniform_texel_buffer_offset_single_texel_alignment
                    .unwrap()
                {
                    if !is_aligned(
                        subbuffer.offset(),
                        properties
                            .uniform_texel_buffer_offset_alignment_bytes
                            .unwrap()
                            .min(element_size),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`subbuffer` was created with the \
                                `BufferUsage::UNIFORM_TEXEL_BUFFER` usage, and the \
                                `uniform_texel_buffer_offset_single_texel_alignment` \
                                property is `false`, but \
                                `subbuffer.offset()` is not a multiple of the \
                                minimum of `create_info.format.block_size()` and the \
                                `uniform_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02751"],
                            ..Default::default()
                        }));
                    }
                } else {
                    if !is_aligned(
                        subbuffer.offset(),
                        properties
                            .uniform_texel_buffer_offset_alignment_bytes
                            .unwrap(),
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`subbuffer` was created with the \
                                `BufferUsage::UNIFORM_TEXEL_BUFFER` usage, and the \
                                `uniform_texel_buffer_offset_single_texel_alignment` \
                                property is `false`, but \
                                `subbuffer.offset()` is not a multiple of the \
                                `uniform_texel_buffer_offset_alignment_bytes` limit"
                                .into(),
                            vuids: &["VUID-VkBufferViewCreateInfo-buffer-02751"],
                            ..Default::default()
                        }));
                    }
                }
            }
        } else {
            if !is_aligned(
                subbuffer.offset(),
                properties.min_texel_buffer_offset_alignment,
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subbuffer.offset()` is not a multiple of the \
                        `min_texel_buffer_offset_alignment` limit"
                        .into(),
                    vuids: &["VUID-VkBufferViewCreateInfo-offset-02749"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        subbuffer: Subbuffer<impl ?Sized>,
        create_info: BufferViewCreateInfo,
    ) -> Result<Arc<BufferView>, VulkanError> {
        let device = subbuffer.device();
        let create_info_vk = create_info.to_vk(subbuffer.as_bytes());

        let fns = device.fns();
        let handle = {
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_buffer_view)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(subbuffer, handle, create_info) })
    }

    /// Creates a new `BufferView` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `subbuffer` and `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        subbuffer: Subbuffer<impl ?Sized>,
        handle: vk::BufferView,
        create_info: BufferViewCreateInfo,
    ) -> Arc<BufferView> {
        let &BufferViewCreateInfo { format, _ne: _ } = &create_info;
        let size = subbuffer.size();
        let format_properties = unsafe {
            subbuffer
                .device()
                .physical_device()
                .format_properties_unchecked(format)
        };
        let format_features = format_properties.buffer_features;

        Arc::new(BufferView {
            handle,
            subbuffer: subbuffer.into_bytes(),
            id: Self::next_id(),
            format,
            format_features,
            range: 0..size,
        })
    }

    /// Returns the buffer associated to this view.
    #[inline]
    pub fn buffer(&self) -> &Subbuffer<[u8]> {
        &self.subbuffer
    }

    /// Returns the format of this view.
    #[inline]
    pub fn format(&self) -> Format {
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
        let fns = self.subbuffer.device().fns();
        unsafe {
            (fns.v1_0.destroy_buffer_view)(
                self.subbuffer.device().handle(),
                self.handle,
                ptr::null(),
            )
        };
    }
}

unsafe impl VulkanObject for BufferView {
    type Handle = vk::BufferView;

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
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

    pub _ne: crate::NonExhaustive,
}

impl Default for BufferViewCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            format: Format::UNDEFINED,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl BufferViewCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self { format, _ne: _ } = self;

        format.validate_device(device).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-VkBufferViewCreateInfo-format-parameter"])
        })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self, subbuffer: &Subbuffer<[u8]>) -> vk::BufferViewCreateInfo<'static> {
        let &Self { format, _ne: _ } = self;

        vk::BufferViewCreateInfo::default()
            .flags(vk::BufferViewCreateFlags::empty())
            .buffer(subbuffer.buffer().handle())
            .format(format.into())
            .offset(subbuffer.offset())
            .range(subbuffer.size())
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferView, BufferViewCreateInfo};
    use crate::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        format::Format,
        memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    };
    use std::sync::Arc;

    #[test]
    fn create_uniform() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let buffer = Buffer::new_slice::<[u8; 4]>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_TEXEL_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            128,
        )
        .unwrap();

        BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_storage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let buffer = Buffer::new_slice::<[u8; 4]>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_TEXEL_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            128,
        )
        .unwrap();
        BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_storage_atomic() {
        // `VK_FORMAT_R32_UINT` guaranteed to be a supported format for atomics
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let buffer = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_TEXEL_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            128,
        )
        .unwrap();
        BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Format::R32_UINT,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn wrong_usage() {
        // `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let buffer = Buffer::new_slice::<[u8; 4]>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST, // Dummy value
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            128,
        )
        .unwrap();

        match BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                ..Default::default()
            },
        ) {
            Err(_) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn unsupported_format() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let buffer = Buffer::new_slice::<[f64; 4]>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_TEXEL_BUFFER | BufferUsage::STORAGE_TEXEL_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            128,
        )
        .unwrap();

        // TODO: what if R64G64B64A64_SFLOAT is supported?
        match BufferView::new(
            buffer,
            BufferViewCreateInfo {
                format: Format::R64G64B64A64_SFLOAT,
                ..Default::default()
            },
        ) {
            Err(_) => (),
            _ => panic!(),
        }
    }
}
