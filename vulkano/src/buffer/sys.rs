// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low level implementation of buffers.
//!
//! Wraps directly around Vulkan buffers, with the exceptions of a few safety checks.
//!
//! The `UnsafeBuffer` type is the lowest-level buffer object provided by this library. It is used
//! internally by the higher-level buffer types. You are strongly encouraged to have excellent
//! knowledge of the Vulkan specs if you want to use an `UnsafeBuffer`.
//!
//! Here is what you must take care of when you use an `UnsafeBuffer`:
//!
//! - Synchronization, ie. avoid reading and writing simultaneously to the same buffer.
//! - Memory aliasing considerations. If you use the same memory to back multiple resources, you
//!   must ensure that they are not used together and must enable some additional flags.
//! - Binding memory correctly and only once. If you use sparse binding, respect the rules of
//!   sparse binding.
//! - Type safety.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::memory::DeviceMemory;
use crate::memory::DeviceMemoryAllocError;
use crate::memory::MemoryRequirements;
use crate::sync::Sharing;
use crate::DeviceSize;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use crate::{buffer::BufferUsage, Version};
use ash::vk::Handle;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Data storage in a GPU-accessible location.
pub struct UnsafeBuffer {
    buffer: ash::vk::Buffer,
    device: Arc<Device>,
    size: DeviceSize,
    usage: BufferUsage,
}

impl UnsafeBuffer {
    /// Creates a new buffer of the given size.
    ///
    /// See the module's documentation for information about safety.
    ///
    /// # Panic
    ///
    /// - Panics if `sparse.sparse` is false and `sparse.sparse_residency` or `sparse.sparse_aliased` is true.
    /// - Panics if `usage` is empty.
    ///
    pub unsafe fn new<'a, I>(
        device: Arc<Device>,
        size: DeviceSize,
        mut usage: BufferUsage,
        sharing: Sharing<I>,
        sparse: Option<SparseLevel>,
    ) -> Result<(UnsafeBuffer, MemoryRequirements), BufferCreationError>
    where
        I: Iterator<Item = u32>,
    {
        let fns = device.fns();

        // Ensure we're not trying to create an empty buffer.
        let size = if size == 0 {
            // To avoid panicking when allocating 0 bytes, use a 1-byte buffer.
            1
        } else {
            size
        };

        // Checking sparse features.
        let flags = if let Some(sparse_level) = sparse {
            if !device.enabled_features().sparse_binding {
                return Err(BufferCreationError::SparseBindingFeatureNotEnabled);
            }

            if sparse_level.sparse_residency && !device.enabled_features().sparse_residency_buffer {
                return Err(BufferCreationError::SparseResidencyBufferFeatureNotEnabled);
            }

            if sparse_level.sparse_aliased && !device.enabled_features().sparse_residency_aliased {
                return Err(BufferCreationError::SparseResidencyAliasedFeatureNotEnabled);
            }

            sparse_level.into()
        } else {
            ash::vk::BufferCreateFlags::empty()
        };

        if usage.device_address && !device.enabled_features().buffer_device_address {
            usage.device_address = false;
            if ash::vk::BufferUsageFlags::from(usage).is_empty() {
                // return an error iff device_address was the only requested usage and the
                // feature isn't enabled. Otherwise we'll hit that assert below.
                // TODO: This is weird, why not just return an error always if the feature is not enabled?
                // You can't use BufferUsage::all() anymore, but is that a good idea anyway?
                return Err(BufferCreationError::DeviceAddressFeatureNotEnabled);
            }
        }

        let usage_bits = ash::vk::BufferUsageFlags::from(usage);
        // Checking for empty BufferUsage.
        assert!(
            !usage_bits.is_empty(),
            "Can't create buffer with empty BufferUsage"
        );

        let buffer = {
            let (sh_mode, sh_indices) = match sharing {
                Sharing::Exclusive => {
                    (ash::vk::SharingMode::EXCLUSIVE, SmallVec::<[u32; 8]>::new())
                }
                Sharing::Concurrent(ids) => (ash::vk::SharingMode::CONCURRENT, ids.collect()),
            };

            let infos = ash::vk::BufferCreateInfo {
                flags,
                size,
                usage: usage_bits,
                sharing_mode: sh_mode,
                queue_family_index_count: sh_indices.len() as u32,
                p_queue_family_indices: sh_indices.as_ptr(),
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_buffer(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let mem_reqs = {
            #[inline]
            fn align(val: DeviceSize, al: DeviceSize) -> DeviceSize {
                al * (1 + (val - 1) / al)
            }

            let mut output = if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_get_memory_requirements2
            {
                let infos = ash::vk::BufferMemoryRequirementsInfo2 {
                    buffer: buffer,
                    ..Default::default()
                };

                let mut output2 = if device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_dedicated_allocation
                {
                    Some(ash::vk::MemoryDedicatedRequirementsKHR::default())
                } else {
                    None
                };

                let mut output = ash::vk::MemoryRequirements2 {
                    p_next: output2
                        .as_mut()
                        .map(|o| o as *mut ash::vk::MemoryDedicatedRequirementsKHR)
                        .unwrap_or(ptr::null_mut()) as *mut _,
                    ..Default::default()
                };

                if device.api_version() >= Version::V1_1 {
                    fns.v1_1.get_buffer_memory_requirements2(
                        device.internal_object(),
                        &infos,
                        &mut output,
                    );
                } else {
                    fns.khr_get_memory_requirements2
                        .get_buffer_memory_requirements2_khr(
                            device.internal_object(),
                            &infos,
                            &mut output,
                        );
                }

                debug_assert!(output.memory_requirements.size >= size);
                debug_assert!(output.memory_requirements.memory_type_bits != 0);

                let mut out = MemoryRequirements::from(output.memory_requirements);
                if let Some(output2) = output2 {
                    debug_assert_eq!(output2.requires_dedicated_allocation, 0);
                    out.prefer_dedicated = output2.prefers_dedicated_allocation != 0;
                }
                out
            } else {
                let mut output: MaybeUninit<ash::vk::MemoryRequirements> = MaybeUninit::uninit();
                fns.v1_0.get_buffer_memory_requirements(
                    device.internal_object(),
                    buffer,
                    output.as_mut_ptr(),
                );
                let output = output.assume_init();
                debug_assert!(output.size >= size);
                debug_assert!(output.memory_type_bits != 0);
                MemoryRequirements::from(output)
            };

            // We have to manually enforce some additional requirements for some buffer types.
            let properties = device.physical_device().properties();
            if usage.uniform_texel_buffer || usage.storage_texel_buffer {
                output.alignment = align(
                    output.alignment,
                    properties.min_texel_buffer_offset_alignment,
                );
            }

            if usage.storage_buffer {
                output.alignment = align(
                    output.alignment,
                    properties.min_storage_buffer_offset_alignment,
                );
            }

            if usage.uniform_buffer {
                output.alignment = align(
                    output.alignment,
                    properties.min_uniform_buffer_offset_alignment,
                );
            }

            output
        };

        let obj = UnsafeBuffer {
            buffer,
            device: device.clone(),
            size,
            usage,
        };

        Ok((obj, mem_reqs))
    }

    /// Binds device memory to this buffer.
    pub unsafe fn bind_memory(
        &self,
        memory: &DeviceMemory,
        offset: DeviceSize,
    ) -> Result<(), OomError> {
        let fns = self.device.fns();

        // We check for correctness in debug mode.
        debug_assert!({
            let mut mem_reqs = MaybeUninit::uninit();
            fns.v1_0.get_buffer_memory_requirements(
                self.device.internal_object(),
                self.buffer,
                mem_reqs.as_mut_ptr(),
            );

            let mem_reqs = mem_reqs.assume_init();
            mem_reqs.size <= (memory.size() - offset)
                && (offset % mem_reqs.alignment) == 0
                && mem_reqs.memory_type_bits & (1 << memory.memory_type().id()) != 0
        });

        // Check for alignment correctness.
        {
            let properties = self.device().physical_device().properties();
            if self.usage().uniform_texel_buffer || self.usage().storage_texel_buffer {
                debug_assert!(offset % properties.min_texel_buffer_offset_alignment == 0);
            }
            if self.usage().storage_buffer {
                debug_assert!(offset % properties.min_storage_buffer_offset_alignment == 0);
            }
            if self.usage().uniform_buffer {
                debug_assert!(offset % properties.min_uniform_buffer_offset_alignment == 0);
            }
        }

        check_errors(fns.v1_0.bind_buffer_memory(
            self.device.internal_object(),
            self.buffer,
            memory.internal_object(),
            offset,
        ))?;
        Ok(())
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the buffer the image was created with.
    #[inline]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Returns a key unique to each `UnsafeBuffer`. Can be used for the `conflicts_key` method.
    #[inline]
    pub fn key(&self) -> u64 {
        self.buffer.as_raw()
    }
}

unsafe impl VulkanObject for UnsafeBuffer {
    type Object = ash::vk::Buffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::Buffer {
        self.buffer
    }
}

unsafe impl DeviceOwned for UnsafeBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl fmt::Debug for UnsafeBuffer {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan buffer {:?}>", self.buffer)
    }
}

impl Drop for UnsafeBuffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_buffer(self.device.internal_object(), self.buffer, ptr::null());
        }
    }
}

impl PartialEq for UnsafeBuffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer && self.device == other.device
    }
}

impl Eq for UnsafeBuffer {}

impl Hash for UnsafeBuffer {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
        self.device.hash(state);
    }
}

/// The level of sparse binding that a buffer should be created with.
#[derive(Debug, Copy, Clone)]
pub struct SparseLevel {
    pub sparse_residency: bool,
    pub sparse_aliased: bool,
}

impl SparseLevel {
    #[inline]
    pub fn none() -> SparseLevel {
        SparseLevel {
            sparse_residency: false,
            sparse_aliased: false,
        }
    }
}

impl From<SparseLevel> for ash::vk::BufferCreateFlags {
    #[inline]
    fn from(val: SparseLevel) -> Self {
        let mut result = ash::vk::BufferCreateFlags::SPARSE_BINDING;
        if val.sparse_residency {
            result |= ash::vk::BufferCreateFlags::SPARSE_RESIDENCY;
        }
        if val.sparse_aliased {
            result |= ash::vk::BufferCreateFlags::SPARSE_ALIASED;
        }
        result
    }
}

/// The device address usage flag was not set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceAddressUsageNotEnabledError;
impl error::Error for DeviceAddressUsageNotEnabledError {}
impl fmt::Display for DeviceAddressUsageNotEnabledError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("the device address usage flag was not set on this buffer")
    }
}

/// Error that can happen when creating a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferCreationError {
    /// Allocating memory failed.
    AllocError(DeviceMemoryAllocError),
    /// Sparse binding was requested but the corresponding feature wasn't enabled.
    SparseBindingFeatureNotEnabled,
    /// Sparse residency was requested but the corresponding feature wasn't enabled.
    SparseResidencyBufferFeatureNotEnabled,
    /// Sparse aliasing was requested but the corresponding feature wasn't enabled.
    SparseResidencyAliasedFeatureNotEnabled,
    /// Device address was requested but the corresponding feature wasn't enabled.
    DeviceAddressFeatureNotEnabled,
}

impl error::Error for BufferCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            BufferCreationError::AllocError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for BufferCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                BufferCreationError::AllocError(_) => "allocating memory failed",
                BufferCreationError::SparseBindingFeatureNotEnabled => {
                    "sparse binding was requested but the corresponding feature wasn't enabled"
                }
                BufferCreationError::SparseResidencyBufferFeatureNotEnabled => {
                    "sparse residency was requested but the corresponding feature wasn't enabled"
                }
                BufferCreationError::SparseResidencyAliasedFeatureNotEnabled => {
                    "sparse aliasing was requested but the corresponding feature wasn't enabled"
                }
                BufferCreationError::DeviceAddressFeatureNotEnabled => {
                    "device address was requested but the corresponding feature wasn't enabled"
                }
            }
        )
    }
}

impl From<OomError> for BufferCreationError {
    #[inline]
    fn from(err: OomError) -> BufferCreationError {
        BufferCreationError::AllocError(err.into())
    }
}

impl From<Error> for BufferCreationError {
    #[inline]
    fn from(err: Error) -> BufferCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                BufferCreationError::AllocError(DeviceMemoryAllocError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                BufferCreationError::AllocError(DeviceMemoryAllocError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::Empty;

    use super::BufferCreationError;
    use super::BufferUsage;
    use super::SparseLevel;
    use super::UnsafeBuffer;

    use crate::device::Device;
    use crate::device::DeviceOwned;
    use crate::sync::Sharing;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let (buf, reqs) = unsafe {
            UnsafeBuffer::new(
                device.clone(),
                128,
                BufferUsage::all(),
                Sharing::Exclusive::<Empty<_>>,
                None,
            )
        }
        .unwrap();

        assert!(reqs.size >= 128);
        assert_eq!(buf.size(), 128);
        assert_eq!(&**buf.device() as *const Device, &*device as *const Device);
    }

    #[test]
    fn missing_feature_sparse_binding() {
        let (device, _) = gfx_dev_and_queue!();
        let sparse = Some(SparseLevel::none());
        unsafe {
            match UnsafeBuffer::new(
                device,
                128,
                BufferUsage::all(),
                Sharing::Exclusive::<Empty<_>>,
                sparse,
            ) {
                Err(BufferCreationError::SparseBindingFeatureNotEnabled) => (),
                _ => panic!(),
            }
        };
    }

    #[test]
    fn missing_feature_sparse_residency() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        let sparse = Some(SparseLevel {
            sparse_residency: true,
            sparse_aliased: false,
        });
        unsafe {
            match UnsafeBuffer::new(
                device,
                128,
                BufferUsage::all(),
                Sharing::Exclusive::<Empty<_>>,
                sparse,
            ) {
                Err(BufferCreationError::SparseResidencyBufferFeatureNotEnabled) => (),
                _ => panic!(),
            }
        };
    }

    #[test]
    fn missing_feature_sparse_aliased() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        let sparse = Some(SparseLevel {
            sparse_residency: false,
            sparse_aliased: true,
        });
        unsafe {
            match UnsafeBuffer::new(
                device,
                128,
                BufferUsage::all(),
                Sharing::Exclusive::<Empty<_>>,
                sparse,
            ) {
                Err(BufferCreationError::SparseResidencyAliasedFeatureNotEnabled) => (),
                _ => panic!(),
            }
        };
    }

    #[test]
    fn create_empty_buffer() {
        let (device, _) = gfx_dev_and_queue!();

        unsafe {
            let _ = UnsafeBuffer::new(
                device,
                0,
                BufferUsage::all(),
                Sharing::Exclusive::<Empty<_>>,
                None,
            );
        };
    }
}
