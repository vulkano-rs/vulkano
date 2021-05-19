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

use crate::buffer::BufferUsage;
use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::memory::DeviceMemory;
use crate::memory::DeviceMemoryAllocError;
use crate::memory::MemoryRequirements;
use crate::sync::Sharing;
use crate::vk;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Data storage in a GPU-accessible location.
pub struct UnsafeBuffer {
    buffer: vk::Buffer,
    device: Arc<Device>,
    size: usize,
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
        size: usize,
        mut usage: BufferUsage,
        sharing: Sharing<I>,
        sparse: Option<SparseLevel>,
    ) -> Result<(UnsafeBuffer, MemoryRequirements), BufferCreationError>
    where
        I: Iterator<Item = u32>,
    {
        let vk = device.pointers();

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
            0
        };

        if usage.device_address
            && !(device.enabled_features().buffer_device_address
                || device.enabled_features().ext_buffer_device_address)
        {
            usage.device_address = false;
            if vk::BufferUsageFlags::from(usage) == 0 {
                // return an error iff device_address was the only requested usage and the
                // feature isn't enabled. Otherwise we'll hit that assert below.
                // TODO: This is weird, why not just return an error always if the feature is not enabled?
                // You can't use BufferUsage::all() anymore, but is that a good idea anyway?
                return Err(BufferCreationError::DeviceAddressFeatureNotEnabled);
            }
        }

        let usage_bits = usage.into();
        // Checking for empty BufferUsage.
        assert!(
            usage_bits != 0,
            "Can't create buffer with empty BufferUsage"
        );

        let buffer = {
            let (sh_mode, sh_indices) = match sharing {
                Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
                Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect()),
            };

            let infos = vk::BufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags,
                size: size as u64,
                usage: usage_bits,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_indices.len() as u32,
                pQueueFamilyIndices: sh_indices.as_ptr(),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(vk.CreateBuffer(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let mem_reqs = {
            #[inline]
            fn align(val: usize, al: usize) -> usize {
                al * (1 + (val - 1) / al)
            }

            let mut output = if device.loaded_extensions().khr_get_memory_requirements2 {
                let infos = vk::BufferMemoryRequirementsInfo2KHR {
                    sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR,
                    pNext: ptr::null_mut(),
                    buffer: buffer,
                };

                let mut output2 = if device.loaded_extensions().khr_dedicated_allocation {
                    Some(vk::MemoryDedicatedRequirementsKHR {
                        sType: vk::STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
                        pNext: ptr::null(),
                        prefersDedicatedAllocation: mem::zeroed(),
                        requiresDedicatedAllocation: mem::zeroed(),
                    })
                } else {
                    None
                };

                let mut output = vk::MemoryRequirements2KHR {
                    sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
                    pNext: output2
                        .as_mut()
                        .map(|o| o as *mut vk::MemoryDedicatedRequirementsKHR)
                        .unwrap_or(ptr::null_mut()) as *mut _,
                    memoryRequirements: mem::zeroed(),
                };

                vk.GetBufferMemoryRequirements2KHR(device.internal_object(), &infos, &mut output);
                debug_assert!(output.memoryRequirements.size >= size as u64);
                debug_assert!(output.memoryRequirements.memoryTypeBits != 0);

                let mut out = MemoryRequirements::from(output.memoryRequirements);
                if let Some(output2) = output2 {
                    debug_assert_eq!(output2.requiresDedicatedAllocation, 0);
                    out.prefer_dedicated = output2.prefersDedicatedAllocation != 0;
                }
                out
            } else {
                let mut output: MaybeUninit<vk::MemoryRequirements> = MaybeUninit::uninit();
                vk.GetBufferMemoryRequirements(
                    device.internal_object(),
                    buffer,
                    output.as_mut_ptr(),
                );
                let output = output.assume_init();
                debug_assert!(output.size >= size as u64);
                debug_assert!(output.memoryTypeBits != 0);
                MemoryRequirements::from(output)
            };

            // We have to manually enforce some additional requirements for some buffer types.
            let limits = device.physical_device().limits();
            if usage.uniform_texel_buffer || usage.storage_texel_buffer {
                output.alignment = align(
                    output.alignment,
                    limits.min_texel_buffer_offset_alignment() as usize,
                );
            }

            if usage.storage_buffer {
                output.alignment = align(
                    output.alignment,
                    limits.min_storage_buffer_offset_alignment() as usize,
                );
            }

            if usage.uniform_buffer {
                output.alignment = align(
                    output.alignment,
                    limits.min_uniform_buffer_offset_alignment() as usize,
                );
            }

            output
        };

        let obj = UnsafeBuffer {
            buffer: buffer,
            device: device.clone(),
            size: size as usize,
            usage,
        };

        Ok((obj, mem_reqs))
    }

    /// Binds device memory to this buffer.
    pub unsafe fn bind_memory(&self, memory: &DeviceMemory, offset: usize) -> Result<(), OomError> {
        let vk = self.device.pointers();

        // We check for correctness in debug mode.
        debug_assert!({
            let mut mem_reqs = MaybeUninit::uninit();
            vk.GetBufferMemoryRequirements(
                self.device.internal_object(),
                self.buffer,
                mem_reqs.as_mut_ptr(),
            );

            let mem_reqs = mem_reqs.assume_init();
            mem_reqs.size <= (memory.size() - offset) as u64
                && (offset as u64 % mem_reqs.alignment) == 0
                && mem_reqs.memoryTypeBits & (1 << memory.memory_type().id()) != 0
        });

        // Check for alignment correctness.
        {
            let limits = self.device().physical_device().limits();
            if self.usage().uniform_texel_buffer || self.usage().storage_texel_buffer {
                debug_assert!(offset % limits.min_texel_buffer_offset_alignment() as usize == 0);
            }
            if self.usage().storage_buffer {
                debug_assert!(offset % limits.min_storage_buffer_offset_alignment() as usize == 0);
            }
            if self.usage().uniform_buffer {
                debug_assert!(offset % limits.min_uniform_buffer_offset_alignment() as usize == 0);
            }
        }

        check_errors(vk.BindBufferMemory(
            self.device.internal_object(),
            self.buffer,
            memory.internal_object(),
            offset as vk::DeviceSize,
        ))?;
        Ok(())
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> usize {
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
        self.buffer
    }
}

unsafe impl VulkanObject for UnsafeBuffer {
    type Object = vk::Buffer;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_BUFFER;

    #[inline]
    fn internal_object(&self) -> vk::Buffer {
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
            let vk = self.device.pointers();
            vk.DestroyBuffer(self.device.internal_object(), self.buffer, ptr::null());
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

impl From<SparseLevel> for vk::BufferCreateFlags {
    #[inline]
    fn from(val: SparseLevel) -> Self {
        let mut result = vk::BUFFER_CREATE_SPARSE_BINDING_BIT;
        if val.sparse_residency {
            result |= vk::BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
        }
        if val.sparse_aliased {
            result |= vk::BUFFER_CREATE_SPARSE_ALIASED_BIT;
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
