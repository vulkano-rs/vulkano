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
#[derive(Debug)]
pub struct UnsafeBuffer {
    handle: ash::vk::Buffer,
    device: Arc<Device>,
    size: DeviceSize,
    usage: BufferUsage,
}

impl UnsafeBuffer {
    /// Creates a new `UnsafeBuffer`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.sharing` is [`Concurrent`](Sharing::Concurrent) with less than 2
    ///   items.
    /// - Panics if `create_info.size` is zero.
    /// - Panics if `create_info.usage` is empty.
    pub fn new(
        device: Arc<Device>,
        create_info: UnsafeBufferCreateInfo,
    ) -> Result<UnsafeBuffer, BufferCreationError> {
        let UnsafeBufferCreateInfo {
            mut sharing,
            size,
            sparse,
            usage,
        } = create_info;

        // VUID-VkBufferCreateInfo-size-00912
        assert!(size != 0);

        // VUID-VkBufferCreateInfo-usage-requiredbitmask
        assert!(usage != BufferUsage::none());

        let mut flags = ash::vk::BufferCreateFlags::empty();

        // Check sparse features
        if let Some(sparse_level) = sparse {
            // VUID-VkBufferCreateInfo-flags-00915
            if !device.enabled_features().sparse_binding {
                return Err(BufferCreationError::FeatureNotEnabled {
                    feature: "sparse_binding",
                    reason: "sparse was `Some`",
                });
            }

            // VUID-VkBufferCreateInfo-flags-00916
            if sparse_level.sparse_residency && !device.enabled_features().sparse_residency_buffer {
                return Err(BufferCreationError::FeatureNotEnabled {
                    feature: "sparse_residency_buffer",
                    reason: "sparse was `Some` and `sparse_residency` was set",
                });
            }

            // VUID-VkBufferCreateInfo-flags-00917
            if sparse_level.sparse_aliased && !device.enabled_features().sparse_residency_aliased {
                return Err(BufferCreationError::FeatureNotEnabled {
                    feature: "sparse_residency_aliased",
                    reason: "sparse was `Some` and `sparse_aliased` was set",
                });
            }

            // VUID-VkBufferCreateInfo-flags-00918
            flags |= sparse_level.into();
        }

        // Check sharing mode and queue families
        let (sharing_mode, queue_family_indices) = match &mut sharing {
            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, &[] as _),
            Sharing::Concurrent(ids) => {
                // VUID-VkBufferCreateInfo-sharingMode-00914
                ids.sort_unstable();
                ids.dedup();
                assert!(ids.len() >= 2);

                for &id in ids.iter() {
                    // VUID-VkBufferCreateInfo-sharingMode-01419
                    if device.physical_device().queue_family_by_id(id).is_none() {
                        return Err(BufferCreationError::SharingInvalidQueueFamilyId { id });
                    }
                }

                (ash::vk::SharingMode::CONCURRENT, ids.as_slice())
            }
        };

        if let Some(max_buffer_size) = device.physical_device().properties().max_buffer_size {
            // VUID-VkBufferCreateInfo-size-06409
            if size > max_buffer_size {
                return Err(BufferCreationError::MaxBufferSizeExceeded {
                    size,
                    max: max_buffer_size,
                });
            }
        }

        // Everything now ok. Creating the buffer.
        let create_info = ash::vk::BufferCreateInfo::builder()
            .flags(flags)
            .size(size)
            .usage(usage.into())
            .sharing_mode(sharing_mode)
            .queue_family_indices(queue_family_indices);

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_buffer(
                device.internal_object(),
                &create_info.build(),
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let buffer = UnsafeBuffer {
            handle,
            device,
            size,
            usage,
        };

        Ok(buffer)
    }

    /// Returns the memory requirements for this buffer.
    pub fn memory_requirements(&self) -> MemoryRequirements {
        #[inline]
        fn align(val: DeviceSize, al: DeviceSize) -> DeviceSize {
            al * (1 + (val - 1) / al)
        }

        let buffer_memory_requirements_info2 = ash::vk::BufferMemoryRequirementsInfo2 {
            buffer: self.handle,
            ..Default::default()
        };
        let mut memory_requirements2 = ash::vk::MemoryRequirements2::default();

        let mut memory_dedicated_requirements = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_dedicated_allocation
        {
            Some(ash::vk::MemoryDedicatedRequirementsKHR::default())
        } else {
            None
        };

        if let Some(next) = memory_dedicated_requirements.as_mut() {
            next.p_next = memory_requirements2.p_next;
            memory_requirements2.p_next = next as *mut _ as *mut _;
        }

        unsafe {
            let fns = self.device.fns();

            if self.device.api_version() >= Version::V1_1
                || self
                    .device
                    .enabled_extensions()
                    .khr_get_memory_requirements2
            {
                if self.device.api_version() >= Version::V1_1 {
                    fns.v1_1.get_buffer_memory_requirements2(
                        self.device.internal_object(),
                        &buffer_memory_requirements_info2,
                        &mut memory_requirements2,
                    );
                } else {
                    fns.khr_get_memory_requirements2
                        .get_buffer_memory_requirements2_khr(
                            self.device.internal_object(),
                            &buffer_memory_requirements_info2,
                            &mut memory_requirements2,
                        );
                }
            } else {
                fns.v1_0.get_buffer_memory_requirements(
                    self.device.internal_object(),
                    self.handle,
                    &mut memory_requirements2.memory_requirements,
                );
            }
        }

        debug_assert!(memory_requirements2.memory_requirements.size >= self.size);
        debug_assert!(memory_requirements2.memory_requirements.memory_type_bits != 0);

        let mut memory_requirements = MemoryRequirements {
            prefer_dedicated: memory_dedicated_requirements
                .map_or(false, |dreqs| dreqs.prefers_dedicated_allocation != 0),
            ..MemoryRequirements::from(memory_requirements2.memory_requirements)
        };

        // We have to manually enforce some additional requirements for some buffer types.
        let properties = self.device.physical_device().properties();
        if self.usage.uniform_texel_buffer || self.usage.storage_texel_buffer {
            memory_requirements.alignment = align(
                memory_requirements.alignment,
                properties.min_texel_buffer_offset_alignment,
            );
        }

        if self.usage.storage_buffer {
            memory_requirements.alignment = align(
                memory_requirements.alignment,
                properties.min_storage_buffer_offset_alignment,
            );
        }

        if self.usage.uniform_buffer {
            memory_requirements.alignment = align(
                memory_requirements.alignment,
                properties.min_uniform_buffer_offset_alignment,
            );
        }

        memory_requirements
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
                self.handle,
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
            self.handle,
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
        self.handle.as_raw()
    }
}

unsafe impl VulkanObject for UnsafeBuffer {
    type Object = ash::vk::Buffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::Buffer {
        self.handle
    }
}

unsafe impl DeviceOwned for UnsafeBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for UnsafeBuffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_buffer(self.device.internal_object(), self.handle, ptr::null());
        }
    }
}

impl PartialEq for UnsafeBuffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device == other.device
    }
}

impl Eq for UnsafeBuffer {}

impl Hash for UnsafeBuffer {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device.hash(state);
    }
}

/// Parameters to construct a new `UnsafeBuffer`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct UnsafeBufferCreateInfo {
    /// Whether the buffer can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<SmallVec<[u32; 4]>>,

    /// The size in bytes of the buffer.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// Create a buffer with sparsely bound memory.
    ///
    /// The default value is `None`.
    pub sparse: Option<SparseLevel>,

    /// How the buffer is going to be used.
    ///
    /// The default value is [`BufferUsage::none()`], which must be overridden.
    pub usage: BufferUsage,
}

impl Default for UnsafeBufferCreateInfo {
    fn default() -> Self {
        Self {
            sharing: Sharing::Exclusive,
            size: 0,
            sparse: None,
            usage: BufferUsage::none(),
        }
    }
}

/// Error that can happen when creating a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferCreationError {
    /// Allocating memory failed.
    AllocError(DeviceMemoryAllocError),

    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// The specified size exceeded the value of the `max_buffer_size` limit.
    MaxBufferSizeExceeded { size: DeviceSize, max: DeviceSize },

    /// The sharing mode was set to `Concurrent`, but one of the specified queue family ids was not
    /// valid.
    SharingInvalidQueueFamilyId { id: u32 },
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
        match *self {
            Self::AllocError(_) => write!(fmt, "allocating memory failed"),
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::MaxBufferSizeExceeded { .. } => write!(
                fmt,
                "the specified size exceeded the value of the `max_buffer_size` limit"
            ),
            Self::SharingInvalidQueueFamilyId { id } => {
                write!(fmt, "the sharing mode was set to `Concurrent`, but one of the specified queue family ids was not valid")
            }
        }
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

/// The level of sparse binding that a buffer should be created with.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
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

#[cfg(test)]
mod tests {
    use super::BufferCreationError;
    use super::BufferUsage;
    use super::SparseLevel;
    use super::UnsafeBuffer;
    use super::UnsafeBufferCreateInfo;
    use crate::device::Device;
    use crate::device::DeviceOwned;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let buf = UnsafeBuffer::new(
            device.clone(),
            UnsafeBufferCreateInfo {
                size: 128,
                usage: BufferUsage::all(),
                ..Default::default()
            },
        )
        .unwrap();
        let reqs = buf.memory_requirements();

        assert!(reqs.size >= 128);
        assert_eq!(buf.size(), 128);
        assert_eq!(&**buf.device() as *const Device, &*device as *const Device);
    }

    #[test]
    fn missing_feature_sparse_binding() {
        let (device, _) = gfx_dev_and_queue!();
        match UnsafeBuffer::new(
            device,
            UnsafeBufferCreateInfo {
                size: 128,
                sparse: Some(SparseLevel::none()),
                usage: BufferUsage::all(),
                ..Default::default()
            },
        ) {
            Err(BufferCreationError::FeatureNotEnabled {
                feature: "sparse_binding",
                ..
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn missing_feature_sparse_residency() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        match UnsafeBuffer::new(
            device,
            UnsafeBufferCreateInfo {
                size: 128,
                sparse: Some(SparseLevel {
                    sparse_residency: true,
                    sparse_aliased: false,
                }),
                usage: BufferUsage::all(),
                ..Default::default()
            },
        ) {
            Err(BufferCreationError::FeatureNotEnabled {
                feature: "sparse_residency_buffer",
                ..
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn missing_feature_sparse_aliased() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        match UnsafeBuffer::new(
            device,
            UnsafeBufferCreateInfo {
                size: 128,
                sparse: Some(SparseLevel {
                    sparse_residency: false,
                    sparse_aliased: true,
                }),
                usage: BufferUsage::all(),
                ..Default::default()
            },
        ) {
            Err(BufferCreationError::FeatureNotEnabled {
                feature: "sparse_residency_aliased",
                ..
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn create_empty_buffer() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            UnsafeBuffer::new(
                device,
                UnsafeBufferCreateInfo {
                    size: 0,
                    usage: BufferUsage::all(),
                    ..Default::default()
                },
            )
        });
    }
}
