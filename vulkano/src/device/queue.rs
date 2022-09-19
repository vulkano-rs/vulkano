// Copyright (c) 2022 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{Device, DeviceOwned};
use crate::{
    command_buffer::PrimaryCommandBuffer,
    instance::debug::DebugUtilsLabel,
    macros::vulkan_bitflags,
    sync::{PipelineStage, PipelineStages, Semaphore},
    OomError, RequiresOneOf, SynchronizedVulkanObject, VulkanError,
};
use parking_lot::{Mutex, MutexGuard};
use std::{
    error::Error,
    ffi::CString,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: Mutex<ash::vk::Queue>,
    device: Arc<Device>,
    queue_family_index: u32,
    id: u32, // id within family
}

impl Queue {
    #[inline]
    pub(super) fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Queue,
        queue_family_index: u32,
        id: u32,
    ) -> Arc<Self> {
        Arc::new(Queue {
            handle: Mutex::new(handle),
            device,
            queue_family_index,
            id,
        })
    }

    /// Returns the device that this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the index of the queue family that this queue belongs to.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the index of this queue within its queue family.
    #[inline]
    pub fn id_within_family(&self) -> u32 {
        self.id
    }

    /// Locks the queue, making it possible to perform operations on the queue, such as submissions.
    #[inline]
    pub fn lock(&self) -> QueueGuard {
        QueueGuard {
            queue: self,
            handle: self.handle.lock(),
        }
    }
}

unsafe impl SynchronizedVulkanObject for Queue {
    type Object = ash::vk::Queue;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<Self::Object> {
        self.handle.lock()
    }
}

unsafe impl DeviceOwned for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.queue_family_index == other.queue_family_index
            && self.device == other.device
    }
}

impl Eq for Queue {}

impl Hash for Queue {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.queue_family_index.hash(state);
        self.device.hash(state);
    }
}

pub struct QueueGuard<'a> {
    queue: &'a Queue,
    handle: MutexGuard<'a, ash::vk::Queue>,
}

impl<'a> QueueGuard<'a> {
    /// Waits until all work on this queue has finished.
    ///
    /// Just like [`Device::wait_idle`], you shouldn't have to call this function in a typical
    /// program.
    #[inline]
    pub fn wait_idle(&mut self) -> Result<(), OomError> {
        unsafe {
            let fns = self.queue.device.fns();
            (fns.v1_0.queue_wait_idle)(*self.handle)
                .result()
                .map_err(VulkanError::from)?;
            Ok(())
        }
    }

    /// Opens a queue debug label region.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    #[inline]
    pub fn begin_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        self.validate_begin_debug_utils_label(&label_info)?;

        unsafe {
            self.begin_debug_utils_label_unchecked(label_info);
            Ok(())
        }
    }

    fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`begin_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_debug_utils_label_unchecked(&mut self, label_info: DebugUtilsLabel) {
        let DebugUtilsLabel {
            label_name,
            color,
            _ne: _,
        } = label_info;

        let label_name_vk = CString::new(label_name.as_str()).unwrap();
        let label_info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: label_name_vk.as_ptr(),
            color,
            ..Default::default()
        };

        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_begin_debug_utils_label_ext)(*self.handle, &label_info);
    }

    /// Closes a queue debug label region.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    ///
    /// # Safety
    ///
    /// - There must be an outstanding queue label region begun with `begin_debug_utils_label` in
    ///   the queue.
    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<(), DebugUtilsError> {
        self.validate_end_debug_utils_label()?;

        self.end_debug_utils_label_unchecked();
        Ok(())
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), DebugUtilsError> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`end_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkQueueEndDebugUtilsLabelEXT-None-01911
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) {
        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_end_debug_utils_label_ext)(*self.handle);
    }

    /// Inserts a queue debug label.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    #[inline]
    pub fn insert_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        self.validate_insert_debug_utils_label(&label_info)?;

        unsafe {
            self.insert_debug_utils_label_unchecked(label_info);
            Ok(())
        }
    }

    fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`insert_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn insert_debug_utils_label_unchecked(&mut self, label_info: DebugUtilsLabel) {
        let DebugUtilsLabel {
            label_name,
            color,
            _ne: _,
        } = label_info;

        let label_name_vk = CString::new(label_name.as_str()).unwrap();
        let label_info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: label_name_vk.as_ptr(),
            color,
            ..Default::default()
        };

        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_insert_debug_utils_label_ext)(*self.handle, &label_info);
    }
}

#[derive(Clone, Debug)]
pub struct SubmitInfo {
    pub wait_semaphores: Vec<SemaphoreSubmitInfo>,
    pub command_buffers: Vec<Arc<dyn PrimaryCommandBuffer>>,
    pub signal_semaphores: Vec<SemaphoreSubmitInfo>,
    pub _ne: crate::NonExhaustive,
}

#[derive(Clone, Debug)]
pub struct SemaphoreSubmitInfo {
    pub semaphore: Arc<Semaphore>,
    pub stages: PipelineStages,
    pub _ne: crate::NonExhaustive,
}

/// Properties of a queue family in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QueueFamilyProperties {
    /// Attributes of the queue family.
    pub queue_flags: QueueFlags,

    /// The number of queues available in this family.
    ///
    /// This guaranteed to be at least 1 (or else that family wouldn't exist).
    pub queue_count: u32,

    /// If timestamps are supported, the number of bits supported by timestamp operations.
    /// The returned value will be in the range 36..64.
    ///
    /// If timestamps are not supported, this is `None`.
    pub timestamp_valid_bits: Option<u32>,

    /// The minimum granularity supported for image transfers, in terms of `[width, height, depth]`.
    pub min_image_transfer_granularity: [u32; 3],
}

impl QueueFamilyProperties {
    /// Returns whether the queues of this family support a particular pipeline stage.
    #[inline]
    pub fn supports_stage(&self, stage: PipelineStage) -> bool {
        ash::vk::QueueFlags::from(self.queue_flags).contains(stage.required_queue_flags())
    }
}

impl From<ash::vk::QueueFamilyProperties> for QueueFamilyProperties {
    #[inline]
    fn from(val: ash::vk::QueueFamilyProperties) -> Self {
        Self {
            queue_flags: val.queue_flags.into(),
            queue_count: val.queue_count,
            timestamp_valid_bits: (val.timestamp_valid_bits != 0)
                .then_some(val.timestamp_valid_bits),
            min_image_transfer_granularity: [
                val.min_image_transfer_granularity.width,
                val.min_image_transfer_granularity.height,
                val.min_image_transfer_granularity.depth,
            ],
        }
    }
}

vulkan_bitflags! {
    /// Attributes of a queue or queue family.
    #[non_exhaustive]
    QueueFlags = QueueFlags(u32);

    /// Queues of this family can execute graphics operations.
    graphics = GRAPHICS,

    /// Queues of this family can execute compute operations.
    compute = COMPUTE,

    /// Queues of this family can execute transfer operations.
    transfer = TRANSFER,

    /// Queues of this family can execute sparse memory management operations.
    sparse_binding = SPARSE_BINDING,

    /// Queues of this family can be created using the `protected` flag.
    protected = PROTECTED {
        api_version: V1_1,
    },

    /// Queues of this family can execute video decode operations.
    video_decode = VIDEO_DECODE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Queues of this family can execute video encode operations.
    video_encode = VIDEO_ENCODE_KHR {
        device_extensions: [khr_video_encode_queue],
    },
}

/// Error that can happen when submitting a debug utils command to a queue.
#[derive(Clone, Debug)]
pub enum DebugUtilsError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for DebugUtilsError {}

impl Display for DebugUtilsError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}
