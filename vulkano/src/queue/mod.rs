use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};
use crate::device::{DebugUtilsError, Device, DeviceOwned};
use crate::device::physical::QueueFamily;
use crate::instance::debug::DebugUtilsLabel;
use crate::{check_errors, OomError, SynchronizedVulkanObject};

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: Mutex<ash::vk::Queue>,
    device: Arc<Device>,
    family: u32,
    id: u32, // id within family
}

impl Queue {

    pub(crate) fn new(handle: Mutex<ash::vk::Queue>, device: Arc<Device>, family: u32, id: u32) -> Queue {
        Self {
            handle,
            device,
            family,
            id
        }
    }

    /// Returns the device this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the family this queue belongs to.
    #[inline]
    pub fn family(&self) -> QueueFamily {
        self.device
            .physical_device()
            .queue_family_by_id(self.family)
            .unwrap()
    }

    /// Returns the index of this queue within its family.
    #[inline]
    pub fn id_within_family(&self) -> u32 {
        self.id
    }

    /// Waits until all work on this queue has finished.
    ///
    /// Just like `Device::wait()`, you shouldn't have to call this function in a typical program.
    #[inline]
    pub fn wait(&self) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();
            let handle = self.handle.lock().unwrap();
            check_errors((fns.v1_0.queue_wait_idle)(*handle))?;
            Ok(())
        }
    }

    /// Opens a queue debug label region.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    #[inline]
    pub fn begin_debug_utils_label(
        &self,
        mut label_info: DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        self.validate_begin_debug_utils_label(&mut label_info)?;

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

        unsafe {
            let fns = self.device.instance().fns();
            let handle = self.handle.lock().unwrap();
            (fns.ext_debug_utils.queue_begin_debug_utils_label_ext)(*handle, &label_info);
        }

        Ok(())
    }

    fn validate_begin_debug_utils_label(
        &self,
        label_info: &mut DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to submit a debug utils command",
            });
        }

        Ok(())
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
    pub unsafe fn end_debug_utils_label(&self) -> Result<(), DebugUtilsError> {
        self.validate_end_debug_utils_label()?;

        {
            let fns = self.device.instance().fns();
            let handle = self.handle.lock().unwrap();
            (fns.ext_debug_utils.queue_end_debug_utils_label_ext)(*handle);
        }

        Ok(())
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to submit a debug utils command",
            });
        }

        // VUID-vkQueueEndDebugUtilsLabelEXT-None-01911
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    /// Inserts a queue debug label.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    #[inline]
    pub fn insert_debug_utils_label(
        &mut self,
        mut label_info: DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        self.validate_insert_debug_utils_label(&mut label_info)?;

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

        unsafe {
            let fns = self.device.instance().fns();
            let handle = self.handle.lock().unwrap();
            (fns.ext_debug_utils.queue_insert_debug_utils_label_ext)(*handle, &label_info);
        }

        Ok(())
    }

    fn validate_insert_debug_utils_label(
        &self,
        label_info: &mut DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to submit a debug utils command",
            });
        }

        Ok(())
    }
}

unsafe impl SynchronizedVulkanObject for Queue {
    type Object = ash::vk::Queue;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<Self::Object> {
        self.handle.lock().unwrap()
    }
}

unsafe impl DeviceOwned for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.family == other.family && self.device == other.device
    }
}

impl Eq for Queue {}

impl Hash for Queue {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.family.hash(state);
        self.device.hash(state);
    }
}
