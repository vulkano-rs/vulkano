//! Store user-defined data associated with Vulkan objects.
//!
//! A private data slot allows you to associate an arbitrary `u64` value with any device-owned
//! Vulkan object. Each private data slot can store one value per object, but you can use the
//! value to look up a larger amount of data in a collection such as a `HashMap`.
//!
//! The intended usage is to create one private data slot for every subsystem in your program
//! that needs to assign data to objects independently of the others. That way, different parts
//! of a program manage their own private data and don't interfere with each other's data.
//!
//! When creating a device, it is possible to reserve private data slots ahead of time using
//! [`DeviceCreateInfo::private_data_slot_request_count`]. This is not necessary, but it can
//! speed up the use of data slots later.
//!
//! [`DeviceCreateInfo::private_data_slot_request_count`]: super::DeviceCreateInfo::private_data_slot_request_count

use super::{Device, DeviceOwned};
use crate::{
    instance::InstanceOwnedDebugWrapper, Requires, RequiresAllOf, RequiresOneOf, Validated,
    ValidationError, Version, VulkanError, VulkanObject,
};
use ash::vk::{self, Handle};
use std::{mem::MaybeUninit, ptr, sync::Arc};

/// An object that stores one `u64` value per Vulkan object.
#[derive(Debug)]
pub struct PrivateDataSlot {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    handle: vk::PrivateDataSlot,
}

impl PrivateDataSlot {
    /// Creates a new `PrivateDataSlot`.
    ///
    /// The `private_data` feature must be enabled on the device.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: PrivateDataSlotCreateInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &PrivateDataSlotCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_features().private_data {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "private_data",
                )])]),
                vuids: &["VUID-vkCreatePrivateDataSlot-privateData-04564"],
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: PrivateDataSlotCreateInfo,
    ) -> Result<Self, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();

            if device.api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.create_private_data_slot)(
                        device.handle(),
                        &create_info_vk,
                        ptr::null(),
                        output.as_mut_ptr(),
                    )
                }
            } else {
                unsafe {
                    (fns.ext_private_data.create_private_data_slot_ext)(
                        device.handle(),
                        &create_info_vk,
                        ptr::null(),
                        output.as_mut_ptr(),
                    )
                }
            }
            .result()
            .map_err(VulkanError::from)?;

            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    /// Creates a new `PrivateDataSlot` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: vk::PrivateDataSlot,
        _create_info: PrivateDataSlotCreateInfo,
    ) -> Self {
        Self {
            device: InstanceOwnedDebugWrapper(device),
            handle,
        }
    }

    /// Sets the private data that is associated with `object` to `data`.
    ///
    /// If `self` already has data for `object`, that data is replaced with the new value.
    #[inline]
    pub fn set_private_data<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        data: u64,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_set_private_data(object)?;

        Ok(unsafe { self.set_private_data_unchecked(object, data) }?)
    }

    fn validate_set_private_data<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
    ) -> Result<(), Box<ValidationError>> {
        assert_eq!(self.device(), object.device());

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_private_data_unchecked<T: VulkanObject + DeviceOwned>(
        &self,
        object: &T,
        data: u64,
    ) -> Result<(), VulkanError> {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            unsafe {
                (fns.v1_3.set_private_data)(
                    self.device.handle(),
                    T::Handle::TYPE,
                    object.handle().as_raw(),
                    self.handle,
                    data,
                )
            }
        } else {
            unsafe {
                (fns.ext_private_data.set_private_data_ext)(
                    self.device.handle(),
                    T::Handle::TYPE,
                    object.handle().as_raw(),
                    self.handle,
                    data,
                )
            }
        }
        .result()
        .map_err(VulkanError::from)
    }

    /// Returns the private data in `self` that is associated with `object`.
    ///
    /// If no private data was previously set, 0 is returned.
    pub fn get_private_data<T: VulkanObject + DeviceOwned>(&self, object: &T) -> u64 {
        let fns = self.device.fns();
        let mut output = MaybeUninit::uninit();

        if self.device.api_version() >= Version::V1_3 {
            unsafe {
                (fns.v1_3.get_private_data)(
                    self.device.handle(),
                    T::Handle::TYPE,
                    object.handle().as_raw(),
                    self.handle,
                    output.as_mut_ptr(),
                )
            }
        } else {
            unsafe {
                (fns.ext_private_data.get_private_data_ext)(
                    self.device.handle(),
                    T::Handle::TYPE,
                    object.handle().as_raw(),
                    self.handle,
                    output.as_mut_ptr(),
                )
            }
        }

        unsafe { output.assume_init() }
    }
}

impl Drop for PrivateDataSlot {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            unsafe {
                (fns.v1_3.destroy_private_data_slot)(self.device.handle(), self.handle, ptr::null())
            };
        } else {
            unsafe {
                (fns.ext_private_data.destroy_private_data_slot_ext)(
                    self.device.handle(),
                    self.handle,
                    ptr::null(),
                )
            };
        }
    }
}

unsafe impl VulkanObject for PrivateDataSlot {
    type Handle = vk::PrivateDataSlot;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for PrivateDataSlot {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// Parameters to create a new `PrivateDataSlot`.
#[derive(Clone, Debug)]
pub struct PrivateDataSlotCreateInfo {
    pub _ne: crate::NonExhaustive,
}

impl Default for PrivateDataSlotCreateInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PrivateDataSlotCreateInfo {
    /// Returns a default `PrivateDataSlotCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, _device: &Device) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PrivateDataSlotCreateInfo<'static> {
        let &Self { _ne: _ } = self;

        vk::PrivateDataSlotCreateInfo::default().flags(vk::PrivateDataSlotCreateFlags::empty())
    }
}
