// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    device::{Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum},
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

/// A fence is used to know when a command buffer submission has finished its execution.
///
/// When a command buffer accesses a resource, you have to ensure that the CPU doesn't access
/// the same resource simultaneously (except for concurrent reads). Therefore in order to know
/// when the CPU can access a resource again, a fence has to be used.
#[derive(Debug)]
pub struct Fence {
    handle: ash::vk::Fence,
    device: Arc<Device>,

    _export_handle_types: ExternalFenceHandleTypes,

    // If true, we know that the `Fence` is signaled. If false, we don't know.
    // This variable exists so that we don't need to call `vkGetFenceStatus` or `vkWaitForFences`
    // multiple times.
    is_signaled: AtomicBool,

    // Indicates whether this fence was taken from the fence pool.
    // If true, will be put back into fence pool on drop.
    must_put_in_pool: bool,
}

impl Fence {
    /// Creates a new `Fence`.
    #[inline]
    pub fn new(device: Arc<Device>, create_info: FenceCreateInfo) -> Result<Fence, FenceError> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(device: &Device, create_info: &FenceCreateInfo) -> Result<(), FenceError> {
        let FenceCreateInfo {
            signaled: _,
            export_handle_types,
            _ne: _,
        } = create_info;

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_fence)
            {
                return Err(FenceError::RequirementNotMet {
                    required_for: "`create_info.export_handle_types` is not empty",
                    requires_one_of: RequiresOneOf {
                        api_version: Some(Version::V1_1),
                        device_extensions: &["khr_external_fence"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkExportFenceCreateInfo-handleTypes-01446
            export_handle_types.validate_device(device)?;

            // VUID-VkExportFenceCreateInfo-handleTypes-01446
            // TODO: `vkGetPhysicalDeviceExternalFenceProperties` can only be called with one
            // handle type, so which one do we give it?
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: FenceCreateInfo,
    ) -> Result<Fence, VulkanError> {
        let FenceCreateInfo {
            signaled,
            export_handle_types,
            _ne: _,
        } = create_info;

        let mut flags = ash::vk::FenceCreateFlags::empty();

        if signaled {
            flags |= ash::vk::FenceCreateFlags::SIGNALED;
        }

        let mut create_info_vk = ash::vk::FenceCreateInfo {
            flags,
            ..Default::default()
        };
        let mut export_fence_create_info_vk = None;

        if !export_handle_types.is_empty() {
            let _ = export_fence_create_info_vk.insert(ash::vk::ExportFenceCreateInfo {
                handle_types: export_handle_types.into(),
                ..Default::default()
            });
        }

        if let Some(info) = export_fence_create_info_vk.as_mut() {
            info.p_next = create_info_vk.p_next;
            create_info_vk.p_next = info as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_fence)(
                device.internal_object(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Fence {
            handle,
            device,
            _export_handle_types: export_handle_types,
            is_signaled: AtomicBool::new(signaled),
            must_put_in_pool: false,
        })
    }

    /// Takes a fence from the vulkano-provided fence pool.
    /// If the pool is empty, a new fence will be created.
    /// Upon `drop`, the fence is put back into the pool.
    ///
    /// For most applications, using the fence pool should be preferred,
    /// in order to avoid creating new fences every frame.
    pub fn from_pool(device: Arc<Device>) -> Result<Fence, FenceError> {
        let handle = device.fence_pool().lock().pop();
        let fence = match handle {
            Some(handle) => {
                unsafe {
                    // Make sure the fence isn't signaled
                    let fns = device.fns();
                    (fns.v1_0.reset_fences)(device.internal_object(), 1, &handle)
                        .result()
                        .map_err(VulkanError::from)?;
                }

                Fence {
                    handle,
                    device,
                    _export_handle_types: ExternalFenceHandleTypes::empty(),
                    is_signaled: AtomicBool::new(false),
                    must_put_in_pool: true,
                }
            }
            None => {
                // Pool is empty, alloc new fence
                let mut fence = Fence::new(device, FenceCreateInfo::default())?;
                fence.must_put_in_pool = true;
                fence
            }
        };

        Ok(fence)
    }

    /// Creates a new `Fence` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Fence,
        create_info: FenceCreateInfo,
    ) -> Fence {
        let FenceCreateInfo {
            signaled,
            export_handle_types,
            _ne: _,
        } = create_info;

        Fence {
            handle,
            device,
            _export_handle_types: export_handle_types,
            is_signaled: AtomicBool::new(signaled),
            must_put_in_pool: false,
        }
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn is_signaled(&self) -> Result<bool, OomError> {
        unsafe {
            if self.is_signaled.load(Ordering::Relaxed) {
                return Ok(true);
            }

            let fns = self.device.fns();
            let result = (fns.v1_0.get_fence_status)(self.device.internal_object(), self.handle);
            match result {
                ash::vk::Result::SUCCESS => {
                    self.is_signaled.store(true, Ordering::Relaxed);
                    Ok(true)
                }
                ash::vk::Result::NOT_READY => Ok(false),
                err => Err(VulkanError::from(err).into()),
            }
        }
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed.
    ///
    /// Returns `Ok` if the fence is now signaled. Returns `Err` if the timeout was reached instead.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), FenceError> {
        unsafe {
            if self.is_signaled.load(Ordering::Relaxed) {
                return Ok(());
            }

            let timeout_ns = if let Some(timeout) = timeout {
                timeout
                    .as_secs()
                    .saturating_mul(1_000_000_000)
                    .saturating_add(timeout.subsec_nanos() as u64)
            } else {
                u64::MAX
            };

            let fns = self.device.fns();
            let result = (fns.v1_0.wait_for_fences)(
                self.device.internal_object(),
                1,
                &self.handle,
                ash::vk::TRUE,
                timeout_ns,
            );

            match result {
                ash::vk::Result::SUCCESS => {
                    self.is_signaled.store(true, Ordering::Relaxed);
                    Ok(())
                }
                ash::vk::Result::TIMEOUT => Err(FenceError::Timeout),
                err => Err(VulkanError::from(err).into()),
            }
        }
    }

    /// Waits for multiple fences at once.
    ///
    /// # Panic
    ///
    /// Panics if not all fences belong to the same device.
    pub fn multi_wait<'a, I>(iter: I, timeout: Option<Duration>) -> Result<(), FenceError>
    where
        I: IntoIterator<Item = &'a Fence>,
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[ash::vk::Fence; 8]> = iter
            .into_iter()
            .filter_map(|fence| {
                match &mut device {
                    dev @ &mut None => *dev = Some(&*fence.device),
                    &mut Some(dev) if std::ptr::eq(dev, &*fence.device) => {}
                    _ => panic!(
                        "Tried to wait for multiple fences that didn't belong to the \
                                 same device"
                    ),
                };

                if fence.is_signaled.load(Ordering::Relaxed) {
                    None
                } else {
                    Some(fence.handle)
                }
            })
            .collect();

        let timeout_ns = if let Some(timeout) = timeout {
            timeout
                .as_secs()
                .saturating_mul(1_000_000_000)
                .saturating_add(timeout.subsec_nanos() as u64)
        } else {
            u64::MAX
        };

        let result = if let Some(device) = device {
            unsafe {
                let fns = device.fns();
                (fns.v1_0.wait_for_fences)(
                    device.internal_object(),
                    fences.len() as u32,
                    fences.as_ptr(),
                    ash::vk::TRUE,
                    timeout_ns,
                )
            }
        } else {
            return Ok(());
        };

        match result {
            ash::vk::Result::SUCCESS => Ok(()),
            ash::vk::Result::TIMEOUT => Err(FenceError::Timeout),
            err => Err(VulkanError::from(err).into()),
        }
    }

    /// Resets the fence.
    // This function takes a `&mut self` because the Vulkan API requires that the fence be
    // externally synchronized.
    #[inline]
    pub fn reset(&mut self) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.reset_fences)(self.device.internal_object(), 1, &self.handle)
                .result()
                .map_err(VulkanError::from)?;
            self.is_signaled.store(false, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Resets multiple fences at once.
    ///
    /// # Panic
    ///
    /// - Panics if not all fences belong to the same device.
    ///
    pub fn multi_reset<'a, I>(iter: I) -> Result<(), OomError>
    where
        I: IntoIterator<Item = &'a mut Fence>,
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[ash::vk::Fence; 8]> = iter
            .into_iter()
            .map(|fence| {
                match &mut device {
                    dev @ &mut None => *dev = Some(&*fence.device),
                    &mut Some(dev) if std::ptr::eq(dev, &*fence.device) => {}
                    _ => panic!(
                        "Tried to reset multiple fences that didn't belong to the same \
                                 device"
                    ),
                };

                fence.is_signaled.store(false, Ordering::Relaxed);
                fence.handle
            })
            .collect();

        if let Some(device) = device {
            unsafe {
                let fns = device.fns();
                (fns.v1_0.reset_fences)(
                    device.internal_object(),
                    fences.len() as u32,
                    fences.as_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
            }
        }
        Ok(())
    }
}

impl Drop for Fence {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_fence = self.handle;
                self.device.fence_pool().lock().push(raw_fence);
            } else {
                let fns = self.device.fns();
                (fns.v1_0.destroy_fence)(self.device.internal_object(), self.handle, ptr::null());
            }
        }
    }
}

unsafe impl VulkanObject for Fence {
    type Object = ash::vk::Fence;

    #[inline]
    fn internal_object(&self) -> ash::vk::Fence {
        self.handle
    }
}

unsafe impl DeviceOwned for Fence {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Fence {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for Fence {}

impl Hash for Fence {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Parameters to create a new `Fence`.
#[derive(Clone, Debug)]
pub struct FenceCreateInfo {
    /// Whether the fence should be created in the signaled state.
    ///
    /// The default value is `false`.
    pub signaled: bool,

    pub export_handle_types: ExternalFenceHandleTypes,

    pub _ne: crate::NonExhaustive,
}

impl Default for FenceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            signaled: false,
            export_handle_types: ExternalFenceHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

vulkan_enum! {
    /// The handle type used for Vulkan external fence APIs.
    #[non_exhaustive]
    ExternalFenceHandleType = ExternalFenceHandleTypeFlags(u32);

    // TODO: document
    OpaqueFd = OPAQUE_FD,

    // TODO: document
    OpaqueWin32 = OPAQUE_WIN32,

    // TODO: document
    OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    // TODO: document
    SyncFd = SYNC_FD,
}

vulkan_bitflags! {
    /// A mask of multiple external fence handle types.
    #[non_exhaustive]
    ExternalFenceHandleTypes = ExternalFenceHandleTypeFlags(u32);

    // TODO: document
    opaque_fd = OPAQUE_FD,

    // TODO: document
    opaque_win32 = OPAQUE_WIN32,

    // TODO: document
    opaque_win32_kmt = OPAQUE_WIN32_KMT,

    // TODO: document
    sync_fd = SYNC_FD,
}

impl From<ExternalFenceHandleType> for ExternalFenceHandleTypes {
    #[inline]
    fn from(val: ExternalFenceHandleType) -> Self {
        let mut result = Self::empty();

        match val {
            ExternalFenceHandleType::OpaqueFd => result.opaque_fd = true,
            ExternalFenceHandleType::OpaqueWin32 => result.opaque_win32 = true,
            ExternalFenceHandleType::OpaqueWin32Kmt => result.opaque_win32_kmt = true,
            ExternalFenceHandleType::SyncFd => result.sync_fd = true,
        }

        result
    }
}

vulkan_bitflags! {
    /// Additional parameters for a fence payload import.
    #[non_exhaustive]
    FenceImportFlags = FenceImportFlags(u32);

    /// The fence payload will be imported only temporarily, regardless of the permanence of the
    /// imported handle type.
    temporary = TEMPORARY,
}

/// The fence configuration to query in
/// [`PhysicalDevice::external_fence_properties`](crate::device::physical::PhysicalDevice::external_fence_properties).
#[derive(Clone, Debug)]
pub struct ExternalFenceInfo {
    /// The external handle type that will be used with the fence.
    pub handle_type: ExternalFenceHandleType,

    pub _ne: crate::NonExhaustive,
}

impl ExternalFenceInfo {
    /// Returns an `ExternalFenceInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            handle_type,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The properties for exporting or importing external handles, when a fence is created
/// with a specific configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalFenceProperties {
    /// Whether a handle can be exported to an external source with the queried
    /// external handle type.
    pub exportable: bool,

    /// Whether a handle can be imported from an external source with the queried
    /// external handle type.
    pub importable: bool,

    /// Which external handle types can be re-exported after the queried external handle type has
    /// been imported.
    pub export_from_imported_handle_types: ExternalFenceHandleTypes,

    /// Which external handle types can be enabled along with the queried external handle type
    /// when creating the fence.
    pub compatible_handle_types: ExternalFenceHandleTypes,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FenceError {
    /// Not enough memory available.
    OomError(OomError),

    /// The device has been lost.
    DeviceLost,

    /// The specified timeout wasn't long enough.
    Timeout,

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for FenceError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl Display for FenceError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::DeviceLost => write!(f, "the device was lost"),
            Self::Timeout => write!(f, "the timeout has been reached"),

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

impl From<VulkanError> for FenceError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            VulkanError::DeviceLost => Self::DeviceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for FenceError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for FenceError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sync::{fence::FenceCreateInfo, Fence},
        VulkanObject,
    };
    use std::time::Duration;

    #[test]
    fn fence_create() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(device, Default::default()).unwrap();
        assert!(!fence.is_signaled().unwrap());
    }

    #[test]
    fn fence_create_signaled() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device,
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fence.is_signaled().unwrap());
    }

    #[test]
    fn fence_signaled_wait() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device,
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        fence.wait(Some(Duration::new(0, 10))).unwrap();
    }

    #[test]
    fn fence_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let mut fence = Fence::new(
            device,
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        fence.reset().unwrap();
        assert!(!fence.is_signaled().unwrap());
    }

    #[test]
    fn multiwait_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!(
            "Tried to wait for multiple fences that didn't belong \
                              to the same device",
            {
                let fence1 = Fence::new(
                    device1.clone(),
                    FenceCreateInfo {
                        signaled: true,
                        ..Default::default()
                    },
                )
                .unwrap();
                let fence2 = Fence::new(
                    device2.clone(),
                    FenceCreateInfo {
                        signaled: true,
                        ..Default::default()
                    },
                )
                .unwrap();

                let _ = Fence::multi_wait(
                    [&fence1, &fence2].iter().cloned(),
                    Some(Duration::new(0, 10)),
                );
            }
        );
    }

    #[test]
    fn multireset_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!(
            "Tried to reset multiple fences that didn't belong \
                              to the same device",
            {
                let mut fence1 = Fence::new(
                    device1.clone(),
                    FenceCreateInfo {
                        signaled: true,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut fence2 = Fence::new(
                    device2.clone(),
                    FenceCreateInfo {
                        signaled: true,
                        ..Default::default()
                    },
                )
                .unwrap();

                let _ = Fence::multi_reset([&mut fence1, &mut fence2]);
            }
        );
    }

    #[test]
    fn fence_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.fence_pool().lock().len(), 0);
        let fence1_internal_obj = {
            let fence = Fence::from_pool(device.clone()).unwrap();
            assert_eq!(device.fence_pool().lock().len(), 0);
            fence.internal_object()
        };

        assert_eq!(device.fence_pool().lock().len(), 1);
        let fence2 = Fence::from_pool(device.clone()).unwrap();
        assert_eq!(device.fence_pool().lock().len(), 0);
        assert_eq!(fence2.internal_object(), fence1_internal_obj);
    }
}
