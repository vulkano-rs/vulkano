// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    device::{Device, DeviceOwned, Queue},
    macros::{vulkan_bitflags, vulkan_enum},
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::{Arc, Weak},
    time::Duration,
};

/// A two-state synchronization primitive that is signalled by the device and waited on by the host.
///
/// # Queue-to-host synchronization
///
/// The primary use of a fence is to know when execution of a queue has reached a particular point.
/// When adding a command to a queue, a fence can be provided with the command, to be signaled
/// when the operation finishes. You can check for a fence's current status by calling
/// `is_signaled` or `wait` on it. If the fence is found to be signaled, that means that the queue
/// has completed the operation that is associated with the fence, and all operations that were
/// submitted before it have been completed as well.
///
/// When a queue command accesses a resource, it must be kept alive until the queue command has
/// finished executing, and you may not be allowed to perform certain other operations (or even any)
/// while the resource is in use. By calling `is_signaled` or `wait`, the queue will be notified
/// when the fence is signaled, so that all resources of the associated queue operation and
/// preceding operations can be released.
///
/// Because of this, it is highly recommended to call `is_signaled` or `wait` on your fences.
/// Otherwise, the queue will hold onto resources indefinitely (using up memory)
/// and resource locks will not be released, which may cause errors when submitting future
/// queue operations. It is not strictly necessary to wait for *every* fence, as a fence
/// that was signaled later in the queue will automatically clean up resources associated with
/// earlier fences too.
#[derive(Debug)]
pub struct Fence {
    handle: ash::vk::Fence,
    device: Arc<Device>,

    // Indicates whether this fence was taken from the fence pool.
    // If true, will be put back into fence pool on drop.
    must_put_in_pool: bool,

    _export_handle_types: ExternalFenceHandleTypes,

    state: Mutex<FenceState>,
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
            must_put_in_pool: false,

            _export_handle_types: export_handle_types,

            state: Mutex::new(FenceState {
                is_signaled: signaled,
                ..Default::default()
            }),
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
                    must_put_in_pool: true,

                    _export_handle_types: ExternalFenceHandleTypes::empty(),

                    state: Mutex::new(Default::default()),
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
            must_put_in_pool: false,

            _export_handle_types: export_handle_types,

            state: Mutex::new(FenceState {
                is_signaled: signaled,
                ..Default::default()
            }),
        }
    }

    /// Returns true if the fence is signaled.
    pub fn is_signaled(&self) -> Result<bool, OomError> {
        let queue_to_signal = {
            let mut state = self.lock();

            // If the fence is already signaled, or it's unsignaled but there's no queue that
            // could signal it, return the currently known value.
            if let Some(is_signaled) = state.status() {
                return Ok(is_signaled);
            }

            // We must ask Vulkan for the state.
            let result = unsafe {
                let fns = self.device.fns();
                (fns.v1_0.get_fence_status)(self.device.internal_object(), self.handle)
            };

            match result {
                ash::vk::Result::SUCCESS => unsafe { state.set_signaled() },
                ash::vk::Result::NOT_READY => return Ok(false),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        // If we have a queue that we need to signal our status to,
        // do so now after the state lock is dropped, to avoid deadlocks.
        if let Some(queue) = queue_to_signal {
            unsafe {
                queue.with(|mut q| q.fence_signaled(self));
            }
        }

        Ok(true)
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed.
    ///
    /// Returns `Ok` if the fence is now signaled. Returns `Err` if the timeout was reached instead.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), FenceError> {
        let queue_to_signal = {
            let mut state = self.state.lock();

            // If the fence is already signaled, we don't need to wait.
            if let Some(true) = state.status() {
                return Ok(());
            }

            let timeout_ns = timeout.map_or(u64::MAX, |timeout| {
                timeout
                    .as_secs()
                    .saturating_mul(1_000_000_000)
                    .saturating_add(timeout.subsec_nanos() as u64)
            });

            let result = unsafe {
                let fns = self.device.fns();
                (fns.v1_0.wait_for_fences)(
                    self.device.internal_object(),
                    1,
                    &self.handle,
                    ash::vk::TRUE,
                    timeout_ns,
                )
            };

            match result {
                ash::vk::Result::SUCCESS => unsafe { state.set_signaled() },
                ash::vk::Result::TIMEOUT => return Err(FenceError::Timeout),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        // If we have a queue that we need to signal our status to,
        // do so now after the state lock is dropped, to avoid deadlocks.
        if let Some(queue) = queue_to_signal {
            unsafe {
                queue.with(|mut q| q.fence_signaled(self));
            }
        }

        Ok(())
    }

    /// Waits for multiple fences at once.
    ///
    /// # Panic
    ///
    /// Panics if not all fences belong to the same device.
    #[inline]
    pub fn multi_wait<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
        timeout: Option<Duration>,
    ) -> Result<(), FenceError> {
        let fences: SmallVec<[_; 8]> = fences.into_iter().collect();
        Self::validate_multi_wait(&fences, timeout)?;

        unsafe { Self::multi_wait_unchecked(fences, timeout) }
    }

    fn validate_multi_wait(
        fences: &[&Fence],
        _timeout: Option<Duration>,
    ) -> Result<(), FenceError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for fence in fences {
            // VUID-vkWaitForFences-pFences-parent
            assert_eq!(device, &fence.device);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn multi_wait_unchecked<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
        timeout: Option<Duration>,
    ) -> Result<(), FenceError> {
        let queues_to_signal: SmallVec<[_; 8]> = {
            let iter = fences.into_iter();
            let mut fences_vk: SmallVec<[_; 8]> = SmallVec::new();
            let mut fences: SmallVec<[_; 8]> = SmallVec::new();
            let mut states: SmallVec<[_; 8]> = SmallVec::new();

            for fence in iter {
                let state = fence.state.lock();

                // Skip the fences that are already signaled.
                if !state.status().unwrap_or(false) {
                    fences_vk.push(fence.handle);
                    fences.push(fence);
                    states.push(state);
                }
            }

            // VUID-vkWaitForFences-fenceCount-arraylength
            // If there are no fences, or all the fences are signaled, we don't need to wait.
            if fences_vk.is_empty() {
                return Ok(());
            }

            let device = &fences[0].device;
            let timeout_ns = timeout.map_or(u64::MAX, |timeout| {
                timeout
                    .as_secs()
                    .saturating_mul(1_000_000_000)
                    .saturating_add(timeout.subsec_nanos() as u64)
            });

            let result = {
                let fns = device.fns();
                (fns.v1_0.wait_for_fences)(
                    device.internal_object(),
                    fences_vk.len() as u32,
                    fences_vk.as_ptr(),
                    ash::vk::TRUE, // TODO: let the user choose false here?
                    timeout_ns,
                )
            };

            match result {
                ash::vk::Result::SUCCESS => fences
                    .into_iter()
                    .zip(&mut states)
                    .filter_map(|(fence, state)| state.set_signaled().map(|state| (state, fence)))
                    .collect(),
                ash::vk::Result::TIMEOUT => return Err(FenceError::Timeout),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        // If we have queues that we need to signal our status to,
        // do so now after the state locks are dropped, to avoid deadlocks.
        for (queue, fence) in queues_to_signal {
            queue.with(|mut q| q.fence_signaled(fence));
        }

        Ok(())
    }

    /// Resets the fence.
    ///
    /// The fence must not be in use by a queue operation.
    #[inline]
    pub fn reset(&self) -> Result<(), FenceError> {
        let mut state = self.state.lock();
        self.validate_reset(&state)?;

        unsafe { Ok(self.reset_unchecked_locked(&mut state)?) }
    }

    fn validate_reset(&self, state: &FenceState) -> Result<(), FenceError> {
        // VUID-vkResetFences-pFences-01123
        if state.is_in_use() {
            return Err(FenceError::InUse);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_unchecked(&self) -> Result<(), VulkanError> {
        let mut state = self.state.lock();
        self.reset_unchecked_locked(&mut state)
    }

    unsafe fn reset_unchecked_locked(&self, state: &mut FenceState) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        (fns.v1_0.reset_fences)(self.device.internal_object(), 1, &self.handle)
            .result()
            .map_err(VulkanError::from)?;

        state.reset();

        Ok(())
    }

    /// Resets multiple fences at once.
    ///
    /// The fences must not be in use by a queue operation.
    ///
    /// # Panic
    ///
    /// - Panics if not all fences belong to the same device.
    #[inline]
    pub fn multi_reset<'a>(fences: impl IntoIterator<Item = &'a Fence>) -> Result<(), FenceError> {
        let (fences, mut states): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = fences
            .into_iter()
            .map(|fence| {
                let state = fence.state.lock();
                (fence, state)
            })
            .unzip();
        Self::validate_multi_reset(&fences, &states)?;

        unsafe { Ok(Self::multi_reset_unchecked_locked(&fences, &mut states)?) }
    }

    fn validate_multi_reset(
        fences: &[&Fence],
        states: &[MutexGuard<'_, FenceState>],
    ) -> Result<(), FenceError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for (fence, state) in fences.iter().zip(states) {
            // VUID-vkResetFences-pFences-parent
            assert_eq!(device, &fence.device);

            // VUID-vkResetFences-pFences-01123
            if state.is_in_use() {
                return Err(FenceError::InUse);
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn multi_reset_unchecked<'a>(
        fences: impl IntoIterator<Item = &'a Fence>,
    ) -> Result<(), VulkanError> {
        let (fences, mut states): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = fences
            .into_iter()
            .map(|fence| {
                let state = fence.state.lock();
                (fence, state)
            })
            .unzip();
        Self::multi_reset_unchecked_locked(&fences, &mut states)
    }

    unsafe fn multi_reset_unchecked_locked(
        fences: &[&Fence],
        states: &mut [MutexGuard<'_, FenceState>],
    ) -> Result<(), VulkanError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;
        let fences_vk: SmallVec<[_; 8]> = fences.iter().map(|fence| fence.handle).collect();

        let fns = device.fns();
        (fns.v1_0.reset_fences)(
            device.internal_object(),
            fences_vk.len() as u32,
            fences_vk.as_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        for state in states {
            state.reset();
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn lock(&self) -> MutexGuard<'_, FenceState> {
        self.state.lock()
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

#[derive(Debug, Default)]
pub struct FenceState {
    is_signaled: bool,
    in_use_by: Option<Weak<Queue>>,
}

impl FenceState {
    /// If the fence is already signaled, or it's unsignaled but there's no queue that
    /// could signal it, returns the currently known value.
    #[inline]
    pub(crate) fn status(&self) -> Option<bool> {
        (self.is_signaled || self.in_use_by.is_none()).then_some(self.is_signaled)
    }

    #[inline]
    pub(crate) fn is_in_use(&self) -> bool {
        self.in_use_by.is_some()
    }

    #[inline]
    pub(crate) unsafe fn add_to_queue(&mut self, queue: &Arc<Queue>) {
        self.is_signaled = false;
        self.in_use_by = Some(Arc::downgrade(queue));
    }

    /// Called when a fence first discovers that it is signaled.
    /// Returns the queue that should be informed about it.
    #[inline]
    pub(crate) unsafe fn set_signaled(&mut self) -> Option<Arc<Queue>> {
        self.is_signaled = true;
        self.in_use_by.take().and_then(|queue| queue.upgrade())
    }

    /// Called when a queue is unlocking resources.
    #[inline]
    pub(crate) unsafe fn set_finished(&mut self) {
        self.is_signaled = true;
        self.in_use_by = None;
    }

    #[inline]
    pub(crate) unsafe fn reset(&mut self) {
        debug_assert!(self.in_use_by.is_none());
        self.is_signaled = false;
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

    /// The fence is currently in use by a queue.
    InUse,
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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
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

            Self::InUse => write!(f, "the fence is currently in use by a queue"),
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

        let fence = Fence::new(
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

        assert_should_panic!({
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
        });
    }

    #[test]
    fn multireset_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!({
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

            let _ = Fence::multi_reset([&fence1, &fence2]);
        });
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
