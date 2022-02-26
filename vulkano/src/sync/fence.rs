// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    check_errors,
    device::{Device, DeviceOwned},
    Error, OomError, Success, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    error, fmt,
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

    // If true, we know that the `Fence` is signaled. If false, we don't know.
    // This variable exists so that we don't need to call `vkGetFenceStatus` or `vkWaitForFences`
    // multiple times.
    signaled: AtomicBool,

    // Indicates whether this fence was taken from the fence pool.
    // If true, will be put back into fence pool on drop.
    must_put_in_pool: bool,
}

impl Fence {
    /// Creates a new `Fence`.
    pub fn new(device: Arc<Device>, create_info: FenceCreateInfo) -> Result<Fence, OomError> {
        let FenceCreateInfo { signaled, _ne: _ } = create_info;

        let mut flags = ash::vk::FenceCreateFlags::empty();

        if signaled {
            flags |= ash::vk::FenceCreateFlags::SIGNALED;
        }

        let create_info = ash::vk::FenceCreateInfo {
            flags,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_fence(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Fence {
            handle,
            device,
            signaled: AtomicBool::new(signaled),
            must_put_in_pool: false,
        })
    }

    /// Takes a fence from the vulkano-provided fence pool.
    /// If the pool is empty, a new fence will be created.
    /// Upon `drop`, the fence is put back into the pool.
    ///
    /// For most applications, using the fence pool should be preferred,
    /// in order to avoid creating new fences every frame.
    pub fn from_pool(device: Arc<Device>) -> Result<Fence, OomError> {
        let handle = device.fence_pool().lock().unwrap().pop();
        let fence = match handle {
            Some(handle) => {
                unsafe {
                    // Make sure the fence isn't signaled
                    let fns = device.fns();
                    check_errors(fns.v1_0.reset_fences(device.internal_object(), 1, &handle))?;
                }

                Fence {
                    handle,
                    device,
                    signaled: AtomicBool::new(false),
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

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn ready(&self) -> Result<bool, OomError> {
        unsafe {
            if self.signaled.load(Ordering::Relaxed) {
                return Ok(true);
            }

            let fns = self.device.fns();
            let result = check_errors(
                fns.v1_0
                    .get_fence_status(self.device.internal_object(), self.handle),
            )?;
            match result {
                Success::Success => {
                    self.signaled.store(true, Ordering::Relaxed);
                    Ok(true)
                }
                Success::NotReady => Ok(false),
                _ => unreachable!(),
            }
        }
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed.
    ///
    /// Returns `Ok` if the fence is now signaled. Returns `Err` if the timeout was reached instead.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), FenceWaitError> {
        unsafe {
            if self.signaled.load(Ordering::Relaxed) {
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
            let r = check_errors(fns.v1_0.wait_for_fences(
                self.device.internal_object(),
                1,
                &self.handle,
                ash::vk::TRUE,
                timeout_ns,
            ))?;

            match r {
                Success::Success => {
                    self.signaled.store(true, Ordering::Relaxed);
                    Ok(())
                }
                Success::Timeout => Err(FenceWaitError::Timeout),
                _ => unreachable!(),
            }
        }
    }

    /// Waits for multiple fences at once.
    ///
    /// # Panic
    ///
    /// Panics if not all fences belong to the same device.
    pub fn multi_wait<'a, I>(iter: I, timeout: Option<Duration>) -> Result<(), FenceWaitError>
    where
        I: IntoIterator<Item = &'a Fence>,
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[ash::vk::Fence; 8]> = iter
            .into_iter()
            .filter_map(|fence| {
                match &mut device {
                    dev @ &mut None => *dev = Some(&*fence.device),
                    &mut Some(ref dev)
                        if &**dev as *const Device == &*fence.device as *const Device => {}
                    _ => panic!(
                        "Tried to wait for multiple fences that didn't belong to the \
                                 same device"
                    ),
                };

                if fence.signaled.load(Ordering::Relaxed) {
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

        let r = if let Some(device) = device {
            unsafe {
                let fns = device.fns();
                check_errors(fns.v1_0.wait_for_fences(
                    device.internal_object(),
                    fences.len() as u32,
                    fences.as_ptr(),
                    ash::vk::TRUE,
                    timeout_ns,
                ))?
            }
        } else {
            return Ok(());
        };

        match r {
            Success::Success => Ok(()),
            Success::Timeout => Err(FenceWaitError::Timeout),
            _ => unreachable!(),
        }
    }

    /// Resets the fence.
    // This function takes a `&mut self` because the Vulkan API requires that the fence be
    // externally synchronized.
    #[inline]
    pub fn reset(&mut self) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();
            check_errors(
                fns.v1_0
                    .reset_fences(self.device.internal_object(), 1, &self.handle),
            )?;
            self.signaled.store(false, Ordering::Relaxed);
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
                    &mut Some(ref dev)
                        if &**dev as *const Device == &*fence.device as *const Device => {}
                    _ => panic!(
                        "Tried to reset multiple fences that didn't belong to the same \
                                 device"
                    ),
                };

                fence.signaled.store(false, Ordering::Relaxed);
                fence.handle
            })
            .collect();

        if let Some(device) = device {
            unsafe {
                let fns = device.fns();
                check_errors(fns.v1_0.reset_fences(
                    device.internal_object(),
                    fences.len() as u32,
                    fences.as_ptr(),
                ))?;
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
                self.device.fence_pool().lock().unwrap().push(raw_fence);
            } else {
                let fns = self.device.fns();
                fns.v1_0
                    .destroy_fence(self.device.internal_object(), self.handle, ptr::null());
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

    pub _ne: crate::NonExhaustive,
}

impl Default for FenceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            signaled: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Error that can be returned when waiting on a fence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FenceWaitError {
    /// Not enough memory to complete the wait.
    OomError(OomError),

    /// The specified timeout wasn't long enough.
    Timeout,

    /// The device has been lost.
    DeviceLostError,
}

impl error::Error for FenceWaitError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            FenceWaitError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for FenceWaitError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                FenceWaitError::OomError(_) => "no memory available",
                FenceWaitError::Timeout => "the timeout has been reached",
                FenceWaitError::DeviceLostError => "the device was lost",
            }
        )
    }
}

impl From<Error> for FenceWaitError {
    #[inline]
    fn from(err: Error) -> FenceWaitError {
        match err {
            Error::OutOfHostMemory => FenceWaitError::OomError(From::from(err)),
            Error::OutOfDeviceMemory => FenceWaitError::OomError(From::from(err)),
            Error::DeviceLost => FenceWaitError::DeviceLostError,
            _ => panic!("Unexpected error value: {}", err as i32),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sync::fence::FenceCreateInfo;
    use crate::sync::Fence;
    use crate::VulkanObject;
    use std::time::Duration;

    #[test]
    fn fence_create() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(device.clone(), Default::default()).unwrap();
        assert!(!fence.ready().unwrap());
    }

    #[test]
    fn fence_create_signaled() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device.clone(),
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fence.ready().unwrap());
    }

    #[test]
    fn fence_signaled_wait() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            device.clone(),
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
            device.clone(),
            FenceCreateInfo {
                signaled: true,
                ..Default::default()
            },
        )
        .unwrap();
        fence.reset().unwrap();
        assert!(!fence.ready().unwrap());
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

        assert_eq!(device.fence_pool().lock().unwrap().len(), 0);
        let fence1_internal_obj = {
            let fence = Fence::from_pool(device.clone()).unwrap();
            assert_eq!(device.fence_pool().lock().unwrap().len(), 0);
            fence.internal_object()
        };

        assert_eq!(device.fence_pool().lock().unwrap().len(), 1);
        let fence2 = Fence::from_pool(device.clone()).unwrap();
        assert_eq!(device.fence_pool().lock().unwrap().len(), 0);
        assert_eq!(fence2.internal_object(), fence1_internal_obj);
    }
}
