// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use Error;
use OomError;
use SafeDeref;
use Success;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

/// A fence is used to know when a command buffer submission has finished its execution.
///
/// When a command buffer accesses a resource, you have to ensure that the CPU doesn't access
/// the same resource simultaneously (except for concurrent reads). Therefore in order to know
/// when the CPU can access a resource again, a fence has to be used.
#[derive(Debug)]
pub struct Fence<D = Arc<Device>>
    where D: SafeDeref<Target = Device>
{
    fence: vk::Fence,

    device: D,

    // If true, we know that the `Fence` is signaled. If false, we don't know.
    // This variable exists so that we don't need to call `vkGetFenceStatus` or `vkWaitForFences`
    // multiple times.
    signaled: AtomicBool,

    // Indicates whether this fence was taken from the fence pool.
    // If true, will be put back into fence pool on drop.
    must_put_in_pool: bool,
}

impl<D> Fence<D>
    where D: SafeDeref<Target = Device>
{
    /// Takes a fence from the vulkano-provided fence pool.
    /// If the pool is empty, a new fence will be allocated.
    /// Upon `drop`, the fence is put back into the pool.
    ///
    /// For most applications, using the fence pool should be preferred,
    /// in order to avoid creating new fences every frame.
    pub fn from_pool(device: D) -> Result<Fence<D>, OomError> {
        let maybe_raw_fence = device.fence_pool().lock().unwrap().pop();
        match maybe_raw_fence {
            Some(raw_fence) => {
                unsafe {
                    // Make sure the fence isn't signaled
                    let vk = device.pointers();
                    check_errors(vk.ResetFences(device.internal_object(), 1, &raw_fence))?;
                }
                Ok(Fence {
                       fence: raw_fence,
                       device: device,
                       signaled: AtomicBool::new(false),
                       must_put_in_pool: true,
                   })
            },
            None => {
                // Pool is empty, alloc new fence
                Fence::alloc_impl(device, false, true)
            },
        }
    }

    /// Builds a new fence.
    #[inline]
    pub fn alloc(device: D) -> Result<Fence<D>, OomError> {
        Fence::alloc_impl(device, false, false)
    }

    /// Builds a new fence in signaled state.
    #[inline]
    pub fn alloc_signaled(device: D) -> Result<Fence<D>, OomError> {
        Fence::alloc_impl(device, true, false)
    }

    fn alloc_impl(device: D, signaled: bool, must_put_in_pool: bool) -> Result<Fence<D>, OomError> {
        let fence = unsafe {
            let infos = vk::FenceCreateInfo {
                sType: vk::STRUCTURE_TYPE_FENCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: if signaled {
                    vk::FENCE_CREATE_SIGNALED_BIT
                } else {
                    0
                },
            };

            let vk = device.pointers();
            let mut output = MaybeUninit::uninit();
            check_errors(vk.CreateFence(device.internal_object(),
                                        &infos,
                                        ptr::null(),
                                        output.as_mut_ptr()))?;
            output.assume_init()
        };

        Ok(Fence {
               fence: fence,
               device: device,
               signaled: AtomicBool::new(signaled),
               must_put_in_pool: must_put_in_pool,
           })
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn ready(&self) -> Result<bool, OomError> {
        unsafe {
            if self.signaled.load(Ordering::Relaxed) {
                return Ok(true);
            }

            let vk = self.device.pointers();
            let result = check_errors(vk.GetFenceStatus(self.device.internal_object(),
                                                        self.fence))?;
            match result {
                Success::Success => {
                    self.signaled.store(true, Ordering::Relaxed);
                    Ok(true)
                },
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
                u64::max_value()
            };

            let vk = self.device.pointers();
            let r = check_errors(vk.WaitForFences(self.device.internal_object(),
                                                  1,
                                                  &self.fence,
                                                  vk::TRUE,
                                                  timeout_ns))?;

            match r {
                Success::Success => {
                    self.signaled.store(true, Ordering::Relaxed);
                    Ok(())
                },
                Success::Timeout => {
                    Err(FenceWaitError::Timeout)
                },
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
        where I: IntoIterator<Item = &'a Fence<D>>,
              D: 'a
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[vk::Fence; 8]> = iter.into_iter()
            .filter_map(|fence| {
                match &mut device {
                    dev @ &mut None => *dev = Some(&*fence.device),
                    &mut Some(ref dev)
                        if &**dev as *const Device == &*fence.device as *const Device => {},
                    _ => panic!("Tried to wait for multiple fences that didn't belong to the \
                                 same device"),
                };

                if fence.signaled.load(Ordering::Relaxed) {
                    None
                } else {
                    Some(fence.fence)
                }
            })
            .collect();

        let timeout_ns = if let Some(timeout) = timeout {
            timeout
                .as_secs()
                .saturating_mul(1_000_000_000)
                .saturating_add(timeout.subsec_nanos() as u64)
        } else {
            u64::max_value()
        };

        let r = if let Some(device) = device {
            unsafe {
                let vk = device.pointers();
                check_errors(vk.WaitForFences(device.internal_object(),
                                              fences.len() as u32,
                                              fences.as_ptr(),
                                              vk::TRUE,
                                              timeout_ns))?
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
            let vk = self.device.pointers();
            check_errors(vk.ResetFences(self.device.internal_object(), 1, &self.fence))?;
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
        where I: IntoIterator<Item = &'a mut Fence<D>>,
              D: 'a
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[vk::Fence; 8]> = iter.into_iter()
            .map(|fence| {
                match &mut device {
                    dev @ &mut None => *dev = Some(&*fence.device),
                    &mut Some(ref dev)
                        if &**dev as *const Device == &*fence.device as *const Device => {},
                    _ => panic!("Tried to reset multiple fences that didn't belong to the same \
                                 device"),
                };

                fence.signaled.store(false, Ordering::Relaxed);
                fence.fence
            })
            .collect();

        if let Some(device) = device {
            unsafe {
                let vk = device.pointers();
                check_errors(vk.ResetFences(device.internal_object(),
                                            fences.len() as u32,
                                            fences.as_ptr()))?;
            }
        }
        Ok(())
    }
}

unsafe impl DeviceOwned for Fence {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<D> VulkanObject for Fence<D>
    where D: SafeDeref<Target = Device>
{
    type Object = vk::Fence;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_FENCE;

    #[inline]
    fn internal_object(&self) -> vk::Fence {
        self.fence
    }
}

impl<D> Drop for Fence<D>
    where D: SafeDeref<Target = Device>
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_fence = self.fence;
                self.device.fence_pool().lock().unwrap().push(raw_fence);
            } else {
                let vk = self.device.pointers();
                vk.DestroyFence(self.device.internal_object(), self.fence, ptr::null());
            }
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
    fn description(&self) -> &str {
        match *self {
            FenceWaitError::OomError(_) => "no memory available",
            FenceWaitError::Timeout => "the timeout has been reached",
            FenceWaitError::DeviceLostError => "the device was lost",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&dyn error::Error> {
        match *self {
            FenceWaitError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for FenceWaitError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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
    use VulkanObject;
    use std::time::Duration;
    use sync::Fence;

    #[test]
    fn fence_create() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::alloc(device.clone()).unwrap();
        assert!(!fence.ready().unwrap());
    }

    #[test]
    fn fence_create_signaled() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::alloc_signaled(device.clone()).unwrap();
        assert!(fence.ready().unwrap());
    }

    #[test]
    fn fence_signaled_wait() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::alloc_signaled(device.clone()).unwrap();
        fence.wait(Some(Duration::new(0, 10))).unwrap();
    }

    #[test]
    fn fence_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let mut fence = Fence::alloc_signaled(device.clone()).unwrap();
        fence.reset().unwrap();
        assert!(!fence.ready().unwrap());
    }

    #[test]
    fn multiwait_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!("Tried to wait for multiple fences that didn't belong \
                              to the same device",
                             {
                                 let fence1 = Fence::alloc_signaled(device1.clone()).unwrap();
                                 let fence2 = Fence::alloc_signaled(device2.clone()).unwrap();

                                 let _ = Fence::multi_wait([&fence1, &fence2].iter().cloned(),
                                                           Some(Duration::new(0, 10)));
                             });
    }

    #[test]
    fn multireset_different_devices() {
        use std::iter::once;

        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!("Tried to reset multiple fences that didn't belong \
                              to the same device",
                             {
                                 let mut fence1 = Fence::alloc_signaled(device1.clone()).unwrap();
                                 let mut fence2 = Fence::alloc_signaled(device2.clone()).unwrap();

                                 let _ = Fence::multi_reset(once(&mut fence1)
                                                                .chain(once(&mut fence2)));
                             });
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
