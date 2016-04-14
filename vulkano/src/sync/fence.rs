// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use smallvec::SmallVec;

use device::Device;
use Error;
use OomError;
use Success;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// A fence is used to know when a command buffer submission has finished its execution.
///
/// When a command buffer accesses a ressource, you have to ensure that the CPU doesn't access
/// the same ressource simultaneously (except for concurrent reads). Therefore in order to know
/// when the CPU can access a ressource again, a fence has to be used.
#[derive(Debug)]
pub struct Fence<D = Arc<Device>> where D: Deref<Target = Device> {
    fence: vk::Fence,

    device: D,
    device_raw: vk::Device,

    // If true, we know that the `Fence` is signaled. If false, we don't know.
    // This variable exists so that we don't need to call `vkGetFenceStatus` or `vkWaitForFences`
    // multiple times.
    signaled: AtomicBool,
}

impl<D> Fence<D> where D: Deref<Target = Device> {
    /// Builds a new fence.
    #[inline]
    pub fn raw(device: &D) -> Result<Fence<D>, OomError>
        where D: Clone
    {
        Fence::new_impl(device, false)
    }

    /// Builds a new fence.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn new(device: &D) -> Arc<Fence<D>>
        where D: Clone
    {
        Arc::new(Fence::raw(device).unwrap())
    }

    /// Builds a new fence already in the "signaled" state.
    #[inline]
    pub fn signaled_raw(device: &D) -> Result<Fence<D>, OomError>
        where D: Clone
    {
        Fence::new_impl(device, true)
    }

    /// Builds a new fence already in the "signaled" state.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn signaled(device: &D) -> Arc<Fence<D>>
        where D: Clone
    {
        Arc::new(Fence::signaled_raw(device).unwrap())
    }

    fn new_impl(device_ptr: &D, signaled: bool) -> Result<Fence<D>, OomError>
        where D: Clone
    {
        let device: &Device = &*device_ptr;
        let vk = device.pointers();

        let fence = unsafe {
            let infos = vk::FenceCreateInfo {
                sType: vk::STRUCTURE_TYPE_FENCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: if signaled { vk::FENCE_CREATE_SIGNALED_BIT } else { 0 },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateFence(device.internal_object(), &infos,
                                             ptr::null(), &mut output)));
            output
        };

        Ok(Fence {
            fence: fence,
            device: device_ptr.clone(),
            device_raw: device.internal_object(),
            signaled: AtomicBool::new(signaled),
        })
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn ready(&self) -> Result<bool, OomError> {
        unsafe {
            let device: &Device = &*self.device;
            assert_eq!(device.internal_object(), self.device_raw);

            if self.signaled.load(Ordering::Relaxed) { return Ok(true); }

            let vk = device.pointers();
            let result = try!(check_errors(vk.GetFenceStatus(device.internal_object(),
                                                             self.fence)));
            match result {
                Success::Success => {
                    self.signaled.store(true, Ordering::Relaxed);
                    Ok(true)
                },
                Success::NotReady => Ok(false),
                _ => unreachable!()
            }
        }
    }

    /// Waits until the fence is signaled, or at least until the number of nanoseconds of the
    /// timeout has elapsed.
    ///
    /// Returns `Ok` if the fence is now signaled. Returns `Err` if the timeout was reached instead.
    pub fn wait(&self, timeout: Duration) -> Result<(), FenceWaitError> {
        unsafe {
            let device: &Device = &*self.device;
            assert_eq!(device.internal_object(), self.device_raw);

            if self.signaled.load(Ordering::Relaxed) { return Ok(()); }

            let timeout_ns = timeout.as_secs().saturating_mul(1_000_000_000)
                                              .saturating_add(timeout.subsec_nanos() as u64);

            let vk = device.pointers();
            let r = try!(check_errors(vk.WaitForFences(device.internal_object(), 1,
                                                       &self.fence, vk::TRUE, timeout_ns)));

            match r {
                Success::Success => {
                    self.signaled.store(true, Ordering::Relaxed);
                    Ok(())
                },
                Success::Timeout => {
                    Err(FenceWaitError::Timeout)
                },
                _ => unreachable!()
            }
        }
    }

    /// Waits for multiple fences at once.
    ///
    /// # Panic
    ///
    /// Panicks if not all fences belong to the same device.
    pub fn multi_wait<'a, I>(iter: I, timeout: Duration) -> Result<(), FenceWaitError>
        where I: IntoIterator<Item = &'a Fence<D>>, D: 'a
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[vk::Fence; 8]> = iter.into_iter().filter_map(|fence| {
            match &mut device {
                dev @ &mut None => *dev = Some(&*fence.device),
                &mut Some(ref dev) if &**dev as *const Device == &*fence.device as *const Device => {},
                _ => panic!("Tried to wait for multiple fences that didn't belong to the \
                             same device"),
            };

            if fence.signaled.load(Ordering::Relaxed) {
                None
            } else {
                Some(fence.fence)
            }
        }).collect();

        let timeout_ns = timeout.as_secs().saturating_mul(1_000_000_000)
                                          .saturating_add(timeout.subsec_nanos() as u64);

        let r = if let Some(device) = device {
            unsafe {
                let vk = device.pointers();
                try!(check_errors(vk.WaitForFences(device.internal_object(), fences.len() as u32,
                                                   fences.as_ptr(), vk::TRUE, timeout_ns)))
            }
        } else {
            return Ok(());
        };

        match r {
            Success::Success => Ok(()),
            Success::Timeout => Err(FenceWaitError::Timeout),
            _ => unreachable!()
        }
    }

    /// Resets the fence.
    // FIXME: must synchronize the fence
    #[inline]
    pub fn reset(&self) {
        unsafe {
            let device: &Device = &*self.device;
            assert_eq!(device.internal_object(), self.device_raw);

            let vk = device.pointers();
            vk.ResetFences(device.internal_object(), 1, &self.fence);
            self.signaled.store(false, Ordering::Relaxed);
        }
    }

    /// Resets multiple fences at once.
    ///
    /// # Panic
    ///
    /// Panicks if not all fences belong to the same device.
    pub fn multi_reset<'a, I>(iter: I)
        where I: IntoIterator<Item = &'a Fence<D>>, D: 'a
    {
        let mut device: Option<&Device> = None;

        let fences: SmallVec<[vk::Fence; 8]> = iter.into_iter().map(|fence| {
            match &mut device {
                dev @ &mut None => *dev = Some(&*fence.device),
                &mut Some(ref dev) if &**dev as *const Device == &*fence.device as *const Device => {},
                _ => panic!("Tried to reset multiple fences that didn't belong to the same device"),
            };

            fence.signaled.store(false, Ordering::Relaxed);
            fence.fence
        }).collect();

        if let Some(device) = device {
            unsafe {
                let vk = device.pointers();
                vk.ResetFences(device.internal_object(), fences.len() as u32, fences.as_ptr());
            }
        }
    }
}

unsafe impl<D> VulkanObject for Fence<D> where D: Deref<Target = Device> {
    type Object = vk::Fence;

    #[inline]
    fn internal_object(&self) -> vk::Fence {
        self.fence
    }
}

impl<D> Drop for Fence<D> where D: Deref<Target = Device> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let device: &Device = &*self.device;
            assert_eq!(device.internal_object(), self.device_raw);

            let vk = device.pointers();
            vk.DestroyFence(device.internal_object(), self.fence, ptr::null());
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
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            FenceWaitError::OomError(ref err) => Some(err),
            _ => None
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
            _ => panic!("Unexpected error value: {}", err as i32)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use sync::Fence;

    #[test]
    #[ignore]       // TODO: fails on AMD + Windows
    fn fence_create() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(&device).unwrap();
        assert!(!fence.ready().unwrap());
    }

    #[test]
    fn fence_create_signaled() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::signaled(&device).unwrap();
        assert!(fence.ready().unwrap());
    }

    #[test]
    fn fence_signaled_wait() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::signaled(&device).unwrap();
        fence.wait(Duration::new(0, 10)).unwrap();
    }

    #[test]
    #[ignore]       // TODO: fails on AMD + Windows
    fn fence_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::signaled(&device).unwrap();
        fence.reset();
        assert!(!fence.ready().unwrap());
    }

    #[test]
    #[should_panic = "Tried to wait for multiple fences that didn't belong to the same device"]
    fn multiwait_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let fence1 = Fence::signaled(&device1).unwrap();
        let fence2 = Fence::signaled(&device2).unwrap();

        let _ = Fence::multi_wait([&*fence1, &*fence2].iter().cloned(), Duration::new(0, 10));
    }

    #[test]
    #[should_panic = "Tried to reset multiple fences that didn't belong to the same device"]
    fn multireset_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let fence1 = Fence::signaled(&device1).unwrap();
        let fence2 = Fence::signaled(&device2).unwrap();

        let _ = Fence::multi_reset([&*fence1, &*fence2].iter().cloned());
    }
}
