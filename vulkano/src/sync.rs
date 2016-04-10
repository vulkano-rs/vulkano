// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Synchronization primitives for Vulkan objects.
//! 
//! In Vulkan, you have to manually ensure two things:
//! 
//! - That a buffer or an image are not read and written simultaneously (similarly to the CPU).
//! - That writes to a buffer or an image are propagated to other queues by inserting memory
//!   barriers.
//!
//! But don't worry ; this is automatically enforced by this library (as long as you don't use
//! any unsafe function). See the `memory` module for more info.
//!
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use smallvec::SmallVec;

use device::Device;
use device::Queue;
use Error;
use OomError;
use Success;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Base trait for objects that can be used as resources and must be synchronized.
pub unsafe trait Resource {
    /// Returns in which queue family or families this resource can be used.
    fn sharing_mode(&self) -> &SharingMode;

    /// Returns true if the `gpu_access` function should be passed a fence.
    #[inline]
    fn requires_fence(&self) -> bool {
        true
    }

    /// Returns true if the `gpu_access` function should be passed a semaphore.
    #[inline]
    fn requires_semaphore(&self) -> bool {
        true
    }
}

/// Declares in which queue(s) a resource can be used.
///
/// When you create a buffer or an image, you have to tell the Vulkan library in which queue
/// families it will be used. The vulkano library requires you to tell in which queue famiily
/// the resource will be used, even for exclusive mode.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: remove
pub enum SharingMode {
    /// The resource is used is only one queue family.
    Exclusive(u32),
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(Vec<u32>),       // TODO: Vec is too expensive here
}

impl<'a> From<&'a Arc<Queue>> for SharingMode {
    #[inline]
    fn from(queue: &'a Arc<Queue>) -> SharingMode {
        SharingMode::Exclusive(queue.family().id())
    }
}

impl<'a> From<&'a [&'a Arc<Queue>]> for SharingMode {
    #[inline]
    fn from(queues: &'a [&'a Arc<Queue>]) -> SharingMode {
        SharingMode::Concurrent(queues.iter().map(|queue| {
            queue.family().id()
        }).collect())
    }
}

/// Declares in which queue(s) a resource can be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sharing<I> where I: Iterator<Item = u32> {
    /// The resource is used is only one queue family.
    Exclusive,
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(I),
}

/// A fence is used to know when a command buffer submission has finished its execution.
///
/// When a command buffer accesses a ressource, you have to ensure that the CPU doesn't access
/// the same ressource simultaneously (except for concurrent reads). Therefore in order to know
/// when the CPU can access a ressource again, a fence has to be used.
#[derive(Debug)]
pub struct Fence {
    device: Arc<Device>,
    fence: vk::Fence,

    // If true, we know that the `Fence` is signaled. If false, we don't know.
    // This variable exists so that we don't need to call `vkGetFenceStatus` or `vkWaitForFences`
    // multiple times.
    signaled: AtomicBool,
}

impl Fence {
    /// Builds a new fence.
    #[inline]
    pub fn new(device: &Arc<Device>) -> Result<Arc<Fence>, OomError> {
        Fence::new_impl(device, false)
    }

    /// Builds a new fence already in the "signaled" state.
    #[inline]
    pub fn signaled(device: &Arc<Device>) -> Result<Arc<Fence>, OomError> {
        Fence::new_impl(device, true)
    }

    fn new_impl(device: &Arc<Device>, signaled: bool) -> Result<Arc<Fence>, OomError> {
        let vk = device.pointers();

        let fence = unsafe {
            let infos = vk::FenceCreateInfo {
                sType: vk::STRUCTURE_TYPE_FENCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: if signaled { vk::FENCE_CREATE_SIGNALED_BIT } else { 0 },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateFence(device.internal_object(), &infos, ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Fence {
            device: device.clone(),
            fence: fence,
            signaled: AtomicBool::new(signaled),
        }))
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn ready(&self) -> Result<bool, OomError> {
        unsafe {
            if self.signaled.load(Ordering::Relaxed) { return Ok(true); }

            let vk = self.device.pointers();
            let result = try!(check_errors(vk.GetFenceStatus(self.device.internal_object(),
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
            if self.signaled.load(Ordering::Relaxed) { return Ok(()); }

            let timeout_ns = timeout.as_secs().saturating_mul(1_000_000_000)
                                              .saturating_add(timeout.subsec_nanos() as u64);

            let vk = self.device.pointers();
            let r = try!(check_errors(vk.WaitForFences(self.device.internal_object(), 1,
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
        where I: IntoIterator<Item = &'a Fence>
    {
        let mut device = None;

        let fences: SmallVec<[vk::Fence; 8]> = iter.into_iter().filter_map(|fence| {
            match &mut device {
                dev @ &mut None => *dev = Some(fence.device.clone()),
                &mut Some(ref dev) if &**dev as *const Device == &*fence.device as *const Device => {},
                _ => panic!("Tried to wait for  multiple fences that didn't belong to the same device"),
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
            let vk = self.device.pointers();
            vk.ResetFences(self.device.internal_object(), 1, &self.fence);
            self.signaled.store(false, Ordering::Relaxed);
        }
    }

    /// Resets multiple fences at once.
    ///
    /// # Panic
    ///
    /// Panicks if not all fences belong to the same device.
    pub fn multi_reset<'a, I>(iter: I)
        where I: IntoIterator<Item = &'a Fence>
    {
        let mut device = None;

        let fences: SmallVec<[vk::Fence; 8]> = iter.into_iter().map(|fence| {
            match &mut device {
                dev @ &mut None => *dev = Some(fence.device.clone()),
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

unsafe impl VulkanObject for Fence {
    type Object = vk::Fence;

    #[inline]
    fn internal_object(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for Fence {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyFence(self.device.internal_object(), self.fence, ptr::null());
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


/// Used to provide synchronization between command buffers during their execution.
/// 
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore {
    device: Arc<Device>,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    /// Builds a new semaphore.
    #[inline]
    pub fn new(device: &Arc<Device>) -> Result<Arc<Semaphore>, OomError> {
        let vk = device.pointers();

        let semaphore = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            static mut INFOS: vk::SemaphoreCreateInfo = vk::SemaphoreCreateInfo {
                sType: vk::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext: 0 as *const _,   // ptr::null()
                flags: 0,   // reserved
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSemaphore(device.internal_object(), &INFOS,
                                                 ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Semaphore {
            device: device.clone(),
            semaphore: semaphore,
        }))
    }
}

unsafe impl VulkanObject for Semaphore {
    type Object = vk::Semaphore;

    #[inline]
    fn internal_object(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for Semaphore {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySemaphore(self.device.internal_object(), self.semaphore, ptr::null());
        }
    }
}

/// Used to block the GPU execution until an event on the CPU occurs.
///
/// Note that Vulkan implementations may have limits on how long a command buffer will wait for an
/// event to be signaled, in order to avoid interfering with progress of other clients of the GPU.
/// If the event isn't signaled within these limits, results are undefined and may include
/// device loss.
#[derive(Debug)]
pub struct Event {
    device: Arc<Device>,
    event: Mutex<vk::Event>,
}

impl Event {
    /// Builds a new event.
    #[inline]
    pub fn new(device: &Arc<Device>) -> Result<Arc<Event>, OomError> {
        let vk = device.pointers();

        let event = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            static mut INFOS: vk::EventCreateInfo = vk::EventCreateInfo {
                sType: vk::STRUCTURE_TYPE_EVENT_CREATE_INFO,
                pNext: 0 as *const _, //ptr::null(),
                flags: 0,   // reserved
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateEvent(device.internal_object(), &INFOS,
                                             ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Event {
            device: device.clone(),
            event: Mutex::new(event),
        }))
    }

    /// Returns true if the event is signaled.
    #[inline]
    pub fn signaled(&self) -> Result<bool, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let event = self.event.lock().unwrap();
            let result = try!(check_errors(vk.GetEventStatus(self.device.internal_object(),
                                                             *event)));
            match result {
                Success::EventSet => Ok(true),
                Success::EventReset => Ok(false),
                _ => unreachable!()
            }
        }
    }

    /// Changes the `Event` to the signaled state.
    ///
    /// If a command buffer is waiting on this event, it is then unblocked.
    #[inline]
    pub fn set(&self) -> Result<(), OomError> {
        unsafe {
            let vk = self.device.pointers();
            let event = self.event.lock().unwrap();
            try!(check_errors(vk.SetEvent(self.device.internal_object(), *event)).map(|_| ()));
            Ok(())
        }
    }

    /// Changes the `Event` to the unsignaled state.
    #[inline]
    pub fn reset(&self) -> Result<(), OomError> {
        unsafe {
            let vk = self.device.pointers();
            let event = self.event.lock().unwrap();
            try!(check_errors(vk.ResetEvent(self.device.internal_object(), *event)).map(|_| ()));
            Ok(())
        }
    }
}

unsafe impl SynchronizedVulkanObject for Event {
    type Object = vk::Event;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<vk::Event> {
        self.event.lock().unwrap()
    }
}

impl Drop for Event {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            let event = self.event.lock().unwrap();
            vk.DestroyEvent(self.device.internal_object(), *event, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use sync::Event;
    use sync::Fence;
    use sync::Semaphore;

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
    fn semaphore_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = Semaphore::new(&device).unwrap();
    }

    #[test]
    fn event_create() {
        let (device, _) = gfx_dev_and_queue!();
        let event = Event::new(&device).unwrap();
        assert!(!event.signaled().unwrap());
    }

    #[test]
    fn event_set() {
        let (device, _) = gfx_dev_and_queue!();
        let event = Event::new(&device).unwrap();
        assert!(!event.signaled().unwrap());

        event.set().unwrap();
        assert!(event.signaled().unwrap());
    }

    #[test]
    fn event_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let event = Event::new(&device).unwrap();
        event.set().unwrap();
        assert!(event.signaled().unwrap());

        event.reset().unwrap();
        assert!(!event.signaled().unwrap());
    }
}
