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
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
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
pub struct Fence {
    device: Arc<Device>,
    fence: vk::Fence,
}

impl Fence {
    /// Builds a new fence.
    #[inline]
    pub fn new(device: &Arc<Device>) -> Arc<Fence> {
        Fence::new_impl(device, false)
    }

    /// Builds a new fence already in the "signaled" state.
    #[inline]
    pub fn signaled(device: &Arc<Device>) -> Arc<Fence> {
        Fence::new_impl(device, true)
    }

    fn new_impl(device: &Arc<Device>, signaled: bool) -> Arc<Fence> {
        let vk = device.pointers();

        let fence = unsafe {
            let infos = vk::FenceCreateInfo {
                sType: vk::STRUCTURE_TYPE_FENCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: if signaled { vk::FENCE_CREATE_SIGNALED_BIT } else { 0 },
            };

            let mut output = mem::uninitialized();
            vk.CreateFence(device.internal_object(), &infos, ptr::null(), &mut output);
            output
        };

        Arc::new(Fence {
            device: device.clone(),
            fence: fence,
        })
    }

    /// Returns true if the fence is signaled.
    #[inline]
    pub fn ready(&self) -> Result<bool, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let result = try!(check_errors(vk.GetFenceStatus(self.device.internal_object(),
                                                             self.fence)));
            match result {
                Success::Success => Ok(true),
                Success::NotReady => Ok(false),
                _ => unreachable!()
            }
        }
    }

    /// Waits until the fence is signaled, or at least until the number of nanoseconds of the
    /// timeout has elapsed.
    ///
    /// Returns `Ok` if the fence is now signaled. Returns `Err` if the timeout was reached instead.
    pub fn wait(&self, timeout_ns: u64) -> Result<(), ()> {
        unsafe {
            let vk = self.device.pointers();
            vk.WaitForFences(self.device.internal_object(), 1, &self.fence, vk::TRUE, timeout_ns);
            Ok(())      // FIXME: 
        }
    }

    /// Resets the fence.
    #[inline]
    pub fn reset(&self) {
        unsafe {
            let vk = self.device.pointers();
            vk.ResetFences(self.device.internal_object(), 1, &self.fence);
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

        let fences: Vec<vk::Fence> = iter.into_iter().map(|fence| {
            match &mut device {
                dev @ &mut None => *dev = Some(fence.device.clone()),
                &mut Some(ref dev) if &**dev as *const Device == &*fence.device as *const Device => {},
                _ => panic!("Tried to reset multiple fences that didn't belong to the same device"),
            };

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

impl Drop for Fence {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyFence(self.device.internal_object(), self.fence, ptr::null());
        }
    }
}

/// Used to provide synchronization between command buffers during their execution.
/// 
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
///
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
pub struct Event {
    device: Arc<Device>,
    event: vk::Event,
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
            event: event,
        }))
    }

    /// Returns true if the event is signaled.
    #[inline]
    pub fn signaled(&self) -> Result<bool, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let result = try!(check_errors(vk.GetEventStatus(self.device.internal_object(),
                                                             self.event)));
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
            try!(check_errors(vk.SetEvent(self.device.internal_object(), self.event)).map(|_| ()));
            Ok(())
        }
    }

    /// Changes the `Event` to the unsignaled state.
    #[inline]
    pub fn reset(&self) -> Result<(), OomError> {
        unsafe {
            let vk = self.device.pointers();
            try!(check_errors(vk.ResetEvent(self.device.internal_object(), self.event)).map(|_| ()));
            Ok(())
        }
    }
}

impl Drop for Event {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyEvent(self.device.internal_object(), self.event, ptr::null());
        }
    }
}
