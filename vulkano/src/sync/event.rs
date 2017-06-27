// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;

use OomError;
use Success;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

/// Used to block the GPU execution until an event on the CPU occurs.
///
/// Note that Vulkan implementations may have limits on how long a command buffer will wait for an
/// event to be signaled, in order to avoid interfering with progress of other clients of the GPU.
/// If the event isn't signaled within these limits, results are undefined and may include
/// device loss.
#[derive(Debug)]
pub struct Event {
    // The event.
    event: vk::Event,
    // The device.
    device: Arc<Device>,
}

impl Event {
    /// See the docs of new().
    #[inline]
    pub fn raw(device: Arc<Device>) -> Result<Event, OomError> {
        let event = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            static mut INFOS: vk::EventCreateInfo = vk::EventCreateInfo {
                sType: vk::STRUCTURE_TYPE_EVENT_CREATE_INFO,
                pNext: 0 as *const _, //ptr::null(),
                flags: 0, // reserved
            };

            let mut output = mem::uninitialized();
            let vk = device.pointers();
            check_errors(vk.CreateEvent(device.internal_object(),
                                        &INFOS,
                                        ptr::null(),
                                        &mut output))?;
            output
        };

        Ok(Event {
               device: device,
               event: event,
           })
    }

    /// Builds a new event.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    #[inline]
    pub fn new(device: Arc<Device>) -> Arc<Event> {
        Arc::new(Event::raw(device).unwrap())
    }

    /// Returns true if the event is signaled.
    #[inline]
    pub fn signaled(&self) -> Result<bool, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let result = check_errors(vk.GetEventStatus(self.device.internal_object(),
                                                        self.event))?;
            match result {
                Success::EventSet => Ok(true),
                Success::EventReset => Ok(false),
                _ => unreachable!(),
            }
        }
    }

    /// See the docs of set().
    #[inline]
    pub fn set_raw(&mut self) -> Result<(), OomError> {
        unsafe {
            let vk = self.device.pointers();
            check_errors(vk.SetEvent(self.device.internal_object(), self.event))?;
            Ok(())
        }
    }

    /// Changes the `Event` to the signaled state.
    ///
    /// If a command buffer is waiting on this event, it is then unblocked.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    #[inline]
    pub fn set(&mut self) {
        self.set_raw().unwrap();
    }

    /// See the docs of reset().
    #[inline]
    pub fn reset_raw(&mut self) -> Result<(), OomError> {
        unsafe {
            let vk = self.device.pointers();
            check_errors(vk.ResetEvent(self.device.internal_object(), self.event))?;
            Ok(())
        }
    }

    /// Changes the `Event` to the unsignaled state.
    ///
    /// # Panic
    ///
    /// - Panics if the device or host ran out of memory.
    ///
    #[inline]
    pub fn reset(&mut self) {
        self.reset_raw().unwrap();
    }
}

unsafe impl DeviceOwned for Event {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for Event {
    type Object = vk::Event;

    #[inline]
    fn internal_object(&self) -> vk::Event {
        self.event
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use sync::Event;

    #[test]
    fn event_create() {
        let (device, _) = gfx_dev_and_queue!();
        let event = Event::new(device);
        assert!(!event.signaled().unwrap());
    }

    #[test]
    fn event_set() {
        let (device, _) = gfx_dev_and_queue!();
        let mut event = Event::new(device);
        assert!(!event.signaled().unwrap());

        Arc::get_mut(&mut event).unwrap().set();
        assert!(event.signaled().unwrap());
    }

    #[test]
    fn event_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let mut event = Event::new(device);
        Arc::get_mut(&mut event).unwrap().set();
        assert!(event.signaled().unwrap());

        Arc::get_mut(&mut event).unwrap().reset();
        assert!(!event.signaled().unwrap());
    }
}
