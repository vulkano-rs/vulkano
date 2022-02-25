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
    OomError, Success, VulkanObject,
};
use std::{
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Used to block the GPU execution until an event on the CPU occurs.
///
/// Note that Vulkan implementations may have limits on how long a command buffer will wait for an
/// event to be signaled, in order to avoid interfering with progress of other clients of the GPU.
/// If the event isn't signaled within these limits, results are undefined and may include
/// device loss.
#[derive(Debug)]
pub struct Event {
    handle: ash::vk::Event,
    device: Arc<Device>,
    must_put_in_pool: bool,
}

impl Event {
    /// Creates a new `Event`.
    pub fn new(device: Arc<Device>, create_info: EventCreateInfo) -> Result<Event, OomError> {
        let create_info = ash::vk::EventCreateInfo {
            flags: ash::vk::EventCreateFlags::empty(),
            ..Default::default()
        };

        let handle = unsafe {
            let mut output = MaybeUninit::uninit();
            let fns = device.fns();
            check_errors(fns.v1_0.create_event(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Event {
            device,
            handle,
            must_put_in_pool: false,
        })
    }

    /// Takes an event from the vulkano-provided event pool.
    /// If the pool is empty, a new event will be allocated.
    /// Upon `drop`, the event is put back into the pool.
    ///
    /// For most applications, using the event pool should be preferred,
    /// in order to avoid creating new events every frame.
    pub fn from_pool(device: Arc<Device>) -> Result<Event, OomError> {
        let handle = device.event_pool().lock().unwrap().pop();
        let event = match handle {
            Some(handle) => {
                unsafe {
                    // Make sure the event isn't signaled
                    let fns = device.fns();
                    check_errors(fns.v1_0.reset_event(device.internal_object(), handle))?;
                }
                Event {
                    handle,
                    device,
                    must_put_in_pool: true,
                }
            }
            None => {
                // Pool is empty, alloc new event
                let mut event = Event::new(device, Default::default())?;
                event.must_put_in_pool = true;
                event
            }
        };

        Ok(event)
    }

    /// Returns true if the event is signaled.
    #[inline]
    pub fn signaled(&self) -> Result<bool, OomError> {
        unsafe {
            let fns = self.device.fns();
            let result = check_errors(
                fns.v1_0
                    .get_event_status(self.device.internal_object(), self.handle),
            )?;
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
            let fns = self.device.fns();
            check_errors(
                fns.v1_0
                    .set_event(self.device.internal_object(), self.handle),
            )?;
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
            let fns = self.device.fns();
            check_errors(
                fns.v1_0
                    .reset_event(self.device.internal_object(), self.handle),
            )?;
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

impl Drop for Event {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_event = self.handle;
                self.device.event_pool().lock().unwrap().push(raw_event);
            } else {
                let fns = self.device.fns();
                fns.v1_0
                    .destroy_event(self.device.internal_object(), self.handle, ptr::null());
            }
        }
    }
}

unsafe impl VulkanObject for Event {
    type Object = ash::vk::Event;

    #[inline]
    fn internal_object(&self) -> ash::vk::Event {
        self.handle
    }
}

unsafe impl DeviceOwned for Event {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Event {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for Event {}

impl Hash for Event {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Parameters to create a new `Event`.
#[derive(Clone, Debug)]
pub struct EventCreateInfo {
    pub _ne: crate::NonExhaustive,
}

impl Default for EventCreateInfo {
    fn default() -> Self {
        Self {
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sync::Event;
    use crate::VulkanObject;

    #[test]
    fn event_create() {
        let (device, _) = gfx_dev_and_queue!();
        let event = Event::new(device, Default::default()).unwrap();
        assert!(!event.signaled().unwrap());
    }

    #[test]
    fn event_set() {
        let (device, _) = gfx_dev_and_queue!();
        let mut event = Event::new(device, Default::default()).unwrap();
        assert!(!event.signaled().unwrap());

        event.set();
        assert!(event.signaled().unwrap());
    }

    #[test]
    fn event_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let mut event = Event::new(device, Default::default()).unwrap();
        event.set();
        assert!(event.signaled().unwrap());

        event.reset();
        assert!(!event.signaled().unwrap());
    }

    #[test]
    fn event_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.event_pool().lock().unwrap().len(), 0);
        let event1_internal_obj = {
            let event = Event::from_pool(device.clone()).unwrap();
            assert_eq!(device.event_pool().lock().unwrap().len(), 0);
            event.internal_object()
        };

        assert_eq!(device.event_pool().lock().unwrap().len(), 1);
        let event2 = Event::from_pool(device.clone()).unwrap();
        assert_eq!(device.event_pool().lock().unwrap().len(), 0);
        assert_eq!(event2.internal_object(), event1_internal_obj);
    }
}
