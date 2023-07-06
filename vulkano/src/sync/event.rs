// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! An event provides fine-grained synchronization within a single queue, or from the host to a
//! queue.
//!
//! When an event is signaled from a queue using the [`set_event`] command buffer command,
//! an event acts similar to a [pipeline barrier], but the synchronization scopes are split:
//! the source synchronization scope includes only commands before the `set_event` command,
//! while the destination synchronization scope includes only commands after the
//! [`wait_events`] command. Commands in between the two are not included.
//!
//! An event can also be signaled from the host, by calling the [`set`] method directly on the
//! [`Event`].
//!
//! [`set_event`]: crate::command_buffer::CommandBufferBuilder::set_event
//! [pipeline barrier]: crate::command_buffer::CommandBufferBuilder::pipeline_barrier
//! [`wait_events`]: crate::command_buffer::CommandBufferBuilder::wait_events
//! [`set`]: Event::set

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    OomError, Requires, RequiresAllOf, RequiresOneOf, VulkanError, VulkanObject,
};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
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
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,
    must_put_in_pool: bool,
}

impl Event {
    /// Creates a new `Event`.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, the
    /// [`events`](crate::device::Features::events)
    /// feature must be enabled on the device.
    #[inline]
    pub fn new(device: Arc<Device>, _create_info: EventCreateInfo) -> Result<Event, EventError> {
        // VUID-vkCreateEvent-events-04468
        if device.enabled_extensions().khr_portability_subset && !device.enabled_features().events {
            return Err(EventError::RequirementNotMet {
                required_for: "this device is a portability subset device, and `Event::new` was \
                    called",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("events")])]),
            });
        }

        let create_info = ash::vk::EventCreateInfo {
            flags: ash::vk::EventCreateFlags::empty(),
            ..Default::default()
        };

        let handle = unsafe {
            let mut output = MaybeUninit::uninit();
            let fns = device.fns();
            (fns.v1_0.create_event)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Event {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),
            must_put_in_pool: false,
        })
    }

    /// Takes an event from the vulkano-provided event pool.
    /// If the pool is empty, a new event will be allocated.
    /// Upon `drop`, the event is put back into the pool.
    ///
    /// For most applications, using the event pool should be preferred,
    /// in order to avoid creating new events every frame.
    #[inline]
    pub fn from_pool(device: Arc<Device>) -> Result<Event, EventError> {
        let handle = device.event_pool().lock().pop();
        let event = match handle {
            Some(handle) => {
                unsafe {
                    // Make sure the event isn't signaled
                    let fns = device.fns();
                    (fns.v1_0.reset_event)(device.handle(), handle)
                        .result()
                        .map_err(VulkanError::from)?;
                }
                Event {
                    handle,
                    device: InstanceOwnedDebugWrapper(device),
                    id: Self::next_id(),
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

    /// Creates a new `Event` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Event,
        _create_info: EventCreateInfo,
    ) -> Event {
        Event {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),
            must_put_in_pool: false,
        }
    }

    /// Returns true if the event is signaled.
    #[inline]
    pub fn signaled(&self) -> Result<bool, OomError> {
        unsafe {
            let fns = self.device.fns();
            let result = (fns.v1_0.get_event_status)(self.device.handle(), self.handle);
            match result {
                ash::vk::Result::EVENT_SET => Ok(true),
                ash::vk::Result::EVENT_RESET => Ok(false),
                err => Err(VulkanError::from(err).into()),
            }
        }
    }

    /// See the docs of set().
    #[inline]
    pub fn set_raw(&mut self) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.set_event)(self.device.handle(), self.handle)
                .result()
                .map_err(VulkanError::from)?;
            Ok(())
        }
    }

    /// Changes the `Event` to the signaled state.
    ///
    /// If a command buffer is waiting on this event, it is then unblocked.
    ///
    /// # Panics
    ///
    /// - Panics if the device or host ran out of memory.
    #[inline]
    pub fn set(&mut self) {
        self.set_raw().unwrap();
    }

    /// See the docs of reset().
    #[inline]
    pub fn reset_raw(&mut self) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.reset_event)(self.device.handle(), self.handle)
                .result()
                .map_err(VulkanError::from)?;
            Ok(())
        }
    }

    /// Changes the `Event` to the unsignaled state.
    ///
    /// # Panics
    ///
    /// - Panics if the device or host ran out of memory.
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
                self.device.event_pool().lock().push(raw_event);
            } else {
                let fns = self.device.fns();
                (fns.v1_0.destroy_event)(self.device.handle(), self.handle, ptr::null());
            }
        }
    }
}

unsafe impl VulkanObject for Event {
    type Handle = ash::vk::Event;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Event {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Event);

/// Parameters to create a new `Event`.
#[derive(Clone, Debug)]
pub struct EventCreateInfo {
    pub _ne: crate::NonExhaustive,
}

impl Default for EventCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EventError {
    /// Not enough memory available.
    OomError(OomError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for EventError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for EventError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
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

impl From<VulkanError> for EventError {
    fn from(err: VulkanError) -> Self {
        match err {
            e @ VulkanError::OutOfHostMemory | e @ VulkanError::OutOfDeviceMemory => {
                Self::OomError(e.into())
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{sync::event::Event, VulkanObject};

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

        assert_eq!(device.event_pool().lock().len(), 0);
        let event1_internal_obj = {
            let event = Event::from_pool(device.clone()).unwrap();
            assert_eq!(device.event_pool().lock().len(), 0);
            event.handle()
        };

        assert_eq!(device.event_pool().lock().len(), 1);
        let event2 = Event::from_pool(device.clone()).unwrap();
        assert_eq!(device.event_pool().lock().len(), 0);
        assert_eq!(event2.handle(), event1_internal_obj);
    }
}
