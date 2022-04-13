// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![cfg(test)]

/// Creates an instance or returns if initialization fails.
macro_rules! instance {
    () => {{
        use crate::instance::Instance;
        let entry = Instance::entry();

        match Instance::new(entry, Default::default()) {
            Ok(i) => i,
            Err(_) => return,
        }
    }};
}

/// Creates a device and a queue for graphics operations.
macro_rules! gfx_dev_and_queue {
    ($($feature:ident),*) => ({
        use crate::device::physical::{PhysicalDevice, PhysicalDeviceType};
        use crate::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
        use crate::device::Features;

        let instance = instance!();
        let enabled_extensions = DeviceExtensions::none();
        let enabled_features = Features {
            $(
                $feature: true,
            )*
            .. Features::none()
        };

        let select = PhysicalDevice::enumerate(&instance)
            .filter(|&p| {
                p.supported_extensions().is_superset_of(&enabled_extensions) &&
                p.supported_features().is_superset_of(&enabled_features)
            })
            .filter_map(|p| {
                p.queue_families()
                    .find(|&q| q.supports_graphics())
                    .map(|q| (p, q))
            })
            .min_by_key(|(p, _)| {
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                }
            });

        let (physical_device, queue_family) = match select {
            Some(x) => x,
            None => return,
        };

        let (device, mut queues) = match Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                enabled_extensions,
                enabled_features,
                ..Default::default()
            }
        ) {
            Ok(r) => r,
            Err(_) => return,
        };

        (device, queues.next().unwrap())
    });
}

macro_rules! assert_should_panic {
    ($msg:expr, $code:block) => {{
        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $code));

        match res {
            Ok(_) => panic!("Test expected to panic but didn't"),
            Err(err) => {
                if let Some(msg) = err.downcast_ref::<String>() {
                    assert!(msg.contains($msg));
                } else if let Some(&msg) = err.downcast_ref::<&str>() {
                    assert!(msg.contains($msg));
                } else {
                    panic!("Couldn't decipher the panic message of the test")
                }
            }
        }
    }};

    ($code:block) => {{
        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $code));

        match res {
            Ok(_) => panic!("Test expected to panic but didn't"),
            Err(_) => {}
        }
    }};
}
