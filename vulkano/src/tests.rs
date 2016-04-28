// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![cfg(test)]

/// Creates an instance or returns if initialization fails.
macro_rules! instance {
    () => ({
        use instance;

        let app = instance::ApplicationInfo {
            application_name: "vulkano tests", application_version: 1,
            engine_name: "vulkano tests", engine_version: 1
        };

        match instance::Instance::new(Some(&app), &instance::InstanceExtensions::none(), None) {
            Ok(i) => i,
            Err(_) => return
        }
    })
}

/// Creates a device and a queue for graphics operations.
macro_rules! gfx_dev_and_queue {
    ($($feature:ident),*) => ({
        use instance;
        use device::Device;
        use device::DeviceExtensions;
        use features::Features;

        let instance = instance!();

        let physical = match instance::PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return
        };

        let queue = match physical.queue_families().find(|q| q.supports_graphics()) {
            Some(q) => q,
            None => return
        };

        let extensions = DeviceExtensions::none();

        let features = Features {
            $(
                $feature: true,
            )*
            .. Features::none()
        };

        // If the physical device doesn't support the requested features, just return.
        if !physical.supported_features().superset_of(&features) {
            return;
        }

        let (device, queues) = match Device::new(&physical, &features,
                                                 &extensions, None, [(queue, 0.5)].iter().cloned())
        {
            Ok(r) => r,
            Err(_) => return
        };

        (device, queues.into_iter().next().unwrap())
    });
}
