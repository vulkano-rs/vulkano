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
        use instance;

        match instance::Instance::new(None, &instance::InstanceExtensions::none(), None) {
            Ok(i) => i,
            Err(_) => return,
        }
    }};
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

        let (device, mut queues) = match Device::new(physical, &features,
                                                     &extensions, [(queue, 0.5)].iter().cloned())
        {
            Ok(r) => r,
            Err(_) => return
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
