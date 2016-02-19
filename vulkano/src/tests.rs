#![cfg(test)]

/// Creates an instance or returns if initialization fails.
macro_rules! instance {
    () => ({
        use instance;

        let app = instance::ApplicationInfo {
            application_name: "vulkano tests", application_version: 1,
            engine_name: "vulkano tests", engine_version: 1
        };

        match instance::Instance::new(Some(&app), None) {
            Ok(i) => i,
            Err(_) => return
        }
    })
}

/// Creates a device and a queue for graphics operations.
macro_rules! gfx_dev_and_queue {
    () => ({
        use instance;
        use device::Device;

        let instance = instance!();

        let physical = instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");

        let queue = physical.queue_families().find(|q| q.supports_graphics())
                                                .expect("couldn't find a graphical queue family");

        let (device, queues) = Device::new(&physical, physical.supported_features(),
                                                        [(queue, 0.5)].iter().cloned(), None)
                                                                .expect("failed to create device");
        (device, queues.into_iter().next().unwrap())
    })
}
