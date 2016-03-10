#![cfg(test)]

/// Creates an instance or returns if initialization fails.
macro_rules! instance {
    () => ({
        use instance;

        let app = instance::ApplicationInfo {
            application_name: "vulkano tests", application_version: 1,
            engine_name: "vulkano tests", engine_version: 1
        };

        match instance::Instance::new(Some(&app), None, &instance::Extensions::none()) {
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
        use device::DeviceExtensions;

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

        let (device, queues) = match Device::new(&physical, physical.supported_features(),
                                                 [(queue, 0.5)].iter().cloned(), None, &extensions)
        {
            Ok(r) => r,
            Err(_) => return
        };

        (device, queues.into_iter().next().unwrap())
    })
}
