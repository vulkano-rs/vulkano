use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
    },
    VulkanLibrary,
};

fn main() {
    // Vulkano Debugging Example Code
    //
    // This example code will demonstrate using the debug functions of the Vulkano API.
    //
    // There is documentation here about how to setup debugging:
    // https://vulkan.lunarg.com/doc/view/1.0.13.0/windows/layers.html
    //
    // .. but if you just want a template of code that has everything ready to go then follow
    // this example. First, enable debugging using this extension: `VK_EXT_debug_utils`.
    let extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..InstanceExtensions::empty()
    };

    let library = VulkanLibrary::new().unwrap();

    // You also need to specify (unless you've used the methods linked above) which debugging
    // layers your code should use. Each layer is a bunch of checks or messages that provide
    // information of some sort.
    //
    // The main layer you might want is `VK_LAYER_LUNARG_standard_validation`. This includes a
    // number of the other layers for you and is quite detailed.
    //
    // Additional layers can be installed (gpu vendor provided, something you found on GitHub, etc)
    // and you should verify that list for safety - Vulkano will return an error if you specify
    // any layers that are not installed on this system. That code to do could look like this:
    println!("List of Vulkan debugging layers available to use:");
    let layers = library.layer_properties().unwrap();
    for l in layers {
        println!("\t{}", l.name());
    }

    // NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers
    // list.
    let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];

    // Important: pass the extension(s) and layer(s) when creating the vulkano instance.
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_layers: layers,
            enabled_extensions: extensions,
            ..Default::default()
        },
    )
    .expect("failed to create Vulkan instance");

    // After creating the instance we must register the debug callback.
    //
    // NOTE: If you let this debug_callback binding fall out of scope then the callback will stop
    // providing events.
    let _debug_callback = unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity::ERROR
                    | DebugUtilsMessageSeverity::WARNING
                    | DebugUtilsMessageSeverity::INFO
                    | DebugUtilsMessageSeverity::VERBOSE,
                message_type: DebugUtilsMessageType::GENERAL
                    | DebugUtilsMessageType::VALIDATION
                    | DebugUtilsMessageType::PERFORMANCE,
                ..DebugUtilsMessengerCreateInfo::new(DebugUtilsMessengerCallback::new(
                    |message_severity, message_type, callback_data| {
                        let severity = if message_severity
                            .intersects(DebugUtilsMessageSeverity::ERROR)
                        {
                            "error"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                            "warning"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::INFO) {
                            "information"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                            "verbose"
                        } else {
                            panic!("no-impl");
                        };

                        let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL) {
                            "general"
                        } else if message_type.intersects(DebugUtilsMessageType::VALIDATION) {
                            "validation"
                        } else if message_type.intersects(DebugUtilsMessageType::PERFORMANCE) {
                            "performance"
                        } else {
                            panic!("no-impl");
                        };

                        println!(
                            "{} {} {}: {}",
                            callback_data.message_id_name.unwrap_or("unknown"),
                            ty,
                            severity,
                            callback_data.message
                        );
                    },
                ))
            },
        )
    }
    .ok();

    // Create Vulkan objects in the same way as the other examples.
    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .map(|p| {
            (!p.queue_family_properties().is_empty())
                .then_some((p, 0))
                .expect("couldn't find a queue family")
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no device available");

    let (_device, _queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("failed to create device");

    // (At this point you should see a bunch of messages printed to the terminal window -
    // have fun debugging!)
}
