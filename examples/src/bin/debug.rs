// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{ImageDimensions, ImmutableImage, MipmapsCount},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, Instance, InstanceCreateInfo, InstanceExtensions, VulkanLibrary,
    },
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
    // this example. First, enable debugging using this extension: VK_EXT_debug_utils
    let extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..InstanceExtensions::none()
    };

    // You also need to specify (unless you've used the methods linked above) which debugging layers
    // your code should use. Each layer is a bunch of checks or messages that provide information of
    // some sort.
    //
    // The main layer you might want is: VK_LAYER_LUNARG_standard_validation
    // This includes a number of the other layers for you and is quite detailed.
    //
    // Additional layers can be installed (gpu vendor provided, something you found on GitHub, etc)
    // and you should verify that list for safety - Vulkano will return an error if you specify
    // any layers that are not installed on this system. That code to do could look like this:
    println!("List of Vulkan debugging layers available to use:");
    let lib = VulkanLibrary::default();
    let mut layers = layers_list(&lib).unwrap();
    while let Some(l) = layers.next() {
        println!("\t{}", l.name());
    }

    // NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers list:
    #[cfg(not(target_os = "macos"))]
    let layers = vec!["VK_LAYER_LUNARG_standard_validation".to_owned()];

    #[cfg(target_os = "macos")]
    let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];

    // Important: pass the extension(s) and layer(s) when creating the vulkano instance
    let instance = Instance::new(
        lib,
        InstanceCreateInfo {
            enabled_extensions: extensions,
            enabled_layers: layers,
            ..Default::default()
        },
    )
    .expect("failed to create Vulkan instance");

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // After creating the instance we must register the debugging callback.                                      //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Note: If you let this debug_callback binding fall out of scope then the callback will stop providing events
    // Note: There is a helper method too: DebugCallback::errors_and_warnings(&instance, |msg| {...

    let severity = MessageSeverity {
        error: true,
        warning: true,
        information: true,
        verbose: true,
    };

    let ty = MessageType::all();

    let _debug_callback = DebugCallback::new(&instance, severity, ty, |msg| {
        let severity = if msg.severity.error {
            "error"
        } else if msg.severity.warning {
            "warning"
        } else if msg.severity.information {
            "information"
        } else if msg.severity.verbose {
            "verbose"
        } else {
            panic!("no-impl");
        };

        let ty = if msg.ty.general {
            "general"
        } else if msg.ty.validation {
            "validation"
        } else if msg.ty.performance {
            "performance"
        } else {
            panic!("no-impl");
        };

        println!(
            "{} {} {}: {}",
            msg.layer_prefix.unwrap_or("unknown"),
            ty,
            severity,
            msg.description
        );
    })
    .ok();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create Vulkan objects in the same way as the other examples                                               //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            Some(
                p.queue_families()
                    .next()
                    .map(|q| (p, q))
                    .expect("couldn't find a queue family"),
            )
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");

    let (_, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    // Create an image in order to generate some additional logging:
    let pixel_format = Format::R8G8B8A8_UINT;
    let dimensions = ImageDimensions::Dim2d {
        width: 4096,
        height: 4096,
        array_layers: 1,
    };
    static DATA: [[u8; 4]; 4096 * 4096] = [[0; 4]; 4096 * 4096];
    let _ = ImmutableImage::from_iter(
        DATA.iter().copied(),
        dimensions,
        MipmapsCount::One,
        pixel_format,
        queue.clone(),
    )
    .unwrap();

    // (At this point you should see a bunch of messages printed to the terminal window - have fun debugging!)
}
