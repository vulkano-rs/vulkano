// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::ImageViewDimensions;
use vulkano::image::ImmutableImage;
use vulkano::image::MipmapsCount;
use vulkano::instance;
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};

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
    let mut layers = instance::layers_list().unwrap();
    while let Some(l) = layers.next() {
        println!("\t{}", l.name());
    }

    // NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers list:
    #[cfg(not(target_os = "macos"))]
    let layers = vec!["VK_LAYER_LUNARG_standard_validation"];

    #[cfg(target_os = "macos")]
    let layers = vec!["VK_LAYER_KHRONOS_validation"];

    // Important: pass the extension(s) and layer(s) when creating the vulkano instance
    let instance =
        Instance::new(None, &extensions, layers).expect("failed to create Vulkan instance");

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

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");
    let queue_family = physical
        .queue_families()
        .next()
        .expect("couldn't find a queue family");
    let (_, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions::required_extensions(physical),
        vec![(queue_family, 0.5)],
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    // Create an image in order to generate some additional logging:
    let pixel_format = Format::R8G8B8A8Uint;
    let dimensions = ImageViewDimensions::Dim2d {
        width: 4096,
        height: 4096,
    };
    const DATA: [[u8; 4]; 4096 * 4096] = [[0; 4]; 4096 * 4096];
    let _ = ImmutableImage::from_iter(
        DATA.iter().cloned(),
        dimensions,
        MipmapsCount::One,
        pixel_format,
        queue.clone(),
    )
    .unwrap();

    // (At this point you should see a bunch of messages printed to the terminal window - have fun debugging!)
}
