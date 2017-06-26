// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate vulkano;

use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::ImmutableImage;
use vulkano::image::Dimensions;
use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::instance::debug::{DebugCallback, MessageTypes};

fn main() {
    // Vulkano Debugging Example Code
    //
    // This example code will demonstrate using the debug functions of the Vulkano api.
    //
    // There is documentation here about how to setup debugging:
    // https://vulkan.lunarg.com/doc/view/1.0.13.0/windows/layers.html
    //
    // .. but if you just want a template of code that has everything ready to go then follow
    // this example. First, enable debugging using this extension: VK_EXT_debug_report
    let extensions = InstanceExtensions {
        ext_debug_report: true,
        ..InstanceExtensions::none()
    };

    // You also need to specify (unless you've used the methods linked above) which debugging layers
    // your code should use. Each layer is a bunch of checks or messages that provide information of
    // some sort.
    //
    // The main layer you might want is: VK_LAYER_LUNARG_standard_validation
    // This includes a number of the other layers for you and is quite detailed.
    //
    // Additional layers can be installed (gpu vendor provided, something you found on github, etc)
    // and you should verify that list for safety - Volkano will return an error if you specify
    // any layers that are not installed on this system. That code to do could look like this:
    println!("List of Vulkan debugging layers available to use:");
    let mut layers = instance::layers_list().unwrap();
    while let Some(l) = layers.next() {
        println!("\t{}", l.name());
    }

    // NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers list:
    let layer = "VK_LAYER_LUNARG_standard_validation";
    let layers = vec![&layer];

    // Important: pass the extension(s) and layer(s) when creating the vulkano instance
    let instance = Instance::new(None, &extensions, layers).expect("failed to create Vulkan instance");

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // After creating the instance we must register the debugging callback.                                      //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Note: If you let this debug_callback binding fall out of scope then the callback will stop providing events
    // Note: There is a helper method too: DebugCallback::errors_and_warnings(&instance, |msg| {...

    let all = MessageTypes {
        error: true,
        warning: true,
        performance_warning: true,
        information: true,
        debug: true,
    };

    let _debug_callback = DebugCallback::new(&instance, all, |msg| {
        let ty = if msg.ty.error {
            "error"
        } else if msg.ty.warning {
            "warning"
        } else if msg.ty.performance_warning {
            "performance_warning"
        } else if msg.ty.information {
            "information"
        } else if msg.ty.debug {
            "debug"
        } else {
            panic!("no-impl");
        };
        println!("{} {}: {}", msg.layer_prefix, ty, msg.description);
    }).ok();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create vulkan objects in the same way as the other examples                                               //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    let queue = physical.queue_families().next().expect("couldn't find a queue family");
    let (device, mut queues) = Device::new(&physical, physical.supported_features(), &DeviceExtensions::none(), vec![(queue, 0.5)]).expect("failed to create device");
    let queue = queues.next().unwrap();

    // Create an image in order to generate some additional logging:
    let pixel_format = Format::R8G8B8A8Uint;
    let dimensions = Dimensions::Dim2d { width: 4096, height: 4096 };
    ImmutableImage::new(device.clone(), dimensions, pixel_format, Some(queue.family())).unwrap();

    // (At this point you should see a bunch of messages printed to the terminal window - have fun debugging!)
}
