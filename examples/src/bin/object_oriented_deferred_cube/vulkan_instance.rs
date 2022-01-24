use std::sync::Arc;

use log::{error, info, trace, warn};
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::Version;

/// Vulkan instance is the entry point to everything Vulkan.
/// It's not tied to anything AFAIK - it's still valid even after a Nvidia driver force restart on Windows 10.
/// We'll also create the debug callback here because debugging requires extra extensions and layers.
pub fn new<'a>(
    version: Version,
    debug: bool,
    extra_instance_extensions: InstanceExtensions,
    extra_layers: impl IntoIterator<Item = &'a str>,
) -> anyhow::Result<(Arc<Instance>, Option<DebugCallback>)> {
    // Instance extensions are something the average developer probably won't need to care about...
    // But you can still learn the list here: https://vulkan.gpuinfo.org/listinstanceextensions.php
    let instance_extensions = extra_instance_extensions.union(&InstanceExtensions {
        ext_debug_utils: debug, // Required by the debugging layers
        ..vulkano_win::required_extensions()
    });

    // Layers are external programs that can hijack your Vulkan calls to add functionalities.
    // Common use cases are debugging, screen capturing and in-game overlays.
    // Some layers need explicit enabling,
    // while others are automatically enabled (sounds like security issues and source of bugs...)
    // Read more at https://renderdoc.org/vulkan-layer-guide.html
    let layers = extra_layers.into_iter().chain(
        // TODO Please replace with the actual layers available with your Vulkan SDK
        // (can be listed with Sascha Willems' Capability Viewer https://vulkan.gpuinfo.org/download.php)
        // or just remove them all and use vkconfig to enable them.
        // I just hard coded these layers... which have a history of changing their names.
        if debug {
            vec![
                "VK_LAYER_KHRONOS_validation",       // The main validation layer
                "VK_LAYER_KHRONOS_synchronization2", // I don't know if this is actually useful but it won't hurt to include
                "VK_LAYER_LUNARG_monitor",           // Convenient FPS display in title bar
            ]
        } else {
            vec![]
        },
    );

    // Finally create the instance.
    // For simplicity we'll not pass an ApplicationInfo here, which is only some debug information for the end user.
    // TODO You probably want to pass in one when you have finished your app.
    let instance = Instance::new(None, version, &instance_extensions, layers)?;

    // Setup debug callback if desired.
    // The callback object needs to be NOT DROPPED in order to work.
    let mut debug_callback = None;
    if debug {
        let severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        };
        let ty = MessageType::all();
        debug_callback = Some(DebugCallback::new(&instance, severity, ty, |msg| {
            let ty = if msg.ty.general {
                "General"
            } else if msg.ty.validation {
                "Validation"
            } else if msg.ty.performance {
                "Performance"
            } else {
                "????"
            };

            let layer = msg.layer_prefix.unwrap_or("unknown");

            if msg.severity.error {
                error!("{} {}: {}", ty, layer, msg.description);
            } else if msg.severity.warning {
                warn!("{} {}: {}", ty, layer, msg.description);
            } else if msg.severity.information {
                info!("{} {}: {}", ty, layer, msg.description);
            } else if msg.severity.verbose {
                trace!("{} {}: {}", ty, layer, msg.description);
            } else {
                warn!("{} {}: {}", ty, layer, msg.description);
                warn!("Unknown Vulkan debug message severity");
            };
        })?);
    }

    Ok((instance, debug_callback))
}
