use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    VulkanLibrary,
};

fn main() {
    let library = VulkanLibrary::new().unwrap();

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            // Enable enumerating devices that use non-conformant Vulkan implementations.
            // (e.g. MoltenVK)
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_video_queue: true,
        khr_video_decode_queue: true,
        khr_video_decode_h264: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, graphics_queue_family_index, video_queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            let graphic_pos = p
                .queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::GRAPHICS));
            let video_pos = p
                .queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::VIDEO_DECODE));
            graphic_pos
                .zip(video_pos)
                .map(|(g, v)| (p, g as u32, v as u32))
        })
        .next()
        .expect("no suitable physical device found");

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        Arc::clone(&physical_device),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: if video_queue_family_index == graphics_queue_family_index {
                vec![QueueCreateInfo {
                    queue_family_index: graphics_queue_family_index,
                    ..Default::default()
                }]
            } else {
                vec![
                    QueueCreateInfo {
                        queue_family_index: graphics_queue_family_index,
                        ..Default::default()
                    },
                    QueueCreateInfo {
                        queue_family_index: video_queue_family_index,
                        ..Default::default()
                    },
                ]
            },

            ..Default::default()
        },
    )
    .unwrap();

    let (graphics_queue, video_queue) = if graphics_queue_family_index == video_queue_family_index {
        let queue = queues.next().unwrap();
        (Arc::clone(&queue), Arc::clone(&queue))
    } else {
        (queues.next().unwrap(), queues.next().unwrap())
    };

    let video_properties = physical_device.queue_family_properties()
        [video_queue_family_index as usize]
        .video_properties
        .as_ref()
        .unwrap();

    println!(
        "Video queue supports the following codecs: {:?}",
        video_properties.video_codec_operations
    );
}
