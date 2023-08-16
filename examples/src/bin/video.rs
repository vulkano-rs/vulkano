use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags},
    image::ImageUsage,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    video::{
        CodecCapabilities, VideoDecodeCapabilityFlags, VideoDecodeH264PictureLayoutFlags,
        VideoDecodeH264ProfileInfo, VideoFormatInfo, VideoProfileInfo, VideoProfileListInfo,
    },
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

    // Video profiles are provided as input to video capability queries such as
    // vkGetPhysicalDeviceVideoCapabilitiesKHR or
    // vkGetPhysicalDeviceVideoFormatPropertiesKHR, as well as when creating
    // resources to be used by video coding operations such as images, buffers,
    // query pools, and video sessions.
    //
    // You must parse the bitstream to correctly construct the profile info.
    // This is hardcoded for the bitstream in this example.
    let profile_info = VideoProfileInfo {
        video_codec_operation: vulkano::video::VideoCodecOperation::DecodeH264,
        chroma_subsampling: vulkano::video::VideoChromaSubsampling::Type420,
        luma_bit_depth: vulkano::video::VideoComponentBitDepth::Type8,
        chroma_bit_depth: Some(vulkano::video::VideoComponentBitDepth::Type8),
        codec_profile_info: vulkano::video::VideoDecodeProfileInfo::H264(
            VideoDecodeH264ProfileInfo {
                std_profile_idc: 0,
                picture_layout: VideoDecodeH264PictureLayoutFlags::PROGRESSIVE,
                ..Default::default()
            },
        ),
        ..Default::default()
    };

    let video_caps = physical_device
        .video_capabilities(profile_info.clone())
        .unwrap();
    println!("Video capabilities: {:#?}", video_caps);

    let CodecCapabilities::VideoDecode(video_decode_caps) = video_caps.codec_capabilities;

    let video_format_info = VideoFormatInfo {
        image_usage: if !video_decode_caps
            .flags
            .intersects(VideoDecodeCapabilityFlags::DPB_AND_OUTPUT_COINCIDE)
        {
            ImageUsage::VIDEO_DECODE_DPB
        } else {
            ImageUsage::VIDEO_DECODE_DPB
                | ImageUsage::VIDEO_DECODE_DST
                | ImageUsage::TRANSFER_SRC
                | ImageUsage::SAMPLED
        },
        profile_list_info: VideoProfileListInfo {
            profiles: vec![profile_info.clone()],
            ..Default::default()
        },
    };

    let formats = physical_device
        .video_format_properties(video_format_info)
        .unwrap();

    println!("video formats: {:#?}", formats);
}
