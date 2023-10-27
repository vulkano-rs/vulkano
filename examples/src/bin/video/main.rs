use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags},
    image::ImageUsage,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::{DeviceMemory, MemoryAllocateFlags, MemoryAllocateInfo},
    query::{QueryPool, QueryPoolCreateInfo, QueryType},
    video::{
        BindVideoSessionMemoryInfo, CodecCapabilities, VideoDecodeCapabilityFlags,
        VideoDecodeH264PictureLayoutFlags, VideoDecodeH264ProfileInfo, VideoFormatInfo,
        VideoProfileInfo, VideoProfileListInfo, VideoSession, VideoSessionCreateInfo,
        VideoSessionMemoryRequirements, VideoSessionParameters, VideoSessionParametersCreateFlags,
        VideoSessionParametersCreateInfo,
    },
    VulkanLibrary,
};

fn find_suitable_memory(
    device: Arc<Device>,
    memory_requirements: VideoSessionMemoryRequirements,
) -> Option<u32> {
    let mem_props = device.physical_device().memory_properties();

    for (i, mem_type) in mem_props.memory_types.iter().enumerate() {
        /* The memory type must agree with the memory requirements */
        if memory_requirements.memory_requirements.memory_type_bits & (1 << i) == 0 {
            continue;
        }

        return Some(i as u32);
    }

    None
}

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

    let mut formats = physical_device
        .video_format_properties(video_format_info)
        .unwrap();

    println!("video formats: {:#?}", formats);

    let format = formats.pop().unwrap();

    let video_session_create_info = VideoSessionCreateInfo {
        queue_family_index: video_queue_family_index,
        video_profile: profile_info,
        picture_format: format.format,
        max_coded_extent: video_caps.max_coded_extent,
        reference_picture_format: format.format,
        max_dpb_slots: video_caps.max_dpb_slots,
        max_active_reference_pictures: video_caps.max_active_reference_pictures,
        std_header_version: video_caps.std_header_version,
        ..Default::default()
    };

    let video_session = VideoSession::new(Arc::clone(&device), video_session_create_info).unwrap();
    println!("video session: {:#?}", video_session);

    let video_session_mem_requirements = video_session.get_memory_requirements().unwrap();
    println!(
        "video session memory requirements: {:?}",
        video_session_mem_requirements
    );

    let mut mems: Vec<_> = video_session_mem_requirements
        .iter()
        .map(|mem_req| {
            let mem_idx = find_suitable_memory(Arc::clone(&device), *mem_req)
                .expect("no suitable memory found");
            DeviceMemory::allocate(
                Arc::clone(&device),
                MemoryAllocateInfo {
                    allocation_size: mem_req.memory_requirements.layout.size(),
                    memory_type_index: mem_idx,
                    dedicated_allocation: None,
                    export_handle_types: Default::default(),
                    flags: MemoryAllocateFlags::empty(),
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    let bind_video_session_memory_infos = video_session_mem_requirements
        .iter()
        .map(|mem_req| {
            BindVideoSessionMemoryInfo::new(
                mem_req.memory_bind_index,
                mems.pop().unwrap(),
                0,
                mem_req.memory_requirements.layout.size(),
            )
        })
        .collect();

    video_session
        .bind_video_session_memory(bind_video_session_memory_infos)
        .unwrap();

    let video_session_parameters_create_info = VideoSessionParametersCreateInfo::new(
        VideoSessionParametersCreateFlags::empty(), None, Arc::clone(&video_session), vulkano::video::VideoSessionParametersCreateInfoNext::VideoDecodeH264SessionParametersCreateInfo { max_std_sps_count: 0, max_std_pps_count: 0, parameter_add_info: Some(vulkano::video::h264::VideoDecodeH264SessionParametersAddInfo {
            std_sp_ss: vec![],
            std_pp_ss: vec![],
        }) }
    );

    let empty_session_parameters =
        VideoSessionParameters::new(Arc::clone(&device), video_session_parameters_create_info)
            .unwrap();
    println!("empty session parameters: {:#?}", empty_session_parameters);

    // A 64x64 progressive byte-stream encoded I-frame.
    // Encoded with the following GStreamer pipeline:
    //
    // gst-launch-1.0 videotestsrc num-buffers=1 ! video/x-raw,format=I420,width=64,height=64 ! x264enc ! video/x-h264,profile=constrained-baseline,stream-format=byte-stream ! filesink location="64x64-I.h264"
    let h264_stream = include_bytes!("64x64-I.h264");
    println!("loaded {} bytes of h264 data", h264_stream.len());

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let command_buffer = command_buffer_allocator
        .allocate(
            video_queue_family_index,
            vulkano::command_buffer::CommandBufferLevel::Primary,
            1,
        )
        .unwrap();

    let mut query_pool_create_info = QueryPoolCreateInfo::query_type(QueryType::ResultStatusOnly);
    query_pool_create_info.query_count = 1;
    let query_pool = QueryPool::new(Arc::clone(&device), query_pool_create_info).unwrap();
}
