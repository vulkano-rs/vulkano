// Copyright (c) 2023 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{
    mem::MaybeUninit,
    sync::{Arc, Mutex},
};

use ash::vk::{
    native::{StdVideoH264LevelIdc, StdVideoH264ProfileIdc},
    DeviceSize,
};

use crate::{
    device::Device,
    format::Format,
    image::{sampler::ComponentMapping, ImageCreateFlags, ImageTiling, ImageType, ImageUsage},
    instance::InstanceOwnedDebugWrapper,
    macros::{vulkan_bitflags, vulkan_bitflags_enum},
    ExtensionProperties, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError,
    Version, VulkanError, VulkanObject,
};

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// The type of video coding operation and video compression standard used
    /// by a video profile
    VideoCodecOperations,

    VideoCodecOperation,

    = VideoCodecOperationFlagsKHR(u32);

    /// Specifies support for H.264 video decode operations
    DECODE_H264, DecodeH264 = DECODE_H264
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
        RequiresAllOf([DeviceExtension(khr_video_decode_h264)])]
    ),

    /// Specifies support for H.265 video decode operations
    DECODE_H265, DecodeH265 = DECODE_H265
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
        RequiresAllOf([DeviceExtension(khr_video_decode_h265)])]
    ),
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    VideoChromaSubsamplings,

    VideoChromaSubsampling,

    = VideoChromaSubsamplingFlagsKHR(u32);

    /// Specifies that the format is monochrome.
    MONOCHROME, Monochrome = MONOCHROME,
    /// Specified that the format is 4:2:0 chroma subsampled, i.e. the two
    /// chroma components are sampled horizontally and vertically at half the
    /// sample rate of the luma component.
    TYPE_420, Type420 = TYPE_420,
    /// The format is 4:2:2 chroma subsampled, i.e. the two chroma components
    /// are sampled horizontally at half the sample rate of luma component.
    TYPE_422, Type422 = TYPE_422,
    /// The format is 4:4:4 chroma sampled, i.e. all three components of the
    /// Y′CBCR format are sampled at the same rate, thus there is no chroma
    /// subsampling.
    TYPE_444, Type444 = TYPE_444,
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    VideoComponentBitDepths,

    VideoComponentBitDepth,

    = VideoComponentBitDepthFlagsKHR(u32);

    /// Specifies a component bit depth of 8 bits.
    TYPE_8, Type8 = TYPE_8,
    /// Specifies a component bit depth of 10 bits.
    TYPE_10, Type10 = TYPE_10,
    /// Specifies a component bit depth of 12 bits.
    TYPE_12, Type12 = TYPE_12,
}

vulkan_bitflags! {
    #[non_exhaustive]

    VideoCapabilityFlags = VideoCapabilityFlagsKHR(u32);

    PROTECTED_CONTENT = PROTECTED_CONTENT,
    SEPARATE_REFERENCE_IMAGES = SEPARATE_REFERENCE_IMAGES,
}

vulkan_bitflags! {
    #[non_exhaustive]

    VideoDecodeCapabilityFlags = VideoDecodeCapabilityFlagsKHR(u32);

    DPB_AND_OUTPUT_COINCIDE = DPB_AND_OUTPUT_COINCIDE,
    DPB_AND_OUTPUT_DISTINCT = DPB_AND_OUTPUT_DISTINCT,
}

vulkan_bitflags! {
    #[non_exhaustive]

    VideoDecodeH264PictureLayoutFlags = VideoDecodeH264PictureLayoutFlagsKHR(u32);

    PROGRESSIVE = PROGRESSIVE,
    INTERLACED_INTERLEAVED_LINES_BIT_KHR = INTERLACED_INTERLEAVED_LINES,
    INTERLACED_SEPARATE_PLANES_BIT_KHR = INTERLACED_SEPARATE_PLANES,
}

pub enum VideoDecodeProfileInfoNextVk {
    H264(ash::vk::VideoDecodeH264ProfileInfoKHR),
    H265(ash::vk::VideoDecodeH265ProfileInfoKHR),
}

pub enum VideoDecodeCapabilitiesNextVk {
    H264(ash::vk::VideoDecodeH264CapabilitiesKHR),
    H265(ash::vk::VideoDecodeH265CapabilitiesKHR),
}

#[derive(Clone, Debug)]
pub struct VideoProfileInfo {
    pub video_codec_operation: VideoCodecOperation,
    pub chroma_subsampling: VideoChromaSubsampling,
    pub luma_bit_depth: VideoComponentBitDepth,
    pub chroma_bit_depth: Option<VideoComponentBitDepth>,
    pub codec_profile_info: VideoDecodeProfileInfo,
    pub _ne: crate::NonExhaustive,
}

impl VideoProfileInfo {
    pub(crate) fn validate(&self) -> Result<(), Box<ValidationError>> {
        let &Self {
            chroma_subsampling,
            chroma_bit_depth,
            _ne,
            ..
        } = self;

        if !matches!(chroma_subsampling, VideoChromaSubsampling::Monochrome)
            && chroma_bit_depth.is_none()
        {
            return Err(Box::new(ValidationError {
                context: "chroma_bit_depth".into(),
                problem: "is `Invalid`".into(),
                vuids: &["VUID-VkVideoProfileInfoKHR-chromaSubsampling-07015"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    /// Safety: `ash::vk::VideoProfileInfoKHR` is only valid so long as the
    /// arguments to this function are not moved or dropped.
    pub(crate) unsafe fn to_vulkan(
        self,
        video_decode_profile_info_next_vk: &mut Option<VideoDecodeProfileInfoNextVk>,
    ) -> ash::vk::VideoProfileInfoKHR {
        let VideoProfileInfo {
            video_codec_operation,
            chroma_subsampling,
            luma_bit_depth,
            chroma_bit_depth,
            codec_profile_info,
            _ne: _,
        } = self;

        let mut video_profile_info_vk = ash::vk::VideoProfileInfoKHR {
            video_codec_operation: video_codec_operation.into(),
            chroma_subsampling: chroma_subsampling.into(),
            luma_bit_depth: luma_bit_depth.into(),
            chroma_bit_depth: if let Some(chroma_bit_depth) = chroma_bit_depth {
                chroma_bit_depth.into()
            } else {
                ash::vk::VideoComponentBitDepthFlagsKHR::INVALID
            },
            ..Default::default()
        };

        match video_codec_operation {
            VideoCodecOperation::DecodeH264 => {
                let video_decode_h264_profile_info = match codec_profile_info {
                    VideoDecodeProfileInfo::H264(p) => p,
                    _ => panic!("Wrong codec profile info type for H264"),
                };

                let video_decode_h264_profile_info = video_decode_profile_info_next_vk.insert(
                    VideoDecodeProfileInfoNextVk::H264(ash::vk::VideoDecodeH264ProfileInfoKHR {
                        std_profile_idc: video_decode_h264_profile_info.std_profile_idc,
                        picture_layout: video_decode_h264_profile_info.picture_layout.into(),
                        ..Default::default()
                    }),
                );

                let video_decode_h264_profile_info = match video_decode_h264_profile_info {
                    VideoDecodeProfileInfoNextVk::H264(v) => v,
                    VideoDecodeProfileInfoNextVk::H265(_) => unreachable!(),
                };

                // VUID-VkVideoProfileInfoKHR-videoCodecOperation-07179
                video_profile_info_vk.p_next =
                    video_decode_h264_profile_info as *const _ as *const _;
            }
            VideoCodecOperation::DecodeH265 => todo!(),
        }

        video_profile_info_vk
    }

    pub(crate) unsafe fn to_vulkan_video_capabilities(
        &self,
        video_decode_capabilities_vk: &mut Option<ash::vk::VideoDecodeCapabilitiesKHR>,
        video_decode_capabilities_next_vk: &mut Option<VideoDecodeCapabilitiesNextVk>,
    ) -> ash::vk::VideoCapabilitiesKHR {
        let mut video_capabilities_vk = ash::vk::VideoCapabilitiesKHR::default();

        let VideoProfileInfo {
            video_codec_operation,
            ..
        } = self;

        let specifies_decode_operation = match video_codec_operation {
            VideoCodecOperation::DecodeH264 | VideoCodecOperation::DecodeH265 => true,
        };

        if specifies_decode_operation {
            let video_decode_capabilities_vk =
                video_decode_capabilities_vk.insert(ash::vk::VideoDecodeCapabilitiesKHR::default());

            video_capabilities_vk.p_next = video_decode_capabilities_vk as *mut _ as *mut _;
        }

        match video_codec_operation {
            VideoCodecOperation::DecodeH264 => {
                let video_decode_h264_capabilities_vk =
                    video_decode_capabilities_next_vk.insert(VideoDecodeCapabilitiesNextVk::H264(
                        ash::vk::VideoDecodeH264CapabilitiesKHR::default(),
                    ));

                let video_decode_h264_capabilities_vk = match video_decode_h264_capabilities_vk {
                    VideoDecodeCapabilitiesNextVk::H264(v) => v,
                    _ => unreachable!(),
                };

                let video_decode_capabilities_vk = video_decode_capabilities_vk.as_mut().unwrap();

                video_decode_capabilities_vk.p_next =
                    video_decode_h264_capabilities_vk as *mut _ as *mut _;
            }
            VideoCodecOperation::DecodeH265 => todo!(),
        }

        video_capabilities_vk
    }
}

impl Default for VideoProfileInfo {
    fn default() -> Self {
        Self {
            video_codec_operation: VideoCodecOperation::DecodeH264,
            chroma_subsampling: VideoChromaSubsampling::Monochrome,
            luma_bit_depth: VideoComponentBitDepth::Type8,
            chroma_bit_depth: Some(VideoComponentBitDepth::Type8),
            codec_profile_info: VideoDecodeProfileInfo::H264(VideoDecodeH264ProfileInfo {
                std_profile_idc: 66,
                picture_layout: VideoDecodeH264PictureLayoutFlags::PROGRESSIVE,
                _ne: crate::NonExhaustive(()),
            }),
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VideoDecodeH264ProfileInfo {
    pub std_profile_idc: StdVideoH264ProfileIdc,
    pub picture_layout: VideoDecodeH264PictureLayoutFlags,
    pub _ne: crate::NonExhaustive,
}

impl Default for VideoDecodeH264ProfileInfo {
    fn default() -> Self {
        Self {
            std_profile_idc: Default::default(),
            picture_layout: Default::default(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[derive(Clone, Debug)]
pub enum VideoDecodeProfileInfo {
    H264(VideoDecodeH264ProfileInfo),
    H265, /* TODO */
}

#[derive(Clone, Debug)]
pub struct VideoCapabilities {
    pub flags: VideoCapabilityFlags,
    pub min_bitstream_buffer_offset_alignment: DeviceSize,
    pub min_bitstream_buffer_size_alignment: DeviceSize,
    pub picture_access_granularity: [u32; 2],
    pub min_coded_extent: [u32; 2],
    pub max_coded_extent: [u32; 2],
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub std_header_version: ExtensionProperties,
    pub codec_capabilities: CodecCapabilities,
    pub _ne: crate::NonExhaustive,
}

#[derive(Clone, Debug)]
pub enum CodecCapabilities {
    VideoDecode(VideoDecodeCapabilities),
}

#[derive(Clone, Debug)]
pub struct VideoDecodeCapabilities {
    pub flags: VideoDecodeCapabilityFlags,
    pub codec_capabilities: VideoDecodeCodecCapabilities,
    pub _ne: crate::NonExhaustive,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum VideoDecodeCodecCapabilities {
    H264(VideoDecodeH264Capabilities),
    /* todo */
}

#[derive(Clone, Debug)]
pub struct VideoDecodeH264Capabilities {
    pub max_level_idc: StdVideoH264LevelIdc,
    pub field_offset_granularity: [i32; 2],
    pub _ne: crate::NonExhaustive,
}

#[derive(Clone, Debug)]
pub struct VideoProfileListInfo {
    pub profiles: Vec<VideoProfileInfo>,
    pub _ne: crate::NonExhaustive,
}

impl VideoProfileListInfo {
    /// Safety: The type returned by this function is only valid so long as the
    /// arguments to this function are not moved or dropped.
    pub(crate) unsafe fn to_vulkan(
        &self,
        video_profile_info_vk: &mut Vec<ash::vk::VideoProfileInfoKHR>,
        video_decode_profile_info_vk: &mut Option<VideoDecodeProfileInfoNextVk>,
    ) -> ash::vk::VideoProfileListInfoKHR {
        *video_profile_info_vk = self
            .profiles
            .iter()
            .cloned()
            .map(|video_profile_info| video_profile_info.to_vulkan(video_decode_profile_info_vk))
            .collect::<Vec<_>>();

        ash::vk::VideoProfileListInfoKHR {
            profile_count: video_profile_info_vk.len() as _,
            p_profiles: video_profile_info_vk.as_ptr() as _,
            ..Default::default()
        }
    }
}

impl Default for VideoProfileListInfo {
    fn default() -> Self {
        Self {
            profiles: Default::default(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VideoFormatInfo {
    pub image_usage: ImageUsage,
    pub profile_list_info: VideoProfileListInfo,
}

impl VideoFormatInfo {
    /// Safety: The type returned by this function is only valid so long as the
    /// arguments to this function are not moved or dropped.
    pub(crate) unsafe fn to_vulkan(
        &self,
        video_profile_list_info_vk: &mut Option<ash::vk::VideoProfileListInfoKHR>,
        video_profile_info_vk: &mut Vec<ash::vk::VideoProfileInfoKHR>,
        video_decode_profile_info_vk: &mut Option<VideoDecodeProfileInfoNextVk>,
    ) -> ash::vk::PhysicalDeviceVideoFormatInfoKHR {
        let video_profile_list_info_vk = video_profile_list_info_vk.insert(
            self.profile_list_info
                .to_vulkan(video_profile_info_vk, video_decode_profile_info_vk),
        );

        ash::vk::PhysicalDeviceVideoFormatInfoKHR {
            p_next: video_profile_list_info_vk as *const _ as _,
            image_usage: self.image_usage.into(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct VideoFormatProperties {
    pub format: Format,
    pub component_mapping: ComponentMapping,
    pub image_create_flags: ImageCreateFlags,
    pub image_type: ImageType,
    pub image_tiling: ImageTiling,
    pub image_usage_flags: ImageUsage,
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    VideoSessionCreateFlags,

    VideoSessionCreateFlag,

    = VideoSessionCreateFlagsKHR(u32);

    PROTECTED_CONTENT, ProtectedContent = PROTECTED_CONTENT,
}

#[derive(Debug)]
pub struct VideoSession {
    handle: ash::vk::VideoSessionKHR,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    /// The `VideoSessionCreateInfo` that created `self`.
    create_info: VideoSessionCreateInfo,
}

impl VideoSession {
    pub fn new(
        device: Arc<Device>,
        create_info: VideoSessionCreateInfo,
    ) -> Result<Arc<VideoSession>, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &VideoSessionCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !device
            .physical_device()
            .supported_extensions()
            .khr_video_queue
            || device.physical_device().api_version() < Version::V1_3
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                    Requires::DeviceExtension("khr_video_queue"),
                    // Requires::APIVersion(Version::V1_3), // ?
                ])]),
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: VideoSessionCreateInfo,
    ) -> Result<Arc<VideoSession>, VulkanError> {
        let &VideoSessionCreateInfo {
            queue_family_index,
            flags,
            ref video_profile,
            picture_format,
            max_coded_extent,
            reference_picture_format,
            max_dpb_slots,
            max_active_reference_pictures,
            std_header_version:
                ExtensionProperties {
                    ref extension_name,
                    spec_version,
                },
            ..
        } = &create_info;
        let mut video_decode_h264_profile_info_vk = None;
        let video_profile_vk = video_profile
            .clone()
            .to_vulkan(&mut video_decode_h264_profile_info_vk);

        let extension_properties_vk = ash::vk::ExtensionProperties {
            extension_name: {
                let c_str = std::ffi::CString::new(extension_name.clone()).unwrap();
                let mut bytes = [0; 256];
                bytes[0..c_str.as_bytes_with_nul().len()]
                    .copy_from_slice(c_str.as_bytes_with_nul());
                bytes.map(|b| b as _)
            },
            spec_version,
        };

        let create_info_vk = ash::vk::VideoSessionCreateInfoKHR {
            queue_family_index,
            flags: flags.into(),
            p_video_profile: &video_profile_vk,
            picture_format: picture_format.into(),
            max_coded_extent: ash::vk::Extent2D {
                width: max_coded_extent[0],
                height: max_coded_extent[1],
            },
            reference_picture_format: reference_picture_format.into(),
            max_dpb_slots,
            max_active_reference_pictures,
            p_std_header_version: &extension_properties_vk,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_video_queue.create_video_session_khr)(
                device.handle(),
                &create_info_vk,
                std::ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(handle, create_info, device))
    }

    /// Creates a new `VideoSession` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        handle: ash::vk::VideoSessionKHR,
        create_info: VideoSessionCreateInfo,
        device: Arc<Device>,
    ) -> Arc<Self> {
        Arc::new(VideoSession {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            create_info,
        })
    }
}

impl Drop for VideoSession {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.khr_video_queue.destroy_video_session_khr)(
                self.device.handle(),
                self.handle,
                std::ptr::null(),
            );
        }
    }
}

#[derive(Clone, Debug)]
pub struct VideoSessionCreateInfo {
    pub queue_family_index: u32,
    pub flags: VideoSessionCreateFlags,
    pub video_profile: VideoProfileInfo,
    pub picture_format: Format,
    pub max_coded_extent: [u32; 2],
    pub reference_picture_format: Format,
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub std_header_version: ExtensionProperties,
    pub _ne: crate::NonExhaustive,
}

impl VideoSessionCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            max_coded_extent,
            reference_picture_format,
            max_dpb_slots,
            max_active_reference_pictures,
            video_profile:
                VideoProfileInfo {
                    video_codec_operation,
                    ..
                },
            ..
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkSwapchainCreateInfoKHR-flags-parameter"])
        })?;

        let video_capabilities = device
            .physical_device()
            .video_capabilities(self.video_profile.clone())
            .map_err(|err| {
                Box::new(ValidationError::from_error(err))
                    .add_context("pVideoProfile")
                    .set_vuids(&["VUID-VkVideoSessionCreateInfoKHR-pVideoProfile-04845"])
            })?;

        if max_dpb_slots > video_capabilities.max_dpb_slots {
            return Err(Box::new(ValidationError {
                context: "max_dpb_slots".into(),
                problem:
                    "is greater than the maximum number of DPB slots supported by the video profile"
                        .into(),
                vuids: &["VUID-VkVideoSessionCreateInfoKHR-maxDPBSlots-04847"],
                ..Default::default()
            }));
        }

        if max_active_reference_pictures > video_capabilities.max_active_reference_pictures {
            return Err(Box::new(ValidationError {
                context: "max_active_reference_pictures".into(),
                problem: "is greater than the maximum number of active reference pictures supported by the video profile".into(),
                vuids: &["VUID-VkVideoSessionCreateInfoKHR-maxActiveReferencePictures-04849"],
                ..Default::default()
            }));
        }

        if (max_dpb_slots == 0 && max_active_reference_pictures != 0)
            || (max_active_reference_pictures == 0 && max_dpb_slots != 0)
        {
            return Err(Box::new(ValidationError {
                context: "max_dpb_slots".into(),
                problem: "is 0 while max_active_reference_pictures is not 0, or vice versa".into(),
                vuids: &["VUID-VkVideoSessionCreateInfoKHR-maxDpbSlots-04850"],
                ..Default::default()
            }));
        }

        if max_coded_extent < video_capabilities.min_coded_extent
            || max_coded_extent > video_capabilities.max_coded_extent
        {
            return Err(Box::new(ValidationError {
                context: "max_coded_extent".into(),
                problem: "is not within the range of supported coded extents".into(),
                vuids: &["VUID-VkVideoSessionCreateInfoKHR-maxCodedExtent-04851"],
                ..Default::default()
            }));
        }

        let is_decode_operation = match video_codec_operation {
            VideoCodecOperation::DecodeH264 | VideoCodecOperation::DecodeH265 => true,
        };

        if is_decode_operation && max_active_reference_pictures > 0 {
            let video_format_info = VideoFormatInfo {
                image_usage: ImageUsage::VIDEO_DECODE_DPB,
                profile_list_info: VideoProfileListInfo {
                    profiles: vec![self.video_profile.clone()],
                    ..Default::default()
                },
            };

            let formats = device
                .physical_device()
                .video_format_properties(video_format_info)
                .map_err(|err| {
                    Box::new(ValidationError::from_error(err)).add_context("pVideoProfile")
                })?;

            if !formats
                .into_iter()
                .any(|f| f.format == reference_picture_format)
            {
                return Err(Box::new(ValidationError {
                    context: "referencePictureFormat".into(),
                    problem: "must be one of the supported decode DPB formats".into(),
                    vuids: &["VUID-VkVideoSessionCreateInfoKHR-referencePictureFormat-04852"],
                    ..Default::default()
                }));
            }
        }

        /* encode: */
        /* VUID-VkVideoSessionCreateInfoKHR-referencePictureFormat-06814 */

        if is_decode_operation {
            let video_format_info = VideoFormatInfo {
                image_usage: ImageUsage::VIDEO_DECODE_DST,
                profile_list_info: VideoProfileListInfo {
                    profiles: vec![self.video_profile.clone()],
                    ..Default::default()
                },
            };

            let formats = device
                .physical_device()
                .video_format_properties(video_format_info)
                .map_err(|err| {
                    Box::new(ValidationError::from_error(err)).add_context("pVideoProfile")
                })?;

            if !formats
                .into_iter()
                .any(|f| f.format == reference_picture_format)
            {
                return Err(Box::new(ValidationError {
                    context: "pictureFormat".into(),
                    problem: "must be one of the supported decode output formats".into(),
                    vuids: &["VUID-VkVideoSessionCreateInfoKHR-pictureFormat-04853"],
                    ..Default::default()
                }));
            }
        }

        /* encode: */
        /* VUID-VkVideoSessionCreateInfoKHR-pictureFormat-04854 */

        if self.std_header_version.extension_name
            != video_capabilities.std_header_version.extension_name
        {
            return Err(Box::new(ValidationError {
                context: "std_header_version.extensionName".into(),
                problem: "does not match video_capabilities.std_header_version.extension_name"
                    .into(),
                vuids: &["VUID-VkVideoSessionCreateInfoKHR-pStdHeaderVersion-07190"],
                ..Default::default()
            }));
        }

        if self.std_header_version.spec_version > video_capabilities.std_header_version.spec_version
        {
            return Err(Box::new(ValidationError {
                context: "std_header_version.specVersion".into(),
                problem: "is greater than video_capabilities.std_header_version.spec_version"
                    .into(),
                vuids: &["VUID-VkVideoSessionCreateInfoKHR-pStdHeaderVersion-07191"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

impl Default for VideoSessionCreateInfo {
    fn default() -> Self {
        Self {
            queue_family_index: Default::default(),
            flags: Default::default(),
            video_profile: Default::default(),
            picture_format: Default::default(),
            max_coded_extent: Default::default(),
            reference_picture_format: Default::default(),
            max_dpb_slots: Default::default(),
            max_active_reference_pictures: Default::default(),
            std_header_version: ExtensionProperties::from(ash::vk::ExtensionProperties::default()),
            _ne: crate::NonExhaustive(()),
        }
    }
}
