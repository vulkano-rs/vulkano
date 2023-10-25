// Copyright (c) 2023 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use ash::vk::native::StdVideoH264ChromaFormatIdc;
use ash::vk::native::StdVideoH264LevelIdc;
use ash::vk::native::StdVideoH264PocType;
use ash::vk::native::StdVideoH264PpsFlags;
use ash::vk::native::StdVideoH264ProfileIdc;
use ash::vk::native::StdVideoH264WeightedBipredIdc;

#[derive(Clone, Debug)]
pub struct VideoH264SpsFlags(ash::vk::native::StdVideoH264SpsFlags);

impl VideoH264SpsFlags {
    pub fn new(
        constraint_set0_flag: u32,
        constraint_set1_flag: u32,
        constraint_set2_flag: u32,
        constraint_set3_flag: u32,
        constraint_set4_flag: u32,
        constraint_set5_flag: u32,
        direct_8x8_inference_flag: u32,
        mb_adaptive_frame_field_flag: u32,
        frame_mbs_only_flag: u32,
        delta_pic_order_always_zero_flag: u32,
        separate_colour_plane_flag: u32,
        gaps_in_frame_num_value_allowed_flag: u32,
        qpprime_y_zero_transform_bypass_flag: u32,
        frame_cropping_flag: u32,
        seq_scaling_matrix_present_flag: u32,
        vui_parameters_present_flag: u32,
    ) -> Self {
        let _bitfield_1 = ash::vk::native::StdVideoH264SpsFlags::new_bitfield_1(
            constraint_set0_flag,
            constraint_set1_flag,
            constraint_set2_flag,
            constraint_set3_flag,
            constraint_set4_flag,
            constraint_set5_flag,
            direct_8x8_inference_flag,
            mb_adaptive_frame_field_flag,
            frame_mbs_only_flag,
            delta_pic_order_always_zero_flag,
            separate_colour_plane_flag,
            gaps_in_frame_num_value_allowed_flag,
            qpprime_y_zero_transform_bypass_flag,
            frame_cropping_flag,
            seq_scaling_matrix_present_flag,
            vui_parameters_present_flag,
        );

        Self(ash::vk::native::StdVideoH264SpsFlags {
            _bitfield_align_1: Default::default(),
            _bitfield_1,
            __bindgen_padding_0: Default::default(),
        })
    }
}

// Using u32 for now. This macro does not work if the path is not reexported
// into ash::vk
// ash::vk_bitflags_wrapped! {
//
//     VideoH264ProfileIdc = StdVideoH264ProfileIdc(u32);
//
//     VIDEO_H264_PROFILE_IDC_BASELINE = StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
//     VIDEO_H264_PROFILE_IDC_MAIN = StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
//     VIDEO_H264_PROFILE_IDC_HIGH = StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
//     VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE = StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
//     VIDEO_H264_PROFILE_IDC_INVALID = StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_INVALID,
// }

#[derive(Clone, Debug)]
pub struct VideoH264SequenceParameterSet {
    pub flags: VideoH264SpsFlags,
    pub profile_idc: StdVideoH264ProfileIdc,
    pub level_idc: StdVideoH264LevelIdc,
    pub chroma_format_idc: StdVideoH264ChromaFormatIdc,
    pub seq_parameter_set_id: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: StdVideoH264PocType,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub num_ref_frames_in_pic_order_cnt_cycle: u8,
    pub max_num_ref_frames: u8,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    pub offset_for_ref_frame: [i32; 255],
}

#[derive(Clone, Debug)]
pub struct VideoH264PpsFlags(ash::vk::native::StdVideoH264PpsFlags);

impl VideoH264PpsFlags {
    pub fn new(
        transform_8x8_mode_flag: u32,
        redundant_pic_cnt_present_flag: u32,
        constrained_intra_pred_flag: u32,
        deblocking_filter_control_present_flag: u32,
        weighted_pred_flag: u32,
        bottom_field_pic_order_in_frame_present_flag: u32,
        entropy_coding_mode_flag: u32,
        pic_scaling_matrix_present_flag: u32,
    ) -> Self {
        let _bitfield_1 = ash::vk::native::StdVideoH264PpsFlags::new_bitfield_1(
            transform_8x8_mode_flag,
            redundant_pic_cnt_present_flag,
            constrained_intra_pred_flag,
            deblocking_filter_control_present_flag,
            weighted_pred_flag,
            bottom_field_pic_order_in_frame_present_flag,
            entropy_coding_mode_flag,
            pic_scaling_matrix_present_flag,
        );

        Self(ash::vk::native::StdVideoH264PpsFlags {
            _bitfield_align_1: Default::default(),
            _bitfield_1,
            __bindgen_padding_0: Default::default(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct VideoH264ScalingLists {
    pub scaling_list_present_mask: u16,
    pub use_default_scaling_matrix_mask: u16,
    pub scaling_list_4x4: [[u8; 16usize]; 6usize],
    pub scaling_list_8x8: [[u8; 64usize]; 6usize],
}

#[derive(Clone, Debug)]
pub struct VideoH264PictureParameterSet {
    pub flags: StdVideoH264PpsFlags,
    pub seq_parameter_set_id: u8,
    pub pic_parameter_set_id: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_bipred_idc: StdVideoH264WeightedBipredIdc,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub scaling_lists: VideoH264ScalingLists,
}

#[derive(Clone, Debug)]
pub struct VideoDecodeH264SessionParametersAddInfo {
    pub std_sp_ss: Vec<VideoH264SequenceParameterSet>,
    pub std_pp_ss: Vec<VideoH264PictureParameterSet>,
}
