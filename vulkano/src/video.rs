// Copyright (c) 2023 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::macros::vulkan_bitflags;

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
