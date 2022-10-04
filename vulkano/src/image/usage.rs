// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::macros::vulkan_bitflags;

vulkan_bitflags! {
    /// Describes how an image is going to be used. This is **not** just an optimization.
    ///
    /// If you try to use an image in a way that you didn't declare, an error will occur.
    #[non_exhaustive]
    ImageUsage = ImageUsageFlags(u32);

    /// The image can be used as a source for transfer, blit, resolve and clear commands.
    transfer_src = TRANSFER_SRC,

    /// The image can be used as a destination for transfer, blit, resolve and clear commands.
    transfer_dst = TRANSFER_DST,

    /// The image can be used as a sampled image in a shader.
    sampled = SAMPLED,

    /// The image can be used as a storage image in a shader.
    storage = STORAGE,

    /// The image can be used as a color attachment in a render pass/framebuffer.
    color_attachment = COLOR_ATTACHMENT,

    /// The image can be used as a depth/stencil attachment in a render pass/framebuffer.
    depth_stencil_attachment = DEPTH_STENCIL_ATTACHMENT,

    /// The image will be used as an attachment, and will only ever be used temporarily.
    /// As soon as you leave a render pass, the content of transient images becomes undefined.
    ///
    /// This is a hint to the Vulkan implementation that it may not need allocate any memory for
    /// this image if the image can live entirely in some cache.
    ///
    /// If `transient_attachment` is true, then only `color_attachment`, `depth_stencil_attachment`
    /// and `input_attachment` can be true as well. The rest must be false or an error will be
    /// returned when creating the image.
    transient_attachment = TRANSIENT_ATTACHMENT,

    /// The image can be used as an input attachment in a render pass/framebuffer.
    input_attachment = INPUT_ATTACHMENT,

    /*
    // TODO: document
    video_decode_dst = VIDEO_DECODE_DST_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    video_decode_src = VIDEO_DECODE_SRC_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    video_decode_dpb = VIDEO_DECODE_DPB_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    fragment_density_map = FRAGMENT_DENSITY_MAP_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    // TODO: document
    fragment_shading_rate_attachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    // TODO: document
    video_encode_dst = VIDEO_ENCODE_DST_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    // TODO: document
    video_encode_src = VIDEO_ENCODE_SRC_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    // TODO: document
    video_encode_dpb = VIDEO_ENCODE_DPB_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    // TODO: document
    invocation_mask = INVOCATION_MASK_HUAWEI {
        device_extensions: [huawei_invocation_mask],
    },
     */
}
