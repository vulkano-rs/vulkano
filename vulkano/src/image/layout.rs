// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{descriptor_set::layout::DescriptorType, macros::vulkan_enum};

vulkan_enum! {
    /// In-memory layout of the pixel data of an image.
    ///
    /// The pixel data of a Vulkan image is arranged in a particular way, which is called its
    /// *layout*. Each image subresource (mipmap level and array layer) in an image can have a
    /// different layout, but usually the whole image has its data in the same layout. Layouts are
    /// abstract in the sense that the user does not know the specific details of each layout; the
    /// device driver is free to implement each layout in the way it sees fit.
    ///
    /// The layout of a newly created image is either `Undefined` or `Preinitialized`. Every
    /// operation that can be performed on an image is only possible with specific layouts, so
    /// before the operation is performed, the user must perform a *layout transition* on the
    /// image. This rearranges the pixel data from one layout into another. Layout transitions are
    /// performed as part of pipeline barriers in a command buffer.
    ///
    /// The `General` layout is compatible with any operation, so layout transitions are never
    /// needed. However, the other layouts, while more restricted, are usually better optimised for
    /// a particular type of operation than `General`, so they are usually preferred.
    ///
    /// Vulkan does not keep track of layouts itself, so it is the responsibility of the user to
    /// keep track of this information. When performing a layout transition, the previous layout
    /// must be specified as well. Some operations allow for different layouts, but require the
    /// user to specify which one. Vulkano helps with this by providing sensible defaults,
    /// automatically tracking the layout of each image when creating a command buffer, and adding
    /// layout transitions where needed.
    #[non_exhaustive]
    ImageLayout = ImageLayout(i32);

    /// The layout of the data is unknown, and the image is treated as containing no valid data.
    /// Transitioning from `Undefined` will discard any existing pixel data.
    Undefined = UNDEFINED,

    /// A general-purpose layout that can be used for any operation. Some operations may only allow
    /// `General`, such as storage images, but many have a more specific layout that is better
    /// optimized for that purpose.
    General = GENERAL,

    /// For a color image used as a color or resolve attachment in a framebuffer. Images that are
    /// transitioned into this layout must have the `color_attachment` usage enabled.
    ColorAttachmentOptimal = COLOR_ATTACHMENT_OPTIMAL,

    /// For a depth/stencil image used as a depth/stencil attachment in a framebuffer.
    DepthStencilAttachmentOptimal = DEPTH_STENCIL_ATTACHMENT_OPTIMAL,

    /// For a depth/stencil image used as a read-only depth/stencil attachment in a framebuffer, or
    /// as a (combined) sampled image or input attachment in a shader.
    DepthStencilReadOnlyOptimal = DEPTH_STENCIL_READ_ONLY_OPTIMAL,

    /// For a color image used as a (combined) sampled image or input attachment in a shader.
    /// Images that are transitioned into this layout must have the `sampled` or `input_attachment`
    /// usages enabled.
    ShaderReadOnlyOptimal = SHADER_READ_ONLY_OPTIMAL,

    /// For operations that transfer data from an image (copy, blit).
    TransferSrcOptimal = TRANSFER_SRC_OPTIMAL,

    /// For operations that transfer data to an image (copy, blit, clear).
    TransferDstOptimal = TRANSFER_DST_OPTIMAL,

    /// When creating an image, this specifies that the initial data is going to be directly
    /// written to from the CPU. Unlike `Undefined`, the image is assumed to contain valid data when
    /// transitioning from this layout. However, this only works right when the image has linear
    /// tiling, optimal tiling gives undefined results.
    Preinitialized = PREINITIALIZED,

    /// A combination of `DepthStencilReadOnlyOptimal` for the depth aspect of the image,
    /// and `DepthStencilAttachmentOptimal` for the stencil aspect of the image.
    DepthReadOnlyStencilAttachmentOptimal = DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL {
        api_version: V1_1,
        device_extensions: [khr_maintenance2],
    },

    /// A combination of `DepthStencilAttachmentOptimal` for the depth aspect of the image,
    /// and `DepthStencilReadOnlyOptimal` for the stencil aspect of the image.
    DepthAttachmentStencilReadOnlyOptimal = DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL {
        api_version: V1_1,
        device_extensions: [khr_maintenance2],
    },
    /*
    // TODO: document
    DepthAttachmentOptimal = DEPTH_ATTACHMENT_OPTIMAL {
        api_version: V1_2,
        device_extensions: [khr_separate_depth_stencil_layouts],
    },

    // TODO: document
    DepthReadOnlyOptimal = DEPTH_READ_ONLY_OPTIMAL {
        api_version: V1_2,
        device_extensions: [khr_separate_depth_stencil_layouts],
    },

    // TODO: document
    StencilAttachmentOptimal = STENCIL_ATTACHMENT_OPTIMAL {
        api_version: V1_2,
        device_extensions: [khr_separate_depth_stencil_layouts],
    },

    // TODO: document
    StencilReadOnlyOptimal = STENCIL_READ_ONLY_OPTIMAL {
        api_version: V1_2,
        device_extensions: [khr_separate_depth_stencil_layouts],
    },

    // TODO: document
    ReadOnlyOptimal = READ_ONLY_OPTIMAL {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    // TODO: document
    AttachmentOptimal = ATTACHMENT_OPTIMAL {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },
     */

    /// The layout of images that are held in a swapchain. Images are in this layout when they are
    /// acquired from the swapchain, and must be transitioned back into this layout before
    /// presenting them.
    PresentSrc = PRESENT_SRC_KHR {
        device_extensions: [khr_swapchain],
    },

    /*
    // TODO: document
    VideoDecodeDst = VIDEO_DECODE_DST_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    VideoDecodeSrc = VIDEO_DECODE_SRC_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    VideoDecodeDpb = VIDEO_DECODE_DPB_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    SharedPresent = SHARED_PRESENT_KHR {
        device_extensions: [khr_shared_presentable_image],
    },

    // TODO: document
    FragmentDensityMapOptimal = FRAGMENT_DENSITY_MAP_OPTIMAL_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    // TODO: document
    FragmentShadingRateAttachmentOptimal = FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    // TODO: document
    VideoEncodeDst = VIDEO_ENCODE_DST_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    // TODO: document
    VideoEncodeSrc = VIDEO_ENCODE_SRC_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    // TODO: document
    VideoEncodeDpb = VIDEO_ENCODE_DPB_KHR {
        device_extensions: [khr_video_encode_queue],
    },
     */
}

/// The set of layouts to use for an image when used in descriptor of various kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDescriptorLayouts {
    /// The image layout to use in a descriptor as a storage image.
    pub storage_image: ImageLayout,
    /// The image layout to use in a descriptor as a combined image sampler.
    pub combined_image_sampler: ImageLayout,
    /// The image layout to use in a descriptor as a sampled image.
    pub sampled_image: ImageLayout,
    /// The image layout to use in a descriptor as an input attachment.
    pub input_attachment: ImageLayout,
}

impl ImageDescriptorLayouts {
    /// Returns the layout for the given descriptor type. Panics if `descriptor_type` is not an
    /// image descriptor type.
    #[inline]
    pub fn layout_for(&self, descriptor_type: DescriptorType) -> ImageLayout {
        match descriptor_type {
            DescriptorType::CombinedImageSampler => self.combined_image_sampler,
            DescriptorType::SampledImage => self.sampled_image,
            DescriptorType::StorageImage => self.storage_image,
            DescriptorType::InputAttachment => self.input_attachment,
            _ => panic!("{:?} is not an image descriptor type", descriptor_type),
        }
    }
}
