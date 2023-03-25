// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::RenderPass;
use crate::{
    device::{Device, DeviceOwned},
    format::Format,
    image::{view::ImageViewType, ImageDimensions, ImageUsage, ImageViewAbstract, SampleCount},
    macros::impl_id_counter,
    OomError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Range,
    ptr,
    sync::Arc,
};

/// The image views that are attached to a render pass during drawing.
///
/// A framebuffer is a collection of images, and supplies the actual inputs and outputs of each
/// attachment within a render pass. Each attachment point in the render pass must have a matching
/// image in the framebuffer.
///
/// ```
/// # use std::sync::Arc;
/// # use vulkano::render_pass::RenderPass;
/// # use vulkano::image::AttachmentImage;
/// # use vulkano::image::view::ImageView;
/// use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
///
/// # let render_pass: Arc<RenderPass> = return;
/// # let view: Arc<ImageView<AttachmentImage>> = return;
/// // let render_pass: Arc<_> = ...;
/// let framebuffer = Framebuffer::new(
///     render_pass.clone(),
///     FramebufferCreateInfo {
///         attachments: vec![view],
///         ..Default::default()
///     },
/// ).unwrap();
/// ```
#[derive(Debug)]
pub struct Framebuffer {
    handle: ash::vk::Framebuffer,
    render_pass: Arc<RenderPass>,
    id: NonZeroU64,

    attachments: Vec<Arc<dyn ImageViewAbstract>>,
    extent: [u32; 2],
    layers: u32,
}

impl Framebuffer {
    /// Creates a new `Framebuffer`.
    pub fn new(
        render_pass: Arc<RenderPass>,
        create_info: FramebufferCreateInfo,
    ) -> Result<Arc<Framebuffer>, FramebufferCreationError> {
        let FramebufferCreateInfo {
            attachments,
            mut extent,
            mut layers,
            _ne: _,
        } = create_info;

        let device = render_pass.device();

        // VUID-VkFramebufferCreateInfo-attachmentCount-00876
        if attachments.len() != render_pass.attachments().len() {
            return Err(FramebufferCreationError::AttachmentCountMismatch {
                provided: attachments.len() as u32,
                required: render_pass.attachments().len() as u32,
            });
        }

        let auto_extent = extent[0] == 0 || extent[1] == 0;
        let auto_layers = layers == 0;

        // VUID-VkFramebufferCreateInfo-width-00885
        // VUID-VkFramebufferCreateInfo-height-00887
        if auto_extent {
            if attachments.is_empty() {
                return Err(FramebufferCreationError::AutoExtentAttachmentsEmpty);
            }

            extent = [u32::MAX, u32::MAX];
        }

        // VUID-VkFramebufferCreateInfo-layers-00889
        if auto_layers {
            if attachments.is_empty() {
                return Err(FramebufferCreationError::AutoLayersAttachmentsEmpty);
            }

            if render_pass.views_used() != 0 {
                // VUID-VkFramebufferCreateInfo-renderPass-02531
                layers = 1;
            } else {
                layers = u32::MAX;
            }
        } else {
            // VUID-VkFramebufferCreateInfo-renderPass-02531
            if render_pass.views_used() != 0 && layers != 1 {
                return Err(FramebufferCreationError::MultiviewLayersInvalid);
            }
        }

        let attachments_vk = attachments
            .iter()
            .zip(render_pass.attachments())
            .enumerate()
            .map(|(attachment_num, (image_view, attachment_desc))| {
                let attachment_num = attachment_num as u32;
                assert_eq!(device, image_view.device());

                for subpass in render_pass.subpasses() {
                    // VUID-VkFramebufferCreateInfo-pAttachments-00877
                    if subpass
                        .color_attachments
                        .iter()
                        .flatten()
                        .any(|atch_ref| atch_ref.attachment == attachment_num)
                    {
                        if !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                            return Err(FramebufferCreationError::AttachmentMissingUsage {
                                attachment: attachment_num,
                                usage: "color_attachment",
                            });
                        }
                    }

                    // VUID-VkFramebufferCreateInfo-pAttachments-02633
                    if let Some(atch_ref) = &subpass.depth_stencil_attachment {
                        if atch_ref.attachment == attachment_num {
                            if !image_view
                                .usage()
                                .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                            {
                                return Err(FramebufferCreationError::AttachmentMissingUsage {
                                    attachment: attachment_num,
                                    usage: "depth_stencil",
                                });
                            }
                        }
                    }

                    // VUID-VkFramebufferCreateInfo-pAttachments-00879
                    if subpass
                        .input_attachments
                        .iter()
                        .flatten()
                        .any(|atch_ref| atch_ref.attachment == attachment_num)
                    {
                        if !image_view.usage().intersects(ImageUsage::INPUT_ATTACHMENT) {
                            return Err(FramebufferCreationError::AttachmentMissingUsage {
                                attachment: attachment_num,
                                usage: "input_attachment",
                            });
                        }
                    }
                }

                // VUID-VkFramebufferCreateInfo-pAttachments-00880
                if image_view.format() != attachment_desc.format {
                    return Err(FramebufferCreationError::AttachmentFormatMismatch {
                        attachment: attachment_num,
                        provided: image_view.format(),
                        required: attachment_desc.format,
                    });
                }

                // VUID-VkFramebufferCreateInfo-pAttachments-00881
                if image_view.image().samples() != attachment_desc.samples {
                    return Err(FramebufferCreationError::AttachmentSamplesMismatch {
                        attachment: attachment_num,
                        provided: image_view.image().samples(),
                        required: attachment_desc.samples,
                    });
                }

                let image_view_extent = image_view.image().dimensions().width_height();
                let image_view_array_layers = image_view.subresource_range().array_layers.end
                    - image_view.subresource_range().array_layers.start;

                // VUID-VkFramebufferCreateInfo-renderPass-04536
                if image_view_array_layers < render_pass.views_used() {
                    return Err(
                        FramebufferCreationError::MultiviewAttachmentNotEnoughLayers {
                            attachment: attachment_num,
                            provided: image_view_array_layers,
                            min: render_pass.views_used(),
                        },
                    );
                }

                // VUID-VkFramebufferCreateInfo-flags-04533
                // VUID-VkFramebufferCreateInfo-flags-04534
                if auto_extent {
                    extent[0] = extent[0].min(image_view_extent[0]);
                    extent[1] = extent[1].min(image_view_extent[1]);
                } else if image_view_extent[0] < extent[0] || image_view_extent[1] < extent[1] {
                    return Err(FramebufferCreationError::AttachmentExtentTooSmall {
                        attachment: attachment_num,
                        provided: image_view_extent,
                        min: extent,
                    });
                }

                // VUID-VkFramebufferCreateInfo-flags-04535
                if auto_layers {
                    layers = layers.min(image_view_array_layers);
                } else if image_view_array_layers < layers {
                    return Err(FramebufferCreationError::AttachmentNotEnoughLayers {
                        attachment: attachment_num,
                        provided: image_view_array_layers,
                        min: layers,
                    });
                }

                // VUID-VkFramebufferCreateInfo-pAttachments-00883
                if image_view.subresource_range().mip_levels.end
                    - image_view.subresource_range().mip_levels.start
                    != 1
                {
                    return Err(FramebufferCreationError::AttachmentMultipleMipLevels {
                        attachment: attachment_num,
                    });
                }

                // VUID-VkFramebufferCreateInfo-pAttachments-00884
                if !image_view.component_mapping().is_identity() {
                    return Err(
                        FramebufferCreationError::AttachmentComponentMappingNotIdentity {
                            attachment: attachment_num,
                        },
                    );
                }

                // VUID-VkFramebufferCreateInfo-pAttachments-00891
                if matches!(
                    image_view.view_type(),
                    ImageViewType::Dim2d | ImageViewType::Dim2dArray
                ) && matches!(
                    image_view.image().dimensions(),
                    ImageDimensions::Dim3d { .. }
                ) && image_view.format().unwrap().type_color().is_none()
                {
                    return Err(
                        FramebufferCreationError::Attachment2dArrayCompatibleDepthStencil {
                            attachment: attachment_num,
                        },
                    );
                }

                // VUID-VkFramebufferCreateInfo-flags-04113
                if image_view.view_type() == ImageViewType::Dim3d {
                    return Err(FramebufferCreationError::AttachmentViewType3d {
                        attachment: attachment_num,
                    });
                }

                Ok(image_view.handle())
            })
            .collect::<Result<SmallVec<[_; 4]>, _>>()?;

        {
            let properties = device.physical_device().properties();

            // VUID-VkFramebufferCreateInfo-width-00886
            // VUID-VkFramebufferCreateInfo-height-00888
            if extent[0] > properties.max_framebuffer_width
                || extent[1] > properties.max_framebuffer_height
            {
                return Err(FramebufferCreationError::MaxFramebufferExtentExceeded {
                    provided: extent,
                    max: [
                        properties.max_framebuffer_width,
                        properties.max_framebuffer_height,
                    ],
                });
            }

            // VUID-VkFramebufferCreateInfo-layers-00890
            if layers > properties.max_framebuffer_layers {
                return Err(FramebufferCreationError::MaxFramebufferLayersExceeded {
                    provided: layers,
                    max: properties.max_framebuffer_layers,
                });
            }
        }

        let create_info = ash::vk::FramebufferCreateInfo {
            flags: ash::vk::FramebufferCreateFlags::empty(),
            render_pass: render_pass.handle(),
            attachment_count: attachments_vk.len() as u32,
            p_attachments: attachments_vk.as_ptr(),
            width: extent[0],
            height: extent[1],
            layers,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_framebuffer)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Framebuffer {
            handle,
            render_pass,
            id: Self::next_id(),
            attachments,
            extent,
            layers,
        }))
    }

    /// Creates a new `Framebuffer` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `render_pass`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        render_pass: Arc<RenderPass>,
        handle: ash::vk::Framebuffer,
        create_info: FramebufferCreateInfo,
    ) -> Arc<Framebuffer> {
        let FramebufferCreateInfo {
            attachments,
            extent,
            layers,
            _ne: _,
        } = create_info;

        Arc::new(Framebuffer {
            handle,
            render_pass,
            id: Self::next_id(),
            attachments,
            extent,
            layers,
        })
    }
    /// Returns the renderpass that was used to create this framebuffer.
    #[inline]
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }

    /// Returns the attachments of the framebuffer.
    #[inline]
    pub fn attachments(&self) -> &[Arc<dyn ImageViewAbstract>] {
        &self.attachments
    }

    /// Returns the extent (width and height) of the framebuffer.
    #[inline]
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the number of layers of the framebuffer.
    #[inline]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Returns the layer ranges for all attachments.
    #[inline]
    pub fn attached_layers_ranges(&self) -> SmallVec<[Range<u32>; 4]> {
        self.attachments
            .iter()
            .map(|img| img.subresource_range().array_layers.clone())
            .collect()
    }
}

impl Drop for Framebuffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device().fns();
            (fns.v1_0.destroy_framebuffer)(self.device().handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Framebuffer {
    type Handle = ash::vk::Framebuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Framebuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.render_pass.device()
    }
}

impl_id_counter!(Framebuffer);

/// Parameters to create a new `Framebuffer`.
#[derive(Clone, Debug)]
pub struct FramebufferCreateInfo {
    /// The attachment images that are to be used in the framebuffer.
    ///
    /// Attachments are specified in the same order as they are defined in the render pass, and
    /// there must be exactly as many. This implies that the list must be empty if the render pass
    /// specifies no attachments. Each image must have the correct usages set to be used for the
    /// types of attachment that the render pass will use it as.
    ///
    /// The attachment images must not be smaller than `extent` and `layers`, but can be larger and
    /// have different sizes from each other. Any leftover parts of an image will be left untouched
    /// during rendering.
    ///
    /// If the render pass has multiview enabled (`views_used` does not return 0), then each
    /// image must have at least `views_used` array layers.
    ///
    /// The default value is empty.
    pub attachments: Vec<Arc<dyn ImageViewAbstract>>,

    /// The extent (width and height) of the framebuffer.
    ///
    /// This must be no larger than the smallest width and height of the images in `attachments`.
    /// If one of the elements is set to 0, the extent will be calculated automatically from the
    /// extents of the attachment images to be the largest allowed. At least one attachment image
    /// must be specified in that case.
    ///
    /// The extent, whether automatically calculated or specified explicitly, must not be larger
    /// than the [`max_framebuffer_width`](crate::device::Properties::max_framebuffer_width) and
    /// [`max_framebuffer_height`](crate::device::Properties::max_framebuffer_height) limits.
    ///
    /// The default value is `[0, 0]`.
    pub extent: [u32; 2],

    /// The number of layers of the framebuffer.
    ///
    /// This must be no larger than the smallest number of array layers of the images in
    /// `attachments`. If set to 0, the number of layers will be calculated automatically from the
    /// layer ranges of the attachment images to be the largest allowed. At least one attachment
    /// image must be specified in that case.
    ///
    /// The number of layers, whether automatically calculated or specified explicitly, must not be
    /// larger than the
    /// [`max_framebuffer_layers`](crate::device::Properties::max_framebuffer_layers) limit.
    ///
    /// If the render pass has multiview enabled (`views_used` does not return 0), then this value
    /// must be 0 or 1.
    ///
    /// The default value is `0`.
    pub layers: u32,

    pub _ne: crate::NonExhaustive,
}

impl Default for FramebufferCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            attachments: Vec::new(),
            extent: [0, 0],
            layers: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Error that can happen when creating a `Framebuffer`.
#[derive(Copy, Clone, Debug)]
pub enum FramebufferCreationError {
    /// Out of memory.
    OomError(OomError),

    /// An attachment image is a 2D image view created from a 3D image, and has a depth/stencil
    /// format.
    Attachment2dArrayCompatibleDepthStencil { attachment: u32 },

    /// An attachment image has a non-identity component mapping.
    AttachmentComponentMappingNotIdentity { attachment: u32 },

    /// The number of attachments doesn't match the number expected by the render pass.
    AttachmentCountMismatch { provided: u32, required: u32 },

    /// An attachment image has an extent smaller than the provided `extent`.
    AttachmentExtentTooSmall {
        attachment: u32,
        provided: [u32; 2],
        min: [u32; 2],
    },

    /// An attachment image has a `format` different from what the render pass requires.
    AttachmentFormatMismatch {
        attachment: u32,
        provided: Option<Format>,
        required: Option<Format>,
    },

    /// An attachment image is missing a usage that the render pass requires it to have.
    AttachmentMissingUsage {
        attachment: u32,
        usage: &'static str,
    },

    /// An attachment image has multiple mip levels.
    AttachmentMultipleMipLevels { attachment: u32 },

    /// An attachment image has less array layers than the provided `layers`.
    AttachmentNotEnoughLayers {
        attachment: u32,
        provided: u32,
        min: u32,
    },

    /// An attachment image has a `samples` different from what the render pass requires.
    AttachmentSamplesMismatch {
        attachment: u32,
        provided: SampleCount,
        required: SampleCount,
    },

    /// An attachment image has a `ty` of [`ImageViewType::Dim3d`].
    AttachmentViewType3d { attachment: u32 },

    /// One of the elements of `extent` is zero, but no attachment images were given to calculate
    /// the extent from.
    AutoExtentAttachmentsEmpty,

    /// `layers` is zero, but no attachment images were given to calculate the number of layers
    /// from.
    AutoLayersAttachmentsEmpty,

    /// The provided `extent` exceeds the `max_framebuffer_width` or `max_framebuffer_height`
    /// limits.
    MaxFramebufferExtentExceeded { provided: [u32; 2], max: [u32; 2] },

    /// The provided `layers` exceeds the `max_framebuffer_layers` limit.
    MaxFramebufferLayersExceeded { provided: u32, max: u32 },

    /// The render pass has multiview enabled, and an attachment image has less layers than the
    /// number of views in the render pass.
    MultiviewAttachmentNotEnoughLayers {
        attachment: u32,
        provided: u32,
        min: u32,
    },

    /// The render pass has multiview enabled, but `layers` was not 0 or 1.
    MultiviewLayersInvalid,
}

impl From<OomError> for FramebufferCreationError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl Error for FramebufferCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for FramebufferCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "no memory available",),
            Self::Attachment2dArrayCompatibleDepthStencil { attachment } => write!(
                f,
                "attachment image {} is a 2D image view created from a 3D image, and has a \
                depth/stencil format",
                attachment,
            ),
            Self::AttachmentComponentMappingNotIdentity { attachment } => write!(
                f,
                "attachment image {} has a non-identity component mapping",
                attachment,
            ),
            Self::AttachmentCountMismatch { .. } => write!(
                f,
                "the number of attachments doesn't match the number expected by the render pass",
            ),
            Self::AttachmentExtentTooSmall {
                attachment,
                provided,
                min,
            } => write!(
                f,
                "attachment image {} has an extent ({:?}) smaller than the provided `extent` \
                ({:?})",
                attachment, provided, min,
            ),
            Self::AttachmentFormatMismatch {
                attachment,
                provided,
                required,
            } => write!(
                f,
                "attachment image {} has a `format` ({:?}) different from what the render pass \
                requires ({:?})",
                attachment, provided, required,
            ),
            Self::AttachmentMissingUsage { attachment, usage } => write!(
                f,
                "attachment image {} is missing usage `{}` that the render pass requires it to \
                have",
                attachment, usage,
            ),
            Self::AttachmentMultipleMipLevels { attachment } => {
                write!(f, "attachment image {} has multiple mip levels", attachment)
            }
            Self::AttachmentNotEnoughLayers {
                attachment,
                provided,
                min,
            } => write!(
                f,
                "attachment image {} has less layers ({}) than the provided `layers` ({})",
                attachment, provided, min,
            ),
            Self::AttachmentSamplesMismatch {
                attachment,
                provided,
                required,
            } => write!(
                f,
                "attachment image {} has a `samples` ({:?}) different from what the render pass \
                requires ({:?})",
                attachment, provided, required,
            ),
            Self::AttachmentViewType3d { attachment } => write!(
                f,
                "attachment image {} has a `ty` of `ImageViewType::Dim3d`",
                attachment,
            ),
            Self::AutoExtentAttachmentsEmpty => write!(
                f,
                "one of the elements of `extent` is zero, but no attachment images were given to \
                calculate the extent from",
            ),
            Self::AutoLayersAttachmentsEmpty => write!(
                f,
                "`layers` is zero, but no attachment images were given to calculate the number of \
                layers from",
            ),
            Self::MaxFramebufferExtentExceeded { provided, max } => write!(
                f,
                "the provided `extent` ({:?}) exceeds the `max_framebuffer_width` or \
                `max_framebuffer_height` limits ({:?})",
                provided, max,
            ),
            Self::MaxFramebufferLayersExceeded { provided, max } => write!(
                f,
                "the provided `layers` ({}) exceeds the `max_framebuffer_layers` limit ({})",
                provided, max,
            ),
            Self::MultiviewAttachmentNotEnoughLayers {
                attachment,
                provided,
                min,
            } => write!(
                f,
                "the render pass has multiview enabled, and attachment image {} has less layers \
                ({}) than the number of views in the render pass ({})",
                attachment, provided, min,
            ),
            Self::MultiviewLayersInvalid => write!(
                f,
                "the render pass has multiview enabled, but `layers` was not 0 or 1",
            ),
        }
    }
}

impl From<VulkanError> for FramebufferCreationError {
    fn from(err: VulkanError) -> Self {
        Self::from(OomError::from(err))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        format::Format,
        image::{attachment::AttachmentImage, view::ImageView},
        memory::allocator::StandardMemoryAllocator,
        render_pass::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass},
    };

    #[test]
    fn simple_create() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let view = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [1024, 768], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let _ = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn check_device_limits() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = RenderPass::empty_single_pass(device).unwrap();
        let res = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                extent: [0xffffffff, 0xffffffff],
                layers: 1,
                ..Default::default()
            },
        );
        match res {
            Err(FramebufferCreationError::MaxFramebufferExtentExceeded { .. }) => (),
            _ => panic!(),
        }

        let res = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                extent: [1, 1],
                layers: 0xffffffff,
                ..Default::default()
            },
        );

        match res {
            Err(FramebufferCreationError::MaxFramebufferLayersExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn attachment_format_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let view = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [1024, 768], Format::R8_UNORM).unwrap(),
        )
        .unwrap();

        match Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        ) {
            Err(FramebufferCreationError::AttachmentFormatMismatch { .. }) => (),
            _ => panic!(),
        }
    }

    // TODO: check samples mismatch

    #[test]
    fn attachment_dims_larger_than_specified_valid() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let view = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [600, 600], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let _ = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                extent: [512, 512],
                layers: 1,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn attachment_dims_smaller_than_specified() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let view = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [512, 700], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        match Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                extent: [600, 600],
                layers: 1,
                ..Default::default()
            },
        ) {
            Err(FramebufferCreationError::AttachmentExtentTooSmall { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn multi_attachments_auto_smaller() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                a: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                b: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [a, b],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let a = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let b = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [512, 128], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let framebuffer = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![a, b],
                ..Default::default()
            },
        )
        .unwrap();

        match (framebuffer.extent(), framebuffer.layers()) {
            ([256, 128], 1) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn not_enough_attachments() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                a: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                b: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [a, b],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let view = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let res = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        );

        match res {
            Err(FramebufferCreationError::AttachmentCountMismatch {
                required: 2,
                provided: 1,
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn too_many_attachments() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                a: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
            },
            pass: {
                color: [a],
                depth_stencil: {},
            },
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let a = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let b = ImageView::new_default(
            AttachmentImage::new(&memory_allocator, [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let res = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![a, b],
                ..Default::default()
            },
        );

        match res {
            Err(FramebufferCreationError::AttachmentCountMismatch {
                required: 1,
                provided: 2,
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn empty_working() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = RenderPass::empty_single_pass(device).unwrap();
        let _ = Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                extent: [512, 512],
                layers: 1,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn cant_determine_dimensions_auto() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = RenderPass::empty_single_pass(device).unwrap();
        let res = Framebuffer::new(render_pass, FramebufferCreateInfo::default());
        match res {
            Err(FramebufferCreationError::AutoExtentAttachmentsEmpty) => (),
            _ => panic!(),
        }
    }
}
