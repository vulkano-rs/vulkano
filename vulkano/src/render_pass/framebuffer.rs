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
    image::{
        view::{ImageView, ImageViewType},
        ImageAspects, ImageType, ImageUsage,
    },
    macros::{impl_id_counter, vulkan_bitflags},
    RuntimeError, ValidationError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::MaybeUninit, num::NonZeroU64, ops::Range, ptr, sync::Arc};

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

    flags: FramebufferCreateFlags,
    attachments: Vec<Arc<ImageView>>,
    extent: [u32; 2],
    layers: u32,
}

impl Framebuffer {
    /// Creates a new `Framebuffer`.
    pub fn new(
        render_pass: Arc<RenderPass>,
        mut create_info: FramebufferCreateInfo,
    ) -> Result<Arc<Framebuffer>, VulkanError> {
        create_info.set_auto_extent_layers(&render_pass);
        Self::validate_new(&render_pass, &create_info)?;

        unsafe { Ok(Self::new_unchecked(render_pass, create_info)?) }
    }

    fn validate_new(
        render_pass: &RenderPass,
        create_info: &FramebufferCreateInfo,
    ) -> Result<(), ValidationError> {
        // VUID-vkCreateFramebuffer-pCreateInfo-parameter
        create_info
            .validate(render_pass.device())
            .map_err(|err| err.add_context("create_info"))?;

        let &FramebufferCreateInfo {
            flags: _,
            ref attachments,
            extent,
            layers,
            _ne,
        } = create_info;

        if attachments.len() != render_pass.attachments().len() {
            return Err(ValidationError {
                problem: "`create_info.attachments` does not have the same length as \
                    `render_pass.attachments()`"
                    .into(),
                vuids: &["VUID-VkFramebufferCreateInfo-attachmentCount-00876"],
                ..Default::default()
            });
        }

        for (index, ((image_view, attachment_desc), attachment_use)) in attachments
            .iter()
            .zip(render_pass.attachments())
            .zip(&render_pass.attachment_use)
            .enumerate()
        {
            if attachment_use.color_attachment
                && !image_view.usage().intersects(ImageUsage::COLOR_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: format!(
                        "`render_pass` uses `create_info.attachments[{}]` as \
                        a color attachment, but it was not created with the \
                        `ImageUsage::COLOR_ATTACHMENT` usage",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-00877"],
                    ..Default::default()
                });
            }

            if attachment_use.depth_stencil_attachment
                && !image_view
                    .usage()
                    .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: format!(
                        "`render_pass` uses `create_info.attachments[{}]` as \
                        a depth or stencil attachment, but it was not created with the \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT` usage",
                        index,
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkFramebufferCreateInfo-pAttachments-02633",
                        "VUID-VkFramebufferCreateInfo-pAttachments-02634",
                    ],
                    ..Default::default()
                });
            }

            if attachment_use.input_attachment
                && !image_view.usage().intersects(ImageUsage::INPUT_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: format!(
                        "`render_pass` uses `create_info.attachments[{}]` as \
                        an input attachment, but it was not created with the \
                        `ImageUsage::INPUT_ATTACHMENT` usage",
                        index,
                    )
                    .into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-02633"],
                    ..Default::default()
                });
            }

            if image_view.format() != attachment_desc.format {
                return Err(ValidationError {
                    problem: format!(
                        "the format of `create_info.attachments[{}]` does not equal \
                        `render_pass.attachments()[{0}].format`",
                        index,
                    )
                    .into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-00880"],
                    ..Default::default()
                });
            }

            if image_view.image().samples() != attachment_desc.samples {
                return Err(ValidationError {
                    problem: format!(
                        "the samples of `create_info.attachments[{}]` does not equal \
                        `render_pass.attachments()[{0}].samples`",
                        index,
                    )
                    .into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-00881"],
                    ..Default::default()
                });
            }

            let image_view_extent = image_view.image().dimensions().width_height();
            let image_view_array_layers = image_view.subresource_range().array_layers.end
                - image_view.subresource_range().array_layers.start;

            if attachment_use.input_attachment
                || attachment_use.color_attachment
                || attachment_use.depth_stencil_attachment
            {
                if image_view_extent[0] < extent[0] || image_view_extent[1] < extent[1] {
                    return Err(ValidationError {
                        problem: format!(
                            "`render_pass` uses `create_info.attachments[{}]` as an input, color, \
                            depth or stencil attachment, but \
                            its width and height are less than `create_info.extent`",
                            index,
                        )
                        .into(),
                        vuids: &[
                            "VUID-VkFramebufferCreateInfo-flags-04533",
                            "VUID-VkFramebufferCreateInfo-flags-04534",
                        ],
                        ..Default::default()
                    });
                }

                if image_view_array_layers < layers {
                    return Err(ValidationError {
                        problem: format!(
                            "`render_pass` uses `create_info.attachments[{}]` as an input, color, \
                            depth or stencil attachment, but \
                            its layer count is less than `create_info.layers`",
                            index,
                        )
                        .into(),
                        vuids: &["VUID-VkFramebufferCreateInfo-flags-04535"],
                        ..Default::default()
                    });
                }

                if image_view_array_layers < render_pass.views_used() {
                    return Err(ValidationError {
                        problem: format!(
                            "`render_pass` has multiview enabled, and uses \
                            `create_info.attachments[{}]` as an input, color, depth or stencil \
                            attachment, but its layer count is less than the number of views used \
                            by `render_pass`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkFramebufferCreateInfo-renderPass-04536"],
                        ..Default::default()
                    });
                }
            }

            if render_pass.views_used() != 0 && layers != 1 {
                return Err(ValidationError {
                    problem: "`render_pass` has multiview enabled, but \
                        `create_info.layers` is not 1"
                        .into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-renderPass-02531"],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        render_pass: Arc<RenderPass>,
        mut create_info: FramebufferCreateInfo,
    ) -> Result<Arc<Framebuffer>, RuntimeError> {
        create_info.set_auto_extent_layers(&render_pass);

        let &FramebufferCreateInfo {
            flags,
            ref attachments,
            extent,
            layers,
            _ne: _,
        } = &create_info;

        let attachments_vk: SmallVec<[_; 4]> =
            attachments.iter().map(VulkanObject::handle).collect();

        let create_info_vk = ash::vk::FramebufferCreateInfo {
            flags: flags.into(),
            render_pass: render_pass.handle(),
            attachment_count: attachments_vk.len() as u32,
            p_attachments: attachments_vk.as_ptr(),
            width: extent[0],
            height: extent[1],
            layers,
            ..Default::default()
        };

        let handle = unsafe {
            let fns = render_pass.device().fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_framebuffer)(
                render_pass.device().handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(render_pass, handle, create_info))
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
        mut create_info: FramebufferCreateInfo,
    ) -> Arc<Framebuffer> {
        create_info.set_auto_extent_layers(&render_pass);

        let FramebufferCreateInfo {
            flags,
            attachments,
            extent,
            layers,
            _ne: _,
        } = create_info;

        Arc::new(Framebuffer {
            handle,
            render_pass,
            id: Self::next_id(),

            flags,
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

    /// Returns the flags that the framebuffer was created with.
    #[inline]
    pub fn flags(&self) -> FramebufferCreateFlags {
        self.flags
    }

    /// Returns the attachments of the framebuffer.
    #[inline]
    pub fn attachments(&self) -> &[Arc<ImageView>] {
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
    /// Additional properties of the framebuffer.
    ///
    /// The default value is empty.
    pub flags: FramebufferCreateFlags,

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
    pub attachments: Vec<Arc<ImageView>>,

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
            flags: FramebufferCreateFlags::empty(),
            attachments: Vec::new(),
            extent: [0, 0],
            layers: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl FramebufferCreateInfo {
    fn set_auto_extent_layers(&mut self, render_pass: &RenderPass) {
        let Self {
            flags: _,
            attachments,
            extent,
            layers,
            _ne: _,
        } = self;

        let is_auto_extent = extent[0] == 0 || extent[1] == 0;
        let is_auto_layers = *layers == 0;

        if (is_auto_extent || is_auto_layers) && !attachments.is_empty() {
            let mut auto_extent = [u32::MAX, u32::MAX];
            let mut auto_layers = if render_pass.views_used() != 0 {
                // VUID-VkFramebufferCreateInfo-renderPass-02531
                1
            } else {
                u32::MAX
            };

            for image_view in attachments.iter() {
                let image_view_extent = image_view.image().dimensions().width_height();
                let image_view_array_layers = image_view.subresource_range().array_layers.end
                    - image_view.subresource_range().array_layers.start;

                auto_extent[0] = auto_extent[0].min(image_view_extent[0]);
                auto_extent[1] = auto_extent[1].min(image_view_extent[1]);
                auto_layers = auto_layers.min(image_view_array_layers);
            }

            if is_auto_extent {
                *extent = auto_extent;
            }

            if is_auto_layers {
                *layers = auto_layers;
            }
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            ref attachments,
            extent,
            layers,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        for (index, image_view) in attachments.iter().enumerate() {
            assert_eq!(device, image_view.device().as_ref());

            let image_view_mip_levels = image_view.subresource_range().mip_levels.end
                - image_view.subresource_range().mip_levels.start;

            if image_view_mip_levels != 1 {
                return Err(ValidationError {
                    context: format!("attachments[{}]", index).into(),
                    problem: "has more than one mip level".into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-00883"],
                    ..Default::default()
                });
            }

            if !image_view.component_mapping().is_identity() {
                return Err(ValidationError {
                    context: format!("attachments[{}]", index).into(),
                    problem: "is not identity swizzled".into(),
                    vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-00884"],
                    ..Default::default()
                });
            }

            match image_view.view_type() {
                ImageViewType::Dim2d | ImageViewType::Dim2dArray => {
                    if image_view.image().dimensions().image_type() == ImageType::Dim3d
                        && (image_view.format().unwrap().aspects())
                            .intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    {
                        return Err(ValidationError {
                            context: format!("attachments[{}]", index).into(),
                            problem: "is a 2D or 2D array image view, but its format is a \
                                depth/stencil format"
                                .into(),
                            vuids: &["VUID-VkFramebufferCreateInfo-pAttachments-00891"],
                            ..Default::default()
                        });
                    }
                }
                ImageViewType::Dim3d => {
                    return Err(ValidationError {
                        context: format!("attachments[{}]", index).into(),
                        problem: "is a 3D image view".into(),
                        vuids: &["VUID-VkFramebufferCreateInfo-flags-04113"],
                        ..Default::default()
                    });
                }
                _ => (),
            }
        }

        let properties = device.physical_device().properties();

        if extent[0] == 0 {
            return Err(ValidationError {
                context: "extent[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-width-00885"],
                ..Default::default()
            });
        }

        if extent[0] > properties.max_framebuffer_width {
            return Err(ValidationError {
                context: "extent[0]".into(),
                problem: "exceeds the `max_framebuffer_width` limit".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-width-00886"],
                ..Default::default()
            });
        }

        if extent[1] == 0 {
            return Err(ValidationError {
                context: "extent[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-height-00887"],
                ..Default::default()
            });
        }

        if extent[1] > properties.max_framebuffer_height {
            return Err(ValidationError {
                context: "extent[1]".into(),
                problem: "exceeds the `max_framebuffer_height` limit".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-height-00888"],
                ..Default::default()
            });
        }

        if layers == 0 {
            return Err(ValidationError {
                context: "layers".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-layers-00889"],
                ..Default::default()
            });
        }

        if layers > properties.max_framebuffer_layers {
            return Err(ValidationError {
                context: "layers".into(),
                problem: "exceeds the `max_framebuffer_layers` limit".into(),
                vuids: &["VUID-VkFramebufferCreateInfo-layers-00890"],
                ..Default::default()
            });
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a framebuffer.
    FramebufferCreateFlags = FramebufferCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    IMAGELESS = IMAGELESS {
        api_version: V1_2,
        device_extensions: [khr_imageless_framebuffer],
    }, */
}

#[cfg(test)]
mod tests {
    use crate::{
        format::Format,
        image::{view::ImageView, Image, ImageCreateInfo, ImageDimensions},
        memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
        render_pass::{
            Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo,
            SubpassDescription,
        },
    };

    #[test]
    fn simple_create() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 1024,
                        height: 768,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
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

        let render_pass = RenderPass::new(
            device,
            RenderPassCreateInfo {
                subpasses: vec![SubpassDescription::default()],
                ..Default::default()
            },
        )
        .unwrap();

        if Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                extent: [0xffffffff, 0xffffffff],
                layers: 1,
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }

        if Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                extent: [1, 1],
                layers: 0xffffffff,
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }
    }

    #[test]
    fn attachment_format_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 1024,
                        height: 768,
                        array_layers: 1,
                    },
                    format: Some(Format::R8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        if Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
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
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 600,
                        height: 600,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
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
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 512,
                        height: 700,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        if Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                extent: [600, 600],
                layers: 1,
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }
    }

    #[test]
    fn multi_attachments_auto_smaller() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                a: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                b: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 256,
                        height: 512,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let b = ImageView::new_default(
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 512,
                        height: 128,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
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
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                b: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 256,
                        height: 512,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        if Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }
    }

    #[test]
    fn too_many_attachments() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass!(
            device.clone(),
            attachments: {
                a: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 256,
                        height: 512,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let b = ImageView::new_default(
            Image::new(
                &memory_allocator,
                ImageCreateInfo {
                    dimensions: ImageDimensions::Dim2d {
                        width: 256,
                        height: 512,
                        array_layers: 1,
                    },
                    format: Some(Format::R8G8B8A8_UNORM),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        if Framebuffer::new(
            render_pass,
            FramebufferCreateInfo {
                attachments: vec![a, b],
                ..Default::default()
            },
        )
        .is_ok()
        {
            panic!()
        }
    }

    #[test]
    fn empty_working() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = RenderPass::new(
            device,
            RenderPassCreateInfo {
                subpasses: vec![SubpassDescription::default()],
                ..Default::default()
            },
        )
        .unwrap();
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

        let render_pass = RenderPass::new(
            device,
            RenderPassCreateInfo {
                subpasses: vec![SubpassDescription::default()],
                ..Default::default()
            },
        )
        .unwrap();

        if Framebuffer::new(render_pass, FramebufferCreateInfo::default()).is_ok() {
            panic!()
        }
    }
}
