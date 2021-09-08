// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::view::ImageViewAbstract;
use crate::render_pass::ensure_image_view_compatible;
use crate::render_pass::AttachmentsList;
use crate::render_pass::IncompatibleRenderPassAttachmentError;
use crate::render_pass::RenderPass;
use crate::Error;
use crate::OomError;
use crate::SafeDeref;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::cmp;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// The image views that are attached to a render pass during drawing.
///
/// A framebuffer is a collection of images, and supplies the actual inputs and outputs of each
/// subpass within a render pass. It is created from a subpass and must match it: each attachment
/// point in the subpass must have a matching image in the framebuffer.
///
/// Creating a framebuffer is done by calling `Framebuffer::start`, which returns a
/// `FramebufferBuilder` object. You can then add the framebuffer attachments one by one by
/// calling `add(image)`. When you are done, call `build()`.
///
/// Both the `add` and the `build` functions perform various checks to make sure that the number
/// of images is correct and that each image is compatible with the attachment definition in the
/// render pass.
///
/// ```
/// # use std::sync::Arc;
/// # use vulkano::render_pass::RenderPass;
/// use vulkano::render_pass::Framebuffer;
///
/// # let render_pass: Arc<RenderPass> = return;
/// # let view: Arc<vulkano::image::view::ImageView<Arc<vulkano::image::AttachmentImage<vulkano::format::Format>>>> = return;
/// // let render_pass: Arc<_> = ...;
/// let framebuffer = Framebuffer::start(render_pass.clone())
///     .add(view).unwrap()
///     .build().unwrap();
/// ```
///
/// All framebuffer objects implement the `FramebufferAbstract` trait. This means that you can cast
/// any `Arc<Framebuffer<..>>` into an `Arc<FramebufferAbstract + Send + Sync>` for easier storage.
///
/// ## Framebuffer dimensions
///
/// If you use `Framebuffer::start()` to create a framebuffer then vulkano will automatically
/// make sure that all the attachments have the same dimensions, as this is the most common
/// situation.
///
/// Alternatively you can also use `with_intersecting_dimensions`, in which case the dimensions of
/// the framebuffer will be the intersection of the dimensions of all attachments, or
/// `with_dimensions` if you want to specify exact dimensions. If you use `with_dimensions`, you
/// are allowed to attach images that are larger than these dimensions.
///
/// If the dimensions of the framebuffer don't match the dimensions of one of its attachment, then
/// only the top-left hand corner of the image will be drawn to.
///
#[derive(Debug)]
pub struct Framebuffer<A> {
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    framebuffer: ash::vk::Framebuffer,
    dimensions: [u32; 3],
    resources: A,
}

impl Framebuffer<()> {
    /// Starts building a framebuffer.
    pub fn start(render_pass: Arc<RenderPass>) -> FramebufferBuilder<()> {
        FramebufferBuilder {
            render_pass,
            raw_ids: SmallVec::new(),
            dimensions: FramebufferBuilderDimensions::AutoIdentical(None),
            attachments: (),
        }
    }

    /// Starts building a framebuffer. The dimensions of the framebuffer will automatically be
    /// the intersection of the dimensions of all the attachments.
    pub fn with_intersecting_dimensions(render_pass: Arc<RenderPass>) -> FramebufferBuilder<()> {
        FramebufferBuilder {
            render_pass,
            raw_ids: SmallVec::new(),
            dimensions: FramebufferBuilderDimensions::AutoSmaller(None),
            attachments: (),
        }
    }

    /// Starts building a framebuffer.
    pub fn with_dimensions(
        render_pass: Arc<RenderPass>,
        dimensions: [u32; 3],
    ) -> FramebufferBuilder<()> {
        FramebufferBuilder {
            render_pass,
            raw_ids: SmallVec::new(),
            dimensions: FramebufferBuilderDimensions::Specific(dimensions),
            attachments: (),
        }
    }
}

/// Prototype of a framebuffer.
pub struct FramebufferBuilder<A> {
    render_pass: Arc<RenderPass>,
    raw_ids: SmallVec<[ash::vk::ImageView; 8]>,
    dimensions: FramebufferBuilderDimensions,
    attachments: A,
}

impl<A> fmt::Debug for FramebufferBuilder<A>
where
    A: fmt::Debug,
{
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("FramebufferBuilder")
            .field("render_pass", &self.render_pass)
            .field("dimensions", &self.dimensions)
            .field("attachments", &self.attachments)
            .finish()
    }
}

#[derive(Debug)]
enum FramebufferBuilderDimensions {
    AutoIdentical(Option<[u32; 3]>),
    AutoSmaller(Option<[u32; 3]>),
    Specific([u32; 3]),
}

impl<A> FramebufferBuilder<A>
where
    A: AttachmentsList,
{
    /// Appends an attachment to the prototype of the framebuffer.
    ///
    /// Attachments must be added in the same order as the one defined in the render pass.
    pub fn add<T>(
        self,
        attachment: T,
    ) -> Result<FramebufferBuilder<(A, T)>, FramebufferCreationError>
    where
        T: ImageViewAbstract,
    {
        if self.raw_ids.len() >= self.render_pass.desc().attachments().len() {
            return Err(FramebufferCreationError::AttachmentsCountMismatch {
                expected: self.render_pass.desc().attachments().len(),
                obtained: self.raw_ids.len() + 1,
            });
        }

        match ensure_image_view_compatible(self.render_pass.desc(), self.raw_ids.len(), &attachment)
        {
            Ok(()) => (),
            Err(err) => return Err(FramebufferCreationError::IncompatibleAttachment(err)),
        };

        let image_dimensions = attachment.image().dimensions();
        let array_layers = attachment.array_layers();
        debug_assert_eq!(image_dimensions.depth(), 1);

        let view_dimensions = [
            image_dimensions.width(),
            image_dimensions.height(),
            array_layers.end - array_layers.start,
        ];

        let dimensions = match self.dimensions {
            FramebufferBuilderDimensions::AutoIdentical(None) => {
                FramebufferBuilderDimensions::AutoIdentical(Some(view_dimensions))
            }
            FramebufferBuilderDimensions::AutoIdentical(Some(current)) => {
                if view_dimensions != current {
                    return Err(FramebufferCreationError::AttachmentDimensionsIncompatible {
                        expected: current,
                        obtained: view_dimensions,
                    });
                }

                FramebufferBuilderDimensions::AutoIdentical(Some(current))
            }
            FramebufferBuilderDimensions::AutoSmaller(None) => {
                FramebufferBuilderDimensions::AutoSmaller(Some(view_dimensions))
            }
            FramebufferBuilderDimensions::AutoSmaller(Some(current)) => {
                let new_dims = [
                    cmp::min(current[0], view_dimensions[0]),
                    cmp::min(current[1], view_dimensions[1]),
                    cmp::min(current[2], view_dimensions[2]),
                ];

                FramebufferBuilderDimensions::AutoSmaller(Some(new_dims))
            }
            FramebufferBuilderDimensions::Specific(current) => {
                if view_dimensions[0] < current[0]
                    || view_dimensions[1] < current[1]
                    || view_dimensions[2] < current[2]
                {
                    return Err(FramebufferCreationError::AttachmentDimensionsIncompatible {
                        expected: current,
                        obtained: view_dimensions,
                    });
                }

                FramebufferBuilderDimensions::Specific(view_dimensions)
            }
        };

        let mut raw_ids = self.raw_ids;
        raw_ids.push(attachment.inner().internal_object());

        Ok(FramebufferBuilder {
            render_pass: self.render_pass,
            raw_ids,
            dimensions,
            attachments: (self.attachments, attachment),
        })
    }

    /// Turns this builder into a `FramebufferBuilder<Rp, Box<AttachmentsList>>`.
    ///
    /// This allows you to store the builder in situations where you don't know in advance the
    /// number of attachments.
    ///
    /// > **Note**: This is a very rare corner case and you shouldn't have to use this function
    /// > in most situations.
    #[inline]
    pub fn boxed(self) -> FramebufferBuilder<Box<dyn AttachmentsList>>
    where
        A: 'static,
    {
        FramebufferBuilder {
            render_pass: self.render_pass,
            raw_ids: self.raw_ids,
            dimensions: self.dimensions,
            attachments: Box::new(self.attachments) as Box<_>,
        }
    }

    /// Builds the framebuffer.
    pub fn build(self) -> Result<Framebuffer<A>, FramebufferCreationError> {
        let device = self.render_pass.device().clone();

        // Check the number of attachments.
        if self.raw_ids.len() != self.render_pass.desc().attachments().len() {
            return Err(FramebufferCreationError::AttachmentsCountMismatch {
                expected: self.render_pass.desc().attachments().len(),
                obtained: self.raw_ids.len(),
            });
        }

        // Compute the dimensions.
        let dimensions = match self.dimensions {
            FramebufferBuilderDimensions::Specific(dims)
            | FramebufferBuilderDimensions::AutoIdentical(Some(dims))
            | FramebufferBuilderDimensions::AutoSmaller(Some(dims)) => dims,
            FramebufferBuilderDimensions::AutoIdentical(None)
            | FramebufferBuilderDimensions::AutoSmaller(None) => {
                return Err(FramebufferCreationError::CantDetermineDimensions);
            }
        };

        // Checking the dimensions against the limits.
        {
            let properties = device.physical_device().properties();
            let limits = [
                properties.max_framebuffer_width,
                properties.max_framebuffer_height,
                properties.max_framebuffer_layers,
            ];
            if dimensions[0] > limits[0] || dimensions[1] > limits[1] || dimensions[2] > limits[2] {
                return Err(FramebufferCreationError::DimensionsTooLarge);
            }
        }

        let mut layers = 1;

        if let Some(multiview) = self.render_pass.desc().multiview() {
            // There needs to be at least as many layers in the framebuffer
            // as the highest layer that gets referenced by the multiview masking.
            if multiview.highest_used_layer() > dimensions[2] {
                return Err(FramebufferCreationError::InsufficientLayerCount {
                    minimum: multiview.highest_used_layer(),
                    current: dimensions[2],
                });
            }

            // VUID-VkFramebufferCreateInfo-renderPass-02531
            // The framebuffer has to be created with one layer if multiview is enabled even though
            // the underlying images generally have more layers
            // but these layers get used by the multiview functionality.
            if multiview.view_masks.iter().any(|&mask| mask != 0) {
                layers = 1;
            }
        }

        let framebuffer = unsafe {
            let fns = device.fns();

            let infos = ash::vk::FramebufferCreateInfo {
                flags: ash::vk::FramebufferCreateFlags::empty(),
                render_pass: self.render_pass.inner().internal_object(),
                attachment_count: self.raw_ids.len() as u32,
                p_attachments: self.raw_ids.as_ptr(),
                width: dimensions[0],
                height: dimensions[1],
                layers,
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_framebuffer(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Framebuffer {
            device,
            render_pass: self.render_pass,
            framebuffer,
            dimensions,
            resources: self.attachments,
        })
    }
}

impl<A> Framebuffer<A> {
    /// Returns the width, height and layers of this framebuffer.
    #[inline]
    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    /// Returns the width of the framebuffer in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.dimensions[0]
    }

    /// Returns the height of the framebuffer in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.dimensions[1]
    }

    /// Returns the number of layers (or depth) of the framebuffer.
    #[inline]
    pub fn layers(&self) -> u32 {
        self.dimensions[2]
    }

    /// Returns the device that was used to create this framebuffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the renderpass that was used to create this framebuffer.
    #[inline]
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }
}

/// Trait for objects that contain a Vulkan framebuffer object.
///
/// Any `Framebuffer` object implements this trait. You can therefore turn a `Arc<Framebuffer<_>>`
/// into a `Arc<FramebufferAbstract + Send + Sync>` for easier storage.
pub unsafe trait FramebufferAbstract {
    /// Returns an opaque struct that represents the framebuffer's internals.
    fn inner(&self) -> FramebufferSys;

    /// Returns the width, height and array layers of the framebuffer.
    fn dimensions(&self) -> [u32; 3];

    /// Returns the render pass this framebuffer was created for.
    fn render_pass(&self) -> &Arc<RenderPass>;

    /// Returns the attachment of the framebuffer with the given index.
    ///
    /// If the `index` is not between `0` and `num_attachments`, then `None` should be returned.
    fn attached_image_view(&self, index: usize) -> Option<&dyn ImageViewAbstract>;

    /// Returns the width of the framebuffer in pixels.
    #[inline]
    fn width(&self) -> u32 {
        self.dimensions()[0]
    }

    /// Returns the height of the framebuffer in pixels.
    #[inline]
    fn height(&self) -> u32 {
        self.dimensions()[1]
    }

    /// Returns the number of layers (or depth) of the framebuffer.
    #[inline]
    fn layers(&self) -> u32 {
        self.dimensions()[2]
    }
}

unsafe impl<T> FramebufferAbstract for T
where
    T: SafeDeref,
    T::Target: FramebufferAbstract,
{
    #[inline]
    fn inner(&self) -> FramebufferSys {
        FramebufferAbstract::inner(&**self)
    }

    #[inline]
    fn dimensions(&self) -> [u32; 3] {
        (**self).dimensions()
    }

    #[inline]
    fn render_pass(&self) -> &Arc<RenderPass> {
        (**self).render_pass()
    }

    #[inline]
    fn attached_image_view(&self, index: usize) -> Option<&dyn ImageViewAbstract> {
        (**self).attached_image_view(index)
    }
}

unsafe impl<A> FramebufferAbstract for Framebuffer<A>
where
    A: AttachmentsList,
{
    #[inline]
    fn inner(&self) -> FramebufferSys {
        FramebufferSys(self.framebuffer, PhantomData)
    }

    #[inline]
    fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    #[inline]
    fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }

    #[inline]
    fn attached_image_view(&self, index: usize) -> Option<&dyn ImageViewAbstract> {
        self.resources.as_image_view_access(index)
    }
}

unsafe impl<A> DeviceOwned for Framebuffer<A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<A> Drop for Framebuffer<A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0.destroy_framebuffer(
                self.device.internal_object(),
                self.framebuffer,
                ptr::null(),
            );
        }
    }
}

/// Opaque object that represents the internals of a framebuffer.
#[derive(Debug, Copy, Clone)]
pub struct FramebufferSys<'a>(ash::vk::Framebuffer, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for FramebufferSys<'a> {
    type Object = ash::vk::Framebuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::Framebuffer {
        self.0
    }
}

/// Error that can happen when creating a framebuffer object.
#[derive(Copy, Clone, Debug)]
pub enum FramebufferCreationError {
    /// Out of memory.
    OomError(OomError),
    /// The requested dimensions exceed the device's limits.
    DimensionsTooLarge,
    /// The number of minimum layers expected by the render pass exceed the framebuffer layers.
    /// This can happen when the multiview feature is enabled and the specified view or correlation
    /// masks refer to more layers than the framebuffer has.
    InsufficientLayerCount {
        /// Minimum number of layers.
        minimum: u32,
        /// Number of framebuffer layers.
        current: u32,
    },
    /// The attachment has a size that isn't compatible with the requested framebuffer dimensions.
    AttachmentDimensionsIncompatible {
        /// Expected dimensions.
        expected: [u32; 3],
        /// Attachment dimensions.
        obtained: [u32; 3],
    },
    /// The number of attachments doesn't match the number expected by the render pass.
    AttachmentsCountMismatch {
        /// Expected number of attachments.
        expected: usize,
        /// Number of attachments that were given.
        obtained: usize,
    },
    /// One of the images cannot be used as the requested attachment.
    IncompatibleAttachment(IncompatibleRenderPassAttachmentError),
    /// The framebuffer has no attachment and no dimension was specified.
    CantDetermineDimensions,
}

impl From<OomError> for FramebufferCreationError {
    #[inline]
    fn from(err: OomError) -> FramebufferCreationError {
        FramebufferCreationError::OomError(err)
    }
}

impl error::Error for FramebufferCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            FramebufferCreationError::OomError(ref err) => Some(err),
            FramebufferCreationError::IncompatibleAttachment(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for FramebufferCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                FramebufferCreationError::OomError(_) => "no memory available",
                FramebufferCreationError::DimensionsTooLarge => {
                    "the dimensions of the framebuffer are too large"
                }
                FramebufferCreationError::InsufficientLayerCount { .. } => {
                    "the number of minimum layers expected by the render pass exceed the framebuffer layers"
                }
                FramebufferCreationError::AttachmentDimensionsIncompatible { .. } => {
                    "the attachment has a size that isn't compatible with the framebuffer dimensions"
                }
                FramebufferCreationError::AttachmentsCountMismatch { .. } => {
                    "the number of attachments doesn't match the number expected by the render pass"
                }
                FramebufferCreationError::IncompatibleAttachment(_) => {
                    "one of the images cannot be used as the requested attachment"
                }
                FramebufferCreationError::CantDetermineDimensions => {
                    "the framebuffer has no attachment and no dimension was specified"
                }
            }
        )
    }
}

impl From<Error> for FramebufferCreationError {
    #[inline]
    fn from(err: Error) -> FramebufferCreationError {
        FramebufferCreationError::from(OomError::from(err))
    }
}

#[cfg(test)]
mod tests {
    use crate::format::Format;
    use crate::image::attachment::AttachmentImage;
    use crate::image::view::ImageView;
    use crate::render_pass::Framebuffer;
    use crate::render_pass::FramebufferCreationError;
    use crate::render_pass::RenderPass;
    use std::sync::Arc;

    #[test]
    fn simple_create() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let view = ImageView::new(
            AttachmentImage::new(device.clone(), [1024, 768], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let _ = Framebuffer::start(render_pass)
            .add(view)
            .unwrap()
            .build()
            .unwrap();
    }

    #[test]
    fn check_device_limits() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = Arc::new(RenderPass::empty_single_pass(device).unwrap());
        let res = Framebuffer::with_dimensions(rp, [0xffffffff, 0xffffffff, 0xffffffff]).build();
        match res {
            Err(FramebufferCreationError::DimensionsTooLarge) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn attachment_format_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let view = ImageView::new(
            AttachmentImage::new(device.clone(), [1024, 768], Format::R8_UNORM).unwrap(),
        )
        .unwrap();

        match Framebuffer::start(render_pass).add(view) {
            Err(FramebufferCreationError::IncompatibleAttachment(_)) => (),
            _ => panic!(),
        }
    }

    // TODO: check samples mismatch

    #[test]
    fn attachment_dims_larger_than_specified_valid() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let view = ImageView::new(
            AttachmentImage::new(device.clone(), [600, 600], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let _ = Framebuffer::with_dimensions(render_pass, [512, 512, 1])
            .add(view)
            .unwrap()
            .build()
            .unwrap();
    }

    #[test]
    fn attachment_dims_smaller_than_specified() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let view = ImageView::new(
            AttachmentImage::new(device.clone(), [512, 700], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        match Framebuffer::with_dimensions(render_pass, [600, 600, 1]).add(view) {
            Err(FramebufferCreationError::AttachmentDimensionsIncompatible {
                expected,
                obtained,
            }) => {
                assert_eq!(expected, [600, 600, 1]);
                assert_eq!(obtained, [512, 700, 1]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn multi_attachments_dims_not_identical() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
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
                    }
                },
                pass: {
                    color: [a, b],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let a = ImageView::new(
            AttachmentImage::new(device.clone(), [512, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let b = ImageView::new(
            AttachmentImage::new(device.clone(), [512, 513], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        match Framebuffer::start(render_pass).add(a).unwrap().add(b) {
            Err(FramebufferCreationError::AttachmentDimensionsIncompatible {
                expected,
                obtained,
            }) => {
                assert_eq!(expected, [512, 512, 1]);
                assert_eq!(obtained, [512, 513, 1]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn multi_attachments_auto_smaller() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
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
                    }
                },
                pass: {
                    color: [a, b],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let a = ImageView::new(
            AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let b = ImageView::new(
            AttachmentImage::new(device.clone(), [512, 128], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let fb = Framebuffer::with_intersecting_dimensions(render_pass)
            .add(a)
            .unwrap()
            .add(b)
            .unwrap()
            .build()
            .unwrap();

        match (fb.width(), fb.height(), fb.layers()) {
            (256, 128, 1) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn not_enough_attachments() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
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
                    }
                },
                pass: {
                    color: [a, b],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let view = ImageView::new(
            AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let res = Framebuffer::with_intersecting_dimensions(render_pass)
            .add(view)
            .unwrap()
            .build();

        match res {
            Err(FramebufferCreationError::AttachmentsCountMismatch {
                expected: 2,
                obtained: 1,
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn too_many_attachments() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    a: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [a],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        );

        let a = ImageView::new(
            AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();
        let b = ImageView::new(
            AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8_UNORM).unwrap(),
        )
        .unwrap();

        let res = Framebuffer::with_intersecting_dimensions(render_pass)
            .add(a)
            .unwrap()
            .add(b);

        match res {
            Err(FramebufferCreationError::AttachmentsCountMismatch {
                expected: 1,
                obtained: 2,
            }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn empty_working() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = Arc::new(RenderPass::empty_single_pass(device).unwrap());
        let _ = Framebuffer::with_dimensions(rp, [512, 512, 1])
            .build()
            .unwrap();
    }

    #[test]
    fn cant_determine_dimensions_auto() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = Arc::new(RenderPass::empty_single_pass(device).unwrap());
        let res = Framebuffer::start(rp).build();
        match res {
            Err(FramebufferCreationError::CantDetermineDimensions) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn cant_determine_dimensions_intersect() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = Arc::new(RenderPass::empty_single_pass(device).unwrap());
        let res = Framebuffer::with_intersecting_dimensions(rp).build();
        match res {
            Err(FramebufferCreationError::CantDetermineDimensions) => (),
            _ => panic!(),
        }
    }
}
