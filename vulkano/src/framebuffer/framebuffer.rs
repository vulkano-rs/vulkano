// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use device::Device;
use device::DeviceOwned;
use format::ClearValue;
use framebuffer::AttachmentsList;
use framebuffer::FramebufferAbstract;
use framebuffer::IncompatibleRenderPassAttachmentError;
use framebuffer::LayoutAttachmentDescription;
use framebuffer::LayoutPassDependencyDescription;
use framebuffer::LayoutPassDescription;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassDescClearValues;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassSys;
use framebuffer::ensure_image_view_compatible;
use image::ImageView;
use image::ImageViewAccess;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Contains the list of images attached to a render pass.
///
/// Creating a framebuffer is done by passing the render pass object, the dimensions of the
/// framebuffer, and the list of attachments to `Framebuffer::new()`.
///
/// Just like all render pass objects implement the `RenderPassAbstract` trait, all framebuffer
/// objects implement the `FramebufferAbstract` trait. This means that you can cast any
/// `Arc<Framebuffer<..>>` into an `Arc<FramebufferAbstract + Send + Sync>` for easier storage.
///
/// ## With a generic list of attachments
///
/// The list of attachments passed to `Framebuffer::new()` can be of various types, but one of the
/// possibilities is to pass an object of type `Vec<Arc<ImageView + Send + Sync>>`.
///
/// > **Note**: If you access a render pass object through the `RenderPassAbstract` trait, passing
/// > a `Vec<Arc<ImageView + Send + Sync>>` is the only possible method.
///
/// The framebuffer constructor will perform various checks to make sure that the number of images
/// is correct and that each image can be used with this render pass.
///
/// ```ignore       // FIXME: unignore
/// # use std::sync::Arc;
/// # use vulkano::framebuffer::RenderPassAbstract;
/// use vulkano::framebuffer::Framebuffer;
///
/// # let render_pass: Arc<RenderPassAbstract + Send + Sync> = return;
/// # let my_image: Arc<vulkano::image::ImageViewAccess> = return;
/// // let render_pass: Arc<RenderPassAbstract + Send + Sync> = ...;
/// let framebuffer = Framebuffer::new(render_pass.clone(), [1024, 768, 1],
///                                    vec![my_image.clone() as Arc<_>]).unwrap();
/// ```
///
/// ## With a specialized list of attachments
///
/// The list of attachments can also be of any type `T`, as long as the render pass description
/// implements the trait `RenderPassDescAttachmentsList<T>`.
///
/// For example if you pass a render pass object that implements
/// `RenderPassDescAttachmentsList<Foo>`, then you can pass a `Foo` as the list of attachments.
///
/// > **Note**: The reason why `Vec<Arc<ImageView + Send + Sync>>` always works (see previous section) is that
/// > render pass descriptions are required to always implement
/// > `RenderPassDescAttachmentsList<Vec<Arc<ImageViewAccess + Send + Sync>>>`.
///
/// When it comes to the `single_pass_renderpass!` and `ordered_passes_renderpass!` macros, you can
/// build a list of attachments by calling `start_attachments()` on the render pass description,
/// which will return an object that has a method whose name is the name of the first attachment
/// and that can be used to specify it. This method will return another object that has a method
/// whose name is the name of the second attachment, and so on. See the documentation of the macros
/// for more details. TODO: put link here
///
/// ```ignore       // FIXME: unignore
/// # #[macro_use] extern crate vulkano;
/// # fn main() {
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// use std::sync::Arc;
/// use vulkano::format::Format;
/// use vulkano::framebuffer::Framebuffer;
///
/// let render_pass = single_pass_renderpass!(device.clone(),
///     attachments: {
///         // `foo` is a custom name we give to the first and only attachment.
///         foo: {
///             load: Clear,
///             store: Store,
///             format: Format::R8G8B8A8Unorm,
///             samples: 1,
///         }
///     },
///     pass: {
///         color: [foo],       // Repeat the attachment name here.
///         depth_stencil: {}
///     }
/// ).unwrap();
///
/// # let my_image: Arc<vulkano::image::ImageViewAccess> = return;
/// let framebuffer = {
///     let atch = render_pass.desc().start_attachments().foo(my_image.clone() as Arc<_>);
///     Framebuffer::new(render_pass, [1024, 768, 1], atch).unwrap()
/// };
/// # }
/// ```
#[derive(Debug)]
pub struct Framebuffer<Rp, A> {
    // TODO: is this field really needed?
    device: Arc<Device>,
    render_pass: Rp,
    framebuffer: vk::Framebuffer,
    dimensions: [u32; 3],
    resources: A,
}

impl<Rp> Framebuffer<Rp, ()> {
    /// Starts building a framebuffer.
    pub fn start(render_pass: Rp) -> FramebufferBuilder<Rp, ()> {
        FramebufferBuilder {
            render_pass: render_pass,
            raw_ids: SmallVec::new(),
            dimensions: FramebufferBuilderDimensions::AutoIdentical,
            num_attachments: 0,
            attachments: (),
        }
    }

    /// Starts building a framebuffer. The dimensions of the framebuffer will automatically be
    /// the intersection of the dimensions of all the attachments.
    pub fn with_intersecting_dimensions(render_pass: Rp) -> FramebufferBuilder<Rp, ()> {
        FramebufferBuilder {
            render_pass: render_pass,
            raw_ids: SmallVec::new(),
            dimensions: FramebufferBuilderDimensions::AutoSmaller(None),
            num_attachments: 0,
            attachments: (),
        }
    }

    /// Starts building a framebuffer.
    pub fn with_dimensions(render_pass: Rp, dimensions: [u32; 3]) -> FramebufferBuilder<Rp, ()> {
        FramebufferBuilder {
            render_pass: render_pass,
            raw_ids: SmallVec::new(),
            dimensions: FramebufferBuilderDimensions::Specific(dimensions),
            num_attachments: 0,
            attachments: (),
        }
    }
}

/// Prototype of a framebuffer.
pub struct FramebufferBuilder<Rp, A> {
    render_pass: Rp,
    raw_ids: SmallVec<[vk::ImageView; 8]>,
    dimensions: FramebufferBuilderDimensions,
    num_attachments: usize,
    attachments: A,
}

enum FramebufferBuilderDimensions {
    AutoIdentical,
    AutoSmaller(Option<[u32; 3]>),
    Specific([u32; 3]),
}

impl<Rp, A> FramebufferBuilder<Rp, A>
    where Rp: RenderPassAbstract,
          A: AttachmentsList,
{
    /// Appends an attachment to the prototype of the framebuffer.
    ///
    /// Attachments must be added in the same order as the one defined in the render pass.
    pub fn add<T>(self, attachment: T)
                  -> Result<FramebufferBuilder<Rp, (A, T::Access)>, FramebufferCreationError>
        where T: ImageView
    {
        let access = attachment.access();

        if self.num_attachments >= self.render_pass.num_attachments() {
            return Err(FramebufferCreationError::AttachmentsCountMismatch {
                expected: self.render_pass.num_attachments(),
                obtained: self.num_attachments,
            });
        }

        match ensure_image_view_compatible(&self.render_pass, self.num_attachments, &access) {
            Ok(()) => (),
            Err(err) => return Err(FramebufferCreationError::IncompatibleAttachment(err))
        };

        let img_dims = access.dimensions();
        debug_assert_eq!(img_dims.depth(), 1);

        let dimensions = match self.dimensions {
            FramebufferBuilderDimensions::AutoIdentical => {
                let dims = [img_dims.width(), img_dims.height(), img_dims.array_layers()];
                FramebufferBuilderDimensions::Specific(dims)
            },
            FramebufferBuilderDimensions::AutoSmaller(None) => {
                let dims = [img_dims.width(), img_dims.height(), img_dims.array_layers()];
                FramebufferBuilderDimensions::AutoSmaller(Some(dims))
            },
            FramebufferBuilderDimensions::AutoSmaller(Some(current)) => {
                let new_dims = [
                    cmp::min(current[0], img_dims.width()),
                    cmp::min(current[1], img_dims.height()),
                    cmp::min(current[2], img_dims.array_layers())
                ];

                FramebufferBuilderDimensions::AutoSmaller(Some(new_dims))
            },
            FramebufferBuilderDimensions::Specific(current) => {
                if img_dims.width() != current[0] || img_dims.height() != current[1] ||
                   img_dims.array_layers() != current[2]
                {
                    return Err(FramebufferCreationError::AttachmentTooSmall);  // TODO: more precise error?
                }

                FramebufferBuilderDimensions::Specific([
                    img_dims.width(),
                    img_dims.height(),
                    img_dims.array_layers()
                ])
            }
        };
        
        let mut raw_ids = self.raw_ids;
        raw_ids.push(access.inner().internal_object());

        Ok(FramebufferBuilder {
            render_pass: self.render_pass,
            raw_ids: raw_ids,
            dimensions: dimensions,
            num_attachments: self.num_attachments + 1,
            attachments: (self.attachments, access),
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
    pub fn boxed(self) -> FramebufferBuilder<Rp, Box<AttachmentsList>>
        where A: 'static
    {
        FramebufferBuilder {
            render_pass: self.render_pass,
            raw_ids: self.raw_ids,
            dimensions: self.dimensions,
            num_attachments: self.num_attachments,
            attachments: Box::new(self.attachments) as Box<_>,
        }
    }

    /// Builds the framebuffer.
    pub fn build(self) -> Result<Framebuffer<Rp, A>, FramebufferCreationError> {
        let device = self.render_pass.device().clone();

        // Check the number of attachments.
        if self.num_attachments != self.render_pass.num_attachments() {
            return Err(FramebufferCreationError::AttachmentsCountMismatch {
                expected: self.render_pass.num_attachments(),
                obtained: self.num_attachments,
            });
        }

        // Compute the dimensions.
        let dimensions = match self.dimensions {
            FramebufferBuilderDimensions::Specific(dims) |
            FramebufferBuilderDimensions::AutoSmaller(Some(dims)) => {
                dims
            },
            FramebufferBuilderDimensions::AutoIdentical |
            FramebufferBuilderDimensions::AutoSmaller(None) => {
                panic!()        // TODO: what if 0 attachment?
            },
        };

        // Checking the dimensions against the limits.
        {
            let limits = device.physical_device().limits();
            let limits = [limits.max_framebuffer_width(), limits.max_framebuffer_height(),
                          limits.max_framebuffer_layers()];
            if dimensions[0] > limits[0] || dimensions[1] > limits[1] ||
               dimensions[2] > limits[2]
            {
                return Err(FramebufferCreationError::DimensionsTooLarge);
            }
        }

        let framebuffer = unsafe {
            let vk = device.pointers();

            let infos = vk::FramebufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                renderPass: self.render_pass.inner().internal_object(),
                attachmentCount: self.raw_ids.len() as u32,
                pAttachments: self.raw_ids.as_ptr(),
                width: dimensions[0],
                height: dimensions[1],
                layers: dimensions[2],
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateFramebuffer(device.internal_object(), &infos,
                                                   ptr::null(), &mut output)));
            output
        };

        Ok(Framebuffer {
            device: device,
            render_pass: self.render_pass,
            framebuffer: framebuffer,
            dimensions: dimensions,
            resources: self.attachments,
        })
    }
}

impl<Rp, A> Framebuffer<Rp, A> {
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
    pub fn render_pass(&self) -> &Rp {
        &self.render_pass
    }
}

unsafe impl<Rp, A> FramebufferAbstract for Framebuffer<Rp, A>
    where Rp: RenderPassAbstract,
          A: AttachmentsList
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
    fn attachments(&self) -> Vec<&ImageViewAccess> {
        self.resources.as_image_view_accesses()
    }
}

unsafe impl<Rp, A> RenderPassDesc for Framebuffer<Rp, A> where Rp: RenderPassDesc {
    #[inline]
    fn num_attachments(&self) -> usize {
        self.render_pass.num_attachments()
    }

    #[inline]
    fn attachment_desc(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        self.render_pass.attachment_desc(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        self.render_pass.num_subpasses()
    }
    
    #[inline]
    fn subpass_desc(&self, num: usize) -> Option<LayoutPassDescription> {
        self.render_pass.subpass_desc(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        self.render_pass.num_dependencies()
    }

    #[inline]
    fn dependency_desc(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        self.render_pass.dependency_desc(num)
    }
}

unsafe impl<C, Rp, A> RenderPassDescClearValues<C> for Framebuffer<Rp, A>
    where Rp: RenderPassDescClearValues<C>
{
    #[inline]
    fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
        self.render_pass.convert_clear_values(vals)
    }
}

unsafe impl<Rp, A> RenderPassAbstract for Framebuffer<Rp, A> where Rp: RenderPassAbstract {
    #[inline]
    fn inner(&self) -> RenderPassSys {
        self.render_pass.inner()
    }
}

unsafe impl<Rp, A> DeviceOwned for Framebuffer<Rp, A> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<Rp, A> Drop for Framebuffer<Rp, A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyFramebuffer(self.device.internal_object(), self.framebuffer, ptr::null());
        }
    }
}

/// Opaque object that represents the internals of a framebuffer.
#[derive(Debug, Copy, Clone)]
pub struct FramebufferSys<'a>(vk::Framebuffer, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for FramebufferSys<'a> {
    type Object = vk::Framebuffer;

    #[inline]
    fn internal_object(&self) -> vk::Framebuffer {
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
    /// One of the attachments is too small compared to the requested framebuffer dimensions.
    AttachmentTooSmall,
    /// The number of attachments doesn't match the number expected by the render pass.
    AttachmentsCountMismatch {
        /// Expected number of attachments.
        expected: usize,
        /// Number of attachments that were given.
        obtained: usize,
    },
    /// One of the images cannot be used as the requested attachment.
    IncompatibleAttachment(IncompatibleRenderPassAttachmentError),
}

impl From<OomError> for FramebufferCreationError {
    #[inline]
    fn from(err: OomError) -> FramebufferCreationError {
        FramebufferCreationError::OomError(err)
    }
}

impl error::Error for FramebufferCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            FramebufferCreationError::OomError(_) => "no memory available",
            FramebufferCreationError::DimensionsTooLarge => "the dimensions of the framebuffer \
                                                             are too large",
            FramebufferCreationError::AttachmentTooSmall => {
                "one of the attachments is too small compared to the requested framebuffer \
                 dimensions"
            },
            FramebufferCreationError::AttachmentsCountMismatch { .. } => {
                "the number of attachments doesn't match the number expected by the render pass"
            },
            FramebufferCreationError::IncompatibleAttachment(_) => {
                "one of the images cannot be used as the requested attachment"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
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
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for FramebufferCreationError {
    #[inline]
    fn from(err: Error) -> FramebufferCreationError {
        FramebufferCreationError::from(OomError::from(err))
    }
}

/* FIXME: restore
#[cfg(test)]
mod tests {
    use format::R8G8B8A8Unorm;
    use framebuffer::Framebuffer;
    use framebuffer::FramebufferCreationError;
    use image::attachment::AttachmentImage;

    #[test]
    fn simple_create() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = single_pass_renderpass! {
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: R8G8B8A8Unorm,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        }.unwrap();

        let image = AttachmentImage::new(&device, [1024, 768], R8G8B8A8Unorm).unwrap();

        let _ = Framebuffer::new(render_pass, [1024, 768, 1], example::AList {
            color: image.clone()
        }).unwrap();
    }

    #[test]
    fn framebuffer_too_large() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = example::CustomRenderPass::new(&device, &example::Formats {
            color: (R8G8B8A8Unorm, 1)
        }).unwrap();

        let image = AttachmentImage::new(&device, [1024, 768], R8G8B8A8Unorm).unwrap();

        let alist = example::AList { color: image.clone() };
        match Framebuffer::new(render_pass, [0xffffffff, 0xffffffff, 0xffffffff], alist) {
            Err(FramebufferCreationError::DimensionsTooLarge) => (),
            _ => panic!()
        }
    }

    #[test]
    fn attachment_too_small() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = example::CustomRenderPass::new(&device, &example::Formats {
            color: (R8G8B8A8Unorm, 1)
        }).unwrap();

        let image = AttachmentImage::new(&device, [512, 512], R8G8B8A8Unorm).unwrap();

        let alist = example::AList { color: image.clone() };
        match Framebuffer::new(render_pass, [600, 600, 1], alist) {
            Err(FramebufferCreationError::AttachmentTooSmall) => (),
            _ => panic!()
        }
    }
}*/
