// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

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
use framebuffer::LayoutAttachmentDescription;
use framebuffer::LayoutPassDependencyDescription;
use framebuffer::LayoutPassDescription;
use framebuffer::RenderPass;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassClearValues;
use framebuffer::RenderPassDescAttachmentsList;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassSys;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Contains the list of images attached to a render pass.
///
/// This is a structure that you must pass when you start recording draw commands in a
/// command buffer.
///
/// A framebuffer can be used alongside with any other render pass object as long as it is
/// compatible with the render pass that his framebuffer was created with. You can determine
/// whether two renderpass objects are compatible by calling `is_compatible_with`.
///
/// The `Framebuffer` struct should always implement the `FramebufferAbstract` trait. Therefore
/// you can turn any `Arc<Framebuffer<Rp, A>>` into a `Arc<FramebufferAbstract>` if you need to.
pub struct Framebuffer<Rp = Arc<RenderPass>, A = Box<AttachmentsList>> {        // TODO: remove default params
    device: Arc<Device>,
    render_pass: Rp,
    framebuffer: vk::Framebuffer,
    dimensions: [u32; 3],
    resources: A,
}

impl<Rp, A> Framebuffer<Rp, A> {
    /// Builds a new framebuffer.
    ///
    /// The `attachments` parameter depends on which render pass implementation is used.
    pub fn new<Ia>(render_pass: Rp, dimensions: [u32; 3], attachments: Ia)
                   -> Result<Arc<Framebuffer<Rp, A>>, FramebufferCreationError>
        where Rp: RenderPassAbstract + RenderPassDescAttachmentsList<Ia, List = A>,
              A: AttachmentsList,
    {
        let device = render_pass.device().clone();

        // This function call is supposed to check whether the attachments are valid.
        // For more safety, we do some additional `debug_assert`s below.
        let attachments = try!(render_pass.check_attachments_list(attachments));

        // Checking the dimensions against the limits.
        {
            let limits = render_pass.device().physical_device().limits();
            let limits = [limits.max_framebuffer_width(), limits.max_framebuffer_height(),
                          limits.max_framebuffer_layers()];
            if dimensions[0] > limits[0] || dimensions[1] > limits[1] ||
               dimensions[2] > limits[2]
            {
                return Err(FramebufferCreationError::DimensionsTooLarge);
            }
        }

        // Checking the dimensions against the attachments.
        if let Some(dims_constraints) = attachments.intersection_dimensions() {
            if dims_constraints[0] < dimensions[0] || dims_constraints[1] < dimensions[1] ||
               dims_constraints[2] < dimensions[2]
            {
                return Err(FramebufferCreationError::AttachmentTooSmall);
            }
        }

        let ids: SmallVec<[vk::ImageView; 8]> =
            attachments.raw_image_view_handles().into_iter().map(|v| v.internal_object()).collect();

        let framebuffer = unsafe {
            let vk = render_pass.device().pointers();

            let infos = vk::FramebufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                renderPass: render_pass.inner().internal_object(),
                attachmentCount: ids.len() as u32,
                pAttachments: ids.as_ptr(),
                width: dimensions[0],
                height: dimensions[1],
                layers: dimensions[2],
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateFramebuffer(device.internal_object(), &infos,
                                                   ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Framebuffer {
            device: device,
            render_pass: render_pass,
            framebuffer: framebuffer,
            dimensions: dimensions,
            resources: attachments,
        }))
    }

    /// Returns true if this framebuffer can be used with the specified renderpass.
    #[inline]
    pub fn is_compatible_with<R>(&self, render_pass: &R) -> bool
        where R: RenderPassAbstract, Rp: RenderPassAbstract
    {
        unimplemented!()
        // FIXME:
        //self.render_pass.is_compatible_with(render_pass)
    }

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
    where Rp: RenderPassAbstract
{
    #[inline]
    fn inner(&self) -> FramebufferSys {
        FramebufferSys(self.framebuffer, PhantomData)
    }

    #[inline]
    fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }
}

unsafe impl<Rp, A> RenderPassDesc for Framebuffer<Rp, A> where Rp: RenderPassDesc {
    #[inline]
    fn num_attachments(&self) -> usize {
        self.render_pass.num_attachments()
    }

    #[inline]
    fn attachment(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        self.render_pass.attachment(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        self.render_pass.num_subpasses()
    }
    
    #[inline]
    fn subpass(&self, num: usize) -> Option<LayoutPassDescription> {
        self.render_pass.subpass(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        self.render_pass.num_dependencies()
    }

    #[inline]
    fn dependency(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        self.render_pass.dependency(num)
    }
}

unsafe impl<At, Rp, A> RenderPassDescAttachmentsList<At> for Framebuffer<Rp, A>
    where Rp: RenderPassDescAttachmentsList<At>
{
    type List = Rp::List;

    #[inline]
    fn check_attachments_list(&self, atch: At) -> Result<Self::List, FramebufferCreationError> {
        self.render_pass.check_attachments_list(atch)
    }
}

unsafe impl<C, Rp, A> RenderPassClearValues<C> for Framebuffer<Rp, A>
    where Rp: RenderPassClearValues<C>
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FramebufferCreationError {
    /// Out of memory.
    OomError(OomError),
    /// The requested dimensions exceed the device's limits.
    DimensionsTooLarge,
    /// One of the attachments has a component swizzle that is different from identity.
    AttachmentNotIdentitySwizzled,
    /// One of the attachments is too small compared to the requested framebuffer dimensions.
    AttachmentTooSmall,
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
            FramebufferCreationError::AttachmentNotIdentitySwizzled => {
                "one of the attachments has a component swizzle that is different from identity"
            },
            FramebufferCreationError::AttachmentTooSmall => {
                "one of the attachments is too small compared to the requested framebuffer \
                 dimensions"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            FramebufferCreationError::OomError(ref err) => Some(err),
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
