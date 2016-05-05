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
use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use device::Device;
use framebuffer::RenderPass;
use framebuffer::RenderPassAttachmentsList;
use image::Layout as ImageLayout;
use image::traits::Image;
use image::traits::ImageView;

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
pub struct Framebuffer<L> {
    device: Arc<Device>,
    render_pass: Arc<L>,
    framebuffer: vk::Framebuffer,
    dimensions: [u32; 3],
    resources: SmallVec<[(Arc<ImageView>, Arc<Image>, ImageLayout, ImageLayout); 8]>,
}

impl<L> Framebuffer<L> {
    /// Builds a new framebuffer.
    ///
    /// The `attachments` parameter depends on which `RenderPass` implementation is used.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the attachments has a different sample count than what the render pass
    ///   describes.
    /// - Additionally, some methods in the `RenderPassAttachmentsList` implementation may panic
    ///   if you pass invalid attachments.      // TODO: should be error instead
    ///
    pub fn new<'a, A>(render_pass: &Arc<L>, dimensions: [u32; 3],
                      attachments: A) -> Result<Arc<Framebuffer<L>>, FramebufferCreationError>
        where L: RenderPass + RenderPassAttachmentsList<A>
    {
        let vk = render_pass.render_pass().device().pointers();
        let device = render_pass.render_pass().device().clone();

        let attachments = try!(render_pass.convert_attachments_list(attachments))
                                .collect::<SmallVec<[_; 8]>>();

        // Checking the dimensions against the limits.
        {
            let limits = render_pass.render_pass().device().physical_device().limits();
            let limits = [limits.max_framebuffer_width(), limits.max_framebuffer_height(),
                          limits.max_framebuffer_layers()];
            if dimensions[0] > limits[0] || dimensions[1] > limits[1] || dimensions[2] > limits[2] {
                return Err(FramebufferCreationError::DimensionsTooLarge);
            }
        }

        let ids = {
            let mut ids = SmallVec::<[_; 8]>::new();

            for &(ref a, _, _, _) in attachments.iter() {
                debug_assert!(a.identity_swizzle());
                // TODO: add more checks with debug_assert!

                let atch_dims = a.parent().dimensions();
                if atch_dims.width() < dimensions[0] || atch_dims.height() < dimensions[1] ||
                   atch_dims.array_layers() < dimensions[2]      // TODO: wrong, since it must be the array layers of the view and not of the image
                {
                    return Err(FramebufferCreationError::AttachmentTooSmall);
                }

                ids.push(a.inner_view().internal_object());
            }

            ids
        };

        let framebuffer = unsafe {
            let infos = vk::FramebufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                renderPass: render_pass.render_pass().internal_object(),
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
            render_pass: render_pass.clone(),
            framebuffer: framebuffer,
            dimensions: dimensions,
            resources: attachments,
        }))
    }

    /// Returns true if this framebuffer can be used with the specified renderpass.
    #[inline]
    pub fn is_compatible_with<R>(&self, render_pass: &Arc<R>) -> bool
        where R: RenderPass, L: RenderPass
    {
        // FIXME: 
        true
        /*(&*self.renderpass as *const UnsafeRenderPass<L> as usize ==
         &**renderpass as *const UnsafeRenderPass<R> as usize) ||
            self.renderpass.is_compatible_with(renderpass)*/
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
    pub fn render_pass(&self) -> &Arc<L> {
        &self.render_pass
    }

    /// Returns all the resources attached to that framebuffer.
    #[inline]
    pub fn attachments(&self) -> &[(Arc<ImageView>, Arc<Image>, ImageLayout, ImageLayout)] {
        &self.resources
    }
}

unsafe impl<L> VulkanObject for Framebuffer<L> {
    type Object = vk::Framebuffer;

    #[inline]
    fn internal_object(&self) -> vk::Framebuffer {
        self.framebuffer
    }
}

impl<L> Drop for Framebuffer<L> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyFramebuffer(self.device.internal_object(), self.framebuffer, ptr::null());
        }
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

#[cfg(test)]
mod tests {
    use format::R8G8B8A8Unorm;
    use framebuffer::Framebuffer;
    use framebuffer::FramebufferCreationError;
    use image::attachment::AttachmentImage;

    mod example {
        use format::R8G8B8A8Unorm;

        single_pass_renderpass! {
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
        }
    }

    #[test]
    fn simple_create() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = example::CustomRenderPass::new(&device, &example::Formats {
            color: (R8G8B8A8Unorm, 1)
        }).unwrap();

        let image = AttachmentImage::new(&device, [1024, 768], R8G8B8A8Unorm).unwrap();

        let _ = Framebuffer::new(&render_pass, [1024, 768, 1], example::AList {
            color: &image
        }).unwrap();
    }

    #[test]
    fn framebuffer_too_large() {
        let (device, _) = gfx_dev_and_queue!();

        let render_pass = example::CustomRenderPass::new(&device, &example::Formats {
            color: (R8G8B8A8Unorm, 1)
        }).unwrap();

        let image = AttachmentImage::new(&device, [1024, 768], R8G8B8A8Unorm).unwrap();

        let alist = example::AList { color: &image };
        match Framebuffer::new(&render_pass, [0xffffffff, 0xffffffff, 0xffffffff], alist) {
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

        let alist = example::AList { color: &image };
        match Framebuffer::new(&render_pass, [600, 600, 1], alist) {
            Err(FramebufferCreationError::AttachmentTooSmall) => (),
            _ => panic!()
        }
    }
}
