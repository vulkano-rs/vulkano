// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::FormatTy;
use crate::framebuffer::AttachmentDescription;
use crate::framebuffer::FramebufferSys;
use crate::framebuffer::PassDescription;
use crate::framebuffer::RenderPass;
use crate::image::view::ImageViewAbstract;
use crate::image::ImageLayout;
use crate::pipeline::shader::ShaderInterfaceDef;
use crate::SafeDeref;
use std::sync::Arc;

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

/// Represents a subpass within a `RenderPass` object.
///
/// This struct doesn't correspond to anything in Vulkan. It is simply an equivalent to a
/// tuple of a render pass and subpass index. Contrary to a tuple, however, the existence of the
/// subpass is checked when the object is created. When you have a `Subpass` you are guaranteed
/// that the given subpass does exist.
#[derive(Debug, Clone)]
pub struct Subpass {
    render_pass: Arc<RenderPass>,
    subpass_id: u32,
}

impl Subpass {
    /// Returns a handle that represents a subpass of a render pass.
    #[inline]
    pub fn from(render_pass: Arc<RenderPass>, id: u32) -> Option<Subpass> {
        if (id as usize) < render_pass.desc().subpasses().len() {
            Some(Subpass {
                render_pass,
                subpass_id: id,
            })
        } else {
            None
        }
    }

    #[inline]
    fn subpass_desc(&self) -> &PassDescription {
        &self.render_pass.desc().subpasses()[self.subpass_id as usize]
    }

    #[inline]
    fn attachment(&self, atch_num: usize) -> &AttachmentDescription {
        &self.render_pass.desc().attachments()[atch_num]
    }

    /// Returns the number of color attachments in this subpass.
    #[inline]
    pub fn num_color_attachments(&self) -> u32 {
        self.subpass_desc().color_attachments.len() as u32
    }

    /// Returns true if the subpass has a depth attachment or a depth-stencil attachment.
    #[inline]
    pub fn has_depth(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        let atch_num = match subpass_desc.depth_stencil {
            Some((d, _)) => d,
            None => return false,
        };

        match self.attachment(atch_num).format.ty() {
            FormatTy::Depth => true,
            FormatTy::Stencil => false,
            FormatTy::DepthStencil => true,
            _ => unreachable!(),
        }
    }

    /// Returns true if the subpass has a depth attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    pub fn has_writable_depth(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        let atch_num = match subpass_desc.depth_stencil {
            Some((d, l)) => {
                if l == ImageLayout::DepthStencilReadOnlyOptimal {
                    return false;
                }
                d
            }
            None => return false,
        };

        match self.attachment(atch_num).format.ty() {
            FormatTy::Depth => true,
            FormatTy::Stencil => false,
            FormatTy::DepthStencil => true,
            _ => unreachable!(),
        }
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment.
    #[inline]
    pub fn has_stencil(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        let atch_num = match subpass_desc.depth_stencil {
            Some((d, _)) => d,
            None => return false,
        };

        match self.attachment(atch_num).format.ty() {
            FormatTy::Depth => false,
            FormatTy::Stencil => true,
            FormatTy::DepthStencil => true,
            _ => unreachable!(),
        }
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    pub fn has_writable_stencil(&self) -> bool {
        let subpass_desc = self.subpass_desc();

        let atch_num = match subpass_desc.depth_stencil {
            Some((d, l)) => {
                if l == ImageLayout::DepthStencilReadOnlyOptimal {
                    return false;
                }
                d
            }
            None => return false,
        };

        match self.attachment(atch_num).format.ty() {
            FormatTy::Depth => false,
            FormatTy::Stencil => true,
            FormatTy::DepthStencil => true,
            _ => unreachable!(),
        }
    }

    /// Returns true if the subpass has any color or depth/stencil attachment.
    #[inline]
    pub fn has_color_or_depth_stencil_attachment(&self) -> bool {
        if self.num_color_attachments() >= 1 {
            return true;
        }

        let subpass_desc = self.subpass_desc();
        match subpass_desc.depth_stencil {
            Some((d, _)) => true,
            None => false,
        }
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<u32> {
        let subpass_desc = self.subpass_desc();

        // TODO: chain input attachments as well?
        subpass_desc
            .color_attachments
            .iter()
            .cloned()
            .chain(subpass_desc.depth_stencil.clone().into_iter())
            .filter_map(|a| self.render_pass.desc().attachments().get(a.0))
            .next()
            .map(|a| a.samples)
    }

    /// Returns the render pass of this subpass.
    #[inline]
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }

    /// Returns the index of this subpass within the renderpass.
    #[inline]
    pub fn index(&self) -> u32 {
        self.subpass_id
    }

    /// Returns `true` if this subpass is compatible with the fragment output definition.
    // TODO: return proper error
    pub fn is_compatible_with<S>(&self, shader_interface: &S) -> bool
    where
        S: ShaderInterfaceDef,
    {
        self.render_pass
            .desc()
            .is_compatible_with_shader(self.subpass_id, shader_interface)
    }
}

impl From<Subpass> for (Arc<RenderPass>, u32) {
    #[inline]
    fn from(value: Subpass) -> (Arc<RenderPass>, u32) {
        (value.render_pass, value.subpass_id)
    }
}
