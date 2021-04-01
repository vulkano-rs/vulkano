// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::framebuffer::FramebufferSys;
use crate::framebuffer::RenderPass;
use crate::framebuffer::RenderPassDesc;
use crate::image::view::ImageViewAbstract;
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

/// Extension trait for `RenderPassDesc` that checks whether a subpass of this render pass accepts
/// the output of a fragment shader.
///
/// The trait is automatically implemented for all type that implement `RenderPassDesc` and
/// `RenderPassDesc`.
///
/// > **Note**: This trait exists so that you can specialize it once specialization lands in Rust.
// TODO: once specialization lands, this trait can be specialized for pairs that are known to
//       always be compatible
pub unsafe trait RenderPassSubpassInterface<Other: ?Sized>: RenderPassDesc
where
    Other: ShaderInterfaceDef,
{
    /// Returns `true` if this subpass is compatible with the fragment output definition.
    /// Also returns `false` if the subpass is out of range.
    // TODO: return proper error
    fn is_compatible_with(&self, subpass: u32, other: &Other) -> bool;
}

unsafe impl<A, B: ?Sized> RenderPassSubpassInterface<B> for A
where
    A: RenderPassDesc,
    B: ShaderInterfaceDef,
{
    fn is_compatible_with(&self, subpass: u32, other: &B) -> bool {
        let pass_descr = match RenderPassDesc::subpass_descs(self)
            .skip(subpass as usize)
            .next()
        {
            Some(s) => s,
            None => return false,
        };

        for element in other.elements() {
            for location in element.location.clone() {
                let attachment_id = match pass_descr.color_attachments.get(location as usize) {
                    Some(a) => a.0,
                    None => return false,
                };

                let attachment_desc = (&self)
                    .attachment_descs()
                    .skip(attachment_id)
                    .next()
                    .unwrap();

                // FIXME: compare formats depending on the number of components and data type
                /*if attachment_desc.format != element.format {
                    return false;
                }*/
            }
        }

        true
    }
}

/// Trait implemented on render pass objects to check whether they are compatible
/// with another render pass.
///
/// The trait is automatically implemented for all type that implement `RenderPassDesc`.
///
/// > **Note**: This trait exists so that you can specialize it once specialization lands in Rust.
// TODO: once specialization lands, this trait can be specialized for pairs that are known to
//       always be compatible
// TODO: maybe this can be unimplemented on some pairs, to provide compile-time checks?
pub unsafe trait RenderPassCompatible<Other: ?Sized>: RenderPassDesc
where
    Other: RenderPassDesc,
{
    /// Returns `true` if this layout is compatible with the other layout, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    // TODO: return proper error
    fn is_compatible_with(&self, other: &Other) -> bool;
}

unsafe impl<A: ?Sized, B: ?Sized> RenderPassCompatible<B> for A
where
    A: RenderPassDesc,
    B: RenderPassDesc,
{
    fn is_compatible_with(&self, other: &B) -> bool {
        if self.num_attachments() != other.num_attachments() {
            return false;
        }

        for atch_num in 0..self.num_attachments() {
            let my_atch = self.attachment_desc(atch_num).unwrap();
            let other_atch = other.attachment_desc(atch_num).unwrap();

            if !my_atch.is_compatible_with(&other_atch) {
                return false;
            }
        }

        return true;

        // FIXME: finish
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
        if (id as usize) < render_pass.num_subpasses() {
            Some(Subpass {
                render_pass,
                subpass_id: id,
            })
        } else {
            None
        }
    }

    /// Returns the number of color attachments in this subpass.
    #[inline]
    pub fn num_color_attachments(&self) -> u32 {
        self.render_pass
            .num_color_attachments(self.subpass_id)
            .unwrap()
    }

    /// Returns true if the subpass has a depth attachment or a depth-stencil attachment.
    #[inline]
    pub fn has_depth(&self) -> bool {
        self.render_pass.has_depth(self.subpass_id).unwrap()
    }

    /// Returns true if the subpass has a depth attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    pub fn has_writable_depth(&self) -> bool {
        self.render_pass
            .has_writable_depth(self.subpass_id)
            .unwrap()
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment.
    #[inline]
    pub fn has_stencil(&self) -> bool {
        self.render_pass.has_stencil(self.subpass_id).unwrap()
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    pub fn has_writable_stencil(&self) -> bool {
        self.render_pass
            .has_writable_stencil(self.subpass_id)
            .unwrap()
    }

    /// Returns true if the subpass has any color or depth/stencil attachment.
    #[inline]
    pub fn has_color_or_depth_stencil_attachment(&self) -> bool {
        self.num_color_attachments() >= 1
            || self
                .render_pass
                .has_depth_stencil_attachment(self.subpass_id)
                .unwrap()
                != (false, false)
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<u32> {
        self.render_pass.num_samples(self.subpass_id)
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
}

impl Into<(Arc<RenderPass>, u32)> for Subpass {
    #[inline]
    fn into(self) -> (Arc<RenderPass>, u32) {
        (self.render_pass, self.subpass_id)
    }
}
