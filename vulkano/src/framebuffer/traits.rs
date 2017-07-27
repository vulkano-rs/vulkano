// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use device::DeviceOwned;
use format::ClearValue;
use framebuffer::FramebufferSys;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassSys;
use image::ImageViewAccess;
use pipeline::shader::ShaderInterfaceDef;

use SafeDeref;

/// Trait for objects that contain a Vulkan framebuffer object.
///
/// Any `Framebuffer` object implements this trait. You can therefore turn a `Arc<Framebuffer<_>>`
/// into a `Arc<FramebufferAbstract + Send + Sync>` for easier storage.
pub unsafe trait FramebufferAbstract: RenderPassAbstract {
    /// Returns an opaque struct that represents the framebuffer's internals.
    fn inner(&self) -> FramebufferSys;

    /// Returns the width, height and array layers of the framebuffer.
    fn dimensions(&self) -> [u32; 3];

    /// Returns the attachment of the framebuffer with the given index.
    ///
    /// If the `index` is not between `0` and `num_attachments`, then `None` should be returned.
    fn attached_image_view(&self, index: usize) -> Option<&ImageViewAccess>;

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
    where T: SafeDeref,
          T::Target: FramebufferAbstract
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
    fn attached_image_view(&self, index: usize) -> Option<&ImageViewAccess> {
        (**self).attached_image_view(index)
    }
}

/// Trait for objects that contain a Vulkan render pass object.
///
/// Any `RenderPass` object implements this trait. You can therefore turn a `Arc<RenderPass<_>>`
/// into a `Arc<RenderPassAbstract + Send + Sync>` for easier storage.
///
/// The `Arc<RenderPassAbstract + Send + Sync>` accepts a `Vec<ClearValue>` for clear values and a
/// `Vec<Arc<ImageView + Send + Sync>>` for the list of attachments.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use vulkano::framebuffer::EmptySinglePassRenderPassDesc;
/// use vulkano::framebuffer::RenderPass;
/// use vulkano::framebuffer::RenderPassAbstract;
///
/// # let device: Arc<vulkano::device::Device> = return;
/// let render_pass = RenderPass::new(device.clone(), EmptySinglePassRenderPassDesc).unwrap();
///
/// // For easier storage, turn this render pass into a `Arc<RenderPassAbstract + Send + Sync>`.
/// let stored_rp = Arc::new(render_pass) as Arc<RenderPassAbstract + Send + Sync>;
/// ```
pub unsafe trait RenderPassAbstract: DeviceOwned + RenderPassDesc {
    /// Returns an opaque object representing the render pass' internals.
    ///
    /// # Safety
    ///
    /// The trait implementation must return the same value every time.
    fn inner(&self) -> RenderPassSys;
}

unsafe impl<T> RenderPassAbstract for T
    where T: SafeDeref,
          T::Target: RenderPassAbstract
{
    #[inline]
    fn inner(&self) -> RenderPassSys {
        (**self).inner()
    }
}

/// Extension trait for `RenderPassDesc`. Defines which types are allowed as a list of clear values.
///
/// When the user enters a render pass, they need to pass a list of clear values to apply to
/// the attachments of the framebuffer. To do so, the render pass object or the framebuffer
/// (depending on the function you use) must implement `RenderPassDescClearValues<C>` where `C` is
/// the parameter that the user passed. The trait method is then responsible for checking the
/// correctness of these values and turning them into a list that can be processed by vulkano.
pub unsafe trait RenderPassDescClearValues<C> {
    /// Decodes a `C` into a list of clear values where each element corresponds
    /// to an attachment. The size of the returned iterator must be the same as the number of
    /// attachments.
    ///
    /// The format of the clear value **must** match the format of the attachment. Attachments
    /// that are not loaded with `LoadOp::Clear` must have an entry equal to `ClearValue::None`.
    ///
    /// Only the attachments whose `LoadOp` is `Clear` should appear in the list returned by the
    /// method. Other attachments simply should not appear. TODO: check that this is correct.
    /// For example if attachments 1, 2 and 4 are `Clear` and attachments 0 and 3 are `Load`, then
    /// the list returned by the function must have three elements which are the clear values of
    /// attachments 1, 2 and 4.
    ///
    /// # Safety
    ///
    /// This trait is unsafe because vulkano doesn't check whether the clear value is in a format
    /// that matches the attachment.
    ///
    // TODO: meh for boxing
    fn convert_clear_values(&self, C) -> Box<Iterator<Item = ClearValue>>;
}

unsafe impl<T, C> RenderPassDescClearValues<C> for T
    where T: SafeDeref,
          T::Target: RenderPassDescClearValues<C>
{
    #[inline]
    fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
        (**self).convert_clear_values(vals)
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
    where Other: ShaderInterfaceDef
{
    /// Returns `true` if this subpass is compatible with the fragment output definition.
    /// Also returns `false` if the subpass is out of range.
    // TODO: return proper error
    fn is_compatible_with(&self, subpass: u32, other: &Other) -> bool;
}

unsafe impl<A, B: ?Sized> RenderPassSubpassInterface<B> for A
    where A: RenderPassDesc,
          B: ShaderInterfaceDef
{
    fn is_compatible_with(&self, subpass: u32, other: &B) -> bool {
        let pass_descr = match RenderPassDesc::subpass_descs(self)
            .skip(subpass as usize)
            .next() {
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
    where Other: RenderPassDesc
{
    /// Returns `true` if this layout is compatible with the other layout, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    // TODO: return proper error
    fn is_compatible_with(&self, other: &Other) -> bool;
}

unsafe impl<A: ?Sized, B: ?Sized> RenderPassCompatible<B> for A
    where A: RenderPassDesc,
          B: RenderPassDesc
{
    fn is_compatible_with(&self, other: &B) -> bool {
        if self.num_attachments() != other.num_attachments() {
            return false;
        }

        for atch_num in 0 .. self.num_attachments() {
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

/// Represents a subpass within a `RenderPassAbstract` object.
///
/// This struct doesn't correspond to anything in Vulkan. It is simply an equivalent to a
/// tuple of a render pass and subpass index. Contrary to a tuple, however, the existence of the
/// subpass is checked when the object is created. When you have a `Subpass` you are guaranteed
/// that the given subpass does exist.
#[derive(Debug, Copy, Clone)]
pub struct Subpass<L> {
    render_pass: L,
    subpass_id: u32,
}

impl<L> Subpass<L>
    where L: RenderPassDesc
{
    /// Returns a handle that represents a subpass of a render pass.
    #[inline]
    pub fn from(render_pass: L, id: u32) -> Option<Subpass<L>> {
        if (id as usize) < render_pass.num_subpasses() {
            Some(Subpass {
                     render_pass: render_pass,
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
        self.num_color_attachments() >= 1 ||
            self.render_pass
                .has_depth_stencil_attachment(self.subpass_id)
                .unwrap() != (false, false)
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<u32> {
        self.render_pass.num_samples(self.subpass_id)
    }
}

impl<L> Subpass<L> {
    /// Returns the render pass of this subpass.
    #[inline]
    pub fn render_pass(&self) -> &L {
        &self.render_pass
    }

    /// Returns the index of this subpass within the renderpass.
    #[inline]
    pub fn index(&self) -> u32 {
        self.subpass_id
    }
}

impl<L> Into<(L, u32)> for Subpass<L> {
    #[inline]
    fn into(self) -> (L, u32) {
        (self.render_pass, self.subpass_id)
    }
}
