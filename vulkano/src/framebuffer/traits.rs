// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use format::ClearValue;
use format::Format;
use format::FormatTy;
use framebuffer::UnsafeRenderPass;
use framebuffer::FramebufferCreationError;
use image::Layout as ImageLayout;
use image::traits::Image;
use image::traits::ImageView;
use pipeline::shader::ShaderInterfaceDef;
use sync::AccessFlagBits;
use sync::PipelineStages;

use vk;

/// Trait for objects that describe a render pass.
///
/// # Safety
///
/// This trait is unsafe because:
///
/// - `render_pass` has to return the same `UnsafeRenderPass` every time.
/// - `num_subpasses` has to return a correct value.
///
pub unsafe trait RenderPass: 'static + Send + Sync {
    /// Returns the underlying `UnsafeRenderPass`. Used by vulkano's internals.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn render_pass(&self) -> &UnsafeRenderPass;
}

pub unsafe trait RenderPassDesc {
    /// Iterator returned by the `attachments` method.
    type AttachmentsIter: ExactSizeIterator<Item = LayoutAttachmentDescription>;
    /// Iterator returned by the `passes` method.
    type PassesIter: ExactSizeIterator<Item = LayoutPassDescription>;
    /// Iterator returned by the `dependencies` method.
    type DependenciesIter: ExactSizeIterator<Item = LayoutPassDependencyDescription>;

    /// Returns an iterator that describes the list of attachments of this render pass.
    fn attachments(&self) -> Self::AttachmentsIter;

    /// Returns an iterator that describes the list of passes of this render pass.
    fn passes(&self) -> Self::PassesIter;

    /// Returns an iterator that describes the list of inter-pass dependencies of this render pass.
    fn dependencies(&self) -> Self::DependenciesIter;

    /// Returns the number of subpasses within the render pass.
    #[inline]
    fn num_subpasses(&self) -> u32 {
        self.passes().len() as u32
    }

    /// Returns the number of color attachments in a subpass. Returns `None` if out of range.
    #[inline]
    fn num_color_attachments(&self, subpass: u32) -> Option<u32> {
        self.passes().skip(subpass as usize).next().map(|p| p.color_attachments.len() as u32)
    }

    /// Returns the number of samples of the attachments of a subpass. Returns `None` if out of
    /// range or if the subpass has no attachment. TODO: return an enum instead?
    #[inline]
    fn num_samples(&self, subpass: u32) -> Option<u32> {
        self.passes().skip(subpass as usize).next().and_then(|p| {
            // TODO: chain input attachments as well?
            p.color_attachments.iter().cloned().chain(p.depth_stencil.clone().into_iter())
                               .filter_map(|a| self.attachments().skip(a.0).next())
                               .next().map(|a| a.samples)
        })
    }

    /// Returns a tuple whose first element is `true` if there's a depth attachment, and whose
    /// second element is `true` if there's a stencil attachment. Returns `None` if out of range.
    #[inline]
    fn has_depth_stencil_attachment(&self, subpass: u32) -> Option<(bool, bool)> {
        self.passes().skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, _)) => d,
                None => return (false, false)
            };

            match self.attachments().skip(atch_num).next().unwrap().format.ty() {
                FormatTy::Depth => (true, false),
                FormatTy::Stencil => (false, true),
                FormatTy::DepthStencil => (true, true),
                _ => unreachable!()
            }
        })
    }

    /// Returns true if a subpass has a depth attachment or a depth-stencil attachment.
    #[inline]
    fn has_depth(&self, subpass: u32) -> Option<bool> {
        self.passes().skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, _)) => d,
                None => return false
            };

            match self.attachments().skip(atch_num).next().unwrap().format.ty() {
                FormatTy::Depth => true,
                FormatTy::Stencil => false,
                FormatTy::DepthStencil => true,
                _ => unreachable!()
            }
        })
    }

    /// Returns true if a subpass has a depth attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    fn has_writable_depth(&self, subpass: u32) -> Option<bool> {
        self.passes().skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, l)) => {
                    if l == ImageLayout::DepthStencilReadOnlyOptimal { return false; }
                    d
                },
                None => return false
            };

            match self.attachments().skip(atch_num).next().unwrap().format.ty() {
                FormatTy::Depth => true,
                FormatTy::Stencil => false,
                FormatTy::DepthStencil => true,
                _ => unreachable!()
            }
        })
    }

    /// Returns true if a subpass has a stencil attachment or a depth-stencil attachment.
    #[inline]
    fn has_stencil(&self, subpass: u32) -> Option<bool> {
        self.passes().skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, _)) => d,
                None => return false
            };

            match self.attachments().skip(atch_num).next().unwrap().format.ty() {
                FormatTy::Depth => false,
                FormatTy::Stencil => true,
                FormatTy::DepthStencil => true,
                _ => unreachable!()
            }
        })
    }

    /// Returns true if a subpass has a stencil attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    fn has_writable_stencil(&self, subpass: u32) -> Option<bool> {
        self.passes().skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, l)) => {
                    if l == ImageLayout::DepthStencilReadOnlyOptimal { return false; }
                    d
                },
                None => return false
            };

            match self.attachments().skip(atch_num).next().unwrap().format.ty() {
                FormatTy::Depth => false,
                FormatTy::Stencil => true,
                FormatTy::DepthStencil => true,
                _ => unreachable!()
            }
        })
    }
}

/// Extension trait for `RenderPass`. Defines which types are allowed as an attachments list.
///
/// # Safety
///
/// This trait is unsafe because it's the job of the implementation to check whether the
/// attachments list is correct. What needs to be checked:
///
/// - That the attachments' format and samples count match the render pass layout.
/// - That the attachments have been created with the proper usage flags.
/// - That the attachments only expose one mipmap.
/// - That the attachments use identity components swizzling.
/// TODO: more stuff with aliasing
///
pub unsafe trait RenderPassAttachmentsList<A>: RenderPass {
    /// A decoded `A`.
    // TODO: crappy way to handle this
    type AttachmentsIter: ExactSizeIterator<Item = (Arc<ImageView>, Arc<Image>, ImageLayout, ImageLayout)>;

    /// Decodes a `A` into a list of attachments.
    ///
    /// Returns an error if one of the attachments is wrong.
    fn convert_attachments_list(&self, A) -> Result<Self::AttachmentsIter,
                                                    FramebufferCreationError>;
}

/// Extension trait for `RenderPass`. Defines which types are allowed as a list of clear values.
///
/// # Safety
///
/// This trait is unsafe because vulkano doesn't check whether the clear value is in a format that
/// matches the attachment.
///
pub unsafe trait RenderPassClearValues<C>: RenderPass {
    /// Iterator that produces one clear value per attachment.
    type ClearValuesIter: Iterator<Item = ClearValue>;

    /// Decodes a `C` into a list of clear values where each element corresponds
    /// to an attachment. The size of the returned iterator must be the same as the number of
    /// attachments.
    ///
    /// The format of the clear value **must** match the format of the attachment. Attachments
    /// that are not loaded with `LoadOp::Clear` must have an entry equal to `ClearValue::None`.
    fn convert_clear_values(&self, C) -> Self::ClearValuesIter;
}

/// Extension trait for `RenderPass` that checks whether a subpass of this render pass accepts
/// the output of a fragment shader.
///
/// The trait is automatically implemented for all type that implement `RenderPass` and
/// `RenderPassDesc`.
// TODO: once specialization lands, this trait can be specialized for pairs that are known to
//       always be compatible
pub unsafe trait RenderPassSubpassInterface<Other>: RenderPass where Other: ShaderInterfaceDef {
    /// Returns `true` if this subpass is compatible with the fragment output definition.
    /// Also returns `false` if the subpass is out of range.
    // TODO: return proper error
    fn is_compatible_with(&self, subpass: u32, other: &Other) -> bool;
}

unsafe impl<A, B> RenderPassSubpassInterface<B> for A
    where A: RenderPass + RenderPassDesc, B: ShaderInterfaceDef
{
    fn is_compatible_with(&self, subpass: u32, other: &B) -> bool {
        let pass_descr = match self.passes().skip(subpass as usize).next() {
            Some(s) => s,
            None => return false,
        };

        for element in other.elements() {
            for location in element.location.clone() {
                let attachment_id = match pass_descr.color_attachments.get(location as usize) {
                    Some(a) => a.0,
                    None => return false,
                };

                let attachment_desc = self.attachments().skip(attachment_id).next().unwrap();

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
/// The trait is automatically implemented for all type that implement `RenderPass`.
// TODO: once specialization lands, this trait can be specialized for pairs that are known to
//       always be compatible
// TODO: maybe this can be unimplemented on some pairs, to provide compile-time checks?
pub unsafe trait RenderPassCompatible<Other>: RenderPass where Other: RenderPass {
    /// Returns `true` if this layout is compatible with the other layout, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    // TODO: return proper error
    fn is_compatible_with(&self, other: &Arc<Other>) -> bool;
}

unsafe impl<A, B> RenderPassCompatible<B> for A
    where A: RenderPass, B: RenderPass
{
    fn is_compatible_with(&self, other: &Arc<B>) -> bool {
        // FIXME:
        /*for (atch1, atch2) in self.attachments().zip(other.attachments()) {
            if !atch1.is_compatible_with(&atch2) {
                return false;
            }
        }*/

        return true;

        // FIXME: finish
    }
}

/// Describes an attachment that will be used in a render pass.
#[derive(Debug, Clone)]
pub struct LayoutAttachmentDescription {
    /// Format of the image that is going to be binded.
    pub format: Format,
    /// Number of samples of the image that is going to be binded.
    pub samples: u32,

    /// What the implementation should do with that attachment at the start of the renderpass.
    pub load: LoadOp,
    /// What the implementation should do with that attachment at the end of the renderpass.
    pub store: StoreOp,

    /// Layout that the image is going to be in at the start of the renderpass.
    ///
    /// The vulkano library will automatically switch to the correct layout if necessary, but it
    /// is more optimal to set this to the correct value.
    pub initial_layout: ImageLayout,

    /// Layout that the image will be transitionned to at the end of the renderpass.
    pub final_layout: ImageLayout,
}

impl LayoutAttachmentDescription {
    /// Returns true if this attachment is compatible with another attachment, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    #[inline]
    pub fn is_compatible_with(&self, other: &LayoutAttachmentDescription) -> bool {
        self.format == other.format && self.samples == other.samples
    }
}

/// Describes one of the passes of a render pass.
///
/// # Restrictions
///
/// All these restrictions are checked when the `UnsafeRenderPass` object is created.
/// TODO: that's not the case ^
///
/// - The number of color attachments must be less than the limit of the physical device.
/// - All the attachments in `color_attachments` and `depth_stencil` must have the same
///   samples count.
/// - If any attachment is used as both an input attachment and a color or
///   depth/stencil attachment, then each use must use the same layout.
/// - Elements of `preserve_attachments` must not be used in any of the other members.
/// - If `resolve_attachments` is not empty, then all the resolve attachments must be attachments
///   with 1 sample and all the color attachments must have more than 1 sample.
/// - If `resolve_attachments` is not empty, all the resolve attachments must have the same format
///   as the color attachments.
/// - If the first use of an attachment in this renderpass is as an input attachment and the
///   attachment is not also used as a color or depth/stencil attachment in the same subpass,
///   then the loading operation must not be `Clear`.
///
// TODO: add tests for all these restrictions
// TODO: allow unused attachments (for example attachment 0 and 2 are used, 1 is unused)
#[derive(Debug, Clone)]
pub struct LayoutPassDescription {
    /// Indices and layouts of attachments to use as color attachments.
    pub color_attachments: Vec<(usize, ImageLayout)>,      // TODO: Vec is slow

    /// Index and layout of the attachment to use as depth-stencil attachment.
    pub depth_stencil: Option<(usize, ImageLayout)>,

    /// Indices and layouts of attachments to use as input attachments.
    pub input_attachments: Vec<(usize, ImageLayout)>,      // TODO: Vec is slow

    /// If not empty, each color attachment will be resolved into each corresponding entry of
    /// this list.
    ///
    /// If this value is not empty, it **must** be the same length as `color_attachments`.
    pub resolve_attachments: Vec<(usize, ImageLayout)>,      // TODO: Vec is slow

    /// Indices of attachments that will be preserved during this pass.
    pub preserve_attachments: Vec<usize>,      // TODO: Vec is slow
}

/// Describes a dependency between two passes of a render pass.
///
/// The implementation is allowed to change the order of the passes within a render pass, unless
/// you specify that there exists a dependency between two passes (ie. the result of one will be
/// used as the input of another one).
// FIXME: finish
#[derive(Debug, Clone)]
pub struct LayoutPassDependencyDescription {
    /// Index of the subpass that writes the data that `destination_subpass` is going to use.
    pub source_subpass: usize,

    /// Index of the subpass that reads the data that `source_subpass` wrote.
    pub destination_subpass: usize,

    pub src_stages: PipelineStages,

    pub dst_stages: PipelineStages,

    pub src_access: AccessFlagBits,

    pub dst_access: AccessFlagBits,

    pub by_region: bool,
}

/// Describes what the implementation should do with an attachment after all the subpasses have
/// completed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum StoreOp {
    /// The attachment will be stored. This is what you usually want.
    Store = vk::ATTACHMENT_STORE_OP_STORE,

    /// What happens is implementation-specific.
    ///
    /// This is purely an optimization compared to `Store`. The implementation doesn't need to copy
    /// from the internal cache to the memory, which saves bandwidth.
    ///
    /// This doesn't mean that the data won't be copied, as an implementation is also free to not
    /// use a cache and write the output directly in memory.
    DontCare = vk::ATTACHMENT_STORE_OP_DONT_CARE,
}

/// Describes what the implementation should do with an attachment at the start of the subpass.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LoadOp {
    /// The attachment will be loaded. This is what you want if you want to draw over
    /// something existing.
    Load = vk::ATTACHMENT_LOAD_OP_LOAD,

    /// The attachment will be cleared by the implementation with a uniform value that you must
    /// provide when you start drawing.
    ///
    /// This is what you usually use at the start of a frame, in order to reset the content of
    /// the color, depth and/or stencil buffers.
    ///
    /// See the `draw_inline` and `draw_secondary` methods of `PrimaryComputeBufferBuilder`.
    Clear = vk::ATTACHMENT_LOAD_OP_CLEAR,

    /// The attachment will have undefined content.
    ///
    /// This is what you should use for attachments that you intend to overwrite entirely.
    DontCare = vk::ATTACHMENT_LOAD_OP_DONT_CARE,
}

/// Represents a subpass within a `RenderPass` object.
///
/// This struct doesn't correspond to anything in Vulkan. It is simply an equivalent to a
/// tuple of a render pass and subpass index. Contrary to a tuple, however, the existence of the
/// subpass is checked when the object is created. When you have a `Subpass` you are guaranteed
/// that the given subpass does exist.
///
pub struct Subpass<'a, L: 'a> {
    render_pass: &'a Arc<L>,
    subpass_id: u32,
}

impl<'a, L: 'a> Subpass<'a, L> where L: RenderPass + RenderPassDesc {
    /// Returns a handle that represents a subpass of a render pass.
    #[inline]
    pub fn from(render_pass: &Arc<L>, id: u32) -> Option<Subpass<L>>
        where L: RenderPass
    {
        if id < render_pass.num_subpasses() {
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
        self.render_pass.num_color_attachments(self.subpass_id).unwrap()
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
        self.render_pass.has_writable_depth(self.subpass_id).unwrap()
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
        self.render_pass.has_writable_stencil(self.subpass_id).unwrap()
    }

    /// Returns true if the subpass has any color or depth/stencil attachment.
    #[inline]
    pub fn has_color_or_depth_stencil_attachment(&self) -> bool {
        self.num_color_attachments() >= 1 ||
        self.render_pass.has_depth_stencil_attachment(self.subpass_id).unwrap() != (false, false)
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<u32> {
        self.render_pass.num_samples(self.subpass_id)
    }
}

impl<'a, L: 'a> Subpass<'a, L> {
    /// Returns the render pass of this subpass.
    #[inline]
    pub fn render_pass(&self) -> &'a Arc<L> {
        self.render_pass
    }

    /// Returns the index of this subpass within the renderpass.
    #[inline]
    pub fn index(&self) -> u32 {
        self.subpass_id
    }
}

// We need manual implementations, otherwise Copy/Clone are only implemented if `L` implements
// Copy/Clone.
impl<'a, L: 'a> Copy for Subpass<'a, L> {}
impl<'a, L: 'a> Clone for Subpass<'a, L> {
    #[inline]
    fn clone(&self) -> Subpass<'a, L> {
        Subpass { render_pass: self.render_pass, subpass_id: self.subpass_id }
    }
}
