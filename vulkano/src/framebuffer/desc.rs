// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use device::Device;
use format::ClearValue;
use format::Format;
use format::FormatTy;
use framebuffer::RenderPass;
use framebuffer::RenderPassCompatible;
use framebuffer::RenderPassCreationError;
use framebuffer::RenderPassDescClearValues;
use image::ImageLayout;
use sync::AccessFlagBits;
use sync::PipelineStages;

use vk;
use SafeDeref;

/// Trait for objects that contain the description of a render pass.
///
/// See also all the traits whose name start with `RenderPassDesc` (eg. `RenderPassDescAttachments`
/// or TODO: rename existing traits to match this). They are extensions to this trait.
///
/// # Safety
///
/// TODO: finish this section
/// - All color and depth/stencil attachments used by any given subpass must have the same number
///   of samples.
/// - The trait methods should always return the same values, unless you modify the description
///   through a mutable borrow. Once you pass the `RenderPassDesc` object to vulkano, you can still
///   access it through the `RenderPass::desc()` method that returns a shared borrow to the
///   description. It must not be possible for a shared borrow to modify the description in such a
///   way that the description changes.
/// - The provided methods shouldn't be overridden with fancy implementations. For example
///   `build_render_pass` must build a render pass from the description and not a different one.
///
pub unsafe trait RenderPassDesc: RenderPassDescClearValues<Vec<ClearValue>> {
    /// Returns the number of attachments of the render pass.
    fn num_attachments(&self) -> usize;

    /// Returns the description of an attachment.
    ///
    /// Returns `None` if `num` is greater than or equal to `num_attachments()`.
    fn attachment_desc(&self, num: usize) -> Option<AttachmentDescription>;

    /// Returns an iterator to the list of attachments.
    #[inline]
    fn attachment_descs(&self) -> RenderPassDescAttachments<Self>
    where
        Self: Sized,
    {
        RenderPassDescAttachments {
            render_pass: self,
            num: 0,
        }
    }

    /// Returns the number of subpasses of the render pass.
    fn num_subpasses(&self) -> usize;

    /// Returns the description of a subpass.
    ///
    /// Returns `None` if `num` is greater than or equal to `num_subpasses()`.
    fn subpass_desc(&self, num: usize) -> Option<PassDescription>;

    /// Returns an iterator to the list of subpasses.
    #[inline]
    fn subpass_descs(&self) -> RenderPassDescSubpasses<Self>
    where
        Self: Sized,
    {
        RenderPassDescSubpasses {
            render_pass: self,
            num: 0,
        }
    }

    /// Returns the number of dependencies of the render pass.
    fn num_dependencies(&self) -> usize;

    /// Returns the description of a dependency.
    ///
    /// Returns `None` if `num` is greater than or equal to `num_dependencies()`.
    fn dependency_desc(&self, num: usize) -> Option<PassDependencyDescription>;

    /// Returns an iterator to the list of dependencies.
    #[inline]
    fn dependency_descs(&self) -> RenderPassDescDependencies<Self>
    where
        Self: Sized,
    {
        RenderPassDescDependencies {
            render_pass: self,
            num: 0,
        }
    }

    /// Returns true if this render pass is compatible with another render pass.
    ///
    /// Two render passes that contain one subpass are compatible if they are identical. Two render
    /// passes that contain more than one subpass are compatible if they are identical except for
    /// the load/store operations and the image layouts.
    ///
    /// This function is just a shortcut for the `RenderPassCompatible` trait.
    #[inline]
    fn is_compatible_with<T>(&self, other: &T) -> bool
    where
        Self: Sized,
        T: ?Sized + RenderPassDesc,
    {
        RenderPassCompatible::is_compatible_with(self, other)
    }

    /// Builds a render pass from this description.
    ///
    /// > **Note**: This function is just a shortcut for `RenderPass::new`.
    #[inline]
    fn build_render_pass(
        self,
        device: Arc<Device>,
    ) -> Result<RenderPass<Self>, RenderPassCreationError>
    where
        Self: Sized,
    {
        RenderPass::new(device, self)
    }

    /// Returns the number of color attachments of a subpass. Returns `None` if out of range.
    #[inline]
    fn num_color_attachments(&self, subpass: u32) -> Option<u32> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .map(|p| p.color_attachments.len() as u32)
    }

    /// Returns the number of samples of the attachments of a subpass. Returns `None` if out of
    /// range or if the subpass has no attachment. TODO: return an enum instead?
    #[inline]
    fn num_samples(&self, subpass: u32) -> Option<u32> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .and_then(|p| {
                // TODO: chain input attachments as well?
                p.color_attachments
                    .iter()
                    .cloned()
                    .chain(p.depth_stencil.clone().into_iter())
                    .filter_map(|a| (&self).attachment_descs().skip(a.0).next())
                    .next()
                    .map(|a| a.samples)
            })
    }

    /// Returns a tuple whose first element is `true` if there's a depth attachment, and whose
    /// second element is `true` if there's a stencil attachment. Returns `None` if out of range.
    #[inline]
    fn has_depth_stencil_attachment(&self, subpass: u32) -> Option<(bool, bool)> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .map(|p| {
                let atch_num = match p.depth_stencil {
                    Some((d, _)) => d,
                    None => return (false, false),
                };

                match (&self)
                    .attachment_descs()
                    .skip(atch_num)
                    .next()
                    .unwrap()
                    .format
                    .ty()
                {
                    FormatTy::Depth => (true, false),
                    FormatTy::Stencil => (false, true),
                    FormatTy::DepthStencil => (true, true),
                    _ => unreachable!(),
                }
            })
    }

    /// Returns true if a subpass has a depth attachment or a depth-stencil attachment.
    #[inline]
    fn has_depth(&self, subpass: u32) -> Option<bool> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .map(|p| {
                let atch_num = match p.depth_stencil {
                    Some((d, _)) => d,
                    None => return false,
                };

                match (&self)
                    .attachment_descs()
                    .skip(atch_num)
                    .next()
                    .unwrap()
                    .format
                    .ty()
                {
                    FormatTy::Depth => true,
                    FormatTy::Stencil => false,
                    FormatTy::DepthStencil => true,
                    _ => unreachable!(),
                }
            })
    }

    /// Returns true if a subpass has a depth attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    fn has_writable_depth(&self, subpass: u32) -> Option<bool> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .map(|p| {
                let atch_num = match p.depth_stencil {
                    Some((d, l)) => {
                        if l == ImageLayout::DepthStencilReadOnlyOptimal {
                            return false;
                        }
                        d
                    }
                    None => return false,
                };

                match (&self)
                    .attachment_descs()
                    .skip(atch_num)
                    .next()
                    .unwrap()
                    .format
                    .ty()
                {
                    FormatTy::Depth => true,
                    FormatTy::Stencil => false,
                    FormatTy::DepthStencil => true,
                    _ => unreachable!(),
                }
            })
    }

    /// Returns true if a subpass has a stencil attachment or a depth-stencil attachment.
    #[inline]
    fn has_stencil(&self, subpass: u32) -> Option<bool> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .map(|p| {
                let atch_num = match p.depth_stencil {
                    Some((d, _)) => d,
                    None => return false,
                };

                match (&self)
                    .attachment_descs()
                    .skip(atch_num)
                    .next()
                    .unwrap()
                    .format
                    .ty()
                {
                    FormatTy::Depth => false,
                    FormatTy::Stencil => true,
                    FormatTy::DepthStencil => true,
                    _ => unreachable!(),
                }
            })
    }

    /// Returns true if a subpass has a stencil attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    fn has_writable_stencil(&self, subpass: u32) -> Option<bool> {
        (&self)
            .subpass_descs()
            .skip(subpass as usize)
            .next()
            .map(|p| {
                let atch_num = match p.depth_stencil {
                    Some((d, l)) => {
                        if l == ImageLayout::DepthStencilReadOnlyOptimal {
                            return false;
                        }
                        d
                    }
                    None => return false,
                };

                match (&self)
                    .attachment_descs()
                    .skip(atch_num)
                    .next()
                    .unwrap()
                    .format
                    .ty()
                {
                    FormatTy::Depth => false,
                    FormatTy::Stencil => true,
                    FormatTy::DepthStencil => true,
                    _ => unreachable!(),
                }
            })
    }
}

unsafe impl<T> RenderPassDesc for T
where
    T: SafeDeref,
    T::Target: RenderPassDesc,
{
    #[inline]
    fn num_attachments(&self) -> usize {
        (**self).num_attachments()
    }

    #[inline]
    fn attachment_desc(&self, num: usize) -> Option<AttachmentDescription> {
        (**self).attachment_desc(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        (**self).num_subpasses()
    }

    #[inline]
    fn subpass_desc(&self, num: usize) -> Option<PassDescription> {
        (**self).subpass_desc(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        (**self).num_dependencies()
    }

    #[inline]
    fn dependency_desc(&self, num: usize) -> Option<PassDependencyDescription> {
        (**self).dependency_desc(num)
    }
}

/// Iterator to the attachments of a `RenderPassDesc`.
#[derive(Debug, Copy, Clone)]
pub struct RenderPassDescAttachments<'a, R: ?Sized + 'a> {
    render_pass: &'a R,
    num: usize,
}

impl<'a, R: ?Sized + 'a> Iterator for RenderPassDescAttachments<'a, R>
where
    R: RenderPassDesc,
{
    type Item = AttachmentDescription;

    fn next(&mut self) -> Option<AttachmentDescription> {
        if self.num < self.render_pass.num_attachments() {
            let n = self.num;
            self.num += 1;
            Some(
                self.render_pass
                    .attachment_desc(n)
                    .expect("Wrong RenderPassDesc implementation"),
            )
        } else {
            None
        }
    }
}

/// Iterator to the subpasses of a `RenderPassDesc`.
#[derive(Debug, Copy, Clone)]
pub struct RenderPassDescSubpasses<'a, R: ?Sized + 'a> {
    render_pass: &'a R,
    num: usize,
}

impl<'a, R: ?Sized + 'a> Iterator for RenderPassDescSubpasses<'a, R>
where
    R: RenderPassDesc,
{
    type Item = PassDescription;

    fn next(&mut self) -> Option<PassDescription> {
        if self.num < self.render_pass.num_subpasses() {
            let n = self.num;
            self.num += 1;
            Some(
                self.render_pass
                    .subpass_desc(n)
                    .expect("Wrong RenderPassDesc implementation"),
            )
        } else {
            None
        }
    }
}

/// Iterator to the subpass dependencies of a `RenderPassDesc`.
#[derive(Debug, Copy, Clone)]
pub struct RenderPassDescDependencies<'a, R: ?Sized + 'a> {
    render_pass: &'a R,
    num: usize,
}

impl<'a, R: ?Sized + 'a> Iterator for RenderPassDescDependencies<'a, R>
where
    R: RenderPassDesc,
{
    type Item = PassDependencyDescription;

    fn next(&mut self) -> Option<PassDependencyDescription> {
        if self.num < self.render_pass.num_dependencies() {
            let n = self.num;
            self.num += 1;
            Some(
                self.render_pass
                    .dependency_desc(n)
                    .expect("Wrong RenderPassDesc implementation"),
            )
        } else {
            None
        }
    }
}

/// Describes an attachment that will be used in a render pass.
#[derive(Debug, Clone)]
pub struct AttachmentDescription {
    /// Format of the image that is going to be bound.
    pub format: Format,
    /// Number of samples of the image that is going to be bound.
    pub samples: u32,

    /// What the implementation should do with that attachment at the start of the render pass.
    pub load: LoadOp,
    /// What the implementation should do with that attachment at the end of the render pass.
    pub store: StoreOp,

    /// Equivalent of `load` for the stencil component of the attachment, if any. Irrelevant if
    /// there is no stencil component.
    pub stencil_load: LoadOp,
    /// Equivalent of `store` for the stencil component of the attachment, if any. Irrelevant if
    /// there is no stencil component.
    pub stencil_store: StoreOp,

    /// Layout that the image is going to be in at the start of the renderpass.
    ///
    /// The vulkano library will automatically switch to the correct layout if necessary, but it
    /// is more efficient to set this to the correct value.
    pub initial_layout: ImageLayout,

    /// Layout that the image will be transitioned to at the end of the renderpass.
    pub final_layout: ImageLayout,
}

impl AttachmentDescription {
    /// Returns true if this attachment is compatible with another attachment, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    #[inline]
    pub fn is_compatible_with(&self, other: &AttachmentDescription) -> bool {
        self.format == other.format && self.samples == other.samples
    }
}

/// Describes one of the passes of a render pass.
///
/// # Restrictions
///
/// All these restrictions are checked when the `RenderPass` object is created.
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
pub struct PassDescription {
    /// Indices and layouts of attachments to use as color attachments.
    pub color_attachments: Vec<(usize, ImageLayout)>, // TODO: Vec is slow

    /// Index and layout of the attachment to use as depth-stencil attachment.
    pub depth_stencil: Option<(usize, ImageLayout)>,

    /// Indices and layouts of attachments to use as input attachments.
    pub input_attachments: Vec<(usize, ImageLayout)>, // TODO: Vec is slow

    /// If not empty, each color attachment will be resolved into each corresponding entry of
    /// this list.
    ///
    /// If this value is not empty, it **must** be the same length as `color_attachments`.
    pub resolve_attachments: Vec<(usize, ImageLayout)>, // TODO: Vec is slow

    /// Indices of attachments that will be preserved during this pass.
    pub preserve_attachments: Vec<usize>, // TODO: Vec is slow
}

/// Describes a dependency between two passes of a render pass.
///
/// The implementation is allowed to change the order of the passes within a render pass, unless
/// you specify that there exists a dependency between two passes (ie. the result of one will be
/// used as the input of another one).
#[derive(Debug, Clone)]
pub struct PassDependencyDescription {
    /// Index of the subpass that writes the data that `destination_subpass` is going to use.
    pub source_subpass: usize,

    /// Index of the subpass that reads the data that `source_subpass` wrote.
    pub destination_subpass: usize,

    /// The pipeline stages that must be finished on the previous subpass before the destination
    /// subpass can start.
    pub source_stages: PipelineStages,

    /// The pipeline stages of the destination subpass that must wait for the source to be finished.
    /// Stages that are earlier of the stages specified here can start before the source is
    /// finished.
    pub destination_stages: PipelineStages,

    /// The way the source subpass accesses the attachments on which we depend.
    pub source_access: AccessFlagBits,

    /// The way the destination subpass accesses the attachments on which we depend.
    pub destination_access: AccessFlagBits,

    /// If false, then the whole subpass must be finished for the next one to start. If true, then
    /// the implementation can start the new subpass for some given pixels as long as the previous
    /// subpass is finished for these given pixels.
    ///
    /// In other words, if the previous subpass has some side effects on other parts of an
    /// attachment, then you should set it to false.
    ///
    /// Passing `false` is always safer than passing `true`, but in practice you rarely need to
    /// pass `false`.
    pub by_region: bool,
}

/// Describes what the implementation should do with an attachment after all the subpasses have
/// completed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum StoreOp {
    /// The attachment will be stored. This is what you usually want.
    ///
    /// While this is the most intuitive option, it is also slower than `DontCare` because it can
    /// take time to write the data back to memory.
    Store = vk::ATTACHMENT_STORE_OP_STORE,

    /// What happens is implementation-specific.
    ///
    /// This is purely an optimization compared to `Store`. The implementation doesn't need to copy
    /// from the internal cache to the memory, which saves memory bandwidth.
    ///
    /// This doesn't mean that the data won't be copied, as an implementation is also free to not
    /// use a cache and write the output directly in memory. In other words, the content of the
    /// image will be undefined.
    DontCare = vk::ATTACHMENT_STORE_OP_DONT_CARE,
}

/// Describes what the implementation should do with an attachment at the start of the subpass.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LoadOp {
    /// The content of the attachment will be loaded from memory. This is what you want if you want
    /// to draw over something existing.
    ///
    /// While this is the most intuitive option, it is also the slowest because it uses a lot of
    /// memory bandwidth.
    Load = vk::ATTACHMENT_LOAD_OP_LOAD,

    /// The content of the attachment will be filled by the implementation with a uniform value
    /// that you must provide when you start drawing.
    ///
    /// This is what you usually use at the start of a frame, in order to reset the content of
    /// the color, depth and/or stencil buffers.
    ///
    /// See the `draw_inline` and `draw_secondary` methods of `PrimaryComputeBufferBuilder`.
    Clear = vk::ATTACHMENT_LOAD_OP_CLEAR,

    /// The attachment will have undefined content.
    ///
    /// This is what you should use for attachments that you intend to entirely cover with draw
    /// commands.
    /// If you are going to fill the attachment with a uniform value, it is better to use `Clear`
    /// instead.
    DontCare = vk::ATTACHMENT_LOAD_OP_DONT_CARE,
}
