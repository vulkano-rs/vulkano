// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::traits::TrackedBuffer;
use command_buffer::std::ResourcesStates;
use command_buffer::sys::PipelineBarrierBuilder;
use command_buffer::submit::SubmitInfo;
use device::Queue;
use format::ClearValue;
use format::Format;
use format::FormatTy;
use framebuffer::UnsafeRenderPass;
use framebuffer::FramebufferCreationError;
use image::Layout as ImageLayout;
use image::traits::TrackedImage;
use pipeline::shader::ShaderInterfaceDef;
use sync::AccessFlagBits;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

use vk;

use VulkanObject;

pub unsafe trait Framebuffer: VulkanObject<Object = vk::Framebuffer> {
    type RenderPass: RenderPass;

    /// Returns the render pass this framebuffer belongs to.
    fn render_pass(&self) -> &Self::RenderPass;

    /// Returns the width, height and number of layers of the framebuffer.
    fn dimensions(&self) -> [u32; 3];
}

unsafe impl<'a, F> Framebuffer for &'a F where F: Framebuffer {
    type RenderPass = F::RenderPass;

    #[inline]
    fn render_pass(&self) -> &Self::RenderPass {
        (**self).render_pass()
    }

    #[inline]
    fn dimensions(&self) -> [u32; 3] {
        (**self).dimensions()
    }
}

unsafe impl<F> Framebuffer for Arc<F> where F: Framebuffer {
    type RenderPass = F::RenderPass;

    #[inline]
    fn render_pass(&self) -> &Self::RenderPass {
        (**self).render_pass()
    }

    #[inline]
    fn dimensions(&self) -> [u32; 3] {
        (**self).dimensions()
    }
}

// TODO: docs
pub unsafe trait TrackedFramebuffer: Framebuffer {
    type State;
    type Finished;

    type SemaphoresWaitIterator: Iterator<Item = (Arc<Semaphore>, PipelineStages)>;
    type SemaphoresSignalIterator: Iterator<Item = Arc<Semaphore>>;

    /// Extracts the states of the framebuffer's attachments from `states`.
    ///
    /// The return values contains:
    ///
    /// - A state object that contains the transitionned states of the framebuffer's attachments.
    /// - The number of command after which the pipeline barrier (the last element of the tuple)
    ///   must happen.
    /// - A pipeline barrier that transitions the attachments to the correct state.
    ///
    unsafe fn extract_and_transition<S>(&self, num_command: usize, states: &mut S)
                                        -> (Self::State, usize, PipelineBarrierBuilder)
        where S: ResourcesStates;

    unsafe fn extract_buffer_state<B>(&self, &mut Self::State, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer;

    unsafe fn extract_image_state<I>(&self, &mut Self::State, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage;

    fn finish(&self, state: Self::State) -> (Self::Finished, PipelineBarrierBuilder);

    unsafe fn on_submit<F>(&self, state: &Self::Finished, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>;
}

unsafe impl<T> TrackedFramebuffer for Arc<T> where T: TrackedFramebuffer {
    type State = T::State;
    type Finished = T::Finished;

    type SemaphoresWaitIterator = T::SemaphoresWaitIterator;
    type SemaphoresSignalIterator = T::SemaphoresSignalIterator;

    #[inline]
    unsafe fn extract_and_transition<S>(&self, num_command: usize, states: &mut S)
                                        -> (Self::State, usize, PipelineBarrierBuilder)
        where S: ResourcesStates
    {
        (**self).extract_and_transition(num_command, states)
    }

    #[inline]
    unsafe fn extract_buffer_state<B>(&self, state: &mut Self::State, buffer: &B) -> Option<B::CommandListState>
        where B: TrackedBuffer
    {
        (**self).extract_buffer_state(state, buffer)
    }

    #[inline]
    unsafe fn extract_image_state<I>(&self, state: &mut Self::State, image: &I) -> Option<I::CommandListState>
        where I: TrackedImage
    {
        (**self).extract_image_state(state, image)
    }

    #[inline]
    fn finish(&self, state: Self::State) -> (Self::Finished, PipelineBarrierBuilder) {
        (**self).finish(state)
    }

    #[inline]
    unsafe fn on_submit<F>(&self, state: &Self::Finished, queue: &Arc<Queue>, fence: F)
                           -> SubmitInfo<Self::SemaphoresWaitIterator,
                                         Self::SemaphoresSignalIterator>
        where F: FnMut() -> Arc<Fence>
    {
        (**self).on_submit(state, queue, fence)
    }
}

/// Trait for objects that describe a render pass.
///
/// # Safety
///
/// This trait is unsafe because:
///
/// - `render_pass` has to return the same `UnsafeRenderPass` every time.
/// - `num_subpasses` has to return a correct value.
///
pub unsafe trait RenderPass {
    /// Returns the underlying `UnsafeRenderPass`. Used by vulkano's internals.
    fn inner(&self) -> &UnsafeRenderPass;
}

unsafe impl<T> RenderPass for Arc<T> where T: RenderPass {
    #[inline]
    fn inner(&self) -> &UnsafeRenderPass {
        (**self).inner()
    }
}

unsafe impl<'a, T> RenderPass for &'a T where T: RenderPass {
    #[inline]
    fn inner(&self) -> &UnsafeRenderPass {
        (**self).inner()
    }
}

pub unsafe trait RenderPassDesc {
    /// Returns the number of attachments of the render pass.
    fn num_attachments(&self) -> usize;
    /// Returns the description of an attachment.
    fn attachment(&self, num: usize) -> Option<LayoutAttachmentDescription>;

    /// Returns the number of subpasses of the render pass.
    fn num_subpasses(&self) -> usize;
    /// Returns the description of a suvpass.
    fn subpass(&self, num: usize) -> Option<LayoutPassDescription>;

    /// Returns the number of dependencies of the render pass.
    fn num_dependencies(&self) -> usize;
    /// Returns the description of a dependency.
    fn dependency(&self, num: usize) -> Option<LayoutPassDependencyDescription>;

    /// Returns the number of color attachments in a subpass. Returns `None` if out of range.
    #[inline]
    fn num_color_attachments(&self, subpass: u32) -> Option<u32> {
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().map(|p| p.color_attachments.len() as u32)
    }

    /// Returns the number of samples of the attachments of a subpass. Returns `None` if out of
    /// range or if the subpass has no attachment. TODO: return an enum instead?
    #[inline]
    fn num_samples(&self, subpass: u32) -> Option<u32> {
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().and_then(|p| {
            // TODO: chain input attachments as well?
            p.color_attachments.iter().cloned().chain(p.depth_stencil.clone().into_iter())
                               .filter_map(|a| (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(a.0).next())
                               .next().map(|a| a.samples)
        })
    }

    /// Returns a tuple whose first element is `true` if there's a depth attachment, and whose
    /// second element is `true` if there's a stencil attachment. Returns `None` if out of range.
    #[inline]
    fn has_depth_stencil_attachment(&self, subpass: u32) -> Option<(bool, bool)> {
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, _)) => d,
                None => return (false, false)
            };

            match (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(atch_num).next().unwrap().format.ty() {
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
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, _)) => d,
                None => return false
            };

            match (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(atch_num).next().unwrap().format.ty() {
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
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, l)) => {
                    if l == ImageLayout::DepthStencilReadOnlyOptimal { return false; }
                    d
                },
                None => return false
            };

            match (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(atch_num).next().unwrap().format.ty() {
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
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, _)) => d,
                None => return false
            };

            match (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(atch_num).next().unwrap().format.ty() {
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
        (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next().map(|p| {
            let atch_num = match p.depth_stencil {
                Some((d, l)) => {
                    if l == ImageLayout::DepthStencilReadOnlyOptimal { return false; }
                    d
                },
                None => return false
            };

            match (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(atch_num).next().unwrap().format.ty() {
                FormatTy::Depth => false,
                FormatTy::Stencil => true,
                FormatTy::DepthStencil => true,
                _ => unreachable!()
            }
        })
    }
}

unsafe impl<T> RenderPassDesc for Arc<T> where T: RenderPassDesc {
    #[inline]
    fn num_attachments(&self) -> usize {
        (**self).num_attachments()
    }

    #[inline]
    fn attachment(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        (**self).attachment(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        (**self).num_subpasses()
    }

    #[inline]
    fn subpass(&self, num: usize) -> Option<LayoutPassDescription> {
        (**self).subpass(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        (**self).num_dependencies()
    }

    #[inline]
    fn dependency(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        (**self).dependency(num)
    }
}

unsafe impl<'a, T> RenderPassDesc for &'a T where T: RenderPassDesc {
    #[inline]
    fn num_attachments(&self) -> usize {
        (**self).num_attachments()
    }

    #[inline]
    fn attachment(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        (**self).attachment(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        (**self).num_subpasses()
    }

    #[inline]
    fn subpass(&self, num: usize) -> Option<LayoutPassDescription> {
        (**self).subpass(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        (**self).num_dependencies()
    }

    #[inline]
    fn dependency(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        (**self).dependency(num)
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
    /// Decodes a `A` into a list of attachments.
    ///
    /// Returns an error if one of the attachments is wrong.
    fn check_attachments_list(&self, &A) -> Result<(), FramebufferCreationError>;
}

unsafe impl<A, Rp> RenderPassAttachmentsList<A> for Arc<Rp>
    where Rp: RenderPassAttachmentsList<A>
{
    #[inline]
    fn check_attachments_list(&self, atch: &A) -> Result<(), FramebufferCreationError> {
        (**self).check_attachments_list(atch)
    }
}

unsafe impl<'a, A, Rp> RenderPassAttachmentsList<A> for &'a Rp
    where Rp: RenderPassAttachmentsList<A>
{
    #[inline]
    fn check_attachments_list(&self, atch: &A) -> Result<(), FramebufferCreationError> {
        (**self).check_attachments_list(atch)
    }
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

unsafe impl<C, Rp> RenderPassClearValues<C> for Arc<Rp> where Rp: RenderPassClearValues<C> {
    type ClearValuesIter = Rp::ClearValuesIter;

    #[inline]
    fn convert_clear_values(&self, values: C) -> Self::ClearValuesIter {
        (**self).convert_clear_values(values)
    }
}

unsafe impl<'a, C, Rp> RenderPassClearValues<C> for &'a Rp where Rp: RenderPassClearValues<C> {
    type ClearValuesIter = Rp::ClearValuesIter;

    #[inline]
    fn convert_clear_values(&self, values: C) -> Self::ClearValuesIter {
        (**self).convert_clear_values(values)
    }
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
        let pass_descr = match (0 .. self.num_subpasses()).map(|p| self.subpass(p).unwrap()).skip(subpass as usize).next() {
            Some(s) => s,
            None => return false,
        };

        for element in other.elements() {
            for location in element.location.clone() {
                let attachment_id = match pass_descr.color_attachments.get(location as usize) {
                    Some(a) => a.0,
                    None => return false,
                };

                let attachment_desc = (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).skip(attachment_id).next().unwrap();

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
        /*for (atch1, atch2) in (0 .. self.num_attachments()).map(|p| self.attachment(p).unwrap()).zip(other.attachments()) {
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
#[derive(Debug, Clone)]
pub struct LayoutPassDependencyDescription {
    /// Index of the subpass that writes the data that `destination_subpass` is going to use.
    pub source_subpass: usize,

    /// Index of the subpass that reads the data that `source_subpass` wrote.
    pub destination_subpass: usize,

    /// The pipeline stages that must be finished on the previous subpass before the destination
    /// subpass can start.
    pub src_stages: PipelineStages,

    /// The pipeline stages of the destination subpass that must wait for the source to be finished.
    /// Stages that are earlier of the stages specified here can start before the source is
    /// finished.
    pub dst_stages: PipelineStages,

    /// The way the source subpass accesses the attachments on which we depend.
    pub src_access: AccessFlagBits,

    /// The way the destination subpass accesses the attachments on which we depend.
    pub dst_access: AccessFlagBits,

    /// If false, then the whole subpass must be finished for the next one to start. If true, then
    /// the implementation can start the new subpass for some given pixels as long as the previous
    /// subpass is finished for these given pixels.
    ///
    /// In other words, if the previous subpass has some side effects on other parts of an
    /// attachment, then you sould set it to false.
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
#[derive(Debug, Copy, Clone)]
pub struct Subpass<L> {
    render_pass: L,
    subpass_id: u32,
}

impl<L> Subpass<L> where L: RenderPass + RenderPassDesc {
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
