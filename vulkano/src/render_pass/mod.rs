// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Description of the steps of the rendering process, and the images used as input or output.
//!
//! # Render passes and framebuffers
//!
//! There are two concepts in Vulkan:
//!
//! - A *render pass* describes the overall process of drawing a frame. It is subdivided into one
//!   or more subpasses.
//! - A *framebuffer* contains the list of image views that are attached during the drawing of
//!   each subpass.
//!
//! Render passes are typically created at initialization only (for example during a loading
//! screen) because they can be costly, while framebuffers can be created and destroyed either at
//! initialization or during the frame.
//!
//! Consequently you can create graphics pipelines from a render pass object alone.
//! A `Framebuffer` object is only needed when you actually add draw commands to a command buffer.

pub use self::{
    create::RenderPassCreationError,
    framebuffer::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError},
};
use crate::{
    device::{Device, DeviceOwned},
    format::Format,
    image::{ImageAspects, ImageLayout, SampleCount},
    macros::{impl_id_counter, vulkan_bitflags_enum, vulkan_enum},
    shader::ShaderInterface,
    sync::{AccessFlags, DependencyFlags, PipelineStages},
    Version, VulkanObject,
};
use std::{cmp::max, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

#[macro_use]
mod macros;
mod create;
mod framebuffer;

/// An object representing the discrete steps in which rendering is done.
///
/// A render pass in Vulkan is made up of three parts:
/// - A list of attachments, which are image views that are inputs, outputs or intermediate stages
///   in the rendering process.
/// - One or more subpasses, which are the steps in which the rendering process, takes place,
///   and the attachments that are used for each step.
/// - Dependencies, which describe how the input and output data of each subpass is to be passed
///   from one subpass to the next.
///
/// ```
/// use vulkano::render_pass::{RenderPass, RenderPassCreateInfo, SubpassDescription};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let render_pass = RenderPass::new(
///     device.clone(),
///     RenderPassCreateInfo {
///         subpasses: vec![SubpassDescription::default()],
///         ..Default::default()
///     },
/// ).unwrap();
/// ```
///
/// This example creates a render pass with no attachment and one single subpass that doesn't draw
/// on anything. While it's sometimes useful, most of the time it's not what you want.
///
/// The easiest way to create a "real" render pass is to use the `single_pass_renderpass!` macro.
///
/// ```
/// # #[macro_use] extern crate vulkano;
/// # fn main() {
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// use vulkano::format::Format;
///
/// let render_pass = single_pass_renderpass!(
///     device.clone(),
///     attachments: {
///         // `foo` is a custom name we give to the first and only attachment.
///         foo: {
///             load: Clear,
///             store: Store,
///             format: Format::R8G8B8A8_UNORM,
///             samples: 1,
///         },
///     },
///     pass: {
///         color: [foo],       // Repeat the attachment name here.
///         depth_stencil: {},
///     },
/// )
/// .unwrap();
/// # }
/// ```
///
/// See the documentation of the macro for more details. TODO: put link here
#[derive(Debug)]
pub struct RenderPass {
    handle: ash::vk::RenderPass,
    device: Arc<Device>,
    id: NonZeroU64,

    attachments: Vec<AttachmentDescription>,
    subpasses: Vec<SubpassDescription>,
    dependencies: Vec<SubpassDependency>,
    correlated_view_masks: Vec<u32>,

    attachment_uses: Vec<Option<AttachmentUse>>,
    granularity: [u32; 2],
    views_used: u32,
}

impl RenderPass {
    /// Creates a new `RenderPass`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.subpasses` is empty.
    /// - Panics if any element of `create_info.attachments` has a `format` of `None`.
    pub fn new(
        device: Arc<Device>,
        mut create_info: RenderPassCreateInfo,
    ) -> Result<Arc<RenderPass>, RenderPassCreationError> {
        Self::validate(&device, &mut create_info)?;

        let handle = unsafe {
            if device.api_version() >= Version::V1_2
                || device.enabled_extensions().khr_create_renderpass2
            {
                Self::create_v2(&device, &create_info)?
            } else {
                Self::create_v1(&device, &create_info)?
            }
        };

        unsafe { Ok(Self::from_handle(device, handle, create_info)) }
    }

    /// Builds a render pass with one subpass and no attachment.
    ///
    /// This method is useful for quick tests.
    #[inline]
    pub fn empty_single_pass(
        device: Arc<Device>,
    ) -> Result<Arc<RenderPass>, RenderPassCreationError> {
        RenderPass::new(
            device,
            RenderPassCreateInfo {
                subpasses: vec![SubpassDescription::default()],
                ..Default::default()
            },
        )
    }

    /// Creates a new `RenderPass` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::RenderPass,
        create_info: RenderPassCreateInfo,
    ) -> Arc<RenderPass> {
        let RenderPassCreateInfo {
            attachments,
            subpasses,
            dependencies,
            correlated_view_masks,
            _ne: _,
        } = create_info;

        let granularity = Self::get_granularity(&device, handle);
        let mut attachment_uses: Vec<Option<AttachmentUse>> = vec![None; attachments.len()];
        let mut views_used = 0;

        for (index, subpass_desc) in subpasses.iter().enumerate() {
            let index = index as u32;
            let &SubpassDescription {
                view_mask,
                ref input_attachments,
                ref color_attachments,
                ref resolve_attachments,
                ref depth_stencil_attachment,
                ..
            } = subpass_desc;

            for atch_ref in (input_attachments.iter().flatten())
                .chain(color_attachments.iter().flatten())
                .chain(resolve_attachments.iter().flatten())
                .chain(depth_stencil_attachment.iter())
            {
                match &mut attachment_uses[atch_ref.attachment as usize] {
                    Some(attachment_use) => attachment_use.last_use_subpass = index,
                    attachment_use @ None => {
                        *attachment_use = Some(AttachmentUse {
                            first_use_subpass: index,
                            last_use_subpass: index,
                        })
                    }
                }
            }

            views_used = max(views_used, u32::BITS - view_mask.leading_zeros());
        }

        Arc::new(RenderPass {
            handle,
            device,
            id: Self::next_id(),

            attachments,
            subpasses,
            dependencies,
            correlated_view_masks,

            attachment_uses,
            granularity,
            views_used,
        })
    }

    unsafe fn get_granularity(device: &Arc<Device>, handle: ash::vk::RenderPass) -> [u32; 2] {
        let fns = device.fns();
        let mut out = MaybeUninit::uninit();
        (fns.v1_0.get_render_area_granularity)(device.handle(), handle, out.as_mut_ptr());

        let out = out.assume_init();
        debug_assert_ne!(out.width, 0);
        debug_assert_ne!(out.height, 0);
        [out.width, out.height]
    }

    /// Returns the attachments of the render pass.
    #[inline]
    pub fn attachments(&self) -> &[AttachmentDescription] {
        &self.attachments
    }

    /// Returns the subpasses of the render pass.
    #[inline]
    pub fn subpasses(&self) -> &[SubpassDescription] {
        &self.subpasses
    }

    /// Returns the dependencies of the render pass.
    #[inline]
    pub fn dependencies(&self) -> &[SubpassDependency] {
        &self.dependencies
    }

    /// Returns the correlated view masks of the render pass.
    #[inline]
    pub fn correlated_view_masks(&self) -> &[u32] {
        &self.correlated_view_masks
    }

    /// If the render pass has multiview enabled, returns the number of views used by the render
    /// pass. Returns 0 if multiview is not enabled.
    #[inline]
    pub fn views_used(&self) -> u32 {
        self.views_used
    }

    /// Returns the granularity of this render pass.
    ///
    /// If the render area of a render pass in a command buffer is a multiple of this granularity,
    /// then the performance will be optimal. Performances are always optimal for render areas
    /// that cover the whole framebuffer.
    #[inline]
    pub fn granularity(&self) -> [u32; 2] {
        self.granularity
    }

    /// Returns the first subpass of the render pass.
    #[inline]
    pub fn first_subpass(self: Arc<Self>) -> Subpass {
        Subpass {
            render_pass: self,
            subpass_id: 0, // Guaranteed to exist
        }
    }

    /// Returns `true` if this render pass is compatible with the other render pass,
    /// as defined in the [`Render Pass Compatibility` section of the Vulkan specs](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap8.html#renderpass-compatibility).
    pub fn is_compatible_with(&self, other: &RenderPass) -> bool {
        if self == other {
            return true;
        }

        let Self {
            handle: _,
            device: _,
            id: _,
            attachments: attachments1,
            subpasses: subpasses1,
            dependencies: dependencies1,
            correlated_view_masks: correlated_view_masks1,
            attachment_uses: _,
            granularity: _,
            views_used: _,
        } = self;
        let Self {
            handle: _,
            device: _,
            id: _,
            attachments: attachments2,
            subpasses: subpasses2,
            dependencies: dependencies2,
            attachment_uses: _,
            correlated_view_masks: correlated_view_masks2,
            granularity: _,
            views_used: _,
        } = other;

        if attachments1.len() != attachments2.len() {
            return false;
        }

        if !attachments1
            .iter()
            .zip(attachments2)
            .all(|(attachment_desc1, attachment_desc2)| {
                let AttachmentDescription {
                    format: format1,
                    samples: samples1,
                    load_op: _,
                    store_op: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    _ne: _,
                } = attachment_desc1;
                let AttachmentDescription {
                    format: format2,
                    samples: samples2,
                    load_op: _,
                    store_op: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    _ne: _,
                } = attachment_desc2;

                format1 == format2 && samples1 == samples2
            })
        {
            return false;
        }

        let are_atch_refs_compatible = |atch_ref1, atch_ref2| match (atch_ref1, atch_ref2) {
            (None, None) => true,
            (Some(atch_ref1), Some(atch_ref2)) => {
                let &AttachmentReference {
                    attachment: attachment1,
                    layout: _,
                    aspects: aspects1,
                    _ne: _,
                } = atch_ref1;
                let AttachmentDescription {
                    format: format1,
                    samples: samples1,
                    load_op: _,
                    store_op: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    _ne: _,
                } = &attachments1[attachment1 as usize];

                let &AttachmentReference {
                    attachment: attachment2,
                    layout: _,
                    aspects: aspects2,
                    _ne: _,
                } = atch_ref2;
                let AttachmentDescription {
                    format: format2,
                    samples: samples2,
                    load_op: _,
                    store_op: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    _ne: _,
                } = &attachments2[attachment2 as usize];

                format1 == format2 && samples1 == samples2 && aspects1 == aspects2
            }
            _ => false,
        };

        if subpasses1.len() != subpasses2.len() {
            return false;
        }

        if !(subpasses1.iter())
            .zip(subpasses2.iter())
            .all(|(subpass1, subpass2)| {
                let SubpassDescription {
                    view_mask: view_mask1,
                    input_attachments: input_attachments1,
                    color_attachments: color_attachments1,
                    resolve_attachments: resolve_attachments1,
                    depth_stencil_attachment: depth_stencil_attachment1,
                    preserve_attachments: _,
                    _ne: _,
                } = subpass1;
                let SubpassDescription {
                    view_mask: view_mask2,
                    input_attachments: input_attachments2,
                    color_attachments: color_attachments2,
                    resolve_attachments: resolve_attachments2,
                    depth_stencil_attachment: depth_stencil_attachment2,
                    preserve_attachments: _,
                    _ne: _,
                } = subpass2;

                if !(0..max(input_attachments1.len(), input_attachments2.len())).all(|i| {
                    are_atch_refs_compatible(
                        input_attachments1.get(i).and_then(|x| x.as_ref()),
                        input_attachments2.get(i).and_then(|x| x.as_ref()),
                    )
                }) {
                    return false;
                }

                if !(0..max(color_attachments1.len(), color_attachments2.len())).all(|i| {
                    are_atch_refs_compatible(
                        color_attachments1.get(i).and_then(|x| x.as_ref()),
                        color_attachments2.get(i).and_then(|x| x.as_ref()),
                    )
                }) {
                    return false;
                }

                if subpasses1.len() > 1
                    && !(0..max(resolve_attachments1.len(), resolve_attachments2.len())).all(|i| {
                        are_atch_refs_compatible(
                            resolve_attachments1.get(i).and_then(|x| x.as_ref()),
                            resolve_attachments2.get(i).and_then(|x| x.as_ref()),
                        )
                    })
                {
                    return false;
                }

                if !are_atch_refs_compatible(
                    depth_stencil_attachment1.as_ref(),
                    depth_stencil_attachment2.as_ref(),
                ) {
                    return false;
                }

                if view_mask1 != view_mask2 {
                    return false;
                }

                true
            })
        {
            return false;
        }

        if dependencies1 != dependencies2 {
            return false;
        }

        if correlated_view_masks1 != correlated_view_masks2 {
            return false;
        }

        true
    }

    /// Returns `true` if the subpass of this description is compatible with the shader's fragment
    /// output definition.
    pub fn is_compatible_with_shader(
        &self,
        subpass: u32,
        shader_interface: &ShaderInterface,
    ) -> bool {
        let subpass_descr = match self.subpasses.get(subpass as usize) {
            Some(s) => s,
            None => return false,
        };

        for element in shader_interface.elements() {
            assert!(!element.ty.is_64bit); // TODO: implement
            let location_range = element.location..element.location + element.ty.num_locations();

            for location in location_range {
                let attachment_id = match subpass_descr.color_attachments.get(location as usize) {
                    Some(Some(atch_ref)) => atch_ref.attachment,
                    _ => return false,
                };

                let _attachment_desc = &self.attachments[attachment_id as usize];

                // FIXME: compare formats depending on the number of components and data type
                /*if attachment_desc.format != element.format {
                    return false;
                }*/
            }
        }

        true
    }
}

impl Drop for RenderPass {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_render_pass)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for RenderPass {
    type Handle = ash::vk::RenderPass;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RenderPass {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(RenderPass);

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
        if (id as usize) < render_pass.subpasses().len() {
            Some(Subpass {
                render_pass,
                subpass_id: id,
            })
        } else {
            None
        }
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

    /// Returns the subpass description for this subpass.
    #[inline]
    pub fn subpass_desc(&self) -> &SubpassDescription {
        &self.render_pass.subpasses()[self.subpass_id as usize]
    }

    /// Returns whether this subpass is the last one in the render pass. If `true` is returned,
    /// calling `next_subpass` will panic.
    #[inline]
    pub fn is_last_subpass(&self) -> bool {
        self.subpass_id as usize == self.render_pass.subpasses().len() - 1
    }

    /// Advances to the next subpass after this one.
    ///
    /// # Panics
    ///
    /// - Panics if there are no more render passes.
    #[inline]
    pub fn next_subpass(&mut self) {
        let next_id = self.subpass_id + 1;
        assert!((next_id as usize) < self.render_pass.subpasses().len());
        self.subpass_id = next_id;
    }

    #[inline]
    fn attachment_desc(&self, atch_num: u32) -> &AttachmentDescription {
        &self.render_pass.attachments()[atch_num as usize]
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
        let atch_num = match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => atch_ref.attachment,
            None => return false,
        };

        self.attachment_desc(atch_num)
            .format
            .map_or(false, |f| f.aspects().intersects(ImageAspects::DEPTH))
    }

    /// Returns true if the subpass has a depth attachment or a depth-stencil attachment whose
    /// layout does not have a read-only depth layout.
    #[inline]
    pub fn has_writable_depth(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        let atch_num = match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => {
                if matches!(
                    atch_ref.layout,
                    ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                ) {
                    return false;
                }
                atch_ref.attachment
            }
            None => return false,
        };

        self.attachment_desc(atch_num)
            .format
            .map_or(false, |f| f.aspects().intersects(ImageAspects::DEPTH))
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment.
    #[inline]
    pub fn has_stencil(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        let atch_num = match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => atch_ref.attachment,
            None => return false,
        };

        self.attachment_desc(atch_num)
            .format
            .map_or(false, |f| f.aspects().intersects(ImageAspects::STENCIL))
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment whose
    /// layout does not have a read-only stencil layout.
    #[inline]
    pub fn has_writable_stencil(&self) -> bool {
        let subpass_desc = self.subpass_desc();

        let atch_num = match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => {
                if matches!(
                    atch_ref.layout,
                    ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                ) {
                    return false;
                }
                atch_ref.attachment
            }
            None => return false,
        };

        self.attachment_desc(atch_num)
            .format
            .map_or(false, |f| f.aspects().intersects(ImageAspects::STENCIL))
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<SampleCount> {
        let subpass_desc = self.subpass_desc();

        // TODO: chain input attachments as well?
        subpass_desc
            .color_attachments
            .iter()
            .flatten()
            .chain(subpass_desc.depth_stencil_attachment.iter())
            .filter_map(|atch_ref| {
                self.render_pass
                    .attachments()
                    .get(atch_ref.attachment as usize)
            })
            .next()
            .map(|atch_desc| atch_desc.samples)
    }

    /// Returns `true` if this subpass is compatible with the fragment output definition.
    // TODO: return proper error
    #[inline]
    pub fn is_compatible_with(&self, shader_interface: &ShaderInterface) -> bool {
        self.render_pass
            .is_compatible_with_shader(self.subpass_id, shader_interface)
    }

    pub(crate) fn load_op(&self, attachment_index: u32) -> Option<LoadOp> {
        self.render_pass.attachment_uses[attachment_index as usize]
            .as_ref()
            .and_then(|attachment_use| {
                (attachment_use.first_use_subpass == self.subpass_id)
                    .then(|| self.render_pass.attachments[attachment_index as usize].load_op)
            })
    }

    pub(crate) fn store_op(&self, attachment_index: u32) -> Option<StoreOp> {
        self.render_pass.attachment_uses[attachment_index as usize]
            .as_ref()
            .and_then(|attachment_use| {
                (attachment_use.last_use_subpass == self.subpass_id)
                    .then(|| self.render_pass.attachments[attachment_index as usize].store_op)
            })
    }

    pub(crate) fn stencil_load_op(&self, attachment_index: u32) -> Option<LoadOp> {
        self.render_pass.attachment_uses[attachment_index as usize]
            .as_ref()
            .and_then(|attachment_use| {
                (attachment_use.first_use_subpass == self.subpass_id).then(|| {
                    self.render_pass.attachments[attachment_index as usize].stencil_load_op
                })
            })
    }

    pub(crate) fn stencil_store_op(&self, attachment_index: u32) -> Option<StoreOp> {
        self.render_pass.attachment_uses[attachment_index as usize]
            .as_ref()
            .and_then(|attachment_use| {
                (attachment_use.last_use_subpass == self.subpass_id).then(|| {
                    self.render_pass.attachments[attachment_index as usize].stencil_store_op
                })
            })
    }
}

impl From<Subpass> for (Arc<RenderPass>, u32) {
    #[inline]
    fn from(value: Subpass) -> (Arc<RenderPass>, u32) {
        (value.render_pass, value.subpass_id)
    }
}

/// Parameters to create a new `RenderPass`.
#[derive(Clone, Debug)]
pub struct RenderPassCreateInfo {
    /// The attachments available for the render pass.
    ///
    /// The default value is empty.
    pub attachments: Vec<AttachmentDescription>,

    /// The subpasses that make up this render pass.
    ///
    /// A render pass must contain at least one subpass.
    ///
    /// The default value is empty, which must be overridden.
    pub subpasses: Vec<SubpassDescription>,

    /// The dependencies between subpasses.
    ///
    /// The default value is empty.
    pub dependencies: Vec<SubpassDependency>,

    /// If multiview rendering is being used (the subpasses have a nonzero `view_mask`),
    /// this specifies sets of views that may be more efficient to render concurrently, for example
    /// because they show the same geometry from almost the same perspective. This is an
    /// optimization hint to the implementation, and does not affect the final result.
    ///
    /// The value is a bitmask, so that that for example `0b11` means that the first two views are
    /// highly correlated, and `0b101` means the first and third view are highly correlated. Each
    /// view bit must appear in at most one element of the list.
    ///
    /// If multiview rendering is not being used, the value must be empty.
    ///
    /// The default value is empty.
    pub correlated_view_masks: Vec<u32>,

    pub _ne: crate::NonExhaustive,
}

impl Default for RenderPassCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            attachments: Vec::new(),
            subpasses: Vec::new(),
            dependencies: Vec::new(),
            correlated_view_masks: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Describes an attachment that will be used in a render pass.
#[derive(Clone, Copy, Debug)]
pub struct AttachmentDescription {
    /// The format of the image that is going to be bound.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The number of samples of the image that is going to be bound.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub samples: SampleCount,

    /// What the implementation should do with the attachment at the start of the subpass that first
    /// uses it.
    ///
    /// The default value is [`LoadOp::DontCare`].
    pub load_op: LoadOp,

    /// What the implementation should do with the attachment at the end of the subpass that last
    /// uses it.
    ///
    /// The default value is [`StoreOp::DontCare`].
    pub store_op: StoreOp,

    /// The equivalent of `load_op` for the stencil component of the attachment, if any. Irrelevant
    /// if there is no stencil component.
    ///
    /// The default value is [`LoadOp::DontCare`].
    pub stencil_load_op: LoadOp,

    /// The equivalent of `store_op` for the stencil component of the attachment, if any. Irrelevant
    /// if there is no stencil component.
    ///
    /// The default value is [`StoreOp::DontCare`].
    pub stencil_store_op: StoreOp,

    /// The layout that the image must in at the start of the render pass.
    ///
    /// The vulkano library will automatically switch to the correct layout if necessary, but it
    /// is more efficient to set this to the correct value.
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub initial_layout: ImageLayout,

    /// The layout that the image will be transitioned to at the end of the render pass.
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub final_layout: ImageLayout,

    pub _ne: crate::NonExhaustive,
}

impl Default for AttachmentDescription {
    #[inline]
    fn default() -> Self {
        Self {
            format: None,
            samples: SampleCount::Sample1,
            load_op: LoadOp::DontCare,
            store_op: StoreOp::DontCare,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::Undefined,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Describes one of the subpasses of a render pass.
///
/// A subpass can use zero or more attachments of various types. Attachment types of which there can
/// be multiple are listed in a `Vec` in this structure. The index in these `Vec`s corresponds to
/// the index used for that attachment type in the shader.
///
/// If a particular index is not used in the shader, it can be set to `None` in this structure.
/// This is useful if an unused index needs to be skipped but a higher index needs to be specified.
///
/// If an attachment is used more than once, i.e. a given `AttachmentReference::attachment` occurs
/// more than once in the `SubpassDescription`, then their `AttachmentReference::layout` must be
/// the same as well.
#[derive(Debug, Clone)]
pub struct SubpassDescription {
    /// If not `0`, enables multiview rendering, and specifies the view indices that are rendered to
    /// in this subpass. The value is a bitmask, so that that for example `0b11` will draw to the
    /// first two views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, it must be nonzero for all subpasses in the render pass, and the
    /// [`multiview`](crate::device::Features::multiview) feature must be enabled on the device.
    ///
    /// The default value is `0`.
    pub view_mask: u32,

    /// The attachments of the render pass that are to be used as input attachments in this subpass.
    ///
    /// If an attachment is used here for the first time in this render pass, and it's is not also
    /// used as a color or depth/stencil attachment in this subpass, then the attachment's `load_op`
    /// must not be [`LoadOp::Clear`].
    ///
    /// The default value is empty.
    pub input_attachments: Vec<Option<AttachmentReference>>,

    /// The attachments of the render pass that are to be used as color attachments in this subpass.
    ///
    /// The number of color attachments must be less than the
    /// [`max_color_attachments`](crate::device::Properties::max_color_attachments) limit of the
    /// physical device. All color attachments must have the same `samples` value.
    ///
    /// The default value is empty.
    pub color_attachments: Vec<Option<AttachmentReference>>,

    /// The attachments of the render pass that are to be used as resolve attachments in this
    /// subpass.
    ///
    /// This list must either be empty or have the same length as `color_attachments`. If it's not
    /// empty, then each resolve attachment is paired with the color attachment of the same index.
    /// The resolve attachments must all have a `samples` value of [`SampleCount::Sample1`], while
    /// the color attachments must have a `samples` value other than [`SampleCount::Sample1`].
    /// Each resolve attachment must have the same `format` as the corresponding color attachment.
    ///
    /// The default value is empty.
    pub resolve_attachments: Vec<Option<AttachmentReference>>,

    /// The single attachment of the render pass that is to be used as depth-stencil attachment in
    /// this subpass.
    ///
    /// If set to `Some`, the referenced attachment must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub depth_stencil_attachment: Option<AttachmentReference>,

    /// The indices of attachments of the render pass that will be preserved during this subpass.
    ///
    /// The referenced attachments must not be used as any other attachment type in the subpass.
    ///
    /// The default value is empty.
    pub preserve_attachments: Vec<u32>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SubpassDescription {
    #[inline]
    fn default() -> Self {
        Self {
            view_mask: 0,
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            input_attachments: Vec::new(),
            resolve_attachments: Vec::new(),
            preserve_attachments: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A reference in a subpass description to a particular attachment of the render pass.
#[derive(Clone, Debug)]
pub struct AttachmentReference {
    /// The number of the attachment being referred to.
    ///
    /// The default value is `0`.
    pub attachment: u32,

    /// The image layout that the attachment should be transitioned to at the start of the subpass.
    ///
    /// The layout is restricted by the type of attachment that an attachment is being used as. A
    /// full listing of allowed layouts per type can be found in
    /// [the Vulkan specification](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap8.html#attachment-type-imagelayout).
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub layout: ImageLayout,

    /// For references to input attachments, the aspects of the image that should be selected.
    /// For attachment types other than input attachments, the value must be empty.
    ///
    /// If empty, all aspects available in the input attachment's `format` will be selected.
    /// If any fields are set, they must be aspects that are available in the `format` of the
    /// attachment.
    ///
    /// If the value is neither empty nor identical to the aspects of the `format`, the device API
    /// version must be at least 1.1, or either the
    /// [`khr_create_renderpass2`](crate::device::DeviceExtensions::khr_create_renderpass2) or the
    /// [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2) extensions must be
    /// enabled on the device.
    ///
    /// The default value is [`ImageAspects::empty()`].
    pub aspects: ImageAspects,

    pub _ne: crate::NonExhaustive,
}

impl Default for AttachmentReference {
    #[inline]
    fn default() -> Self {
        Self {
            attachment: 0,
            layout: ImageLayout::Undefined,
            aspects: ImageAspects::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A dependency between two subpasses of a render pass.
///
/// The implementation is allowed to change the order of the subpasses within a render pass, unless
/// you specify that there exists a dependency between two subpasses (ie. the result of one will be
/// used as the input of another one). Subpass dependencies work similar to pipeline barriers,
/// except that they operate on whole subpasses instead of individual images.
///
/// If `src_subpass` and `dst_subpass` are equal, then this specifies a
/// [subpass self-dependency](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-barriers-subpass-self-dependencies).
/// The `src_stages` must all be
/// [logically earlier in the pipeline](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-order)
/// than the `dst_stages`, and if they both contain a
/// [framebuffer-space stage](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-framebuffer-regions),
/// then `by_region` must be activated.
///
/// If `src_subpass` or `dst_subpass` are set to `None`, this specifies an external
/// dependency. An external dependency specifies a dependency on commands that were submitted before
/// the render pass instance began (for `src_subpass`), or on commands that will be submitted
/// after the render pass instance ends (for `dst_subpass`). The values must not both be
/// `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubpassDependency {
    /// The index of the subpass that writes the data that `dst_subpass` is going to use.
    ///
    /// `None` specifies an external dependency.
    ///
    /// The default value is `None`.
    pub src_subpass: Option<u32>,

    /// The index of the subpass that reads the data that `src_subpass` wrote.
    ///
    /// `None` specifies an external dependency.
    ///
    /// The default value is `None`.
    pub dst_subpass: Option<u32>,

    /// The pipeline stages that must be finished on `src_subpass` before the
    /// `dst_stages` of `dst_subpass` can start.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The pipeline stages of `dst_subpass` that must wait for the `src_stages` of
    /// `src_subpass` to be finished. Stages that are earlier than the stages specified here can
    /// start before the `src_stages` are finished.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The way `src_subpass` accesses the attachments on which we depend.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The way `dst_subpass` accesses the attachments on which we depend.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub dst_access: AccessFlags,

    /// Dependency flags that modify behavior of the subpass dependency.
    ///
    /// If a `src_subpass` equals `dst_subpass`, then:
    /// - If `src_stages` and `dst_stages` both contain framebuffer-space stages,
    ///   this must include [`BY_REGION`].
    /// - If the subpass's `view_mask` has more than one view,
    ///   this must include [`VIEW_LOCAL`].
    ///
    /// The default value is [`DependencyFlags::empty()`].
    ///
    /// [`BY_REGION`]: crate::sync::DependencyFlags::BY_REGION
    /// [`VIEW_LOCAL`]: crate::sync::DependencyFlags::VIEW_LOCAL
    pub dependency_flags: DependencyFlags,

    /// If multiview rendering is being used (the subpasses have a nonzero `view_mask`), and
    /// `dependency_flags` includes [`VIEW_LOCAL`], specifies an offset relative to the view index
    /// of `dst_subpass`: each view `d` in `dst_subpass` depends on view `d + view_offset` in
    /// `src_subpass`. If the source view index does not exist, the dependency is ignored for
    /// that view.
    ///
    /// If `dependency_flags` does not include [`VIEW_LOCAL`], or if `src_subpass` and
    /// `dst_subpass` are the same, the value must be `0`.
    ///
    /// The default value is `0`.
    ///
    /// [`VIEW_LOCAL`]: crate::sync::DependencyFlags::VIEW_LOCAL
    pub view_offset: i32,

    pub _ne: crate::NonExhaustive,
}

impl Default for SubpassDependency {
    #[inline]
    fn default() -> Self {
        Self {
            src_subpass: None,
            dst_subpass: None,
            src_stages: PipelineStages::empty(),
            dst_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_access: AccessFlags::empty(),
            dependency_flags: DependencyFlags::empty(),
            view_offset: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what the implementation should do with an attachment at the start of the subpass.
    LoadOp = AttachmentLoadOp(i32);

    /// The content of the attachment will be loaded from memory. This is what you want if you want
    /// to draw over something existing.
    ///
    /// While this is the most intuitive option, it is also the slowest because it uses a lot of
    /// memory bandwidth.
    Load = LOAD,

    /// The content of the attachment will be filled by the implementation with a uniform value
    /// that you must provide when you start drawing.
    ///
    /// This is what you usually use at the start of a frame, in order to reset the content of
    /// the color, depth and/or stencil buffers.
    Clear = CLEAR,

    /// The attachment will have undefined content.
    ///
    /// This is what you should use for attachments that you intend to entirely cover with draw
    /// commands.
    /// If you are going to fill the attachment with a uniform value, it is better to use `Clear`
    /// instead.
    DontCare = DONT_CARE,

    /* TODO: enable
    // TODO: document
    None = NONE_EXT {
        device_extensions: [ext_load_store_op_none],
    },*/
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what the implementation should do with an attachment after all the subpasses have
    /// completed.
    StoreOp = AttachmentStoreOp(i32);

    /// The attachment will be stored. This is what you usually want.
    ///
    /// While this is the most intuitive option, it is also slower than `DontCare` because it can
    /// take time to write the data back to memory.
    Store = STORE,

    /// What happens is implementation-specific.
    ///
    /// This is purely an optimization compared to `Store`. The implementation doesn't need to copy
    /// from the internal cache to the memory, which saves memory bandwidth.
    ///
    /// This doesn't mean that the data won't be copied, as an implementation is also free to not
    /// use a cache and write the output directly in memory. In other words, the content of the
    /// image will be undefined.
    DontCare = DONT_CARE,

    /* TODO: enable
    // TODO: document
    None = NONE {
        api_version: V1_3,
        device_extensions: [khr_dynamic_rendering, ext_load_store_op_none, qcom_render_pass_store_ops],
    },*/
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`ResolveMode`] values.
    ResolveModes,

    /// Possible resolve modes for attachments.
    ResolveMode,

    = ResolveModeFlags(u32);

    /// The resolved sample is taken from sample number zero, the other samples are ignored.
    ///
    /// This mode is supported for depth and stencil formats, and for color images with an integer
    /// format.
    SAMPLE_ZERO, SampleZero = SAMPLE_ZERO,

    /// The resolved sample is calculated from the average of the samples.
    ///
    /// This mode is supported for depth formats, and for color images with a non-integer format.
    AVERAGE, Average = AVERAGE,

    /// The resolved sample is calculated from the minimum of the samples.
    ///
    /// This mode is supported for depth and stencil formats only.
    MIN, Min = MIN,

    /// The resolved sample is calculated from the maximum of the samples.
    ///
    /// This mode is supported for depth and stencil formats only.
    MAX, Max = MAX,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AttachmentUse {
    first_use_subpass: u32,
    last_use_subpass: u32,
}

#[cfg(test)]
mod tests {
    use crate::{
        format::Format,
        render_pass::{RenderPass, RenderPassCreationError},
    };

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = RenderPass::empty_single_pass(device).unwrap();
    }

    #[test]
    fn too_many_color_atch() {
        let (device, _) = gfx_dev_and_queue!();

        if device.physical_device().properties().max_color_attachments >= 10 {
            return; // test ignored
        }

        let rp = single_pass_renderpass!(
            device,
            attachments: {
                a1: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a2: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a3: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a4: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a5: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a6: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a7: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a8: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a9: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
                a10: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
            },
            pass: {
                color: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10],
                depth_stencil: {},
            },
        );

        match rp {
            Err(RenderPassCreationError::SubpassMaxColorAttachmentsExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn non_zero_granularity() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = single_pass_renderpass!(
            device,
            attachments: {
                a: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, },
            },
            pass: {
                color: [a],
                depth_stencil: {},
            },
        )
        .unwrap();

        let granularity = rp.granularity();
        assert_ne!(granularity[0], 0);
        assert_ne!(granularity[1], 0);
    }
}
