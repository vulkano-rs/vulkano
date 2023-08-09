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

pub use self::framebuffer::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo};
use crate::{
    device::{Device, DeviceOwned, QueueFlags},
    format::{ClearValueType, Format, FormatFeatures, NumericType},
    image::{ImageAspects, ImageLayout, SampleCount},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
    shader::ShaderInterface,
    sync::{AccessFlags, DependencyFlags, MemoryBarrier, PipelineStages},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ahash::HashMap;
use std::{
    cmp::max,
    collections::hash_map::Entry,
    mem::{replace, MaybeUninit},
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

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
///             format: Format::R8G8B8A8_UNORM,
///             samples: 1,
///             load_op: Clear,
///             store_op: Store,
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
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: RenderPassCreateFlags,
    attachments: Vec<AttachmentDescription>,
    subpasses: Vec<SubpassDescription>,
    dependencies: Vec<SubpassDependency>,
    correlated_view_masks: Vec<u32>,

    attachment_use: Vec<AttachmentUse>,
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
    ) -> Result<Arc<RenderPass>, Validated<VulkanError>> {
        for subpass in create_info.subpasses.iter_mut() {
            for input_attachment in subpass.input_attachments.iter_mut().flatten() {
                if input_attachment.aspects.is_empty() {
                    if let Some(attachment_desc) = create_info
                        .attachments
                        .get(input_attachment.attachment as usize)
                    {
                        input_attachment.aspects = attachment_desc.format.aspects();
                    }
                }
            }
        }

        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &RenderPassCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateRenderPass2-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        mut create_info: RenderPassCreateInfo,
    ) -> Result<Arc<RenderPass>, VulkanError> {
        for subpass in create_info.subpasses.iter_mut() {
            for input_attachment in subpass.input_attachments.iter_mut().flatten() {
                if input_attachment.aspects.is_empty() {
                    if let Some(attachment_desc) = create_info
                        .attachments
                        .get(input_attachment.attachment as usize)
                    {
                        input_attachment.aspects = attachment_desc.format.aspects();
                    }
                }
            }
        }

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

    /// Creates a new `RenderPass` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::RenderPass,
        create_info: RenderPassCreateInfo,
    ) -> Arc<RenderPass> {
        let RenderPassCreateInfo {
            flags,
            attachments,
            subpasses,
            dependencies,
            correlated_view_masks,
            _ne: _,
        } = create_info;

        let mut attachment_use = vec![AttachmentUse::default(); attachments.len()];
        let granularity = Self::get_granularity(&device, handle);
        let mut views_used = 0;

        for subpass_desc in &subpasses {
            let &SubpassDescription {
                flags: _,
                view_mask,
                ref input_attachments,
                ref color_attachments,
                ref color_resolve_attachments,
                ref depth_stencil_attachment,
                ref depth_stencil_resolve_attachment,
                depth_resolve_mode: _,
                stencil_resolve_mode: _,
                preserve_attachments: _,
                _ne: _,
            } = subpass_desc;

            for color_attachment in color_attachments.iter().flatten() {
                attachment_use[color_attachment.attachment as usize].color_attachment = true;
            }

            for color_resolve_attachment in color_resolve_attachments.iter().flatten() {
                attachment_use[color_resolve_attachment.attachment as usize].color_attachment =
                    true;
            }

            if let Some(depth_stencil_attachment) = depth_stencil_attachment {
                attachment_use[depth_stencil_attachment.attachment as usize]
                    .depth_stencil_attachment = true;
            }

            if let Some(depth_stencil_resolve_attachment) = depth_stencil_resolve_attachment {
                attachment_use[depth_stencil_resolve_attachment.attachment as usize]
                    .depth_stencil_attachment = true;
            }

            for input_attachment in input_attachments.iter().flatten() {
                attachment_use[input_attachment.attachment as usize].input_attachment = true;
            }

            views_used = max(views_used, u32::BITS - view_mask.leading_zeros());
        }

        Arc::new(RenderPass {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            attachments,
            subpasses,
            dependencies,
            correlated_view_masks,

            attachment_use,
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

    /// Returns the flags that the render pass was created with.
    #[inline]
    pub fn flags(&self) -> RenderPassCreateFlags {
        self.flags
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

            flags: flags1,
            attachments: attachments1,
            subpasses: subpasses1,
            dependencies: dependencies1,
            correlated_view_masks: correlated_view_masks1,

            attachment_use: _,
            granularity: _,
            views_used: _,
        } = self;
        let Self {
            handle: _,
            device: _,
            id: _,

            flags: flags2,
            attachments: attachments2,
            subpasses: subpasses2,
            dependencies: dependencies2,
            correlated_view_masks: correlated_view_masks2,

            attachment_use: _,
            granularity: _,
            views_used: _,
        } = other;

        if flags1 != flags2 {
            return false;
        }

        if attachments1.len() != attachments2.len() {
            return false;
        }

        if !attachments1
            .iter()
            .zip(attachments2)
            .all(|(attachment_desc1, attachment_desc2)| {
                let AttachmentDescription {
                    flags: flags1,
                    format: format1,
                    samples: samples1,
                    load_op: _,
                    store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    stencil_initial_layout: _,
                    stencil_final_layout: _,
                    _ne: _,
                } = attachment_desc1;
                let AttachmentDescription {
                    flags: flags2,
                    format: format2,
                    samples: samples2,
                    load_op: _,
                    store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    stencil_initial_layout: _,
                    stencil_final_layout: _,
                    _ne: _,
                } = attachment_desc2;

                flags1 == flags2 && format1 == format2 && samples1 == samples2
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
                    stencil_layout: _,
                    aspects: aspects1,
                    _ne: _,
                } = atch_ref1;
                let AttachmentDescription {
                    flags: flags1,
                    format: format1,
                    samples: samples1,
                    load_op: _,
                    store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    stencil_initial_layout: _,
                    stencil_final_layout: _,
                    _ne: _,
                } = &attachments1[attachment1 as usize];

                let &AttachmentReference {
                    attachment: attachment2,
                    layout: _,
                    stencil_layout: _,
                    aspects: aspects2,
                    _ne: _,
                } = atch_ref2;
                let AttachmentDescription {
                    flags: flags2,
                    format: format2,
                    samples: samples2,
                    load_op: _,
                    store_op: _,
                    initial_layout: _,
                    final_layout: _,
                    stencil_load_op: _,
                    stencil_store_op: _,
                    stencil_initial_layout: _,
                    stencil_final_layout: _,
                    _ne: _,
                } = &attachments2[attachment2 as usize];

                flags1 == flags2
                    && format1 == format2
                    && samples1 == samples2
                    && aspects1 == aspects2
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
                    flags: flags1,
                    view_mask: view_mask1,
                    input_attachments: input_attachments1,
                    color_attachments: color_attachments1,
                    color_resolve_attachments: color_resolve_attachments1,
                    depth_stencil_attachment: depth_stencil_attachment1,
                    depth_stencil_resolve_attachment: depth_stencil_resolve_attachment1,
                    depth_resolve_mode: depth_resolve_mode1,
                    stencil_resolve_mode: stencil_resolve_mode1,
                    preserve_attachments: _,
                    _ne: _,
                } = subpass1;
                let SubpassDescription {
                    flags: flags2,
                    view_mask: view_mask2,
                    input_attachments: input_attachments2,
                    color_attachments: color_attachments2,
                    color_resolve_attachments: color_resolve_attachments2,
                    depth_stencil_attachment: depth_stencil_attachment2,
                    depth_stencil_resolve_attachment: depth_stencil_resolve_attachment2,
                    depth_resolve_mode: depth_resolve_mode2,
                    stencil_resolve_mode: stencil_resolve_mode2,
                    preserve_attachments: _,
                    _ne: _,
                } = subpass2;

                if flags1 != flags2 {
                    return false;
                }

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
                    && !(0..max(
                        color_resolve_attachments1.len(),
                        color_resolve_attachments2.len(),
                    ))
                        .all(|i| {
                            are_atch_refs_compatible(
                                color_resolve_attachments1.get(i).and_then(|x| x.as_ref()),
                                color_resolve_attachments2.get(i).and_then(|x| x.as_ref()),
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

                if subpasses1.len() > 1 {
                    if !are_atch_refs_compatible(
                        depth_stencil_resolve_attachment1.as_ref(),
                        depth_stencil_resolve_attachment2.as_ref(),
                    ) {
                        return false;
                    }
                }

                if depth_resolve_mode1 != depth_resolve_mode2 {
                    return false;
                }

                if stencil_resolve_mode1 != stencil_resolve_mode2 {
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
                    Some(Some(attachment_ref)) => attachment_ref.attachment,
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

    /// Returns the number of color attachments in this subpass.
    #[inline]
    pub fn num_color_attachments(&self) -> u32 {
        self.subpass_desc().color_attachments.len() as u32
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<SampleCount> {
        let subpass_desc = self.subpass_desc();

        // TODO: chain input attachments as well?
        (subpass_desc.color_attachments.iter().flatten())
            .chain(subpass_desc.depth_stencil_attachment.iter())
            .filter_map(|attachment_ref| {
                self.render_pass
                    .attachments()
                    .get(attachment_ref.attachment as usize)
            })
            .next()
            .map(|attachment_desc| attachment_desc.samples)
    }

    /// Returns `true` if this subpass is compatible with the fragment output definition.
    // TODO: return proper error
    #[inline]
    pub fn is_compatible_with(&self, shader_interface: &ShaderInterface) -> bool {
        self.render_pass
            .is_compatible_with_shader(self.subpass_id, shader_interface)
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
    /// Additional properties of the render pass.
    ///
    /// The default value is empty.
    pub flags: RenderPassCreateFlags,

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
            flags: RenderPassCreateFlags::empty(),
            attachments: Vec::new(),
            subpasses: Vec::new(),
            dependencies: Vec::new(),
            correlated_view_masks: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl RenderPassCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref attachments,
            ref subpasses,
            ref dependencies,
            ref correlated_view_masks,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkRenderPassCreateInfo2-flags-parameter"])
        })?;

        let mut attachment_potential_format_features =
            vec![FormatFeatures::empty(); attachments.len()];

        for (attachment_index, attachment) in attachments.iter().enumerate() {
            // VUID-VkRenderPassCreateInfo2-pAttachments-parameter
            attachment
                .validate(device)
                .map_err(|err| err.add_context(format!("attachments[{}]", attachment_index)))?;

            let &AttachmentDescription {
                flags: _,
                format,
                samples: _,
                load_op: _,
                store_op: _,
                initial_layout: _,
                final_layout: _,
                stencil_load_op: _,
                stencil_store_op: _,
                stencil_initial_layout: _,
                stencil_final_layout: _,
                _ne: _,
            } = attachment;

            // Safety: attachment has been validated
            attachment_potential_format_features[attachment_index] = unsafe {
                device
                    .physical_device()
                    .format_properties_unchecked(format)
                    .potential_format_features()
            };
        }

        if subpasses.is_empty() {
            return Err(Box::new(ValidationError {
                context: "subpasses".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkRenderPassCreateInfo2-subpassCount-arraylength"],
                ..Default::default()
            }));
        }

        let mut attachment_is_used = vec![false; attachments.len()];

        for (subpass_index, subpass_desc) in subpasses.iter().enumerate() {
            // VUID-VkRenderPassCreateInfo2-pSubpasses-parameter
            subpass_desc
                .validate(device)
                .map_err(|err| err.add_context(format!("subpasses[{}]", subpass_index)))?;

            let &SubpassDescription {
                flags: _,
                view_mask,
                ref input_attachments,
                ref color_attachments,
                ref color_resolve_attachments,
                ref depth_stencil_attachment,
                ref depth_stencil_resolve_attachment,
                depth_resolve_mode: _,
                stencil_resolve_mode: _,
                ref preserve_attachments,
                _ne: _,
            } = subpass_desc;

            if (view_mask != 0) != (subpasses[0].view_mask != 0) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`subpasses[{}].view_mask != 0` does not equal \
                        `subpasses[0].view_mask != 0`",
                        subpass_index
                    )
                    .into(),
                    vuids: &["VUID-VkRenderPassCreateInfo2-viewMask-03058"],
                    ..Default::default()
                }));
            }

            let mut color_samples = None;

            for (ref_index, color_attachment) in color_attachments
                .iter()
                .enumerate()
                .flat_map(|(i, a)| a.as_ref().map(|a| (i, a)))
            {
                let &AttachmentReference {
                    attachment,
                    layout,
                    stencil_layout: _,
                    aspects: _,
                    _ne: _,
                } = color_attachment;

                let attachment_desc = attachments.get(attachment as usize).ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "`subpasses[{0}].color_attachments[{1}].attachment` \
                                is not less than the length of `attachments`",
                            subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-attachment-03051"],
                        ..Default::default()
                    })
                })?;

                let is_first_use = !replace(&mut attachment_is_used[attachment as usize], true);

                if is_first_use
                    && attachment_desc.load_op == AttachmentLoadOp::Clear
                    && matches!(
                        layout,
                        ImageLayout::ShaderReadOnlyOptimal
                            | ImageLayout::DepthStencilReadOnlyOptimal
                            | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    )
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "attachment {0} is first used in \
                            `subpasses[{1}].color_attachments[{2}]`, and \
                            `attachments[{0}].load_op` is `AttachmentLoadOp::Clear`, but \
                            `subpasses[{1}].color_attachments[{2}].layout` \
                            does not have a writable color aspect",
                            attachment, subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-pAttachments-02522"],
                        ..Default::default()
                    }));
                }

                if !attachment_potential_format_features[attachment as usize]
                    .intersects(FormatFeatures::COLOR_ATTACHMENT)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "attachment {0} is used in `subpasses[{1}].color_attachments[{2}]`, \
                            but the potential format features of `attachments[{0}].format` \
                            do not include `FormatFeatures::COLOR_ATTACHMENT`",
                            attachment, subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkSubpassDescription2-pColorAttachments-02898"],
                        ..Default::default()
                    }));
                }

                match color_samples {
                    Some(samples) => {
                        if samples != attachment_desc.samples {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`subpasses[{0}].color_attachments[{1}]` uses \
                                    an attachment with a different number of samples than other \
                                    color and depth/stencil attachments in the subpass",
                                    subpass_index, ref_index
                                )
                                .into(),
                                vuids: &["VUID-VkSubpassDescription2-pColorAttachments-03069"],
                                ..Default::default()
                            }));
                        }
                    }
                    None => color_samples = Some(attachment_desc.samples),
                }

                if let Some(color_resolve_attachment) = color_resolve_attachments
                    .get(ref_index)
                    .and_then(Option::as_ref)
                {
                    let &AttachmentReference {
                        attachment: resolve_attachment,
                        layout: _,
                        stencil_layout: _,
                        aspects: _,
                        _ne: _,
                    } = color_resolve_attachment;

                    let resolve_attachment_desc = attachments
                        .get(resolve_attachment as usize)
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                problem: format!(
                                "`subpasses[{0}].color_resolve_attachments[{1}].attachment` is \
                                not less than the length of `attachments`",
                                subpass_index, ref_index
                            )
                                .into(),
                                vuids: &["VUID-VkRenderPassCreateInfo2-attachment-03051"],
                                ..Default::default()
                            })
                        })?;

                    let is_first_use =
                        !replace(&mut attachment_is_used[resolve_attachment as usize], true);

                    if is_first_use
                        && attachment_desc.load_op == AttachmentLoadOp::Clear
                        && matches!(
                            layout,
                            ImageLayout::ShaderReadOnlyOptimal
                                | ImageLayout::DepthStencilReadOnlyOptimal
                                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        )
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is first used in \
                                `subpasses[{1}].color_resolve_attachments[{2}]`, and \
                                `attachments[{0}].load_op` is `AttachmentLoadOp::Clear`, but \
                                `subpasses[{1}].color_resolve_attachments[{2}].layout` \
                                does not have a writable color aspect",
                                attachment, subpass_index, ref_index
                            )
                            .into(),
                            vuids: &["VUID-VkRenderPassCreateInfo2-pAttachments-02522"],
                            ..Default::default()
                        }));
                    }

                    if !attachment_potential_format_features[resolve_attachment as usize]
                        .intersects(FormatFeatures::COLOR_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is used in \
                                `subpasses[{1}].color_resolve_attachments[{2}]`, \
                                but the potential format features of `attachments[{0}].format` \
                                do not include `FormatFeatures::COLOR_ATTACHMENT`",
                                resolve_attachment, subpass_index, ref_index
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescription2-pResolveAttachments-02899"],
                            ..Default::default()
                        }));
                    }

                    if resolve_attachment_desc.samples != SampleCount::Sample1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is used in \
                                `subpasses[{1}].color_resolve_attachments[{2}]`, but \
                                `attachments[{0}].samples` is not `SampleCount::Sample1`",
                                resolve_attachment, subpass_index, ref_index
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescription2-pResolveAttachments-03067"],
                            ..Default::default()
                        }));
                    }

                    if attachment_desc.samples == SampleCount::Sample1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is used in \
                                `subpasses[{1}].color_attachments[{2}]`, and \
                                `subpasses[{1}].color_resolve_attachments[{2}]` is `Some`, but \
                                `attachments[{0}].samples` is `SampleCount::Sample1`",
                                attachment, subpass_index, ref_index
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescription2-pResolveAttachments-03066"],
                            ..Default::default()
                        }));
                    }

                    if resolve_attachment_desc.format != attachment_desc.format {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`attachments[\
                                subpasses[{0}].color_attachments[{1}].attachment\
                                ].format` is not equal to \
                                `attachments[\
                                subpasses[{0}].color_resolve_attachments[{1}].attachment\
                                ].format`",
                                subpass_index, ref_index
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescription2-pResolveAttachments-03068"],
                            ..Default::default()
                        }));
                    }
                }
            }

            if let Some(depth_stencil_attachment) = depth_stencil_attachment.as_ref() {
                let &AttachmentReference {
                    attachment,
                    layout,
                    stencil_layout,
                    aspects: _,
                    _ne: _,
                } = depth_stencil_attachment;

                let attachment_desc = attachments.get(attachment as usize).ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "`subpasses[{}].depth_stencil_attachment.attachment` \
                                is not less than the length of `attachments`",
                            subpass_index,
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-attachment-03051"],
                        ..Default::default()
                    })
                })?;

                let format = attachment_desc.format;

                if !attachment_potential_format_features[attachment as usize]
                    .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "attachment {} is used in `subpasses[{}].depth_stencil_attachment`, \
                            but the potential format features of `attachments[{0}].format` \
                            do not include `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`",
                            attachment, subpass_index,
                        )
                        .into(),
                        vuids: &["VUID-VkSubpassDescription2-pDepthStencilAttachment-02900"],
                        ..Default::default()
                    }));
                }

                if let Some(samples) = color_samples {
                    if samples != attachment_desc.samples {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`subpasses[{}].depth_stencil_attachment` uses an \
                                attachment with a different number of samples than other color or \
                                depth/stencil attachments in the subpass",
                                subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescription2-pColorAttachments-03069"],
                            ..Default::default()
                        }));
                    }
                }

                let is_first_use = !replace(&mut attachment_is_used[attachment as usize], true);

                if is_first_use {
                    if attachment_desc.load_op == AttachmentLoadOp::Clear
                        && matches!(
                            layout,
                            ImageLayout::ShaderReadOnlyOptimal
                                | ImageLayout::DepthStencilReadOnlyOptimal
                                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        )
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is first used in \
                                `subpasses[{1}].depth_stencil_attachment`, and \
                                `attachments[{0}].load_op` is `AttachmentLoadOp::Clear`, but \
                                `depth_stencil_attachment.layout` \
                                does not have a writable depth aspect",
                                attachment, subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkRenderPassCreateInfo2-pAttachments-02522"],
                            ..Default::default()
                        }));
                    }

                    if attachment_desc
                        .stencil_load_op
                        .unwrap_or(attachment_desc.load_op)
                        == AttachmentLoadOp::Clear
                        && matches!(
                            stencil_layout.unwrap_or(layout),
                            ImageLayout::ShaderReadOnlyOptimal
                                | ImageLayout::DepthStencilReadOnlyOptimal
                                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                        )
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is first used in \
                                `subpasses[{1}].depth_stencil_attachment`, and \
                                `attachments[{0}].stencil_load_op` is `AttachmentLoadOp::Clear`, \
                                but `depth_stencil_attachment.stencil_layout` \
                                does not have a writable stencil aspect",
                                attachment, subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkRenderPassCreateInfo2-pAttachments-02523"],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(depth_stencil_resolve_attachment) = depth_stencil_resolve_attachment {
                    let &AttachmentReference {
                        attachment: resolve_attachment,
                        layout: _,
                        stencil_layout: _,
                        aspects: _,
                        _ne: _,
                    } = depth_stencil_resolve_attachment;

                    let resolve_attachment_desc = attachments
                        .get(resolve_attachment as usize)
                        .ok_or_else(|| {
                            Box::new(ValidationError {
                                problem: format!(
                                "`subpasses[{}].depth_stencil_resolve_attachment.attachment` is \
                                not less than the length of `attachments`",
                                subpass_index,
                            )
                                .into(),
                                vuids: &["VUID-VkRenderPassCreateInfo2-pSubpasses-06473"],
                                ..Default::default()
                            })
                        })?;

                    let resolve_format = resolve_attachment_desc.format;

                    if !(resolve_format.components()[0] == format.components()[0]
                        && resolve_format.numeric_format_depth() == format.numeric_format_depth())
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the number of bits and numeric type of the depth component of \
                                `attachments[\
                                subpasses[{0}].depth_stencil_resolve_attachment.attachment\
                                ].format` is not equal to \
                                the number of bits and numeric type of the depth component of \
                                `attachments[\
                                subpasses[{0}].depth_stencil_attachment.attachment\
                                ].format`",
                                subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03181"],
                            ..Default::default()
                        }));
                    }

                    if !(resolve_format.components()[1] == format.components()[1]
                        && resolve_format.numeric_format_stencil()
                            == format.numeric_format_stencil())
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the number of bits and numeric type of the stencil component of \
                                `attachments[\
                                subpasses[{0}].depth_stencil_resolve_attachment.attachment\
                                ].format` is not equal to \
                                the number of bits and numeric type of the stencil component of \
                                `attachments[\
                                subpasses[{0}].depth_stencil_attachment.attachment\
                                ].format`",
                                subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03182"],
                            ..Default::default()
                        }));
                    }

                    let is_first_use =
                        !replace(&mut attachment_is_used[resolve_attachment as usize], true);

                    if is_first_use
                        && attachment_desc.load_op == AttachmentLoadOp::Clear
                        && matches!(
                            layout,
                            ImageLayout::ShaderReadOnlyOptimal
                                | ImageLayout::DepthStencilReadOnlyOptimal
                                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        )
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is first used in \
                                `subpasses[{1}].depth_stencil_resolve_attachment`, and \
                                `attachments[{0}].load_op` is `AttachmentLoadOp::Clear`, but \
                                `depth_stencil_resolve_attachment.layout` \
                                does not have a writable depth aspect",
                                attachment, subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkRenderPassCreateInfo2-pAttachments-02522"],
                            ..Default::default()
                        }));
                    }

                    if !attachment_potential_format_features[attachment as usize]
                        .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {} is used in \
                                `subpasses[{}].depth_stencil_resolve_attachment`, \
                                but the potential format features of `attachments[{0}].format` \
                                do not include `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`",
                                attachment, subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-02651"],
                            ..Default::default()
                        }));
                    }

                    if resolve_attachment_desc.samples != SampleCount::Sample1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {} is used in \
                                `subpasses[{}].depth_stencil_resolve_attachment`, but \
                                `attachments[{0}].samples` is not `SampleCount::Sample1`",
                                resolve_attachment, subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03180"],
                            ..Default::default()
                        }));
                    }

                    if attachment_desc.samples == SampleCount::Sample1 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "attachment {0} is used in subpass {1} in \
                                `depth_stencil_attachment`, and \
                                `depth_stencil_resolve_attachment` is \
                                `Some` , but `attachments[{0}].samples` is `SampleCount::Sample1`",
                                attachment, subpass_index,
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03179"],
                            ..Default::default()
                        }));
                    }
                }
            }

            for (ref_index, input_attachment) in input_attachments
                .iter()
                .enumerate()
                .flat_map(|(i, a)| a.as_ref().map(|a| (i, a)))
            {
                let &AttachmentReference {
                    attachment,
                    layout: _,
                    stencil_layout: _,
                    aspects,
                    _ne: _,
                } = input_attachment;

                let attachment_desc = attachments.get(attachment as usize).ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "`subpasses[{}].input_attachments[{}].attachment` \
                                is not less than the length of `attachments`",
                            subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-attachment-03051"],
                        ..Default::default()
                    })
                })?;

                let format_aspects = attachment_desc.format.aspects();
                let is_first_use = !replace(&mut attachment_is_used[attachment as usize], true);

                if is_first_use && attachment_desc.load_op == AttachmentLoadOp::Clear {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "attachment {0} is first used in \
                            `subpasses[{1}].input_attachments[{2}]`, and \
                            `attachments[{0}].load_op` is `AttachmentLoadOp::Clear`",
                            attachment, subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkSubpassDescription2-loadOp-03064"],
                        ..Default::default()
                    }));
                }

                if !attachment_potential_format_features[attachment as usize].intersects(
                    FormatFeatures::COLOR_ATTACHMENT | FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
                ) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "attachment {} is used in `subpasses[{}].input_attachments[{}]`, \
                            but the potential format features of `attachments[{0}].format` \
                            do not include `FormatFeatures::COLOR_ATTACHMENT` or \
                            `FormatFeatures::DEPTH_STENCIL_ATTACHMENT`",
                            attachment, subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkSubpassDescription2-pInputAttachments-02897"],
                        ..Default::default()
                    }));
                }

                if aspects != format_aspects {
                    if !(device.api_version() >= Version::V1_1
                        || device.enabled_extensions().khr_create_renderpass2
                        || device.enabled_extensions().khr_maintenance2)
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`subpasses[{}].input_attachments[{}].aspects` does not \
                                equal the aspects of \
                                `attachments[subpasses[{0}].input_attachments[{1}].attachment]\
                                .format`",
                                subpass_index, ref_index,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[
                                RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                                RequiresAllOf(&[Requires::DeviceExtension(
                                    "khr_create_renderpass2",
                                )]),
                                RequiresAllOf(&[Requires::DeviceExtension("khr_maintenance2")]),
                            ]),
                            // vuids?
                            ..Default::default()
                        }));
                    }

                    if !format_aspects.contains(aspects) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`subpasses[{}].input_attachments[{}].aspects` is not a subset of \
                                the aspects of \
                                `attachments[subpasses[{0}].input_attachments[{1}].attachment]\
                                .format`",
                                subpass_index, ref_index,
                            )
                            .into(),
                            vuids: &["VUID-VkRenderPassCreateInfo2-attachment-02525"],
                            ..Default::default()
                        }));
                    }
                }
            }

            for (ref_index, &atch) in preserve_attachments.iter().enumerate() {
                if atch as usize >= attachments.len() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`subpasses[{}].preserve_attachments[{}]` \
                            is not less than the length of `attachments`",
                            subpass_index, ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-attachment-03051"],
                        ..Default::default()
                    }));
                }
            }
        }

        for (dependency_index, dependency) in dependencies.iter().enumerate() {
            // VUID-VkRenderPassCreateInfo2-pDependencies-parameter
            dependency
                .validate(device)
                .map_err(|err| err.add_context(format!("dependencies[{}]", dependency_index)))?;

            let &SubpassDependency {
                src_subpass,
                dst_subpass,
                src_stages,
                dst_stages,
                src_access: _,
                dst_access: _,
                dependency_flags,
                view_offset: _,
                _ne: _,
            } = dependency;

            if subpasses[0].view_mask == 0
                && dependency_flags.intersects(DependencyFlags::VIEW_LOCAL)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`subpasses[0].view_mask` is 0, and `dependencies[{}].dependency_flags` \
                        includes `DependencyFlags::VIEW_LOCAL`",
                        dependency_index,
                    )
                    .into(),
                    vuids: &["VUID-VkRenderPassCreateInfo2-viewMask-03059"],
                    ..Default::default()
                }));
            }

            if let Some(src_subpass) = src_subpass {
                if src_subpass as usize >= subpasses.len() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`dependencies[{}].src_subpass` is not less than the length of \
                            `subpasses`",
                            dependency_index,
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-srcSubpass-02526"],
                        ..Default::default()
                    }));
                }
            }

            if let Some(dst_subpass) = dst_subpass {
                if dst_subpass as usize >= subpasses.len() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`dependencies[{}].dst_subpass` is not less than the length of \
                            `subpasses`",
                            dependency_index,
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-dstSubpass-02527"],
                        ..Default::default()
                    }));
                }
            }

            if !PipelineStages::from(QueueFlags::GRAPHICS).contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!("dependencies[{}].src_stages", dependency_index,).into(),
                    problem: "contains a non-graphics stage".into(),
                    vuids: &["VUID-VkRenderPassCreateInfo2-pDependencies-03054"],
                    ..Default::default()
                }));
            }

            if !PipelineStages::from(QueueFlags::GRAPHICS).contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!("dependencies[{}].dst_stages", dependency_index,).into(),
                    problem: "contains a non-graphics stage".into(),
                    vuids: &["VUID-VkRenderPassCreateInfo2-pDependencies-03055"],
                    ..Default::default()
                }));
            }

            if let (Some(src_subpass), Some(dst_subpass)) = (src_subpass, dst_subpass) {
                if src_subpass == dst_subpass
                    && subpasses[src_subpass as usize].view_mask.count_ones() > 1
                    && !dependency_flags.intersects(DependencyFlags::VIEW_LOCAL)
                {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`dependencies[{0}].src_subpass` equals \
                            `dependencies[{0}].dst_subpass`, and \
                            `subpasses[dependencies[{0}].src_subpass].view_mask` has more than \
                            one bit set, and `dependencies[{0}].dependency_flags` does not \
                            contain `DependencyFlags::VIEW_LOCAL`",
                            dependency_index
                        )
                        .into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-pDependencies-03060"],
                        ..Default::default()
                    }));
                }
            }
        }

        if !correlated_view_masks.is_empty() {
            if subpasses[0].view_mask == 0 {
                return Err(Box::new(ValidationError {
                    problem: "`correlated_view_masks` is not empty, but \
                        `subpasses[0].view_mask` is zero"
                        .into(),
                    vuids: &["VUID-VkRenderPassCreateInfo2-viewMask-03057"],
                    ..Default::default()
                }));
            }

            correlated_view_masks.iter().try_fold(0, |total, &mask| {
                if total & mask != 0 {
                    Err(Box::new(ValidationError {
                        context: "correlated_view_masks".into(),
                        problem: "the bit masks overlap with each other".into(),
                        vuids: &["VUID-VkRenderPassCreateInfo2-pCorrelatedViewMasks-03056"],
                        ..Default::default()
                    }))
                } else {
                    Ok(total | mask)
                }
            })?;
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a render pass.
    RenderPassCreateFlags = RenderPassCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    TRANSFORM = TRANSFORM_QCOM {
        device_extensions: [qcom_render_pass_transform],
    }, */
}

/// Describes an attachment that will be used in a render pass.
#[derive(Clone, Copy, Debug)]
pub struct AttachmentDescription {
    /// Additional properties of the attachment.
    ///
    /// The default value is empty.
    pub flags: AttachmentDescriptionFlags,

    /// The format of the image that is going to be bound.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

    /// The number of samples of the image that is going to be bound.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub samples: SampleCount,

    /// What the implementation should do with the attachment at the start of the subpass that
    /// first uses it.
    ///
    /// The default value is [`AttachmentLoadOp::DontCare`].
    pub load_op: AttachmentLoadOp,

    /// What the implementation should do with the attachment at the end of the subpass that
    /// last uses it.
    ///
    /// The default value is [`AttachmentStoreOp::DontCare`].
    pub store_op: AttachmentStoreOp,

    /// The layout that the attachment must in at the start of the render pass.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub initial_layout: ImageLayout,

    /// The layout that the attachment will be transitioned to at the end of the render pass.
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub final_layout: ImageLayout,

    /// The `load_op` for the stencil aspect of the attachment, if different.
    ///
    /// The default value is `None`.
    pub stencil_load_op: Option<AttachmentLoadOp>,

    /// The `store_op` for the stencil aspect of the attachment, if different.
    ///
    /// The default value is `None`.
    pub stencil_store_op: Option<AttachmentStoreOp>,

    /// The `initial_layout` for the stencil aspect of the attachment, if different.
    ///
    /// `stencil_initial_layout` and `stencil_final_layout` must be either both `None`,
    /// or both `Some`.
    ///
    /// If this is `Some`, then the
    /// [`separate_depth_stencil_layouts`](crate::device::Features::separate_depth_stencil_layouts)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub stencil_initial_layout: Option<ImageLayout>,

    /// The `final_layout` for the stencil aspect of the attachment, if different.
    ///
    /// `stencil_initial_layout` and `stencil_final_layout` must be either both `None`,
    /// or both `Some`.
    ///
    /// If this is `Some`, then the
    /// [`separate_depth_stencil_layouts`](crate::device::Features::separate_depth_stencil_layouts)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub stencil_final_layout: Option<ImageLayout>,

    pub _ne: crate::NonExhaustive,
}

impl Default for AttachmentDescription {
    #[inline]
    fn default() -> Self {
        Self {
            flags: AttachmentDescriptionFlags::empty(),
            format: Format::UNDEFINED,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::DontCare,
            store_op: AttachmentStoreOp::DontCare,
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::Undefined,
            stencil_load_op: None,
            stencil_store_op: None,
            stencil_initial_layout: None,
            stencil_final_layout: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl AttachmentDescription {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            format,
            samples,
            load_op,
            store_op,
            initial_layout,
            final_layout,
            stencil_load_op,
            stencil_store_op,
            stencil_initial_layout,
            stencil_final_layout,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkAttachmentDescription2-flags-parameter"])
        })?;

        format.validate_device(device).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-VkAttachmentDescription2-format-parameter"])
        })?;

        samples.validate_device(device).map_err(|err| {
            err.add_context("samples")
                .set_vuids(&["VUID-VkAttachmentDescription2-samples-parameter"])
        })?;

        load_op.validate_device(device).map_err(|err| {
            err.add_context("load_op")
                .set_vuids(&["VUID-VkAttachmentDescription2-loadOp-parameter"])
        })?;

        store_op.validate_device(device).map_err(|err| {
            err.add_context("store_op")
                .set_vuids(&["VUID-VkAttachmentDescription2-storeOp-parameter"])
        })?;

        initial_layout.validate_device(device).map_err(|err| {
            err.add_context("initial_layout")
                .set_vuids(&["VUID-VkAttachmentDescription2-initialLayout-parameter"])
        })?;

        final_layout.validate_device(device).map_err(|err| {
            err.add_context("final_layout")
                .set_vuids(&["VUID-VkAttachmentDescription2-finalLayout-parameter"])
        })?;

        if matches!(
            final_layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized
        ) {
            return Err(Box::new(ValidationError {
                context: "final_layout".into(),
                problem: "is `ImageLayout::Undefined` or `ImageLayout::Preinitialized`".into(),
                vuids: &["VUID-VkAttachmentDescription2-finalLayout-00843"],
                ..Default::default()
            }));
        }

        if !device.enabled_features().separate_depth_stencil_layouts {
            if matches!(
                initial_layout,
                ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "initial_layout".into(),
                    problem: "specifies a layout for only the depth aspect or only the \
                        stencil aspect"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "separate_depth_stencil_layouts",
                    )])]),
                    vuids: &["VUID-VkAttachmentDescription2-separateDepthStencilLayouts-03284"],
                }));
            }

            if matches!(
                final_layout,
                ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "final_layout".into(),
                    problem: "specifies a layout for only the depth aspect or only the \
                        stencil aspect"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "separate_depth_stencil_layouts",
                    )])]),
                    vuids: &["VUID-VkAttachmentDescription2-separateDepthStencilLayouts-03285"],
                }));
            }
        }

        if let Some(stencil_load_op) = stencil_load_op {
            stencil_load_op.validate_device(device).map_err(|err| {
                err.add_context("stencil_load_op")
                    .set_vuids(&["VUID-VkAttachmentDescription2-stencilLoadOp-parameter"])
            })?;
        }

        if let Some(stencil_store_op) = stencil_store_op {
            stencil_store_op.validate_device(device).map_err(|err| {
                err.add_context("stencil_store_op")
                    .set_vuids(&["VUID-VkAttachmentDescription2-stencilStoreOp-parameter"])
            })?;
        }

        if stencil_initial_layout.is_some() != stencil_final_layout.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`stencil_initial_layout` and `stencil_final_layout` are not either both \
                    `None` or both `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        if let Some(stencil_initial_layout) = stencil_initial_layout {
            if !device.enabled_features().separate_depth_stencil_layouts {
                return Err(Box::new(ValidationError {
                    context: "stencil_initial_layout".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "separate_depth_stencil_layouts",
                    )])]),
                    ..Default::default()
                }));
            }

            stencil_initial_layout
                .validate_device(device)
                .map_err(|err| {
                    err.add_context("stencil_initial_layout").set_vuids(&[
                        "VUID-VkAttachmentDescriptionStencilLayout-stencilInitialLayout-parameter",
                    ])
                })?;

            if matches!(
                stencil_initial_layout,
                ImageLayout::ColorAttachmentOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "stencil_initial_layout".into(),
                    problem: "cannot be used with stencil formats".into(),
                    vuids: &[
                        "VUID-VkAttachmentDescriptionStencilLayout-stencilInitialLayout-03308",
                    ],
                    ..Default::default()
                }));
            }
        }

        if let Some(stencil_final_layout) = stencil_final_layout {
            if !device.enabled_features().separate_depth_stencil_layouts {
                return Err(Box::new(ValidationError {
                    context: "stencil_final_layout".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "separate_depth_stencil_layouts",
                    )])]),
                    ..Default::default()
                }));
            }

            stencil_final_layout
                .validate_device(device)
                .map_err(|err| {
                    err.add_context("stencil_final_layout").set_vuids(&[
                        "VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-parameter",
                    ])
                })?;

            if matches!(
                stencil_final_layout,
                ImageLayout::ColorAttachmentOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "stencil_final_layout".into(),
                    problem: "is a color or combined depth/stencil layout".into(),
                    vuids: &["VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-03309"],
                    ..Default::default()
                }));
            }

            if matches!(
                stencil_final_layout,
                ImageLayout::Undefined | ImageLayout::Preinitialized
            ) {
                return Err(Box::new(ValidationError {
                    context: "stencil_final_layout".into(),
                    problem: "is `ImageLayout::Undefined` or `ImageLayout::Preinitialized`".into(),
                    vuids: &["VUID-VkAttachmentDescriptionStencilLayout-stencilFinalLayout-03309"],
                    ..Default::default()
                }));
            }
        }

        if format == Format::UNDEFINED {
            return Err(Box::new(ValidationError {
                context: "format".into(),
                problem: "is `Format::UNDEFINED`".into(),
                vuids: &["VUID-VkAttachmentDescription2-format-06698"],
                ..Default::default()
            }));
        }

        let format_aspects = format.aspects();

        if format_aspects.intersects(ImageAspects::COLOR) {
            if matches!(
                initial_layout,
                ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`format` has a color component, but `initial_layout` cannot be \
                        used with color formats"
                        .into(),
                    vuids: &[
                        "VUID-VkAttachmentDescription2-format-03280",
                        "VUID-VkAttachmentDescription2-format-06487",
                        "VUID-VkAttachmentDescription2-format-03286",
                    ],
                    ..Default::default()
                }));
            }

            if matches!(
                final_layout,
                ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
                    | ImageLayout::StencilAttachmentOptimal
                    | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`format` has a color component, but `final_layout` cannot be \
                        used with color formats"
                        .into(),
                    vuids: &[
                        "VUID-VkAttachmentDescription2-format-03282",
                        "VUID-VkAttachmentDescription2-format-06488",
                        "VUID-VkAttachmentDescription2-format-03287",
                    ],
                    ..Default::default()
                }));
            }

            if load_op == AttachmentLoadOp::Load && initial_layout == ImageLayout::Undefined {
                return Err(Box::new(ValidationError {
                    problem: "`format` has a color component, `load_op` is \
                        `AttachmentLoadOp::Load`, and `initial_layout` is \
                        `ImageLayout::Undefined`"
                        .into(),
                    vuids: &["VUID-VkAttachmentDescription2-format-06699"],
                    ..Default::default()
                }));
            }
        }

        if format_aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            if matches!(initial_layout, ImageLayout::ColorAttachmentOptimal) {
                return Err(Box::new(ValidationError {
                    problem: "`format` has a depth component, but `initial_layout` cannot be \
                        used with depth formats"
                        .into(),
                    vuids: &["VUID-VkAttachmentDescription2-format-03281"],
                    ..Default::default()
                }));
            }

            if matches!(final_layout, ImageLayout::ColorAttachmentOptimal) {
                return Err(Box::new(ValidationError {
                    problem: "`format` has a depth component, but `final_layout` cannot be \
                        used with depth formats"
                        .into(),
                    vuids: &["VUID-VkAttachmentDescription2-format-03283"],
                    ..Default::default()
                }));
            }

            if format_aspects.intersects(ImageAspects::DEPTH) {
                if matches!(
                    initial_layout,
                    ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        problem: "`format` has a depth component, but `initial_layout` \
                            specifies a layout for only the stencil component"
                            .into(),
                        vuids: &[
                            "VUID-VkAttachmentDescription2-format-06906",
                            "VUID-VkAttachmentDescription2-format-03290",
                        ],
                        ..Default::default()
                    }));
                }

                if matches!(
                    final_layout,
                    ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        problem: "`format` has a depth component, but `final_layout` \
                            specifies a layout for only the stencil component"
                            .into(),
                        vuids: &[
                            "VUID-VkAttachmentDescription2-format-06907",
                            "VUID-VkAttachmentDescription2-format-03291",
                        ],
                        ..Default::default()
                    }));
                }

                if load_op == AttachmentLoadOp::Load && initial_layout == ImageLayout::Undefined {
                    return Err(Box::new(ValidationError {
                        problem: "`format` has a depth component, `load_op` is \
                            `AttachmentLoadOp::Load`, and `initial_layout` is \
                            `ImageLayout::Undefined`"
                            .into(),
                        vuids: &["VUID-VkAttachmentDescription2-format-06699"],
                        ..Default::default()
                    }));
                }
            }

            if format_aspects.intersects(ImageAspects::STENCIL) {
                if stencil_load_op.unwrap_or(load_op) == AttachmentLoadOp::Load
                    && stencil_initial_layout.unwrap_or(initial_layout) == ImageLayout::Undefined
                {
                    return Err(Box::new(ValidationError {
                        problem: "`format` has a stencil component, `stencil_load_op` is \
                            `AttachmentLoadOp::Load`, and `stencil_initial_layout` is \
                            `ImageLayout::Undefined`"
                            .into(),
                        vuids: &[
                            "VUID-VkAttachmentDescription2-pNext-06704",
                            "VUID-VkAttachmentDescription2-pNext-06705",
                        ],
                        ..Default::default()
                    }));
                }

                if stencil_initial_layout.is_none() && stencil_final_layout.is_none() {
                    if matches!(
                        initial_layout,
                        ImageLayout::DepthAttachmentOptimal | ImageLayout::DepthReadOnlyOptimal
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`format` has a stencil component, `stencil_initial_layout` \
                                and `stencil_final_layout` are both `None`, and \
                                `initial_layout` does not specify a layout for the stencil aspect"
                                .into(),
                            vuids: &[
                                "VUID-VkAttachmentDescription2-format-06249",
                                "VUID-VkAttachmentDescription2-format-06247",
                            ],
                            ..Default::default()
                        }));
                    }

                    if matches!(
                        final_layout,
                        ImageLayout::DepthAttachmentOptimal | ImageLayout::DepthReadOnlyOptimal
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`format` has a stencil component, `stencil_initial_layout` \
                                and `stencil_final_layout` are both `None`, and \
                                `final_layout` does not specify a layout for the stencil aspect"
                                .into(),
                            vuids: &[
                                "VUID-VkAttachmentDescription2-format-06250",
                                "VUID-VkAttachmentDescription2-format-06248",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        // VUID-VkAttachmentDescription2-samples-08745
        // TODO: How do you check this?

        Ok(())
    }

    pub(crate) fn required_clear_value(&self) -> Option<ClearValueType> {
        if let Some(numeric_format) = self.format.numeric_format_color() {
            (self.load_op == AttachmentLoadOp::Clear).then(|| match numeric_format.numeric_type() {
                NumericType::Float => ClearValueType::Float,
                NumericType::Int => ClearValueType::Int,
                NumericType::Uint => ClearValueType::Uint,
            })
        } else {
            let aspects = self.format.aspects();
            let need_depth =
                aspects.intersects(ImageAspects::DEPTH) && self.load_op == AttachmentLoadOp::Clear;
            let need_stencil = aspects.intersects(ImageAspects::STENCIL)
                && self.stencil_load_op.unwrap_or(self.load_op) == AttachmentLoadOp::Clear;

            match (need_depth, need_stencil) {
                (true, true) => Some(ClearValueType::DepthStencil),
                (true, false) => Some(ClearValueType::Depth),
                (false, true) => Some(ClearValueType::Stencil),
                (false, false) => None,
            }
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a render pass attachment description.
    AttachmentDescriptionFlags = AttachmentDescriptionFlags(u32);

    /* TODO: enable
    // TODO: document
    MAY_ALIAS = MAY_ALIAS, */
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
    /// Additional properties of the subpass.
    ///
    /// The default value is empty.
    pub flags: SubpassDescriptionFlags,

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
    /// must not be [`AttachmentLoadOp::Clear`].
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

    /// The attachments of the render pass that are to be used as color resolve attachments in this
    /// subpass.
    ///
    /// This list must either be empty or have the same length as `color_attachments`. If it's not
    /// empty, then each resolve attachment is paired with the color attachment of the same index.
    /// Each referenced color resolve attachment must have the same `format` as the corresponding
    /// color attachment.
    /// If the color resolve attachment is `Some`, then the referenced color resolve attachment
    /// must have a `samples` value of [`SampleCount::Sample1`], while the corresponding
    /// color attachment must have a `samples` value other than [`SampleCount::Sample1`].
    ///
    /// The default value is empty.
    pub color_resolve_attachments: Vec<Option<AttachmentReference>>,

    /// The single attachment of the render pass that is to be used as depth/stencil attachment in
    /// this subpass.
    ///
    /// If set to `Some`, the referenced attachment must have the same `samples` value as those in
    /// `color_attachments`.
    ///
    /// The default value is `None`.
    pub depth_stencil_attachment: Option<AttachmentReference>,

    /// The single attachment of the render pass that is to be used as depth/stencil resolve
    /// attachment in this subpass.
    ///
    /// The depth/stencil resolve attachment must have the same `format` as the depth/stencil
    /// attachment.
    /// If this is `Some`, then `depth_stencil_attachment` must also be `Some`, and at least one
    /// of `depth_resolve_mode` and `stencil_resolve_mode` must be `Some`. The referenced
    /// depth/stencil resolve attachment must have a `samples` value of [`SampleCount::Sample1`],
    /// while the depth/stencil attachment must have a `samples` value other than
    /// [`SampleCount::Sample1`].
    ///
    /// If this is `Some`, then the device API version must be at least 1.2, or the
    /// [`khr_depth_stencil_resolve`](crate::device::DeviceExtensions::khr_depth_stencil_resolve)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub depth_stencil_resolve_attachment: Option<AttachmentReference>,

    /// How the resolve operation should be performed for the depth aspect. If set to `None`,
    /// no resolve is performed for the depth aspect.
    ///
    /// If `depth_stencil_resolve_attachment` is `None`, this must also be `None`.
    ///
    /// The default value is `None`.
    pub depth_resolve_mode: Option<ResolveMode>,

    /// How the resolve operation should be performed for the stencil aspect. If set to `None`,
    /// no resolve is performed for the stencil aspect.
    ///
    /// If `depth_stencil_resolve_attachment` is `None`, this must also be `None`.
    ///
    /// The default value is `None`.
    pub stencil_resolve_mode: Option<ResolveMode>,

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
            flags: SubpassDescriptionFlags::empty(),
            view_mask: 0,
            color_attachments: Vec::new(),
            color_resolve_attachments: Vec::new(),
            depth_stencil_attachment: None,
            depth_stencil_resolve_attachment: None,
            depth_resolve_mode: None,
            stencil_resolve_mode: None,
            input_attachments: Vec::new(),
            preserve_attachments: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SubpassDescription {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let properties = device.physical_device().properties();

        let &Self {
            flags,
            view_mask,
            ref input_attachments,
            ref color_attachments,
            ref color_resolve_attachments,
            ref depth_stencil_attachment,
            ref depth_stencil_resolve_attachment,
            depth_resolve_mode,
            stencil_resolve_mode,
            ref preserve_attachments,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkSubpassDescription2-flags-parameter"])
        })?;

        if color_attachments.len() as u32 > properties.max_color_attachments {
            return Err(Box::new(ValidationError {
                context: "color_attachments".into(),
                problem: "the number of elements is greater than the `max_color_attachments` limit"
                    .into(),
                vuids: &["VUID-VkSubpassDescription2-colorAttachmentCount-03063"],
                ..Default::default()
            }));
        }

        // Track the layout of each attachment used in this subpass
        #[derive(PartialEq, Eq)]
        struct Layouts {
            layout: ImageLayout,
            stencil_layout: Option<ImageLayout>,
        }

        let mut layouts = HashMap::default();

        if !color_resolve_attachments.is_empty()
            && color_resolve_attachments.len() != color_attachments.len()
        {
            return Err(Box::new(ValidationError {
                problem: "`color_resolve_attachments` is not empty, but the length is not equal \
                    to the length of `color_attachments`"
                    .into(),
                vuids: &["VUID-VkSubpassDescription2-pResolveAttachments-parameter"],
                ..Default::default()
            }));
        }

        for (ref_index, color_attachment) in color_attachments.iter().enumerate() {
            if let Some(color_attachment) = color_attachment {
                // VUID-VkSubpassDescription2-pColorAttachments-parameter
                color_attachment
                    .validate(device)
                    .map_err(|err| err.add_context(format!("color_attachments[{}]", ref_index)))?;

                let &AttachmentReference {
                    attachment,
                    layout,
                    stencil_layout,
                    aspects,
                    _ne: _,
                } = color_attachment;

                if preserve_attachments.contains(&attachment) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`color_attachments[{}].attachment` also occurs in \
                            `preserve_attachments`",
                            ref_index
                        )
                        .into(),
                        vuids: &["VUID-VkSubpassDescription2-pPreserveAttachments-03074"],
                        ..Default::default()
                    }));
                }

                if matches!(
                    layout,
                    ImageLayout::DepthStencilAttachmentOptimal
                        | ImageLayout::ShaderReadOnlyOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        | ImageLayout::DepthAttachmentOptimal
                        | ImageLayout::DepthReadOnlyOptimal
                        | ImageLayout::StencilAttachmentOptimal
                        | ImageLayout::StencilReadOnlyOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        context: format!("color_attachments[{}].layout", ref_index).into(),
                        problem: "cannot be used with color attachments".into(),
                        vuids: &[
                            "VUID-VkSubpassDescription2-attachment-06913",
                            "VUID-VkSubpassDescription2-attachment-06916",
                            "VUID-VkSubpassDescription2-attachment-06919",
                        ],
                        ..Default::default()
                    }));
                }

                if stencil_layout.is_some() {
                    return Err(Box::new(ValidationError {
                        context: format!("color_attachments[{}].stencil_layout", ref_index).into(),
                        problem: "is `Some`".into(),
                        ..Default::default()
                    }));
                }

                let layouts_entry = Layouts {
                    layout,
                    stencil_layout,
                };

                match layouts.entry(attachment) {
                    Entry::Occupied(entry) => {
                        if *entry.get() != layouts_entry {
                            return Err(Box::new(ValidationError {
                                context: format!("color_attachments[{}].layout", ref_index).into(),
                                problem: "is not equal to the layout used for this attachment \
                                    elsewhere in this subpass"
                                    .into(),
                                vuids: &["VUID-VkSubpassDescription2-layout-02528"],
                                ..Default::default()
                            }));
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(layouts_entry);
                    }
                }

                if !aspects.is_empty() {
                    return Err(Box::new(ValidationError {
                        context: format!("color_attachments[{}].aspects", ref_index).into(),
                        problem: "is not empty for a color attachment".into(),
                        // vuids? Not required by spec, but enforced by Vulkano for sanity.
                        ..Default::default()
                    }));
                }

                if let Some(color_resolve_attachment) = color_resolve_attachments
                    .get(ref_index)
                    .and_then(Option::as_ref)
                {
                    // VUID-VkSubpassDescription2-pResolveAttachments-parameter
                    color_resolve_attachment.validate(device).map_err(|err| {
                        err.add_context(format!("color_resolve_attachments[{}]", ref_index))
                    })?;

                    let &AttachmentReference {
                        attachment: resolve_attachment,
                        layout: resolve_layout,
                        stencil_layout: resolve_stencil_layout,
                        aspects: resolve_aspects,
                        _ne: _,
                    } = color_resolve_attachment;

                    if preserve_attachments.contains(&resolve_attachment) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`color_resolve_attachments[{}].attachment` \
                                also occurs in `preserve_attachments`",
                                ref_index
                            )
                            .into(),
                            vuids: &["VUID-VkSubpassDescription2-pPreserveAttachments-03074"],
                            ..Default::default()
                        }));
                    }

                    if matches!(
                        resolve_layout,
                        ImageLayout::DepthStencilAttachmentOptimal
                            | ImageLayout::ShaderReadOnlyOptimal
                            | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                            | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                            | ImageLayout::DepthAttachmentOptimal
                            | ImageLayout::DepthReadOnlyOptimal
                            | ImageLayout::StencilAttachmentOptimal
                            | ImageLayout::StencilReadOnlyOptimal
                    ) {
                        return Err(Box::new(ValidationError {
                            context: format!("color_resolve_attachments[{}].layout", ref_index)
                                .into(),
                            problem: "cannot be used with color resolve attachments".into(),
                            vuids: &[
                                "VUID-VkSubpassDescription2-attachment-06914",
                                "VUID-VkSubpassDescription2-attachment-06917",
                                "VUID-VkSubpassDescription2-attachment-06920",
                            ],
                            ..Default::default()
                        }));
                    }

                    if resolve_stencil_layout.is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!(
                                "color_resolve_attachments[{}].stencil_layout",
                                ref_index
                            )
                            .into(),
                            problem: "is `Some`".into(),
                            ..Default::default()
                        }));
                    }

                    let layouts_entry = Layouts {
                        layout: resolve_layout,
                        stencil_layout: resolve_stencil_layout,
                    };

                    match layouts.entry(resolve_attachment) {
                        Entry::Occupied(entry) => {
                            if *entry.get() != layouts_entry {
                                return Err(Box::new(ValidationError {
                                    context: format!(
                                        "color_resolve_attachments[{}].layout",
                                        ref_index
                                    )
                                    .into(),
                                    problem: "is not equal to the layout used for this attachment \
                                        elsewhere in this subpass"
                                        .into(),
                                    vuids: &["VUID-VkSubpassDescription2-layout-02528"],
                                    ..Default::default()
                                }));
                            }
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(layouts_entry);
                        }
                    }

                    if !resolve_aspects.is_empty() {
                        return Err(Box::new(ValidationError {
                            context: format!("color_resolve_attachments[{}].aspects", ref_index)
                                .into(),
                            problem: "is not empty for a color attachment".into(),
                            // vuids? Not required by spec, but enforced by Vulkano for sanity.
                            ..Default::default()
                        }));
                    }
                }
            } else if color_resolve_attachments
                .get(ref_index)
                .and_then(Option::as_ref)
                .is_some()
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`color_resolve_attachments[{}]` is `Some`, but \
                        `color_attachments[{0}]` is `None`",
                        ref_index,
                    )
                    .into(),
                    vuids: &["VUID-VkSubpassDescription2-pResolveAttachments-03065"],
                    ..Default::default()
                }));
            }
        }

        if let Some(depth_stencil_attachment) = depth_stencil_attachment {
            // VUID-VkSubpassDescription2-pDepthStencilAttachment-parameter
            depth_stencil_attachment
                .validate(device)
                .map_err(|err| err.add_context("depth_stencil_attachment"))?;

            let &AttachmentReference {
                attachment,
                layout,
                stencil_layout,
                aspects,
                _ne: _,
            } = depth_stencil_attachment;

            if preserve_attachments.contains(&attachment) {
                return Err(Box::new(ValidationError {
                    problem: "`depth_stencil_attachment.attachment` also occurs in \
                        `preserve_attachments`"
                        .into(),
                    vuids: &["VUID-VkSubpassDescription2-pPreserveAttachments-03074"],
                    ..Default::default()
                }));
            }

            if matches!(
                layout,
                ImageLayout::ColorAttachmentOptimal | ImageLayout::ShaderReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "depth_stencil_attachment.layout".into(),
                    problem: "cannot be used with depth/stencil attachments`".into(),
                    vuids: &["VUID-VkSubpassDescription2-attachment-06915"],
                    ..Default::default()
                }));
            }

            if stencil_layout.is_some() {
                if matches!(
                    layout,
                    ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
                ) {
                    return Err(Box::new(ValidationError {
                        problem: "`depth_stencil_attachment.stencil_layout` is `Some`, but \
                            `depth_stencil_attachment.layout` is \
                            `ImageLayout::StencilAttachmentOptimal` or \
                            `ImageLayout::StencilReadOnlyOptimal`"
                            .into(),
                        vuids: &["VUID-VkSubpassDescription2-attachment-06251"],
                        ..Default::default()
                    }));
                }
            }

            let layouts_entry = Layouts {
                layout,
                stencil_layout,
            };

            match layouts.entry(attachment) {
                Entry::Occupied(entry) => {
                    if *entry.get() != layouts_entry {
                        return Err(Box::new(ValidationError {
                            context: "depth_stencil_attachment.layout".into(),
                            problem: "is not equal to the layout used for this attachment \
                                elsewhere in this subpass"
                                .into(),
                            vuids: &["VUID-VkSubpassDescription2-layout-02528"],
                            ..Default::default()
                        }));
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(layouts_entry);
                }
            }

            if !aspects.is_empty() {
                return Err(Box::new(ValidationError {
                    context: "depth_stencil_attachment.aspects".into(),
                    problem: "is not empty for a depth/stencil attachment".into(),
                    // vuids? Not required by spec, but enforced by Vulkano for sanity.
                    ..Default::default()
                }));
            }

            if color_attachments
                .iter()
                .flatten()
                .any(|color_atch_ref| color_atch_ref.attachment == attachment)
            {
                return Err(Box::new(ValidationError {
                    problem: "`depth_stencil_attachment.attachment` also occurs in \
                        `color_attachments`"
                        .into(),
                    vuids: &["VUID-VkSubpassDescription2-pDepthStencilAttachment-04440"],
                    ..Default::default()
                }));
            }

            if let Some(depth_stencil_resolve_attachment) = depth_stencil_resolve_attachment {
                if !(device.api_version() >= Version::V1_2
                    || device.enabled_extensions().khr_depth_stencil_resolve)
                {
                    return Err(Box::new(ValidationError {
                        context: "depth_stencil_resolve_attachment".into(),
                        problem: "is `Some`".into(),
                        requires_one_of: RequiresOneOf(&[
                            RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                            RequiresAllOf(&[Requires::DeviceExtension(
                                "khr_depth_stencil_resolve",
                            )]),
                        ]),
                        // vuids?
                        ..Default::default()
                    }));
                }

                depth_stencil_resolve_attachment
                    .validate(device)
                    .map_err(|err| err.add_context("depth_stencil_resolve_attachment"))?;

                let &AttachmentReference {
                    attachment: resolve_attachment,
                    layout: resolve_layout,
                    stencil_layout: resolve_stencil_layout,
                    aspects: resolve_aspects,
                    _ne,
                } = depth_stencil_resolve_attachment;

                if preserve_attachments.contains(&resolve_attachment) {
                    return Err(Box::new(ValidationError {
                        problem: "`depth_stencil_resolve_attachment.attachment` also occurs in \
                            `preserve_attachments`"
                            .into(),
                        vuids: &["VUID-VkSubpassDescription2-pPreserveAttachments-03074"],
                        ..Default::default()
                    }));
                }

                let layouts_entry = Layouts {
                    layout: resolve_layout,
                    stencil_layout: resolve_stencil_layout,
                };

                match layouts.entry(resolve_attachment) {
                    Entry::Occupied(entry) => {
                        if *entry.get() != layouts_entry {
                            return Err(Box::new(ValidationError {
                                context: "depth_attachment.resolve.attachment_ref.layout".into(),
                                problem: "is not equal to the layout used for this attachment \
                                    elsewhere in this subpass"
                                    .into(),
                                vuids: &["VUID-VkSubpassDescription2-layout-02528"],
                                ..Default::default()
                            }));
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(layouts_entry);
                    }
                }

                if !resolve_aspects.is_empty() {
                    return Err(Box::new(ValidationError {
                        context: "depth_stencil_resolve_attachment.aspects".into(),
                        problem: "is not empty for a depth/stencil attachment".into(),
                        // vuids? Not required by spec, but enforced by Vulkano for sanity.
                        ..Default::default()
                    }));
                }

                match (depth_resolve_mode, stencil_resolve_mode) {
                    (None, None) => {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_stencil_resolve_attachment` is `Some`, but \
                                `depth_resolve_mode` and `stencil_resolve_mode` are both `None`".into(),
                            vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03178"],
                            ..Default::default()
                        }));
                    }
                    (None, Some(_)) | (Some(_), None) => {
                        if !properties.independent_resolve_none.unwrap_or(false) {
                            return Err(Box::new(ValidationError {
                                problem: "`depth_stencil_resolve_attachment` is `Some`, and \
                                    the `independent_resolve_none` device property is \
                                    `false`, but one of `depth_resolve_mode` and \
                                    `stencil_resolve_mode` is `Some` while the other is `None`"
                                    .into(),
                                vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03186"],
                                ..Default::default()
                            }));
                        }
                    }
                    (Some(depth_resolve_mode), Some(stencil_resolve_mode)) => {
                        if depth_resolve_mode != stencil_resolve_mode
                            && !properties.independent_resolve.unwrap_or(false)
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`depth_stencil_resolve_attachment` is `Some`, and \
                                    `depth_resolve_mode` and `stencil_resolve_mode` are both \
                                    `Some`, and the `independent_resolve` device property is \
                                    `false`, but `depth_resolve_mode` does not equal \
                                    `stencil_resolve_mode`"
                                    .into(),
                                vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03185"],
                                ..Default::default()
                            }));
                        }
                    }
                }
            }
        } else if depth_stencil_resolve_attachment.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`depth_stencil_resolve_attachment` is `Some`, but \
                    `depth_stencil_attachment` is `None`"
                    .into(),
                vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03177"],
                ..Default::default()
            }));
        }

        if let Some(depth_resolve_mode) = depth_resolve_mode {
            if depth_stencil_resolve_attachment.is_some() {
                return Err(Box::new(ValidationError {
                    problem: "`depth_resolve_mode` is `Some`, but \
                        `depth_stencil_resolve_attachment` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }

            depth_resolve_mode
                .validate_device(device)
                .map_err(|err| err.add_context("depth_resolve_mode"))?;

            if !properties
                .supported_depth_resolve_modes
                .unwrap_or_default()
                .contains_enum(depth_resolve_mode)
            {
                return Err(Box::new(ValidationError {
                    problem: "`depth_resolve_mode` is not one of the modes in the \
                        `supported_depth_resolve_modes` device property"
                        .into(),
                    vuids: &["VUID-VkSubpassDescriptionDepthStencilResolve-depthResolveMode-03183"],
                    ..Default::default()
                }));
            }
        }

        if let Some(stencil_resolve_mode) = stencil_resolve_mode {
            if depth_stencil_resolve_attachment.is_some() {
                return Err(Box::new(ValidationError {
                    problem: "`stencil_resolve_mode` is `Some`, but \
                        `depth_stencil_resolve_attachment` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }

            stencil_resolve_mode
                .validate_device(device)
                .map_err(|err| err.add_context("stencil_resolve_mode"))?;

            if !properties
                .supported_stencil_resolve_modes
                .unwrap_or_default()
                .contains_enum(stencil_resolve_mode)
            {
                return Err(Box::new(ValidationError {
                    problem: "stencil_resolve_mode` is not one of the modes in the \
                        `supported_stencil_resolve_modes` device property"
                        .into(),
                    vuids: &[
                        "VUID-VkSubpassDescriptionDepthStencilResolve-stencilResolveMode-03184",
                    ],
                    ..Default::default()
                }));
            }
        }

        for (ref_index, input_attachment) in input_attachments
            .iter()
            .enumerate()
            .flat_map(|(i, a)| a.as_ref().map(|a| (i, a)))
        {
            // VUID-VkSubpassDescription2-pInputAttachments-parameter
            input_attachment
                .validate(device)
                .map_err(|err| err.add_context(format!("input_attachments[{}]", ref_index)))?;

            let &AttachmentReference {
                attachment,
                layout,
                stencil_layout,
                aspects,
                _ne: _,
            } = input_attachment;

            if preserve_attachments.contains(&attachment) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`input_attachments[{}].attachment` also occurs in \
                        `preserve_attachments`",
                        ref_index
                    )
                    .into(),
                    vuids: &["VUID-VkSubpassDescription2-pPreserveAttachments-03074"],
                    ..Default::default()
                }));
            }

            if matches!(
                layout,
                ImageLayout::ColorAttachmentOptimal
                    | ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::StencilAttachmentOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: format!("input_attachments[{}].layout", ref_index).into(),
                    problem: "cannot be used with input attachments".into(),
                    vuids: &[
                        "VUID-VkSubpassDescription2-attachment-06912",
                        "VUID-VkSubpassDescription2-attachment-06918",
                    ],
                    ..Default::default()
                }));
            }

            let layouts_entry = Layouts {
                layout,
                stencil_layout,
            };

            match layouts.entry(attachment) {
                Entry::Occupied(entry) => {
                    if *entry.get() != layouts_entry {
                        return Err(Box::new(ValidationError {
                            context: format!("input_attachments[{}].layout", ref_index).into(),
                            problem: "is not equal to the layout used for this attachment \
                                elsewhere in this subpass"
                                .into(),
                            vuids: &["VUID-VkSubpassDescription2-layout-02528"],
                            ..Default::default()
                        }));
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(layouts_entry);
                }
            }

            if aspects.is_empty() {
                return Err(Box::new(ValidationError {
                    context: format!("input_attachments[{}].aspects", ref_index).into(),
                    problem: "is empty for an input attachment".into(),
                    vuids: &["VUID-VkSubpassDescription2-attachment-02800"],
                    ..Default::default()
                }));
            }

            if aspects.intersects(ImageAspects::METADATA) {
                return Err(Box::new(ValidationError {
                    context: format!("input_attachments[{}].aspects", ref_index).into(),
                    problem: "contains `ImageAspects::METADATA`".into(),
                    vuids: &["VUID-VkSubpassDescription2-attachment-02801"],
                    ..Default::default()
                }));
            }

            if aspects.intersects(
                ImageAspects::MEMORY_PLANE_0
                    | ImageAspects::MEMORY_PLANE_1
                    | ImageAspects::MEMORY_PLANE_2
                    | ImageAspects::MEMORY_PLANE_3,
            ) {
                return Err(Box::new(ValidationError {
                    context: format!("input_attachments[{}].aspects", ref_index).into(),
                    problem: "contains `ImageAspects::MEMORY_PLANE_0`, \
                        `ImageAspects::MEMORY_PLANE_1`, `ImageAspects::MEMORY_PLANE_2` or \
                        `ImageAspects::MEMORY_PLANE_3`"
                        .into(),
                    vuids: &["VUID-VkSubpassDescription2-attachment-04563"],
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().multiview && view_mask != 0 {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "is not 0".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("multiview")])]),
                vuids: &["VUID-VkSubpassDescription2-multiview-06558"],
            }));
        }

        let highest_view_index = u32::BITS - view_mask.leading_zeros();

        if highest_view_index >= properties.max_multiview_view_count.unwrap_or(0) {
            return Err(Box::new(ValidationError {
                context: "view_mask".into(),
                problem: "the highest enabled view index is not less than the \
                    `max_multiview_view_count` limit"
                    .into(),
                vuids: &["VUID-VkSubpassDescription2-viewMask-06706"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a render pass subpass description.
    SubpassDescriptionFlags = SubpassDescriptionFlags(u32);

    /* TODO: enable
    // TODO: document
    PER_VIEW_ATTRIBUTES = PER_VIEW_ATTRIBUTES_NVX {
        device_extensions: [nvx_multiview_per_view_attributes],
    }, */

    /* TODO: enable
    // TODO: document
    PER_VIEW_POSITION_X_ONLY = PER_VIEW_POSITION_X_ONLY_NVX{
        device_extensions: [nvx_multiview_per_view_attributes],
    }, */

    /* TODO: enable
    // TODO: document
    FRAGMENT_REGION = FRAGMENT_REGION_QCOM {
        device_extensions: [qcom_render_pass_shader_resolve],
    }, */

    /* TODO: enable
    // TODO: document
    SHADER_RESOLVE = SHADER_RESOLVE_QCOM {
        device_extensions: [qcom_render_pass_shader_resolve],
    }, */

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_EXT {
        device_extensions: [ext_rasterization_order_attachment_access, arm_rasterization_order_attachment_access],
    }, */

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT {
        device_extensions: [ext_rasterization_order_attachment_access, arm_rasterization_order_attachment_access],
    }, */

    /* TODO: enable
    // TODO: document
    RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS = RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT {
        device_extensions: [ext_rasterization_order_attachment_access, arm_rasterization_order_attachment_access],
    }, */

    /* TODO: enable
    // TODO: document
    ENABLE_LEGACY_DITHERING = ENABLE_LEGACY_DITHERING_EXT {
        device_extensions: [ext_legacy_dithering],
    }, */
}

/// A reference to an attachment in a subpass description of a render pass.
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

    /// The `layout` of the stencil aspect of the attachment, if different.
    ///
    /// The layout is restricted by the type of attachment that an attachment is being used as. A
    /// full listing of allowed layouts per type can be found in
    /// [the Vulkan specification](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap8.html#attachment-type-imagelayout).
    ///
    /// If this is `Some`, then the
    /// [`separate_depth_stencil_layouts`](crate::device::Features::separate_depth_stencil_layouts)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub stencil_layout: Option<ImageLayout>,

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
            stencil_layout: None,
            aspects: ImageAspects::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl AttachmentReference {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            attachment: _,
            layout,
            stencil_layout,
            aspects,
            _ne,
        } = self;

        layout.validate_device(device).map_err(|err| {
            err.add_context("layout")
                .set_vuids(&["VUID-VkAttachmentReference2-layout-parameter"])
        })?;

        if matches!(
            layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized | ImageLayout::PresentSrc
        ) {
            return Err(Box::new(ValidationError {
                context: "layout".into(),
                problem: "is `ImageLayout::Undefined`, `ImageLayout::Preinitialized` or \
                    `ImageLayout::PresentSrc`"
                    .into(),
                vuids: &["VUID-VkAttachmentReference2-layout-03077"],
                ..Default::default()
            }));
        }

        if matches!(
            layout,
            ImageLayout::DepthAttachmentOptimal
                | ImageLayout::DepthReadOnlyOptimal
                | ImageLayout::StencilAttachmentOptimal
                | ImageLayout::StencilReadOnlyOptimal
        ) && !device.enabled_features().separate_depth_stencil_layouts
        {
            return Err(Box::new(ValidationError {
                context: "layout".into(),
                problem: "specifies a layout for only the depth aspect or only the stencil aspect"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "separate_depth_stencil_layouts",
                )])]),
                vuids: &["VUID-VkAttachmentReference2-separateDepthStencilLayouts-03313"],
            }));
        }

        if let Some(stencil_layout) = stencil_layout {
            if !device.enabled_features().separate_depth_stencil_layouts {
                return Err(Box::new(ValidationError {
                    context: "stencil_layout".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "separate_depth_stencil_layouts",
                    )])]),
                    ..Default::default()
                }));
            }

            stencil_layout.validate_device(device).map_err(|err| {
                err.add_context("stencil_layout")
                    .set_vuids(&["VUID-VkAttachmentReferenceStencilLayout-stencilLayout-parameter"])
            })?;

            if matches!(
                stencil_layout,
                ImageLayout::Undefined | ImageLayout::Preinitialized | ImageLayout::PresentSrc
            ) {
                return Err(Box::new(ValidationError {
                    context: "stencil_layout".into(),
                    problem: "is `ImageLayout::Undefined`, `ImageLayout::Preinitialized` or \
                        `ImageLayout::PresentSrc`"
                        .into(),
                    vuids: &["VUID-VkAttachmentReferenceStencilLayout-stencilLayout-03318"],
                    ..Default::default()
                }));
            }

            if matches!(
                stencil_layout,
                ImageLayout::ColorAttachmentOptimal
                    | ImageLayout::DepthAttachmentOptimal
                    | ImageLayout::DepthReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "stencil_layout".into(),
                    problem: "does not specify a layout for the stencil aspect".into(),
                    vuids: &["VUID-VkAttachmentReferenceStencilLayout-stencilLayout-03318"],
                    ..Default::default()
                }));
            }

            if matches!(
                stencil_layout,
                ImageLayout::DepthStencilAttachmentOptimal
                    | ImageLayout::DepthStencilReadOnlyOptimal
                    | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                    | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
            ) {
                return Err(Box::new(ValidationError {
                    context: "stencil_layout".into(),
                    problem: "specifies a layout for both the depth and the stencil aspect".into(),
                    vuids: &["VUID-VkAttachmentReferenceStencilLayout-stencilLayout-03318"],
                    ..Default::default()
                }));
            }
        }

        aspects
            .validate_device(device)
            .map_err(|err| err.add_context("aspects"))?;

        Ok(())
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

impl SubpassDependency {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_subpass,
            dst_subpass,
            src_stages,
            dst_stages,
            src_access,
            dst_access,
            dependency_flags,
            view_offset,
            _ne: _,
        } = self;

        // A hack to let us re-use the validation of MemoryBarrier
        // without using it directly inside SubpassDependency.
        let temp_barrier = MemoryBarrier {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            ..Default::default()
        };
        temp_barrier.validate(device)?;

        // To use the extra flag bits from synchronization 2, we also need create_renderpass2,
        // for the ability to use extension structs.
        if !(device.api_version() >= Version::V1_2
            || device.enabled_extensions().khr_create_renderpass2)
        {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_create_renderpass2")]),
                    ]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_create_renderpass2")]),
                    ]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_create_renderpass2")]),
                    ]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_create_renderpass2")]),
                    ]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().synchronization2 {
            if src_stages.is_empty() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "is empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "synchronization2",
                    )])]),
                    vuids: &["VUID-VkSubpassDependency2-srcStageMask-03937"],
                }));
            }

            if dst_stages.is_empty() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "is empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "synchronization2",
                    )])]),
                    vuids: &["VUID-VkSubpassDependency2-dstStageMask-03937"],
                }));
            }
        }

        if src_subpass.is_none() && dst_subpass.is_none() {
            return Err(Box::new(ValidationError {
                problem: "`src_subpass` and `dst_subpass` are both `None`".into(),
                vuids: &["VUID-VkSubpassDependency2-srcSubpass-03085"],
                ..Default::default()
            }));
        }

        if let (Some(src_subpass), Some(dst_subpass)) = (src_subpass, dst_subpass) {
            if src_subpass > dst_subpass {
                return Err(Box::new(ValidationError {
                    problem: "`src_subpass` is greater than `dst_subpass`".into(),
                    vuids: &["VUID-VkSubpassDependency2-srcSubpass-03084"],
                    ..Default::default()
                }));
            }

            if src_subpass == dst_subpass {
                let framebuffer_stages = PipelineStages::EARLY_FRAGMENT_TESTS
                    | PipelineStages::FRAGMENT_SHADER
                    | PipelineStages::LATE_FRAGMENT_TESTS
                    | PipelineStages::COLOR_ATTACHMENT_OUTPUT;

                if src_stages.intersects(framebuffer_stages)
                    && !(dst_stages - framebuffer_stages).is_empty()
                {
                    return Err(Box::new(ValidationError {
                        problem: "`src_subpass` equals `dst_subpass`, and `src_stages` includes \
                            a framebuffer-space stage, and `dst_stages` does not contain only \
                            framebuffer-space stages"
                            .into(),
                        vuids: &["VUID-VkSubpassDependency2-srcSubpass-06810"],
                        ..Default::default()
                    }));
                }

                if src_stages.intersects(framebuffer_stages)
                    && dst_stages.intersects(framebuffer_stages)
                    && !dependency_flags.intersects(DependencyFlags::BY_REGION)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`src_subpass` equals `dst_subpass`, and \
                            `src_stages` and `dst_stages` both include a framebuffer-space stage, \
                            and `dependency_flags` does not include `DependencyFlags::BY_REGION`"
                            .into(),
                        vuids: &["VUID-VkSubpassDependency2-srcSubpass-02245"],
                        ..Default::default()
                    }));
                }

                if view_offset != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`src_subpass` equals `dst_subpass`, and `view_offset` is 0"
                            .into(),
                        vuids: &["VUID-VkSubpassDependency2-viewOffset-02530"],
                        ..Default::default()
                    }));
                }
            }
        }

        if dependency_flags.intersects(DependencyFlags::VIEW_LOCAL) {
            if src_subpass.is_none() {
                return Err(Box::new(ValidationError {
                    problem: "`dependency_flags` includes `DependencyFlags::VIEW_LOCAL`, and \
                        `src_subpass` is `None`"
                        .into(),
                    vuids: &["VUID-VkSubpassDependency2-dependencyFlags-03090"],
                    ..Default::default()
                }));
            }

            if dst_subpass.is_none() {
                return Err(Box::new(ValidationError {
                    problem: "`dependency_flags` includes `DependencyFlags::VIEW_LOCAL`, and \
                        `dst_subpass` is `None`"
                        .into(),
                    vuids: &["VUID-VkSubpassDependency2-dependencyFlags-03091"],
                    ..Default::default()
                }));
            }
        } else {
            if view_offset != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`dependency_flags` does not include `DependencyFlags::VIEW_LOCAL`, \
                        and `view_offset` is not 0"
                        .into(),
                    vuids: &["VUID-VkSubpassDependency2-dependencyFlags-03092"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what the implementation should do with an attachment at the start of the subpass.
    AttachmentLoadOp = AttachmentLoadOp(i32);

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
    None = NONE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_load_store_op_none)]),
    ]),*/
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what the implementation should do with an attachment after all the subpasses have
    /// completed.
    AttachmentStoreOp = AttachmentStoreOp(i32);

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

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AttachmentUse {
    pub(crate) color_attachment: bool,
    pub(crate) depth_stencil_attachment: bool,
    pub(crate) input_attachment: bool,
}

#[cfg(test)]
mod tests {
    use super::{RenderPassCreateInfo, SubpassDescription};
    use crate::{format::Format, render_pass::RenderPass};

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = RenderPass::new(
            device,
            RenderPassCreateInfo {
                subpasses: vec![SubpassDescription::default()],
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn too_many_color_atch() {
        let (device, _) = gfx_dev_and_queue!();

        if device.physical_device().properties().max_color_attachments >= 10 {
            return; // test ignored
        }

        single_pass_renderpass!(
            device,
            attachments: {
                a1: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a2: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a3: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a4: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a5: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a6: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a7: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a8: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a9: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
                a10: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
            },
            pass: {
                color: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10],
                depth_stencil: {},
            },
        )
        .unwrap_err();
    }

    #[test]
    fn non_zero_granularity() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = single_pass_renderpass!(
            device,
            attachments: {
                a: { format: Format::R8G8B8A8_UNORM, samples: 1, load_op: Clear, store_op: DontCare, },
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
