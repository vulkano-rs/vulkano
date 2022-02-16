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

pub use self::create::RenderPassCreationError;
pub use self::framebuffer::Framebuffer;
pub use self::framebuffer::FramebufferCreateInfo;
pub use self::framebuffer::FramebufferCreationError;
use crate::{
    device::{Device, DeviceOwned},
    format::{ClearValue, Format},
    image::{ImageAspects, ImageLayout, SampleCount},
    shader::ShaderInterface,
    sync::{AccessFlags, PipelineStages},
    Version, VulkanObject,
};
use std::{
    hash::{Hash, Hasher},
    mem::MaybeUninit,
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
/// let render_pass = single_pass_renderpass!(device.clone(),
///     attachments: {
///         // `foo` is a custom name we give to the first and only attachment.
///         foo: {
///             load: Clear,
///             store: Store,
///             format: Format::R8G8B8A8_UNORM,
///             samples: 1,
///         }
///     },
///     pass: {
///         color: [foo],       // Repeat the attachment name here.
///         depth_stencil: {}
///     }
/// ).unwrap();
/// # }
/// ```
///
/// See the documentation of the macro for more details. TODO: put link here
#[derive(Debug)]
pub struct RenderPass {
    handle: ash::vk::RenderPass,
    device: Arc<Device>,

    attachments: Vec<AttachmentDescription>,
    subpasses: Vec<SubpassDescription>,
    dependencies: Vec<SubpassDependency>,
    correlated_view_masks: Vec<u32>,

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
        let mut views_used = 0;
        Self::validate(&device, &mut create_info, &mut views_used)?;

        let handle = if device.api_version() >= Version::V1_2
            || device.enabled_extensions().khr_create_renderpass2
        {
            Self::create_v2(&device, &create_info)?
        } else {
            Self::create_v1(&device, &create_info)?
        };

        let RenderPassCreateInfo {
            attachments,
            subpasses,
            dependencies,
            correlated_view_masks,
            _ne: _,
        } = create_info;

        let granularity = unsafe {
            let fns = device.fns();
            let mut out = MaybeUninit::uninit();
            fns.v1_0.get_render_area_granularity(
                device.internal_object(),
                handle,
                out.as_mut_ptr(),
            );

            let out = out.assume_init();
            debug_assert_ne!(out.width, 0);
            debug_assert_ne!(out.height, 0);
            [out.width, out.height]
        };

        Ok(Arc::new(RenderPass {
            handle,
            device,

            attachments,
            subpasses,
            dependencies,
            correlated_view_masks,

            granularity,
            views_used,
        }))
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
    /// as defined in the `Render Pass Compatibility` section of the Vulkan specs.
    // TODO: return proper error
    pub fn is_compatible_with(&self, other: &RenderPass) -> bool {
        if self.attachments().len() != other.attachments().len() {
            return false;
        }

        for (my_atch, other_atch) in self.attachments.iter().zip(other.attachments.iter()) {
            if !my_atch.is_compatible_with(&other_atch) {
                return false;
            }
        }

        return true;

        // FIXME: finish
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

                let attachment_desc = &self.attachments[attachment_id as usize];

                // FIXME: compare formats depending on the number of components and data type
                /*if attachment_desc.format != element.format {
                    return false;
                }*/
            }
        }

        true
    }

    /// Decodes `I` into a list of clear values where each element corresponds
    /// to an attachment. The size of the returned iterator must be the same as the number of
    /// attachments.
    ///
    /// When the user enters a render pass, they need to pass a list of clear values to apply to
    /// the attachments of the framebuffer. This method is then responsible for checking the
    /// correctness of these values and turning them into a list that can be processed by vulkano.
    ///
    /// The format of the clear value **must** match the format of the attachment. Attachments
    /// that are not loaded with `LoadOp::Clear` must have an entry equal to `ClearValue::None`.
    pub fn convert_clear_values<I>(&self, values: I) -> impl Iterator<Item = ClearValue>
    where
        I: IntoIterator<Item = ClearValue>,
    {
        // FIXME: safety checks
        values.into_iter()
    }
}

impl Drop for RenderPass {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_render_pass(self.device.internal_object(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for RenderPass {
    type Object = ash::vk::RenderPass;

    #[inline]
    fn internal_object(&self) -> ash::vk::RenderPass {
        self.handle
    }
}

unsafe impl DeviceOwned for RenderPass {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for RenderPass {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for RenderPass {}

impl Hash for RenderPass {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device.hash(state);
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
        if (id as usize) < render_pass.subpasses().len() {
            Some(Subpass {
                render_pass,
                subpass_id: id,
            })
        } else {
            None
        }
    }

    /// Returns the subpass description for this subpass.
    #[inline]
    pub fn subpass_desc(&self) -> &SubpassDescription {
        &self.render_pass.subpasses()[self.subpass_id as usize]
    }

    /// Returns whether this subpass is the last one in the render pass. If `true` is returned,
    /// `next_subpass` will return `None`.
    #[inline]
    pub fn is_last_subpass(&self) -> bool {
        self.subpass_id as usize == self.render_pass.subpasses().len() - 1
    }

    /// Tries to advance to the next subpass after this one, and returns `true` if successful.
    #[inline]
    pub fn try_next_subpass(&mut self) -> bool {
        let next_id = self.subpass_id + 1;

        if (next_id as usize) < self.render_pass.subpasses().len() {
            self.subpass_id = next_id;
            true
        } else {
            false
        }
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
            .map_or(false, |f| f.aspects().depth)
    }

    /// Returns true if the subpass has a depth attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    pub fn has_writable_depth(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        let atch_num = match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => {
                if atch_ref.layout == ImageLayout::DepthStencilReadOnlyOptimal {
                    return false;
                }
                atch_ref.attachment
            }
            None => return false,
        };

        self.attachment_desc(atch_num)
            .format
            .map_or(false, |f| f.aspects().depth)
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
            .map_or(false, |f| f.aspects().stencil)
    }

    /// Returns true if the subpass has a stencil attachment or a depth-stencil attachment whose
    /// layout is not `DepthStencilReadOnlyOptimal`.
    #[inline]
    pub fn has_writable_stencil(&self) -> bool {
        let subpass_desc = self.subpass_desc();

        let atch_num = match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => {
                if atch_ref.layout == ImageLayout::DepthStencilReadOnlyOptimal {
                    return false;
                }
                atch_ref.attachment
            }
            None => return false,
        };

        self.attachment_desc(atch_num)
            .format
            .map_or(false, |f| f.aspects().stencil)
    }

    /// Returns true if the subpass has any depth/stencil attachment.
    #[inline]
    pub fn has_depth_stencil_attachment(&self) -> bool {
        let subpass_desc = self.subpass_desc();
        match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => true,
            None => false,
        }
    }

    /// Returns true if the subpass has any color or depth/stencil attachment.
    #[inline]
    pub fn has_color_or_depth_stencil_attachment(&self) -> bool {
        if self.num_color_attachments() >= 1 {
            return true;
        }

        let subpass_desc = self.subpass_desc();
        match &subpass_desc.depth_stencil_attachment {
            Some(atch_ref) => true,
            None => false,
        }
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

impl AttachmentDescription {
    /// Returns true if this attachment is compatible with another attachment, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    #[inline]
    pub fn is_compatible_with(&self, other: &AttachmentDescription) -> bool {
        self.format == other.format && self.samples == other.samples
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
    /// [the Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap8.html#attachment-type-imagelayout).
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
    /// The default value is [`ImageAspects::none()`].
    pub aspects: ImageAspects,

    pub _ne: crate::NonExhaustive,
}

impl Default for AttachmentReference {
    #[inline]
    fn default() -> Self {
        Self {
            attachment: 0,
            layout: ImageLayout::Undefined,
            aspects: ImageAspects::none(),
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
/// If `source_subpass` and `destination_subpass` are equal, then this specifies a
/// [subpass self-dependency](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-barriers-subpass-self-dependencies).
/// The `source_stages` must all be
/// [logically earlier in the pipeline](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-order)
/// than the `destination_stages`, and if they both contain a
/// [framebuffer-space stage](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-framebuffer-regions),
/// then `by_region` must be activated.
///
/// If `source_subpass` or `destination_subpass` are set to `None`, this specifies an external
/// dependency. An external dependency specifies a dependency on commands that were submitted before
/// the render pass instance began (for `source_subpass`), or on commands that will be submitted
/// after the render pass instance ends (for `destination_subpass`). The values must not both be
/// `None`.
#[derive(Clone, Debug)]
pub struct SubpassDependency {
    /// The index of the subpass that writes the data that `destination_subpass` is going to use.
    ///
    /// `None` specifies an external dependency.
    ///
    /// The default value is `None`.
    pub source_subpass: Option<u32>,

    /// The index of the subpass that reads the data that `source_subpass` wrote.
    ///
    /// `None` specifies an external dependency.
    ///
    /// The default value is `None`.
    pub destination_subpass: Option<u32>,

    /// The pipeline stages that must be finished on `source_subpass` before the
    /// `destination_stages` of `destination_subpass` can start.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub source_stages: PipelineStages,

    /// The pipeline stages of `destination_subpass` that must wait for the `source_stages` of
    /// `source_subpass` to be finished. Stages that are earlier than the stages specified here can
    /// start before the `source_stages` are finished.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub destination_stages: PipelineStages,

    /// The way `source_subpass` accesses the attachments on which we depend.
    ///
    /// The default value is [`AccessFlags::none()`].
    pub source_access: AccessFlags,

    /// The way `destination_subpass` accesses the attachments on which we depend.
    ///
    /// The default value is [`AccessFlags::none()`].
    pub destination_access: AccessFlags,

    /// If false, then the source operations must be fully finished for the destination operations
    /// to start. If true, then the implementation can start the destination operation for some
    /// given pixels as long as the source operation is finished for these given pixels.
    ///
    /// In other words, if the previous subpass has some side effects on other parts of an
    /// attachment, then you should set it to false.
    ///
    /// Passing `false` is always safer than passing `true`, but in practice you rarely need to
    /// pass `false`.
    ///
    /// The default value is `false`.
    pub by_region: bool,

    /// If multiview rendering is being used (the subpasses have a nonzero `view_mask`), then
    /// setting this to `Some` creates a view-local dependency, between views in `source_subpass`
    /// and views in `destination_subpass`.
    ///
    /// The inner value specifies an offset relative to the view index of `destination_subpass`:
    /// each view `d` in `destination_subpass` depends on view `d + view_offset` in
    /// `source_subpass`. If the source view index does not exist, the dependency is ignored for
    /// that view.
    ///
    /// If multiview rendering is not being used, the value must be `None`. If `source_subpass`
    /// and `destination_subpass` are the same, only `Some(0)` and `None` are allowed as values, and
    /// if that subpass also has multiple bits set in its `view_mask`, the value must be `Some(0)`.
    ///
    /// The default value is `None`.
    pub view_local: Option<i32>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SubpassDependency {
    #[inline]
    fn default() -> Self {
        Self {
            source_subpass: None,
            destination_subpass: None,
            source_stages: PipelineStages::none(),
            destination_stages: PipelineStages::none(),
            source_access: AccessFlags::none(),
            destination_access: AccessFlags::none(),
            by_region: false,
            view_local: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Describes what the implementation should do with an attachment at the start of the subpass.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum LoadOp {
    /// The content of the attachment will be loaded from memory. This is what you want if you want
    /// to draw over something existing.
    ///
    /// While this is the most intuitive option, it is also the slowest because it uses a lot of
    /// memory bandwidth.
    Load = ash::vk::AttachmentLoadOp::LOAD.as_raw(),

    /// The content of the attachment will be filled by the implementation with a uniform value
    /// that you must provide when you start drawing.
    ///
    /// This is what you usually use at the start of a frame, in order to reset the content of
    /// the color, depth and/or stencil buffers.
    Clear = ash::vk::AttachmentLoadOp::CLEAR.as_raw(),

    /// The attachment will have undefined content.
    ///
    /// This is what you should use for attachments that you intend to entirely cover with draw
    /// commands.
    /// If you are going to fill the attachment with a uniform value, it is better to use `Clear`
    /// instead.
    DontCare = ash::vk::AttachmentLoadOp::DONT_CARE.as_raw(),
}

impl From<LoadOp> for ash::vk::AttachmentLoadOp {
    #[inline]
    fn from(val: LoadOp) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Describes what the implementation should do with an attachment after all the subpasses have
/// completed.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum StoreOp {
    /// The attachment will be stored. This is what you usually want.
    ///
    /// While this is the most intuitive option, it is also slower than `DontCare` because it can
    /// take time to write the data back to memory.
    Store = ash::vk::AttachmentStoreOp::STORE.as_raw(),

    /// What happens is implementation-specific.
    ///
    /// This is purely an optimization compared to `Store`. The implementation doesn't need to copy
    /// from the internal cache to the memory, which saves memory bandwidth.
    ///
    /// This doesn't mean that the data won't be copied, as an implementation is also free to not
    /// use a cache and write the output directly in memory. In other words, the content of the
    /// image will be undefined.
    DontCare = ash::vk::AttachmentStoreOp::DONT_CARE.as_raw(),
}

impl From<StoreOp> for ash::vk::AttachmentStoreOp {
    #[inline]
    fn from(val: StoreOp) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Possible resolve modes for depth and stencil attachments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum ResolveMode {
    None = ash::vk::ResolveModeFlags::NONE.as_raw(),
    SampleZero = ash::vk::ResolveModeFlags::SAMPLE_ZERO.as_raw(),
    Average = ash::vk::ResolveModeFlags::AVERAGE.as_raw(),
    Min = ash::vk::ResolveModeFlags::MIN.as_raw(),
    Max = ash::vk::ResolveModeFlags::MAX.as_raw(),
}

#[derive(Clone, Copy, Debug)]
pub struct ResolveModes {
    pub none: bool,
    pub sample_zero: bool,
    pub average: bool,
    pub min: bool,
    pub max: bool,
}

impl From<ash::vk::ResolveModeFlags> for ResolveModes {
    #[inline]
    fn from(val: ash::vk::ResolveModeFlags) -> Self {
        Self {
            none: val.intersects(ash::vk::ResolveModeFlags::NONE),
            sample_zero: val.intersects(ash::vk::ResolveModeFlags::SAMPLE_ZERO),
            average: val.intersects(ash::vk::ResolveModeFlags::AVERAGE),
            min: val.intersects(ash::vk::ResolveModeFlags::MIN),
            max: val.intersects(ash::vk::ResolveModeFlags::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::format::Format;
    use crate::render_pass::RenderPass;
    use crate::render_pass::RenderPassCreationError;

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

        let rp = single_pass_renderpass! {
            device.clone(),
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
                a10: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, }
            },
            pass: {
                color: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10],
                depth_stencil: {}
            }
        };

        match rp {
            Err(RenderPassCreationError::SubpassMaxColorAttachmentsExceeded { .. }) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn non_zero_granularity() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = single_pass_renderpass! {
            device.clone(),
            attachments: {
                a: { load: Clear, store: DontCare, format: Format::R8G8B8A8_UNORM, samples: 1, }
            },
            pass: {
                color: [a],
                depth_stencil: {}
            }
        }
        .unwrap();

        let granularity = rp.granularity();
        assert_ne!(granularity[0], 0);
        assert_ne!(granularity[1], 0);
    }
}
