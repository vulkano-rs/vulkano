// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::ClearValue;
use crate::format::Format;
use crate::image::ImageLayout;
use crate::image::SampleCount;
use crate::pipeline::shader::ShaderInterface;
use crate::sync::AccessFlags;
use crate::sync::PipelineStages;

/// The description of a render pass.
#[derive(Clone, Debug)]
pub struct RenderPassDesc {
    attachments: Vec<AttachmentDesc>,
    subpasses: Vec<SubpassDesc>,
    dependencies: Vec<SubpassDependencyDesc>,
    multiview: Option<MultiviewDesc>,
}

impl RenderPassDesc {
    /// Creates a description of a render pass.
    pub fn new(
        attachments: Vec<AttachmentDesc>,
        subpasses: Vec<SubpassDesc>,
        dependencies: Vec<SubpassDependencyDesc>,
    ) -> RenderPassDesc {
        RenderPassDesc {
            attachments,
            subpasses,
            dependencies,
            multiview: None,
        }
    }

    /// Creates a description of a render pass that uses the multiview feature.
    /// See [`MultiviewDesc`] for an explanation of possible configuration options.
    pub fn with_multiview(
        attachments: Vec<AttachmentDesc>,
        subpasses: Vec<SubpassDesc>,
        dependencies: Vec<SubpassDependencyDesc>,
        multiview: MultiviewDesc,
    ) -> RenderPassDesc {
        RenderPassDesc {
            attachments,
            subpasses,
            dependencies,
            multiview: Some(multiview),
        }
    }

    /// Creates a description of an empty render pass, with one subpass and no attachments.
    pub fn empty() -> RenderPassDesc {
        RenderPassDesc {
            attachments: vec![],
            subpasses: vec![SubpassDesc {
                color_attachments: vec![],
                depth_stencil: None,
                input_attachments: vec![],
                resolve_attachments: vec![],
                preserve_attachments: vec![],
            }],
            dependencies: vec![],
            multiview: None,
        }
    }

    // Returns the attachments of the description.
    #[inline]
    pub fn attachments(&self) -> &[AttachmentDesc] {
        &self.attachments
    }

    // Returns the subpasses of the description.
    #[inline]
    pub fn subpasses(&self) -> &[SubpassDesc] {
        &self.subpasses
    }

    // Returns the dependencies of the description.
    #[inline]
    pub fn dependencies(&self) -> &[SubpassDependencyDesc] {
        &self.dependencies
    }

    // Returns the multiview configuration of the description.
    #[inline]
    pub fn multiview(&self) -> &Option<MultiviewDesc> {
        &self.multiview
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

    /// Returns `true` if the subpass of this description is compatible with the shader's fragment
    /// output definition.
    pub fn is_compatible_with_shader(
        &self,
        subpass: u32,
        shader_interface: &ShaderInterface,
    ) -> bool {
        let pass_descr = match self.subpasses.get(subpass as usize) {
            Some(s) => s,
            None => return false,
        };

        for element in shader_interface.elements() {
            for location in element.location.clone() {
                let attachment_id = match pass_descr.color_attachments.get(location as usize) {
                    Some(a) => a.0,
                    None => return false,
                };

                let attachment_desc = &self.attachments[attachment_id];

                // FIXME: compare formats depending on the number of components and data type
                /*if attachment_desc.format != element.format {
                    return false;
                }*/
            }
        }

        true
    }

    /// Returns `true` if this description is compatible with the other description,
    /// as defined in the `Render Pass Compatibility` section of the Vulkan specs.
    // TODO: return proper error
    pub fn is_compatible_with_desc(&self, other: &RenderPassDesc) -> bool {
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
}

impl Default for RenderPassDesc {
    fn default() -> Self {
        Self::empty()
    }
}

/// Describes an attachment that will be used in a render pass.
#[derive(Debug, Clone, Copy)]
pub struct AttachmentDesc {
    /// Format of the image that is going to be bound.
    pub format: Format,
    /// Number of samples of the image that is going to be bound.
    pub samples: SampleCount,

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

impl AttachmentDesc {
    /// Returns true if this attachment is compatible with another attachment, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    #[inline]
    pub fn is_compatible_with(&self, other: &AttachmentDesc) -> bool {
        self.format == other.format && self.samples == other.samples
    }
}

/// Describes one of the subpasses of a render pass.
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
pub struct SubpassDesc {
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

/// Describes a dependency between two subpasses of a render pass.
///
/// The implementation is allowed to change the order of the subpasses within a render pass, unless
/// you specify that there exists a dependency between two subpasses (ie. the result of one will be
/// used as the input of another one).
#[derive(Debug, Clone, Copy)]
pub struct SubpassDependencyDesc {
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
    pub source_access: AccessFlags,

    /// The way the destination subpass accesses the attachments on which we depend.
    pub destination_access: AccessFlags,

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
#[repr(i32)]
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

/// Describes what the implementation should do with an attachment at the start of the subpass.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
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
    ///
    /// See the `draw_inline` and `draw_secondary` methods of `PrimaryComputeBufferBuilder`.
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

/// Describes the `multiview` configuration for the render pass which is used to draw
/// to multiple layers of a framebuffer inside of a single render pass.
#[derive(Debug, Clone)]
pub struct MultiviewDesc {
    /// The view masks indicate which layers of the framebuffer should be rendered for each subpass.
    /// Values are bit masks which means that for example `0b11` will draw to the first two layers
    /// and `0b101` will draw to the first and third layer.
    pub view_masks: Vec<u32>,

    /// The correlation masks indicate sets of views that may be more efficient to render
    /// concurrently (usually because they show the same geometry from almost the same perspective).
    /// Values are bit masks which means that for example `0b11` means the first two layers are
    /// highly correlated and `0b101` means the first and third layer are highly correlated.
    pub correlation_masks: Vec<u32>,

    /// The view offsets contain additional information for each subpass dependency that indicate
    /// which views in the source subpass the views of the destination subpass depend on.
    pub view_offsets: Vec<i32>,
}

impl MultiviewDesc {
    /// Returns the index of the layer with the biggest index that is
    /// referred to by a mask in the multiview description.
    pub fn highest_used_layer(&self) -> u32 {
        self.view_masks
            .iter()
            .chain(self.correlation_masks.iter())
            .map(|&mask| 32 - mask.leading_zeros()) // the highest set bit corresponds to the highest used layer
            .max()
            .unwrap_or(0)
    }

    /// Returns the amount of layers that are used in the multiview description.
    pub fn used_layer_count(&self) -> u32 {
        self.view_masks
            .iter()
            .chain(self.correlation_masks.iter())
            .fold(0, |acc, &mask| acc | mask)
            .count_ones()
    }
}

/// Possible resolve modes for depth and stencil attachments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
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
