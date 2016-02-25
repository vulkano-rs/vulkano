//! Targets on which your draw commands are executed.
//! 
//! There are two concepts in Vulkan:
//! 
//! - A `RenderPass` is a collection of rendering passes (called subpasses). Each subpass contains
//!   the list of attachments that are written to when drawing. The `RenderPass` only defines the
//!   formats and dimensions of all the attachments of the multiple subpasses.
//! - A `Framebuffer` defines the actual images that are attached. The format of the images must
//!   match what the `RenderPass` expects.
//!
//! Creating a `RenderPass` is necessary before you create a graphics pipeline.
//! A `Framebuffer`, however, is only needed when you build the command buffer.
//!
//! # Creating a RenderPass
//! 
//! Creating a `RenderPass` in the vulkano library is best done with the
//! `single_pass_renderpass!` macro.
//!
//! This macro creates an inaccessible struct which implements the `RenderPassLayout` trait. This
//! trait tells vulkano what the characteristics of the renderpass are, and is also used to
//! determine the types of the various parameters later on.
//!
use std::error;
use std::fmt;
use std::iter;
use std::iter::Empty as EmptyIter;
use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::ptr;
use std::sync::Arc;

use device::Device;
use format::ClearValue;
use format::Format;
use format::FormatDesc;
use image::AbstractImageView;
use image::Layout as ImageLayout;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Types that describes the characteristics of a renderpass.
pub unsafe trait RenderPassLayout {
    /// The list of clear values to use when beginning to draw on this renderpass.
    type ClearValues;

    /// Iterator that produces one clear value per attachment.
    type ClearValuesIter: Iterator<Item = ClearValue>;

    /// Decodes a `ClearValues` into a list of clear values where each element corresponds
    /// to an attachment. The size of the returned array must be the same as the number of
    /// attachments.
    ///
    /// The format of the clear value **must** match the format of the attachment. Only attachments
    /// that are loaded with `LoadOp::Clear` must have an entry in the array.
    fn convert_clear_values(&self, Self::ClearValues) -> Self::ClearValuesIter;

    /// Iterator that produces attachments.
    type AttachmentsDescIter: ExactSizeIterator<Item = AttachmentDescription>;

    /// Returns the descriptions of the attachments.
    fn attachments(&self) -> Self::AttachmentsDescIter;

    /// Iterator that produces passes.
    type PassesIter: ExactSizeIterator<Item = PassDescription>;

    /// Returns the descriptions of the passes.
    fn passes(&self) -> Self::PassesIter;

    /// Iterator that produces pass dependencies.
    type PassDependenciesIter: ExactSizeIterator<Item = PassDependencyDescription>;

    /// Returns the descriptions of the dependencies between passes.
    fn pass_dependencies(&self) -> Self::PassDependenciesIter;

    /// List of images that will be binded to attachments.
    ///
    /// A parameter of this type must be passed when creating a `Framebuffer`.
    // TODO: should use HKTs so that attachments list can get passed references to the attachments
    type AttachmentsList;

    /// A decoded `AttachmentsList`.
    // TODO: should be AbstractImageView or something like that, so that images can't get passed
    type AttachmentsIter: ExactSizeIterator<Item = Arc<AbstractImageView>>;

    /// Decodes a `AttachmentsList` into a list of attachments.
    fn convert_attachments_list(&self, Self::AttachmentsList) -> Self::AttachmentsIter;
}

/// Trait implemented on renderpass layouts to check whether they are compatible
/// with another layout.
///
/// The trait is automatically implemented for all type that implement `RenderPassLayout`.
// TODO: once specialization lands, this trait can be specialized for pairs that are known to
//       always be compatible
// TODO: maybe this can be unimplemented on some pairs, to provide compile-time checks?
pub unsafe trait CompatibleLayout<Other>: RenderPassLayout where Other: RenderPassLayout {
    /// Returns `true` if this layout is compatible with the other layout, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    fn is_compatible_with(&self, other: &Other) -> bool;
}

unsafe impl<A, B> CompatibleLayout<B> for A
    where A: RenderPassLayout, B: RenderPassLayout
{
    fn is_compatible_with(&self, other: &B) -> bool {
        for (atch1, atch2) in self.attachments().zip(other.attachments()) {
            if !atch1.is_compatible_with(&atch2) {
                return false;
            }
        }

        return true;

        // FIXME: finish
    }
}

/// Describes an attachment that will be used in a renderpass.
pub struct AttachmentDescription {
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

impl AttachmentDescription {
    /// Returns true if this attachment is compatible with another attachment, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    #[inline]
    pub fn is_compatible_with(&self, other: &AttachmentDescription) -> bool {
        self.format == other.format && self.samples == other.samples
    }
}

/// Describes one of the passes of a renderpass.
///
/// # Restrictions
///
/// All these restrictions are checked when the `RenderPass` object is created.
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
pub struct PassDescription {
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

/// Describes a dependency between two passes of a renderpass.
///
/// The implementation is allowed to change the order of the passes within a renderpass, unless
/// you specify that there exists a dependency between two passes (ie. the result of one will be
/// used as the input of another one).
// FIXME: finish
pub struct PassDependencyDescription {
    /// Index of the subpass that writes the data that `destination_subpass` is going to use.
    pub source_subpass: usize,

    /// Index of the subpass that reads the data that `source_subpass` wrote.
    pub destination_subpass: usize,

    /*VkPipelineStageFlags    srcStageMask;
    VkPipelineStageFlags    dstStageMask;
    VkAccessFlags           srcAccessMask;
    VkAccessFlags           dstAccessMask;*/

    pub by_region: bool,
}

/// Implementation of `RenderPassLayout` with no attachment at all and a single pass.
#[derive(Debug, Copy, Clone)]
pub struct EmptySinglePassLayout;

unsafe impl RenderPassLayout for EmptySinglePassLayout {
    type ClearValues = ();
    type ClearValuesIter = EmptyIter<ClearValue>;

    #[inline]
    fn convert_clear_values(&self, _: Self::ClearValues) -> Self::ClearValuesIter {
        iter::empty()
    }

    type AttachmentsDescIter = EmptyIter<AttachmentDescription>;

    #[inline]
    fn attachments(&self) -> Self::AttachmentsDescIter {
        iter::empty()
    }

    type PassesIter = OptionIntoIter<PassDescription>;

    #[inline]
    fn passes(&self) -> Self::PassesIter {
        Some(PassDescription {
            color_attachments: vec![],
            depth_stencil: None,
            input_attachments: vec![],
            resolve_attachments: vec![],
            preserve_attachments: vec![],
        }).into_iter()
    }

    type PassDependenciesIter = EmptyIter<PassDependencyDescription>;

    #[inline]
    fn pass_dependencies(&self) -> Self::PassDependenciesIter {
        iter::empty()
    }

    type AttachmentsList = ();
    type AttachmentsIter = EmptyIter<Arc<AbstractImageView>>;

    #[inline]
    fn convert_attachments_list(&self, _: Self::AttachmentsList) -> Self::AttachmentsIter {
        iter::empty()
    }
}



/// Builds a `RenderPass` object.
#[macro_export]
macro_rules! single_pass_renderpass {
    (
        device: $device:expr,
        attachments: {
            $(
                $atch_name:ident: {
                    load: $load:ident,
                    store: $store:ident,
                    format: $format:ident,
                }
            ),*
        },
        pass: {
            color: [$($color_atch:ident),*],
            depth_stencil: {$($depth_atch:ident)*}
        }
    ) => (
        {
            use std::sync::Arc;

            struct Layout;
            unsafe impl $crate::framebuffer::RenderPassLayout for Layout {
                type ClearValues = ($(<$crate::format::$format as $crate::format::FormatDesc>::ClearValue,)*);
                type ClearValuesIter = std::vec::IntoIter<$crate::format::ClearValue>;
                type AttachmentsDescIter = std::vec::IntoIter<$crate::framebuffer::AttachmentDescription>;
                type PassesIter = std::option::IntoIter<$crate::framebuffer::PassDescription>;
                type PassDependenciesIter = std::option::IntoIter<$crate::framebuffer::PassDependencyDescription>;
                type AttachmentsIter = std::vec::IntoIter<std::sync::Arc<$crate::image::AbstractImageView>>;

                // FIXME: should be stronger-typed
                type AttachmentsList = (
                    $(
                        Arc<$crate::image::AbstractTypedImageView<$crate::image::Type2d, $crate::format::$format>>,
                    )*
                );

                #[inline]
                fn convert_clear_values(&self, val: Self::ClearValues) -> Self::ClearValuesIter {
                    $crate::format::ClearValuesTuple::iter(val)
                }

                #[inline]
                fn attachments(&self) -> Self::AttachmentsDescIter {
                    vec![
                        $(
                            $crate::framebuffer::AttachmentDescription {
                                format: $crate::format::FormatDesc::format(&$crate::format::$format),      // FIXME: only works with markers
                                samples: 1,                         // FIXME:
                                load: $crate::framebuffer::LoadOp::$load,
                                store: $crate::framebuffer::StoreOp::$store,
                                initial_layout: $crate::image::Layout::PresentSrc,       // FIXME:
                                final_layout: $crate::image::Layout::PresentSrc,       // FIXME:
                            },
                        )*
                    ].into_iter()
                }

                #[inline]
                #[allow(unused_mut)]
                #[allow(unused_assignments)]
                fn passes(&self) -> Self::PassesIter {
                    let mut attachment_num = 0;
                    $(
                        let $atch_name = attachment_num;
                        attachment_num += 1;
                    )*

                    let mut depth = None;
                    $(
                        depth = Some(($depth_atch, $crate::image::Layout::DepthStencilAttachmentOptimal));
                    )*

                    Some(
                        $crate::framebuffer::PassDescription {
                            color_attachments: vec![
                                $(
                                    ($color_atch, $crate::image::Layout::ColorAttachmentOptimal)
                                ),*
                            ],
                            depth_stencil: depth,
                            input_attachments: vec![],
                            resolve_attachments: vec![],
                            preserve_attachments: vec![],
                        }
                    ).into_iter()
                }

                #[inline]
                fn pass_dependencies(&self) -> Self::PassDependenciesIter {
                    None.into_iter()
                }

                #[inline]
                fn convert_attachments_list(&self, l: Self::AttachmentsList) -> Self::AttachmentsIter {
                    $crate::image::AbstractTypedImageViewsTuple::iter(l)
                }
            }

            $crate::framebuffer::RenderPass::<Layout>::new($device, Layout)
        }
    );
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

/// Defines the layout of multiple subpasses.
pub struct RenderPass<L> {
    device: Arc<Device>,
    renderpass: vk::RenderPass,
    num_passes: u32,
    layout: L,
}

impl<L> RenderPass<L> where L: RenderPassLayout {
    /// Builds a new renderpass.
    ///
    /// This function calls the methods of the `RenderPassLayout` implementation and builds the
    /// corresponding Vulkan object.
    ///
    /// # Panic
    ///
    /// - Panicks if the layout described by the `RenderPassLayout` implementation is invalid.
    ///   See the documentation of the various methods and structs related to `RenderPassLayout`
    ///   for more details.
    ///
    pub fn new(device: &Arc<Device>, layout: L) -> Result<Arc<RenderPass<L>>, OomError> {
        let vk = device.pointers();

        // TODO: check the validity of the renderpass layout with debug_assert!

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let attachments = layout.attachments().map(|attachment| {
            vk::AttachmentDescription {
                flags: 0,       // FIXME: may alias flag
                format: attachment.format as u32,
                samples: attachment.samples,
                loadOp: attachment.load as u32,
                storeOp: attachment.store as u32,
                stencilLoadOp: attachment.load as u32,       // TODO: allow user to choose
                stencilStoreOp: attachment.store as u32,      // TODO: allow user to choose
                initialLayout: attachment.initial_layout as u32,
                finalLayout: attachment.final_layout as u32,
            }
        }).collect::<Vec<_>>();

        // We need to pass pointers to vkAttachmentReference structs when creating the renderpass.
        // Therefore we need to allocate them in advance.
        //
        // This block allocates, for each pass, in order, all color attachment references, then all
        // input attachment references, then all resolve attachment references, then the depth
        // stencil attachment reference.
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let attachment_references = layout.passes().flat_map(|pass| {
            debug_assert!(pass.resolve_attachments.is_empty() ||
                          pass.resolve_attachments.len() == pass.color_attachments.len());
            let resolve = pass.resolve_attachments.into_iter().map(|(offset, img_la)| {
                debug_assert!(offset < layout.attachments().len());
                vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, }
            });

            let color = pass.color_attachments.into_iter().map(|(offset, img_la)| {
                debug_assert!(offset < layout.attachments().len());
                vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, }
            });

            let input = pass.input_attachments.into_iter().map(|(offset, img_la)| {
                debug_assert!(offset < layout.attachments().len());
                vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, }
            });

            let depthstencil = if let Some((offset, img_la)) = pass.depth_stencil {
                Some(vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, })
            } else {
                None
            }.into_iter();

            color.chain(input).chain(resolve).chain(depthstencil)
        }).collect::<Vec<_>>();

        // Same as `attachment_references` but only for the preserve attachments.
        // This is separate because attachment references are u32s and not `vkAttachmentReference`
        // structs.
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let preserve_attachments_references = layout.passes().flat_map(|pass| {
            pass.preserve_attachments.into_iter().map(|offset| offset as u32)
        }).collect::<Vec<_>>();

        // Now iterating over passes.
        // `ref_index` and `preserve_ref_index` are increased during the loop and point to the
        // next element to use in respectively `attachment_references` and
        // `preserve_attachments_references`.
        let mut ref_index = 0usize;
        let mut preserve_ref_index = 0usize;
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let passes = layout.passes().map(|pass| {
            unsafe {
                assert!(pass.color_attachments.len() as u32 <=
                        device.physical_device().limits().max_color_attachments());

                let color_attachments = attachment_references.as_ptr().offset(ref_index as isize);
                ref_index += pass.color_attachments.len();
                let input_attachments = attachment_references.as_ptr().offset(ref_index as isize);
                ref_index += pass.input_attachments.len();
                let resolve_attachments = attachment_references.as_ptr().offset(ref_index as isize);
                ref_index += pass.resolve_attachments.len();
                let depth_stencil = if pass.depth_stencil.is_some() {
                    let a = attachment_references.as_ptr().offset(ref_index as isize);
                    ref_index += 1;
                    a
                } else {
                    ptr::null()
                };

                let preserve_attachments = preserve_attachments_references.as_ptr().offset(preserve_ref_index as isize);
                preserve_ref_index += pass.preserve_attachments.len();

                vk::SubpassDescription {
                    flags: 0,   // reserved
                    pipelineBindPoint: vk::PIPELINE_BIND_POINT_GRAPHICS,
                    inputAttachmentCount: pass.input_attachments.len() as u32,
                    pInputAttachments: input_attachments,
                    colorAttachmentCount: pass.color_attachments.len() as u32,
                    pColorAttachments: color_attachments,
                    pResolveAttachments: if pass.resolve_attachments.len() == 0 { ptr::null() } else { resolve_attachments },
                    pDepthStencilAttachment: depth_stencil,
                    preserveAttachmentCount: pass.preserve_attachments.len() as u32,
                    pPreserveAttachments: preserve_attachments,
                }
            }
        }).collect::<Vec<_>>();

        assert!(!passes.is_empty());
        // If these assertions fails, there's a serious bug in the code above ^.
        debug_assert!(ref_index == attachment_references.len());
        debug_assert!(preserve_ref_index == preserve_attachments_references.len());

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let dependencies = layout.pass_dependencies().map(|dependency| {
            debug_assert!(dependency.source_subpass < layout.passes().len());
            debug_assert!(dependency.destination_subpass < layout.passes().len());

            vk::SubpassDependency {
                srcSubpass: dependency.source_subpass as u32,
                dstSubpass: dependency.destination_subpass as u32,
                srcStageMask: vk::PIPELINE_STAGE_ALL_COMMANDS_BIT,      // FIXME:
                dstStageMask: vk::PIPELINE_STAGE_ALL_COMMANDS_BIT,      // FIXME:
                srcAccessMask: vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,       // FIXME:
                dstAccessMask: vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,       // FIXME:
                dependencyFlags: if dependency.by_region { vk::DEPENDENCY_BY_REGION_BIT } else { 0 },
            }
        }).collect::<Vec<_>>();

        let renderpass = unsafe {
            let infos = vk::RenderPassCreateInfo {
                sType: vk::STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                attachmentCount: attachments.len() as u32,
                pAttachments: attachments.as_ptr(),
                subpassCount: passes.len() as u32,
                pSubpasses: passes.as_ptr(),
                dependencyCount: dependencies.len() as u32,
                pDependencies: dependencies.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateRenderPass(device.internal_object(), &infos,
                                                  ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(RenderPass {
            device: device.clone(),
            renderpass: renderpass,
            num_passes: passes.len() as u32,
            layout: layout,
        }))
    }
}

impl RenderPass<EmptySinglePassLayout> {
    /// Builds a `RenderPass` with no attachment and a single pass.
    #[inline]
    pub fn empty_single_pass(device: &Arc<Device>)
                             -> Result<Arc<RenderPass<EmptySinglePassLayout>>, OomError>
    {
        RenderPass::new(device, EmptySinglePassLayout)
    }
}

impl<L> RenderPass<L> where L: RenderPassLayout {
    /// Returns the device that was used to create this renderpass.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the number of subpasses.
    #[inline]
    pub fn num_subpasses(&self) -> u32 {
        self.num_passes
    }

    /// Returns a handle that represents a subpass of this renderpass.
    #[inline]
    pub fn subpass(&self, id: u32) -> Option<Subpass<L>> {
        if id < self.num_passes {
            Some(Subpass {
                renderpass: self,
                subpass_id: id,
            })

        } else {
            None
        }
    }

    /// Returns true if this renderpass is compatible with another one.
    ///
    /// This means that framebuffers created with this renderpass can also be used alongside with
    /// the other renderpass.
    #[inline]
    pub fn is_compatible_with<R2>(&self, other: &RenderPass<R2>) -> bool
        where R2: RenderPassLayout
    {
        self.layout.is_compatible_with(&other.layout)
    }

    /// Returns the layout used to create this renderpass.
    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

unsafe impl<L> VulkanObject for RenderPass<L> {
    type Object = vk::RenderPass;

    #[inline]
    fn internal_object(&self) -> vk::RenderPass {
        self.renderpass
    }
}

impl<L> Drop for RenderPass<L> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyRenderPass(self.device.internal_object(), self.renderpass, ptr::null());
        }
    }
}

/// Represents a subpass within a `RenderPass`.
///
/// This struct doesn't correspond to anything in Vulkan. It is simply an equivalent to a
/// combination of a render pass and subpass ID.
#[derive(Copy, Clone)]
pub struct Subpass<'a, L: 'a> {
    renderpass: &'a RenderPass<L>,
    subpass_id: u32,
}

impl<'a, L: 'a> Subpass<'a, L> {
    /// Returns the renderpass of this subpass.
    #[inline]
    pub fn renderpass(&self) -> &'a RenderPass<L> {
        self.renderpass
    }

    /// Returns the index of this subpass within the renderpass.
    #[inline]
    pub fn index(&self) -> u32 {
        self.subpass_id
    }

    /// Returns true if the subpass has any color or depth/stencil attachment.
    #[inline]
    pub fn has_color_or_depth_stencil_attachment(&self) -> bool {
        unimplemented!()
    }

    /// Returns the number of samples in the color and/or depth/stencil attachments. Returns `None`
    /// if there is no such attachment in this subpass.
    #[inline]
    pub fn num_samples(&self) -> Option<u32> {
        unimplemented!()
    }
}

/// Contains the list of images attached to a renderpass.
///
/// This is a structure that you must pass when you start recording draw commands in a
/// command buffer.
///
/// A framebuffer can be used alongside with any other renderpass object as long as it is
/// compatible with the renderpass that his framebuffer was created with. You can determine whether
/// two renderpass objects are compatible by calling `is_compatible_with`.
pub struct Framebuffer<L> {
    device: Arc<Device>,
    renderpass: Arc<RenderPass<L>>,
    framebuffer: vk::Framebuffer,
    dimensions: (u32, u32, u32),
    resources: Vec<Arc<AbstractImageView>>,
}

impl<L> Framebuffer<L> {
    /// Builds a new framebuffer.
    ///
    /// The `attachments` parameter depends on which struct is used as a template parameter
    /// for the renderpass.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the attachments has a different sample count than what the renderpass
    ///   describes.
    /// - Additionally, some methods in the `RenderPassLayout` implementation may panic if you
    ///   pass invalid attachments.
    ///
    pub fn new<'a>(renderpass: &Arc<RenderPass<L>>, dimensions: (u32, u32, u32),
                   attachments: L::AttachmentsList)
                   -> Result<Arc<Framebuffer<L>>, FramebufferCreationError>
        where L: RenderPassLayout
    {
        let vk = renderpass.device.pointers();
        let device = renderpass.device.clone();

        let attachments = renderpass.layout.convert_attachments_list(attachments).collect::<Vec<_>>();

        // checking the dimensions against the limits
        {
            let limits = renderpass.device().physical_device().limits();
            let limits = [limits.max_framebuffer_width(), limits.max_framebuffer_height(),
                          limits.max_framebuffer_layers()];
            if dimensions.0 > limits[0] || dimensions.1 > limits[1] || dimensions.2 > limits[2] {
                return Err(FramebufferCreationError::DimensionsTooLarge);
            }
        }

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let ids = attachments.iter().map(|a| {
            //assert!(a.is_identity_swizzled());
            a.internal_object()
        }).collect::<Vec<_>>();

        let framebuffer = unsafe {
            let infos = vk::FramebufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                renderPass: renderpass.internal_object(),
                attachmentCount: ids.len() as u32,
                pAttachments: ids.as_ptr(),
                width: dimensions.0,
                height: dimensions.1,
                layers: dimensions.2,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateFramebuffer(device.internal_object(), &infos,
                                                   ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Framebuffer {
            device: device,
            renderpass: renderpass.clone(),
            framebuffer: framebuffer,
            dimensions: dimensions,
            resources: attachments,
        }))
    }

    /// Returns true if this framebuffer can be used with the specified renderpass.
    #[inline]
    pub fn is_compatible_with<R>(&self, renderpass: &Arc<RenderPass<R>>) -> bool
        where R: RenderPassLayout, L: RenderPassLayout
    {
        (&*self.renderpass as *const RenderPass<L> as usize ==
         &**renderpass as *const RenderPass<R> as usize) ||
            self.renderpass.is_compatible_with(renderpass)
    }

    /// Returns the width of the framebuffer in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.dimensions.0
    }

    /// Returns the height of the framebuffer in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.dimensions.1
    }

    /// Returns the number of layers (or depth) of the framebuffer.
    #[inline]
    pub fn layers(&self) -> u32 {
        self.dimensions.2
    }

    /// Returns the renderpass that was used to create this framebuffer.
    #[inline]
    pub fn renderpass(&self) -> &Arc<RenderPass<L>> {
        &self.renderpass
    }

    /// Returns all the resources attached to that framebuffer.
    #[inline]
    pub fn attachments(&self) -> &[Arc<AbstractImageView>] {
        &self.resources
    }
}

unsafe impl<L> VulkanObject for Framebuffer<L> {
    type Object = vk::Framebuffer;

    #[inline]
    fn internal_object(&self) -> vk::Framebuffer {
        self.framebuffer
    }
}

impl<L> Drop for Framebuffer<L> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyFramebuffer(self.device.internal_object(), self.framebuffer, ptr::null());
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FramebufferCreationError {
    OomError(OomError),
    DimensionsTooLarge,
}

impl From<OomError> for FramebufferCreationError {
    #[inline]
    fn from(err: OomError) -> FramebufferCreationError {
        FramebufferCreationError::OomError(err)
    }
}

impl error::Error for FramebufferCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            FramebufferCreationError::OomError(_) => "no memory available",
            FramebufferCreationError::DimensionsTooLarge => "the dimensions of the framebuffer \
                                                             are too large",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            FramebufferCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for FramebufferCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for FramebufferCreationError {
    #[inline]
    fn from(err: Error) -> FramebufferCreationError {
        FramebufferCreationError::from(OomError::from(err))
    }
}

#[cfg(test)]
mod tests {
    use framebuffer::Framebuffer;
    use framebuffer::RenderPass;
    use framebuffer::FramebufferCreationError;

    #[test]
    #[ignore]       // TODO: crashes on AMD+Windows
    fn empty_renderpass_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = RenderPass::empty_single_pass(&device).unwrap();
    }

    #[test]
    #[ignore]       // TODO: crashes on AMD+Windows
    fn framebuffer_too_large() {
        let (device, _) = gfx_dev_and_queue!();
        let renderpass = RenderPass::empty_single_pass(&device).unwrap();

        match Framebuffer::new(&renderpass, (0xffffffff, 0xffffffff, 0xffffffff), ()) {
            Err(FramebufferCreationError::DimensionsTooLarge) => (),
            _ => panic!()
        }
    }
}
