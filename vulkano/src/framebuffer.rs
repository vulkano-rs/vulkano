//! Targets on which your draw commands are executed.
//! 
//! # Render passes and framebuffers
//!
//! There are two concepts in Vulkan:
//! 
//! - A `UnsafeRenderPass` is a collection of rendering passes called subpasses. Each subpass contains
//!   the format and dimensions of the attachments that are part of the subpass. The render
//!   pass only defines the layout of the rendering process.
//! - A `Framebuffer` contains the list of actual images that are attached. It is created from a
//!   `UnsafeRenderPass` and has to match its characteristics.
//!
//! This split means that you can create graphics pipelines from a `UnsafeRenderPass` alone.
//! A `Framebuffer` is only needed when you add draw commands to a command buffer.
//!
//! # Render passes
//!
//! A render pass is composed of three things:
//!
//! - A list of attachments with their format.
//! - A list of subpasses, that defines for each subpass which attachment is used for which
//!   purpose.
//! - A list of dependencies between subpasses. Vulkan implementations are free to reorder the
//!   subpasses, which means that you need to declare dependencies if the output of a subpass
//!   needs to be read in a following subpass.
//!
//! Before you can create a `UnsafeRenderPass` object with the vulkano library, you have to create an
//! object that can describe these three lists through `Layout` trait. This trait is
//! unsafe because the information that its methods return is trusted blindly by vulkano.
//! 
//! There are two ways to do this:   TODO add more ways
//! 
//! - Creating an instance of an `EmptySinglePassLayout`, which describes a renderpass with no
//!   attachment and with one subpass.
//! - Using the `single_pass_renderpass!` macro. See the documentation of this macro.
//!
//! ## Example
//! 
//! With `EmptySinglePassLayout`:
//! 
//! ```no_run
//! use vulkano::framebuffer::UnsafeRenderPass;
//! use vulkano::framebuffer::EmptySinglePassLayout;
//! 
//! # let device: std::sync::Arc<vulkano::device::Device> = unsafe { ::std::mem::uninitialized() };
//! let renderpass = UnsafeRenderPass::new(&device, EmptySinglePassLayout).unwrap();
//! // the type of `renderpass` is `UnsafeRenderPass<EmptySinglePassLayout>`
//! ```
//!
//! # Framebuffers
//!
//! Creating a framebuffer is done by passing the render pass object, the dimensions of the
//! framebuffer, and the list of attachments to `Framebuffer::new()`.
//!
//! The slightly tricky part is that the list of attachments depends on the trait implementation
//! of `Layout`. For example if you use an `EmptySinglePassLayout`, you have to pass
//! `()` for the list of attachments.
//!
//! Some implementations of `Layout` can use strong typing for the attachments list, in
//! order to produce a compilation error if you pass the wrong kind of attachment. Other
//! implementations may have more relaxed rules and check the format of the attachments at runtime
//! instead.
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
pub unsafe trait Layout {
    /// Iterator that produces attachments.
    type AttachmentsDescIter: ExactSizeIterator<Item = LayoutAttachmentDescription>;

    /// Returns the descriptions of the attachments.
    fn attachments(&self) -> Self::AttachmentsDescIter;

    /// Iterator that produces passes.
    type PassesIter: ExactSizeIterator<Item = LayoutPassDescription>;

    /// Returns the descriptions of the passes.
    fn passes(&self) -> Self::PassesIter;

    /// Iterator that produces pass dependencies.
    type PassDependenciesIter: ExactSizeIterator<Item = LayoutPassDependencyDescription>;

    /// Returns the descriptions of the dependencies between passes.
    fn pass_dependencies(&self) -> Self::PassDependenciesIter;
}

/// Extension trait for `Layout`. Defines which types are allowed as an attachments list.
pub unsafe trait LayoutAttachmentsList<A>: Layout {
    /// A decoded `A`.
    type AttachmentsIter: ExactSizeIterator<Item = Arc<AbstractImageView>>;

    /// Decodes a `A` into a list of attachments.
    fn convert_attachments_list(&self, A) -> Self::AttachmentsIter;
}

/// Extension trait for `Layout`. Defines which types are allowed as a list of clear values.
pub unsafe trait LayoutClearValues<C>: Layout {
    /// Iterator that produces one clear value per attachment.
    type ClearValuesIter: Iterator<Item = ClearValue>;

    /// Decodes a `C` into a list of clear values where each element corresponds
    /// to an attachment. The size of the returned array must be the same as the number of
    /// attachments.
    ///
    /// The format of the clear value **must** match the format of the attachment. Only attachments
    /// that are loaded with `LoadOp::Clear` must have an entry in the array.
    fn convert_clear_values(&self, C) -> Self::ClearValuesIter;
}

/// Trait implemented on renderpass layouts to check whether they are compatible
/// with another layout.
///
/// The trait is automatically implemented for all type that implement `Layout`.
// TODO: once specialization lands, this trait can be specialized for pairs that are known to
//       always be compatible
// TODO: maybe this can be unimplemented on some pairs, to provide compile-time checks?
pub unsafe trait LayoutCompatible<Other>: Layout where Other: Layout {
    /// Returns `true` if this layout is compatible with the other layout, as defined in the
    /// `Render Pass Compatibility` section of the Vulkan specs.
    fn is_compatible_with(&self, other: &Other) -> bool;
}

unsafe impl<A, B> LayoutCompatible<B> for A
    where A: Layout, B: Layout
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

/// Describes one of the passes of a renderpass.
///
/// # Restrictions
///
/// All these restrictions are checked when the `UnsafeRenderPass` object is created.
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

/// Describes a dependency between two passes of a renderpass.
///
/// The implementation is allowed to change the order of the passes within a renderpass, unless
/// you specify that there exists a dependency between two passes (ie. the result of one will be
/// used as the input of another one).
// FIXME: finish
pub struct LayoutPassDependencyDescription {
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

/// Implementation of `Layout` with no attachment at all and a single pass.
#[derive(Debug, Copy, Clone)]
pub struct EmptySinglePassLayout;

unsafe impl Layout for EmptySinglePassLayout {
    type AttachmentsDescIter = EmptyIter<LayoutAttachmentDescription>;

    #[inline]
    fn attachments(&self) -> Self::AttachmentsDescIter {
        iter::empty()
    }

    type PassesIter = OptionIntoIter<LayoutPassDescription>;

    #[inline]
    fn passes(&self) -> Self::PassesIter {
        Some(LayoutPassDescription {
            color_attachments: vec![],
            depth_stencil: None,
            input_attachments: vec![],
            resolve_attachments: vec![],
            preserve_attachments: vec![],
        }).into_iter()
    }

    type PassDependenciesIter = EmptyIter<LayoutPassDependencyDescription>;

    #[inline]
    fn pass_dependencies(&self) -> Self::PassDependenciesIter {
        iter::empty()
    }
}

unsafe impl LayoutAttachmentsList<()> for EmptySinglePassLayout {
    type AttachmentsIter = EmptyIter<Arc<AbstractImageView>>;

    #[inline]
    fn convert_attachments_list(&self, _: ()) -> Self::AttachmentsIter {
        iter::empty()
    }
}

unsafe impl LayoutClearValues<()> for EmptySinglePassLayout {
    type ClearValuesIter = EmptyIter<ClearValue>;

    #[inline]
    fn convert_clear_values(&self, _: ()) -> Self::ClearValuesIter {
        iter::empty()
    }
}

/// Builds a `UnsafeRenderPass` object.
#[macro_export]
macro_rules! single_pass_renderpass {
    (
        attachments: { $($a:tt)* },
        pass: {
            color: [$($color_atch:ident),*],
            depth_stencil: {$($depth_atch:ident)*}
        }
    ) => {
        ordered_passes_renderpass!{
            attachments: { $($a)* },
            passes: [
                {
                    color: [$($color_atch),*],
                    depth_stencil: {$($depth_atch)*},
                    input: []
                }
            ]
        }
    }
}

/// Builds a `UnsafeRenderPass` object.
#[macro_export]
macro_rules! ordered_passes_renderpass {
    (
        attachments: {
            $(
                $atch_name:ident: {
                    load: $load:ident,
                    store: $store:ident,
                    format: $format:ident,
                }
            ),*
        },
        passes: [
            $(
                {
                    color: [$($color_atch:ident),*],
                    depth_stencil: {$($depth_atch:ident)*},
                    input: [$($input_atch:ident),*]
                }
            ),*
        ]
    ) => {
        use std;        // TODO: import everything instead
        use std::sync::Arc;
        use $crate::format::ClearValue;

        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct Layout;

        unsafe impl $crate::framebuffer::Layout for Layout {
            type AttachmentsDescIter = std::vec::IntoIter<$crate::framebuffer::LayoutAttachmentDescription>;
            type PassesIter = std::vec::IntoIter<$crate::framebuffer::LayoutPassDescription>;
            type PassDependenciesIter = std::vec::IntoIter<$crate::framebuffer::LayoutPassDependencyDescription>;

            #[inline]
            fn attachments(&self) -> Self::AttachmentsDescIter {
                vec![
                    $(
                        $crate::framebuffer::LayoutAttachmentDescription {
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

                vec![
                    $({
                        let mut depth = None;
                        $(
                            depth = Some(($depth_atch, $crate::image::Layout::DepthStencilAttachmentOptimal));
                        )*

                        $crate::framebuffer::LayoutPassDescription {
                            color_attachments: vec![
                                $(
                                    ($color_atch, $crate::image::Layout::ColorAttachmentOptimal)
                                ),*
                            ],
                            depth_stencil: depth,
                            input_attachments: vec![
                                $(
                                    ($input_atch, $crate::image::Layout::ShaderReadOnlyOptimal)
                                ),*
                            ],
                            resolve_attachments: vec![],
                            preserve_attachments: (0 .. attachment_num).filter(|&a| {
                                $(if a == $color_atch { return false; })*
                                $(if a == $depth_atch { return false; })*
                                $(if a == $input_atch { return false; })*
                                true
                            }).collect()
                        }
                    }),*
                ].into_iter()
            }

            #[inline]
            fn pass_dependencies(&self) -> Self::PassDependenciesIter {
                // TODO: could use a custom iterator
                (1 .. self.passes().len()).flat_map(|p2| {
                    (0 .. p2.clone()).map(move |p1| {
                        $crate::framebuffer::LayoutPassDependencyDescription {
                            source_subpass: p1,
                            destination_subpass: p2,
                            by_region: false,
                        }
                    })
                }).collect::<Vec<_>>().into_iter()
            }
        }

        pub type AList = ($(      // FIXME: should not use a trait
            Arc<$crate::image::AbstractTypedImageView<$crate::image::Type2d, $crate::format::$format>>,
        )*);

        unsafe impl $crate::framebuffer::LayoutAttachmentsList<AList> for Layout {
            // TODO: shouldn't build a Vec
            type AttachmentsIter = std::vec::IntoIter<std::sync::Arc<$crate::image::AbstractImageView>>;

            #[inline]
            fn convert_attachments_list(&self, l: AList) -> Self::AttachmentsIter {
                $crate::image::AbstractTypedImageViewsTuple::iter(l)
            }
        }

        ordered_passes_renderpass!{__impl_clear_values__ [0] [] [$($atch_name $format $load,)*] }

        unsafe impl $crate::framebuffer::LayoutClearValues<ClearValues> for Layout {
            type ClearValuesIter = ClearValuesIter;

            #[inline]
            fn convert_clear_values(&self, val: ClearValues) -> ClearValuesIter {
                ClearValuesIter(val, 0)
            }
        }
    };

    (__impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ident Clear, $($rest:tt)*]) => {
        ordered_passes_renderpass!{__impl_clear_values__ [$num+1] [$($s)* $atch_name [$num] $format,] [$($rest)*] }
    };

    (__impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ident $misc:ident, $($rest:tt)*]) => {
        ordered_passes_renderpass!{__impl_clear_values__ [$num+1] [$($s)*] [$($rest)*] }
    };

    (__impl_clear_values__ [$total:expr] [$($atch:ident [$num:expr] $format:ident,)*] []) => {
        pub struct ClearValues {
            $(
                pub $atch: <$crate::format::$format as $crate::format::FormatDesc>::ClearValue,
            )*
        }

        pub struct ClearValuesIter(ClearValues, usize);

        impl Iterator for ClearValuesIter {
            type Item = $crate::format::ClearValue;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                $(
                    if self.1 == $num {
                        self.1 += 1;
                        return Some(ClearValue::from((self.0).$atch));        // FIXME: should use Format::decode_clear_value instead
                    }
                )+

                if self.1 >= $total {
                    None
                } else {
                    Some(ClearValue::None)
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = $total - self.1;
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for ClearValuesIter {}
    };
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
pub struct UnsafeRenderPass<L> {
    device: Arc<Device>,
    renderpass: vk::RenderPass,
    num_passes: u32,
    layout: L,
}

impl<L> UnsafeRenderPass<L> where L: Layout {
    /// Builds a new renderpass.
    ///
    /// This function calls the methods of the `Layout` implementation and builds the
    /// corresponding Vulkan object.
    ///
    /// # Panic
    ///
    /// - Panicks if this functions detects that the `Layout` trait was not implemented
    ///   correctly and contains an error.
    ///   See the documentation of the various methods and structs related to `Layout`
    ///   for more details.
    ///
    pub fn new(device: &Arc<Device>, layout: L) -> Result<Arc<UnsafeRenderPass<L>>, OomError> {
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
                srcStageMask: vk::PIPELINE_STAGE_ALL_GRAPHICS_BIT,      // FIXME:
                dstStageMask: vk::PIPELINE_STAGE_ALL_GRAPHICS_BIT,      // FIXME:
                srcAccessMask: 0x0001FFFF,       // FIXME:
                dstAccessMask: 0x0001FFFF,       // FIXME:
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

        Ok(Arc::new(UnsafeRenderPass {
            device: device.clone(),
            renderpass: renderpass,
            num_passes: passes.len() as u32,
            layout: layout,
        }))
    }
}

impl UnsafeRenderPass<EmptySinglePassLayout> {
    /// Builds a `UnsafeRenderPass` with no attachment and a single pass.
    #[inline]
    pub fn empty_single_pass(device: &Arc<Device>)
                             -> Result<Arc<UnsafeRenderPass<EmptySinglePassLayout>>, OomError>
    {
        UnsafeRenderPass::new(device, EmptySinglePassLayout)
    }
}

impl<L> UnsafeRenderPass<L> where L: Layout {
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

    // TODO: add a `subpass` method that takes `Arc<Self>` as parameter

    /// Returns true if this renderpass is compatible with another one.
    ///
    /// This means that framebuffers created with this renderpass can also be used alongside with
    /// the other renderpass.
    #[inline]
    pub fn is_compatible_with<R2>(&self, other: &UnsafeRenderPass<R2>) -> bool
        where R2: Layout
    {
        self.layout.is_compatible_with(&other.layout)
    }

    /// Returns the layout used to create this renderpass.
    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

unsafe impl<L> VulkanObject for UnsafeRenderPass<L> {
    type Object = vk::RenderPass;

    #[inline]
    fn internal_object(&self) -> vk::RenderPass {
        self.renderpass
    }
}

impl<L> Drop for UnsafeRenderPass<L> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyRenderPass(self.device.internal_object(), self.renderpass, ptr::null());
        }
    }
}

/// Trait implemented on all render pass objects.
pub trait AbstractRenderPass {}
impl<L> AbstractRenderPass for UnsafeRenderPass<L> {}

/// Represents a subpass within a `UnsafeRenderPass`.
///
/// This struct doesn't correspond to anything in Vulkan. It is simply an equivalent to a
/// combination of a render pass and subpass ID.
pub struct Subpass<'a, L: 'a> {
    render_pass: &'a Arc<UnsafeRenderPass<L>>,
    subpass_id: u32,
}

impl<'a, L: 'a> Subpass<'a, L> {
    /// Returns a handle that represents a subpass of a render pass.
    #[inline]
    pub fn from(render_pass: &Arc<UnsafeRenderPass<L>>, id: u32) -> Option<Subpass<L>> {
        if id < render_pass.num_passes {
            Some(Subpass {
                render_pass: render_pass,
                subpass_id: id,
            })

        } else {
            None
        }
    }

    /// Returns the render pass of this subpass.
    #[inline]
    pub fn render_pass(&self) -> &'a Arc<UnsafeRenderPass<L>> {
        self.render_pass
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

impl<'a, L: 'a> Copy for Subpass<'a, L> {}
impl<'a, L: 'a> Clone for Subpass<'a, L> {
    #[inline]
    fn clone(&self) -> Subpass<'a, L> {
        Subpass { render_pass: self.render_pass, subpass_id: self.subpass_id }
    }
}

/// Contains the list of images attached to a renderpass.
///
/// This is a structure that you must pass when you start recording draw commands in a
/// command buffer.
///
/// A framebuffer can be used alongside with any other render pass object as long as it is
/// compatible with the render pass that his framebuffer was created with. You can determine
/// whether two renderpass objects are compatible by calling `is_compatible_with`.
pub struct Framebuffer<L> {
    device: Arc<Device>,
    renderpass: Arc<UnsafeRenderPass<L>>,
    framebuffer: vk::Framebuffer,
    dimensions: (u32, u32, u32),
    resources: Vec<Arc<AbstractImageView>>,
}

impl<L> Framebuffer<L> {
    /// Builds a new framebuffer.
    ///
    /// The `attachments` parameter depends on which struct is used as a template parameter
    /// for the render pass.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the attachments has a different sample count than what the render pass
    ///   describes.
    /// - Additionally, some methods in the `Layout` implementation may panic if you
    ///   pass invalid attachments.
    ///
    pub fn new<'a, A>(renderpass: &Arc<UnsafeRenderPass<L>>, dimensions: (u32, u32, u32),        // TODO: what about [u32; 3] instead?
                      attachments: A) -> Result<Arc<Framebuffer<L>>, FramebufferCreationError>
        where L: Layout + LayoutAttachmentsList<A>
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
    pub fn is_compatible_with<R>(&self, renderpass: &Arc<UnsafeRenderPass<R>>) -> bool
        where R: Layout, L: Layout
    {
        (&*self.renderpass as *const UnsafeRenderPass<L> as usize ==
         &**renderpass as *const UnsafeRenderPass<R> as usize) ||
            self.renderpass.is_compatible_with(renderpass)
    }

    /// Returns the width, height and layers of this framebuffer.
    #[inline]
    pub fn dimensions(&self) -> [u32; 3] {
        [self.dimensions.0, self.dimensions.1, self.dimensions.2]
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
    pub fn renderpass(&self) -> &Arc<UnsafeRenderPass<L>> {
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

/// Trait implemented on all framebuffers.
pub trait AbstractFramebuffer {}
impl<L> AbstractFramebuffer for Framebuffer<L> {}

/// Error that can happen when creating a framebuffer object.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FramebufferCreationError {
    /// Out of memory.
    OomError(OomError),
    /// The requested dimensions exceed the device's limits.
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
    use framebuffer::UnsafeRenderPass;
    use framebuffer::FramebufferCreationError;

    #[test]
    #[ignore]       // TODO: crashes on AMD+Windows
    fn empty_renderpass_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = UnsafeRenderPass::empty_single_pass(&device).unwrap();
    }

    #[test]
    #[ignore]       // TODO: crashes on AMD+Windows
    fn framebuffer_too_large() {
        let (device, _) = gfx_dev_and_queue!();
        let renderpass = UnsafeRenderPass::empty_single_pass(&device).unwrap();

        match Framebuffer::new(&renderpass, (0xffffffff, 0xffffffff, 0xffffffff), ()) {
            Err(FramebufferCreationError::DimensionsTooLarge) => (),
            _ => panic!()
        }
    }
}
