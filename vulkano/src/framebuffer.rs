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
//! Creating a `RenderPass` in the vulkano library is best done with the `renderpass!` macro.
//!
//! This macro creates an inaccessible struct which implements the `RenderPassLayout` trait. This
//! trait tells vulkano what the characteristics of the renderpass are, and is also used to
//! determine the types of the various parameters later on.
//!
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use formats::Format;
use formats::FormatMarker;
use image::ImageResource;
use image::Layout as ImageLayout;

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
    type AttachmentsIter: ExactSizeIterator<Item = AttachmentDescription>;

    /// Returns the descriptions of the attachments.
    fn attachments(&self) -> Self::AttachmentsIter;

    /// Iterator that produces passes.
    type PassesIter: ExactSizeIterator<Item = PassDescription>;

    /// Returns the descriptions of the passes.
    fn passes(&self) -> Self::PassesIter;

    /// Iterator that produces pass dependencies.
    type PassDependenciesIter: ExactSizeIterator<Item = PassDependencyDescription>;

    /// Returns the descriptions of the dependencies between passes.
    fn pass_dependencies(&self) -> Self::PassDependenciesIter;
}

pub unsafe trait RenderPassLayoutExt<'a, M: 'a>: RenderPassLayout {
    type AttachmentsList;

    fn ids(&self, &Self::AttachmentsList) -> (Vec<Arc<ImageResource>>, Vec<u64>);
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

/// Describes a uniform value that will be used to fill an attachment at the start of the
/// renderpass.
// TODO: should have the same layout as `vk::ClearValue` for performances
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ClearValue {
    /// Entry for attachments that aren't cleared.
    None,
    /// Value for floating-point attachments, including `Unorm`, `Snorm`, `Sfloat`.
    Float([f32; 4]),
    /// Value for integer attachments, including `Int`.
    Int([i32; 4]),
    /// Value for unsigned integer attachments, including `Uint`.
    Uint([u32; 4]),
    /// Value for depth attachments.
    Depth(f32),
    /// Value for stencil attachments.
    Stencil(u32),
    /// Value for depth and stencil attachments.
    DepthStencil((f32, u32)),
}

pub struct AttachmentDescription {
    pub format: Format,
    pub samples: u32,
    pub load: LoadOp,
    pub store: StoreOp,

    pub initial_layout: ImageLayout,
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

pub struct PassDescription {
    /// Indices of attachments to use as color attachments.
    pub color_attachments: Vec<(usize, ImageLayout)>,      // TODO: Vec is slow
    pub depth_stencil: Option<(usize, ImageLayout)>,
    /// Indices of attachments to use as input attachments.
    pub input_attachments: Vec<(usize, ImageLayout)>,      // TODO: Vec is slow
    /// If not empty, each color attachment will be resolved into each corresponding entry of
    /// this list.
    ///
    /// If this value is not empty, it **must** be the same length as `color_attachments`.
    pub resolve_attachments: Vec<(usize, ImageLayout)>,      // TODO: Vec is slow
    /// Indices of attachments that will be preserved during this pass.
    pub preserve_attachments: Vec<usize>,      // TODO: Vec is slow
}

// FIXME: finish
pub struct PassDependencyDescription {
    pub source_subpass: usize,
    pub destination_subpass: usize,
    /*VkPipelineStageFlags    srcStageMask;
    VkPipelineStageFlags    dstStageMask;
    VkAccessFlags           srcAccessMask;
    VkAccessFlags           dstAccessMask;*/
    pub by_region: bool,
}

/// Builds a `RenderPass` object.
#[macro_export]
macro_rules! renderpass {
    (
        device: $device:expr,
        attachments: { $($atch_name:ident [$($attrs:ident),*]),+ }
    ) => (
        {
            use std::sync::Arc;

            struct Layout;
            unsafe impl $crate::framebuffer::RenderPassLayout for Layout {
                type ClearValues = [f32; 4];        // FIXME:
                type ClearValuesIter = std::option::IntoIter<$crate::framebuffer::ClearValue>;
                type AttachmentsIter = std::vec::IntoIter<$crate::framebuffer::AttachmentDescription>;

                #[inline]
                fn convert_clear_values(&self, val: Self::ClearValues) -> Self::ClearValuesIter {
                    Some($crate::framebuffer::ClearValue::Float(val)).into_iter()
                }

                #[inline]
                fn attachments(&self) -> Self::AttachmentsIter {
                    vec![
                        $(
                            $crate::framebuffer::AttachmentDescription {
                                format: $crate::formats::Format::B8G8R8A8Srgb,       // FIXME:
                                samples: 1,                         // FIXME:
                                load: renderpass!(__load_op__ $($attrs),*),
                                store: $crate::framebuffer::StoreOp::Store,     // FIXME:
                                initial_layout: $crate::image::Layout::PresentSrc,       // FIXME:
                                final_layout: $crate::image::Layout::PresentSrc,       // FIXME:
                            }
                        )*
                    ].into_iter()
                }
            }

            unsafe impl<'a, M> $crate::framebuffer::RenderPassLayoutExt<'a, M> for Layout
                where M: $crate::memory::MemorySourceChunk + 'static
            {
                type AttachmentsList = &'a Arc<$crate::image::ImageView<$crate::image::Type2d, $crate::formats::B8G8R8A8Srgb, M>>;      // FIXME:

                fn ids(&self, l: &Self::AttachmentsList) -> (Vec<Arc<$crate::image::ImageResource>>, Vec<u64>) {
                    let a = vec![(**l).clone() as Arc<$crate::image::ImageResource>];
                    let b = vec![l.id()];
                    (a, b)
                }
            }

            $crate::framebuffer::RenderPass::<Layout>::new($device, Layout)
        }
    );

    // Gets the load operation to use for an attachment from the list of attributes.
    (__load_op__ LoadDontCare $($attrs:ident),*) => (
        $crate::framebuffer::LoadOp::DontCare
    );
    (__load_op__ Clear $($attrs:ident),*) => (
        $crate::framebuffer::LoadOp::Clear
    );
    (__load_op__ $first:ident $($attrs:ident),*) => (
        renderpass!(__load_op__ $($attrs),*)
    );
    (__load_op__) => (
        $crate::framebuffer::LoadOp::Load
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
                stencilLoadOp: 0,       // FIXME:
                stencilStoreOp: 0,      // FIXME:,
                initialLayout: attachment.initial_layout as u32,
                finalLayout: attachment.final_layout as u32,
            }
        }).collect::<Vec<_>>();

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

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let preserve_attachments_references = layout.passes().flat_map(|pass| {
            pass.preserve_attachments.into_iter().map(|offset| offset as u32)
        }).collect::<Vec<_>>();

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let mut ref_index = 0usize;
        let mut preserve_ref_index = 0usize;
        let passes = layout.passes().map(|pass| {
            unsafe {
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
                    colorAttachmentCount: pass.input_attachments.len() as u32,
                    pColorAttachments: color_attachments,
                    pResolveAttachments: if pass.resolve_attachments.len() == 0 { ptr::null() } else { resolve_attachments },
                    pDepthStencilAttachment: ptr::null(),       // FIXME:
                    preserveAttachmentCount: pass.preserve_attachments.len() as u32,
                    pPreserveAttachments: preserve_attachments,
                }
            }
        }).collect::<Vec<_>>();
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
}

pub struct Framebuffer<L> {
    device: Arc<Device>,
    renderpass: Arc<RenderPass<L>>,
    framebuffer: vk::Framebuffer,
    dimensions: (u32, u32, u32),
    resources: Vec<Arc<ImageResource>>,
}

impl<L> Framebuffer<L> {
    pub fn new<'a, M>(renderpass: &Arc<RenderPass<L>>, dimensions: (u32, u32, u32),
                      attachments: L::AttachmentsList) -> Result<Arc<Framebuffer<L>>, OomError>
        where L: RenderPassLayoutExt<'a, M>
    {
        let vk = renderpass.device.pointers();
        let device = renderpass.device.clone();

        let (resources, ids) = renderpass.layout.ids(&attachments);

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
            resources: resources,
        }))
    }

    /// Returns true if this framebuffer can be used with the specified renderpass.
    #[inline]
    pub fn is_compatible_with<R>(&self, renderpass: &Arc<RenderPass<R>>) -> bool {
        true        // FIXME:
        //(&*self.renderpass as *const RenderPass<L> as usize == &**renderpass as *const _ as usize)
            //|| self.renderpass.is_compatible_with(renderpass)
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
    pub fn attachments(&self) -> &[Arc<ImageResource>] {
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
