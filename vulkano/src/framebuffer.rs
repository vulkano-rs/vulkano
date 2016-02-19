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

    /// Decodes a `ClearValues` into a list of clear values where each element corresponds
    /// to an attachment. The size of the returned array must be the same as the number of
    /// attachments.
    ///
    /// The format of the clear value **must** match the format of the attachment. Only attachments
    /// that are loaded with `LoadOp::Clear` must have an entry in the array.
    fn convert_clear_values(&self, Self::ClearValues) -> Vec<ClearValue>;

    /// Returns the descriptions of the attachments.
    fn attachments(&self) -> Vec<AttachmentDescription>;     // TODO: static array?
}

pub unsafe trait RenderPassLayoutExt<'a, M: 'a>: RenderPassLayout {
    type AttachmentsList;

    fn ids(&self, &Self::AttachmentsList) -> (Vec<Arc<ImageResource>>, Vec<u64>);
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

                #[inline]
                fn convert_clear_values(&self, val: Self::ClearValues) -> Vec<$crate::framebuffer::ClearValue> {
                    vec![$crate::framebuffer::ClearValue::Float(val)]
                }

                #[inline]
                fn attachments(&self) -> Vec<$crate::framebuffer::AttachmentDescription> {
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
                    ]
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

        let attachments = layout.attachments().iter().map(|attachment| {
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

        // FIXME: totally hacky
        let color_attachment_references = layout.attachments().iter().map(|attachment| {
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            }
        }).collect::<Vec<_>>();

        let passes = (0 .. 1).map(|_| {
            vk::SubpassDescription {
                flags: 0,   // reserved
                pipelineBindPoint: vk::PIPELINE_BIND_POINT_GRAPHICS,
                inputAttachmentCount: 0,        // FIXME:
                pInputAttachments: ptr::null(),     // FIXME:
                colorAttachmentCount: color_attachment_references.len() as u32,     // FIXME:
                pColorAttachments: color_attachment_references.as_ptr(),        // FIXME:
                pResolveAttachments: ptr::null(),       // FIXME:
                pDepthStencilAttachment: ptr::null(),       // FIXME:
                preserveAttachmentCount: 0,     // FIXME:
                pPreserveAttachments: ptr::null(),      // FIXME:
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
                dependencyCount: 0,             // FIXME:
                pDependencies: ptr::null(),     // FIXME:
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
    pub fn is_compatible_with<R2>(&self, other: &RenderPass<R2>) -> bool {
        true        // FIXME: 
    }

    /// Returns the layout used to create this renderpass.
    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

impl<L> VulkanObject for RenderPass<L> {
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

impl<L> VulkanObject for Framebuffer<L> {
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
