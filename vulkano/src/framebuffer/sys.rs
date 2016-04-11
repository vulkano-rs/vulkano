// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use device::Device;
use format::FormatDesc;
use framebuffer::RenderPass;
use framebuffer::LayoutAttachmentDescription;
use framebuffer::LayoutPassDescription;
use framebuffer::LayoutPassDependencyDescription;
use image::traits::Image;
use image::traits::ImageView;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Defines the layout of multiple subpasses.
pub struct UnsafeRenderPass {
    renderpass: vk::RenderPass,
    device: Arc<Device>,
}

impl UnsafeRenderPass {
    /// Builds a new renderpass.
    ///
    /// This function calls the methods of the `Layout` implementation and builds the
    /// corresponding Vulkan object.
    ///
    /// # Safety
    ///
    /// This function doesn't check whether all the restrictions in the attachments, passes and
    /// passes dependencies were enforced.
    ///
    /// See the documentation of the structs of this module for more info about these restrictions.
    ///
    /// # Panic
    ///
    /// Can panick if it detects some violations in the restrictions. Only unexpensive checks are
    /// performed. `debug_assert!` is used, so some restrictions are only checked in debug
    /// mode.
    ///
    pub unsafe fn new<Ia, Ip, Id>(device: &Arc<Device>, attachments: Ia, passes: Ip,
                                  pass_dependencies: Id)
               -> Result<UnsafeRenderPass, OomError>
        where Ia: ExactSizeIterator<Item = LayoutAttachmentDescription> + Clone,        // with specialization we can handle the "Clone" restriction internally
              Ip: ExactSizeIterator<Item = LayoutPassDescription> + Clone,      // with specialization we can handle the "Clone" restriction internally
              Id: ExactSizeIterator<Item = LayoutPassDependencyDescription>
    {
        let vk = device.pointers();

        // TODO: check the validity of the renderpass layout with debug_assert!

        let attachments = attachments.clone().map(|attachment| {
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
        }).collect::<SmallVec<[_; 16]>>();

        // We need to pass pointers to vkAttachmentReference structs when creating the renderpass.
        // Therefore we need to allocate them in advance.
        //
        // This block allocates, for each pass, in order, all color attachment references, then all
        // input attachment references, then all resolve attachment references, then the depth
        // stencil attachment reference.
        let attachment_references = passes.clone().flat_map(|pass| {
            debug_assert!(pass.resolve_attachments.is_empty() ||
                          pass.resolve_attachments.len() == pass.color_attachments.len());
            let resolve = pass.resolve_attachments.into_iter().map(|(offset, img_la)| {
                debug_assert!(offset < attachments.len());
                vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, }
            });

            let color = pass.color_attachments.into_iter().map(|(offset, img_la)| {
                debug_assert!(offset < attachments.len());
                vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, }
            });

            let input = pass.input_attachments.into_iter().map(|(offset, img_la)| {
                debug_assert!(offset < attachments.len());
                vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, }
            });

            let depthstencil = if let Some((offset, img_la)) = pass.depth_stencil {
                Some(vk::AttachmentReference { attachment: offset as u32, layout: img_la as u32, })
            } else {
                None
            }.into_iter();

            color.chain(input).chain(resolve).chain(depthstencil)
        }).collect::<SmallVec<[_; 16]>>();

        // Same as `attachment_references` but only for the preserve attachments.
        // This is separate because attachment references are u32s and not `vkAttachmentReference`
        // structs.
        let preserve_attachments_references = passes.clone().flat_map(|pass| {
            pass.preserve_attachments.into_iter().map(|offset| offset as u32)
        }).collect::<SmallVec<[_; 16]>>();

        // Now iterating over passes.
        // `ref_index` and `preserve_ref_index` are increased during the loop and point to the
        // next element to use in respectively `attachment_references` and
        // `preserve_attachments_references`.
        let mut ref_index = 0usize;
        let mut preserve_ref_index = 0usize;
        let passes = passes.clone().map(|pass| {
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
        }).collect::<SmallVec<[_; 16]>>();

        assert!(!passes.is_empty());
        // If these assertions fails, there's a serious bug in the code above ^.
        debug_assert!(ref_index == attachment_references.len());
        debug_assert!(preserve_ref_index == preserve_attachments_references.len());

        let dependencies = pass_dependencies.map(|dependency| {
            debug_assert!(dependency.source_subpass < passes.len());
            debug_assert!(dependency.destination_subpass < passes.len());

            vk::SubpassDependency {
                srcSubpass: dependency.source_subpass as u32,
                dstSubpass: dependency.destination_subpass as u32,
                srcStageMask: vk::PIPELINE_STAGE_ALL_GRAPHICS_BIT,      // FIXME:
                dstStageMask: vk::PIPELINE_STAGE_ALL_GRAPHICS_BIT,      // FIXME:
                srcAccessMask: 0x0001FFFF,       // FIXME:
                dstAccessMask: 0x0001FFFF,       // FIXME:
                dependencyFlags: if dependency.by_region { vk::DEPENDENCY_BY_REGION_BIT } else { 0 },
            }
        }).collect::<SmallVec<[_; 16]>>();

        let renderpass = {
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

        Ok(UnsafeRenderPass {
            device: device.clone(),
            renderpass: renderpass,
        })
    }

    /// Returns the device that was used to create this render pass.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for UnsafeRenderPass {
    type Object = vk::RenderPass;

    #[inline]
    fn internal_object(&self) -> vk::RenderPass {
        self.renderpass
    }
}

unsafe impl RenderPass for UnsafeRenderPass {
    #[inline]
    fn render_pass(&self) -> &UnsafeRenderPass {
        self
    }
}

impl Drop for UnsafeRenderPass {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyRenderPass(self.device.internal_object(), self.renderpass, ptr::null());
        }
    }
}
