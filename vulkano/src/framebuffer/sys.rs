// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use device::Device;
use device::DeviceOwned;
use format::ClearValue;
use framebuffer::EmptySinglePassRenderPassDesc;
use framebuffer::AttachmentDescription;
use framebuffer::PassDependencyDescription;
use framebuffer::PassDescription;
use framebuffer::LoadOp;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassDescClearValues;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use vk;

/// Defines the layout of multiple subpasses.
///
/// The `RenderPass` struct should always implement the `RenderPassAbstract` trait. Therefore
/// you can turn any `Arc<RenderPass<D>>` into a `Arc<RenderPassAbstract + Send + Sync>` if you need to.
pub struct RenderPass<D> {
    // The internal Vulkan object.
    render_pass: vk::RenderPass,

    // Device this render pass was created from.
    device: Arc<Device>,

    // Description of the render pass.
    desc: D,

    // Cache of the granularity of the render pass.
    granularity: Mutex<Option<[u32; 2]>>,
}

impl<D> RenderPass<D>
    where D: RenderPassDesc
{
    /// Builds a new render pass.
    ///
    /// # Panic
    ///
    /// - Can panic if it detects some violations in the restrictions. Only inexpensive checks are
    /// performed. `debug_assert!` is used, so some restrictions are only checked in debug
    /// mode.
    ///
    pub fn new(device: Arc<Device>, description: D)
               -> Result<RenderPass<D>, RenderPassCreationError> {
        let vk = device.pointers();

        // If the first use of an attachment in this render pass is as an input attachment, and
        // the attachment is not also used as a color or depth/stencil attachment in the same
        // subpass, then loadOp must not be VK_ATTACHMENT_LOAD_OP_CLEAR
        debug_assert!(description.attachment_descs().enumerate().all(|(atch_num,
          attachment)| {
            if attachment.load != LoadOp::Clear {
                return true;
            }

            for p in description.subpass_descs() {
                if p.color_attachments
                    .iter()
                    .find(|&&(a, _)| a == atch_num)
                    .is_some()
                {
                    return true;
                }
                if let Some((a, _)) = p.depth_stencil {
                    if a == atch_num {
                        return true;
                    }
                }
                if p.input_attachments
                    .iter()
                    .find(|&&(a, _)| a == atch_num)
                    .is_some()
                {
                    return false;
                }
            }

            true
        }));

        let attachments = description
            .attachment_descs()
            .map(|attachment| {
                debug_assert!(attachment.samples.is_power_of_two());

                vk::AttachmentDescription {
                    flags: 0, // FIXME: may alias flag
                    format: attachment.format as u32,
                    samples: attachment.samples,
                    loadOp: attachment.load as u32,
                    storeOp: attachment.store as u32,
                    stencilLoadOp: attachment.stencil_load as u32,
                    stencilStoreOp: attachment.stencil_store as u32,
                    initialLayout: attachment.initial_layout as u32,
                    finalLayout: attachment.final_layout as u32,
                }
            })
            .collect::<SmallVec<[_; 16]>>();

        // We need to pass pointers to vkAttachmentReference structs when creating the render pass.
        // Therefore we need to allocate them in advance.
        //
        // This block allocates, for each pass, in order, all color attachment references, then all
        // input attachment references, then all resolve attachment references, then the depth
        // stencil attachment reference.
        let attachment_references = description
            .subpass_descs()
            .flat_map(|pass| {
                // Performing some validation with debug asserts.
                debug_assert!(pass.resolve_attachments.is_empty() ||
                                  pass.resolve_attachments.len() == pass.color_attachments.len());
                debug_assert!(pass.resolve_attachments
                                  .iter()
                                  .all(|a| attachments[a.0].samples == 1));
                debug_assert!(pass.resolve_attachments.is_empty() ||
                                  pass.color_attachments
                                      .iter()
                                      .all(|a| attachments[a.0].samples > 1));
                debug_assert!(pass.resolve_attachments.is_empty() ||
                                  pass.resolve_attachments
                                      .iter()
                                      .zip(pass.color_attachments.iter())
                                      .all(|(r, c)| {
                                               attachments[r.0].format == attachments[c.0].format
                                           }));
                debug_assert!(pass.color_attachments
                                  .iter()
                                  .cloned()
                                  .chain(pass.depth_stencil.clone().into_iter())
                                  .chain(pass.input_attachments.iter().cloned())
                                  .chain(pass.resolve_attachments.iter().cloned())
                                  .all(|(a, _)| {
                                           pass.preserve_attachments
                                               .iter()
                                               .find(|&&b| a == b)
                                               .is_none()
                                       }));
                debug_assert!(
                    pass.color_attachments
                        .iter()
                        .cloned()
                        .chain(pass.depth_stencil.clone().into_iter())
                        .all(|(atch, layout)| if let Some(r) =
                            pass.input_attachments.iter().find(|r| r.0 == atch)
                        {
                            r.1 == layout
                        } else {
                            true
                        })
                );

                let resolve = pass.resolve_attachments
                    .into_iter()
                    .map(|(offset, img_la)| {
                             debug_assert!(offset < attachments.len());
                             vk::AttachmentReference {
                                 attachment: offset as u32,
                                 layout: img_la as u32,
                             }
                         });

                let color = pass.color_attachments.into_iter().map(|(offset, img_la)| {
                    debug_assert!(offset < attachments.len());
                    vk::AttachmentReference {
                        attachment: offset as u32,
                        layout: img_la as u32,
                    }
                });

                let input = pass.input_attachments.into_iter().map(|(offset, img_la)| {
                    debug_assert!(offset < attachments.len());
                    vk::AttachmentReference {
                        attachment: offset as u32,
                        layout: img_la as u32,
                    }
                });

                let depthstencil = if let Some((offset, img_la)) = pass.depth_stencil {
                    Some(vk::AttachmentReference {
                             attachment: offset as u32,
                             layout: img_la as u32,
                         })
                } else {
                    None
                }.into_iter();

                color.chain(input).chain(resolve).chain(depthstencil)
            })
            .collect::<SmallVec<[_; 16]>>();

        // Same as `attachment_references` but only for the preserve attachments.
        // This is separate because attachment references are u32s and not `vkAttachmentReference`
        // structs.
        let preserve_attachments_references = description
            .subpass_descs()
            .flat_map(|pass| {
                          pass.preserve_attachments
                              .into_iter()
                              .map(|offset| offset as u32)
                      })
            .collect::<SmallVec<[_; 16]>>();

        // Now iterating over passes.
        let passes = unsafe {
            // `ref_index` and `preserve_ref_index` are increased during the loop and point to the
            // next element to use in respectively `attachment_references` and
            // `preserve_attachments_references`.
            let mut ref_index = 0usize;
            let mut preserve_ref_index = 0usize;
            let mut out: SmallVec<[_; 16]> = SmallVec::new();

            for pass in description.subpass_descs() {
                if pass.color_attachments.len() as u32 >
                    device.physical_device().limits().max_color_attachments()
                {
                    return Err(RenderPassCreationError::ColorAttachmentsLimitExceeded);
                }

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

                let preserve_attachments = preserve_attachments_references
                    .as_ptr()
                    .offset(preserve_ref_index as isize);
                preserve_ref_index += pass.preserve_attachments.len();

                out.push(vk::SubpassDescription {
                             flags: 0, // reserved
                             pipelineBindPoint: vk::PIPELINE_BIND_POINT_GRAPHICS,
                             inputAttachmentCount: pass.input_attachments.len() as u32,
                             pInputAttachments: if pass.input_attachments.is_empty() {
                                 ptr::null()
                             } else {
                                 input_attachments
                             },
                             colorAttachmentCount: pass.color_attachments.len() as u32,
                             pColorAttachments: if pass.color_attachments.is_empty() {
                                 ptr::null()
                             } else {
                                 color_attachments
                             },
                             pResolveAttachments: if pass.resolve_attachments.is_empty() {
                                 ptr::null()
                             } else {
                                 resolve_attachments
                             },
                             pDepthStencilAttachment: depth_stencil,
                             preserveAttachmentCount: pass.preserve_attachments.len() as u32,
                             pPreserveAttachments: if pass.preserve_attachments.is_empty() {
                                 ptr::null()
                             } else {
                                 preserve_attachments
                             },
                         });
            }

            assert!(!out.is_empty());
            // If these assertions fails, there's a serious bug in the code above ^.
            debug_assert!(ref_index == attachment_references.len());
            debug_assert!(preserve_ref_index == preserve_attachments_references.len());

            out
        };

        let dependencies = description
            .dependency_descs()
            .map(|dependency| {
                debug_assert!(dependency.source_subpass as u32 == vk::SUBPASS_EXTERNAL || dependency.source_subpass < passes.len());
                debug_assert!(dependency.destination_subpass as u32 == vk::SUBPASS_EXTERNAL || dependency.destination_subpass < passes.len());

                vk::SubpassDependency {
                    srcSubpass: dependency.source_subpass as u32,
                    dstSubpass: dependency.destination_subpass as u32,
                    srcStageMask: dependency.source_stages.into_vulkan_bits(),
                    dstStageMask: dependency.destination_stages.into_vulkan_bits(),
                    srcAccessMask: dependency.source_access.into_vulkan_bits(),
                    dstAccessMask: dependency.destination_access.into_vulkan_bits(),
                    dependencyFlags: if dependency.by_region {
                        vk::DEPENDENCY_BY_REGION_BIT
                    } else {
                        0
                    },
                }
            })
            .collect::<SmallVec<[_; 16]>>();

        let render_pass = unsafe {
            let infos = vk::RenderPassCreateInfo {
                sType: vk::STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                attachmentCount: attachments.len() as u32,
                pAttachments: if attachments.is_empty() {
                    ptr::null()
                } else {
                    attachments.as_ptr()
                },
                subpassCount: passes.len() as u32,
                pSubpasses: if passes.is_empty() {
                    ptr::null()
                } else {
                    passes.as_ptr()
                },
                dependencyCount: dependencies.len() as u32,
                pDependencies: if dependencies.is_empty() {
                    ptr::null()
                } else {
                    dependencies.as_ptr()
                },
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateRenderPass(device.internal_object(),
                                             &infos,
                                             ptr::null(),
                                             &mut output))?;
            output
        };

        Ok(RenderPass {
               device: device.clone(),
               render_pass: render_pass,
               desc: description,
               granularity: Mutex::new(None),
           })
    }
}

impl RenderPass<EmptySinglePassRenderPassDesc> {
    /// Builds a render pass with one subpass and no attachment.
    ///
    /// This method is useful for quick tests.
    #[inline]
    pub fn empty_single_pass(
        device: Arc<Device>)
        -> Result<RenderPass<EmptySinglePassRenderPassDesc>, RenderPassCreationError> {
        RenderPass::new(device, EmptySinglePassRenderPassDesc)
    }
}

impl<D> RenderPass<D> {
    /// Returns the granularity of this render pass.
    ///
    /// If the render area of a render pass in a command buffer is a multiple of this granularity,
    /// then the performance will be optimal. Performances are always optimal for render areas
    /// that cover the whole framebuffer.
    pub fn granularity(&self) -> [u32; 2] {
        let mut granularity = self.granularity.lock().unwrap();

        if let Some(&granularity) = granularity.as_ref() {
            return granularity;
        }

        unsafe {
            let vk = self.device.pointers();
            let mut out = mem::uninitialized();
            vk.GetRenderAreaGranularity(self.device.internal_object(), self.render_pass, &mut out);

            debug_assert_ne!(out.width, 0);
            debug_assert_ne!(out.height, 0);
            let gran = [out.width, out.height];
            *granularity = Some(gran);
            gran
        }
    }

    /// Returns the description of the render pass.
    ///
    /// > **Note**: You must not somehow modify the description. This shouldn't be possible anyway
    /// > if `RenderPassDesc` was implemented correctly.
    #[inline]
    pub fn desc(&self) -> &D {
        &self.desc
    }
}

unsafe impl<D> RenderPassDesc for RenderPass<D>
    where D: RenderPassDesc
{
    #[inline]
    fn num_attachments(&self) -> usize {
        self.desc.num_attachments()
    }

    #[inline]
    fn attachment_desc(&self, num: usize) -> Option<AttachmentDescription> {
        self.desc.attachment_desc(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        self.desc.num_subpasses()
    }

    #[inline]
    fn subpass_desc(&self, num: usize) -> Option<PassDescription> {
        self.desc.subpass_desc(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        self.desc.num_dependencies()
    }

    #[inline]
    fn dependency_desc(&self, num: usize) -> Option<PassDependencyDescription> {
        self.desc.dependency_desc(num)
    }
}

unsafe impl<C, D> RenderPassDescClearValues<C> for RenderPass<D>
    where D: RenderPassDescClearValues<C>
{
    #[inline]
    fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
        self.desc.convert_clear_values(vals)
    }
}

unsafe impl<D> RenderPassAbstract for RenderPass<D>
    where D: RenderPassDesc
{
    #[inline]
    fn inner(&self) -> RenderPassSys {
        RenderPassSys(self.render_pass, PhantomData)
    }
}

unsafe impl<D> DeviceOwned for RenderPass<D> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<D> fmt::Debug for RenderPass<D>
    where D: fmt::Debug
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("RenderPass")
            .field("raw", &self.render_pass)
            .field("device", &self.device)
            .field("desc", &self.desc)
            .finish()
    }
}

impl<D> Drop for RenderPass<D> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyRenderPass(self.device.internal_object(), self.render_pass, ptr::null());
        }
    }
}

/// Opaque object that represents the render pass' internals.
#[derive(Debug, Copy, Clone)]
pub struct RenderPassSys<'a>(vk::RenderPass, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for RenderPassSys<'a> {
    type Object = vk::RenderPass;

    const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT;

    #[inline]
    fn internal_object(&self) -> vk::RenderPass {
        self.0
    }
}

/// Error that can happen when creating a compute pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderPassCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// The maximum number of color attachments has been exceeded.
    ColorAttachmentsLimitExceeded,
}

impl error::Error for RenderPassCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            RenderPassCreationError::OomError(_) => "not enough memory available",
            RenderPassCreationError::ColorAttachmentsLimitExceeded => {
                "the maximum number of color attachments has been exceeded"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            RenderPassCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for RenderPassCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for RenderPassCreationError {
    #[inline]
    fn from(err: OomError) -> RenderPassCreationError {
        RenderPassCreationError::OomError(err)
    }
}

impl From<Error> for RenderPassCreationError {
    #[inline]
    fn from(err: Error) -> RenderPassCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                RenderPassCreationError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                RenderPassCreationError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use format::Format;
    use framebuffer::RenderPass;
    use framebuffer::RenderPassCreationError;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = RenderPass::empty_single_pass(device).unwrap();
    }

    #[test]
    fn too_many_color_atch() {
        let (device, _) = gfx_dev_and_queue!();

        if device.physical_device().limits().max_color_attachments() >= 10 {
            return; // test ignored
        }

        let rp =
            single_pass_renderpass! {
            device.clone(),
            attachments: {
                a1: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a2: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a3: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a4: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a5: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a6: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a7: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a8: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a9: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, },
                a10: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, }
            },
            pass: {
                color: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10],
                depth_stencil: {}
            }
        };

        match rp {
            Err(RenderPassCreationError::ColorAttachmentsLimitExceeded) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn non_zero_granularity() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = single_pass_renderpass! {
            device.clone(),
            attachments: {
                a: { load: Clear, store: DontCare, format: Format::R8G8B8A8Unorm, samples: 1, }
            },
            pass: {
                color: [a],
                depth_stencil: {}
            }
        }.unwrap();

        let granularity = rp.granularity();
        assert_ne!(granularity[0], 0);
        assert_ne!(granularity[1], 0);
    }
}
