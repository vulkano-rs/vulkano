// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::iter::Empty as EmptyIter;
use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;

use device::Device;
use format::ClearValue;
use framebuffer::framebuffer::FramebufferCreationError;
use framebuffer::sys::UnsafeRenderPass;
use framebuffer::traits::RenderPass;
use framebuffer::traits::RenderPassDesc;
use framebuffer::traits::RenderPassAttachmentsList;
use framebuffer::traits::RenderPassClearValues;
use framebuffer::traits::LayoutAttachmentDescription;
use framebuffer::traits::LayoutPassDescription;
use framebuffer::traits::LayoutPassDependencyDescription;
use image::Layout as ImageLayout;
use image::traits::Image;
use image::traits::ImageView;

use OomError;

/// Implementation of `RenderPass` with no attachment at all and a single pass.
///
/// When you use a `EmptySinglePassRenderPass`, the list of attachments and clear values must
/// be `()`.
pub struct EmptySinglePassRenderPass {
    render_pass: UnsafeRenderPass,
}

impl EmptySinglePassRenderPass {
    /// See the docs of new().
    pub fn raw(device: &Arc<Device>) -> Result<EmptySinglePassRenderPass, OomError> {
        let rp = try!(unsafe {
            let pass = LayoutPassDescription {
                color_attachments: vec![],
                depth_stencil: None,
                input_attachments: vec![],
                resolve_attachments: vec![],
                preserve_attachments: vec![],
            };

            UnsafeRenderPass::raw(device, iter::empty(), Some(pass).into_iter(), iter::empty())
        });

        Ok(EmptySinglePassRenderPass {
            render_pass: rp
        })
    }
    
    /// Builds the render pass.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn new(device: &Arc<Device>) -> Arc<EmptySinglePassRenderPass> {
        Arc::new(EmptySinglePassRenderPass::raw(device).unwrap())
    }
}

unsafe impl RenderPass for EmptySinglePassRenderPass {
    #[inline]
    fn render_pass(&self) -> &UnsafeRenderPass {
        &self.render_pass
    }
}

unsafe impl RenderPassDesc for EmptySinglePassRenderPass {
    type AttachmentsIter = EmptyIter<LayoutAttachmentDescription>;
    type PassesIter = OptionIntoIter<LayoutPassDescription>;
    type DependenciesIter = EmptyIter<LayoutPassDependencyDescription>;

    #[inline]
    fn attachments(&self) -> Self::AttachmentsIter {
        iter::empty()
    }

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

    #[inline]
    fn dependencies(&self) -> Self::DependenciesIter {
        iter::empty()
    }

    #[inline]
    fn num_subpasses(&self) -> u32 {
        1
    }

    #[inline]
    fn num_color_attachments(&self, subpass: u32) -> Option<u32> {
        if subpass == 0 {
            Some(0)
        } else {
            None
        }
    }

    #[inline]
    fn num_samples(&self, _: u32) -> Option<u32> {
        None
    }

    #[inline]
    fn has_depth_stencil_attachment(&self, subpass: u32) -> Option<(bool, bool)> {
        if subpass == 0 {
            Some((false, false))
        } else {
            None
        }
    }

    #[inline]
    fn has_depth(&self, subpass: u32) -> Option<bool> {
        if subpass == 0 {
            Some(false)
        } else {
            None
        }
    }

    #[inline]
    fn has_writable_depth(&self, subpass: u32) -> Option<bool> {
        if subpass == 0 {
            Some(false)
        } else {
            None
        }
    }

    #[inline]
    fn has_stencil(&self, subpass: u32) -> Option<bool> {
        if subpass == 0 {
            Some(false)
        } else {
            None
        }
    }

    #[inline]
    fn has_writable_stencil(&self, subpass: u32) -> Option<bool> {
        if subpass == 0 {
            Some(false)
        } else {
            None
        }
    }
}

unsafe impl RenderPassAttachmentsList<()> for EmptySinglePassRenderPass {
    type AttachmentsIter = EmptyIter<(Arc<ImageView>, Arc<Image>, ImageLayout, ImageLayout)>;

    #[inline]
    fn convert_attachments_list(&self, _: ())
                                -> Result<Self::AttachmentsIter, FramebufferCreationError>
    {
        Ok(iter::empty())
    }
}

unsafe impl RenderPassClearValues<()> for EmptySinglePassRenderPass {
    type ClearValuesIter = EmptyIter<ClearValue>;

    #[inline]
    fn convert_clear_values(&self, _: ()) -> Self::ClearValuesIter {
        iter::empty()
    }
}

#[cfg(test)]
mod tests {
    use framebuffer::EmptySinglePassRenderPass;

    #[test]
    #[ignore]       // TODO: crashes on AMD+Windows
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = EmptySinglePassRenderPass::new(&device);
    }
}
