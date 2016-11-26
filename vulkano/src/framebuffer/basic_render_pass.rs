// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use format::Format;
use framebuffer::LoadOp;
use framebuffer::StoreOp;
use framebuffer::traits::RenderPassDesc;
use framebuffer::traits::LayoutAttachmentDescription;
use framebuffer::traits::LayoutPassDescription;
use framebuffer::traits::LayoutPassDependencyDescription;
use image::Layout;

/// Description of a render pass with one subpass.
#[derive(Debug, Clone)]
pub struct BasicRenderPassDesc {
    pub color_attachments: Vec<BasicRenderPassDescAttachment>,
    pub depth_stencil: Option<BasicRenderPassDescAttachment>,
    pub samples: u32,
}

#[derive(Debug, Clone)]
pub struct BasicRenderPassDescAttachment {
    pub format: Format,
}

// FIXME: formats must be checked

unsafe impl RenderPassDesc for BasicRenderPassDesc {
    #[inline]
    fn num_attachments(&self) -> usize {
        self.color_attachments.len() + if self.depth_stencil.is_some() { 1 } else { 0 }
    }

    #[inline]
    fn attachment(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        if num < self.color_attachments.len() {
            let atch = &self.color_attachments[num];

            Some(LayoutAttachmentDescription {
                format: atch.format,
                samples: self.samples,

                load: LoadOp::Load,
                store: StoreOp::Store,

                initial_layout: Layout::ColorAttachmentOptimal,
                final_layout: Layout::ColorAttachmentOptimal,
            })

        } else if num == self.color_attachments.len() && self.depth_stencil.is_some() {
            let atch = self.depth_stencil.as_ref().unwrap();

            Some(LayoutAttachmentDescription {
                format: atch.format,
                samples: self.samples,

                load: LoadOp::Load,
                store: StoreOp::Store,

                initial_layout: Layout::DepthStencilAttachmentOptimal,
                final_layout: Layout::DepthStencilAttachmentOptimal,
            })

        } else {
            None
        }
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        1
    }

    #[inline]
    fn subpass(&self, num: usize) -> Option<LayoutPassDescription> {
        if num == 0 {
            Some(LayoutPassDescription {
                color_attachments: (0 .. self.color_attachments.len()).map(|n| {
                    (n, Layout::ColorAttachmentOptimal)
                }).collect(),
                depth_stencil: self.depth_stencil.as_ref().map(|_| {
                    (self.color_attachments.len(), Layout::DepthStencilAttachmentOptimal)
                }),
                input_attachments: vec![],
                resolve_attachments: vec![],
                preserve_attachments: vec![],
            })

        } else {
            None
        }
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        0
    }

    #[inline]
    fn dependency(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        None
    }
}
