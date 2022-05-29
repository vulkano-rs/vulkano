// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::GraphicsPipelineCreationError;
use crate::{format::Format, render_pass::Subpass};

/// Selects the type of render pass that a graphics pipeline is created for.
#[derive(Clone, Debug)]
pub enum PipelineRenderPassType {
    BeginRenderPass(Subpass),
    BeginRendering(PipelineRenderingCreateInfo),
}

impl From<Subpass> for PipelineRenderPassType {
    #[inline]
    fn from(val: Subpass) -> Self {
        Self::BeginRenderPass(val)
    }
}

impl From<PipelineRenderingCreateInfo> for PipelineRenderPassType {
    #[inline]
    fn from(val: PipelineRenderingCreateInfo) -> Self {
        Self::BeginRendering(val)
    }
}

/// The dynamic rendering parameters to create a graphics pipeline.
#[derive(Clone, Debug)]
pub struct PipelineRenderingCreateInfo {
    /// If not `0`, indicates that multiview rendering will be enabled, and specifies the view
    /// indices that are rendered to. The value is a bitmask, so that that for example `0b11` will
    /// draw to the first two views and `0b101` will draw to the first and third view.
    ///
    /// If set to a nonzero value, the [`multiview`](crate::device::Features::multiview) feature
    /// must be enabled on the device.
    ///
    /// The default value is `0`.
    pub view_mask: u32,

    /// The formats of the color attachments that will be used during rendering.
    ///
    /// If an element is `None`, it indicates that the attachment will not be used.
    ///
    /// The default value is empty.
    pub color_attachment_formats: Vec<Option<Format>>,

    /// The format of the depth attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no depth attachment will be used.
    ///
    /// The default value is `None`.
    pub depth_attachment_format: Option<Format>,

    /// The format of the stencil attachment that will be used during rendering.
    ///
    /// If set to `None`, it indicates that no stencil attachment will be used.
    ///
    /// The default value is `None`.
    pub stencil_attachment_format: Option<Format>,

    pub _ne: crate::NonExhaustive,
}

impl Default for PipelineRenderingCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            view_mask: 0,
            color_attachment_formats: Vec::new(),
            depth_attachment_format: None,
            stencil_attachment_format: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl PipelineRenderingCreateInfo {
    pub(super) fn validate(&self) -> Result<(), GraphicsPipelineCreationError> {
        Ok(())
    }
}
