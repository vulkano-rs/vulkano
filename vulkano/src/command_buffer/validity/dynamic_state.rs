// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::SyncCommandBufferBuilder;
use crate::pipeline::GraphicsPipeline;
use std::error;
use std::fmt;

/// Checks whether states that are about to be set are correct.
pub(in super::super) fn check_dynamic_state_validity(
    builder: &SyncCommandBufferBuilder,
    pipeline: &GraphicsPipeline,
) -> Result<(), CheckDynamicStateValidityError> {
    let device = pipeline.device();

    if pipeline.has_dynamic_blend_constants() {
        if builder.current_blend_constants().is_none() {
            return Err(CheckDynamicStateValidityError::BlendConstantsNotSet);
        }
    }

    if pipeline.has_dynamic_depth_bounds() {
        if builder.current_blend_constants().is_none() {
            return Err(CheckDynamicStateValidityError::BlendConstantsNotSet);
        }
    }

    if pipeline.has_dynamic_line_width() {
        if builder.current_line_width().is_none() {
            return Err(CheckDynamicStateValidityError::LineWidthNotSet);
        }
    }

    if pipeline.has_dynamic_scissor() {
        for num in 0..pipeline.num_viewports() {
            if builder.current_scissor(num).is_none() {
                return Err(CheckDynamicStateValidityError::ScissorNotSet { num });
            }
        }
    }

    if pipeline.has_dynamic_stencil_compare_mask() {
        let state = builder.current_stencil_compare_mask();

        if state.front.is_none() || state.back.is_none() {
            return Err(CheckDynamicStateValidityError::StencilCompareMaskNotSet);
        }
    }

    if pipeline.has_dynamic_stencil_reference() {
        let state = builder.current_stencil_reference();

        if state.front.is_none() || state.back.is_none() {
            return Err(CheckDynamicStateValidityError::StencilReferenceNotSet);
        }
    }

    if pipeline.has_dynamic_stencil_write_mask() {
        let state = builder.current_stencil_write_mask();

        if state.front.is_none() || state.back.is_none() {
            return Err(CheckDynamicStateValidityError::StencilWriteMaskNotSet);
        }
    }

    if pipeline.has_dynamic_viewport() {
        for num in 0..pipeline.num_viewports() {
            if builder.current_viewport(num).is_none() {
                return Err(CheckDynamicStateValidityError::ViewportNotSet { num });
            }
        }
    }

    Ok(())
}

/// Error that can happen when validating dynamic states.
#[derive(Debug, Copy, Clone)]
pub enum CheckDynamicStateValidityError {
    /// The pipeline has dynamic blend constants, but no blend constants value was set.
    BlendConstantsNotSet,
    /// The pipeline has dynamic depth bounds, but no depth bounds value was set.
    DepthBoundsNotSet,
    /// The pipeline has a dynamic line width, but no line width value was set.
    LineWidthNotSet,
    /// The pipeline has a dynamic scissor, but the scissor for a slot used by the pipeline was not set.
    ScissorNotSet { num: u32 },
    /// The pipeline has dynamic stencil compare mask, but no compare mask was set for the front or back face.
    StencilCompareMaskNotSet,
    /// The pipeline has dynamic stencil reference, but no reference was set for the front or back face.
    StencilReferenceNotSet,
    /// The pipeline has dynamic stencil write mask, but no write mask was set for the front or back face.
    StencilWriteMaskNotSet,
    /// The pipeline has a dynamic viewport, but the viewport for a slot used by the pipeline was not set.
    ViewportNotSet { num: u32 },
}

impl error::Error for CheckDynamicStateValidityError {}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckDynamicStateValidityError::BlendConstantsNotSet => {
                    "the pipeline has dynamic blend constants, but no blend constants value was set"
                }
                CheckDynamicStateValidityError::DepthBoundsNotSet => {
                    "the pipeline has dynamic depth bounds, but no depth bounds value was set"
                }
                CheckDynamicStateValidityError::LineWidthNotSet => {
                    "the pipeline has a dynamic line width, but no line width value was set"
                }
                CheckDynamicStateValidityError::ScissorNotSet { .. } => {
                    "The pipeline has a dynamic scissor, but the scissor for a slot used by the pipeline was not set"
                }
                CheckDynamicStateValidityError::StencilCompareMaskNotSet => {
                    "the pipeline has dynamic stencil compare mask, but no compare mask was set for the front or back face"
                }
                CheckDynamicStateValidityError::StencilReferenceNotSet => {
                    "the pipeline has dynamic stencil reference, but no reference was set for the front or back face"
                }
                CheckDynamicStateValidityError::StencilWriteMaskNotSet => {
                    "the pipeline has dynamic stencil write mask, but no write mask was set for the front or back face"
                }
                CheckDynamicStateValidityError::ViewportNotSet { .. } => {
                    "the pipeline has a dynamic viewport, but the viewport for a slot used by the pipeline was not set"
                }
            }
        )
    }
}

// TODO: tests
