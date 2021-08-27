// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::synced::SyncCommandBufferBuilder,
    pipeline::{ComputePipeline, GraphicsPipeline},
};
use std::{error, fmt};

pub(in super::super) fn check_pipeline_compute(
    builder: &SyncCommandBufferBuilder,
) -> Result<&ComputePipeline, CheckPipelineError> {
    let pipeline = match builder.bound_pipeline_compute() {
        Some(x) => x,
        None => return Err(CheckPipelineError::PipelineNotBound),
    };

    Ok(pipeline)
}

pub(in super::super) fn check_pipeline_graphics(
    builder: &SyncCommandBufferBuilder,
) -> Result<&GraphicsPipeline, CheckPipelineError> {
    let pipeline = match builder.bound_pipeline_graphics() {
        Some(x) => x,
        None => return Err(CheckPipelineError::PipelineNotBound),
    };

    Ok(pipeline)
}

/// Error that can happen when checking whether the pipeline is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckPipelineError {
    /// No pipeline was bound to the bind point used by the operation.
    PipelineNotBound,
}

impl error::Error for CheckPipelineError {}

impl fmt::Display for CheckPipelineError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            CheckPipelineError::PipelineNotBound => write!(
                fmt,
                "no pipeline was bound to the bind point used by the operation",
            ),
        }
    }
}
