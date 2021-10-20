// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::CommandBufferState;
use crate::command_buffer::synced::SetOrPush;
use crate::descriptor_set::layout::DescriptorSetCompatibilityError;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::PipelineBindPoint;
use crate::VulkanObject;
use std::error;
use std::fmt;

/// Checks whether descriptor sets are compatible with the pipeline.
pub(in super::super) fn check_descriptor_sets_validity(
    current_state: CommandBufferState,
    pipeline_layout: &PipelineLayout,
    pipeline_bind_point: PipelineBindPoint,
) -> Result<(), CheckDescriptorSetsValidityError> {
    if pipeline_layout.descriptor_set_layouts().is_empty() {
        return Ok(());
    }

    let bindings_pipeline_layout = match current_state
        .descriptor_sets_pipeline_layout(pipeline_bind_point)
    {
        Some(x) => x,
        None => return Err(CheckDescriptorSetsValidityError::MissingDescriptorSet { set_num: 0 }),
    };

    if bindings_pipeline_layout.internal_object() != pipeline_layout.internal_object()
        && bindings_pipeline_layout.push_constant_ranges() != pipeline_layout.push_constant_ranges()
    {
        return Err(CheckDescriptorSetsValidityError::IncompatiblePushConstants);
    }

    for (set_num, pipeline_set) in pipeline_layout.descriptor_set_layouts().iter().enumerate() {
        let set_num = set_num as u32;

        let descriptor_set = match current_state.descriptor_set(pipeline_bind_point, set_num) {
            Some(s) => s,
            None => return Err(CheckDescriptorSetsValidityError::MissingDescriptorSet { set_num }),
        };

        let descriptor_set_layout = match descriptor_set {
            SetOrPush::Set(descriptor_set, _dynamic_offsets) => descriptor_set.layout(),
            SetOrPush::Push(descriptor_writes) => descriptor_writes.layout(),
        };

        match pipeline_set.ensure_compatible_with_bind(descriptor_set_layout) {
            Ok(_) => (),
            Err(error) => {
                return Err(
                    CheckDescriptorSetsValidityError::IncompatibleDescriptorSet { error, set_num },
                );
            }
        }
    }

    Ok(())
}

/// Error that can happen when checking descriptor sets validity.
#[derive(Debug, Clone)]
pub enum CheckDescriptorSetsValidityError {
    MissingDescriptorSet {
        set_num: u32,
    },
    IncompatibleDescriptorSet {
        /// The error returned by the descriptor set.
        error: DescriptorSetCompatibilityError,
        /// The index of the set of the descriptor.
        set_num: u32,
    },
    IncompatiblePushConstants,
}

impl error::Error for CheckDescriptorSetsValidityError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::IncompatibleDescriptorSet { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for CheckDescriptorSetsValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::MissingDescriptorSet { set_num } => {
                write!(fmt, "descriptor set {} has not been not bound, but is required by the pipeline layout", set_num)
            }
            Self::IncompatibleDescriptorSet { set_num, .. } => {
                write!(fmt, "compatibility error in descriptor set {}", set_num)
            }
            Self::IncompatiblePushConstants => {
                write!(fmt, "the push constant ranges in the bound pipeline do not match the ranges of layout used to bind the descriptor sets")
            }
        }
    }
}

// TODO: tests
