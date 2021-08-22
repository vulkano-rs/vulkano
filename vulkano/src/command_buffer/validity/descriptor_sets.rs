// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor_set::layout::DescriptorSetCompatibilityError;
use crate::descriptor_set::DescriptorSetWithOffsets;
use crate::pipeline::layout::PipelineLayout;
use std::error;
use std::fmt;

/// Checks whether descriptor sets are compatible with the pipeline.
pub fn check_descriptor_sets_validity(
    pipeline_layout: &PipelineLayout,
    descriptor_sets: &[DescriptorSetWithOffsets],
) -> Result<(), CheckDescriptorSetsValidityError> {
    for (set_index, pipeline_set) in pipeline_layout.descriptor_set_layouts().iter().enumerate() {
        let set_num = set_index as u32;

        let descriptor_set = match descriptor_sets.get(set_index) {
            Some(s) => s,
            None => return Err(CheckDescriptorSetsValidityError::MissingDescriptorSet { set_num }),
        };

        match pipeline_set.ensure_compatible_with_bind(descriptor_set.as_ref().0.layout()) {
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
        }
    }
}

// TODO: tests
