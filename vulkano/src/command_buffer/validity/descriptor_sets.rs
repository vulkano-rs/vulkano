// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use crate::descriptor_set::layout::DescriptorDescSupersetError;
use crate::descriptor_set::DescriptorSetWithOffsets;
use crate::pipeline::layout::PipelineLayout;

/// Checks whether descriptor sets are compatible with the pipeline.
pub fn check_descriptor_sets_validity(
    pipeline_layout: &PipelineLayout,
    descriptor_sets: &[DescriptorSetWithOffsets],
) -> Result<(), CheckDescriptorSetsValidityError> {
    // What's important is not that the pipeline layout and the descriptor sets *match*. Instead
    // what's important is that the descriptor sets are a superset of the pipeline layout. It's not
    // a problem if the descriptor sets provide more elements than expected.

    for (set_num, set) in pipeline_layout.descriptor_set_layouts().iter().enumerate() {
        for (binding_num, pipeline_desc) in
            (0..set.num_bindings()).filter_map(|i| set.descriptor(i).map(|d| (i, d)))
        {
            let set_desc = descriptor_sets
                .get(set_num)
                .and_then(|so| so.as_ref().0.layout().descriptor(binding_num));

            let set_desc = match set_desc {
                Some(s) => s,
                None => {
                    return Err(CheckDescriptorSetsValidityError::MissingDescriptor {
                        set_num: set_num,
                        binding_num: binding_num,
                    })
                }
            };

            if let Err(err) = set_desc.ensure_superset_of(&pipeline_desc) {
                return Err(CheckDescriptorSetsValidityError::IncompatibleDescriptor {
                    error: err,
                    set_num: set_num,
                    binding_num: binding_num,
                });
            }
        }
    }

    Ok(())
}

/// Error that can happen when checking descriptor sets validity.
#[derive(Debug, Clone)]
pub enum CheckDescriptorSetsValidityError {
    /// A descriptor is missing in the descriptor sets that were provided.
    MissingDescriptor {
        /// The index of the set of the descriptor.
        set_num: usize,
        /// The binding number of the descriptor.
        binding_num: usize,
    },

    /// A descriptor in the provided sets is not compatible with what is expected.
    IncompatibleDescriptor {
        /// The reason why the two descriptors aren't compatible.
        error: DescriptorDescSupersetError,
        /// The index of the set of the descriptor.
        set_num: usize,
        /// The binding number of the descriptor.
        binding_num: usize,
    },
}

impl error::Error for CheckDescriptorSetsValidityError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            CheckDescriptorSetsValidityError::IncompatibleDescriptor { ref error, .. } => {
                Some(error)
            }
            _ => None,
        }
    }
}

impl fmt::Display for CheckDescriptorSetsValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckDescriptorSetsValidityError::MissingDescriptor { .. } => {
                    "a descriptor is missing in the descriptor sets that were provided"
                }
                CheckDescriptorSetsValidityError::IncompatibleDescriptor { .. } => {
                    "a descriptor in the provided sets is not compatible with what is expected"
                }
            }
        )
    }
}

// TODO: tests
