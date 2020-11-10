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

use descriptor::descriptor::DescriptorDescSupersetError;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutDesc;

/// Checks whether descriptor sets are compatible with the pipeline.
pub fn check_descriptor_sets_validity<Pl, D>(
    pipeline: &Pl,
    descriptor_sets: &D,
) -> Result<(), CheckDescriptorSetsValidityError>
where
    Pl: ?Sized + PipelineLayoutDesc,
    D: ?Sized + DescriptorSetsCollection,
{
    // What's important is not that the pipeline layout and the descriptor sets *match*. Instead
    // what's important is that the descriptor sets are a superset of the pipeline layout. It's not
    // a problem if the descriptor sets provide more elements than expected.

    for set_num in 0..pipeline.num_sets() {
        for binding_num in 0..pipeline.num_bindings_in_set(set_num).unwrap_or(0) {
            let set_desc = descriptor_sets.descriptor(set_num, binding_num);
            let pipeline_desc = pipeline.descriptor(set_num, binding_num);

            let (set_desc, pipeline_desc) = match (set_desc, pipeline_desc) {
                (Some(s), Some(p)) => (s, p),
                (None, Some(_)) => {
                    return Err(CheckDescriptorSetsValidityError::MissingDescriptor {
                        set_num: set_num,
                        binding_num: binding_num,
                    })
                }
                (Some(_), None) => continue,
                (None, None) => continue,
            };

            if let Err(err) = set_desc.is_superset_of(&pipeline_desc) {
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
    fn cause(&self) -> Option<&dyn error::Error> {
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
