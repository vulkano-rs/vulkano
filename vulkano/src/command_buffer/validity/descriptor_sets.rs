// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use descriptor::pipeline_layout::PipelineLayoutAbstract;

/// Checks whether descriptor sets are compatible with the pipeline.
pub fn check_descriptor_sets_validity<Pl, D>(pipeline: &Pl, descriptor_sets: &D)
                                             -> Result<(), CheckDescriptorSetsValidityError>
    where Pl: ?Sized + PipelineLayoutAbstract,
          D: ?Sized,
{
    // TODO: implement

    Ok(())
}

/// Error that can happen when checking descriptor sets validity.
#[derive(Debug, Copy, Clone)]
pub enum CheckDescriptorSetsValidityError {
    /// The descriptor sets are incompatible with the pipeline layout.
    IncompatibleDescriptorSets,
}

impl error::Error for CheckDescriptorSetsValidityError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckDescriptorSetsValidityError::IncompatibleDescriptorSets => {
                "the descriptor sets are incompatible with the pipeline layout"
            },
        }
    }
}

impl fmt::Display for CheckDescriptorSetsValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
