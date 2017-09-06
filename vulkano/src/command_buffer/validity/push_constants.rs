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
use descriptor::pipeline_layout::PipelineLayoutPushConstantsCompatible;

/// Checks whether push constants are compatible with the pipeline.
pub fn check_push_constants_validity<Pl, Pc>(pipeline: &Pl, push_constants: &Pc)
                                             -> Result<(), CheckPushConstantsValidityError>
    where Pl: ?Sized + PipelineLayoutAbstract + PipelineLayoutPushConstantsCompatible<Pc>,
          Pc: ?Sized
{
    if !pipeline.is_compatible(push_constants) {
        return Err(CheckPushConstantsValidityError::IncompatiblePushConstants);
    }

    Ok(())
}

/// Error that can happen when checking push constants validity.
#[derive(Debug, Copy, Clone)]
pub enum CheckPushConstantsValidityError {
    /// The push constants are incompatible with the pipeline layout.
    IncompatiblePushConstants,
}

impl error::Error for CheckPushConstantsValidityError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckPushConstantsValidityError::IncompatiblePushConstants => {
                "the push constants are incompatible with the pipeline layout"
            },
        }
    }
}

impl fmt::Display for CheckPushConstantsValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
