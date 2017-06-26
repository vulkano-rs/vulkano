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

use command_buffer::DynamicState;
use pipeline::GraphicsPipelineAbstract;

/// Checks whether states that are about to be set are correct.
pub fn check_dynamic_state_validity<Pl>(pipeline: &Pl, state: &DynamicState)
                                        -> Result<(), CheckDynamicStateValidityError>
    where Pl: GraphicsPipelineAbstract
{
    let device = pipeline.device();

    // FIXME:

    Ok(())
}

/// Error that can happen when attempting to add a `fill_buffer` command.
#[derive(Debug, Copy, Clone)]
pub enum CheckDynamicStateValidityError {
}

impl error::Error for CheckDynamicStateValidityError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
        }
    }
}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
