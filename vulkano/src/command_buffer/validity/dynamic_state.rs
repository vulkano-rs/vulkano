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

    if pipeline.has_dynamic_line_width() {
        if let Some(value) = state.line_width {
            if value != 1.0 && !pipeline.device().enabled_features().wide_lines {
                return Err(CheckDynamicStateValidityError::LineWidthMissingExtension);
            }
        } else {
            return Err(CheckDynamicStateValidityError::LineWidthMissing);
        }

    } else {
        if state.line_width.is_some() {
            return Err(CheckDynamicStateValidityError::LineWidthNotDynamic);
        }
    }

    if pipeline.has_dynamic_viewports() {
        if let Some(ref viewports) = state.viewports {
            if viewports.len() != pipeline.num_viewports() as usize {
                return Err(CheckDynamicStateValidityError::ViewportsCountMismatch {
                               expected: pipeline.num_viewports() as usize,
                               obtained: viewports.len(),
                           });
            }
        } else {
            return Err(CheckDynamicStateValidityError::ViewportsMissing);
        }

    } else {
        if state.viewports.is_some() {
            return Err(CheckDynamicStateValidityError::ViewportsNotDynamic);
        }
    }

    if pipeline.has_dynamic_scissors() {
        if let Some(ref scissors) = state.scissors {
            if scissors.len() != pipeline.num_viewports() as usize {
                return Err(CheckDynamicStateValidityError::ScissorsCountMismatch {
                               expected: pipeline.num_viewports() as usize,
                               obtained: scissors.len(),
                           });
            }
        } else {
            return Err(CheckDynamicStateValidityError::ScissorsMissing);
        }

    } else {
        if state.scissors.is_some() {
            return Err(CheckDynamicStateValidityError::ScissorsNotDynamic);
        }
    }

    // TODO: don't forget to implement the rest

    Ok(())
}

/// Error that can happen when validating dynamic states.
#[derive(Debug, Copy, Clone)]
pub enum CheckDynamicStateValidityError {
    /// Passed a dynamic line width, while the pipeline doesn't have line width set as dynamic.
    LineWidthNotDynamic,
    /// The pipeline has a dynamic line width, but no line width value was passed.
    LineWidthMissing,
    /// The `wide_lines` extension must be enabled in order to use line width values different
    /// from 1.0.
    LineWidthMissingExtension,
    /// Passed dynamic viewports, while the pipeline doesn't have viewports set as dynamic.
    ViewportsNotDynamic,
    /// The pipeline has dynamic viewports, but no viewports were passed.
    ViewportsMissing,
    /// The number of dynamic viewports doesn't match the expected number of viewports.
    ViewportsCountMismatch {
        /// Expected number of viewports.
        expected: usize,
        /// Number of viewports that were passed.
        obtained: usize,
    },
    /// Passed dynamic scissors, while the pipeline doesn't have scissors set as dynamic.
    ScissorsNotDynamic,
    /// The pipeline has dynamic scissors, but no scissors were passed.
    ScissorsMissing,
    /// The number of dynamic scissors doesn't match the expected number of scissors.
    ScissorsCountMismatch {
        /// Expected number of scissors.
        expected: usize,
        /// Number of scissors that were passed.
        obtained: usize,
    },
}

impl error::Error for CheckDynamicStateValidityError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckDynamicStateValidityError::LineWidthNotDynamic => {
                "passed a dynamic line width, while the pipeline doesn't have line width set as \
                 dynamic"
            },
            CheckDynamicStateValidityError::LineWidthMissing => {
                "the pipeline has a dynamic line width, but no line width value was passed"
            },
            CheckDynamicStateValidityError::LineWidthMissingExtension => {
                "the `wide_lines` extension must be enabled in order to use line width values \
                 different from 1.0"
            },
            CheckDynamicStateValidityError::ViewportsNotDynamic => {
                "passed dynamic viewports, while the pipeline doesn't have viewports set as \
                 dynamic"
            },
            CheckDynamicStateValidityError::ViewportsMissing => {
                "the pipeline has dynamic viewports, but no viewports were passed"
            },
            CheckDynamicStateValidityError::ViewportsCountMismatch { .. } => {
                "the number of dynamic viewports doesn't match the expected number of viewports"
            },
            CheckDynamicStateValidityError::ScissorsNotDynamic => {
                "passed dynamic scissors, while the pipeline doesn't have scissors set as dynamic"
            },
            CheckDynamicStateValidityError::ScissorsMissing => {
                "the pipeline has dynamic scissors, but no scissors were passed"
            },
            CheckDynamicStateValidityError::ScissorsCountMismatch { .. } => {
                "the number of dynamic scissors doesn't match the expected number of scissors"
            },
        }
    }
}

impl fmt::Display for CheckDynamicStateValidityError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

// TODO: tests
