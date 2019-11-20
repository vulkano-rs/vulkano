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

    if pipeline.has_dynamic_stencil_compare_mask() {
        if let None = state.compare_mask {
            return Err(CheckDynamicStateValidityError::CompareMaskMissing);
        }

    } else {
        if state.compare_mask.is_some() {
            return Err(CheckDynamicStateValidityError::CompareMaskNotDynamic);
        }
    }

    if pipeline.has_dynamic_stencil_write_mask() {
        if let None = state.write_mask {
            return Err(CheckDynamicStateValidityError::WriteMaskMissing);
        }

    } else {
        if state.write_mask.is_some() {
            return Err(CheckDynamicStateValidityError::WriteMaskNotDynamic);
        }
    }

    if pipeline.has_dynamic_stencil_reference() {
        if let None = state.reference {
            return Err(CheckDynamicStateValidityError::ReferenceMissing);
        }

    } else {
        if state.reference.is_some() {
            return Err(CheckDynamicStateValidityError::ReferenceNotDynamic);
        }
    }

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
    /// Passed dynamic compare mask, while the pipeline doesn't have the compare mask set as dynamic.
    CompareMaskNotDynamic,
    /// The pipeline has dynamic compare mask, but no compare mask was passed.
    CompareMaskMissing,
    /// Passed dynamic write mask, while the pipeline doesn't have the write mask set as dynamic.
    WriteMaskNotDynamic,
    /// The pipeline has dynamic write mask, but no write mask was passed.
    WriteMaskMissing,
    /// Passed dynamic reference, while the pipeline doesn't have the reference set as dynamic.
    ReferenceNotDynamic,
    /// The pipeline has dynamic reference, but no reference was passed.
    ReferenceMissing,
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
            CheckDynamicStateValidityError::CompareMaskNotDynamic => {
                "passed dynamic compare mask, while the pipeline doesn't have compare mask set as dynamic"
            },
            CheckDynamicStateValidityError::CompareMaskMissing => {
                "the pipeline has dynamic compare mask, but no compare mask was passed"
            },
            CheckDynamicStateValidityError::WriteMaskNotDynamic => {
                "passed dynamic write mask, while the pipeline doesn't have write mask set as dynamic"
            },
            CheckDynamicStateValidityError::WriteMaskMissing => {
                "the pipeline has dynamic write mask, but no write mask was passed"
            },
            CheckDynamicStateValidityError::ReferenceNotDynamic => {
                "passed dynamic Reference, while the pipeline doesn't have reference set as dynamic"
            },
            CheckDynamicStateValidityError::ReferenceMissing => {
                "the pipeline has dynamic reference, but no reference was passed"
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
