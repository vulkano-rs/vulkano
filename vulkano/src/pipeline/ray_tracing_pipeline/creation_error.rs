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

use descriptor::pipeline_layout::PipelineLayoutNotSupersetError;
use Error;
use OomError;

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RayTracingPipelineCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The pipeline layout is not compatible with what the shaders expect.
    IncompatiblePipelineLayout(PipelineLayoutNotSupersetError),

    /// There are no raygen shaders passed to pipeline.
    NoRaygenShader,

    /// The wrong type of shader has been passed.
    ///
    /// For example you passed a hit shader as the raygen.
    WrongShaderType,

    /// The `rayTracing` feature must be enabled in order to use ray tracing.
    RayTracingFeatureNotEnabled,

    /// The maximum recursion depth of the pipeline has been exceeded.
    MaxRecursionDepthExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },
}

impl error::Error for RayTracingPipelineCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            RayTracingPipelineCreationError::OomError(_) => "not enough memory available",
            RayTracingPipelineCreationError::IncompatiblePipelineLayout(_) => {
                "the pipeline layout is not compatible with what the shaders expect"
            }
            RayTracingPipelineCreationError::NoRaygenShader => {
                "at least one of the shader stages must be a raygen shader"
            }
            RayTracingPipelineCreationError::WrongShaderType => {
                "the wrong type of shader has been passed"
            }
            RayTracingPipelineCreationError::RayTracingFeatureNotEnabled => {
                "the `rayTracing` feature must be enabled in order to use ray tracing."
            }
            RayTracingPipelineCreationError::MaxRecursionDepthExceeded { .. } => {
                "the maximum recursion depth has been exceeded"
            }
        }
    }

    #[inline]
    fn cause(&self) -> Option<&dyn error::Error> {
        match *self {
            RayTracingPipelineCreationError::OomError(ref err) => Some(err),
            RayTracingPipelineCreationError::IncompatiblePipelineLayout(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for RayTracingPipelineCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for RayTracingPipelineCreationError {
    #[inline]
    fn from(err: OomError) -> RayTracingPipelineCreationError {
        RayTracingPipelineCreationError::OomError(err)
    }
}

impl From<PipelineLayoutNotSupersetError> for RayTracingPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutNotSupersetError) -> RayTracingPipelineCreationError {
        RayTracingPipelineCreationError::IncompatiblePipelineLayout(err)
    }
}

impl From<Error> for RayTracingPipelineCreationError {
    #[inline]
    fn from(err: Error) -> RayTracingPipelineCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                RayTracingPipelineCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                RayTracingPipelineCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
