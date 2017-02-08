// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use command_buffer::cb::AddCommand;
use command_buffer::cmd::CmdBindDescriptorSets;
use command_buffer::cmd::CmdBindDescriptorSetsError;
use command_buffer::cmd::CmdBindPipeline;
use command_buffer::cmd::CmdDispatchRaw;
use command_buffer::cmd::CmdDispatchRawError;
use command_buffer::cmd::CmdPushConstants;
use command_buffer::cmd::CmdPushConstantsError;
use descriptor::descriptor_set::DescriptorSetsCollection;
use pipeline::ComputePipelineAbstract;

/// Command that executes a compute shader.
pub struct CmdDispatch<P, S, Pc> {
    push_constants: CmdPushConstants<Pc, P>,
    descriptor_sets: CmdBindDescriptorSets<S, P>,
    bind_pipeline: CmdBindPipeline<P>,
    dispatch_raw: CmdDispatchRaw,
}

impl<P, S, Pc> CmdDispatch<P, S, Pc>
    where P: ComputePipelineAbstract, S: DescriptorSetsCollection
{
    /// See the documentation of the `dispatch` method.
    pub fn new(dimensions: [u32; 3], pipeline: P, sets: S, push_constants: Pc)
               -> Result<CmdDispatch<P, S, Pc>, CmdDispatchError>
        where P: Clone
    {
        let bind_pipeline = CmdBindPipeline::bind_compute_pipeline(pipeline.clone());
        let descriptor_sets = try!(CmdBindDescriptorSets::new(true, pipeline.clone(), sets));
        let push_constants = try!(CmdPushConstants::new(pipeline.clone(), push_constants));
        let dispatch_raw = try!(unsafe { CmdDispatchRaw::new(dimensions) });

        Ok(CmdDispatch {
            push_constants: push_constants,
            descriptor_sets: descriptor_sets,
            bind_pipeline: bind_pipeline,
            dispatch_raw: dispatch_raw,
        })
    }
}

unsafe impl<Cb, P, S, Pc, O, O1, O2, O3> AddCommand<CmdDispatch<P, S, Pc>> for Cb
    where Cb: AddCommand<CmdPushConstants<Pc, P>, Out = O1>,
          O1: AddCommand<CmdBindDescriptorSets<S, P>, Out = O2>,
          O2: AddCommand<CmdBindPipeline<P>, Out = O3>,
          O3: AddCommand<CmdDispatchRaw, Out = O>
{
    type Out = O;

    #[inline]
    fn add(self, command: CmdDispatch<P, S, Pc>) -> O {
        self.add(command.push_constants)
            .add(command.descriptor_sets)
            .add(command.bind_pipeline)
            .add(command.dispatch_raw)
    }
}

/// Error that can happen when creating a `CmdDispatch`.
#[derive(Debug, Copy, Clone)]
pub enum CmdDispatchError {
    /// The dispatch dimensions are larger than the hardware limits.
    DispatchRawError(CmdDispatchRawError),
    /// Error while binding descriptor sets.
    BindDescriptorSetsError(CmdBindDescriptorSetsError),
    /// Error while setting push constants.
    PushConstantsError(CmdPushConstantsError),
}

impl From<CmdDispatchRawError> for CmdDispatchError {
    #[inline]
    fn from(err: CmdDispatchRawError) -> CmdDispatchError {
        CmdDispatchError::DispatchRawError(err)
    }
}

impl From<CmdBindDescriptorSetsError> for CmdDispatchError {
    #[inline]
    fn from(err: CmdBindDescriptorSetsError) -> CmdDispatchError {
        CmdDispatchError::BindDescriptorSetsError(err)
    }
}

impl From<CmdPushConstantsError> for CmdDispatchError {
    #[inline]
    fn from(err: CmdPushConstantsError) -> CmdDispatchError {
        CmdDispatchError::PushConstantsError(err)
    }
}

impl error::Error for CmdDispatchError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdDispatchError::DispatchRawError(_) => {
                "the dispatch dimensions are larger than the hardware limits"
            },
            CmdDispatchError::BindDescriptorSetsError(_) => {
                "error while binding descriptor sets"
            },
            CmdDispatchError::PushConstantsError(_) => {
                "error while setting push constants"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            CmdDispatchError::DispatchRawError(ref err) => Some(err),
            CmdDispatchError::BindDescriptorSetsError(ref err) => Some(err),
            CmdDispatchError::PushConstantsError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for CmdDispatchError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
