// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::sync::Arc;

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that binds a pipeline to a command buffer.
pub struct CmdBindPipeline<P> {
    // The raw pipeline object to bind.
    raw_pipeline: vk::Pipeline,
    // The raw Vulkan enum representing the kind of pipeline.
    pipeline_ty: vk::PipelineBindPoint,
    // The device of the pipeline object, so that we can compare it with the command buffer's
    // device.
    device: Arc<Device>,
    // The pipeline object to bind. Unused, but we need to keep it alive.
    pipeline: P,
}

impl<P> CmdBindPipeline<P> {
    /// Builds a command that binds a compute pipeline to the compute pipeline bind point.
    ///
    /// Use this command right before a compute dispatch.
    #[inline]
    pub fn bind_compute_pipeline(pipeline: P) -> CmdBindPipeline<P>
        where P: ComputePipelineAbstract
    {
        let raw_pipeline = pipeline.inner().internal_object();
        let device = pipeline.device().clone();

        CmdBindPipeline {
            raw_pipeline: raw_pipeline,
            pipeline_ty: vk::PIPELINE_BIND_POINT_COMPUTE,
            device: device,
            pipeline: pipeline,
        }
    }

    /// Builds a command that binds a graphics pipeline to the graphics pipeline bind point.
    ///
    /// Use this command right before a draw command.
    #[inline]
    pub fn bind_graphics_pipeline(pipeline: P) -> CmdBindPipeline<P>
        where P: GraphicsPipelineAbstract
    {
        let raw_pipeline = GraphicsPipelineAbstract::inner(&pipeline).internal_object();
        let device = pipeline.device().clone();

        CmdBindPipeline {
            raw_pipeline: raw_pipeline,
            pipeline_ty: vk::PIPELINE_BIND_POINT_GRAPHICS,
            device: device,
            pipeline: pipeline,
        }
    }

    /// This disables the command but keeps it alive. All getters still return the same value, but
    /// executing the command will not do anything.
    #[inline]
    pub fn disabled(mut self) -> CmdBindPipeline<P> {
        self.raw_pipeline = 0;
        self
    }

    /// Returns the device the pipeline is assocated with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// True if this is the graphics pipeline. False if the compute pipeline.
    // TODO: should be an enum?
    #[inline]
    pub fn is_graphics(&self) -> bool {
        self.pipeline_ty == vk::PIPELINE_BIND_POINT_GRAPHICS
    }

    /// Returns an object giving access to the pipeline object that will be bound.
    #[inline]
    pub fn sys(&self) -> CmdBindPipelineSys {
        CmdBindPipelineSys(self.raw_pipeline, PhantomData)
    }
}

unsafe impl<'a, P, Pl> AddCommand<&'a CmdBindPipeline<Pl>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdBindPipeline<Pl>) -> Self::Out {
        if command.raw_pipeline != 0 {
            unsafe {
                let vk = self.device().pointers();
                let cmd = self.internal_object();
                vk.CmdBindPipeline(cmd, command.pipeline_ty, command.raw_pipeline);
            }
        }

        self
    }
}

/// Object that represents the internals of the bind pipeline command.
#[derive(Debug, Copy, Clone)]
pub struct CmdBindPipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for CmdBindPipelineSys<'a> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.0
    }
}
