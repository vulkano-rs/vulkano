// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use command_buffer::CommandBufferBuilder;
use command_buffer::cmd::CommandsList;
use device::Device;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds a command that binds pipeline at the end of it.
///
/// > **Note**: In reality, what this command does is ensure that the requested pipeline is bound
/// > after it is executed. In other words, if the command is aware that the same pipeline is
/// > already bound, then it won't bind it again. This optimization is essential, as binding a
/// > pipeline has a non-negligible overhead.
pub struct CmdBindPipeline<L, P> where L: CommandsList {
    // Parent commands list.
    previous: L,
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

impl<L> CmdBindPipeline<L, ()> where L: CommandsList {
    /// Builds a command that binds a compute pipeline to the compute pipeline bind point.
    ///
    /// Use this command right before a compute dispatch.
    #[inline]
    pub fn bind_compute_pipeline<Pl>(previous: L, pipeline: Arc<ComputePipeline<Pl>>)
                                     -> CmdBindPipeline<L, Arc<ComputePipeline<Pl>>>
    {
        let raw_pipeline = pipeline.internal_object();
        let device = pipeline.device().clone();

        CmdBindPipeline {
            previous: previous,
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
    pub fn bind_graphics_pipeline<V, Pl, R>(previous: L, pipeline: Arc<GraphicsPipeline<V, Pl, R>>)
                                            -> CmdBindPipeline<L, Arc<GraphicsPipeline<V, Pl, R>>>
    {
        let raw_pipeline = pipeline.internal_object();
        let device = pipeline.device().clone();

        CmdBindPipeline {
            previous: previous,
            raw_pipeline: raw_pipeline,
            pipeline_ty: vk::PIPELINE_BIND_POINT_GRAPHICS,
            device: device,
            pipeline: pipeline,
        }
    }
}

impl<L, P> CmdBindPipeline<L, P> where L: CommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: CommandBufferBuilder<'a>) -> CommandBufferBuilder<'a> {
        let mut builder = self.previous.append(builder);

        assert_eq!(self.device.internal_object(), builder.device.internal_object());

        // Returning now if the pipeline object is already bound.
        // Note that we need to perform this check after validating the device, otherwise the
        // pipeline ID could match by mistake.
        match self.pipeline_ty {
            vk::PIPELINE_BIND_POINT_GRAPHICS => {
                if builder.bound_graphics_pipeline == self.raw_pipeline {
                    return builder;
                } else {
                    builder.bound_graphics_pipeline = self.raw_pipeline;
                }
            },
            vk::PIPELINE_BIND_POINT_COMPUTE => {
                if builder.bound_compute_pipeline == self.raw_pipeline {
                    return builder;
                } else {
                    builder.bound_compute_pipeline = self.raw_pipeline;
                }
            },
            _ => unreachable!()
        }

        // Binding for real.
        unsafe {
            let vk = builder.device.pointers();
            let cmd = builder.command_buffer.clone().take().unwrap();
            vk.CmdBindPipeline(cmd, self.pipeline_ty, self.raw_pipeline);
        }

        builder
    }
}

// TODO:
/*unsafe impl<'a, L, B, D: ?Sized> CommandsListPossibleOutsideRenderPass
    for CmdUnsyncedUpdate<'a, L, B, D>
    where B: Buffer,
          L: CommandsList,
          D: Copy + 'static,
{
    #[inline]
    fn is_outside_render_pass(&self) -> bool {
        true
    }
}*/
