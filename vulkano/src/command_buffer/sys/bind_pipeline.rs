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
use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::DynamicState;
use pipeline::GraphicsPipeline;
use pipeline::ComputePipeline;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that copies data between buffers.
pub struct GraphicsPipelineBindCommand {
    keep_alive: Arc<KeepAlive + 'static>,
    device: vk::Device,

    pipeline: vk::Pipeline,

    viewports: Option<(u32, SmallVec<[vk::Viewport; 2]>)>,
    scissors: Option<(u32, SmallVec<[vk::Rect2D; 2]>)>,
    line_width: Option<f32>,
    depth_bias: Option<(f32, f32, f32)>,
    blend_constants: Option<[f32; 4]>,
    depth_bounds: Option<(f32, f32)>,
    stencil_compare_mask_front: Option<u32>,
    stencil_compare_mask_back: Option<u32>,
    stencil_write_mask_front: Option<u32>,
    stencil_write_mask_back: Option<u32>,
    stencil_reference_front: Option<u32>,
    stencil_reference_back: Option<u32>,
}

impl GraphicsPipelineBindCommand {
    pub fn new<V, Pl, Rp>(pipeline: &Arc<GraphicsPipeline<V, Pl, Rp>>, dynamic: &DynamicState)
                          -> Result<GraphicsPipelineBindCommand, GfxPipelineBindError>
        where V: Send + Sync + 'static,
              Pl: Send + Sync + 'static,
              Rp: Send + Sync + 'static
    {
        let limits = pipeline.device().physical_device().limits();

        // FIXME: return errors instead
        assert!(pipeline.has_dynamic_line_width());

        if let Some(line_width) = dynamic.line_width {
            if line_width < limits.line_width_range()[0] ||
               line_width > limits.line_width_range()[1]
            {
                return Err(GfxPipelineBindError::LineWidthOutOfRange);
            }
        }

        // FIXME: check all states

        Ok(GraphicsPipelineBindCommand {
            keep_alive: pipeline.clone(),
            device: pipeline.device().internal_object(),

            pipeline: pipeline.internal_object(),

            viewports: if let Some(ref list) = dynamic.viewports {
                Some((0, list.iter().map(|elem| elem.clone().into()).collect()))
            } else {
                None
            },
            scissors: if let Some(ref list) = dynamic.scissors {
                Some((0, list.iter().map(|elem| elem.clone().into()).collect()))
            } else {
                None
            },
            line_width: dynamic.line_width,
            depth_bias: None,       // TODO:
            blend_constants: None,      // TODO:
            depth_bounds: None,     // TODO:
            stencil_compare_mask_front: None,       // TODO:
            stencil_compare_mask_back: None,        // TODO:
            stencil_write_mask_front: None,     // TODO:
            stencil_write_mask_back: None,      // TODO:
            stencil_reference_front: None,      // TODO:
            stencil_reference_back: None,       // TODO:
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is not within a render pass.
    /// - Panicks if the pipeline was not allocated with the same device as the command buffer.
    /// - Panicks if the queue doesn't not support graphics operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            assert!(cb.within_render_pass);
            assert_eq!(self.device, cb.device().internal_object());
            assert!(cb.pool().queue_family().supports_graphics());

            cb.keep_alive.push(self.keep_alive.clone());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();

                if cb.current_graphics_pipeline != Some(self.pipeline) {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, self.pipeline);
                    cb.current_graphics_pipeline = Some(self.pipeline);
                }

                if let Some((first, ref viewports)) = self.viewports {
                    // TODO: check existing state
                    vk.CmdSetViewport(cmd, first, viewports.len() as u32, viewports.as_ptr());
                }

                if let Some((first, ref scissors)) = self.scissors {
                    // TODO: check existing state
                    vk.CmdSetScissor(cmd, first, scissors.len() as u32, scissors.as_ptr());
                }

                if let Some(l) = self.line_width {
                    if cb.current_dynamic_state.line_width != Some(l) {
                        vk.CmdSetLineWidth(cmd, l);
                        cb.current_dynamic_state.line_width = Some(l);
                    }
                }

                if let Some((co, cl, s)) = self.depth_bias {
                    // TODO: check existing state
                    vk.CmdSetDepthBias(cmd, co, cl, s);
                }

                if let Some(bc) = self.blend_constants {
                    // TODO: check existing state
                    vk.CmdSetBlendConstants(cmd, bc);
                }

                if let Some((mi, ma)) = self.depth_bounds {
                    // TODO: check existing state
                    vk.CmdSetDepthBounds(cmd, mi, ma);
                }

                // TODO: check existing state
                match (self.stencil_compare_mask_front, self.stencil_compare_mask_back) {
                    (Some(f), Some(b)) if f == b => {
                        vk.CmdSetStencilCompareMask(cmd, vk::STENCIL_FRONT_AND_BACK, f);
                    },
                    (Some(f), Some(b)) => {
                        vk.CmdSetStencilCompareMask(cmd, vk::STENCIL_FACE_FRONT_BIT, f);
                        vk.CmdSetStencilCompareMask(cmd, vk::STENCIL_FACE_BACK_BIT, b);
                    },
                    (None, Some(b)) => {
                        vk.CmdSetStencilCompareMask(cmd, vk::STENCIL_FACE_BACK_BIT, b);
                    },
                    (Some(f), None) => {
                        vk.CmdSetStencilCompareMask(cmd, vk::STENCIL_FACE_FRONT_BIT, f);
                    },
                    _ => ()
                }

                // TODO: check existing state
                match (self.stencil_write_mask_front, self.stencil_write_mask_back) {
                    (Some(f), Some(b)) if f == b => {
                        vk.CmdSetStencilWriteMask(cmd, vk::STENCIL_FRONT_AND_BACK, f);
                    },
                    (Some(f), Some(b)) => {
                        vk.CmdSetStencilWriteMask(cmd, vk::STENCIL_FACE_FRONT_BIT, f);
                        vk.CmdSetStencilWriteMask(cmd, vk::STENCIL_FACE_BACK_BIT, b);
                    },
                    (None, Some(b)) => {
                        vk.CmdSetStencilWriteMask(cmd, vk::STENCIL_FACE_BACK_BIT, b);
                    },
                    (Some(f), None) => {
                        vk.CmdSetStencilWriteMask(cmd, vk::STENCIL_FACE_FRONT_BIT, f);
                    },
                    _ => ()
                }

                // TODO: check existing state
                match (self.stencil_reference_front, self.stencil_reference_back) {
                    (Some(f), Some(b)) if f == b => {
                        vk.CmdSetStencilReference(cmd, vk::STENCIL_FRONT_AND_BACK, f);
                    },
                    (Some(f), Some(b)) => {
                        vk.CmdSetStencilReference(cmd, vk::STENCIL_FACE_FRONT_BIT, f);
                        vk.CmdSetStencilReference(cmd, vk::STENCIL_FACE_BACK_BIT, b);
                    },
                    (None, Some(b)) => {
                        vk.CmdSetStencilReference(cmd, vk::STENCIL_FACE_BACK_BIT, b);
                    },
                    (Some(f), None) => {
                        vk.CmdSetStencilReference(cmd, vk::STENCIL_FACE_FRONT_BIT, f);
                    },
                    _ => ()
                }
            }

            cb
        }
    }
}

error_ty!{GfxPipelineBindError => "Error that can happen when binding a graphics pipeline.",
    LineWidthOutOfRange => "the requested line width is out of range",
}

/// Prototype for a command that copies data between buffers.
pub struct ComputePipelineBindCommand {
    keep_alive: Arc<KeepAlive + 'static>,
    device: vk::Device,
    pipeline: vk::Pipeline,
}

impl ComputePipelineBindCommand {
    /// Builds a new compute pipeline binding command.
    #[inline]
    pub fn new<Pl>(pipeline: &Arc<ComputePipeline<Pl>>) -> ComputePipelineBindCommand
        where Pl: Send + Sync + 'static,
    {
        ComputePipelineBindCommand {
            keep_alive: pipeline.clone(),
            device: pipeline.device().internal_object(),
            pipeline: pipeline.internal_object(),
        }
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is within a render pass.
    /// - Panicks if the pipeline was not allocated with the same device as the command buffer.
    /// - Panicks if the queue doesn't not support compute operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            // Various checks.
            assert!(!cb.within_render_pass);
            assert_eq!(self.device, cb.device().internal_object());
            assert!(cb.pool().queue_family().supports_compute());

            cb.keep_alive.push(self.keep_alive.clone());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();

                if cb.current_compute_pipeline != Some(self.pipeline) {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
                    cb.current_compute_pipeline = Some(self.pipeline);
                }
            }

            cb
        }
    }
}
