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

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::DynamicState;
use command_buffer::pool::CommandPool;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::bind_pipeline::GraphicsPipelineBindCommand;
use command_buffer::sys::bind_pipeline::GfxPipelineBindError;
use command_buffer::sys::bind_ib::IndexBufferBindCommand;
use command_buffer::sys::bind_ib::IndexBufferBindError;
use command_buffer::sys::bind_sets::DescriptorSetsBindCommand;
use command_buffer::sys::bind_sets::DescriptorSetsBindError;
use command_buffer::sys::bind_vb::VertexSourceBindCommand;
use command_buffer::sys::bind_vb::VertexBufferBindError;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutSetsCompatible;
use descriptor::pipeline_layout::PipelineLayoutPushConstantsCompatible;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Source;

use VulkanPointers;
use vk;

/// Prototype for a command that draws.
pub struct DrawCommand {
    bind_pipeline: GraphicsPipelineBindCommand,
    bind_vertex_buffer: VertexSourceBindCommand,
    bind_index_buffer: Option<IndexBufferBindCommand>,
    bind_descriptor_sets: DescriptorSetsBindCommand,

    ty: DrawTyInternal,
}

impl DrawCommand {
    /// Builds a new command that draws.
    ///
    /// # Panic
    ///
    /// - Panicks if some of the parameters were not created with the same device.
    ///
    pub fn new<'a, I, Ib, Vb, S, P, V, Pl, Rp>
              (ty: DrawTy<'a, I, Ib>, pipeline: &Arc<GraphicsPipeline<V, Pl, Rp>>,
               dynamic: &DynamicState, vertex_buffers: Vb,
               sets: S, push_constants: &P) -> Result<DrawCommand, DrawError>
        where Pl: PipelineLayout + PipelineLayoutSetsCompatible<S> +
                  PipelineLayoutPushConstantsCompatible<P>,
              S: DescriptorSetsCollection,
              P: Copy,
              V: Source<Vb>,
              Rp: Send + Sync + 'static,
              I: Index + 'static,
              Ib: Buffer,
    {
        let (vertex_buffers, vertex_count, instance_count) = {
            pipeline.vertex_definition().decode(vertex_buffers)
        };

        // TODO: doesn't detect that the vertex and index buffers have the same device as the rest

        let (internal_ty, bind_index_buffer) = match ty {
            DrawTy::Simple => {
                let ty = DrawTyInternal::CmdDraw {
                    vertex_count: vertex_count as u32,
                    instance_count: instance_count as u32,
                    first_vertex: 0,        // TODO: allow to choose
                    first_instance: 0,      // TODO: allow to choose
                };

                (ty, None)
            },
            DrawTy::Indexed { index_buffer, first_index } => {
                // TODO: correct error instead
                assert!(first_index < index_buffer.len() as u32);

                let ty = DrawTyInternal::CmdDrawIndexed {
                    index_count: index_buffer.len() as u32 - first_index,
                    instance_count: 1,      // TODO: allow to choose
                    first_index: first_index,
                    vertex_offset: 0,       // TODO: allow to choose
                    first_instance: 0,      // TODO: allow to choose
                };
                
                let bind = try!(IndexBufferBindCommand::new(index_buffer));

                (ty, Some(bind))
            },
        };

        Ok(DrawCommand {
            bind_pipeline: try!({
                GraphicsPipelineBindCommand::new(pipeline, dynamic)
            }),
            bind_vertex_buffer: try!({
                VertexSourceBindCommand::new(vertex_buffers.map(|v| (v, 0)))        // FIXME: offset
            }),
            bind_index_buffer: bind_index_buffer,
            bind_descriptor_sets: try!({
                let pipeline_layout = pipeline.layout();
                DescriptorSetsBindCommand::new(true, &**pipeline_layout, sets, push_constants)
            }),
            ty: internal_ty,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the various parameters were not created with the same device as the
    ///   command buffer.
    /// - Panicks if the command buffer is not inside a render pass.
    ///
    /// # Safety
    ///
    /// - Doesn't check framebuffer and render pass compatibility.
    ///
    pub unsafe fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                            -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        // Various checks.
        assert!(cb.within_render_pass);
        debug_assert!(cb.pool().queue_family().supports_graphics());

        // Binding stuff.
        cb = self.bind_pipeline.submit(cb);
        cb = self.bind_vertex_buffer.submit(cb);
        cb = if let Some(ref mut b) = self.bind_index_buffer { b.submit(cb) } else { cb };
        cb = self.bind_descriptor_sets.submit(cb);

        // Drawing.
        {
            let _pool_lock = cb.pool().lock();
            let vk = cb.device.pointers();
            let cmd = cb.cmd.clone().unwrap();

            match self.ty {
                DrawTyInternal::CmdDraw { vertex_count, instance_count, first_vertex,
                                          first_instance } =>
                {
                    vk.CmdDraw(cmd, vertex_count, instance_count, first_vertex,
                               first_instance);
                },
                DrawTyInternal::CmdDrawIndexed { index_count, instance_count, first_index,
                                                 vertex_offset, first_instance } =>
                {
                    vk.CmdDrawIndexed(cmd, index_count, instance_count, first_index,
                                      vertex_offset, first_instance);
                },

                DrawTyInternal::CmdDrawIndirect { buffer, offset, draw_count, stride } =>
                {
                    vk.CmdDrawIndirect(cmd, buffer, offset, draw_count, stride);
                },

                DrawTyInternal::CmdDrawIndexedIndirect { buffer, offset, draw_count,
                                                         stride } =>
                {
                    vk.CmdDrawIndexedIndirect(cmd, buffer, offset, draw_count, stride);
                },
            }
        }

        cb
    }
}

pub enum DrawTy<'a, I, Ib> where I: Index, Ib: Buffer {
    Simple,
    Indexed { index_buffer: BufferSlice<'a, [I], Ib>, first_index: u32 },
    // TODO: indirect and indexed-indirect
}

enum DrawTyInternal {
    CmdDraw { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 },
    CmdDrawIndexed { index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32,
                     first_instance: u32 },
    CmdDrawIndirect { buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32 },
    CmdDrawIndexedIndirect { buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32,
                             stride: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawError {
    GfxPipelineBindError(GfxPipelineBindError),
    DescriptorSetsBindError(DescriptorSetsBindError),
    VertexBufferBindError(VertexBufferBindError),
    IndexBufferBindError(IndexBufferBindError),
}

impl error::Error for DrawError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DrawError::GfxPipelineBindError(_) => {
                "error when binding the graphics pipeline"
            },
            DrawError::DescriptorSetsBindError(_) => {
                "error when binding the descriptor sets and push constants"
            },
            DrawError::VertexBufferBindError(_) => {
                "error when binding the vertex source"
            },
            DrawError::IndexBufferBindError(_) => {
                "error when binding the index source"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            DrawError::GfxPipelineBindError(ref err) => Some(err),
            DrawError::DescriptorSetsBindError(ref err) => Some(err),
            DrawError::VertexBufferBindError(ref err) => Some(err),
            DrawError::IndexBufferBindError(ref err) => Some(err),
        }
    }
}

impl From<GfxPipelineBindError> for DrawError {
    #[inline]
    fn from(err: GfxPipelineBindError) -> DrawError {
        DrawError::GfxPipelineBindError(err)
    }
}

impl From<DescriptorSetsBindError> for DrawError {
    #[inline]
    fn from(err: DescriptorSetsBindError) -> DrawError {
        DrawError::DescriptorSetsBindError(err)
    }
}

impl From<VertexBufferBindError> for DrawError {
    #[inline]
    fn from(err: VertexBufferBindError) -> DrawError {
        DrawError::VertexBufferBindError(err)
    }
}

impl From<IndexBufferBindError> for DrawError {
    #[inline]
    fn from(err: IndexBufferBindError) -> DrawError {
        DrawError::IndexBufferBindError(err)
    }
}

impl fmt::Display for DrawError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
