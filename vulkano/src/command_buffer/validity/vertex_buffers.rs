// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::SyncCommandBufferBuilder;
use crate::pipeline::vertex::VertexInputRate;
use crate::pipeline::GraphicsPipeline;
use crate::DeviceSize;
use std::convert::TryInto;
use std::error;
use std::fmt;

pub(in super::super) fn check_vertex_buffers(
    builder: &SyncCommandBufferBuilder,
    pipeline: &GraphicsPipeline,
    vertices: Option<(u32, u32)>,
    instances: Option<(u32, u32)>,
) -> Result<(), CheckVertexBufferError> {
    let vertex_input = pipeline.vertex_input();
    let mut max_vertex_count: Option<u32> = None;
    let mut max_instance_count: Option<u32> = None;

    for (binding_num, binding_desc) in vertex_input.bindings() {
        let vertex_buffer = match builder.bound_vertex_buffer(binding_num) {
            Some(x) => x,
            None => return Err(CheckVertexBufferError::BufferNotBound { binding_num }),
        };

        let mut num_elements = (vertex_buffer.size() / binding_desc.stride as DeviceSize)
            .try_into()
            .unwrap_or(u32::MAX);

        match binding_desc.input_rate {
            VertexInputRate::Vertex => {
                max_vertex_count = Some(if let Some(x) = max_vertex_count {
                    x.min(num_elements)
                } else {
                    num_elements
                });
            }
            VertexInputRate::Instance { divisor } => {
                if divisor == 0 {
                    // A divisor of 0 means the same instance data is used for all instances,
                    // so we can draw any number of instances from a single element.
                    // The buffer must contain at least one element though.
                    if num_elements != 0 {
                        num_elements = u32::MAX;
                    }
                } else {
                    // If divisor is 2, we use only half the amount of data from the source buffer,
                    // so the number of instances that can be drawn is twice as large.
                    num_elements = num_elements.saturating_mul(divisor);
                }

                max_instance_count = Some(if let Some(x) = max_instance_count {
                    x.min(num_elements)
                } else {
                    num_elements
                });
            }
        };
    }

    if let Some((first_vertex, vertex_count)) = vertices {
        if let Some(max_vertex_count) = max_vertex_count {
            if first_vertex + vertex_count > max_vertex_count {
                return Err(CheckVertexBufferError::TooManyVertices {
                    vertex_count,
                    max_vertex_count,
                });
            }
        }
    }

    if let Some((first_instance, instance_count)) = instances {
        if let Some(max_instance_count) = max_instance_count {
            if first_instance + instance_count > max_instance_count {
                return Err(CheckVertexBufferError::TooManyInstances {
                    instance_count,
                    max_instance_count,
                }
                .into());
            }
        }

        if pipeline
            .subpass()
            .render_pass()
            .desc()
            .multiview()
            .is_some()
        {
            let max_instance_index = pipeline
                .device()
                .physical_device()
                .properties()
                .max_multiview_instance_index
                .unwrap_or(0);

            if first_instance + instance_count > max_instance_index + 1 {
                return Err(CheckVertexBufferError::TooManyInstances {
                    instance_count,
                    max_instance_count: max_instance_index + 1, // TODO: this can overflow
                }
                .into());
            }
        }
    }

    Ok(())
}

/// Error that can happen when checking whether the vertex buffers are valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckVertexBufferError {
    /// No buffer was bound to a binding slot needed by the pipeline.
    BufferNotBound { binding_num: u32 },

    /// The "vertex buffer" usage must be enabled on the buffer.
    BufferMissingUsage {
        /// Index of the buffer that is missing usage.
        binding_num: u32,
    },

    /// A draw command requested too many vertices.
    TooManyVertices {
        /// The used amount of vertices.
        vertex_count: u32,
        /// The allowed amount of vertices.
        max_vertex_count: u32,
    },

    /// A draw command requested too many instances.
    ///
    /// When the `multiview` feature is used the maximum amount of instances may be reduced
    /// because the implementation may use instancing internally to implement `multiview`.
    TooManyInstances {
        /// The used amount of instances.
        instance_count: u32,
        /// The allowed amount of instances.
        max_instance_count: u32,
    },
}

impl error::Error for CheckVertexBufferError {}

impl fmt::Display for CheckVertexBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckVertexBufferError::BufferNotBound { .. } => {
                    "no buffer was bound to a binding slot needed by the pipeline"
                }
                CheckVertexBufferError::BufferMissingUsage { .. } => {
                    "the vertex buffer usage is missing on a vertex buffer"
                }
                CheckVertexBufferError::TooManyVertices { .. } => {
                    "the draw command requested too many vertices"
                }
                CheckVertexBufferError::TooManyInstances { .. } => {
                    "the draw command requested too many instances"
                }
            }
        )
    }
}
