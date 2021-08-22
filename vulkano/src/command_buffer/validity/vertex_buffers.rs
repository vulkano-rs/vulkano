// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::device::DeviceOwned;
use crate::pipeline::GraphicsPipeline;
use crate::VulkanObject;
use std::error;
use std::fmt;

/// Checks whether vertex buffers can be bound.
///
/// # Panic
///
/// - Panics if one of the vertex buffers was not created with the same device as `pipeline`.
///
pub fn check_vertex_buffers(
    pipeline: &GraphicsPipeline,
    vertex_buffers: &[Box<dyn BufferAccess + Send + Sync>],
) -> Result<(), CheckVertexBufferError> {
    for (num, buf) in vertex_buffers.iter().enumerate() {
        assert_eq!(
            buf.inner().buffer.device().internal_object(),
            pipeline.device().internal_object()
        );

        if !buf.inner().buffer.usage().vertex_buffer {
            return Err(CheckVertexBufferError::BufferMissingUsage { num_buffer: num });
        }
    }

    Ok(())
}

/// Error that can happen when checking whether the vertex buffers are valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckVertexBufferError {
    /// The "vertex buffer" usage must be enabled on the buffer.
    BufferMissingUsage {
        /// Index of the buffer that is missing usage.
        num_buffer: usize,
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

    /// A draw command requested too many indices.
    TooManyIndices {
        /// The used amount of indices.
        index_count: u32,
        /// The allowed amount of indices.
        max_index_count: u32,
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
                CheckVertexBufferError::BufferMissingUsage { .. } => {
                    "the vertex buffer usage is missing on a vertex buffer"
                }
                CheckVertexBufferError::TooManyVertices { .. } => {
                    "the draw command requested too many vertices"
                }
                CheckVertexBufferError::TooManyInstances { .. } => {
                    "the draw command requested too many instances"
                }
                CheckVertexBufferError::TooManyIndices { .. } => {
                    "the draw command requested too many indices"
                }
            }
        )
    }
}
