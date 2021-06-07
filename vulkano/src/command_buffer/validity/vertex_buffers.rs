// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use crate::buffer::BufferAccess;
use crate::device::DeviceOwned;
use crate::pipeline::vertex::VertexSource;
use crate::pipeline::GraphicsPipelineAbstract;
use crate::VulkanObject;

/// Checks whether vertex buffers can be bound.
///
/// # Panic
///
/// - Panics if one of the vertex buffers was not created with the same device as `pipeline`.
///
pub fn check_vertex_buffers<GP, V>(
    pipeline: &GP,
    vertex_buffers: V,
) -> Result<CheckVertexBuffer, CheckVertexBufferError>
where
    GP: GraphicsPipelineAbstract + DeviceOwned + VertexSource<V>,
{
    let (vertex_buffers, vertex_count, instance_count) = pipeline.decode(vertex_buffers);

    for (num, buf) in vertex_buffers.iter().enumerate() {
        assert_eq!(
            buf.inner().buffer.device().internal_object(),
            pipeline.device().internal_object()
        );

        if !buf.inner().buffer.usage().vertex_buffer {
            return Err(CheckVertexBufferError::BufferMissingUsage { num_buffer: num });
        }
    }

    if let Some(multiview) = pipeline.subpass().render_pass().desc().multiview() {
        let max_instance_index = pipeline
            .device()
            .physical_device()
            .extended_properties()
            .max_multiview_instance_index()
            .unwrap_or(0) as usize;

        // vulkano currently always uses `0` as the first instance which means the highest
        // used index will just be `instance_count - 1`
        if instance_count > max_instance_index + 1 {
            return Err(CheckVertexBufferError::TooManyInstances {
                instance_count,
                max_instance_count: max_instance_index + 1,
            });
        }
    }

    Ok(CheckVertexBuffer {
        vertex_buffers,
        vertex_count: vertex_count as u32,
        instance_count: instance_count as u32,
    })
}

/// Information returned if `check_vertex_buffer` succeeds.
pub struct CheckVertexBuffer {
    /// The list of vertex buffers.
    pub vertex_buffers: Vec<Box<dyn BufferAccess + Send + Sync>>,
    /// Number of vertices available in the intersection of the buffers.
    pub vertex_count: u32,
    /// Number of instances available in the intersection of the buffers.
    pub instance_count: u32,
}

/// Error that can happen when checking whether the vertex buffers are valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckVertexBufferError {
    /// The "vertex buffer" usage must be enabled on the buffer.
    BufferMissingUsage {
        /// Index of the buffer that is missing usage.
        num_buffer: usize,
    },

    /// The vertex buffer has too many instances.
    /// When the `multiview` feature is used the maximum amount of instances may be reduced
    /// because the implementation may use instancing internally to implement `multiview`.
    TooManyInstances {
        /// The used amount of instances.
        instance_count: usize,
        /// The allowed amount of instances.
        max_instance_count: usize,
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
                CheckVertexBufferError::TooManyInstances { .. } => {
                    "the vertex buffer has too many instances"
                }
            }
        )
    }
}
