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

use command_buffer::DrawIndexedIndirectCommand;
use command_buffer::DrawIndirectCommand;

use super::CheckVertexBuffer;
use super::CheckIndexBuffer;

/// Checks whether subset is a valid subset of a vertex buffer.
pub fn check_subset_validity(vb_infos: &CheckVertexBuffer, subset: &DrawIndirectCommand)
                             -> Result<(), CheckSubsetError>
{
    if vb_infos.vertex_count <= subset.first_vertex
        || vb_infos.vertex_count < (subset.first_vertex + subset.vertex_count) {
        return Err(CheckSubsetError::VerticesOutOfRange);
    }
    if vb_infos.instance_count <= subset.first_instance
        || vb_infos.instance_count < (subset.first_instance + subset.instance_count) {
        return Err(CheckSubsetError::InstancesOutOfRange);
    }
    Ok(())
}

/// Checks whether subset is a valid subset of a vertex-index buffers pair.
pub fn check_indexed_subset_validity(vb_infos: &CheckVertexBuffer,
                                     ib_infos: &CheckIndexBuffer,
                                     subset: &DrawIndexedIndirectCommand)
                                     -> Result<(), CheckIndexedSubsetError>
{
    // TODO: Possible to validate vertex_offset ?

    if vb_infos.instance_count <= subset.first_instance
        || vb_infos.instance_count < (subset.first_instance + subset.instance_count) {
        return Err(CheckIndexedSubsetError::InstancesOutOfRange);
    }
    if ib_infos.num_indices <= subset.first_index as usize
        || ib_infos.num_indices < (subset.first_index as usize + subset.index_count as usize) {
        return Err(CheckIndexedSubsetError::IndexOutOfRange);
    }
    Ok(())
}

/// Error that can happen when checking whether the subset is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckSubsetError {
    /// Vertex range is not a valid subset of vertex buffer.
    VerticesOutOfRange,
    /// Instance range is not a valid subset of instance buffer.
    InstancesOutOfRange,
}

/// Error that can happen when checking whether the indexed subset is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckIndexedSubsetError {
    /// Instance range is not a valid subset of instance buffer.
    InstancesOutOfRange,
    /// Index range is not a valid subset of index buffer.
    IndexOutOfRange,
}

impl error::Error for CheckSubsetError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckSubsetError::VerticesOutOfRange => {
                "vertex range is not a valid subset of vertex buffer"
            },
            CheckSubsetError::InstancesOutOfRange => {
                "instance range is not a valid subset of instance buffer"
            },
        }
    }
}

impl fmt::Display for CheckSubsetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl error::Error for CheckIndexedSubsetError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CheckIndexedSubsetError::IndexOutOfRange => {
                "index range is not a valid subset of index buffer"
            },
            CheckIndexedSubsetError::InstancesOutOfRange => {
                "instance range is not a valid subset of instance buffer"
            },
        }
    }
}

impl fmt::Display for CheckIndexedSubsetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
