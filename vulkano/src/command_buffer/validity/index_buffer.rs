// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::synced::CommandBufferState;
use std::error;
use std::fmt;

/// Checks whether an index buffer can be bound.
///
/// # Panic
///
/// - Panics if the buffer was not created with `device`.
///
pub(in super::super) fn check_index_buffer(
    current_state: CommandBufferState,
    indices: Option<(u32, u32)>,
) -> Result<(), CheckIndexBufferError> {
    let (index_buffer, index_type) = match current_state.index_buffer() {
        Some(x) => x,
        None => return Err(CheckIndexBufferError::BufferNotBound),
    };

    if let Some((first_index, index_count)) = indices {
        let max_index_count = (index_buffer.size() / index_type.size()) as u32;

        if first_index + index_count > max_index_count {
            return Err(CheckIndexBufferError::TooManyIndices {
                index_count,
                max_index_count,
            }
            .into());
        }
    }

    Ok(())
}

/// Error that can happen when checking whether binding an index buffer is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckIndexBufferError {
    /// No index buffer was bound.
    BufferNotBound,
    /// A draw command requested too many indices.
    TooManyIndices {
        /// The used amount of indices.
        index_count: u32,
        /// The allowed amount of indices.
        max_index_count: u32,
    },
    /// The "index buffer" usage must be enabled on the index buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
    /// The type of the indices is not supported by the device.
    UnsupportIndexType,
}

impl error::Error for CheckIndexBufferError {}

impl fmt::Display for CheckIndexBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckIndexBufferError::BufferNotBound => {
                    "no index buffer was bound"
                }
                CheckIndexBufferError::TooManyIndices { .. } => {
                    "the draw command requested too many indices"
                }
                CheckIndexBufferError::BufferMissingUsage => {
                    "the index buffer usage must be enabled on the index buffer"
                }
                CheckIndexBufferError::WrongAlignment => {
                    "the sum of offset and the address of the range of VkDeviceMemory object that is \
                 backing buffer, must be a multiple of the type indicated by indexType"
                }
                CheckIndexBufferError::UnsupportIndexType => {
                    "the type of the indices is not supported by the device"
                }
            }
        )
    }
}
