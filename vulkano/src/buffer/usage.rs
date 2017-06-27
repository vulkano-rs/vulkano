// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;
use vk;

/// Describes how a buffer is going to be used. This is **not** an optimization.
///
/// If you try to use a buffer in a way that you didn't declare, a panic will happen.
///
/// Some methods are provided to build `BufferUsage` structs for some common situations. However
/// there is no restriction in the combination of BufferUsages that can be enabled.
#[derive(Debug, Copy, Clone)]
pub struct BufferUsage {
    pub transfer_source: bool,
    pub transfer_dest: bool,
    pub uniform_texel_buffer: bool,
    pub storage_texel_buffer: bool,
    pub uniform_buffer: bool,
    pub storage_buffer: bool,
    pub index_buffer: bool,
    pub vertex_buffer: bool,
    pub indirect_buffer: bool,
}

impl BufferUsage {
    /// Builds a `BufferUsage` with all values set to false.
    #[inline]
    pub fn none() -> BufferUsage {
        BufferUsage {
            transfer_source: false,
            transfer_dest: false,
            uniform_texel_buffer: false,
            storage_texel_buffer: false,
            uniform_buffer: false,
            storage_buffer: false,
            index_buffer: false,
            vertex_buffer: false,
            indirect_buffer: false,
        }
    }

    /// Builds a `BufferUsage` with all values set to true. Can be used for quick prototyping.
    #[inline]
    pub fn all() -> BufferUsage {
        BufferUsage {
            transfer_source: true,
            transfer_dest: true,
            uniform_texel_buffer: true,
            storage_texel_buffer: true,
            uniform_buffer: true,
            storage_buffer: true,
            index_buffer: true,
            vertex_buffer: true,
            indirect_buffer: true,
        }
    }

    /// Builds a `BufferUsage` with `transfer_source` set to true and the rest to false.
    #[inline]
    pub fn transfer_source() -> BufferUsage {
        BufferUsage {
            transfer_source: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `transfer_dest` set to true and the rest to false.
    #[inline]
    pub fn transfer_dest() -> BufferUsage {
        BufferUsage {
            transfer_dest: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `vertex_buffer` set to true and the rest to false.
    #[inline]
    pub fn vertex_buffer() -> BufferUsage {
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `vertex_buffer` and `transfer_dest` set to true and the rest
    /// to false.
    #[inline]
    pub fn vertex_buffer_transfer_dest() -> BufferUsage {
        BufferUsage {
            vertex_buffer: true,
            transfer_dest: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `index_buffer` set to true and the rest to false.
    #[inline]
    pub fn index_buffer() -> BufferUsage {
        BufferUsage {
            index_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `index_buffer` and `transfer_dest` set to true and the rest to false.
    #[inline]
    pub fn index_buffer_transfer_dest() -> BufferUsage {
        BufferUsage {
            index_buffer: true,
            transfer_dest: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `uniform_buffer` set to true and the rest to false.
    #[inline]
    pub fn uniform_buffer() -> BufferUsage {
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `uniform_buffer` and `transfer_dest` set to true and the rest
    /// to false.
    #[inline]
    pub fn uniform_buffer_transfer_dest() -> BufferUsage {
        BufferUsage {
            uniform_buffer: true,
            transfer_dest: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `indirect_buffer` set to true and the rest to false.
    #[inline]
    pub fn indirect_buffer() -> BufferUsage {
        BufferUsage {
            indirect_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `indirect_buffer` and `transfer_dest` set to true and the rest
    /// to false.
    #[inline]
    pub fn indirect_buffer_transfer_dest() -> BufferUsage {
        BufferUsage {
            indirect_buffer: true,
            transfer_dest: true,
            ..BufferUsage::none()
        }
    }
}

impl BitOr for BufferUsage {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        BufferUsage {
            transfer_source: self.transfer_source || rhs.transfer_source,
            transfer_dest: self.transfer_dest || rhs.transfer_dest,
            uniform_texel_buffer: self.uniform_texel_buffer || rhs.uniform_texel_buffer,
            storage_texel_buffer: self.storage_texel_buffer || rhs.storage_texel_buffer,
            uniform_buffer: self.uniform_buffer || rhs.uniform_buffer,
            storage_buffer: self.storage_buffer || rhs.storage_buffer,
            index_buffer: self.index_buffer || rhs.index_buffer,
            vertex_buffer: self.vertex_buffer || rhs.vertex_buffer,
            indirect_buffer: self.indirect_buffer || rhs.indirect_buffer,
        }
    }
}

/// Turns a `BufferUsage` into raw bits.
#[inline]
pub fn usage_to_bits(usage: BufferUsage) -> vk::BufferUsageFlagBits {
    let mut result = 0;
    if usage.transfer_source {
        result |= vk::BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if usage.transfer_dest {
        result |= vk::BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if usage.uniform_texel_buffer {
        result |= vk::BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
    }
    if usage.storage_texel_buffer {
        result |= vk::BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    }
    if usage.uniform_buffer {
        result |= vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
    if usage.storage_buffer {
        result |= vk::BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if usage.index_buffer {
        result |= vk::BUFFER_USAGE_INDEX_BUFFER_BIT;
    }
    if usage.vertex_buffer {
        result |= vk::BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }
    if usage.indirect_buffer {
        result |= vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }
    result
}
