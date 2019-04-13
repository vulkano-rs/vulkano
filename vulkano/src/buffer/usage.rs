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

/// Describes how a buffer is going to be used. This is **not** just an optimization.
///
/// If you try to use a buffer in a way that you didn't declare, a panic will happen.
///
/// Some methods are provided to build `BufferUsage` structs for some common situations. However
/// there is no restriction in the combination of BufferUsages that can be enabled.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BufferUsage {
    pub transfer_source: bool,
    pub transfer_destination: bool,
    pub uniform_texel_buffer: bool,
    pub storage_texel_buffer: bool,
    pub uniform_buffer: bool,
    pub storage_buffer: bool,
    pub index_buffer: bool,
    pub vertex_buffer: bool,
    pub indirect_buffer: bool,
}

impl BufferUsage {
    /// Turns this `BufferUsage` into raw Vulkan bits.
    pub(crate) fn to_vulkan_bits(&self) -> vk::BufferUsageFlagBits {
        let mut result = 0;
        if self.transfer_source {
            result |= vk::BUFFER_USAGE_TRANSFER_SRC_BIT;
        }
        if self.transfer_destination {
            result |= vk::BUFFER_USAGE_TRANSFER_DST_BIT;
        }
        if self.uniform_texel_buffer {
            result |= vk::BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
        }
        if self.storage_texel_buffer {
            result |= vk::BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
        }
        if self.uniform_buffer {
            result |= vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        }
        if self.storage_buffer {
            result |= vk::BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }
        if self.index_buffer {
            result |= vk::BUFFER_USAGE_INDEX_BUFFER_BIT;
        }
        if self.vertex_buffer {
            result |= vk::BUFFER_USAGE_VERTEX_BUFFER_BIT;
        }
        if self.indirect_buffer {
            result |= vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }
        result
    }

    /// Builds a `BufferUsage` with all values set to false.
    #[inline]
    pub fn none() -> BufferUsage {
        BufferUsage {
            transfer_source: false,
            transfer_destination: false,
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
            transfer_destination: true,
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

    /// Builds a `BufferUsage` with `transfer_destination` set to true and the rest to false.
    #[inline]
    pub fn transfer_destination() -> BufferUsage {
        BufferUsage {
            transfer_destination: true,
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

    /// Builds a `BufferUsage` with `vertex_buffer` and `transfer_destination` set to true and the rest
    /// to false.
    #[inline]
    pub fn vertex_buffer_transfer_destination() -> BufferUsage {
        BufferUsage {
            vertex_buffer: true,
            transfer_destination: true,
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

    /// Builds a `BufferUsage` with `index_buffer` and `transfer_destination` set to true and the rest to false.
    #[inline]
    pub fn index_buffer_transfer_destination() -> BufferUsage {
        BufferUsage {
            index_buffer: true,
            transfer_destination: true,
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

    /// Builds a `BufferUsage` with `uniform_buffer` and `transfer_destination` set to true and the rest
    /// to false.
    #[inline]
    pub fn uniform_buffer_transfer_destination() -> BufferUsage {
        BufferUsage {
            uniform_buffer: true,
            transfer_destination: true,
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

    /// Builds a `BufferUsage` with `indirect_buffer` and `transfer_destination` set to true and the rest
    /// to false.
    #[inline]
    pub fn indirect_buffer_transfer_destination() -> BufferUsage {
        BufferUsage {
            indirect_buffer: true,
            transfer_destination: true,
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
            transfer_destination: self.transfer_destination || rhs.transfer_destination,
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
