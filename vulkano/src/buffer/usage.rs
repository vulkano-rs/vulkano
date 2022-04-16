// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;

/// Describes how a buffer is going to be used. This is **not** just an optimization.
///
/// If you try to use a buffer in a way that you didn't declare, a panic will happen.
///
/// Some methods are provided to build `BufferUsage` structs for some common situations. However
/// there is no restriction in the combination of BufferUsages that can be enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferUsage {
    pub transfer_src: bool,
    pub transfer_dst: bool,
    pub uniform_texel_buffer: bool,
    pub storage_texel_buffer: bool,
    pub uniform_buffer: bool,
    pub storage_buffer: bool,
    pub index_buffer: bool,
    pub vertex_buffer: bool,
    pub indirect_buffer: bool,
    pub device_address: bool,
    pub _ne: crate::NonExhaustive,
}

impl BufferUsage {
    /// Builds a `BufferUsage` with all values set to false.
    #[inline]
    pub const fn none() -> BufferUsage {
        BufferUsage {
            transfer_src: false,
            transfer_dst: false,
            uniform_texel_buffer: false,
            storage_texel_buffer: false,
            uniform_buffer: false,
            storage_buffer: false,
            index_buffer: false,
            vertex_buffer: false,
            indirect_buffer: false,
            device_address: false,
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Builds a `BufferUsage` with all values set to true. Can be used for quick prototyping.
    #[inline]
    pub const fn all() -> BufferUsage {
        BufferUsage {
            transfer_src: true,
            transfer_dst: true,
            uniform_texel_buffer: true,
            storage_texel_buffer: true,
            uniform_buffer: true,
            storage_buffer: true,
            index_buffer: true,
            vertex_buffer: true,
            indirect_buffer: true,
            device_address: true,
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Builds a `BufferUsage` with `transfer_src` set to true and the rest to false.
    #[inline]
    pub const fn transfer_src() -> BufferUsage {
        BufferUsage {
            transfer_src: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `transfer_dst` set to true and the rest to false.
    #[inline]
    pub const fn transfer_dst() -> BufferUsage {
        BufferUsage {
            transfer_dst: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `vertex_buffer` set to true and the rest to false.
    #[inline]
    pub const fn vertex_buffer() -> BufferUsage {
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `vertex_buffer` and `transfer_dst` set to true and the rest
    /// to false.
    #[inline]
    pub const fn vertex_buffer_transfer_dst() -> BufferUsage {
        BufferUsage {
            vertex_buffer: true,
            transfer_dst: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `index_buffer` set to true and the rest to false.
    #[inline]
    pub const fn index_buffer() -> BufferUsage {
        BufferUsage {
            index_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `index_buffer` and `transfer_dst` set to true and the rest to false.
    #[inline]
    pub const fn index_buffer_transfer_dst() -> BufferUsage {
        BufferUsage {
            index_buffer: true,
            transfer_dst: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `uniform_buffer` set to true and the rest to false.
    #[inline]
    pub const fn uniform_buffer() -> BufferUsage {
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with only `storage_buffer` set, while rest are not
    #[inline]
    pub const fn storage_buffer() -> BufferUsage {
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `uniform_buffer` and `transfer_dst` set to true and the rest
    /// to false.
    #[inline]
    pub const fn uniform_buffer_transfer_dst() -> BufferUsage {
        BufferUsage {
            uniform_buffer: true,
            transfer_dst: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `indirect_buffer` set to true and the rest to false.
    #[inline]
    pub const fn indirect_buffer() -> BufferUsage {
        BufferUsage {
            indirect_buffer: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `indirect_buffer` and `transfer_dst` set to true and the rest
    /// to false.
    #[inline]
    pub const fn indirect_buffer_transfer_dst() -> BufferUsage {
        BufferUsage {
            indirect_buffer: true,
            transfer_dst: true,
            ..BufferUsage::none()
        }
    }

    /// Builds a `BufferUsage` with `device_address` set to true and the rest to false.
    #[inline]
    pub const fn device_address() -> BufferUsage {
        BufferUsage {
            device_address: true,
            ..BufferUsage::none()
        }
    }
}

impl From<BufferUsage> for ash::vk::BufferUsageFlags {
    fn from(val: BufferUsage) -> Self {
        let mut result = ash::vk::BufferUsageFlags::empty();
        if val.transfer_src {
            result |= ash::vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if val.transfer_dst {
            result |= ash::vk::BufferUsageFlags::TRANSFER_DST;
        }
        if val.uniform_texel_buffer {
            result |= ash::vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
        }
        if val.storage_texel_buffer {
            result |= ash::vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
        }
        if val.uniform_buffer {
            result |= ash::vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if val.storage_buffer {
            result |= ash::vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if val.index_buffer {
            result |= ash::vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if val.vertex_buffer {
            result |= ash::vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if val.indirect_buffer {
            result |= ash::vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        if val.device_address {
            result |= ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        result
    }
}

impl BitOr for BufferUsage {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        BufferUsage {
            transfer_src: self.transfer_src || rhs.transfer_src,
            transfer_dst: self.transfer_dst || rhs.transfer_dst,
            uniform_texel_buffer: self.uniform_texel_buffer || rhs.uniform_texel_buffer,
            storage_texel_buffer: self.storage_texel_buffer || rhs.storage_texel_buffer,
            uniform_buffer: self.uniform_buffer || rhs.uniform_buffer,
            storage_buffer: self.storage_buffer || rhs.storage_buffer,
            index_buffer: self.index_buffer || rhs.index_buffer,
            vertex_buffer: self.vertex_buffer || rhs.vertex_buffer,
            indirect_buffer: self.indirect_buffer || rhs.indirect_buffer,
            device_address: self.device_address || rhs.device_address,
            _ne: crate::NonExhaustive(()),
        }
    }
}
