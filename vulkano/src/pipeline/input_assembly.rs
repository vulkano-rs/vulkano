// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Assembling vertices into primitives.
//!
//! The input assembly is the stage where lists of vertices are turned into primitives.
//!


/// How the input assembly stage should behave.
#[derive(Copy, Clone, Debug)]
#[deprecated]
pub struct InputAssembly {
    /// The type of primitives.
    ///
    /// Note that some topologies don't support primitive restart.
    pub topology: PrimitiveTopology,

    /// If true, then the special index value `0xffff` or `0xffffffff` will tell the GPU that it is
    /// the end of the current primitive. A new primitive will restart at the next index.
    ///
    /// Note that some topologies don't support primitive restart.
    pub primitive_restart_enable: bool,
}

impl InputAssembly {
    /// Builds an `InputAssembly` struct with the `TriangleList` topology.
    #[inline]
    pub fn triangle_list() -> InputAssembly {
        InputAssembly {
            topology: PrimitiveTopology::TriangleList,
            primitive_restart_enable: false,
        }
    }
}

/// Describes how vertices must be grouped together to form primitives.
///
/// Note that some topologies don't support primitive restart.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
    LineListWithAdjacency,
    LineStripWithAdjacency,
    TriangleListWithAdjacency,
    TriangleStripWithAdjacency,
    PatchList { vertices_per_patch: u32 },
}

impl From<PrimitiveTopology> for ash::vk::PrimitiveTopology {
    #[inline]
    fn from(val: PrimitiveTopology) -> ash::vk::PrimitiveTopology {
        match val {
            PrimitiveTopology::PointList => ash::vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => ash::vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => ash::vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleList => ash::vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => ash::vk::PrimitiveTopology::TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => ash::vk::PrimitiveTopology::TRIANGLE_FAN,
            PrimitiveTopology::LineListWithAdjacency => {
                ash::vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY
            }
            PrimitiveTopology::LineStripWithAdjacency => {
                ash::vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY
            }
            PrimitiveTopology::TriangleListWithAdjacency => {
                ash::vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY
            }
            PrimitiveTopology::TriangleStripWithAdjacency => {
                ash::vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY
            }
            PrimitiveTopology::PatchList { .. } => ash::vk::PrimitiveTopology::PATCH_LIST,
        }
    }
}

impl PrimitiveTopology {
    /// Returns true if this primitive topology supports using primitives restart.
    #[inline]
    pub fn supports_primitive_restart(&self) -> bool {
        match *self {
            PrimitiveTopology::LineStrip => true,
            PrimitiveTopology::TriangleStrip => true,
            PrimitiveTopology::TriangleFan => true,
            PrimitiveTopology::LineStripWithAdjacency => true,
            PrimitiveTopology::TriangleStripWithAdjacency => true,
            _ => false,
        }
    }
}

/// Trait for types that can be used as indices by the GPU.
pub unsafe trait Index {
    /// Returns the type of data.
    fn ty() -> IndexType;
}

unsafe impl Index for u16 {
    #[inline(always)]
    fn ty() -> IndexType {
        IndexType::U16
    }
}

unsafe impl Index for u32 {
    #[inline(always)]
    fn ty() -> IndexType {
        IndexType::U32
    }
}

/// An enumeration of all valid index types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
#[repr(i32)]
pub enum IndexType {
    U16 = ash::vk::IndexType::UINT16.as_raw(),
    U32 = ash::vk::IndexType::UINT32.as_raw(),
}

impl From<IndexType> for ash::vk::IndexType {
    #[inline]
    fn from(val: IndexType) -> Self {
        Self::from_raw(val as i32)
    }
}
