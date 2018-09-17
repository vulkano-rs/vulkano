// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Assembling vertices into primitives.
//!
//! The input assembly is the stage where lists of vertices are turned into primitives.
//!

use vk;

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

impl Into<vk::PrimitiveTopology> for PrimitiveTopology {
    #[inline]
    fn into(self) -> vk::PrimitiveTopology {
        match self {
            PrimitiveTopology::PointList => vk::PRIMITIVE_TOPOLOGY_POINT_LIST,
            PrimitiveTopology::LineList => vk::PRIMITIVE_TOPOLOGY_LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PRIMITIVE_TOPOLOGY_LINE_STRIP,
            PrimitiveTopology::TriangleList => vk::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => vk::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => vk::PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
            PrimitiveTopology::LineListWithAdjacency =>
                vk::PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
            PrimitiveTopology::LineStripWithAdjacency =>
                vk::PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
            PrimitiveTopology::TriangleListWithAdjacency =>
                vk::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
            PrimitiveTopology::TriangleStripWithAdjacency =>
                vk::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
            PrimitiveTopology::PatchList { .. } => vk::PRIMITIVE_TOPOLOGY_PATCH_LIST,
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
#[repr(u32)]
pub enum IndexType {
    U16 = vk::INDEX_TYPE_UINT16,
    U32 = vk::INDEX_TYPE_UINT32,
}
