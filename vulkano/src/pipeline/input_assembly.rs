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

use crate::DeviceSize;

/// The state in a graphics pipeline describing how the input assembly stage should behave.
#[derive(Clone, Copy, Debug)]
pub struct InputAssemblyState {
    /// The type of primitives.
    ///
    /// Note that some topologies require a feature to be enabled on the device.
    ///
    /// If set to `None`, then this state will be considered as dynamic and the value will
    /// need to be set when building a command buffer. This requires the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature to be
    /// enabled on the device.
    pub topology: Option<PrimitiveTopology>,

    /// If true, then when drawing with an index buffer, the special index value consisting of the
    /// maximum unsigned value (`0xff`, `0xffff` or `0xffffffff`) will tell the GPU that it is the
    /// end of the current primitive. A new primitive will restart at the next index.
    ///
    /// Primitive restart is mostly useful in combination with "strip" and "fan" topologies. "List"
    /// topologies require a feature to be enabled on the device when combined with primitive
    /// restart.
    ///
    /// If set to `None`, then this state will be considered as dynamic and the value will
    /// need to be set when building a command buffer. This requires the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature to be
    /// enabled on the device.
    pub primitive_restart_enable: Option<bool>,
}

impl InputAssemblyState {
    /// Builds an `InputAssembly` struct with the `TriangleList` topology and primitive restart
    /// disabled.
    #[inline]
    pub fn triangle_list() -> InputAssemblyState {
        InputAssemblyState {
            topology: Some(PrimitiveTopology::TriangleList),
            primitive_restart_enable: Some(false),
        }
    }
}

impl Default for InputAssemblyState {
    fn default() -> Self {
        Self::triangle_list()
    }
}

/// Describes how vertices must be grouped together to form primitives.
///
/// When enabling primitive restart, "list" topologies require a feature to be enabled on the
/// device:
/// - The `PatchList` topology requires the
///   [`primitive_topology_patch_list_restart`](crate::device::Features::primitive_topology_patch_list_restart)
///   feature.
/// - All other "list" topologies require the
///   [`primitive_topology_list_restart`](crate::device::Features::primitive_topology_list_restart)
///   feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum PrimitiveTopology {
    /// A series of separate point primitives.
    PointList = ash::vk::PrimitiveTopology::POINT_LIST.as_raw(),
    /// A series of separate line primitives.
    LineList = ash::vk::PrimitiveTopology::LINE_LIST.as_raw(),
    /// A series of consecutive line primitives, with consecutive lines sharing a vertex.
    LineStrip = ash::vk::PrimitiveTopology::LINE_STRIP.as_raw(),
    /// A series of separate triangle primitives.
    TriangleList = ash::vk::PrimitiveTopology::TRIANGLE_LIST.as_raw(),
    /// A series of consecutive triangle primitives, with consecutive triangles sharing an edge (two vertices).
    TriangleStrip = ash::vk::PrimitiveTopology::TRIANGLE_STRIP.as_raw(),
    /// A series of consecutive triangle primitives, with all triangles sharing a common vertex (the first).
    TriangleFan = ash::vk::PrimitiveTopology::TRIANGLE_FAN.as_raw(),
    /// As `LineList, but with adjacency, used in combination with geometry shaders. Requires the
    /// [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    LineListWithAdjacency = ash::vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY.as_raw(),
    /// As `LineStrip`, but with adjacency, used in combination with geometry shaders. Requires the
    /// [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    LineStripWithAdjacency = ash::vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY.as_raw(),
    /// As `TriangleList`, but with adjacency, used in combination with geometry shaders. Requires
    /// the [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    TriangleListWithAdjacency = ash::vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY.as_raw(),
    /// As `TriangleStrip`, but with adjacency, used in combination with geometry shaders. Requires
    /// the [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    TriangleStripWithAdjacency = ash::vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY.as_raw(),
    /// Separate patch primitives, used in combination with tessellation shaders. Requires the
    /// [`tessellation_shader`](crate::device::Features::tessellation_shader) feature.
    PatchList = ash::vk::PrimitiveTopology::PATCH_LIST.as_raw(),
}

// TODO: use the #[default] attribute once it's stable.
// See: https://github.com/rust-lang/rust/issues/87517
impl Default for PrimitiveTopology {
    fn default() -> Self {
        PrimitiveTopology::TriangleList
    }
}

impl From<PrimitiveTopology> for ash::vk::PrimitiveTopology {
    #[inline]
    fn from(val: PrimitiveTopology) -> ash::vk::PrimitiveTopology {
        Self::from_raw(val as i32)
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

unsafe impl Index for u8 {
    #[inline(always)]
    fn ty() -> IndexType {
        IndexType::U8
    }
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
    U8 = ash::vk::IndexType::UINT8_EXT.as_raw(),
    U16 = ash::vk::IndexType::UINT16.as_raw(),
    U32 = ash::vk::IndexType::UINT32.as_raw(),
}

impl IndexType {
    /// Returns the size in bytes of indices of this type.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        match self {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        }
    }
}

impl From<IndexType> for ash::vk::IndexType {
    #[inline]
    fn from(val: IndexType) -> Self {
        Self::from_raw(val as i32)
    }
}
