// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures how input vertices are assembled into primitives.

use crate::{
    macros::vulkan_enum,
    pipeline::{PartialStateMode, StateMode},
    DeviceSize,
};

/// The state in a graphics pipeline describing how the input assembly stage should behave.
#[derive(Clone, Copy, Debug)]
pub struct InputAssemblyState {
    /// The type of primitives.
    ///
    /// Note that some topologies require a feature to be enabled on the device.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature must be
    /// enabled on the device.
    pub topology: PartialStateMode<PrimitiveTopology, PrimitiveTopologyClass>,

    /// If true, then when drawing with an index buffer, the special index value consisting of the
    /// maximum unsigned value (`0xff`, `0xffff` or `0xffffffff`) will tell the GPU that it is the
    /// end of the current primitive. A new primitive will restart at the next index.
    ///
    /// Primitive restart is mostly useful in combination with "strip" and "fan" topologies. "List"
    /// topologies require a feature to be enabled on the device when combined with primitive
    /// restart.
    ///
    /// If set to `Dynamic`, the device API version must be at least 1.3, or the
    /// [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2) feature must
    /// be enabled on the device.
    pub primitive_restart_enable: StateMode<bool>,
}

impl InputAssemblyState {
    /// Creates an `InputAssemblyState` with the `TriangleList` topology and primitive restart
    /// disabled.
    #[inline]
    pub fn new() -> Self {
        Self {
            topology: PartialStateMode::Fixed(PrimitiveTopology::TriangleList),
            primitive_restart_enable: StateMode::Fixed(false),
        }
    }

    /// Sets the primitive topology.
    #[inline]
    pub fn topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = PartialStateMode::Fixed(topology);
        self
    }

    /// Sets the primitive topology to dynamic.
    #[inline]
    pub fn topology_dynamic(mut self, topology_class: PrimitiveTopologyClass) -> Self {
        self.topology = PartialStateMode::Dynamic(topology_class);
        self
    }

    /// Enables primitive restart.
    #[inline]
    pub fn primitive_restart_enable(mut self) -> Self {
        self.primitive_restart_enable = StateMode::Fixed(true);
        self
    }

    /// Sets primitive restart enable to dynmamic.
    #[inline]
    pub fn primitive_restart_enable_dynamic(mut self) -> Self {
        self.primitive_restart_enable = StateMode::Dynamic;
        self
    }
}

impl Default for InputAssemblyState {
    /// Returns [`InputAssemblyState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

vulkan_enum! {
    #[non_exhaustive]

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
    PrimitiveTopology impl {
        /// Returns the topology class of this topology.
        #[inline]
        pub fn class(self) -> PrimitiveTopologyClass {
            match self {
                Self::PointList => PrimitiveTopologyClass::Point,
                Self::LineList
                | Self::LineStrip
                | Self::LineListWithAdjacency
                | Self::LineStripWithAdjacency => PrimitiveTopologyClass::Line,
                Self::TriangleList
                | Self::TriangleStrip
                | Self::TriangleFan
                | Self::TriangleListWithAdjacency
                | Self::TriangleStripWithAdjacency => PrimitiveTopologyClass::Triangle,
                Self::PatchList => PrimitiveTopologyClass::Patch,
            }
        }
    }
    = PrimitiveTopology(i32);

    /// A series of separate point primitives.
    PointList = POINT_LIST,

    /// A series of separate line primitives.
    LineList = LINE_LIST,

    /// A series of consecutive line primitives, with consecutive lines sharing a vertex.
    LineStrip = LINE_STRIP,

    /// A series of separate triangle primitives.
    TriangleList = TRIANGLE_LIST,

    /// A series of consecutive triangle primitives, with consecutive triangles sharing an edge (two vertices).
    TriangleStrip = TRIANGLE_STRIP,

    /// A series of consecutive triangle primitives, with all triangles sharing a common vertex (the first).
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, the [`triangle_fans`](crate::device::Features::triangle_fans)
    /// feature must be enabled on the device.
    TriangleFan = TRIANGLE_FAN,

    /// As `LineList, but with adjacency, used in combination with geometry shaders. Requires the
    /// [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    LineListWithAdjacency = LINE_LIST_WITH_ADJACENCY,

    /// As `LineStrip`, but with adjacency, used in combination with geometry shaders. Requires the
    /// [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    LineStripWithAdjacency = LINE_STRIP_WITH_ADJACENCY,

    /// As `TriangleList`, but with adjacency, used in combination with geometry shaders. Requires
    /// the [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    TriangleListWithAdjacency = TRIANGLE_LIST_WITH_ADJACENCY,

    /// As `TriangleStrip`, but with adjacency, used in combination with geometry shaders. Requires
    /// the [`geometry_shader`](crate::device::Features::geometry_shader) feature.
    TriangleStripWithAdjacency = TRIANGLE_STRIP_WITH_ADJACENCY,

    /// Separate patch primitives, used in combination with tessellation shaders. Requires the
    /// [`tessellation_shader`](crate::device::Features::tessellation_shader) feature.
    PatchList = PATCH_LIST,

}

// TODO: use the #[default] attribute once it's stable.
// See: https://github.com/rust-lang/rust/issues/87517
impl Default for PrimitiveTopology {
    #[inline]
    fn default() -> Self {
        PrimitiveTopology::TriangleList
    }
}

/// Describes the shape of a primitive topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveTopologyClass {
    Point,
    Line,
    Triangle,
    Patch,
}

impl PrimitiveTopologyClass {
    /// Returns a representative example of this topology class.
    pub(crate) fn example(self) -> PrimitiveTopology {
        match self {
            Self::Point => PrimitiveTopology::PointList,
            Self::Line => PrimitiveTopology::LineList,
            Self::Triangle => PrimitiveTopology::TriangleList,
            Self::Patch => PrimitiveTopology::PatchList,
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

vulkan_enum! {
    #[non_exhaustive]

    /// An enumeration of all valid index types.
    IndexType = IndexType(i32);

    // TODO: document
    U8 = UINT8_EXT {
        device_extensions: [ext_index_type_uint8],
    },

    // TODO: document
    U16 = UINT16,

    // TODO: document
    U32 = UINT32,

    /* TODO: enable
    // TODO: document
    None = NONE_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },*/
}

impl IndexType {
    /// Returns the size in bytes of indices of this type.
    #[inline]
    pub fn size(self) -> DeviceSize {
        match self {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        }
    }
}
