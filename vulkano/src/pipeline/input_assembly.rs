use vk;

/// How the input assembly stage should behave.
#[derive(Copy, Clone, Debug)]
pub struct InputAssembly {
    /// The type of primitives.
    ///
    /// Note that some tologies don't support primitive restart.
    pub topology: PrimitiveTopology,

    /// If true, then the special index value `0xffff` or `0xffffffff` will tell the GPU that it is
    /// the end of the current primitive. A new primitive will restart at the next index.
    ///
    /// Note that some tologies don't support primitive restart.
    pub primitive_restart_enable: bool,
}

/// Describes how vertices must be grouped together to form primitives.
///
/// Note that some tologies don't support primitive restart.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum PrimitiveTopology {
    PointList = vk::PRIMITIVE_TOPOLOGY_POINT_LIST,
    LineList = vk::PRIMITIVE_TOPOLOGY_LINE_LIST,
    LineStrip = vk::PRIMITIVE_TOPOLOGY_LINE_STRIP,
    TriangleList = vk::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    TriangleStrip = vk::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    TriangleFan = vk::PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    LineListWithAdjacency = vk::PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
    LineStripWithAdjacency = vk::PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
    TriangleListWithAdjancecy = vk::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
    TriangleStripWithAdjacency = vk::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
    PatchList = vk::PRIMITIVE_TOPOLOGY_PATCH_LIST,
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
            _ => false
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
#[derive(Copy, Clone, Debug)]
#[allow(missing_docs)]
#[repr(u32)]
pub enum IndexType {
    U16 = vk::INDEX_TYPE_UINT16,
    U32 = vk::INDEX_TYPE_UINT32,
}
