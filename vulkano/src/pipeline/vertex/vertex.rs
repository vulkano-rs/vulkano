// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use format::Format;

/// Describes an individual `Vertex`. In other words a collection of attributes that can be read
/// from a vertex shader.
///
/// At this stage, the vertex is in a "raw" format. For example a `[f32; 4]` can match both a
/// `vec4` or a `float[4]`. The way the things are bound depends on the shader.
pub unsafe trait Vertex: 'static + Send + Sync {
    /// Returns the characteristics of a vertex member by its name.
    fn member(name: &str) -> Option<VertexMemberInfo>;
}

unsafe impl Vertex for () {
    #[inline]
    fn member(_: &str) -> Option<VertexMemberInfo> {
        None
    }
}

/// Information about a member of a vertex struct.
pub struct VertexMemberInfo {
    /// Offset of the member in bytes from the start of the struct.
    pub offset: usize,
    /// Type of data. This is used to check that the interface is matching.
    pub ty: VertexMemberTy,
    /// Number of consecutive elements of that type.
    pub array_size: usize,
}

/// Type of a member of a vertex struct.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum VertexMemberTy {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl VertexMemberTy {
    /// Returns true if a combination of `(type, array_size)` matches a format.
    #[inline]
    pub fn matches(&self, array_size: usize, format: Format, num_locs: u32) -> bool {
        // TODO: implement correctly
        let my_size = match *self {
            VertexMemberTy::I8 => 1,
            VertexMemberTy::U8 => 1,
            VertexMemberTy::I16 => 2,
            VertexMemberTy::U16 => 2,
            VertexMemberTy::I32 => 4,
            VertexMemberTy::U32 => 4,
            VertexMemberTy::F32 => 4,
            VertexMemberTy::F64 => 8,
        };

        let format_size = match format.size() {
            None => return false,
            Some(s) => s,
        };

        array_size * my_size == format_size * num_locs as usize
    }
}
