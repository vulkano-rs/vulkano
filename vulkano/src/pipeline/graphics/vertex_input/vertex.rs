// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::Format;
use bytemuck::Pod;

/// Describes an individual `Vertex`. In other words a collection of attributes that can be read
/// from a vertex shader.
///
/// At this stage, the vertex is in a "raw" format. For example a `[f32; 4]` can match both a
/// `vec4` or a `float[4]`. The way the things are bound depends on the shader.
pub unsafe trait Vertex: Pod + Send + Sync + 'static {
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexMemberInfo {
    /// Offset of the member in bytes from the start of the struct.
    pub offset: usize,
    /// Attribute format of the member. Implicitly provides number of components.
    pub format: Format,
    /// Number of consecutive array elements or matrix columns using format.
    pub num_elements: u32,
}
