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
pub use vulkano_macros::Vertex;

/// Describes an individual `Vertex`. In other words a collection of attributes that can be read
/// from a vertex shader.
///
/// At this stage, the vertex is in a "raw" format. For example a `[f32; 4]` can match both a
/// `vec4` or a `float[4]`. The way the things are bound depends on the shader.
///
/// The vertex trait can be derived and the format has to be specified using the `format`
/// field-level attribute:
/// ```
/// #[repr(C)]
/// #[derive(Clone, Copy, Debug, Default, Vertex)]
/// struct MyVertex {A
///     // Every field needs to explicitly state the desired shader input format
///     #[format(R32G32B32_SFLOAT)]
///     pos: [f32; 3],
///     // The `name` attribute can be used to specify shader input names to match.
///     // By default the field-name is used.
///     #[name("in_proj", "cam_proj")]
///     #[format(R32G32B32_SFLOAT)]
///     proj: [f32; 16],
/// }
/// ```
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
    /// Number of consecutive array elements or matrix columns using format. The corresponding
    /// number of locations might defer depending on the size of the format.
    pub num_elements: u32,
}

impl VertexMemberInfo {
    pub fn num_components(&self) -> u32 {
        self.format
            .components()
            .iter()
            .filter(|&bits| *bits > 0)
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use crate::format::Format;
    use crate::pipeline::graphics::vertex_input::Vertex;

    use bytemuck::{Pod, Zeroable};

    #[test]
    fn derive_vertex_infer() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
        struct TestVertex {
            matrix: [f32; 16],
            vector: [f32; 4],
            scalar: u16,
            _padding: u16,
        }

        let matrix = TestVertex::member("matrix").unwrap();
        let vector = TestVertex::member("vector").unwrap();
        let scalar = TestVertex::member("scalar").unwrap();
        assert_eq!(matrix.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(matrix.offset, 0);
        assert_eq!(matrix.num_elements, 4);
        assert_eq!(vector.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(vector.offset, 16 * 4);
        assert_eq!(vector.num_elements, 1);
        assert_eq!(scalar.format, Format::R16_UINT);
        assert_eq!(scalar.offset, 16 * 5);
        assert_eq!(scalar.num_elements, 1);
    }

    #[test]
    fn derive_vertex_multiple_names() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
        struct TestVertex {
            #[name("b", "c")]
            a: [f32; 16],
        }

        let b = TestVertex::member("b").unwrap();
        let c = TestVertex::member("c").unwrap();
        assert_eq!(b.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(c.format, Format::R32G32B32A32_SFLOAT);
    }

    #[test]
    fn derive_vertex_format() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
        struct TestVertex {
            #[format(R8_UNORM)]
            unorm: u8,
        }

        let unorm = TestVertex::member("unorm").unwrap();
        assert_eq!(unorm.format, Format::R8_UNORM);
    }
}
