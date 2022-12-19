// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::Format;

/// Implements the `Vertex` trait on a struct.
///
/// # Examples
///
/// ```
/// # use bytemuck::{Zeroable, Pod};
/// #[repr(C)]
/// #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
/// struct Vertex{
///     position: [f32; 3],
///     color: [f32; 4],
/// }
///
/// vulkano::impl_vertex!(Vertex, position, color);
/// ```
#[macro_export]
macro_rules! impl_vertex {
    ($out:ty $(, $member:ident)*) => (
        #[allow(unsafe_code)]
        unsafe impl $crate::pipeline::graphics::vertex_input::Vertex for $out {
            #[inline(always)]
            fn member(name: &str) -> Option<$crate::pipeline::graphics::vertex_input::VertexMemberInfo> {
                #[allow(unused_imports)]
                use $crate::format::Format;
                use $crate::pipeline::graphics::vertex_input::VertexMemberInfo;
                use $crate::pipeline::graphics::vertex_input::VertexMember;

                $(
                    if name == stringify!($member) {
                        let dummy = <$out>::default();
                        #[inline] fn f<T: VertexMember>(_: &T) -> Format { T::format() }
                        let format = f(&dummy.$member);
                        let field_size = {
                            let p = unsafe {
                                core::ptr::addr_of!((*(&dummy as *const _ as *const $out)).$member)
                            };
                            const fn size_of_raw<T>(_: *const T) -> usize {
                                core::mem::size_of::<T>()
                            }
                            size_of_raw(p)
                        } as u32;
                        let format_size = format.block_size().expect("no block size for format") as u32;
                        let num_locations = field_size / format_size;
                        let remainder = field_size % format_size;
                        assert!(remainder == 0, "struct field `{}` size does not fit multiple of format size", name);

                        let dummy_ptr = (&dummy) as *const _;
                        let member_ptr = (&dummy.$member) as *const _;

                        return Some(VertexMemberInfo {
                            offset: member_ptr as usize - dummy_ptr as usize,
                            format,
                            num_locations,
                        });
                    }
                )*

                None
            }
        }
    )
}

/// Trait for data types that can be used as vertex members. Used by the `impl_vertex!` macro.
pub unsafe trait VertexMember {
    /// Returns the format and array size of the member.
    fn format() -> Format;
}

#[macro_export]
macro_rules! impl_vertex_member {
    ($out:ty, $format:ident) => {
        unsafe impl VertexMember for $out {
            #[inline]
            fn format() -> Format {
                Format::$format
            }
        }
    };
}

impl_vertex_member!(i8, R8_SINT);
impl_vertex_member!(u8, R8_UINT);
impl_vertex_member!(i16, R16_SINT);
impl_vertex_member!(u16, R16_UINT);
impl_vertex_member!(i32, R32_SINT);
impl_vertex_member!(u32, R32_UINT);
impl_vertex_member!(f32, R32_SFLOAT);
impl_vertex_member!(f64, R64_SFLOAT);
impl_vertex_member!([i8; 2], R8G8_SINT);
impl_vertex_member!([u8; 2], R8G8_UINT);
impl_vertex_member!([i16; 2], R16G16_SINT);
impl_vertex_member!([u16; 2], R16G16_UINT);
impl_vertex_member!([i32; 2], R32G32_SINT);
impl_vertex_member!([u32; 2], R32G32_UINT);
impl_vertex_member!([f32; 2], R32G32_SFLOAT);
impl_vertex_member!([f64; 2], R64G64_SFLOAT);
impl_vertex_member!([i8; 3], R8G8B8_SINT);
impl_vertex_member!([u8; 3], R8G8B8_UINT);
impl_vertex_member!([i16; 3], R16G16B16_SINT);
impl_vertex_member!([u16; 3], R16G16B16_UINT);
impl_vertex_member!([i32; 3], R32G32B32_SINT);
impl_vertex_member!([u32; 3], R32G32B32_UINT);
impl_vertex_member!([f32; 3], R32G32B32_SFLOAT);
impl_vertex_member!([f64; 3], R64G64B64_SFLOAT);
impl_vertex_member!([i8; 4], R8G8B8A8_SINT);
impl_vertex_member!([u8; 4], R8G8B8A8_UINT);
impl_vertex_member!([i16; 4], R16G16B16A16_SINT);
impl_vertex_member!([u16; 4], R16G16B16A16_UINT);
impl_vertex_member!([i32; 4], R32G32B32A32_SINT);
impl_vertex_member!([u32; 4], R32G32B32A32_UINT);
impl_vertex_member!([f32; 4], R32G32B32A32_SFLOAT);
impl_vertex_member!([f64; 4], R64G64B64A64_SFLOAT);
impl_vertex_member!([f32; 9], R32G32B32_SFLOAT);
impl_vertex_member!([f32; 16], R32G32B32A32_SFLOAT);

unsafe impl<T> VertexMember for [T; 1]
where
    T: VertexMember,
{
    fn format() -> Format {
        <T as VertexMember>::format()
    }
}

#[cfg(test)]
mod tests {
    use crate::format::Format;
    use crate::pipeline::graphics::vertex_input::Vertex;

    use bytemuck::{Pod, Zeroable};

    #[test]
    fn vertex_impl() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
        struct TestVertex {
            matrix: [f32; 16],
            vector: [f32; 4],
            scalar: u16,
            _padding: u16,
        }
        impl_vertex!(TestVertex, scalar, vector, matrix);

        let matrix = TestVertex::member("matrix").unwrap();
        let vector = TestVertex::member("vector").unwrap();
        let scalar = TestVertex::member("scalar").unwrap();
        assert_eq!(matrix.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(matrix.offset, 0);
        assert_eq!(matrix.num_locations, 4);
        assert_eq!(vector.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(vector.offset, 16 * 4);
        assert_eq!(vector.num_locations, 1);
        assert_eq!(scalar.format, Format::R16_UINT);
        assert_eq!(scalar.offset, 16 * 5);
        assert_eq!(scalar.num_locations, 1);
    }
}
