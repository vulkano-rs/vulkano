// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use pipeline::vertex::VertexMemberTy;

/// Implements the `Vertex` trait on a struct.
///# Example
///
///```
///#[derive(Default, Copy, Clone)]
///struct Vertex{
///  position: [f32; 3],
///  color: [f32; 4]
///}
///
///vulkano::impl_vertex!(Vertex, position, color);
///
///```
#[macro_export]
macro_rules! impl_vertex {
    ($out:ty $(, $member:ident)*) => (
        #[allow(unsafe_code)]
        unsafe impl $crate::pipeline::vertex::Vertex for $out {
            #[inline(always)]
            fn member(name: &str) -> Option<$crate::pipeline::vertex::VertexMemberInfo> {
                use std::ptr;
                #[allow(unused_imports)]
                use $crate::format::Format;
                use $crate::pipeline::vertex::VertexMemberInfo;
                use $crate::pipeline::vertex::VertexMemberTy;
                use $crate::pipeline::vertex::VertexMember;

                $(
                    if name == stringify!($member) {
                        let dummy = <$out>::default();
                        #[inline] fn f<T: VertexMember>(_: &T) -> (VertexMemberTy, usize) { T::format() }
                        let (ty, array_size) = f(&dummy.$member);

                        let dummy_ptr = (&dummy) as *const _;
                        let member_ptr = (&dummy.$member) as *const _;

                        return Some(VertexMemberInfo {
                            offset: member_ptr as usize - dummy_ptr as usize,
                            ty: ty,
                            array_size: array_size,
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
    fn format() -> (VertexMemberTy, usize);
}

unsafe impl VertexMember for i8 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::I8, 1)
    }
}

unsafe impl VertexMember for u8 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::U8, 1)
    }
}

unsafe impl VertexMember for i16 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::I16, 1)
    }
}

unsafe impl VertexMember for u16 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::U16, 1)
    }
}

unsafe impl VertexMember for i32 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::I32, 1)
    }
}

unsafe impl VertexMember for u32 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::U32, 1)
    }
}

unsafe impl VertexMember for f32 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::F32, 1)
    }
}

unsafe impl VertexMember for f64 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::F64, 1)
    }
}

unsafe impl<T> VertexMember for (T,)
where
    T: VertexMember,
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        <T as VertexMember>::format()
    }
}

unsafe impl<T> VertexMember for (T, T)
where
    T: VertexMember,
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 2)
    }
}

unsafe impl<T> VertexMember for (T, T, T)
where
    T: VertexMember,
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 3)
    }
}

unsafe impl<T> VertexMember for (T, T, T, T)
where
    T: VertexMember,
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 4)
    }
}

macro_rules! impl_vm_array {
    ($sz:expr) => {
        unsafe impl<T> VertexMember for [T; $sz]
        where
            T: VertexMember,
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                let (ty, sz) = <T as VertexMember>::format();
                (ty, sz * $sz)
            }
        }
    };
}

impl_vm_array!(1);
impl_vm_array!(2);
impl_vm_array!(3);
impl_vm_array!(4);
impl_vm_array!(5);
impl_vm_array!(6);
impl_vm_array!(7);
impl_vm_array!(8);
impl_vm_array!(9);
impl_vm_array!(10);
impl_vm_array!(11);
impl_vm_array!(12);
impl_vm_array!(13);
impl_vm_array!(14);
impl_vm_array!(15);
impl_vm_array!(16);
impl_vm_array!(32);
impl_vm_array!(64);
