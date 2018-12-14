// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use pipeline::vertex::VertexMemberTy;

/// Implements the `Vertex` trait on a struct.
// TODO: add example
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
                        let (ty, array_size) = unsafe {
                            #[inline] fn f<T: VertexMember>(_: &T) -> (VertexMemberTy, usize)
                                      { T::format() }
                            let dummy: *const $out = ptr::null();
                            f(&(&*dummy).$member)
                        };

                        return Some(VertexMemberInfo {
                            offset: unsafe {
                                let dummy: *const $out = ptr::null();
                                let member = (&(&*dummy).$member) as *const _;
                                member as usize
                            },

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
///
/// By default, Vulkano implements this for integers and floats and tuples of them up to 4 elements. Vulkano also
/// implements this for integer and float arrays up to 16 elements and then 32 and 64 elements exactly. If the "cgmath"
/// or "nalgebra" [features are
/// enabled](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#choosing-features), Vulkano will
/// implement this trait for the respective vector types from those crates.
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
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        <T as VertexMember>::format()
    }
}

unsafe impl<T> VertexMember for (T, T)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 2)
    }
}

unsafe impl<T> VertexMember for (T, T, T)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 3)
    }
}

unsafe impl<T> VertexMember for (T, T, T, T)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 4)
    }
}

macro_rules! impl_vm_array {
    ($sz:expr) => (
        unsafe impl<T> VertexMember for [T; $sz]
            where T: VertexMember
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                let (ty, sz) = <T as VertexMember>::format();
                (ty, sz * $sz)
            }
        }
    );
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

macro_rules! impl_cgmath_vector {
    ($vector:ident, $components:expr) => (
        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<i8>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::I8, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<u8>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::U8, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<i16>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::I16, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<u16>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::U16, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<i32>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::I32, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<u32>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::U32, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<f32>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::F32, $components)
            }
        }

        #[cfg(feature = "cgmath")]
        unsafe impl VertexMember for cgmath::$vector<f64>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::F64, $components)
            }
        }
    );
}

impl_cgmath_vector!(Vector1, 1);
impl_cgmath_vector!(Vector2, 2);
impl_cgmath_vector!(Vector3, 3);
impl_cgmath_vector!(Vector4, 4);

macro_rules! impl_nalgebra_vector {
    ($vector:ident, $components:expr) => (
        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<i8>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::I8, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<u8>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::U8, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<i16>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::I16, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<u16>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::U16, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<i32>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::I32, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<u32>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::U32, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<f32>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::F32, $components)
            }
        }

        #[cfg(feature = "nalgebra")]
        unsafe impl VertexMember for nalgebra::$vector<f64>
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                (VertexMemberTy::F64, $components)
            }
        }
    );
}

impl_nalgebra_vector!(Vector1, 1);
impl_nalgebra_vector!(Vector2, 2);
impl_nalgebra_vector!(Vector3, 3);
impl_nalgebra_vector!(Vector4, 4);
