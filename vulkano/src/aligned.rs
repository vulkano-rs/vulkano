// This is a modified version of https://github.com/japaric/aligned.
// Please see https://github.com/japaric/aligned#license for the licensing.
//
// The modifications made include removing the dependency for `as-slice`, implementing bytemuck's
// `AnyBitPattern` trait for the wrapper, and adjusting the documentation.

//! A newtype wrapper for enforcing correct alignment for external types.

use bytemuck::{AnyBitPattern, Zeroable};
use std::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};

/// A newtype wrapper around `T`, with an alignment of at least `A` bytes.
///
/// In Vulkan, the layout of buffer contents is not necessarily as one would expect from the type
/// signature in the shader code. For example, the *extended layout* or *std140 layout* in GLSL,
/// which is used for uniform buffers by default, requires that array elements are aligned to 16
/// bytes at minimum. That means that even if the array contains a scalar type like `u32` for
/// example, it must be aligned to 16 bytes. We can not enforce that with primitive Rust types
/// alone. In such cases, we can use `Aligned` to enforce correct alignment on the Rust side.
///
/// See also [the `shader` module documentation] for more information about layout in shaders.
///
/// [the `shader` module documentation]: crate::shader
#[repr(C)]
pub struct Aligned<A, T>
where
    A: sealed::Alignment,
{
    _alignment: [A; 0],
    value: T,
}

#[allow(non_snake_case)]
#[doc(hidden)]
#[inline(always)]
pub const fn Aligned<A, T>(value: T) -> Aligned<A, T>
where
    A: sealed::Alignment,
{
    Aligned {
        _alignment: [],
        value,
    }
}

impl<A, T> Clone for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Clone,
{
    fn clone(&self) -> Self {
        Aligned {
            _alignment: [],
            value: self.value.clone(),
        }
    }
}

impl<A, T> Copy for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Copy,
{
}

impl<A, T> Debug for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.value.fmt(f)
    }
}

impl<A, T> Default for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Default,
{
    fn default() -> Self {
        Aligned {
            _alignment: [],
            value: T::default(),
        }
    }
}

impl<A, T> Display for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.value.fmt(f)
    }
}

impl<A, T> Deref for Aligned<A, T>
where
    A: sealed::Alignment,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<A, T> DerefMut for Aligned<A, T>
where
    A: sealed::Alignment,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<A, T> PartialEq for Aligned<A, T>
where
    A: sealed::Alignment,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<A, T> Eq for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Eq,
{
}

impl<A, T> Hash for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<A, T> PartialOrd for Aligned<A, T>
where
    A: sealed::Alignment,
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<A, T> Ord for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

// SAFETY: All that's required to derive `AnyBitPattern` for structs is that all the fields are
// `AnyBitPattern`, and the only field that is stored is of type `T`. We enforce that it is
// `AnyBitPattern` with the bound.
unsafe impl<A, T> AnyBitPattern for Aligned<A, T>
where
    A: sealed::Alignment,
    T: AnyBitPattern,
{
}

// SAFETY: Same reasoning as for `AnyBitPattern` above.
unsafe impl<A, T> Zeroable for Aligned<A, T>
where
    A: sealed::Alignment,
    T: Zeroable,
{
}

/// 2-byte alignment.
#[derive(Clone, Copy)]
#[repr(align(2))]
pub struct A2;

/// 4-byte alignment.
#[derive(Clone, Copy)]
#[repr(align(4))]
pub struct A4;

/// 8-byte alignment.
#[derive(Clone, Copy)]
#[repr(align(8))]
pub struct A8;

/// 16-byte alignment.
#[derive(Clone, Copy)]
#[repr(align(16))]
pub struct A16;

mod sealed {
    pub trait Alignment: Copy + 'static {}

    impl Alignment for super::A2 {}
    impl Alignment for super::A4 {}
    impl Alignment for super::A8 {}
    impl Alignment for super::A16 {}
}
