// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A newtype wrapper for enforcing correct alignment for external types.

use crate::buffer::{BufferContents, BufferContentsLayout};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    alloc::Layout,
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    mem::{align_of, size_of, MaybeUninit},
    ops::{Deref, DerefMut},
};

/// A newtype wrapper around `T`, with `N` bytes of trailing padding.
///
/// In Vulkan, the layout of buffer contents is not necessarily as one would expect from the type
/// signature in the shader code. For example, the *extended layout* or *std140 layout* in GLSL,
/// which is used for uniform buffers by default, requires that array elements are aligned to 16
/// bytes at minimum. That means that even if the array contains a scalar type like `u32` for
/// example, it must be aligned to 16 bytes. We can not enforce that with primitive Rust types
/// alone. In such cases, we can use `Padded` to enforce correct alignment on the Rust side.
///
/// See also [the `shader` module documentation] for more information about layout in shaders.
///
/// [the `shader` module documentation]: crate::shader
#[repr(C)]
pub struct Padded<T, const N: usize> {
    value: T,
    _padding: [MaybeUninit<u8>; N],
}

#[allow(non_snake_case)]
#[doc(hidden)]
#[inline(always)]
pub const fn Padded<T, const N: usize>(value: T) -> Padded<T, N> {
    Padded {
        value,
        _padding: [MaybeUninit::uninit(); N],
    }
}

impl<T, const N: usize> Clone for Padded<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Padded(self.value.clone())
    }
}

impl<T, const N: usize> Copy for Padded<T, N> where T: Copy {}

impl<T, const N: usize> Debug for Padded<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.value.fmt(f)
    }
}

impl<T, const N: usize> Default for Padded<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Padded(T::default())
    }
}

impl<T, const N: usize> Deref for Padded<T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, const N: usize> DerefMut for Padded<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T, const N: usize> Display for Padded<T, N>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.value.fmt(f)
    }
}

impl<T, const N: usize> From<T> for Padded<T, N> {
    fn from(value: T) -> Self {
        Padded(value)
    }
}

impl<T, const N: usize> PartialEq for Padded<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T, const N: usize> Eq for Padded<T, N> where T: Eq {}

impl<T, const N: usize> Hash for Padded<T, N>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T, const N: usize> PartialOrd for Padded<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T, const N: usize> Ord for Padded<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

unsafe impl<T, const N: usize> BufferContents for Padded<T, N>
where
    T: BufferContents,
{
    const LAYOUT: BufferContentsLayout =
        if let Some(layout) = BufferContentsLayout::from_sized(Layout::new::<Self>()) {
            layout
        } else {
            panic!("zero-sized types are not valid buffer contents");
        };

    unsafe fn from_ffi(data: *const std::ffi::c_void, range: usize) -> *const Self {
        debug_assert!(range == size_of::<Self>());
        debug_assert!(data as usize % align_of::<Self>() == 0);

        data.cast()
    }

    unsafe fn from_ffi_mut(data: *mut std::ffi::c_void, range: usize) -> *mut Self {
        debug_assert!(range == size_of::<Self>());
        debug_assert!(data as usize % align_of::<Self>() == 0);

        data.cast()
    }
}

#[cfg(feature = "serde")]
impl<T, const N: usize> Serialize for Padded<T, N>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Deserialize<'de> for Padded<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Padded)
    }
}
