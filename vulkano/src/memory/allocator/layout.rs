// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::align_up;
use crate::{DeviceSize, NonZeroDeviceSize};
use std::{
    alloc::Layout,
    cmp::{self, Ordering},
    error::Error,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    mem::{self, align_of, size_of},
};

/// Vulkan analog of std's [`Layout`], represented using [`DeviceSize`]s.
///
/// Unlike `Layout`s, `DeviceLayout`s are required to have non-zero size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceLayout {
    size: NonZeroDeviceSize,
    alignment: DeviceAlignment,
}

impl DeviceLayout {
    /// The maximum size of a memory block after its layout's size has been rounded up to the
    /// nearest multiple of its layout's alignment.
    ///
    /// This invariant is enforced to avoid arithmetic overflows when constructing layouts and when
    /// allocating memory. Any layout that doesn't uphold this invariant will therefore *lead to
    /// undefined behavior*.
    pub const MAX_SIZE: DeviceSize = DeviceAlignment::MAX.as_devicesize() - 1;

    /// Creates a new `DeviceLayout` from a [`Layout`], or returns an error if the `Layout` has
    /// zero size.
    #[inline]
    pub const fn from_layout(layout: Layout) -> Result<Self, TryFromLayoutError> {
        let (size, alignment) = (layout.size(), layout.align());

        #[cfg(any(
            target_pointer_width = "64",
            target_pointer_width = "32",
            target_pointer_width = "16",
        ))]
        {
            const _: () = assert!(size_of::<DeviceSize>() >= size_of::<usize>());
            const _: () = assert!(DeviceLayout::MAX_SIZE >= isize::MAX as DeviceSize);

            if let Some(size) = NonZeroDeviceSize::new(size as DeviceSize) {
                // SAFETY: `Layout`'s alignment-invariant guarantees that it is a power of two.
                let alignment = unsafe { DeviceAlignment::new_unchecked(alignment as DeviceSize) };

                // SAFETY: Under the precondition that `usize` can't overflow `DeviceSize`, which
                // we checked above, `Layout`'s overflow-invariant is the same if not stricter than
                // that of `DeviceLayout`.
                Ok(unsafe { DeviceLayout::new_unchecked(size, alignment) })
            } else {
                Err(TryFromLayoutError)
            }
        }
    }

    /// Converts the `DeviceLayout` into a [`Layout`], or returns an error if the `DeviceLayout`
    /// doesn't meet the invariants of `Layout`.
    #[inline]
    pub const fn into_layout(self) -> Result<Layout, TryFromDeviceLayoutError> {
        let (size, alignment) = (self.size(), self.alignment().as_devicesize());

        #[cfg(target_pointer_width = "64")]
        {
            const _: () = assert!(size_of::<DeviceSize>() <= size_of::<usize>());
            const _: () = assert!(DeviceLayout::MAX_SIZE as usize <= isize::MAX as usize);

            // SAFETY: Under the precondition that `DeviceSize` can't overflow `usize`, which we
            // checked above, `DeviceLayout`'s overflow-invariant is the same if not stricter that
            // of `Layout`.
            Ok(unsafe { Layout::from_size_align_unchecked(size as usize, alignment as usize) })
        }
        #[cfg(any(target_pointer_width = "32", target_pointer_width = "16"))]
        {
            const _: () = assert!(size_of::<DeviceSize>() > size_of::<usize>());
            const _: () = assert!(DeviceLayout::MAX_SIZE > isize::MAX as DeviceSize);

            if size > usize::MAX as DeviceSize || alignment > usize::MAX as DeviceSize {
                Err(TryFromDeviceLayoutError)
            } else if let Ok(layout) = Layout::from_size_align(size as usize, alignment as usize) {
                Ok(layout)
            } else {
                Err(TryFromDeviceLayoutError)
            }
        }
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment`.
    ///
    /// Returns [`None`] if `size` is zero, `alignment` is not a power of two, or if `size` would
    /// exceed [`DeviceLayout::MAX_SIZE`] when rounded up to the nearest multiple of `alignment`.
    #[inline]
    pub const fn from_size_alignment(size: DeviceSize, alignment: DeviceSize) -> Option<Self> {
        if let (Some(size), Some(alignment)) = (
            NonZeroDeviceSize::new(size),
            DeviceAlignment::new(alignment),
        ) {
            DeviceLayout::new(size, alignment)
        } else {
            None
        }
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment` without doing any
    /// checks.
    ///
    /// # Safety
    ///
    /// - `size` must be non-zero.
    /// - `alignment` must be a power of two, which also means it must be non-zero.
    /// - `size`, when rounded up to the nearest multiple of `alignment`, must not exceed
    ///   [`DeviceLayout::MAX_SIZE`].
    #[inline]
    pub const unsafe fn from_size_alignment_unchecked(
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> Self {
        DeviceLayout::new_unchecked(
            NonZeroDeviceSize::new_unchecked(size),
            DeviceAlignment::new_unchecked(alignment),
        )
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment`.
    ///
    /// Returns [`None`] if `size` would exceed [`DeviceLayout::MAX_SIZE`] when rounded up to the
    /// nearest multiple of `alignment`.
    #[inline]
    pub const fn new(size: NonZeroDeviceSize, alignment: DeviceAlignment) -> Option<Self> {
        if size.get() > Self::max_size_for_alignment(alignment) {
            None
        } else {
            // SAFETY: We checked that the rounded-up size won't exceed `DeviceLayout::MAX_SIZE`.
            Some(unsafe { DeviceLayout::new_unchecked(size, alignment) })
        }
    }

    #[inline(always)]
    const fn max_size_for_alignment(alignment: DeviceAlignment) -> DeviceSize {
        // `DeviceLayout::MAX_SIZE` is `DeviceAlignment::MAX - 1`, so this can't overflow.
        DeviceLayout::MAX_SIZE - (alignment.as_devicesize() - 1)
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment` without checking for
    /// potential overflow.
    ///
    /// # Safety
    ///
    /// - `size`, when rounded up to the nearest multiple of `alignment`, must not exceed
    ///   [`DeviceLayout::MAX_SIZE`].
    #[inline]
    pub const unsafe fn new_unchecked(size: NonZeroDeviceSize, alignment: DeviceAlignment) -> Self {
        debug_assert!(size.get() <= Self::max_size_for_alignment(alignment));

        DeviceLayout { size, alignment }
    }

    /// Returns the minimum size in bytes for a memory block of this layout.
    #[inline]
    pub const fn size(&self) -> DeviceSize {
        self.size.get()
    }

    /// Returns the minimum alignment for a memory block of this layout.
    #[inline]
    pub const fn alignment(&self) -> DeviceAlignment {
        self.alignment
    }

    /// Creates a new `DeviceLayout` from `self` that is also aligned to `alignment` at minimum.
    ///
    /// Returns [`None`] if `self.size()` would overflow [`DeviceLayout::MAX_SIZE`] when rounded up
    /// to the nearest multiple of `alignment`.
    #[inline]
    pub fn align_to(&self, alignment: DeviceAlignment) -> Option<Self> {
        DeviceLayout::new(self.size, cmp::max(self.alignment, alignment))
    }

    /// Returns the amount of padding that needs to be added to `self.size()` such that the result
    /// is a multiple of `alignment`.
    #[inline]
    pub const fn padding_needed_for(&self, alignment: DeviceAlignment) -> DeviceSize {
        let size = self.size();

        align_up(size, alignment).wrapping_sub(size)
    }

    /// Creates a new `DeviceLayout` by rounding up `self.size()` to the nearest multiple of
    /// `self.alignment()`.
    #[inline]
    pub const fn pad_to_alignment(&self) -> Self {
        // SAFETY: `DeviceLayout`'s invariant guarantees that the rounded up size won't exceed
        // `DeviceLayout::MAX_SIZE`.
        unsafe { DeviceLayout::new_unchecked(self.padded_size(), self.alignment) }
    }

    #[inline(always)]
    const fn padded_size(&self) -> NonZeroDeviceSize {
        let size = align_up(self.size(), self.alignment);

        // SAFETY: `DeviceLayout`'s invariant guarantees that the rounded up size won't overflow.
        unsafe { NonZeroDeviceSize::new_unchecked(size) }
    }

    /// Creates a new `DeviceLayout` describing the record for `n` instances of `self`, possibly
    /// with padding at the end of each to ensure correct alignment of all instances.
    ///
    /// Returns a tuple consisting of the new layout and the stride, in bytes, of `self`, or
    /// returns [`None`] on arithmetic overflow or when the total size would exceed
    /// [`DeviceLayout::MAX_SIZE`].
    #[inline]
    pub fn repeat(&self, n: NonZeroDeviceSize) -> Option<(Self, DeviceSize)> {
        let stride = self.padded_size();
        let size = stride.checked_mul(n)?;

        DeviceLayout::new(size, self.alignment).map(|layout| (layout, stride.get()))
    }

    /// Creates a new `DeviceLayout` describing the record for `self` followed by `next`, including
    /// potential padding between them to ensure `next` will be properly aligned, but without any
    /// trailing padding. You should use [`pad_to_alignment`] after you are done extending the
    /// layout with all fields to get a valid `#[repr(C)]` layout.
    ///
    /// The alignments of the two layouts get combined by picking the maximum between them.
    ///
    /// Returns a tuple consisting of the resulting layout as well as the offset, in bytes, of
    /// `next`.
    ///
    /// Returns [`None`] on arithmetic overflow or when the total size rounded up to the nearest
    /// multiple of the combined alignment would exceed [`DeviceLayout::MAX_SIZE`].
    ///
    /// [`pad_to_alignment`]: Self::pad_to_alignment
    #[inline]
    pub fn extend(&self, next: Self) -> Option<(Self, DeviceSize)> {
        let padding = self.padding_needed_for(next.alignment);
        let offset = self.size.checked_add(padding)?;
        let size = offset.checked_add(next.size())?;
        let alignment = cmp::max(self.alignment, next.alignment);

        DeviceLayout::new(size, alignment).map(|layout| (layout, offset.get()))
    }
}

impl TryFrom<Layout> for DeviceLayout {
    type Error = TryFromLayoutError;

    #[inline]
    fn try_from(layout: Layout) -> Result<Self, Self::Error> {
        DeviceLayout::from_layout(layout)
    }
}

impl TryFrom<DeviceLayout> for Layout {
    type Error = TryFromDeviceLayoutError;

    #[inline]
    fn try_from(device_layout: DeviceLayout) -> Result<Self, Self::Error> {
        DeviceLayout::into_layout(device_layout)
    }
}

/// Error that can happen when converting a [`Layout`] to a [`DeviceLayout`].
///
/// It occurs when the `Layout` has zero size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryFromLayoutError;

impl Error for TryFromLayoutError {}

impl Display for TryFromLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("attempted to convert a zero-size `Layout` to a `DeviceLayout`")
    }
}

/// Error that can happen when converting a [`DeviceLayout`] to a [`Layout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryFromDeviceLayoutError;

impl Error for TryFromDeviceLayoutError {}

impl Display for TryFromDeviceLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(
            "attempted to convert a `DeviceLayout` to a `Layout` which would result in a \
            violation of `Layout`'s overflow-invariant",
        )
    }
}

/// Vulkan analog of std's [`Alignment`], stored as a [`DeviceSize`] that is guaranteed to be a
/// valid Vulkan alignment.
///
/// [`Alignment`]: std::ptr::Alignment
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DeviceAlignment(AlignmentEnum);

const _: () = assert!(size_of::<DeviceAlignment>() == size_of::<DeviceSize>());
const _: () = assert!(align_of::<DeviceAlignment>() == align_of::<DeviceSize>());

impl DeviceAlignment {
    /// The smallest possible alignment, 1.
    pub const MIN: Self = Self(AlignmentEnum::_Align1Shl0);

    /// The largest possible alignment, 2<sup>63</sup>.
    pub const MAX: Self = Self(AlignmentEnum::_Align1Shl63);

    /// Returns the alignment for a type.
    #[inline]
    pub const fn of<T>() -> Self {
        #[cfg(any(
            target_pointer_width = "64",
            target_pointer_width = "32",
            target_pointer_width = "16",
        ))]
        {
            const _: () = assert!(size_of::<DeviceSize>() >= size_of::<usize>());

            // SAFETY: rustc guarantees that the alignment of types is a power of two.
            unsafe { DeviceAlignment::new_unchecked(align_of::<T>() as DeviceSize) }
        }
    }

    /// Tries to create a `DeviceAlignment` from a [`DeviceSize`], returning [`None`] if it's not a
    /// power of two.
    #[inline]
    pub const fn new(alignment: DeviceSize) -> Option<Self> {
        if alignment.is_power_of_two() {
            Some(unsafe { DeviceAlignment::new_unchecked(alignment) })
        } else {
            None
        }
    }

    /// Creates a `DeviceAlignment` from a [`DeviceSize`] without checking if it's a power of two.
    ///
    /// # Safety
    ///
    /// - `alignment` must be a power of two, which also means it must be non-zero.
    #[inline]
    pub const unsafe fn new_unchecked(alignment: DeviceSize) -> Self {
        debug_assert!(alignment.is_power_of_two());

        unsafe { mem::transmute::<DeviceSize, DeviceAlignment>(alignment) }
    }

    /// Returns the alignment as a [`DeviceSize`].
    #[inline]
    pub const fn as_devicesize(self) -> DeviceSize {
        self.0 as DeviceSize
    }

    /// Returns the alignment as a [`NonZeroDeviceSize`].
    #[inline]
    pub const fn as_nonzero(self) -> NonZeroDeviceSize {
        // SAFETY: All the discriminants are non-zero.
        unsafe { NonZeroDeviceSize::new_unchecked(self.as_devicesize()) }
    }

    /// Returns the base-2 logarithm of the alignment.
    #[inline]
    pub const fn log2(self) -> u32 {
        self.as_nonzero().trailing_zeros()
    }
}

impl Debug for DeviceAlignment {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?} (1 << {:?})", self.as_nonzero(), self.log2())
    }
}

impl TryFrom<NonZeroDeviceSize> for DeviceAlignment {
    type Error = TryFromIntError;

    #[inline]
    fn try_from(alignment: NonZeroDeviceSize) -> Result<Self, Self::Error> {
        if alignment.is_power_of_two() {
            Ok(unsafe { DeviceAlignment::new_unchecked(alignment.get()) })
        } else {
            Err(TryFromIntError)
        }
    }
}

impl TryFrom<DeviceSize> for DeviceAlignment {
    type Error = TryFromIntError;

    #[inline]
    fn try_from(alignment: DeviceSize) -> Result<Self, Self::Error> {
        DeviceAlignment::new(alignment).ok_or(TryFromIntError)
    }
}

impl From<DeviceAlignment> for NonZeroDeviceSize {
    #[inline]
    fn from(alignment: DeviceAlignment) -> Self {
        alignment.as_nonzero()
    }
}

impl From<DeviceAlignment> for DeviceSize {
    #[inline]
    fn from(alignment: DeviceAlignment) -> Self {
        alignment.as_devicesize()
    }
}

// This is a false-positive, the underlying values that this impl and the derived `PartialEq` work
// with are the same.
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for DeviceAlignment {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_nonzero().hash(state);
    }
}

impl PartialOrd for DeviceAlignment {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_nonzero().partial_cmp(&other.as_nonzero())
    }
}

impl Ord for DeviceAlignment {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_nonzero().cmp(&other.as_nonzero())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
enum AlignmentEnum {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
    _Align1Shl16 = 1 << 16,
    _Align1Shl17 = 1 << 17,
    _Align1Shl18 = 1 << 18,
    _Align1Shl19 = 1 << 19,
    _Align1Shl20 = 1 << 20,
    _Align1Shl21 = 1 << 21,
    _Align1Shl22 = 1 << 22,
    _Align1Shl23 = 1 << 23,
    _Align1Shl24 = 1 << 24,
    _Align1Shl25 = 1 << 25,
    _Align1Shl26 = 1 << 26,
    _Align1Shl27 = 1 << 27,
    _Align1Shl28 = 1 << 28,
    _Align1Shl29 = 1 << 29,
    _Align1Shl30 = 1 << 30,
    _Align1Shl31 = 1 << 31,
    _Align1Shl32 = 1 << 32,
    _Align1Shl33 = 1 << 33,
    _Align1Shl34 = 1 << 34,
    _Align1Shl35 = 1 << 35,
    _Align1Shl36 = 1 << 36,
    _Align1Shl37 = 1 << 37,
    _Align1Shl38 = 1 << 38,
    _Align1Shl39 = 1 << 39,
    _Align1Shl40 = 1 << 40,
    _Align1Shl41 = 1 << 41,
    _Align1Shl42 = 1 << 42,
    _Align1Shl43 = 1 << 43,
    _Align1Shl44 = 1 << 44,
    _Align1Shl45 = 1 << 45,
    _Align1Shl46 = 1 << 46,
    _Align1Shl47 = 1 << 47,
    _Align1Shl48 = 1 << 48,
    _Align1Shl49 = 1 << 49,
    _Align1Shl50 = 1 << 50,
    _Align1Shl51 = 1 << 51,
    _Align1Shl52 = 1 << 52,
    _Align1Shl53 = 1 << 53,
    _Align1Shl54 = 1 << 54,
    _Align1Shl55 = 1 << 55,
    _Align1Shl56 = 1 << 56,
    _Align1Shl57 = 1 << 57,
    _Align1Shl58 = 1 << 58,
    _Align1Shl59 = 1 << 59,
    _Align1Shl60 = 1 << 60,
    _Align1Shl61 = 1 << 61,
    _Align1Shl62 = 1 << 62,
    _Align1Shl63 = 1 << 63,
}

/// Error that can happen when trying to convert an integer to a `DeviceAlignment`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryFromIntError;

impl Error for TryFromIntError {}

impl Display for TryFromIntError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("attempted to convert a non-power-of-two integer to a `DeviceAlignment`")
    }
}
