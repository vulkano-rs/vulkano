#![allow(unused_unsafe, clippy::mut_from_ref)]

use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    cell::Cell,
    cmp,
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
};

const INITIAL_REGION_SIZE: usize = 4096;
const MIN_REGION_ALIGNMENT: usize = 16;

/// A bump allocator, growing down like the stack typically does.
//
// INVARIANT: All of `free_start_ptr`, `free_end_ptr` and `block_footer_ptr` must be derived from
// the allocation of the current block, with all provenance preserved, except when no allocations
// from the arena have been made yet, in which case `free_start_ptr` and `free_end_ptr` must be
// dangling and identical and `block_footer_ptr` must be `None`.
pub(crate) struct Arena {
    /// The start of the free bytes of the current region. Since we grow down, this is always
    /// greater than or equal to `free_end_ptr`.
    //
    // INVARIANT: Must be within our current region (assuming there is one), be greater than or
    // equal to `free_end_ptr`, and not alias any previous allocations in the region. But keep in
    // mind that since we grow down, `free_start_ptr` actually has the highest address in the free
    // part of our current region, which means that it's at the edge of the previous allocation (or
    // the footer). This isn't aliasing since because we grow down, this is the one-past-the-end
    // pointer of a potential next allocation. When the next allocation happens, `free_start_ptr`
    // is decreased to account for the new allocation, which may or may not be touching the
    // previous allocation that follows it.
    free_start_ptr: Cell<NonNull<u8>>,

    /// The end of the free bytes of the current region.
    //
    // INVARIANT: Must be at the lowest address (since we grow down) in our current region
    // (assuming there is one).
    free_end_ptr: Cell<NonNull<u8>>,

    /// The footer of the current block.
    //
    // INVARIANT: Must refer to a correctly-initialized block footer of the current block and be
    // disjoint from the current region (assuming there is a block).
    block_footer_ptr: Cell<Option<NonNull<BlockFooter>>>,
}

#[repr(C, align(16))]
struct BlockFooter {
    /// The footer of the next block.
    //
    // INVARIANT: Must refer to a correctly-initialized block footer of the next block and be
    // derived from the allocation of the next block, with all provenance preserved (assuming
    // there is a next block).
    next_block_footer_ptr: Option<NonNull<Self>>,

    /// The beginning of the block.
    //
    // INVARIANT: Must be the pointer gotten from allocating the block.
    block_start_ptr: NonNull<u8>,
}

const _: () = assert!(is_aligned(size_of::<BlockFooter>(), MIN_REGION_ALIGNMENT));
const _: () = assert!(align_of::<BlockFooter>() <= MIN_REGION_ALIGNMENT);

impl Arena {
    pub(crate) fn new() -> Self {
        let dangling_ptr = NonNull::dangling();

        // SAFETY: `free_start_ptr` and `free_end_ptr` are dangling and identical, and
        // `block_footer_ptr` is `None`.
        unsafe {
            Arena {
                free_start_ptr: Cell::new(dangling_ptr),
                free_end_ptr: Cell::new(dangling_ptr),
                block_footer_ptr: Cell::new(None),
            }
        }
    }

    #[inline(always)]
    pub(super) fn allocate_sized<T>(&self) -> &mut MaybeUninit<T> {
        let layout = Layout::new::<T>();

        // SAFETY: `Layout::new` returns a layout that's padded to its alignment.
        let mut ptr = unsafe { self.allocate(layout) }.cast::<MaybeUninit<T>>();

        // SAFETY: `Self::allocate` returns a unique, aligned, dereferenceable pointer.
        unsafe { ptr.as_mut() }
    }

    #[inline(always)]
    pub(super) fn allocate_slice<T>(&self, len: usize) -> &mut [MaybeUninit<T>] {
        let layout = Layout::array::<T>(len).unwrap();

        // SAFETY: `Layout::array` returns a layout that's padded to its alignment.
        let ptr = unsafe { self.allocate(layout) }.cast::<MaybeUninit<T>>();

        let mut ptr = NonNull::slice_from_raw_parts(ptr, len);

        // SAFETY: `Self::allocate` returns a unique, aligned, dereferenceable pointer.
        unsafe { ptr.as_mut() }
    }

    #[inline(always)]
    pub(super) fn allocate_slice_copy<T>(&self, slice: &[T]) -> &mut [T] {
        let out = self.allocate_slice(slice.len());
        let out_ptr = out.as_mut_ptr().cast::<T>();

        unsafe { ptr::copy_nonoverlapping(slice.as_ptr(), out_ptr, slice.len()) };

        // SAFETY: We initialized the slice above.
        unsafe { slice_assume_init_mut(out) }
    }

    #[inline(always)]
    pub(super) fn allocate_str_copy(&self, s: &str) -> &mut str {
        let out = self.allocate_slice::<u8>(s.len());
        let out_ptr = out.as_mut_ptr().cast::<u8>();

        unsafe { ptr::copy_nonoverlapping(s.as_ptr(), out_ptr, s.len()) };

        // SAFETY: We initialized the slice above.
        let out = unsafe { slice_assume_init_mut(out) };

        // SAFETY: The bytes are the same as that of the input str, so they must be valid UTF-8.
        unsafe { str_from_utf8_unchecked_mut(out) }
    }

    /// # Safety
    ///
    /// - `layout` must be padded to its alignment.
    #[inline(always)]
    unsafe fn allocate(&self, layout: Layout) -> NonNull<u8> {
        // SAFETY: Enforced by the caller.
        if let Some(ptr) = unsafe { self.allocate_fast(layout) } {
            ptr
        } else {
            // SAFETY: Enforced by the caller.
            unsafe { self.allocate_slow(layout) }
        }
    }

    /// # Safety
    ///
    /// - `layout` must be padded to its alignment.
    #[inline(always)]
    unsafe fn allocate_fast(&self, layout: Layout) -> Option<NonNull<u8>> {
        debug_assert!(is_aligned(layout.size(), layout.align()));

        let free_start_ptr = self.free_start_ptr.get().as_ptr();
        let free_end_ptr = self.free_end_ptr.get().as_ptr();

        let addr = align_down(free_start_ptr.addr(), layout.align());

        // We don't need to align the size up because of our precondition that `layout` must be
        // padded to its alignment.
        let addr = addr.wrapping_sub(layout.size());

        let ptr = free_start_ptr.with_addr(addr);

        // Given the guarantee of `Layout`, this is equivalent to doing the naive
        // `(free_end_ptr..=free_start_ptr).contains(&ptr)` check.
        //
        // To prove it, we must first derive a few things. The guarantee of `Layout` states:
        //
        // > `layout.size()`, when rounded up to the nearest multiple of `layout.align()`, must not
        // > overflow `isize`.
        //
        // Let's establish the relationship between a layout's size and alignment by first looking
        // at the two extremes:
        //
        // - For a layout with an alignment of `1`, the maximum size is `isize::MAX`. The size can
        //   be any number since everything is aligned to `1`, but any greater size would trivially
        //   violate the guarantee.
        // - For a layout with an alignment of `0b1000...0000` (`1 << (usize::BITS - 1)`), the only
        //   allowed size is `0`. If the size were any larger, say `1`, then aligning it up to
        //   `0b1000...0000` would yield `0b1000...0000`, violating the guarantee.
        //
        // In general, the maximum size for a given alignment is given by:
        //
        // ```
        // max_size = align_down(isize::MAX as usize, alignment)
        // ```
        //
        // Or more visually:
        //
        // ```
        //            MSB                          LSB
        //             ▼                            ▼
        // alignment = 00000000 00010000 ... 00000000
        // max_size  = 01111111 11110000 ... 00000000
        //                          └───────┬───────┘
        //                           trailing zeroes
        // ```
        //
        // As you can see, the maximum size has to have as many trailing zeros as the alignment.
        // This is because if one of those zeroes were a one, aligning up would overflow all of the
        // other ones, resulting in `0b1000...0000` once again.
        //
        // The maximum distance between a given value and the value aligned up or down for a given
        // alignment is given by:
        //
        // ```
        // max_aligning_distance = alignment - 1
        // ```
        //
        // This is because aligning a value that's already aligned doesn't affect the value, but
        // every other value that isn't aligned is affected, and there are `alignment - 1` of those
        // values between an aligned value and the next closest aligned value.
        //
        // Now let us piece it together. Our algorithm first aligns `free_start_ptr` down to the
        // layout's alignment, then subtracts the layout's size, resulting in the allocated
        // pointer. That's a total maximum distance from `free_start_ptr` of:
        //
        // ```
        // allocation_distance = alignment - 1 + size
        // ```
        //
        // And thanks to the relationship between the size and alignment, we know that the maximum
        // distance from `free_start_ptr` is:
        //
        // ```
        // max_allocation_distance = isize::MAX as usize
        // ```
        //
        // We want to know the distance of the result of the aforementioned algorithm and
        // `free_end_ptr`, so we subtract `free_end_ptr` from the allocated pointer:
        //
        // ```
        // end_distance = allocated_ptr - free_end_ptr
        // ```
        //
        // If the allocated pointer is in bounds of the region, the end distance can be at most
        // `isize::MAX` because a region has to obey `Layout` too. If the allocated pointer is out
        // of bounds of the region, the end distance can be at most `isize::MAX` in the case that
        // `free_start_ptr` is equal to `free_end_ptr` and the the maximum allocation distance is
        // subtracted from `free_start_ptr`. In both cases, the maximum distance of the allocated
        // pointer from `free_end_ptr` is `isize::MAX`:
        //
        // ```
        // max_end_distance = isize::MAX as usize
        // ```
        //
        // And since we subtract `free_end_ptr` from the allocated pointer, the end distance is
        // positive if the allocated pointer is in bounds of the region and negative otherwise. Or
        // in unsigned integer math, the allocated pointer is in bounds of the region if the end
        // distance is less than or equal to `isize::MAX`. We know that the end distance can't wrap
        // around so much as to end up in the region again because the address space is
        // `isize::MAX as usize * 2 + 1` and our maximum region size and maximum end distance are
        // both `isize::MAX`.
        //
        // Q.E.D.
        //
        // Doing this instead of the naive `(free_end_ptr..=free_start_ptr).contains(&ptr)` check
        // allows this to compile down to just a subtraction, comparison, and jump instruction.
        let is_in_bounds_of_region = addr.wrapping_sub(free_end_ptr.addr()) <= isize::MAX as usize;

        if !is_in_bounds_of_region {
            return None;
        }

        // SAFETY: We checked that `ptr` is in bounds of the region.
        let ptr = unsafe { NonNull::new_unchecked(ptr) };

        // SAFETY: We made sure that `ptr` is in bounds of the region and that it doesn't point
        // inside any previous allocations, including the current one.
        unsafe { self.free_start_ptr.set(ptr) };

        Some(ptr)
    }

    /// # Safety
    ///
    /// - `layout` must be padded to its alignment.
    #[cold]
    unsafe fn allocate_slow(&self, layout: Layout) -> NonNull<u8> {
        let new_region_size = if let Some(ptr) = self.block_footer_ptr.get() {
            // SAFETY: By our own invariant, `self.block_footer_ptr` must be a valid pointer to an
            // initialized footer.
            let block_start_ptr = unsafe { ptr.as_ref() }.block_start_ptr;

            // SAFETY: By our own invariants, `self.block_start_ptr` and `block_footer_ptr` must be
            // derived from the same allocation, and `block_start_ptr` must be less than
            // `block_footer_ptr`.
            let region_size = unsafe { ptr.cast::<u8>().offset_from_unsigned(block_start_ptr) };

            // This cannot overflow because `region_size` cannot exceed `isize::MAX`.
            region_size * 2
        } else {
            INITIAL_REGION_SIZE
        };

        // We allocate all regions with an alignment of `MIN_REGION_ALIGNMENT`, so if we want to
        // guarantee that an allocation with a greater alignment fits in the new region, we have to
        // account for the scenario where the region is only aligned to `MIN_REGION_ALIGNMENT`, in
        // which case the following `Self::allocate_fast` would align down the `free_start_ptr`
        // (which will point one past the end of the region) to the allocation's alignment and
        // subtract the size of the allocation (which has to be padded to its alignment, remember).
        // The maximum aligning down is given by `layout.align() - MIN_REGION_ALIGNMENT`. So if we
        // allocate a region that fits the allocation's size plus this number, we guarantee that
        // the allocation fits in the region. This doesn't add any memory otherhead for allocations
        // that are `MIN_REGION_ALIGNMENT`-aligned or less (which is always true in our case).
        let min_region_size = if layout.align() > MIN_REGION_ALIGNMENT {
            // The addition can't overflow because of `Layout`'s guarantee. In fact, `layout.size()
            // + (layout.align() - 1)` can be at most `isize::MAX`.
            layout.size() + (layout.align() - MIN_REGION_ALIGNMENT)
        } else {
            // This can't overflow because of `Layout`'s guarantee too. `align_up(layout.size(),
            // arbitrary_alignment)` can be at most `isize::MAX as usize + 1`.
            align_up(layout.size(), MIN_REGION_ALIGNMENT)
        };

        let region_size = cmp::max(new_region_size, min_region_size);

        // We know this is true because there are 2 possible starting values for `region_size`:
        //
        // - `INITIAL_REGION_SIZE`, which is aligned to `MIN_REGION_ALIGNMENT`, or
        // - `min_region_size`, which has 2 further possibilities:
        //   - if `layout.align()` is greater than `MIN_REGION_ALIGNMENT`, `layout.size()`, which
        //     is more aligned than `MIN_REGION_ALIGNMENT` (because `layout` has to be padded to
        //     its alignment), plus `layout.align() - MIN_REGION_ALIGNMENT`, which is aligned to
        //     `MIN_REGION_ALIGNMENT`, or
        //   - `layout.size()` that we explicitly align to `MIN_REGION_ALIGNMENT`.
        //
        // This means that the initial region size is aligned to `MIN_REGION_ALIGNMENT`. The
        // next region size, then, has to either:
        //
        // - be this region size times 2, which is aligned to `MIN_REGION_ALIGNMENT`, or
        // - be `min_region_size`, which is also aligned to `MIN_REGION_ALIGNMENT` for the same
        //   reasons as before.
        debug_assert!(is_aligned(region_size, MIN_REGION_ALIGNMENT));

        // Nothing we can do about these checks.
        let block_size = region_size.checked_add(size_of::<BlockFooter>()).unwrap();
        let block_layout = Layout::from_size_align(block_size, MIN_REGION_ALIGNMENT).unwrap();

        // SAFETY: `block_layout` has a nonzero size.
        let Some(block_start_ptr) = NonNull::new(unsafe { alloc(block_layout) }) else {
            handle_alloc_error(block_layout);
        };

        // SAFETY: We allocated enough memory for the region and footer above.
        let block_footer_ptr = unsafe { block_start_ptr.add(region_size) };

        // SAFETY: `block_footer_ptr` is the start of our newly-allocated region.
        unsafe { self.free_start_ptr.set(block_footer_ptr) };

        // SAFETY: `block_start_ptr` is the end of our newly-allocated region.
        unsafe { self.free_end_ptr.set(block_start_ptr) };

        let block_footer_ptr = block_footer_ptr.cast::<BlockFooter>();

        let block_footer = BlockFooter {
            next_block_footer_ptr: self.block_footer_ptr.get(),
            block_start_ptr,
        };

        // SAFETY: We allocated enough memory for the region and footer above.
        unsafe { block_footer_ptr.write(block_footer) };

        // SAFETY: We have correctly initialized the block footer above.
        unsafe { self.block_footer_ptr.set(Some(block_footer_ptr)) };

        // This has to succeed because we made sure to allocate a large enough region.
        //
        // SAFETY: Enforced by the caller.
        unsafe { self.allocate_fast(layout) }.unwrap()
    }

    #[cfg(test)]
    fn block_count(&self) -> usize {
        let mut block_count = 0;
        let mut block_footer_ptr = self.block_footer_ptr.get();

        while let Some(ptr) = block_footer_ptr {
            block_count += 1;
            block_footer_ptr = unsafe { ptr.as_ref() }.next_block_footer_ptr;
        }

        block_count
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let mut block_footer_ptr = self.block_footer_ptr.get();

        while let Some(ptr) = block_footer_ptr {
            let block_start_ptr = unsafe { ptr.as_ref() }.block_start_ptr;
            let next_block_footer_ptr = unsafe { ptr.as_ref() }.next_block_footer_ptr;

            // SAFETY: By our own invariants, `block_start_ptr` and `self.block_footer_ptr` must be
            // derived from the same allocation, and `block_start_ptr` must be less than
            // `block_footer_ptr`.
            let region_size = unsafe { ptr.cast::<u8>().offset_from_unsigned(block_start_ptr) };

            // This cannot overflow because we've already allocated the block.
            let block_size = region_size + size_of::<BlockFooter>();

            let block_alignment = MIN_REGION_ALIGNMENT;

            // SAFETY: We've already allocated the block with this layout.
            let layout = unsafe { Layout::from_size_align_unchecked(block_size, block_alignment) };

            // SAFETY: By our own invariants, `block_start_ptr` must be the pointer gotten from
            // allocating the block. `layout` is the layout we allocated the block with.
            unsafe { dealloc(block_start_ptr.as_ptr(), layout) };

            block_footer_ptr = next_block_footer_ptr;
        }
    }
}

#[inline(always)]
const fn align_up(offset: usize, alignment: usize) -> usize {
    align_down(offset.wrapping_add(alignment.wrapping_sub(1)), alignment)
}

#[inline(always)]
const fn align_down(offset: usize, alignment: usize) -> usize {
    offset & !alignment.wrapping_sub(1)
}

#[inline(always)]
pub(super) const fn is_aligned(offset: usize, alignment: usize) -> bool {
    offset & alignment.wrapping_sub(1) == 0
}

#[inline(always)]
pub(super) unsafe fn slice_assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    // SAFETY:
    // - DST pointers with the same metadata are guaranteed to have the same representation.
    // - `MaybeUninit<T>` has the same layout as `T`. The caller must ensure that every element of
    //   the slice is initialized.
    unsafe { mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(slice) }
}

#[allow(clippy::transmute_bytes_to_str)]
#[inline(always)]
unsafe fn str_from_utf8_unchecked_mut(bytes: &mut [u8]) -> &mut str {
    // SAFETY:
    // - DST pointers with the same metadata are guaranteed to have the same representation.
    // - `[u8]` has the same layout as `str`. The caller must ensure that the bytes are valid
    //   UTF-8.
    unsafe { mem::transmute::<&mut [u8], &mut str>(bytes) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment() {
        let arena = Arena::new();

        for alignment_log2 in 0..24 {
            let alignment = 1 << alignment_log2;
            let layout = Layout::from_size_align(alignment, alignment).unwrap();

            let ptr = unsafe { arena.allocate(layout) };

            assert!(is_aligned(ptr.as_ptr().addr(), alignment));
        }
    }

    #[test]
    fn small_sizes() {
        let arena = Arena::new();

        for size_log2 in 0..12 {
            let size = 1 << size_log2;
            let layout = Layout::from_size_align(size, 1).unwrap();

            unsafe { arena.allocate(layout) };
        }

        assert_eq!(arena.block_count(), 1);
    }

    #[test]
    fn large_sizes() {
        let arena = Arena::new();

        for size_log2 in 12..24 {
            let size = 1 << size_log2;
            let layout = Layout::from_size_align(size, 1).unwrap();

            unsafe { arena.allocate(layout) };
        }

        assert_eq!(arena.block_count(), 12);
    }
}
