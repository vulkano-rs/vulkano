#![allow(dead_code)]

use std::{
    cmp,
    cmp::Ordering,
    collections::{btree_map, BTreeMap},
    fmt::{Debug, Error as FmtError, Formatter},
    iter::{FromIterator, FusedIterator, Peekable},
    ops::{Add, Bound, Range, Sub},
};

/// A map whose keys are stored as (half-open) ranges bounded
/// inclusively below and exclusively above `(start..end)`.
///
/// Contiguous and overlapping ranges that map to the same value
/// are coalesced into a single range.
#[derive(Clone)]
pub struct RangeMap<K, V> {
    // Wrap ranges so that they are `Ord`.
    // See `range_wrapper.rs` for explanation.
    btm: BTreeMap<RangeStartWrapper<K>, V>,
}

impl<K, V> Default for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    /// Makes a new empty `RangeMap`.
    pub fn new() -> Self {
        RangeMap {
            btm: BTreeMap::new(),
        }
    }

    /// Returns a reference to the value corresponding to the given key,
    /// if the key is covered by any range in the map.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.get_key_value(key).map(|(_range, value)| value)
    }

    /// Returns the range-value pair (as a pair of references) corresponding
    /// to the given key, if the key is covered by any range in the map.
    pub fn get_key_value(&self, key: &K) -> Option<(&Range<K>, &V)> {
        // The only stored range that could contain the given key is the
        // last stored range whose start is less than or equal to this key.
        let key_as_start = RangeStartWrapper::new(key.clone()..key.clone());

        self.btm
            .range((Bound::Unbounded, Bound::Included(key_as_start)))
            .next_back()
            .filter(|(range_start_wrapper, _value)| {
                // Does the only candidate range contain
                // the requested key?
                range_start_wrapper.range.contains(key)
            })
            .map(|(range_start_wrapper, value)| (&range_start_wrapper.range, value))
    }

    /// Returns `true` if any range in the map covers the specified key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns `true` if any part of the provided range overlaps with a range in the map.
    #[inline]
    pub fn contains_any(&self, range: &Range<K>) -> bool {
        self.range(range).next().is_some()
    }

    /// Returns `true` if all of the provided range is covered by ranges in the map.
    #[inline]
    pub fn contains_all(&self, range: &Range<K>) -> bool {
        self.gaps(range).next().is_none()
    }

    /// Gets an iterator over all pairs of key range and value,
    /// ordered by key range.
    ///
    /// The iterator element type is `(&'a Range<K>, &'a V)`.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.btm.iter(),
        }
    }

    /// Gets a mutable iterator over all pairs of key range and value,
    /// ordered by key range.
    ///
    /// The iterator element type is `(&'a Range<K>, &'a mut V)`.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.btm.iter_mut(),
        }
    }

    /// Insert a pair of key range and value into the map.
    ///
    /// If the inserted range partially or completely overlaps any
    /// existing range in the map, then the existing range (or ranges) will be
    /// partially or completely replaced by the inserted range.
    ///
    /// If the inserted range either overlaps or is immediately adjacent
    /// any existing range _mapping to the same value_, then the ranges
    /// will be coalesced into a single contiguous range.
    ///
    /// # Panics
    ///
    /// Panics if range `start >= end`.
    pub fn insert(&mut self, range: Range<K>, value: V) {
        // We don't want to have to make empty ranges make sense;
        // they don't represent anything meaningful in this structure.
        assert!(range.start < range.end);

        // Wrap up the given range so that we can "borrow"
        // it as a wrapper reference to either its start or end.
        // See `range_wrapper.rs` for explanation of these hacks.
        let mut new_range_start_wrapper: RangeStartWrapper<K> = RangeStartWrapper::new(range);
        let new_value = value;

        // Is there a stored range either overlapping the start of
        // the range to insert or immediately preceding it?
        //
        // If there is any such stored range, it will be the last
        // whose start is less than or equal to the start of the range to insert,
        // or the one before that if both of the above cases exist.
        let mut candidates = self
            .btm
            .range((Bound::Unbounded, Bound::Included(&new_range_start_wrapper)))
            .rev()
            .take(2)
            .filter(|(stored_range_start_wrapper, _stored_value)| {
                // Does the candidate range either overlap
                // or immediately precede the range to insert?
                // (Remember that it might actually cover the _whole_
                // range to insert and then some.)
                stored_range_start_wrapper
                    .range
                    .touches(&new_range_start_wrapper.range)
            });

        if let Some(mut candidate) = candidates.next() {
            // Or the one before it if both cases described above exist.
            if let Some(another_candidate) = candidates.next() {
                candidate = another_candidate;
            }

            let (stored_range_start_wrapper, stored_value) =
                (candidate.0.clone(), candidate.1.clone());

            self.adjust_touching_ranges_for_insert(
                stored_range_start_wrapper,
                stored_value,
                &mut new_range_start_wrapper.range,
                &new_value,
            );
        }

        // Are there any stored ranges whose heads overlap or immediately
        // follow the range to insert?
        //
        // If there are any such stored ranges (that weren't already caught above),
        // their starts will fall somewhere after the start of the range to insert,
        // and on or before its end.
        //
        // This time around, if the latter holds, it also implies
        // the former so we don't need to check here if they touch.
        //
        // REVISIT: Possible micro-optimisation: `impl Borrow<T> for RangeStartWrapper<T>`
        // and use that to search here, to avoid constructing another `RangeStartWrapper`.
        let new_range_end_as_start = RangeStartWrapper::new(
            new_range_start_wrapper.range.end.clone()..new_range_start_wrapper.range.end.clone(),
        );

        while let Some((stored_range_start_wrapper, stored_value)) = self
            .btm
            .range((
                Bound::Included(&new_range_start_wrapper),
                Bound::Included(&new_range_end_as_start),
            ))
            .next()
        {
            // One extra exception: if we have different values,
            // and the stored range starts at the end of the range to insert,
            // then we don't want to keep looping forever trying to find more!
            #[allow(clippy::suspicious_operation_groupings)]
            if stored_range_start_wrapper.range.start == new_range_start_wrapper.range.end
                && *stored_value != new_value
            {
                // We're beyond the last stored range that could be relevant.
                // Avoid wasting time on irrelevant ranges, or even worse, looping forever.
                // (`adjust_touching_ranges_for_insert` below assumes that the given range
                // is relevant, and behaves very poorly if it is handed a range that it
                // shouldn't be touching.)
                break;
            }

            let stored_range_start_wrapper = stored_range_start_wrapper.clone();
            let stored_value = stored_value.clone();

            self.adjust_touching_ranges_for_insert(
                stored_range_start_wrapper,
                stored_value,
                &mut new_range_start_wrapper.range,
                &new_value,
            );
        }

        // Insert the (possibly expanded) new range, and we're done!
        self.btm.insert(new_range_start_wrapper, new_value);
    }

    /// Removes a range from the map, if all or any of it was present.
    ///
    /// If the range to be removed _partially_ overlaps any ranges
    /// in the map, then those ranges will be contracted to no
    /// longer cover the removed range.
    ///
    ///
    /// # Panics
    ///
    /// Panics if range `start >= end`.
    pub fn remove(&mut self, range: Range<K>) {
        // We don't want to have to make empty ranges make sense;
        // they don't represent anything meaningful in this structure.
        assert!(range.start < range.end);

        let range_start_wrapper: RangeStartWrapper<K> = RangeStartWrapper::new(range);
        let range = &range_start_wrapper.range;

        // Is there a stored range overlapping the start of
        // the range to insert?
        //
        // If there is any such stored range, it will be the last
        // whose start is less than or equal to the start of the range to insert.
        if let Some((stored_range_start_wrapper, stored_value)) = self
            .btm
            .range((Bound::Unbounded, Bound::Included(&range_start_wrapper)))
            .next_back()
            .filter(|(stored_range_start_wrapper, _stored_value)| {
                // Does the only candidate range overlap
                // the range to insert?
                stored_range_start_wrapper.range.overlaps(range)
            })
            .map(|(stored_range_start_wrapper, stored_value)| {
                (stored_range_start_wrapper.clone(), stored_value.clone())
            })
        {
            self.adjust_overlapping_ranges_for_remove(
                stored_range_start_wrapper,
                stored_value,
                range,
            );
        }

        // Are there any stored ranges whose heads overlap the range to insert?
        //
        // If there are any such stored ranges (that weren't already caught above),
        // their starts will fall somewhere after the start of the range to insert,
        // and before its end.
        //
        // REVISIT: Possible micro-optimisation: `impl Borrow<T> for RangeStartWrapper<T>`
        // and use that to search here, to avoid constructing another `RangeStartWrapper`.
        let new_range_end_as_start = RangeStartWrapper::new(range.end.clone()..range.end.clone());

        while let Some((stored_range_start_wrapper, stored_value)) = self
            .btm
            .range((
                Bound::Excluded(&range_start_wrapper),
                Bound::Excluded(&new_range_end_as_start),
            ))
            .next()
            .map(|(stored_range_start_wrapper, stored_value)| {
                (stored_range_start_wrapper.clone(), stored_value.clone())
            })
        {
            self.adjust_overlapping_ranges_for_remove(
                stored_range_start_wrapper,
                stored_value,
                range,
            );
        }
    }

    fn adjust_touching_ranges_for_insert(
        &mut self,
        stored_range_start_wrapper: RangeStartWrapper<K>,
        stored_value: V,
        new_range: &mut Range<K>,
        new_value: &V,
    ) {
        if stored_value == *new_value {
            // The ranges have the same value, so we can "adopt"
            // the stored range.
            //
            // This means that no matter how big or where the stored range is,
            // we will expand the new range's bounds to subsume it,
            // and then delete the stored range.
            new_range.start =
                cmp::min(&new_range.start, &stored_range_start_wrapper.range.start).clone();
            new_range.end = cmp::max(&new_range.end, &stored_range_start_wrapper.range.end).clone();
            self.btm.remove(&stored_range_start_wrapper);
        } else {
            // The ranges have different values.
            if new_range.overlaps(&stored_range_start_wrapper.range) {
                // The ranges overlap. This is a little bit more complicated.
                // Delete the stored range, and then add back between
                // 0 and 2 subranges at the ends of the range to insert.
                self.btm.remove(&stored_range_start_wrapper);
                if stored_range_start_wrapper.range.start < new_range.start {
                    // Insert the piece left of the range to insert.
                    self.btm.insert(
                        RangeStartWrapper::new(
                            stored_range_start_wrapper.range.start..new_range.start.clone(),
                        ),
                        stored_value.clone(),
                    );
                }
                if stored_range_start_wrapper.range.end > new_range.end {
                    // Insert the piece right of the range to insert.
                    self.btm.insert(
                        RangeStartWrapper::new(
                            new_range.end.clone()..stored_range_start_wrapper.range.end,
                        ),
                        stored_value,
                    );
                }
            } else {
                // No-op; they're not overlapping,
                // so we can just keep both ranges as they are.
            }
        }
    }

    fn adjust_overlapping_ranges_for_remove(
        &mut self,
        stored_range_start_wrapper: RangeStartWrapper<K>,
        stored_value: V,
        range_to_remove: &Range<K>,
    ) {
        // Delete the stored range, and then add back between
        // 0 and 2 subranges at the ends of the range to insert.
        self.btm.remove(&stored_range_start_wrapper);
        let stored_range = stored_range_start_wrapper.range;

        if stored_range.start < range_to_remove.start {
            // Insert the piece left of the range to insert.
            self.btm.insert(
                RangeStartWrapper::new(stored_range.start..range_to_remove.start.clone()),
                stored_value.clone(),
            );
        }

        if stored_range.end > range_to_remove.end {
            // Insert the piece right of the range to insert.
            self.btm.insert(
                RangeStartWrapper::new(range_to_remove.end.clone()..stored_range.end),
                stored_value,
            );
        }
    }

    /// Splits a range in two at the provided key.
    ///
    /// Does nothing if no range exists at the key, or if the key is at a range boundary.
    pub fn split_at(&mut self, key: &K) {
        // Find a range that contains the key, but doesn't start or end with the key.
        let bounds = (
            Bound::Unbounded,
            Bound::Excluded(RangeStartWrapper::new(key.clone()..key.clone())),
        );

        let range_to_remove = match self
            .btm
            .range(bounds)
            .next_back()
            .filter(|(range_start_wrapper, _value)| range_start_wrapper.range.contains(key))
        {
            Some((k, _v)) => k.clone(),
            None => return,
        };

        // Remove the range, and re-insert two new ranges with the same value, separated by the key.
        let value = self.btm.remove(&range_to_remove).unwrap();
        self.btm.insert(
            RangeStartWrapper::new(range_to_remove.range.start..key.clone()),
            value.clone(),
        );
        self.btm.insert(
            RangeStartWrapper::new(key.clone()..range_to_remove.range.end),
            value,
        );
    }

    /// Gets an iterator over all pairs of key range and value, where the key range overlaps with
    /// the provided range.
    ///
    /// The iterator element type is `(&Range<K>, &V)`.
    pub fn range(&self, range: &Range<K>) -> RangeIter<'_, K, V> {
        let start = self
            .get_key_value(&range.start)
            .map_or(&range.start, |(k, _v)| &k.start);
        let end = &range.end;

        RangeIter {
            inner: self.btm.range((
                Bound::Included(RangeStartWrapper::new(start.clone()..start.clone())),
                Bound::Excluded(RangeStartWrapper::new(end.clone()..end.clone())),
            )),
        }
    }

    /// Gets a mutable iterator over all pairs of key range and value, where the key range overlaps
    /// with the provided range.
    ///
    /// The iterator element type is `(&Range<K>, &mut V)`.
    pub fn range_mut(&mut self, range: &Range<K>) -> RangeMutIter<'_, K, V> {
        let start = self
            .get_key_value(&range.start)
            .map_or(&range.start, |(k, _v)| &k.start);
        let end = &range.end;
        let bounds = (
            Bound::Included(RangeStartWrapper::new(start.clone()..start.clone())),
            Bound::Excluded(RangeStartWrapper::new(end.clone()..end.clone())),
        );

        RangeMutIter {
            inner: self.btm.range_mut(bounds),
        }
    }

    /// Gets an iterator over all the maximally-sized ranges
    /// contained in `outer_range` that are not covered by
    /// any range stored in the map.
    ///
    /// The iterator element type is `Range<K>`.
    ///
    /// NOTE: Calling `gaps` eagerly finds the first gap,
    /// even if the iterator is never consumed.
    pub fn gaps<'a>(&'a self, outer_range: &'a Range<K>) -> Gaps<'a, K, V> {
        let mut keys = self.btm.keys().peekable();
        // Find the first potential gap.
        let mut candidate_start = &outer_range.start;

        while let Some(item) = keys.peek() {
            if item.range.end <= outer_range.start {
                // This range sits entirely before the start of
                // the outer range; just skip it.
                let _ = keys.next();
            } else if item.range.start <= outer_range.start {
                // This range overlaps the start of the
                // outer range, so the first possible candidate
                // range begins at its end.
                candidate_start = &item.range.end;
                let _ = keys.next();
            } else {
                // The rest of the items might contribute to gaps.
                break;
            }
        }

        Gaps {
            outer_range,
            keys,
            candidate_start,
        }
    }
}

/// An iterator over the entries of a `RangeMap`, ordered by key range.
///
/// The iterator element type is `(&'a Range<K>, &'a V)`.
///
/// This `struct` is created by the [`iter`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`iter`]: RangeMap::iter
pub struct Iter<'a, K, V> {
    inner: btree_map::Iter<'a, RangeStartWrapper<K>, V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    type Item = (&'a Range<K>, &'a V);

    fn next(&mut self) -> Option<(&'a Range<K>, &'a V)> {
        self.inner.next().map(|(by_start, v)| (&by_start.range, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> where K: Ord + Clone {}
impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> where K: Ord + Clone {}

/// An iterator over the entries of a `RangeMap`, ordered by key range.
///
/// The iterator element type is `(&'a Range<K>, &'a V)`.
///
/// This `struct` is created by the [`iter`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`iter`]: RangeMap::iter
pub struct IterMut<'a, K, V> {
    inner: btree_map::IterMut<'a, RangeStartWrapper<K>, V>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    type Item = (&'a Range<K>, &'a mut V);

    fn next(&mut self) -> Option<(&'a Range<K>, &'a mut V)> {
        self.inner.next().map(|(by_start, v)| (&by_start.range, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V> where K: Ord + Clone {}
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> where K: Ord + Clone {}

/// An owning iterator over the entries of a `RangeMap`, ordered by key range.
///
/// The iterator element type is `(Range<K>, V)`.
///
/// This `struct` is created by the [`into_iter`] method on [`RangeMap`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
pub struct IntoIter<K, V> {
    inner: btree_map::IntoIter<RangeStartWrapper<K>, V>,
}

impl<K, V> IntoIterator for RangeMap<K, V> {
    type Item = (Range<K>, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.btm.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (Range<K>, V);

    fn next(&mut self) -> Option<(Range<K>, V)> {
        self.inner.next().map(|(by_start, v)| (by_start.range, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> where K: Ord + Clone {}
impl<K, V> ExactSizeIterator for IntoIter<K, V> where K: Ord + Clone {}

// We can't just derive this automatically, because that would
// expose irrelevant (and private) implementation details.
// Instead implement it in the same way that the underlying BTreeMap does.
impl<K: Debug, V: Debug> Debug for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> FromIterator<(Range<K>, V)> for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn from_iter<T: IntoIterator<Item = (Range<K>, V)>>(iter: T) -> Self {
        let mut range_map = RangeMap::new();
        range_map.extend(iter);
        range_map
    }
}

impl<K, V> Extend<(Range<K>, V)> for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn extend<T: IntoIterator<Item = (Range<K>, V)>>(&mut self, iter: T) {
        iter.into_iter().for_each(move |(k, v)| {
            self.insert(k, v);
        })
    }
}

/// An iterator over entries of a `RangeMap` whose range overlaps with a specified range.
///
/// The iterator element type is `(&'a Range<K>, &'a V)`.
///
/// This `struct` is created by the [`range`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`range`]: RangeMap::range
pub struct RangeIter<'a, K, V> {
    inner: btree_map::Range<'a, RangeStartWrapper<K>, V>,
}

impl<'a, K, V> FusedIterator for RangeIter<'a, K, V> where K: Ord + Clone {}

impl<'a, K, V> Iterator for RangeIter<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    type Item = (&'a Range<K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(by_start, v)| (&by_start.range, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// A mutable iterator over entries of a `RangeMap` whose range overlaps with a specified range.
///
/// The iterator element type is `(&'a Range<K>, &'a mut V)`.
///
/// This `struct` is created by the [`range_mut`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`range_mut`]: RangeMap::range_mut
pub struct RangeMutIter<'a, K, V> {
    inner: btree_map::RangeMut<'a, RangeStartWrapper<K>, V>,
}

impl<'a, K, V> FusedIterator for RangeMutIter<'a, K, V> where K: Ord + Clone {}

impl<'a, K, V> Iterator for RangeMutIter<'a, K, V>
where
    K: 'a,
    V: 'a,
{
    type Item = (&'a Range<K>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(by_start, v)| (&by_start.range, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// An iterator over all ranges not covered by a `RangeMap`.
///
/// The iterator element type is `Range<K>`.
///
/// This `struct` is created by the [`gaps`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`gaps`]: RangeMap::gaps
pub struct Gaps<'a, K, V> {
    outer_range: &'a Range<K>,
    keys: Peekable<btree_map::Keys<'a, RangeStartWrapper<K>, V>>,
    candidate_start: &'a K,
}

// `Gaps` is always fused. (See definition of `next` below.)
impl<'a, K, V> FusedIterator for Gaps<'a, K, V> where K: Ord + Clone {}

impl<'a, K, V> Iterator for Gaps<'a, K, V>
where
    K: Ord + Clone,
{
    type Item = Range<K>;

    fn next(&mut self) -> Option<Self::Item> {
        if *self.candidate_start >= self.outer_range.end {
            // We've already passed the end of the outer range;
            // there are no more gaps to find.
            return None;
        }

        // Figure out where this gap ends.
        let (gap_end, mut next_candidate_start) = if let Some(next_item) = self.keys.next() {
            if next_item.range.start < self.outer_range.end {
                // The gap goes up until the start of the next item,
                // and the next candidate starts after it.
                (&next_item.range.start, &next_item.range.end)
            } else {
                // The item sits after the end of the outer range,
                // so this gap ends at the end of the outer range.
                // This also means there will be no more gaps.
                (&self.outer_range.end, &self.outer_range.end)
            }
        } else {
            // There's no next item; the end is at the
            // end of the outer range.
            // This also means there will be no more gaps.
            (&self.outer_range.end, &self.outer_range.end)
        };

        // Find the start of the next gap.
        while let Some(next_item) = self.keys.peek() {
            if next_item.range.start == *next_candidate_start {
                // There's another item at the start of our candidate range.
                // Gaps can't have zero width, so skip to the end of this
                // item and try again.
                next_candidate_start = &next_item.range.end;
                self.keys.next().expect("We just checked that this exists");
            } else {
                // We found an item that actually has a gap before it.
                break;
            }
        }

        // Move the next candidate gap start past the end
        // of this gap, and yield the gap we found.
        let gap = self.candidate_start.clone()..gap_end.clone();
        self.candidate_start = next_candidate_start;
        Some(gap)
    }
}

// Wrappers to allow storing (and sorting/searching)
// ranges as the keys of a `BTreeMap`.
//
// We can do this because we maintain the invariants
// that the order of range starts is the same as the order
// of range ends, and that no two stored ranges have the
// same start or end as each other.
//
// NOTE: Be very careful not to accidentally use these
// if you really do want to compare equality of the
// inner range!

//
// Range start wrapper
//

#[derive(Eq, Debug, Clone)]
pub struct RangeStartWrapper<T> {
    pub range: Range<T>,
}

impl<T> RangeStartWrapper<T> {
    pub fn new(range: Range<T>) -> RangeStartWrapper<T> {
        RangeStartWrapper { range }
    }
}

impl<T> PartialEq for RangeStartWrapper<T>
where
    T: Eq,
{
    fn eq(&self, other: &RangeStartWrapper<T>) -> bool {
        self.range.start == other.range.start
    }
}

impl<T> Ord for RangeStartWrapper<T>
where
    T: Ord,
{
    fn cmp(&self, other: &RangeStartWrapper<T>) -> Ordering {
        self.range.start.cmp(&other.range.start)
    }
}

impl<T> PartialOrd for RangeStartWrapper<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &RangeStartWrapper<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub trait RangeExt<T> {
    fn overlaps(&self, other: &Self) -> bool;
    fn touches(&self, other: &Self) -> bool;
}

impl<T> RangeExt<T> for Range<T>
where
    T: Ord,
{
    fn overlaps(&self, other: &Self) -> bool {
        // Strictly less than, because ends are excluded.
        cmp::max(&self.start, &other.start) < cmp::min(&self.end, &other.end)
    }

    fn touches(&self, other: &Self) -> bool {
        // Less-than-or-equal-to because if one end is excluded, the other is included.
        // I.e. the two could be joined into a single range, because they're overlapping
        // or immediately adjacent.
        cmp::max(&self.start, &other.start) <= cmp::min(&self.end, &other.end)
    }
}

/// Minimal version of unstable [`Step`](std::iter::Step) trait
/// from the Rust standard library.
///
/// This is needed for [`RangeInclusiveMap`](crate::RangeInclusiveMap)
/// because ranges stored as its keys interact with each other
/// when the start of one is _adjacent_ the end of another.
/// I.e. we need a concept of successor values rather than just
/// equality, and that is what `Step` will
/// eventually provide once it is stabilized.
///
/// **NOTE:** This will likely be deprecated and then eventually
/// removed once the standard library's `Step`
/// trait is stabilised, as most crates will then likely implement `Step`
/// for their types where appropriate.
///
/// See [this issue](https://github.com/rust-lang/rust/issues/42168)
/// for details about that stabilization process.
pub trait StepLite {
    /// Returns the _successor_ of `self`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this function is allowed to panic, wrap, or saturate.
    /// The suggested behavior is to panic when debug assertions are enabled,
    /// and to wrap or saturate otherwise.
    fn add_one(&self) -> Self;

    /// Returns the _predecessor_ of `self`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this function is allowed to panic, wrap, or saturate.
    /// The suggested behavior is to panic when debug assertions are enabled,
    /// and to wrap or saturate otherwise.
    fn sub_one(&self) -> Self;
}

// Implement for all common integer types.
macro_rules! impl_step_lite {
    ($($t:ty)*) => ($(
        impl StepLite for $t {
            #[inline]
            fn add_one(&self) -> Self {
                Add::add(*self, 1)
            }

            #[inline]
            fn sub_one(&self) -> Self {
                Sub::sub(*self, 1)
            }
        }
    )*)
}

impl_step_lite!(usize u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

// TODO: When on nightly, a blanket implementation for
// all types that implement `std::iter::Step` instead
// of the auto-impl above.

/// Successor and predecessor functions defined for `T`,
/// but as free functions rather than methods on `T` itself.
///
/// This is useful as a workaround for Rust's "orphan rules",
/// which prevent you from implementing [`StepLite`](crate::StepLite) for `T` if `T`
/// is a foreign type.
///
/// **NOTE:** This will likely be deprecated and then eventually
/// removed once the standard library's [`Step`](std::iter::Step)
/// trait is stabilised, as most crates will then likely implement `Step`
/// for their types where appropriate.
///
/// See [this issue](https://github.com/rust-lang/rust/issues/42168)
/// for details about that stabilization process.
///
/// There is also a blanket implementation of `StepFns` for all
/// types implementing `StepLite`. Consumers of this crate should
/// prefer to implement `StepLite` for their own types, and only
/// fall back to `StepFns` when dealing with foreign types.
pub trait StepFns<T> {
    /// Returns the _successor_ of value `start`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this function is allowed to panic, wrap, or saturate.
    /// The suggested behavior is to panic when debug assertions are enabled,
    /// and to wrap or saturate otherwise.
    fn add_one(start: &T) -> T;

    /// Returns the _predecessor_ of value `start`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this function is allowed to panic, wrap, or saturate.
    /// The suggested behavior is to panic when debug assertions are enabled,
    /// and to wrap or saturate otherwise.
    fn sub_one(start: &T) -> T;
}

impl<T> StepFns<T> for T
where
    T: StepLite,
{
    fn add_one(start: &T) -> T {
        start.add_one()
    }

    fn sub_one(start: &T) -> T {
        start.sub_one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{format, vec, vec::Vec};

    trait RangeMapExt<K, V> {
        fn to_vec(&self) -> Vec<(Range<K>, V)>;
    }

    impl<K, V> RangeMapExt<K, V> for RangeMap<K, V>
    where
        K: Ord + Clone,
        V: Eq + Clone,
    {
        fn to_vec(&self) -> Vec<(Range<K>, V)> {
            self.iter().map(|(kr, v)| (kr.clone(), v.clone())).collect()
        }
    }

    //
    // Insertion tests
    //

    #[test]
    fn empty_map_is_empty() {
        let range_map: RangeMap<u32, bool> = RangeMap::new();
        assert_eq!(range_map.to_vec(), vec![]);
    }

    #[test]
    fn insert_into_empty_map() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(0..50, false);
        assert_eq!(range_map.to_vec(), vec![(0..50, false)]);
    }

    #[test]
    fn new_same_value_immediately_following_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_immediately_following_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..3, false), (3..5, true)]);
    }

    #[test]
    fn new_same_value_overlapping_end_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-----◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_overlapping_end_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-----◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..3, false), (3..5, true)]);
    }

    #[test]
    fn new_same_value_immediately_preceding_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_immediately_preceding_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..3, false), (3..5, true)]);
    }

    #[test]
    fn new_same_value_wholly_inside_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        range_map.insert(1..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(2..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_wholly_inside_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-------◇ ◌ ◌ ◌ ◌
        range_map.insert(1..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(2..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◌ ●-◌ ◌ ◌ ◌ ◌
        assert_eq!(
            range_map.to_vec(),
            vec![(1..2, true), (2..4, false), (4..5, true)]
        );
    }

    #[test]
    fn replace_at_end_of_existing_range_should_coalesce() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    //
    // Get* tests
    //

    #[test]
    fn get() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(0..50, false);
        assert_eq!(range_map.get(&49), Some(&false));
        assert_eq!(range_map.get(&50), None);
    }

    #[test]
    fn get_key_value() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(0..50, false);
        assert_eq!(range_map.get_key_value(&49), Some((&(0..50), &false)));
        assert_eq!(range_map.get_key_value(&50), None);
    }

    //
    // Removal tests
    //

    #[test]
    fn remove_from_empty_map() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.remove(0..50);
        assert_eq!(range_map.to_vec(), vec![]);
    }

    #[test]
    fn remove_non_covered_range_before_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(0..25);
        assert_eq!(range_map.to_vec(), vec![(25..75, false)]);
    }

    #[test]
    fn remove_non_covered_range_after_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(75..100);
        assert_eq!(range_map.to_vec(), vec![(25..75, false)]);
    }

    #[test]
    fn remove_overlapping_start_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(0..30);
        assert_eq!(range_map.to_vec(), vec![(30..75, false)]);
    }

    #[test]
    fn remove_middle_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(30..70);
        assert_eq!(range_map.to_vec(), vec![(25..30, false), (70..75, false)]);
    }

    #[test]
    fn remove_overlapping_end_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(70..100);
        assert_eq!(range_map.to_vec(), vec![(25..70, false)]);
    }

    #[test]
    fn remove_exactly_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(25..75);
        assert_eq!(range_map.to_vec(), vec![]);
    }

    #[test]
    fn remove_superset_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(0..100);
        assert_eq!(range_map.to_vec(), vec![]);
    }

    // Gaps tests

    #[test]
    fn whole_range_is_a_gap() {
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌
        let range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-------------◇ ◌
        let outer_range = 1..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield the entire outer range.
        assert_eq!(gaps.next(), Some(1..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn whole_range_is_covered_exactly() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---------◌ ◌ ◌ ◌
        range_map.insert(1..6, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆---------◇ ◌ ◌ ◌
        let outer_range = 1..6;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield no gaps.
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_before_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◆-----◇ ◌
        let outer_range = 5..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield the entire outer range.
        assert_eq!(gaps.next(), Some(5..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_touching_start_of_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        range_map.insert(1..5, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◆-----◇ ◌
        let outer_range = 5..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield the entire outer range.
        assert_eq!(gaps.next(), Some(5..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_overlapping_start_of_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---------◌ ◌ ◌ ◌
        range_map.insert(1..6, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◆-----◇ ◌
        let outer_range = 5..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield from the end of the stored item
        // to the end of the outer range.
        assert_eq!(gaps.next(), Some(6..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_starting_at_start_of_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ●-◌ ◌ ◌ ◌
        range_map.insert(5..6, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◆-----◇ ◌
        let outer_range = 5..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield from the item onwards.
        assert_eq!(gaps.next(), Some(6..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn items_floating_inside_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ●-◌ ◌ ◌ ◌
        range_map.insert(5..6, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●-◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(3..4, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-------------◇ ◌
        let outer_range = 1..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield gaps at start, between items,
        // and at end.
        assert_eq!(gaps.next(), Some(1..3));
        assert_eq!(gaps.next(), Some(4..5));
        assert_eq!(gaps.next(), Some(6..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_ending_at_end_of_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◌ ◌ ●-◌ ◌
        range_map.insert(7..8, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◆-----◇ ◌
        let outer_range = 5..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield from the start of the outer range
        // up to the start of the stored item.
        assert_eq!(gaps.next(), Some(5..7));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_overlapping_end_of_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ●---◌ ◌ ◌ ◌
        range_map.insert(4..6, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◆-----◇ ◌ ◌ ◌ ◌
        let outer_range = 2..5;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield from the start of the outer range
        // up to the start of the stored item.
        assert_eq!(gaps.next(), Some(2..4));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_touching_end_of_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ●-------◌ ◌
        range_map.insert(4..8, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-----◇ ◌ ◌ ◌ ◌ ◌
        let outer_range = 1..4;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield the entire outer range.
        assert_eq!(gaps.next(), Some(1..4));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn item_after_outer_range() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◌ ●---◌ ◌
        range_map.insert(6..7, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-----◇ ◌ ◌ ◌ ◌ ◌
        let outer_range = 1..4;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield the entire outer range.
        assert_eq!(gaps.next(), Some(1..4));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn empty_outer_range_with_items_away_from_both_sides() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆---◇ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◌ ◆---◇ ◌ ◌
        range_map.insert(5..7, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◆ ◌ ◌ ◌ ◌ ◌
        let outer_range = 4..4;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield no gaps.
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn empty_outer_range_with_items_touching_both_sides() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(2..4, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌
        range_map.insert(4..6, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◆ ◌ ◌ ◌ ◌ ◌
        let outer_range = 4..4;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield no gaps.
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn empty_outer_range_with_item_straddling() {
        let mut range_map: RangeMap<u32, ()> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◆-----◇ ◌ ◌ ◌ ◌ ◌
        range_map.insert(2..5, ());
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ◆ ◌ ◌ ◌ ◌ ◌
        let outer_range = 4..4;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield no gaps.
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    #[test]
    fn no_empty_gaps() {
        // Make two ranges different values so they don't
        // get coalesced.
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◌ ●-◌ ◌ ◌ ◌ ◌
        range_map.insert(4..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●-◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(3..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-------------◇ ◌
        let outer_range = 1..8;
        let mut gaps = range_map.gaps(&outer_range);
        // Should yield gaps at start and end, but not between the
        // two touching items. (4 is covered, so there should be no gap.)
        assert_eq!(gaps.next(), Some(1..3));
        assert_eq!(gaps.next(), Some(5..8));
        assert_eq!(gaps.next(), None);
        // Gaps iterator should be fused.
        assert_eq!(gaps.next(), None);
        assert_eq!(gaps.next(), None);
    }

    ///
    /// impl Debug
    ///

    #[test]
    fn map_debug_repr_looks_right() {
        let mut map: RangeMap<u32, ()> = RangeMap::new();

        // Empty
        assert_eq!(format!("{:?}", map), "{}");

        // One entry
        map.insert(2..5, ());
        assert_eq!(format!("{:?}", map), "{2..5: ()}");

        // Many entries
        map.insert(6..7, ());
        map.insert(8..9, ());
        assert_eq!(format!("{:?}", map), "{2..5: (), 6..7: (), 8..9: ()}");
    }

    // Iterator Tests

    #[test]
    fn into_iter_matches_iter() {
        // Just use vec since that's the same implementation we'd expect
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(1..3, false);
        range_map.insert(3..5, true);

        let cloned = range_map.to_vec();
        let consumed = range_map.into_iter().collect::<Vec<_>>();

        // Correct value
        assert_eq!(cloned, vec![(1..3, false), (3..5, true)]);

        // Equality
        assert_eq!(cloned, consumed);
    }
}
