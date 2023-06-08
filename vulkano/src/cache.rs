// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use ahash::HashMap;
use parking_lot::RwLock;
use std::{
    collections::hash_map::Entry,
    hash::Hash,
    sync::{Arc, Weak},
};

/// A map specialized to caching properties that are specific to a Vulkan implementation.
///
/// Readers never block each other, except when an entry is vacant. In that case it gets written to
/// once and then never again, entries are immutable after insertion.
#[derive(Debug)]
pub(crate) struct OnceCache<K, V> {
    inner: RwLock<HashMap<K, V>>,
}

impl<K, V> Default for OnceCache<K, V> {
    fn default() -> Self {
        Self {
            inner: RwLock::new(HashMap::default()),
        }
    }
}

impl<K, V> OnceCache<K, V> {
    /// Creates a new `OnceCache`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V> OnceCache<K, V>
where
    K: Eq + Hash,
    V: Clone,
{
    /// Returns the value for the specified `key`. The entry gets written to with the value
    /// returned by `f` if it doesn't exist.
    pub fn get_or_insert(&self, key: K, f: impl FnOnce(&K) -> V) -> V {
        if let Some(value) = self.inner.read().get(&key) {
            return value.clone();
        }

        match self.inner.write().entry(key) {
            Entry::Occupied(entry) => {
                // This can happen if someone else inserted an entry between when we released
                // the read lock and acquired the write lock.
                entry.get().clone()
            }
            Entry::Vacant(entry) => {
                let value = f(entry.key());
                entry.insert(value.clone());
                value
            }
        }
    }

    /// Returns the value for the specified `key`. The entry gets written to with the value
    /// returned by `f` if it doesn't exist. If `f` returns [`Err`], the error is propagated and
    /// the entry isn't written to.
    pub fn get_or_try_insert<E>(&self, key: K, f: impl FnOnce(&K) -> Result<V, E>) -> Result<V, E> {
        if let Some(value) = self.inner.read().get(&key) {
            return Ok(value.clone());
        }

        match self.inner.write().entry(key) {
            Entry::Occupied(entry) => {
                // This can happen if someone else inserted an entry between when we released
                // the read lock and acquired the write lock.
                Ok(entry.get().clone())
            }
            Entry::Vacant(entry) => {
                let value = f(entry.key())?;
                entry.insert(value.clone());

                Ok(value)
            }
        }
    }
}

/// Like `OnceCache`, but the cache stores weak `Arc` references. If the weak reference cannot
/// be upgraded, then it acts as if the entry has become vacant again.
#[derive(Debug)]
pub(crate) struct WeakArcOnceCache<K, V> {
    inner: RwLock<HashMap<K, Weak<V>>>,
}

impl<K, V> Default for WeakArcOnceCache<K, V> {
    fn default() -> Self {
        Self {
            inner: RwLock::new(HashMap::default()),
        }
    }
}

impl<K, V> WeakArcOnceCache<K, V> {
    /// Creates a new `OnceCache`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V> WeakArcOnceCache<K, V>
where
    K: Eq + Hash,
{
    /// Returns the value for the specified `key`. The entry gets written to with the value
    /// returned by `f` if it doesn't exist.
    #[allow(dead_code)]
    pub fn get_or_insert(&self, key: K, f: impl FnOnce(&K) -> Arc<V>) -> Arc<V> {
        if let Some(arc) = self
            .inner
            .read()
            .get(&key)
            .and_then(|weak| Weak::upgrade(weak))
        {
            return arc;
        }

        match self.inner.write().entry(key) {
            Entry::Occupied(mut entry) => {
                if let Some(arc) = Weak::upgrade(entry.get()) {
                    // This can happen if someone else inserted an entry between when we released
                    // the read lock and acquired the write lock.
                    arc
                } else {
                    // The weak reference could not be upgraded, so create a new one.
                    let arc = f(entry.key());
                    entry.insert(Arc::downgrade(&arc));
                    arc
                }
            }
            Entry::Vacant(entry) => {
                let arc = f(entry.key());
                entry.insert(Arc::downgrade(&arc));
                arc
            }
        }
    }

    /// Returns the value for the specified `key`. The entry gets written to with the value
    /// returned by `f` if it doesn't exist. If `f` returns [`Err`], the error is propagated and
    /// the entry isn't written to.
    pub fn get_or_try_insert<E>(
        &self,
        key: K,
        f: impl FnOnce(&K) -> Result<Arc<V>, E>,
    ) -> Result<Arc<V>, E> {
        if let Some(arc) = self
            .inner
            .read()
            .get(&key)
            .and_then(|weak| Weak::upgrade(weak))
        {
            return Ok(arc);
        }

        match self.inner.write().entry(key) {
            Entry::Occupied(mut entry) => {
                if let Some(arc) = Weak::upgrade(entry.get()) {
                    // This can happen if someone else inserted an entry between when we released
                    // the read lock and acquired the write lock.
                    Ok(arc)
                } else {
                    // The weak reference could not be upgraded, so create a new one.
                    let arc = f(entry.key())?;
                    entry.insert(Arc::downgrade(&arc));
                    Ok(arc)
                }
            }
            Entry::Vacant(entry) => {
                let arc = f(entry.key())?;
                entry.insert(Arc::downgrade(&arc));
                Ok(arc)
            }
        }
    }
}
