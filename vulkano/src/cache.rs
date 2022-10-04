use ahash::HashMap;
use parking_lot::RwLock;
use std::{collections::hash_map::Entry, hash::Hash};

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
        OnceCache {
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
