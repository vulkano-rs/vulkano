use std::sync;

use Fence;

/// Defines a strategy for read-write access and relation with fences.
pub trait Lock<T> {
    fn shared_access(&self);
    fn exclusive_access(&self);

    fn shared_access_until(&self, Fence);
    fn exclusive_access_until(&self, Fence);
}

pub struct Mutex<T> {
    inner: sync::Mutex<T>,
    fence: Option<Fence>,
}

impl<T> Mutex<T> {
    pub fn lock(&self) -> Result<MutexLock<T>, MutexlockError> {
        self.fence.wait();
        let inner = try!(self.inner.lock());
    }
}

impl<T> Lock<T> for Mutex<T> {

}

pub struct MutexLock<'a, T> {
    inner: sync::LockGuard<'a, T>,
}
