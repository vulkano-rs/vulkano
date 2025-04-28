use std::{
    fmt,
    ops::{Deref, DerefMut},
    panic::{RefUnwindSafe, UnwindSafe},
    ptr::NonNull,
};

#[repr(transparent)]
pub struct AliasableBox<T> {
    ptr: NonNull<T>,
}

unsafe impl<T: Send> Send for AliasableBox<T> {}
unsafe impl<T: Sync> Sync for AliasableBox<T> {}

impl<T: UnwindSafe> UnwindSafe for AliasableBox<T> {}
impl<T: RefUnwindSafe> RefUnwindSafe for AliasableBox<T> {}

impl<T> Unpin for AliasableBox<T> {}

impl<T> AliasableBox<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        let ptr = Box::into_raw(Box::new(value));
        let ptr = unsafe { NonNull::new_unchecked(ptr) };

        AliasableBox { ptr }
    }
}

impl<T: fmt::Debug> fmt::Debug for AliasableBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T> Deref for AliasableBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> DerefMut for AliasableBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Drop for AliasableBox<T> {
    #[inline]
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.ptr.as_ptr()) });
    }
}
