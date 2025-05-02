use std::{
    fmt,
    ops::{Deref, DerefMut},
    panic::{RefUnwindSafe, UnwindSafe},
    ptr::NonNull,
};

macro_rules! self_referential {
    (
        mod $module_name:ident {
            $vis:vis struct $struct_name:ident {
                inner: $borrower_name:ident<'_>,
                $($owner_field_name:ident: $owner_ty:ty,)+
            }

            $(impl $($auto_impl:ident),+)?
        }
    ) => {
        crate::self_referential::self_referential!(
            @unsafe
            $module_name,
            $vis,
            $struct_name,
            $borrower_name<'_>,
            $borrower_name<'static>,
            ($($owner_field_name: $owner_ty,)+),
            impl for<'a> FnOnce($(&'a $owner_ty,)+) -> $borrower_name<'a>,
            $borrower_name<'_>,
            ($($($auto_impl,)+)?),
        );
    };
    (
        mod $module_name:ident {
            $vis:vis struct $struct_name:ident {
                inner: Vec<$borrower_name:ident<'_>>,
                $($owner_field_name:ident: $owner_ty:ty,)+
            }

            $(impl $($auto_impl:ident),+)?
        }
    ) => {
        crate::self_referential::self_referential!(
            @unsafe
            $module_name,
            $vis,
            $struct_name,
            Vec<$borrower_name<'_>>,
            Vec<$borrower_name<'static>>,
            ($($owner_field_name: $owner_ty,)+),
            impl for<'a> FnOnce($(&'a $owner_ty,)+) -> Vec<$borrower_name<'a>>,
            [$borrower_name<'_>],
            ($($($auto_impl,)+)?),
        );
    };
    (
        mod $module_name:ident {
            $vis:vis struct $struct_name:ident {
                inner: Vec<&'_ $borrowed_ty:ty>,
                $($owner_field_name:ident: $owner_ty:ty,)+
            }

            $(impl $($auto_impl:ident),+)?
        }
    ) => {
        crate::self_referential::self_referential!(
            @unsafe
            $module_name,
            $vis,
            $struct_name,
            Vec<&'_ $borrowed_ty>,
            Vec<&'static $borrowed_ty>,
            ($($owner_field_name: $owner_ty,)+),
            impl for<'a> FnOnce($(&'a $owner_ty,)+) -> Vec<&'a $borrowed_ty>,
            [&'_ $borrowed_ty],
            ($($($auto_impl,)+)?),
        );
    };
    (
        mod $module_name:ident {
            $vis:vis struct $struct_name:ident {
                inner: SmallVec<[$borrower_name:ident<'_>; $n:literal]>,
                $($owner_field_name:ident: $owner_ty:ty,)+
            }

            $(impl $($auto_impl:ident),+)?
        }
    ) => {
        crate::self_referential::self_referential!(
            @unsafe
            $module_name,
            $vis,
            $struct_name,
            SmallVec<[$borrower_name<'_>; $n]>,
            SmallVec<[$borrower_name<'static>; $n]>,
            ($($owner_field_name: $owner_ty,)+),
            impl for<'a> FnOnce($(&'a $owner_ty,)+) -> SmallVec<[$borrower_name<'a>; $n]>,
            [$borrower_name<'_>],
            ($($($auto_impl,)+)?),
        );
    };
    (
        @unsafe
        $module_name:ident,
        $vis:vis,
        $struct_name:ident,
        $borrower_ty:ty,
        $erased_borrower_ty:ty,
        ($($owner_field_name:ident: $owner_ty:ty,)+),
        $make_inner_ty:ty,
        $referent_ty:ty,
        ($($auto_impl:ident,)*),
    ) => {
        mod $module_name {
            #[allow(unused_imports)]
            use super::*;

            pub struct $struct_name {
                // The static lifetime is A LIE. We must not expose this fake lifetime to user code.
                inner: $erased_borrower_ty,
                // These fields must remain untouched until the outer struct is dropped.
                #[allow(dead_code)]
                owners: crate::self_referential::AliasableBox<Owners>,
            }

            #[allow(dead_code)]
            struct Owners {
                $($owner_field_name: $owner_ty,)+
            }

            impl $struct_name {
                #[inline]
                pub fn new(
                    $($owner_field_name: $owner_ty,)+
                    make_inner: $make_inner_ty,
                ) -> Self {
                    let owners = crate::self_referential::AliasableBox::new(Owners {
                        $($owner_field_name,)+
                    });

                    let inner = make_inner($(&owners.$owner_field_name,)+);

                    // SAFETY: The borrower can only borrow from the heap thanks to the owners being
                    // wrapped in `AliasableBox`. This type is also safe to move while borrows exist
                    // (unlike `Box`). This, coupled with the fact that we only allow access to the
                    // borrower with its lifetime bound to `self` such that the internal borrows
                    // cannot outlive the owners, ensures that extending the lifetime is sound.
                    let inner = unsafe {
                        std::mem::transmute::<$borrower_ty, $erased_borrower_ty>(inner)
                    };

                    Self { inner, owners }
                }

                #[inline]
                pub fn as_ref(&self) -> &$referent_ty {
                    &self.inner
                }
            }

            impl std::fmt::Debug for $struct_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    std::fmt::Debug::fmt(self.as_ref(), f)
                }
            }

            $(crate::self_referential::self_referential!(@$auto_impl $struct_name);)*
        }

        $vis use $module_name::$struct_name;
    };
    (@PartialEq $struct_name:ident) => {
        impl PartialEq for $struct_name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_ref() == other.as_ref()
            }
        }
    };
    (@Eq $struct_name:ident) => {
        impl Eq for $struct_name {}
    };
    (@Hash $struct_name:ident) => {
        impl std::hash::Hash for $struct_name {
            #[inline]
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.as_ref().hash(state);
            }
        }
    };
}
pub(crate) use self_referential;

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
