use std::{
    fmt,
    ops::{Deref, DerefMut},
    panic::{RefUnwindSafe, UnwindSafe},
    ptr::NonNull,
};

/// Creates a self-referential struct while encapsulating all the unsafety.
///
/// The macro is given a single module with a single struct. This struct must have one `inner`
/// field followed by one or more "owner" fields. The `inner` field is the type with internal
/// borrows and where the lifetime goes is given by wherever the `'_` lifetime is used in the type.
/// The owner fields can be of any type. It's additionally possible to provide an `impl` keyword
/// followed by a comma-separated list of traits to automatically implement. These traits forward
/// to the `inner` field's implementation. This is because for use as map keys, the comparison and
/// hash implementations must match between the owned and borrowed structs (and it's also
/// unnecessary to use the other fields when they are borrowed in the `inner` field anyway).
///
/// What the macro does under the hood is that it generates the given module containing the given
/// struct. The module is used in order to prevent the parent module at looking at the struct
/// fields which would be unsound. However, is is meant to "look" like it's generating the struct
/// inline, which is achieved by reexporting the generated struct. The struct with its visibility
/// will therefore be available in the containing module with that visibility. The struct has a
/// `new` method which safely constructs the self-referential struct: it allocates the owners in an
/// [`AliasableBox`] to allow the owners to be stored alongside the `inner` field and calls the
/// provided closure with borrows to these heap-allocated owners. Then it extends the lifetime of
/// the `inner` field to `'static` in order to allow it to be stored in an owned manner and the box
/// to be moved. This is why it would be unsound to allow looking at the generated fields: you
/// could observe the fake static lifetime and/or drop or otherwise mutate the owners. The only way
/// to access the `inner` field must be using the generated `as_ref` method, which ensures that the
/// fake static lifetime is not observable, and the owner fields must be inaccessible.
///
/// We need self-referential structs in order to implement APIs that return the same info structs
/// that are used when creating the object, and these all have a lifetime. If the struct holds
/// references to some other structs/vectors, these have to be stored somewhere.
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
        // SAFETY: The `$referent_ty` doesn't expose the static lifetime.
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
        // SAFETY: The `$referent_ty` doesn't expose the static lifetime.
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
        // SAFETY: The `$referent_ty` doesn't expose the static lifetime.
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
        // SAFETY: The `$referent_ty` doesn't expose the static lifetime.
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

/// At the time of writing, using the standard library [`Box`], moving a `Box` invalidated pointers
/// to its contents. This is why using this type for unsafe code is very ill-advised; our case is
/// no different. If we used a struct like `(Inner<'static>, Box<Owners>)` for our self-referential
/// structs, moving this struct would invalidate `Inner`'s internal borrows of `Owners`.
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
