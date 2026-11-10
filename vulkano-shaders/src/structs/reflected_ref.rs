#![allow(unused_unsafe)]

use super::{
    Constant, TypeArray, TypeFloat, TypeInt, TypeMatrix, TypePointer, TypeStruct, TypeVector,
};
use crate::arena;
use std::{marker::PhantomData, mem, ptr::NonNull};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(super) struct ReflectedRef<'arena> {
    // INVARIANT: The pointer with the low 3 bits zeroed must be a valid pointer to a reflected
    // SPIR-V construct, and the low 3 bits must be that reflected construct's `ReflectedKind`.
    ptr: NonNull<()>,
    marker: PhantomData<&'arena ()>,
}

#[derive(Clone, Copy)]
#[repr(u8)]
#[rustfmt::skip]
enum ReflectedKind {
    TypeInt     = 0b_000,
    TypeFloat   = 0b_001,
    TypePointer = 0b_010,
    TypeVector  = 0b_011,
    TypeMatrix  = 0b_100,
    TypeArray   = 0b_101,
    TypeStruct  = 0b_110,
    Constant    = 0b_111,
}

impl<'arena> ReflectedRef<'arena> {
    /// # Safety
    ///
    /// - `reflected_kind` must correspond to what's being pointed to by `ptr`.
    /// - `ptr` must be valid and have a lifetime of `'arena`.
    #[inline(always)]
    unsafe fn new(ptr: NonNull<()>, reflected_kind: ReflectedKind) -> Self {
        // This assert is going to get murdered by LLVM.
        assert!((reflected_kind as u8) < 8);

        let addr = ptr.addr();

        // This assert is going to get murdered by LLVM once it's inlined into the outer function.
        assert!(arena::is_aligned(addr.get(), 8));

        let ptr = ptr.with_addr(addr | reflected_kind as usize);

        // SAFETY: We checked that `ptr` is aligned to `8`, which means that the low 3 bits are
        // zero, and we ORed it with `reflected_kind`, which we checked is a 3-bit value. The
        // caller must ensure that `reflected_kind` corresponds to what's being pointed to by `ptr`
        // and that the pointer's lifetime is `'arena`.
        unsafe {
            ReflectedRef {
                ptr,
                marker: PhantomData,
            }
        }
    }

    #[inline(always)]
    pub(super) fn from_constant(constant: &'arena Constant) -> Self {
        let ptr = NonNull::from(constant).cast::<()>();

        // SAFETY: `ptr` was derived from a reference to a `Constant` with lifetime `'arena`, and
        // `ReflectedKind::Constant` is its kind.
        unsafe { Self::new(ptr, ReflectedKind::Constant) }
    }

    #[inline(always)]
    fn kind(self) -> ReflectedKind {
        let addr = self.ptr.as_ptr().addr();

        // SAFETY: By our own invariant, `self.ptr`'s low 3 bits must be the value of a
        // `ReflectedKind`.
        unsafe { mem::transmute::<u8, ReflectedKind>((addr & 0b111) as u8) }
    }

    #[inline(always)]
    fn ptr(self) -> NonNull<()> {
        let ptr = self.ptr.as_ptr().map_addr(|addr| addr & !0b111);

        // SAFETY: By our own invariant, `self.ptr` with the low 3 bits zeroed must be a valid
        // pointer, so it can't be null.
        unsafe { NonNull::new_unchecked(ptr) }
    }

    #[inline(always)]
    pub(super) fn to_type_ref(self) -> Option<TypeRef<'arena>> {
        if (self.kind() as u8) < 8 {
            // SAFETY: By our own invariant, `self.kind()` must be the kind of reflected construct
            // being pointed to by `self.ptr()`, and we checked that the kind is that of a type.
            Some(unsafe { TypeRef(self) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub(super) fn to_type_scalar_ref(self) -> Option<TypeScalarRef<'arena>> {
        if (self.kind() as u8) < 3 {
            // SAFETY: By our own invariant, `self.kind()` must be the kind of reflected construct
            // being pointed to by `self.ptr()`, and we checked that the kind is that of a scalar
            // type.
            Some(unsafe { TypeScalarRef(TypeRef(self)) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub(super) fn to_type_vector(self) -> Option<&'arena TypeVector<'arena>> {
        if matches!(self.kind(), ReflectedKind::TypeVector) {
            // SAFETY: By our own invariant, `self.kind()` must be the kind of reflected construct
            // being pointed to by `self.ptr()`.
            Some(unsafe { self.ptr().cast::<TypeVector<'arena>>().as_ref() })
        } else {
            None
        }
    }

    #[inline(always)]
    pub(super) fn to_type_struct(self) -> Option<&'arena TypeStruct<'arena>> {
        if matches!(self.kind(), ReflectedKind::TypeStruct) {
            // SAFETY: By our own invariant, `self.kind()` must be the kind of reflected construct
            // being pointed to by `self.ptr()`.
            Some(unsafe { self.ptr().cast::<TypeStruct<'arena>>().as_ref() })
        } else {
            None
        }
    }

    #[inline(always)]
    pub(super) fn to_constant(self) -> Option<&'arena Constant> {
        if matches!(self.kind(), ReflectedKind::Constant) {
            // SAFETY: By our own invariant, `self.kind()` must be the kind of reflected construct
            // being pointed to by `self.ptr()`.
            Some(unsafe { self.ptr().cast::<Constant>().as_ref() })
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(super) struct TypeRef<'arena>(
    // INVARIANT: Must have a `ReflectedKind` of a type.
    ReflectedRef<'arena>,
);

// These variants *are* constructed, just in a way that Rust doesn't understand.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, u8)]
pub(super) enum Type<'arena> {
    Int(&'arena TypeInt) = ReflectedKind::TypeInt as u8,
    Float(&'arena TypeFloat) = ReflectedKind::TypeFloat as u8,
    Pointer(&'arena TypePointer) = ReflectedKind::TypePointer as u8,
    Vector(&'arena TypeVector<'arena>) = ReflectedKind::TypeVector as u8,
    Matrix(&'arena TypeMatrix<'arena>) = ReflectedKind::TypeMatrix as u8,
    Array(&'arena TypeArray<'arena>) = ReflectedKind::TypeArray as u8,
    Struct(&'arena TypeStruct<'arena>) = ReflectedKind::TypeStruct as u8,
}

#[repr(C)]
struct TypeRepr {
    kind: ReflectedKind,
    ptr: NonNull<()>,
}

impl<'arena> TypeRef<'arena> {
    #[allow(private_bounds)]
    #[inline(always)]
    pub(super) fn new<T: IsType>(ty: &'arena T) -> Self {
        let ptr = NonNull::from(ty).cast::<()>();

        // SAFETY: `ptr` was derived from a reference to an `IsType` with lifetime `'arena`, and
        // `T::KIND` is the `IsType`'s kind.
        unsafe { TypeRef(ReflectedRef::new(ptr, T::KIND)) }
    }

    #[inline(always)]
    pub(super) fn to_reflected_ref(self) -> ReflectedRef<'arena> {
        self.0
    }

    #[inline(always)]
    pub(super) fn to_type(self) -> Type<'arena> {
        let kind = self.0.kind();
        let ptr = self.0.ptr();

        // SAFETY: `TypeRepr` has the same representation as `Type`. We made sure of this by making
        // `Type` and `TypeRepr` `#[repr(C)]` and by making `Type` and `ReflectedKind`
        // `#[repr(u8)]`. By `ReflectedRef`'s invariant, `kind` must be the kind of the reflected
        // construct being pointed to by `ptr`, and by our own invariant, the reflected construct
        // must be a type.
        unsafe { mem::transmute::<TypeRepr, Type<'arena>>(TypeRepr { kind, ptr }) }
    }
}

impl PartialEq for TypeRef<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_type() == other.to_type()
    }
}

impl Eq for TypeRef<'_> {}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(super) struct TypeScalarRef<'arena>(
    // INVARIANT: Must have a `ReflectedKind` of a scalar type.
    TypeRef<'arena>,
);

// These variants *are* constructed, just in a way that Rust doesn't understand.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, u8)]
pub(super) enum TypeScalar<'arena> {
    Int(&'arena TypeInt) = ReflectedKind::TypeInt as u8,
    Float(&'arena TypeFloat) = ReflectedKind::TypeFloat as u8,
}

impl<'arena> TypeScalarRef<'arena> {
    #[inline(always)]
    pub(super) fn from_type_int(type_int: &'arena TypeInt) -> Self {
        // SAFETY: `type_int` is scalar.
        unsafe { TypeScalarRef(TypeRef::new(type_int)) }
    }

    #[inline(always)]
    pub(super) fn from_type_float(type_float: &'arena TypeFloat) -> Self {
        // SAFETY: `type_float` is scalar.
        unsafe { TypeScalarRef(TypeRef::new(type_float)) }
    }

    #[inline(always)]
    pub(super) fn to_type_scalar(self) -> TypeScalar<'arena> {
        // SAFETY: `Type` and `TypeScalar` have the same layout. We made sure of this by making
        // `Type` and `TypeScalar` `#[repr(C, u8)]`. `TypeScalar` is transmutable to `Type` because
        // the former is a subtype of the latter. We made sure of this by making them have the same
        // discriminants. In this case, since, by our own invariant, `self.0` must be scalar, we
        // can also transmute the other way.
        unsafe { mem::transmute::<Type<'arena>, TypeScalar<'arena>>(self.0.to_type()) }
    }
}

impl PartialEq for TypeScalarRef<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_type_scalar() == other.to_type_scalar()
    }
}

impl Eq for TypeScalarRef<'_> {}

/// # Safety
///
/// - `KIND` must be that of a type.
/// - `KIND` must be unique for each implementor.
unsafe trait IsType {
    const KIND: ReflectedKind;
}

// SAFETY: These are the only implementations of `IsType`, and they all have unique `KIND` values.
unsafe impl IsType for TypeInt {
    const KIND: ReflectedKind = ReflectedKind::TypeInt;
}

// SAFETY: Same as the previous.
unsafe impl IsType for TypeFloat {
    const KIND: ReflectedKind = ReflectedKind::TypeFloat;
}

// SAFETY: Same as the previous.
unsafe impl IsType for TypePointer {
    const KIND: ReflectedKind = ReflectedKind::TypePointer;
}

// SAFETY: Same as the previous.
unsafe impl IsType for TypeVector<'_> {
    const KIND: ReflectedKind = ReflectedKind::TypeVector;
}

// SAFETY: Same as the previous.
unsafe impl IsType for TypeMatrix<'_> {
    const KIND: ReflectedKind = ReflectedKind::TypeMatrix;
}

// SAFETY: Same as the previous.
unsafe impl IsType for TypeArray<'_> {
    const KIND: ReflectedKind = ReflectedKind::TypeArray;
}

// SAFETY: Same as the previous.
unsafe impl IsType for TypeStruct<'_> {
    const KIND: ReflectedKind = ReflectedKind::TypeStruct;
}
