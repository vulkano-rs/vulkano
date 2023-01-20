// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::bail;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse_quote, Data, DeriveInput, Fields, FieldsNamed, FieldsUnnamed, Meta, MetaList, NestedMeta,
    Result, Type, WherePredicate,
};

pub fn derive_buffer_contents(mut ast: DeriveInput) -> Result<TokenStream> {
    let crate_ident = crate::crate_ident();

    let struct_ident = &ast.ident;

    let (impl_generics, type_generics, where_clause) = {
        let predicates = ast
            .generics
            .type_params()
            .map(|param| {
                let param_ident = &param.ident;
                parse_quote! { #param_ident: ::#crate_ident::buffer::BufferContents }
            })
            .collect::<Vec<WherePredicate>>();
        ast.generics
            .make_where_clause()
            .predicates
            .extend(predicates);

        ast.generics.split_for_impl()
    };

    let data = match &ast.data {
        Data::Struct(data) => data,
        Data::Enum(_) => bail!("deriving `BufferContents` for enums is not supported"),
        Data::Union(_) => bail!("deriving `BufferContents` for unions is not supported"),
    };

    let fields = match &data.fields {
        Fields::Named(FieldsNamed { named, .. }) => named,
        Fields::Unnamed(FieldsUnnamed { unnamed, .. }) => unnamed,
        Fields::Unit => bail!("zero-sized types are not valid buffer contents"),
    };

    if !ast
        .attrs
        .iter()
        .filter_map(|attr| {
            attr.path
                .is_ident("repr")
                .then(|| attr.parse_meta().unwrap())
        })
        .any(|meta| match meta {
            Meta::List(MetaList { nested, .. }) => {
                nested.iter().any(|nested_meta| match nested_meta {
                    NestedMeta::Meta(Meta::Path(path)) => {
                        path.is_ident("C") || path.is_ident("transparent")
                    }
                    _ => false,
                })
            }
            _ => false,
        })
    {
        bail!(
            "deriving `BufferContents` is only supported for types that are marked `#[repr(C)]` \
            or `#[repr(transparent)]`",
        );
    }

    let mut field_types = fields.iter().map(|field| &field.ty);
    let first_field_type = field_types.next().unwrap();
    let mut layout = quote! {
        <#first_field_type as ::#crate_ident::buffer::BufferContents>::LAYOUT
    };
    for field_type in field_types {
        layout = quote! {
            // TODO: Replace with `Option::unwrap` once its constness is stabilized.
            if let ::std::option::Option::Some(layout) =
                #layout.extend(<#field_type as ::#crate_ident::buffer::BufferContents>::LAYOUT)
            {
                layout
            } else {
                ::std::unreachable!()
            }
        };
    }

    let is_unsized = matches!(fields.last().unwrap().ty, Type::Slice(_))
        || ast
            .attrs
            .iter()
            .any(|attr| attr.path.is_ident("dynamically_sized")); // unsized is a reserved keyword.

    let (from_ffi, from_ffi_mut);

    if is_unsized {
        let components = quote! {
            let alignment = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .alignment()
                .as_devicesize() as usize;
            ::std::debug_assert!(data as usize % alignment == 0);

            let padded_head_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .padded_head_size() as usize;
            let element_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .element_size()
                .unwrap() as usize;

            ::std::debug_assert!(range >= padded_head_size);
            let tail_size = range - padded_head_size;
            ::std::debug_assert!(tail_size % element_size == 0);
            let len = tail_size / element_size;

            let components = PtrComponents { data, len };
        };

        from_ffi = quote! {
            #[repr(C)]
            struct PtrComponents {
                data: *const ::std::ffi::c_void,
                len: usize,
            }

            #components

            // SAFETY: All fields must implement `BufferContents`. The last field, if it is
            // unsized, must therefore be a slice or a DST derived from a slice. It can not be any
            // other kind of DST, unless unsafe code was used to achieve that.
            //
            // That means we can safely rely on knowing what kind of DST the implementing type is,
            // but it doesn't tell us what the correct representation for the pointer of this kind
            // of DST is. For that we have to rely on what the docs tell us, namely that for
            // structs where the last field is a DST, the metadata is the same as the last field's.
            // We also know that the metadata of a slice is its length measured in the number of
            // elements. This tells us that the components of a pointer to the implementing type
            // are the address to the start of the data, and a length. It still does not tell us
            // what the representation of the pointer is though.
            //
            // In fact, there is no way to be certain that this representation is correct.
            // *Theoretically* rustc could decide tomorrow that the metadata comes first and the
            // address comes last, but the chance of that ever happening is zero.
            //
            // But what if the implementing type is actually sized? In that case the size of a
            // pointer to the type will by definition be smaller, and since transmuting types of
            // different sizes never works, it will cause a compilation error on this line.
            //
            // TODO: HACK: Replace with `std::ptr::from_raw_parts` once it is stabilized.
            ::std::mem::transmute::<PtrComponents, *const Self>(components)
        };

        from_ffi_mut = quote! {
            #[repr(C)]
            struct PtrComponents {
                data: *mut ::std::ffi::c_void,
                len: usize,
            }

            #components

            // SAFETY: Please read the docs in `from_ffi` above.
            // TODO: HACK: Replace with `std::ptr::from_raw_parts_mut` once it is stabilized.
            ::std::mem::transmute::<PtrComponents, *mut Self>(components)
        };
    } else {
        from_ffi = quote! {
            ::std::debug_assert!(range == ::std::mem::size_of::<Self>());
            ::std::debug_assert!(data as usize % ::std::mem::align_of::<Self>() == 0);

            data.cast()
        };

        from_ffi_mut = from_ffi.clone();
    }

    Ok(quote! {
        #[allow(unsafe_code)]
        unsafe impl #impl_generics ::#crate_ident::buffer::BufferContents
            for #struct_ident #type_generics #where_clause
        {
            const LAYOUT: ::#crate_ident::buffer::BufferContentsLayout = #layout;

            #[inline(always)]
            unsafe fn from_ffi(data: *const ::std::ffi::c_void, range: usize) -> *const Self {
                #from_ffi
            }

            #[inline(always)]
            unsafe fn from_ffi_mut(data: *mut ::std::ffi::c_void, range: usize) -> *mut Self {
                #from_ffi_mut
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr() {
        let default_repr = parse_quote! {
            struct Test(u8, [u8]);
        };
        assert!(derive_buffer_contents(default_repr).is_err());

        let irellevant_reprs = parse_quote! {
            #[repr(packed(2), align(16))]
            struct Test(u8, [u8]);
        };
        assert!(derive_buffer_contents(irellevant_reprs).is_err());

        let transparent_repr = parse_quote! {
            #[repr(transparent)]
            struct Test([u8]);
        };
        assert!(derive_buffer_contents(transparent_repr).is_ok());

        let multiple_reprs = parse_quote! {
            #[repr(align(16))]
            #[repr(C)]
            #[repr(packed)]
            struct Test(u8, [u8]);
        };
        assert!(derive_buffer_contents(multiple_reprs).is_ok());
    }

    #[test]
    fn zero_sized() {
        let unit = parse_quote! {
            struct Test;
        };
        assert!(derive_buffer_contents(unit).is_err());
    }

    #[test]
    fn unsupported_datatype() {
        let enum_ = parse_quote! {
            #[repr(C)]
            enum Test { A, B, C }
        };
        assert!(derive_buffer_contents(enum_).is_err());

        let union = parse_quote! {
            #[repr(C)]
            union Test {
                a: u32,
                b: f32,
            }
        };
        assert!(derive_buffer_contents(union).is_err());
    }
}
