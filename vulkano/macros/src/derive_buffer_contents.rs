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
    parse_quote, punctuated::Punctuated, Data, DeriveInput, Field, Fields, FieldsNamed,
    FieldsUnnamed, Ident, Meta, MetaList, NestedMeta, Result, Token, Type, TypeArray, TypeSlice,
    WherePredicate,
};

pub fn derive_buffer_contents(mut ast: DeriveInput) -> Result<TokenStream> {
    let crate_ident = crate::crate_ident();

    let struct_ident = &ast.ident;

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

    let is_unsized = matches!(fields.last().unwrap().ty, Type::Slice(_))
        || ast
            .attrs
            .iter()
            .any(|attr| attr.path.is_ident("dynamically_sized")); // unsized is a reserved keyword.

    let (layout, bound_types) = write_layout(&crate_ident, fields, is_unsized);

    let (impl_generics, type_generics, where_clause) = {
        let predicates = bound_types.iter().map(|ty| -> WherePredicate {
            parse_quote! { #ty: ::#crate_ident::buffer::BufferContents }
        });

        ast.generics
            .make_where_clause()
            .predicates
            .extend(predicates);

        ast.generics.split_for_impl()
    };

    let (from_ffi, from_ffi_mut);

    if is_unsized {
        let components = quote! {
            let alignment = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .alignment()
                .as_devicesize() as usize;
            ::std::debug_assert!(data as usize % alignment == 0);

            let head_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .head_size() as usize;
            let element_size = <Self as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .element_size()
                .unwrap() as usize;

            ::std::debug_assert!(range >= head_size);
            let tail_size = range - head_size;
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

fn write_layout<'a>(
    crate_ident: &Ident,
    fields: &'a Punctuated<Field, Token![,]>,
    is_unsized: bool,
) -> (TokenStream, Vec<&'a Type>) {
    let mut bound_types = Vec::new();

    let mut field_types = fields.iter().map(|field| &field.ty);
    let last_field_type = field_types.next_back().unwrap();
    let mut layout = quote! { ::std::alloc::Layout::new::<()>() };

    // Construct the layout of the head and accumulate the types that have to implement
    // `BufferContents` in order for the struct to implement the trait as well.
    for field_type in field_types {
        bound_types.push(find_innermost_element_type(field_type));

        layout = quote! {
            extend_layout(#layout, ::std::alloc::Layout::new::<#field_type>())
        };
    }

    // The last field needs special treatment depending on whether it's a slice or not.
    if let Type::Slice(TypeSlice { elem, .. }) = last_field_type {
        bound_types.push(find_innermost_element_type(elem));

        layout = quote! {
            ::#crate_ident::buffer::BufferContentsLayout::from_head_element_layout(
                #layout,
                ::std::alloc::Layout::new::<#elem>(),
            )
        };
    } else if is_unsized {
        // We don't need to add `last_field_type` to the bounds because it is enforced here in the
        // `LAYOUT` constant.
        layout = quote! {
            <#last_field_type as ::#crate_ident::buffer::BufferContents>::LAYOUT
                .extend_from_layout(&#layout)
        };
    } else {
        bound_types.push(find_innermost_element_type(last_field_type));

        layout = quote! {
            ::#crate_ident::buffer::BufferContentsLayout::from_sized(
                ::std::alloc::Layout::new::<Self>()
            )
        };
    }

    let layout = quote! {
        {
            // HACK: Very depressingly, `Layout::extend` is not const.
            const fn extend_layout(
                layout: ::std::alloc::Layout,
                next: ::std::alloc::Layout,
            ) -> ::std::alloc::Layout {
                let padded_size = if let Some(val) =
                    layout.size().checked_add(layout.align() - 1)
                {
                    val & !(layout.align() - 1)
                } else {
                    ::std::unreachable!()
                };

                // TODO: Replace with `Ord::max` once its constness is stabilized.
                let align = if layout.align() >= next.align() {
                    layout.align()
                } else {
                    next.align()
                };

                if let Some(size) = padded_size.checked_add(next.size()) {
                    if let Ok(layout) = ::std::alloc::Layout::from_size_align(size, align) {
                        layout
                    } else {
                        ::std::unreachable!()
                    }
                } else {
                    ::std::unreachable!()
                }
            }

            if let Some(layout) = #layout {
                if let Some(layout) = layout.pad_to_alignment() {
                    layout
                } else {
                    ::std::unreachable!()
                }
            } else {
                ::std::panic!("zero-sized types are not valid buffer contents")
            }
        }
    };

    (layout, bound_types)
}

// HACK: This works around an inherent limitation of bytemuck, namely that an array
// where the element is `AnyBitPattern` is itself not `AnyBitPattern`, by only
// requiring that the innermost type in the array implements `BufferContents`.
fn find_innermost_element_type(mut field_type: &Type) -> &Type {
    while let Type::Array(TypeArray { elem, .. }) = field_type {
        field_type = elem;
    }

    field_type
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
