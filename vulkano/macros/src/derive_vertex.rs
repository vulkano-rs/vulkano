// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Data, DataStruct, Error, Fields, Ident, LitStr, Result, Token,
};

pub fn derive_vertex(ast: syn::DeriveInput) -> Result<TokenStream> {
    let struct_name = &ast.ident;

    let fields = match &ast.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => {
            return Err(Error::new_spanned(
                ast,
                "Expected a struct with named fields",
            ));
        }
    };

    let found_crate = crate_name("vulkano").expect("vulkano is present in `Cargo.toml`");

    let crate_ident = match found_crate {
        FoundCrate::Itself => Ident::new("crate", Span::call_site()),
        FoundCrate::Name(name) => Ident::new(&name, Span::call_site()),
    };

    let mut member_cases = quote! {
        let mut offset = 0;
    };

    for field in fields.iter() {
        let field_name = field.ident.to_owned().unwrap();
        let field_ty = &field.ty;
        let mut names = vec![LitStr::new(&field_name.to_string(), Span::call_site())];
        let mut format = quote! {};
        for attr in &field.attrs {
            let attr_ident = if let Some(ident) = attr.path.get_ident() {
                ident
            } else {
                continue;
            };
            if attr_ident == "name" {
                let meta = attr.parse_args_with(NameMeta::parse)?;
                names = meta.lit_str_list.into_iter().collect();
            } else if attr_ident == "format" {
                let format_ident = attr.parse_args_with(Ident::parse)?;
                format = quote! {
                    let format = Format::#format_ident;
                };
            }
        }
        if format.is_empty() {
            return Err(Error::new(
                field_name.span(),
                "Expected `#[format(...)]`-attribute with valid `vulkano::format::Format`",
            ));
        }
        for name in &names {
            member_cases = quote! {
                #member_cases

                let field_size = std::mem::size_of::<#field_ty>() as u32;
                if name == #name {
                    #format
                    let format_size = format.block_size().expect("no block size for format") as u32;
                    let num_elements = field_size / format_size;
                    let remainder = field_size % format_size;
                    assert!(remainder == 0, "struct field `{}` size does not fit multiple of format size", name);
                    return Some(VertexMemberInfo {
                        offset,
                        format,
                        num_elements,
                    });
                }
                offset += field_size as usize;
            };
        }
    }

    Ok(TokenStream::from(quote! {
        #[allow(unsafe_code)]
        unsafe impl #crate_ident::pipeline::graphics::vertex_input::Vertex for #struct_name {
            #[inline(always)]
            fn member(name: &str) -> Option<#crate_ident::pipeline::graphics::vertex_input::VertexMemberInfo> {
                #[allow(unused_imports)]
                use #crate_ident::format::Format;
                use #crate_ident::pipeline::graphics::vertex_input::VertexMemberInfo;
                use #crate_ident::pipeline::graphics::vertex_input::VertexMember;

                #member_cases

                None
            }
        }
    }))
}

struct NameMeta {
    lit_str_list: Punctuated<LitStr, Token![,]>,
}

impl Parse for NameMeta {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            lit_str_list: input.parse_terminated(<LitStr as Parse>::parse)?,
        })
    }
}
