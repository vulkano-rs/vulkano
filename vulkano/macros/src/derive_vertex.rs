// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::bail;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Data, DataStruct, Fields, Ident, LitStr, Result, Token,
};

pub fn derive_vertex(ast: syn::DeriveInput) -> Result<TokenStream> {
    let struct_name = &ast.ident;

    let fields = match &ast.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => bail!("expected a struct with named fields"),
    };

    let crate_ident = crate::crate_ident();

    let mut members = quote! {
        let mut offset = 0;
        let mut members = ::std::collections::HashMap::default();
    };

    for field in fields.iter() {
        let field_name = field.ident.to_owned().unwrap();
        let field_name_lit = LitStr::new(&field_name.to_string(), Span::call_site());
        let field_ty = &field.ty;
        let mut names = vec![field_name_lit.clone()];
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
                    let format = ::#crate_ident::format::Format::#format_ident;
                };
            }
        }
        if format.is_empty() {
            bail!(
                field_name,
                "expected `#[format(...)]`-attribute with valid `vulkano::format::Format`",
            );
        }
        for name in &names {
            members = quote! {
                #members

                let field_size = ::std::mem::size_of::<#field_ty>() as u32;
                {
                    #format
                    let format_size = format.block_size().expect("no block size for format") as u32;
                    let num_elements = field_size / format_size;
                    let remainder = field_size % format_size;
                    ::std::assert!(remainder == 0, "struct field `{}` size does not fit multiple of format size", #field_name_lit);
                    members.insert(
                        #name.to_string(),
                        ::#crate_ident::pipeline::graphics::vertex_input::VertexMemberInfo {
                            offset,
                            format,
                            num_elements,
                        },
                    );
                }
                offset += field_size as usize;
            };
        }
    }

    let function_body = quote! {
        #members

        ::#crate_ident::pipeline::graphics::vertex_input::VertexBufferDescription {
            members,
            stride: ::std::mem::size_of::<#struct_name>() as u32,
            input_rate: ::#crate_ident::pipeline::graphics::vertex_input::VertexInputRate::Vertex,
        }
    };

    Ok(quote! {
        #[allow(unsafe_code)]
        unsafe impl ::#crate_ident::pipeline::graphics::vertex_input::Vertex for #struct_name {
            #[inline(always)]
            fn per_vertex() -> ::#crate_ident::pipeline::graphics::vertex_input::VertexBufferDescription {
                #function_body
            }

            #[inline(always)]
            fn per_instance() -> ::#crate_ident::pipeline::graphics::vertex_input::VertexBufferDescription {
                Self::per_vertex().per_instance()
            }

            #[inline(always)]
            fn per_instance_with_divisor(divisor: u32) -> ::#crate_ident::pipeline::graphics::vertex_input::VertexBufferDescription {
                Self::per_vertex().per_instance_with_divisor(divisor)
            }
        }
    })
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
