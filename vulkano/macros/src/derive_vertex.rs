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

    let mut member_cases = quote! {};

    for field in fields.iter() {
        let field_name = field.ident.to_owned().unwrap();
        let mut names = vec![LitStr::new(&field_name.to_string(), Span::call_site())];
        let mut format = quote! {
            let dummy = <#struct_name>::default();
            #[inline] fn f<T: VertexMember>(_: &T) -> Format { T::format() }
            let format = f(&dummy.#field_name);
        };
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
        for name in &names {
            member_cases = quote! {
                #member_cases

                if name == #name {
                    #format
                    let field_size = {
                        let m = core::mem::MaybeUninit::<#struct_name>::uninit();
                        let p = unsafe { core::ptr::addr_of!((*(&m as *const _ as *const #struct_name)).#field_name) };
                        const fn size_of_raw<T>(_: *const T) -> usize {
                            core::mem::size_of::<T>()
                        }
                        size_of_raw(p)
                    } as u32;
                    let format_size = format.block_size().expect("no block size for format") as u32;
                    let num_elements = field_size / format_size;
                    let remainder = field_size % format_size;
                    assert!(remainder == 0, "struct field `{}` size does not fit multiple of format size", name);
                    let offset = {
                        let dummy = ::core::mem::MaybeUninit::<#struct_name>::uninit();
                        let dummy_ptr = dummy.as_ptr();
                        let member_ptr = unsafe { ::core::ptr::addr_of!((*dummy_ptr).#field_name) };
                        member_ptr as usize - dummy_ptr as usize
                    };
                    return Some(VertexMemberInfo {
                        offset,
                        format,
                        num_elements,
                    });
                }
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
