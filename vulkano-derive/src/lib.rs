extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DataStruct, DeriveInput, Fields};

#[proc_macro_derive(Vertex)]
pub fn vertex_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as DeriveInput);

    let fields = match &ast.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => panic!("expected struct with named fields"),
    };

    let member = fields.iter().map(|field| &field.ident);
    let ty = fields.iter().map(|field| &field.ty);
    let name = &ast.ident;

    let output = quote! {
        #[allow(unsafe_code)]
        unsafe impl ::vulkano::pipeline::vertex::Vertex for #name {
            #[inline(always)]
            fn member(name: &str)
            -> Option<::vulkano::pipeline::vertex::VertexMemberInfo> {
                use std::ptr;

                #[allow(unused_imports)]
                use ::vulkano::format::Format;
                use ::vulkano::pipeline::vertex::VertexMemberInfo;
                use ::vulkano::pipeline::vertex::VertexMemberTy;
                use ::vulkano::pipeline::vertex::VertexMember;

                #(
                    if name == stringify!(#member) {
                        let dummy = <#name>::default();
                        let (ty, array_size) = <#ty>::format();
                        let dummy_ptr = (&dummy) as *const _;
                        let member_ptr = (&dummy.#member) as *const _;

                        return Some(VertexMemberInfo {
                            offset: member_ptr as usize - dummy_ptr as usize,
                            ty: ty,
                            array_size: array_size,
                        })
                    }
                )*

                None
            }
        }
    };

    output.into()
}
