use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod derive_vertex;

#[proc_macro_derive(Vertex, attributes(name, format))]
pub fn proc_derive_vertex(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    derive_vertex::derive_vertex(ast).unwrap_or_else(|err| err.to_compile_error().into())
}
