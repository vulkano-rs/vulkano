// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod derive_vertex;

#[proc_macro_derive(Vertex, attributes(name, format))]
pub fn proc_derive_vertex(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    derive_vertex::derive_vertex(ast).unwrap_or_else(|err| err.to_compile_error().into())
}
