//! The codegen for [vulkano-macros].
//!
//! vulkano-macros is a shim over this crate so that the codegen can be used by crates other than
//! vulkano-macros as proc-macro crates can't export anything but proc macros. The intended use
//! case for this crate is for implementing your own shim over it with a different [vulkano path]
//! such that your library's users don't have to have vulkano in their dependencies.
//!
//! [vulkano path]: CodegenContext::vulkano_path
//! [vulkano-macros]: https://docs.rs/vulkano-macros/latest/vulkano_macros/

use proc_macro2::TokenStream;
use quote::quote;
use std::marker::PhantomData;
use syn::Error;

mod derive_buffer_contents;
mod derive_vertex;
mod shader;
#[cfg(test)]
mod tests;

/// See [`vulkano_macros::BufferContents`].
///
/// [`vulkano_macros::BufferContents`]: https://docs.rs/vulkano-macros/latest/vulkano_macros/derive.BufferContents.html
pub fn derive_buffer_contents(ccx: &CodegenContext<'_>, input: TokenStream) -> TokenStream {
    derive_buffer_contents::derive_buffer_contents(ccx, input)
        .unwrap_or_else(Error::into_compile_error)
}

/// See [`vulkano_macros::Vertex`].
///
/// [`vulkano_macros::Vertex`]: https://docs.rs/vulkano-macros/latest/vulkano_macros/derive.Vertex.html
pub fn derive_vertex(ccx: &CodegenContext<'_>, input: TokenStream) -> TokenStream {
    derive_vertex::derive_vertex(ccx, input).unwrap_or_else(Error::into_compile_error)
}

/// See [`vulkano_macros::include_vulkano_glsl`].
///
/// [`vulkano_macros::include_vulkano_glsl`]: https://docs.rs/vulkano-macros/latest/vulkano_macros/macro.include_vulkano_glsl.html
pub fn include_vulkano_glsl(input: TokenStream) -> TokenStream {
    if let Err(err) = syn::parse2::<syn::parse::Nothing>(input) {
        return err.into_compile_error();
    }

    let include_string = include_str!("../include/vulkano.glsl");

    quote! { #include_string }
}

/// See [`vulkano_macros::include_vulkano_slang`].
///
/// [`vulkano_macros::include_vulkano_slang`]: https://docs.rs/vulkano-macros/latest/vulkano_macros/macro.include_vulkano_slang.html
pub fn include_vulkano_slang(input: TokenStream) -> TokenStream {
    if let Err(err) = syn::parse2::<syn::parse::Nothing>(input) {
        return err.into_compile_error();
    }

    let include_string = include_str!("../include/vulkano.slang");

    quote! { #include_string }
}

/// See [`vulkano_macros::shader`].
///
/// [`vulkano_macros::shader`]: https://docs.rs/vulkano-macros/latest/vulkano_macros/macro.shader.html
pub fn shader(ccx: &CodegenContext<'_>, input: TokenStream) -> TokenStream {
    shader::shader(ccx, input).unwrap_or_else(Error::into_compile_error)
}

/// Parameters for code generation.
#[derive(Clone, Debug)]
pub struct CodegenContext<'a> {
    /// The path to the vulkano module.
    ///
    /// Paths to vulkano items are generated as `"::{vulkano_path}::{item_path}"`. Note that the
    /// leading `::` is already present and as such must not be included in `vulkano_path`.
    ///
    /// This must either be the default, which is `"vulkano"`, or the path of a reexported vulkano
    /// (e.g, `"my_library::vulkano"`).
    pub vulkano_path: &'a str,

    pub _ne: NonExhaustive<'a>,
}

impl Default for CodegenContext<'_> {
    #[inline]
    fn default() -> Self {
        CodegenContext {
            vulkano_path: "vulkano",
            _ne: NonExhaustive(PhantomData),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NonExhaustive<'a>(PhantomData<&'a ()>);

macro_rules! bail {
    ($msg:expr $(,)?) => {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format_args!($msg),
        ))
    };
    ($span:expr, $msg:expr $(,)?) => {
        return Err(syn::Error::new_spanned($span, format_args!($msg)))
    };
}
use bail;
