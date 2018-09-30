//! Procedural Macro glue for compile-time compilation of GLSL into SPIR-V
//!
//! # Basic usage
//!
//! ```
//! #[macro_use]
//! extern crate vulkano_shader_derive;
//! extern crate vulkano;
//! # fn main() {}
//! #[allow(unused)]
//! mod vertex_shader {
//!     #[derive(VulkanoShader)]
//!     #[ty = "vertex"]
//!     #[src = "
//! #version 450
//!
//! layout(location = 0) in vec3 position;
//!
//! void main() {
//!     gl_Position = vec4(position, 1.0);
//! }
//! "]
//!     struct Dummy;
//! }
//! ```
//!
//! # Details
//!
//! Due to the current limitations of procedural shaders in Rust, the current
//! functionality of this crate is to base everything off of deriving
//! `VulkanoShader` for a dummy struct that never actually gets used. When
//! derived, the unused struct itself will be replaced by the functionality
//! needed to use the shader in a Vulkano application. Due to the fact that
//! a lot of what is generated will never be used, it's a good idea to put
//! `#[allow(unused)]` on the module itself if you don't want to see irrelevant
//! errors.
//! 
//! If you want to take a look at what the macro generates, your best options
//! are to either read through the code that handles the generation (the
//! [`reflect`][reflect] function in the `vulkano-shaders` crate) or use a tool
//! such as [cargo-expand][cargo-expand] to view the expansion of the macro in your
//! own code. It is unfortunately not possible to provide a `generated_example`
//! module like some normal macro crates do since derive macros cannot be used from
//! the crate they are declared in. On the other hand, if you are looking for a
//! high-level overview, you can see the below section.
//! 
//! # Generated code overview
//! 
//! The macro generates the following items of interest:
//! * The `Shader` struct. This contains a single field, `shader`, which is an
//! `Arc<ShaderModule>`.
//! * The `Shader::load` constructor. This method takes an `Arc<Device>`, calls
//! [`ShaderModule::new`][ShaderModule::new] with the passed-in device and the
//! shader data provided via the macro, and returns `Result<Shader, OomError>`.
//! Before doing so, it loops through every capability instruction in the shader
//! data, verifying that the passed-in `Device` has the appropriate features
//! enabled. **This function currently panics if a feature required by the shader
//! is not enabled on the device.** At some point in the future it will return
//! an error instead.
//! * The `Shader::module` method. This method simply returns a reference to the
//! `Arc<ShaderModule>` contained within the `shader` field of the `Shader`
//! struct.
//! * Methods for each entry point of the shader module. These construct and
//! return the various entry point structs that can be found in the
//! [vulkano::pipeline::shader][pipeline::shader] module.
//! * A Rust struct translated from each struct contained in the shader data.
//! * The `Layout` newtype. This contains a [`ShaderStages`][ShaderStages] struct.
//! An implementation of [`PipelineLayoutDesc`][PipelineLayoutDesc] is also
//! generated for the newtype.
//! * The `SpecializationConstants` struct. This contains a field for every
//! specialization constant found in the shader data. Implementations of
//! `Default` and [`SpecializationConstants`][SpecializationConstants] are also
//! generated for the struct.
//! 
//! All of these generated items will be accessed through the module that you
//! wrote to use the derive macro in. If you wanted to store the `Shader` in
//! a struct of your own, you could do something like this:
//! 
//! ```
//! # #[macro_use]
//! # extern crate vulkano_shader_derive;
//! # extern crate vulkano;
//! # fn main() {}
//! # use std::sync::Arc;
//! # use vulkano::OomError;
//! # use vulkano::device::Device;
//! #
//! # #[allow(unused)]
//! # mod vertex_shader {
//! #     #[derive(VulkanoShader)]
//! #     #[ty = "vertex"]
//! #     #[src = "
//! # #version 450
//! #
//! # layout(location = 0) in vec3 position;
//! #
//! # void main() {
//! #     gl_Position = vec4(position, 1.0);
//! # }
//! # "]
//! #     struct Dummy;
//! # }
//! // various use statements
//! // `vertex_shader` module with shader derive
//! 
//! pub struct Shaders {
//!     pub vertex_shader: vertex_shader::Shader
//! }
//! 
//! impl Shaders {
//!     pub fn load(device: Arc<Device>) -> Result<Self, OomError> {
//!         Ok(Self {
//!             vertex_shader: vertex_shader::Shader::load(device)?,
//!         })
//!     }
//! }
//! ```
//! 
//! # Options
//! 
//! The options available are in the form of the following attributes:
//!
//! ## `#[ty = "..."]`
//!
//! This defines what shader type the given GLSL source will be compiled into.
//! The type can be any of the following:
//!
//! * `vertex`
//! * `fragment`
//! * `geometry`
//! * `tess_ctrl`
//! * `tess_eval`
//! * `compute`
//!
//! For details on what these shader types mean, [see Vulkano's documentation][pipeline].
//!
//! ## `#[src = "..."]`
//!
//! Provides the raw GLSL source to be compiled in the form of a string. Cannot
//! be used in conjunction with the `#[path]` attribute.
//!
//! ## `#[path = "..."]`
//!
//! Provides the path to the GLSL source to be compiled, relative to `Cargo.toml`.
//! Cannot be used in conjunction with the `#[src]` attribute.
//! 
//! [reflect]: https://github.com/vulkano-rs/vulkano/blob/master/vulkano-shaders/src/lib.rs#L67
//! [cargo-expand]: https://github.com/dtolnay/cargo-expand
//! [ShaderModule::new]: https://docs.rs/vulkano/*/vulkano/pipeline/shader/struct.ShaderModule.html#method.new
//! [OomError]: https://docs.rs/vulkano/*/vulkano/enum.OomError.html
//! [pipeline::shader]: https://docs.rs/vulkano/*/vulkano/pipeline/shader/index.html
//! [descriptor]: https://docs.rs/vulkano/*/vulkano/descriptor/index.html
//! [ShaderStages]: https://docs.rs/vulkano/*/vulkano/descriptor/descriptor/struct.ShaderStages.html
//! [PipelineLayoutDesc]: https://docs.rs/vulkano/*/vulkano/descriptor/pipeline_layout/trait.PipelineLayoutDesc.html
//! [SpecializationConstants]: https://docs.rs/vulkano/*/vulkano/pipeline/shader/trait.SpecializationConstants.html
//! [pipeline]: https://docs.rs/vulkano/*/vulkano/pipeline/index.html

extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;
extern crate vulkano_shaders;

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;

enum SourceKind {
    Src(String),
    Path(String),
}

#[proc_macro_derive(VulkanoShader, attributes(src, path, ty))]
pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let syn_item: syn::DeriveInput = syn::parse(input).unwrap();

    let source_code = {
        let mut iter = syn_item.attrs.iter().filter_map(|attr| {
            attr.interpret_meta().and_then(|meta| {
                match meta {
                    syn::Meta::NameValue(syn::MetaNameValue { ident, lit: syn::Lit::Str(lit_str), .. }) => {
                        match ident.to_string().as_ref() {
                            "src"  => Some(SourceKind::Src(lit_str.value())),
                            "path" => Some(SourceKind::Path(lit_str.value())),
                            _      => None,
                        }
                    },

                    _ => None
                }
            })
        });

        let source = iter.next().expect("No source attribute given ; put #[src = \"...\"] or #[path = \"...\"]");

        if iter.next().is_some() {
            panic!("Multiple src or path attributes given ; please provide only one");
        }

        match source {
            SourceKind::Src(source) => source,

            SourceKind::Path(path) => {
                let root = env::var("CARGO_MANIFEST_DIR").unwrap_or(".".into());
                let full_path = Path::new(&root).join(&path);

                if full_path.is_file() {
                    let mut buf = String::new();
                    File::open(full_path)
                        .and_then(|mut file| file.read_to_string(&mut buf))
                        .expect(&format!("Error reading source from {:?}", path));
                    buf
                } else {
                    panic!("File {:?} was not found ; note that the path must be relative to your Cargo.toml", path);
                }
            }
        }
    };

    let ty_str = syn_item.attrs.iter().filter_map(|attr| {
        attr.interpret_meta().and_then(|meta| {
            match meta {
                syn::Meta::NameValue(syn::MetaNameValue { ident, lit: syn::Lit::Str(lit_str), .. }) => {
                    match ident.to_string().as_ref() {
                        "ty" => Some(lit_str.value()),
                        _    => None
                    }
                }

                _ => None
            }
        })
    }).next().expect("Can't find `ty` attribute ; put #[ty = \"vertex\"] for example.");

    let ty = match &ty_str[..] {
        "vertex" => vulkano_shaders::ShaderKind::Vertex,
        "fragment" => vulkano_shaders::ShaderKind::Fragment,
        "geometry" => vulkano_shaders::ShaderKind::Geometry,
        "tess_ctrl" => vulkano_shaders::ShaderKind::TessControl,
        "tess_eval" => vulkano_shaders::ShaderKind::TessEvaluation,
        "compute" => vulkano_shaders::ShaderKind::Compute,
        _ => panic!("Unexpected shader type ; valid values: vertex, fragment, geometry, tess_ctrl, tess_eval, compute")
    };
    let content = vulkano_shaders::compile(&source_code, ty).unwrap();
    
    vulkano_shaders::reflect("Shader", content.as_binary()).unwrap().into()
}
