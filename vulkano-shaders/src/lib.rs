//! The procedural macro for vulkano's shader system.
//! Manages the compile-time compilation of GLSL into SPIR-V and generation of assosciated rust code.
//!
//! # Basic usage
//!
//! ```
//! mod vs {
//!     vulkano_shaders::shader!{
//!         ty: "vertex",
//!         src: "
//! #version 450
//!
//! layout(location = 0) in vec3 position;
//!
//! void main() {
//!     gl_Position = vec4(position, 1.0);
//! }"
//!     }
//! }
//! # fn main() {}
//! ```
//!
//! # Details
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
//! All of these generated items will be accessed through the module specified
//! by `mod_name: foo` If you wanted to store the `Shader` in a struct of your own,
//! you could do something like this:
//!
//! ```
//! # fn main() {}
//! # use std::sync::Arc;
//! # use vulkano::OomError;
//! # use vulkano::device::Device;
//! #
//! # mod vs {
//! #     vulkano_shaders::shader!{
//! #         ty: "vertex",
//! #         src: "
//! # #version 450
//! #
//! # layout(location = 0) in vec3 position;
//! #
//! # void main() {
//! #     gl_Position = vec4(position, 1.0);
//! # }"
//! #     }
//! # }
//! // various use statements
//! // `vertex_shader` module with shader derive
//!
//! pub struct Shaders {
//!     pub vs: vs::Shader
//! }
//!
//! impl Shaders {
//!     pub fn load(device: Arc<Device>) -> Result<Self, OomError> {
//!         Ok(Self {
//!             vs: vs::Shader::load(device)?,
//!         })
//!     }
//! }
//! ```
//!
//! # Options
//!
//! The options available are in the form of the following attributes:
//!
//! ## `ty: "..."`
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
//! ## `src: "..."`
//!
//! Provides the raw GLSL source to be compiled in the form of a string. Cannot
//! be used in conjunction with the `path` field.
//!
//! ## `path: "..."`
//!
//! Provides the path to the GLSL source to be compiled, relative to `Cargo.toml`.
//! Cannot be used in conjunction with the `src` field.
//!
//! ## `include: ["...", "...", ..., "..."]`
//!
//! Specifies the standard include directories to be searched through when using the
//! `#include <...>` directive within a shader source. Include directories can be absolute
//! or relative to `Cargo.toml`.
//! If `path` was specified, relative paths can also be used (`#include "..."`), without the need
//! to specify one or more standard include directories. Relative paths are relative to the
//! directory, which contains the source file the `#include "..."` directive is declared in.
//!
//! ## `define: [("NAME", "VALUE"), ...]`
//!
//! Adds the given macro definitions to the pre-processor. This is equivalent to passing `-DNAME=VALUE`
//! on the command line.
//!
//! ## `dump: true`
//!
//! The crate fails to compile but prints the generated rust code to stdout.
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

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]

#![recursion_limit = "1024"]
#[macro_use] extern crate quote;
#[macro_use] extern crate syn;
             extern crate proc_macro;

use std::env;
use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;

use syn::parse::{Parse, ParseStream, Result};
use syn::{Ident, LitStr, LitBool};

mod codegen;
mod descriptor_sets;
mod entry_point;
mod enums;
mod parse;
mod spec_consts;
mod structs;
mod spirv_search;

use crate::codegen::ShaderKind;

enum SourceKind {
    Src(String),
    Path(String),
}

struct MacroInput {
    shader_kind: ShaderKind,
    source_kind: SourceKind,
    include_directories: Vec<String>,
    macro_defines: Vec<(String, String)>,
    dump: bool,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut dump = None;
        let mut shader_kind = None;
        let mut source_kind = None;
        let mut include_directories = Vec::new();
        let mut macro_defines = Vec::new();

        while !input.is_empty() {
            let name: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match name.to_string().as_ref() {
                "ty" => {
                    if shader_kind.is_some() {
                        panic!("Only one `ty` can be defined")
                    }

                    let ty: LitStr = input.parse()?;
                    let ty = match ty.value().as_ref() {
                        "vertex" => ShaderKind::Vertex,
                        "fragment" => ShaderKind::Fragment,
                        "geometry" => ShaderKind::Geometry,
                        "tess_ctrl" => ShaderKind::TessControl,
                        "tess_eval" => ShaderKind::TessEvaluation,
                        "compute" => ShaderKind::Compute,
                        _ => panic!("Unexpected shader type, valid values: vertex, fragment, geometry, tess_ctrl, tess_eval, compute")
                    };
                    shader_kind = Some(ty);
                }
                "src" => {
                    if source_kind.is_some() {
                        panic!("Only one `src` or `path` can be defined")
                    }

                    let src: LitStr = input.parse()?;
                    source_kind = Some(SourceKind::Src(src.value()));
                }
                "path" => {
                    if source_kind.is_some() {
                        panic!("Only one `src` or `path` can be defined")
                    }

                    let path: LitStr = input.parse()?;
                    source_kind = Some(SourceKind::Path(path.value()));
                }
                "define" => {
                    let array_input;
                    bracketed!(array_input in input);

                    while !array_input.is_empty() {
                        let tuple_input;
                        parenthesized!(tuple_input in array_input);

                        let name: LitStr = tuple_input.parse()?;
                        tuple_input.parse::<Token![,]>()?;
                        let value: LitStr = tuple_input.parse()?;
                        macro_defines.push((name.value(), value.value()));

                        if !array_input.is_empty() {
                            array_input.parse::<Token![,]>()?;
                        }
                    }
                }
                "include" => {
                    let in_brackets;
                    bracketed!(in_brackets in input);

                    while !in_brackets.is_empty() {
                        let path: LitStr = in_brackets.parse()?;

                        include_directories.push(path.value());

                        if !in_brackets.is_empty() {
                            in_brackets.parse::<Token![,]>()?;
                        }
                    }
                }
                "dump" => {
                    if dump.is_some() {
                        panic!("Only one `dump` can be defined")
                    }
                    let dump_lit: LitBool = input.parse()?;
                    dump = Some(dump_lit.value);
                }
                name => panic!(format!("Unknown field name: {}", name))
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let shader_kind = match shader_kind {
            Some(shader_kind) => shader_kind,
            None => panic!("Please provide a shader type e.g. `ty: \"vertex\"`")
        };

        let source_kind = match source_kind {
            Some(source_kind) => source_kind,
            None => panic!("Please provide a source e.g. `path: \"foo.glsl\"` or `src: \"glsl source code here ...\"`")
        };

        let dump = dump.unwrap_or(false);

        Ok(MacroInput { shader_kind, source_kind, include_directories, dump, macro_defines })
    }
}

pub(self) fn read_file_to_string(full_path: &Path) -> IoResult<String> {
    let mut buf = String::new();
    File::open(full_path)
        .and_then(|mut file| file.read_to_string(&mut buf))?;
    Ok(buf)
}

#[proc_macro]
pub fn shader(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    let root = env::var("CARGO_MANIFEST_DIR").unwrap_or(".".into());
    let root_path = Path::new(&root);

    let (path, source_code) = match input.source_kind {
        SourceKind::Src(source) => (None, source),
        SourceKind::Path(path) => (Some(path.clone()), {
            let full_path = root_path.join(&path);

            if full_path.is_file() {
                read_file_to_string(&full_path)
                    .expect(&format!("Error reading source from {:?}", path))
            } else {
                panic!("File {:?} was not found ; note that the path must be relative to your Cargo.toml", path);
            }
        })
    };

    let include_paths = input.include_directories.iter().map(|include_directory| {
        let include_path = Path::new(include_directory);
        let mut full_include_path = root_path.to_owned();
        full_include_path.push(include_path);
        full_include_path
    }).collect::<Vec<_>>();

    let content = match codegen::compile(path, &root_path, &source_code, input.shader_kind, &include_paths, &input.macro_defines) {
        Ok(ok) => ok,
        Err(e) => panic!(e.replace("(s): ", "(s):\n"))
    };

    codegen::reflect("Shader", content.as_binary(), input.dump).unwrap().into()
}
