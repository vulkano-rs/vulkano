//! The procedural macro for vulkano's shader system.
//! Manages the compile-time compilation of GLSL into SPIR-V and generation of associated Rust
//! code.
//!
//! # Basic usage
//!
//! ```
//! mod vs {
//!     vulkano_shaders::shader! {
//!         ty: "vertex",
//!         src: r"
//!             #version 450
//!
//!             layout(location = 0) in vec3 position;
//!
//!             void main() {
//!                 gl_Position = vec4(position, 1.0);
//!             }
//!         ",
//!     }
//! }
//! # fn main() {}
//! ```
//!
//! # Details
//!
//! If you want to take a look at what the macro generates, your best option is to use
//! [cargo-expand] to view the expansion of the macro in your own code. On the other hand, if you
//! are looking for a high-level overview, you can see the below section.
//!
//! # Generated code overview
//!
//! The macro generates the following items of interest:
//!
//! - The `load` constructor. This function takes an `Arc<Device>`, constructs a [`ShaderModule`]
//!   with the passed-in device and the shader data provided via the macro, and returns
//!   `Result<Arc<ShaderModule>, Validated<VulkanError>>`. Before doing so, it checks every
//!   capability instruction in the shader data, verifying that the passed-in `Device` has the
//!   appropriate features enabled.
//! - If the `shaders` option is used, then instead of one `load` constructor, there is one for
//!   each shader. They are named based on the provided names, `load_first`, `load_second` etc.
//! - A Rust struct translated from each struct contained in the shader data. By default, each
//!   structure has a `Clone` and a `Copy` implementation. This behavior could be customized
//!   through the `custom_derives` macro option (see below for details). Each struct also has an
//!   implementation of [`BufferContents`], so that it can be read from/written to a buffer.
//!
//! All of these generated items will be accessed through the module where the macro was invoked.
//! If you wanted to store the `ShaderModule` in a struct of your own, you could do something like
//! this:
//!
//! ```
//! # fn main() {}
//! # use std::sync::Arc;
//! # use vulkano::{device::Device, shader::ShaderModule, Validated, VulkanError};
//! #
//! # mod vs {
//! #     vulkano_shaders::shader!{
//! #         ty: "vertex",
//! #         src: r"
//! #             #version 450
//! #
//! #             layout(location = 0) in vec3 position;
//! #
//! #             void main() {
//! #                 gl_Position = vec4(position, 1.0);
//! #             }
//! #         ",
//! #     }
//! # }
//! // ...various use statements...
//! // ...`vs` module containing a `shader!` call...
//!
//! pub struct Shaders {
//!     pub vs: Arc<ShaderModule>,
//! }
//!
//! impl Shaders {
//!     pub fn load(device: Arc<Device>) -> Result<Self, Validated<VulkanError>> {
//!         Ok(Self {
//!             vs: vs::load(device)?,
//!         })
//!     }
//! }
//! ```
//!
//! # Options
//!
//! The options available are in the form of the following fields:
//!
//! ## `ty: "..."`
//!
//! This defines what shader type the given GLSL source will be compiled into. The type can be any
//! of the following:
//!
//! - `vertex`
//! - `tess_ctrl`
//! - `tess_eval`
//! - `geometry`
//! - `task`
//! - `mesh`
//! - `fragment`
//! - `compute`
//! - `raygen`
//! - `anyhit`
//! - `closesthit`
//! - `miss`
//! - `intersection`
//! - `callable`
//!
//! For details on what these shader types mean, [see Vulkano's documentation][pipeline].
//!
//! ## `src: "..."`
//!
//! Provides the raw GLSL source to be compiled in the form of a string. Cannot be used in
//! conjunction with the `path` or `bytes` field.
//!
//! ## `path: "..."`
//!
//! Provides the path to the GLSL source to be compiled, relative to your `Cargo.toml`. Cannot be
//! used in conjunction with the `src` or `bytes` field.
//!
//! ## `bytes: "..."`
//!
//! Provides the path to precompiled SPIR-V bytecode, relative to your `Cargo.toml`. Cannot be used
//! in conjunction with the `src` or `path` field. This allows using shaders compiled through a
//! separate build system.
//!
//! ## `root_path_env: "..."`
//!
//! Instead of searching relative to your `Cargo.toml`, search relative to some other folder
//! specified by this env variable. The intended use case is using `OUT_DIR` to be able to load
//! shaders generated by your build script. Defaults to `CARGO_MANIFEST_DIR` corresponding to the
//! folder of your `Cargo.toml`.
//!
//! See [`cargo-env-vars`] for a full set of env variables set by cargo. It is also possible to
//! specify env variables from within the build script using the following:
//! ```rust
//! # let shader_out_dir = "";
//! println!("cargo:rustc-env=SHADER_OUT_DIR={shader_out_dir}");
//! ```
//!
//! ## `shaders: { first: { src: "...", ty: "..." }, ... }`
//!
//! With these options the user can compile several shaders in a single macro invocation. Each
//! entry key will be the suffix of the generated `load` function (`load_first` in this case).
//! However, all other Rust structs translated from the shader source will be shared between
//! shaders. The macro checks that the source structs with the same names between different shaders
//! have the same declaration signature, and throws a compile-time error if they don't.
//!
//! Each entry expects a `src`, `path`, `bytes`, and `ty` pairs same as above.
//!
//! ## `include: ["...", "...", ...]`
//!
//! Specifies the standard include directories to be searched through when using the
//! `#include <...>` directive within a shader source. Include directories can be absolute or
//! relative to your `Cargo.toml`. If `path` was specified, relative paths can also be used
//! (`#include "..."`), without the need to specify one or more standard include directories.
//! Relative paths are relative to the directory which contains the source file the
//! `#include "..."` directive is declared in.
//!
//! ## `define: [("NAME", "VALUE"), ...]`
//!
//! Adds the given macro definitions to the pre-processor. This is equivalent to passing the
//! `-DNAME=VALUE` argument on the command line.
//!
//! ## `vulkan_version: "major.minor"` and `spirv_version: "major.minor"`
//!
//! Sets the Vulkan and SPIR-V versions to compile into, respectively. These map directly to the
//! [`set_target_env`] and [`set_target_spirv`] compile options. If neither option is specified,
//! then SPIR-V 1.0 code targeting Vulkan 1.0 will be generated.
//!
//! The generated code must be supported by the device at runtime. If not, then an error will be
//! returned when calling `load`.
//!
//! ## `generate_structs: true`
//!
//! Generate rust structs that represent the structs contained in the shader. They all implement
//! [`BufferContents`], which allows then to be passed to the shader, without having to worry about
//! the layout of the struct manually. However, some use-cases, such as Rust-GPU, may not have any
//! use for such structs, and may choose to disable them.
//!
//! ## `custom_derives: [Clone, Default, PartialEq, ...]`
//!
//! Extends the list of derive macros that are added to the `derive` attribute of Rust structs that
//! represent shader structs.
//!
//! By default, each generated struct derives `Clone` and `Copy`. If the struct has unsized members
//! none of the derives are applied on the struct, except [`BufferContents`], which is always
//! derived.
//!
//! ## `linalg_type: "..."`
//!
//! Specifies the way that linear algebra types should be generated. It can be any of the
//! following:
//!
//! - `std`
//! - `cgmath`
//! - `nalgebra`
//!
//! The default is `std`, which uses arrays to represent vectors and matrices. Note that if the
//! chosen crate doesn't have a type that represents a certain linear algebra type (e.g. `mat3`, or
//! a rectangular matrix) then the macro will default back to arrays for that type.
//!
//! If you use linear algebra types from a third-party crate, then you have to have the crate in
//! your dependencies with the appropriate feature enabled that adds `bytemuck` support.
//!
//! ## `dump: true`
//!
//! The crate fails to compile but prints the generated Rust code to stdout.
//!
//! # Cargo features
//!
//! | Feature                     | Description                                             |
//! |-----------------------------|---------------------------------------------------------|
//! | `shaderc-build-from-source` | Build the `shaderc` library from source when compiling. |
//! | `shaderc-debug`             | Compile shaders with debug information included.        |
//!
//! [`cargo-env-vars`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html
//! [cargo-expand]: https://github.com/dtolnay/cargo-expand
//! [`ShaderModule`]: vulkano::shader::ShaderModule
//! [pipeline]: vulkano::pipeline
//! [`set_target_env`]: shaderc::CompileOptions::set_target_env
//! [`set_target_spirv`]: shaderc::CompileOptions::set_target_spirv
//! [`BufferContents`]: vulkano::buffer::BufferContents

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![recursion_limit = "1024"]
#![allow(clippy::needless_borrowed_reference)]
#![warn(rust_2018_idioms, rust_2021_compatibility)]

use crate::codegen::ShaderKind;
use ahash::HashMap;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use shaderc::{EnvVersion, SpirvVersion};
use std::{
    env, fs, mem,
    path::{Path, PathBuf},
};
use structs::TypeRegistry;
use syn::{
    braced, bracketed, parenthesized,
    parse::{Parse, ParseStream, Result},
    parse_macro_input, parse_quote, Error, Ident, LitBool, LitStr, Path as SynPath, Token,
};

mod codegen;
mod rust_gpu;
mod structs;

#[proc_macro]
pub fn shader(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as MacroInput);

    shader_inner(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn shader_inner(mut input: MacroInput) -> Result<TokenStream> {
    let (root, relative_path_error_msg) = match input.root_path_env.as_ref() {
        None => (
            env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()),
            "to your Cargo.toml".to_owned(),
        ),
        Some(root_path_env) => {
            let root = match env::var(root_path_env.value()) {
                Ok(e) => e,
                Err(e) => {
                    bail!(
                        root_path_env,
                        "failed to fetch environment variable: {e}; typical parameters are \
                        `OUT_DIR` to gather results from your build script, or left default to \
                        search relative to your Cargo.toml",
                    )
                }
            };
            let env = root_path_env.value();
            let error = format!("to the path `{root}` specified by the env variable `{env:?}`");
            (root, error)
        }
    };

    let root_path = Path::new(&root);
    let shaders = mem::take(&mut input.shaders); // yoink

    let mut shaders_code = Vec::with_capacity(shaders.len());
    let mut types_code = Vec::with_capacity(shaders.len());
    let mut type_registry = TypeRegistry::default();

    for (name, (shader_kind, source_kind)) in shaders {
        let (code, types) = match source_kind {
            SourceKind::Src(source) => {
                let (artifact, includes) =
                    codegen::compile(&input, None, root_path, &source.value(), shader_kind)
                        .map_err(|err| Error::new_spanned(&source, err))?;

                let words = artifact.as_binary();

                codegen::reflect(&input, source, name, words, includes, &mut type_registry)?
            }
            SourceKind::Path(path) => {
                let full_path = root_path.join(path.value());

                if !full_path.is_file() {
                    bail!(
                        path,
                        "file `{full_path:?}` was not found, note that the path must be relative \
                        {relative_path_error_msg}",
                    );
                }

                let source_code = fs::read_to_string(&full_path)
                    .or_else(|err| bail!(path, "failed to read source `{full_path:?}`: {err}"))?;

                let (artifact, mut includes) = codegen::compile(
                    &input,
                    Some(path.value()),
                    root_path,
                    &source_code,
                    shader_kind,
                )
                .map_err(|err| Error::new_spanned(&path, err))?;

                let words = artifact.as_binary();

                includes.push(full_path.into_os_string().into_string().unwrap());

                codegen::reflect(&input, path, name, words, includes, &mut type_registry)?
            }
            SourceKind::Bytes(path) => {
                let full_path = root_path.join(path.value());

                if !full_path.is_file() {
                    bail!(
                        path,
                        "file `{full_path:?}` was not found, note that the path must be relative \
                        {relative_path_error_msg}",
                    );
                }

                let bytes = fs::read(&full_path)
                    .or_else(|err| bail!(path, "failed to read source `{full_path:?}`: {err}"))?;

                let words = vulkano::shader::spirv::bytes_to_words(&bytes)
                    .or_else(|err| bail!(path, "failed to read source `{full_path:?}`: {err}"))?;

                codegen::reflect(&input, path, name, &words, Vec::new(), &mut type_registry)?
            }
        };

        shaders_code.push(code);
        types_code.push(types);
    }

    let result = quote! {
        #( #shaders_code )*
        #( #types_code )*
    };

    if input.dump.value {
        println!("{}", result);
        bail!(input.dump, "`shader!` Rust codegen dumped");
    }

    Ok(result)
}

enum SourceKind {
    Src(LitStr),
    Path(LitStr),
    Bytes(LitStr),
}

struct MacroInput {
    root_path_env: Option<LitStr>,
    include_directories: Vec<PathBuf>,
    macro_defines: Vec<(String, String)>,
    shaders: HashMap<String, (ShaderKind, SourceKind)>,
    spirv_version: Option<SpirvVersion>,
    vulkan_version: Option<EnvVersion>,
    generate_structs: bool,
    custom_derives: Vec<SynPath>,
    linalg_type: LinAlgType,
    dump: LitBool,
}

impl MacroInput {
    #[cfg(test)]
    fn empty() -> Self {
        MacroInput {
            root_path_env: None,
            include_directories: Vec::new(),
            macro_defines: Vec::new(),
            shaders: HashMap::default(),
            vulkan_version: None,
            spirv_version: None,
            generate_structs: true,
            custom_derives: Vec::new(),
            linalg_type: LinAlgType::default(),
            dump: LitBool::new(false, Span::call_site()),
        }
    }
}

impl Parse for MacroInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let root = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());

        let mut root_path_env = None;
        let mut include_directories = Vec::new();
        let mut macro_defines = Vec::new();
        let mut shaders = HashMap::default();
        let mut vulkan_version = None;
        let mut spirv_version = None;
        let mut generate_structs = true;
        let mut custom_derives = None;
        let mut linalg_type = None;
        let mut dump = None;

        fn parse_shader_fields(
            output: &mut (Option<ShaderKind>, Option<SourceKind>),
            name: &str,
            input: ParseStream<'_>,
        ) -> Result<()> {
            match name {
                "ty" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.0.is_some() {
                        bail!(lit, "field `ty` is already defined");
                    }

                    output.0 = Some(match lit.value().as_str() {
                        "vertex" => ShaderKind::Vertex,
                        "tess_ctrl" => ShaderKind::TessControl,
                        "tess_eval" => ShaderKind::TessEvaluation,
                        "geometry" => ShaderKind::Geometry,
                        "task" => ShaderKind::Task,
                        "mesh" => ShaderKind::Mesh,
                        "fragment" => ShaderKind::Fragment,
                        "compute" => ShaderKind::Compute,
                        "raygen" => ShaderKind::RayGeneration,
                        "anyhit" => ShaderKind::AnyHit,
                        "closesthit" => ShaderKind::ClosestHit,
                        "miss" => ShaderKind::Miss,
                        "intersection" => ShaderKind::Intersection,
                        "callable" => ShaderKind::Callable,
                        ty => bail!(
                            lit,
                            "expected `vertex`, `tess_ctrl`, `tess_eval`, `geometry`, `task`, \
                            `mesh`, `fragment` `compute`, `raygen`, `anyhit`, `closesthit`, \
                            `miss`, `intersection` or `callable`, found `{ty}`",
                        ),
                    });
                }
                "bytes" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.1.is_some() {
                        bail!(
                            lit,
                            "only one of `src`, `path`, or `bytes` can be defined per shader entry",
                        );
                    }

                    output.1 = Some(SourceKind::Bytes(lit));
                }
                "path" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.1.is_some() {
                        bail!(
                            lit,
                            "only one of `src`, `path` or `bytes` can be defined per shader entry",
                        );
                    }

                    output.1 = Some(SourceKind::Path(lit));
                }
                "src" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.1.is_some() {
                        bail!(
                            lit,
                            "only one of `src`, `path` or `bytes` can be defined per shader entry",
                        );
                    }

                    output.1 = Some(SourceKind::Src(lit));
                }
                _ => unreachable!(),
            }

            Ok(())
        }

        while !input.is_empty() {
            let field_ident = input.parse::<Ident>()?;
            input.parse::<Token![:]>()?;
            let field = field_ident.to_string();

            match field.as_str() {
                "bytes" | "src" | "path" | "ty" => {
                    if shaders.len() > 1 || (shaders.len() == 1 && !shaders.contains_key("")) {
                        bail!(
                            field_ident,
                            "only one of `src`, `path`, `bytes` or `shaders` can be defined",
                        );
                    }

                    parse_shader_fields(shaders.entry(String::new()).or_default(), &field, input)?;
                }
                "shaders" => {
                    if !shaders.is_empty() {
                        bail!(
                            field_ident,
                            "only one of `src`, `path`, `bytes` or `shaders` can be defined",
                        );
                    }

                    let in_braces;
                    braced!(in_braces in input);

                    while !in_braces.is_empty() {
                        let name_ident = in_braces.parse::<Ident>()?;
                        let name = name_ident.to_string();

                        if shaders.contains_key(&name) {
                            bail!(name_ident, "shader entry `{name}` is already defined");
                        }

                        in_braces.parse::<Token![:]>()?;

                        let in_shader_definition;
                        braced!(in_shader_definition in in_braces);

                        while !in_shader_definition.is_empty() {
                            let field_ident = in_shader_definition.parse::<Ident>()?;
                            in_shader_definition.parse::<Token![:]>()?;
                            let field = field_ident.to_string();

                            match field.as_str() {
                                "bytes" | "src" | "path" | "ty" => {
                                    parse_shader_fields(
                                        shaders.entry(name.clone()).or_default(),
                                        &field,
                                        &in_shader_definition,
                                    )?;
                                }
                                field => bail!(
                                    field_ident,
                                    "expected `bytes`, `src`, `path` or `ty` as a field, found \
                                    `{field}`",
                                ),
                            }

                            if !in_shader_definition.is_empty() {
                                in_shader_definition.parse::<Token![,]>()?;
                            }
                        }

                        if !in_braces.is_empty() {
                            in_braces.parse::<Token![,]>()?;
                        }

                        match shaders.get(&name).unwrap() {
                            (None, _) => bail!(
                                "please specify a type for shader `{name}` e.g. `ty: \"vertex\"`",
                            ),
                            (_, None) => bail!(
                                "please specify a source for shader `{name}` e.g. \
                                `path: \"entry_point.glsl\"`",
                            ),
                            _ => (),
                        }
                    }

                    if shaders.is_empty() {
                        bail!("at least one shader entry must be defined");
                    }
                }
                "define" => {
                    let array_input;
                    bracketed!(array_input in input);

                    while !array_input.is_empty() {
                        let tuple_input;
                        parenthesized!(tuple_input in array_input);

                        let name = tuple_input.parse::<LitStr>()?;
                        tuple_input.parse::<Token![,]>()?;
                        let value = tuple_input.parse::<LitStr>()?;
                        macro_defines.push((name.value(), value.value()));

                        if !array_input.is_empty() {
                            array_input.parse::<Token![,]>()?;
                        }
                    }
                }
                "root_path_env" => {
                    let lit = input.parse::<LitStr>()?;
                    if root_path_env.is_some() {
                        bail!(lit, "field `root_path_env` is already defined");
                    }
                    root_path_env = Some(lit);
                }
                "include" => {
                    let in_brackets;
                    bracketed!(in_brackets in input);

                    while !in_brackets.is_empty() {
                        let path = in_brackets.parse::<LitStr>()?;

                        include_directories.push([&root, &path.value()].into_iter().collect());

                        if !in_brackets.is_empty() {
                            in_brackets.parse::<Token![,]>()?;
                        }
                    }
                }
                "vulkan_version" => {
                    let lit = input.parse::<LitStr>()?;
                    if vulkan_version.is_some() {
                        bail!(lit, "field `vulkan_version` is already defined");
                    }

                    vulkan_version = Some(match lit.value().as_str() {
                        "1.0" => EnvVersion::Vulkan1_0,
                        "1.1" => EnvVersion::Vulkan1_1,
                        "1.2" => EnvVersion::Vulkan1_2,
                        "1.3" => EnvVersion::Vulkan1_3,
                        ver => bail!(lit, "expected `1.0`, `1.1`, `1.2` or `1.3`, found `{ver}`"),
                    });
                }
                "spirv_version" => {
                    let lit = input.parse::<LitStr>()?;
                    if spirv_version.is_some() {
                        bail!(lit, "field `spirv_version` is already defined");
                    }

                    spirv_version = Some(match lit.value().as_str() {
                        "1.0" => SpirvVersion::V1_0,
                        "1.1" => SpirvVersion::V1_1,
                        "1.2" => SpirvVersion::V1_2,
                        "1.3" => SpirvVersion::V1_3,
                        "1.4" => SpirvVersion::V1_4,
                        "1.5" => SpirvVersion::V1_5,
                        "1.6" => SpirvVersion::V1_6,
                        ver => bail!(
                            lit,
                            "expected `1.0`, `1.1`, `1.2`, `1.3`, `1.4`, `1.5` or `1.6`, found \
                            `{ver}`",
                        ),
                    });
                }
                "generate_structs" => {
                    let lit = input.parse::<LitBool>()?;
                    generate_structs = lit.value;
                }
                "custom_derives" => {
                    let in_brackets;
                    bracketed!(in_brackets in input);

                    while !in_brackets.is_empty() {
                        if custom_derives.is_none() {
                            custom_derives = Some(Vec::new());
                        }

                        custom_derives
                            .as_mut()
                            .unwrap()
                            .push(in_brackets.parse::<SynPath>()?);

                        if !in_brackets.is_empty() {
                            in_brackets.parse::<Token![,]>()?;
                        }
                    }
                }
                "types_meta" => {
                    bail!(
                        field_ident,
                        "you no longer need to add any derives to use the generated structs in \
                        buffers, and you also no longer need bytemuck as a dependency, because \
                        `BufferContents` is derived automatically for the generated structs; if \
                        you need to add additional derives (e.g. `Debug`, `PartialEq`) then please \
                        use the `custom_derives` field of the macro",
                    );
                }
                "linalg_type" => {
                    let lit = input.parse::<LitStr>()?;
                    if linalg_type.is_some() {
                        bail!(lit, "field `linalg_type` is already defined");
                    }

                    linalg_type = Some(match lit.value().as_str() {
                        "std" => LinAlgType::Std,
                        "cgmath" => LinAlgType::CgMath,
                        "nalgebra" => LinAlgType::Nalgebra,
                        ty => bail!(lit, "expected `std`, `cgmath` or `nalgebra`, found `{ty}`"),
                    });
                }
                "dump" => {
                    let lit = input.parse::<LitBool>()?;
                    if dump.is_some() {
                        bail!(lit, "field `dump` is already defined");
                    }

                    dump = Some(lit);
                }
                field => bail!(
                    field_ident,
                    "expected `bytes`, `src`, `path`, `ty`, `shaders`, `define`, `include`, \
                    `vulkan_version`, `spirv_version`, `generate_structs`, `custom_derives`, \
                    `linalg_type` or `dump` as a field, found `{field}`",
                ),
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        if shaders.is_empty() {
            bail!(r#"please specify at least one shader e.g. `ty: "vertex", src: "<GLSL code>"`"#);
        }

        match shaders.get("") {
            Some((None, _)) => {
                bail!(r#"please specify the type of the shader e.g. `ty: "vertex"`"#);
            }
            Some((_, None)) => {
                bail!(r#"please specify the source of the shader e.g. `src: "<GLSL code>"`"#);
            }
            _ => {}
        }

        Ok(MacroInput {
            root_path_env,
            include_directories,
            macro_defines,
            shaders: shaders
                .into_iter()
                .map(|(key, (shader_kind, shader_source))| {
                    (key, (shader_kind.unwrap(), shader_source.unwrap()))
                })
                .collect(),
            vulkan_version,
            spirv_version,
            generate_structs,
            custom_derives: custom_derives.unwrap_or_else(|| {
                vec![
                    parse_quote! { ::std::clone::Clone },
                    parse_quote! { ::std::marker::Copy },
                ]
            }),
            linalg_type: linalg_type.unwrap_or_default(),
            dump: dump.unwrap_or_else(|| LitBool::new(false, Span::call_site())),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LinAlgType {
    #[default]
    Std,
    CgMath,
    Nalgebra,
}

macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!($msg),
        ))
    };
    ($span:expr, $msg:literal $(,)?) => {
        return Err(syn::Error::new_spanned(&$span, format!($msg)))
    };
}
use bail;
