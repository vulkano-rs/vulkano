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
//! * The `load` constructor. This method takes an `Arc<Device>`, calls
//! [`ShaderModule::new`][ShaderModule::new] with the passed-in device and the shader data provided
//! via the macro, and returns `Result<Arc<ShaderModule>, ShaderCreationError>`.
//! Before doing so, it loops through every capability instruction in the shader
//! data, verifying that the passed-in `Device` has the appropriate features
//! enabled.
//! * If the `shaders` option is used, then instead of one `load` constructor, there is one for each
//! shader. They are named based on the provided names, `load_first`, `load_second` etc.
//! * A Rust struct translated from each struct contained in the shader data.
//! By default each structure has a `Clone` and a `Copy` implementations. This
//! behavior could be customized through the `types_meta` macro option(see below
//! for details).
//! * The `SpecializationConstants` struct. This contains a field for every
//! specialization constant found in the shader data. Implementations of
//! `Default` and [`SpecializationConstants`][SpecializationConstants] are also
//! generated for the struct.
//!
//! All of these generated items will be accessed through the module when the macro was invoked.
//! If you wanted to store the `Shader` in a struct of your own, you could do something like this:
//!
//! ```
//! # fn main() {}
//! # use std::sync::Arc;
//! # use vulkano::shader::{ShaderCreationError, ShaderModule};
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
//!     pub vs: Arc<ShaderModule>,
//! }
//!
//! impl Shaders {
//!     pub fn load(device: Arc<Device>) -> Result<Self, ShaderCreationError> {
//!         Ok(Self {
//!             vs: vs::load(device)?,
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
//! be used in conjunction with the `path` or `bytes` field.
//!
//! ## `path: "..."`
//!
//! Provides the path to the GLSL source to be compiled, relative to `Cargo.toml`.
//! Cannot be used in conjunction with the `src` or `bytes` field.
//!
//! ## `bytes: "..."`
//!
//! Provides the path to precompiled SPIR-V bytecode, relative to `Cargo.toml`.
//! Cannot be used in conjunction with the `src` or `path` field.
//! This allows using shaders compiled through a separate build system.
//! **Note**: If your shader contains multiple entrypoints with different
//! descriptor sets, you may also need to enable `exact_entrypoint_interface`.
//!
//! ## `shaders: { First: {src: "...", ty: "..."}, ... }`
//!
//! With these options the user can compile several shaders at a single macro invocation.
//! Each entry key is a suffix that will be put after the name of the generated `load` function and
//! `SpecializationConstants` struct(`FirstSpecializationConstants` in this case). However all other
//! Rust structs translated from the shader source will be shared between shaders. The macro checks
//! that the source structs with the same names between different shaders have the same declaration
//! signature, and throws a compile-time error if they don't.
//!
//! Each entry values expecting `src`, `path`, `bytes`, and `ty` pairs same as above.
//!
//! Also `SpecializationConstants` can all be shared between shaders by specifying
//! `shared_constants: true,` entry-flag of the `shaders` map. This feature is turned-off by
//! default.
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
//! ## `vulkan_version: "major.minor"` and `spirv_version: "major.minor"`
//!
//! Sets the Vulkan and SPIR-V versions to compile into, respectively. These map directly to the
//! [`set_target_env`](shaderc::CompileOptions::set_target_env) and
//! [`set_target_spirv`](shaderc::CompileOptions::set_target_spirv) compile options.
//! If neither option is specified, then SPIR-V 1.0 code targeting Vulkan 1.0 will be generated.
//!
//! The generated code must be supported by the device at runtime. If not, then an error will be
//! returned when calling `Shader::load`.
//!
//! ## `types_meta: { use a::b; #[derive(Clone, Default, PartialEq ...)] impl Eq }`
//!
//! Extends implementations of Rust structs that represent Shader structs.
//!
//! By default each generated struct has a `Clone` and a `Copy` implementations
//! only. If the struct has unsized members none of derives or impls applied on
//! this struct.
//!
//! The block may have as many `use`, `derive` or `impl` statements as needed
//! and in any order.
//!
//! Each `use` declaration will be added to generated `ty` module. And each
//! `derive`'s trait and `impl` statement will be applied to each generated
//! struct inside `ty` module.
//!
//! For `Default` derive implementation fills a struct data with all zeroes.
//! For `Display` and `Debug` derive implementation prints all fields except `_dummyX`.
//! For `PartialEq` derive implementation all non-`_dummyX` are checking for equality.
//!
//! The macro performs trivial checking for duplicate declarations. To see the
//! final output of generated code the user can also use `dump` macro
//! option(see below).
//!
//! ## `exact_entrypoint_interface: true`
//!
//! By default, the macro assumes that all resources (Uniforms, Storage Buffers,
//! Images, Samplers, etc) need to be bound into a descriptor set, even if they are
//! not used in the shader code. However, shaders with multiple entrypoints may have
//! conflicting descriptor sets for each entrypoint. Enabling this option will force
//! the macro to only generate descriptor information for resources that are used
//! in each entrypoint.
//!
//! The macro determines which resources are used by looking at each entrypoint's
//! interface and bytecode. See [`src/descriptor_sets.rs`][descriptor_sets]
//! for the exact logic.
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
//! [SpecializationConstants]: https://docs.rs/vulkano/*/vulkano/pipeline/shader/trait.SpecializationConstants.html
//! [pipeline]: https://docs.rs/vulkano/*/vulkano/pipeline/index.html
//! [descriptor_sets]: https://github.com/vulkano-rs/vulkano/blob/master/vulkano-shaders/src/descriptor_sets.rs#L142

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![recursion_limit = "1024"]
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;
extern crate proc_macro;

use crate::codegen::ShaderKind;
use shaderc::{EnvVersion, SpirvVersion};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;
use std::slice::from_raw_parts;
use std::{env, iter::empty};
use syn::parse::{Parse, ParseStream, Result};
use syn::{
    Ident, ItemUse, LitBool, LitStr, Meta, MetaList, NestedMeta, Path as SynPath, TypeImplTrait,
};

mod codegen;
mod entry_point;
mod structs;

enum SourceKind {
    Src(String),
    Path(String),
    Bytes(String),
}

struct TypesMeta {
    custom_derives: Vec<SynPath>,
    clone: bool,
    copy: bool,
    display: bool,
    debug: bool,
    default: bool,
    partial_eq: bool,
    uses: Vec<ItemUse>,
    impls: Vec<TypeImplTrait>,
}

impl Default for TypesMeta {
    #[inline]
    fn default() -> Self {
        Self {
            custom_derives: vec![],
            clone: true,
            copy: true,
            partial_eq: false,
            debug: false,
            display: false,
            default: false,
            uses: Vec::new(),
            impls: Vec::new(),
        }
    }
}

impl TypesMeta {
    #[inline]
    fn empty() -> Self {
        Self {
            custom_derives: Vec::new(),
            clone: false,
            copy: false,
            partial_eq: false,
            debug: false,
            display: false,
            default: false,
            uses: Vec::new(),
            impls: Vec::new(),
        }
    }
}

struct RegisteredType {
    shader: String,
    signature: Vec<(String, Cow<'static, str>)>,
}

impl RegisteredType {
    #[inline]
    fn assert_signatures(&self, type_name: &str, target_type: &Self) {
        if self.signature.len() > target_type.signature.len() {
            panic!(
                "Shaders {shader_a:} and {shader_b:} declare structs with the \
                same name \"`{type_name:}\", but the struct from {shader_a:} shader \
                contains extra field \"{field:}\"",
                shader_a = self.shader,
                shader_b = target_type.shader,
                type_name = type_name,
                field = self.signature[target_type.signature.len()].0
            );
        }

        if self.signature.len() < target_type.signature.len() {
            panic!(
                "Shaders {shader_a:} and {shader_b:} declare structs with the \
                same name \"{type_name:}\", but the struct from {shader_b:} shader \
                contains extra field \"{field:}\"",
                shader_a = self.shader,
                shader_b = target_type.shader,
                type_name = type_name,
                field = target_type.signature[self.signature.len()].0
            );
        }

        let comparison = self
            .signature
            .iter()
            .zip(target_type.signature.iter())
            .enumerate();

        for (index, ((a_name, a_type), (b_name, b_type))) in comparison {
            if a_name != b_name || a_type != b_type {
                panic!(
                    "Shaders {shader_a:} and {shader_b:} declare structs with the \
                    same name \"{type_name:}\", but the struct from {shader_a:} shader \
                    contains field \"{a_name:}\" of type \"{a_type:}\" in position {index:}, \
                    whereas the same struct from {shader_b:} contains field \"{b_name:}\" \
                    of type \"{b_type:}\" in the same position",
                    shader_a = self.shader,
                    shader_b = target_type.shader,
                    type_name = type_name,
                    index = index,
                    a_name = a_name,
                    a_type = a_type,
                    b_name = b_name,
                    b_type = b_type,
                );
            }
        }
    }
}

struct MacroInput {
    dump: bool,
    include_directories: Vec<String>,
    macro_defines: Vec<(String, String)>,
    shared_constants: bool,
    shaders: HashMap<String, (ShaderKind, SourceKind)>,
    spirv_version: Option<SpirvVersion>,
    types_meta: TypesMeta,
    vulkan_version: Option<EnvVersion>,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut dump = None;
        let mut exact_entrypoint_interface = None;
        let mut include_directories = Vec::new();
        let mut macro_defines = Vec::new();
        let mut shared_constants = None;
        let mut shaders = HashMap::new();
        let mut spirv_version = None;
        let mut types_meta = None;
        let mut vulkan_version = None;

        fn parse_shader_fields<'k>(
            output: &mut (Option<ShaderKind>, Option<SourceKind>),
            name: &'k str,
            input: ParseStream,
        ) -> Result<()> {
            match name {
                "ty" => {
                    if output.0.is_some() {
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

                    output.0 = Some(ty);
                }

                "bytes" => {
                    if output.1.is_some() {
                        panic!(
                            "Only one of `src`, `path`, or `bytes` can be defined per Shader entry"
                        )
                    }

                    let path: LitStr = input.parse()?;
                    output.1 = Some(SourceKind::Bytes(path.value()));
                }

                "path" => {
                    if output.1.is_some() {
                        panic!(
                            "Only one of `src`, `path`, or `bytes` can be defined per Shader entry"
                        )
                    }

                    let path: LitStr = input.parse()?;
                    output.1 = Some(SourceKind::Path(path.value()));
                }

                "src" => {
                    if output.1.is_some() {
                        panic!("Only one of `src`, `path`, `bytes` can be defined per Shader entry")
                    }

                    let src: LitStr = input.parse()?;
                    output.1 = Some(SourceKind::Src(src.value()));
                }

                other => unreachable!("Unexpected entry key {:?}", other),
            }

            Ok(())
        }

        while !input.is_empty() {
            let name: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let name = name.to_string();

            match name.as_str() {
                "bytes" | "src" | "path" | "ty" => {
                    if shaders.len() > 1 || (shaders.len() == 1 && !shaders.contains_key("")) {
                        panic!("Only one of `shaders`, `src`, `path`, or `bytes` can be defined");
                    }

                    parse_shader_fields(
                        shaders
                            .entry("".to_string())
                            .or_insert_with(Default::default),
                        name.as_str(),
                        input,
                    )?;
                }
                "shaders" => {
                    if !shaders.is_empty() {
                        panic!("Only one of `shaders`, `src`, `path`, or `bytes` can be defined");
                    }

                    let in_braces;
                    braced!(in_braces in input);

                    while !in_braces.is_empty() {
                        let prefix: Ident = in_braces.parse()?;
                        let prefix = prefix.to_string();

                        if prefix.to_string().as_str() == "shared_constants" {
                            in_braces.parse::<Token![:]>()?;

                            if shared_constants.is_some() {
                                panic!("Only one `shared_constants` can be defined")
                            }
                            let independent_constants_lit: LitBool = in_braces.parse()?;
                            shared_constants = Some(independent_constants_lit.value);

                            if !in_braces.is_empty() {
                                in_braces.parse::<Token![,]>()?;
                            }

                            continue;
                        }

                        if shaders.contains_key(&prefix) {
                            panic!("Shader entry {:?} already defined", prefix);
                        }

                        in_braces.parse::<Token![:]>()?;

                        let in_shader_definition;
                        braced!(in_shader_definition in in_braces);

                        while !in_shader_definition.is_empty() {
                            let name: Ident = in_shader_definition.parse()?;
                            in_shader_definition.parse::<Token![:]>()?;
                            let name = name.to_string();

                            match name.as_ref() {
                                "bytes" | "src" | "path" | "ty" => {
                                    parse_shader_fields(
                                        shaders
                                            .entry(prefix.clone())
                                            .or_insert_with(Default::default),
                                        name.as_str(),
                                        &in_shader_definition,
                                    )?;
                                }

                                name => panic!("Unknown Shader definition field {:?}", name),
                            }

                            if !in_shader_definition.is_empty() {
                                in_shader_definition.parse::<Token![,]>()?;
                            }
                        }

                        if !in_braces.is_empty() {
                            in_braces.parse::<Token![,]>()?;
                        }

                        match shaders.get(&prefix).unwrap() {
                            (None, _) => panic!("Please specify shader's {} type e.g. `ty: \"vertex\"`", prefix),
                            (_, None) => panic!("Please specify shader's {} source e.g. `path: \"entry_point.glsl\"`", prefix),
                            _ => ()
                        }
                    }

                    if shaders.is_empty() {
                        panic!("At least one Shader entry must be defined");
                    }
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
                "dump" => {
                    if dump.is_some() {
                        panic!("Only one `dump` can be defined")
                    }
                    let dump_lit: LitBool = input.parse()?;
                    dump = Some(dump_lit.value);
                }
                "exact_entrypoint_interface" => {
                    if exact_entrypoint_interface.is_some() {
                        panic!("Only one `dump` can be defined")
                    }
                    let lit: LitBool = input.parse()?;
                    exact_entrypoint_interface = Some(lit.value);
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
                "spirv_version" => {
                    let version: LitStr = input.parse()?;
                    spirv_version = Some(match version.value().as_ref() {
                        "1.0" => SpirvVersion::V1_0,
                        "1.1" => SpirvVersion::V1_1,
                        "1.2" => SpirvVersion::V1_2,
                        "1.3" => SpirvVersion::V1_3,
                        "1.4" => SpirvVersion::V1_4,
                        "1.5" => SpirvVersion::V1_5,
                        "1.6" => SpirvVersion::V1_6,
                        _ => panic!("Unknown SPIR-V version: {}", version.value()),
                    });
                }
                "types_meta" => {
                    let in_braces;
                    braced!(in_braces in input);

                    let mut meta = TypesMeta::empty();

                    while !in_braces.is_empty() {
                        if in_braces.peek(Token![#]) {
                            in_braces.parse::<Token![#]>()?;

                            let in_brackets;
                            bracketed!(in_brackets in in_braces);

                            let derive_list: MetaList = in_brackets.parse()?;

                            for derive in derive_list.nested {
                                match derive {
                                    NestedMeta::Meta(Meta::Path(path)) => {
                                        let custom_derive = if let Some(derive_ident) =
                                            path.get_ident()
                                        {
                                            match derive_ident.to_string().as_str() {
                                                "Clone" => {
                                                    if meta.default {
                                                        return Err(in_brackets
                                                            .error("Duplicate Clone derive"));
                                                    }

                                                    meta.clone = true;

                                                    false
                                                }
                                                "Copy" => {
                                                    if meta.copy {
                                                        return Err(in_brackets
                                                            .error("Duplicate Copy derive"));
                                                    }

                                                    meta.copy = true;

                                                    false
                                                }
                                                "PartialEq" => {
                                                    if meta.partial_eq {
                                                        return Err(in_brackets
                                                            .error("Duplicate PartialEq derive"));
                                                    }

                                                    meta.partial_eq = true;

                                                    false
                                                }
                                                "Debug" => {
                                                    if meta.debug {
                                                        return Err(in_brackets
                                                            .error("Duplicate Debug derive"));
                                                    }

                                                    meta.debug = true;

                                                    false
                                                }
                                                "Display" => {
                                                    if meta.display {
                                                        return Err(in_brackets
                                                            .error("Duplicate Display derive"));
                                                    }

                                                    meta.display = true;

                                                    false
                                                }
                                                "Default" => {
                                                    if meta.default {
                                                        return Err(in_brackets
                                                            .error("Duplicate Default derive"));
                                                    }

                                                    meta.default = true;

                                                    false
                                                }
                                                _ => true,
                                            }
                                        } else {
                                            true
                                        };

                                        if custom_derive {
                                            if meta
                                                .custom_derives
                                                .iter()
                                                .any(|candidate| candidate.eq(&path))
                                            {
                                                return Err(
                                                    in_braces.error("Duplicate derive declaration")
                                                );
                                            }

                                            meta.custom_derives.push(path);
                                        }
                                    }
                                    _ => return Err(in_brackets.error("Unsupported syntax")),
                                }
                            }

                            continue;
                        }

                        if in_braces.peek(Token![impl]) {
                            let impl_trait: TypeImplTrait = in_braces.parse()?;

                            if meta.impls.iter().any(|candidate| candidate == &impl_trait) {
                                return Err(in_braces.error("Duplicate \"impl\" declaration"));
                            }

                            meta.impls.push(impl_trait);

                            continue;
                        }

                        if in_braces.peek(Token![use]) {
                            let item_use: ItemUse = in_braces.parse()?;

                            if meta.uses.iter().any(|candidate| candidate == &item_use) {
                                return Err(in_braces.error("Duplicate \"use\" declaration"));
                            }

                            meta.uses.push(item_use);

                            continue;
                        }

                        return Err(in_braces.error("Type meta must by \"use a::b::c\", \"#[derive(Type1, Type2, ..)]\" or \"impl Type\""));
                    }

                    types_meta = Some(meta);
                }
                "vulkan_version" => {
                    let version: LitStr = input.parse()?;
                    vulkan_version = Some(match version.value().as_ref() {
                        "1.0" => EnvVersion::Vulkan1_0,
                        "1.1" => EnvVersion::Vulkan1_1,
                        "1.2" => EnvVersion::Vulkan1_2,
                        _ => panic!("Unknown Vulkan version: {}", version.value()),
                    });
                }
                name => panic!("Unknown field {:?}", name),
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        if shaders.is_empty() {
            panic!("Please specify at least one shader e.g. `ty: \"vertex\", src: \"glsl source code\"`");
        }

        match shaders.get("") {
            Some((None, _)) => panic!("Please specify shader's type e.g. `ty: \"vertex\"`"),
            Some((_, None)) => {
                panic!("Please specify shader's source e.g. `src: \"glsl source code\"`")
            }
            _ => (),
        }

        Ok(Self {
            dump: dump.unwrap_or(false),
            include_directories,
            macro_defines,
            shared_constants: shared_constants.unwrap_or(false),
            shaders: shaders
                .into_iter()
                .map(|(key, (shader_kind, shader_source))| {
                    (key, (shader_kind.unwrap(), shader_source.unwrap()))
                })
                .collect(),
            spirv_version,
            types_meta: types_meta.unwrap_or_else(|| TypesMeta::default()),
            vulkan_version,
        })
    }
}

pub(self) fn read_file_to_string(full_path: &Path) -> IoResult<String> {
    let mut buf = String::new();
    File::open(full_path).and_then(|mut file| file.read_to_string(&mut buf))?;
    Ok(buf)
}

#[proc_macro]
pub fn shader(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as MacroInput);

    let is_single = input.shaders.len() == 1;
    let root = env::var("CARGO_MANIFEST_DIR").unwrap_or(".".into());
    let root_path = Path::new(&root);

    let mut shaders_code = Vec::with_capacity(input.shaders.len());
    let mut types_code = Vec::with_capacity(input.shaders.len());
    let mut types_registry = HashMap::new();

    for (prefix, (shader_kind, shader_source)) in input.shaders {
        let (code, types) = if let SourceKind::Bytes(path) = shader_source {
            let full_path = root_path.join(&path);

            let bytes = if full_path.is_file() {
                fs::read(full_path).expect(&format!("Error reading source from {:?}", path))
            } else {
                panic!(
                    "File {:?} was not found; note that the path must be relative to your Cargo.toml",
                    path
                );
            };

            // The SPIR-V specification essentially guarantees that
            // a shader will always be an integer number of words
            assert_eq!(0, bytes.len() % 4);
            codegen::reflect(
                prefix.as_str(),
                unsafe { from_raw_parts(bytes.as_slice().as_ptr() as *const u32, bytes.len() / 4) },
                &input.types_meta,
                empty(),
                input.shared_constants,
                &mut types_registry,
            )
            .unwrap()
            .into()
        } else {
            let (path, full_path, source_code) = match shader_source {
                SourceKind::Src(source) => (None, None, source),
                SourceKind::Path(path) => {
                    let full_path = root_path.join(&path);
                    let source_code = read_file_to_string(&full_path)
                        .expect(&format!("Error reading source from {:?}", path));

                    if full_path.is_file() {
                        (Some(path.clone()), Some(full_path), source_code)
                    } else {
                        panic!("File {:?} was not found; note that the path must be relative to your Cargo.toml", path);
                    }
                }
                SourceKind::Bytes(_) => unreachable!(),
            };

            let include_paths = input
                .include_directories
                .iter()
                .map(|include_directory| {
                    let include_path = Path::new(include_directory);
                    let mut full_include_path = root_path.to_owned();
                    full_include_path.push(include_path);
                    full_include_path
                })
                .collect::<Vec<_>>();

            let (content, includes) = match codegen::compile(
                path,
                &root_path,
                &source_code,
                shader_kind,
                &include_paths,
                &input.macro_defines,
                input.vulkan_version,
                input.spirv_version,
            ) {
                Ok(ok) => ok,
                Err(e) => {
                    if is_single {
                        panic!("{}", e.replace("(s): ", "(s):\n"))
                    } else {
                        panic!("Shader {:?} {}", prefix, e.replace("(s): ", "(s):\n"))
                    }
                }
            };

            let input_paths = includes.iter().map(|s| s.as_ref()).chain(
                full_path
                    .as_ref()
                    .map(|p| p.as_path())
                    .map(codegen::path_to_str),
            );

            codegen::reflect(
                prefix.as_str(),
                content.as_binary(),
                &input.types_meta,
                input_paths,
                input.shared_constants,
                &mut types_registry,
            )
            .unwrap()
            .into()
        };

        shaders_code.push(code);
        types_code.push(types);
    }

    let uses = &input.types_meta.uses;

    let result = quote! {
        #(
            #shaders_code
        )*

        pub mod ty {
            #( #uses )*

            #(
                #types_code
            )*
        }
    };

    if input.dump {
        println!("{}", result.to_string());
        panic!("`shader!` rust codegen dumped") // TODO: use span from dump
    }

    proc_macro::TokenStream::from(result)
}
