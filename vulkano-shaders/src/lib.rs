//! The procedural macro for vulkano's shader system.
//! Manages the compile-time compilation of shader code into SPIR-V and generation of associated
//! Rust code.
//!
//! # Requirements
//!
//! This crate invokes [`glslc`] at compile time to compile shaders. `glslc` must be installed and
//! available on your `PATH`. It is included in the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
//! or can be installed separately via your package manager.
//!
//! [`glslc`]: https://github.com/google/shaderc/tree/main/glslc
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
//! #
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
//! - The `load` function. This function takes an `&Arc<Device>`, constructs a [`ShaderModule`]
//!   with the passed-in device and the shader data provided via the macro, and returns
//!   `Result<Arc<ShaderModule>, VulkanError>`. It also generates the `try_load` function, which
//!   works the same, but returns a `Result<Arc<ShaderModule>, Validated<VulkanError>>`. These
//!   functions are both unsafe because they delegate to the unsafe [`ShaderModule::try_new`].
//!   Additionally, a `load_unchecked` function is generated, which skips validation entirely and
//!   delegates to [`ShaderModule::new_unchecked`].
//! - If the `shaders` option is used, then instead of one `load` constructor, there is one for
//!   each shader. They are named based on the provided names, `load_first`, `load_second` etc.
//!   `try_` and `_unchecked` variants are also generated for each shader.
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
//! #
//! # fn main() {}
//! #
//! // ...various use statements...
//! // ...`vs` module containing a `shader!` call...
//!
//! pub struct Shaders {
//!     pub vs: Arc<ShaderModule>,
//! }
//!
//! impl Shaders {
//!     pub unsafe fn load(device: &Arc<Device>) -> Result<Self, VulkanError> {
//!         Ok(Self {
//!             vs: unsafe { vs::load(device) }?,
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
//! This defines what shader type the given shader source will be compiled into. The type can be
//! any of the following:
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
//! Provides the raw shader source to be compiled in the form of a string. Cannot be used in
//! conjunction with the `path` or `bytes` field.
//!
//! ## `path: "..."`
//!
//! Provides the path to the shader source to be compiled, relative to the file invoking the macro.
//! Cannot be used in conjunction with the `src` or `bytes` field.
//!
//! ## `bytes: "..."`
//!
//! Provides the path to precompiled SPIR-V bytecode, relative to the file invoking the macro.
//! Cannot be used in conjunction with the `src` or `path` field, and may also not specify a shader
//! `ty` type. This allows using shaders compiled through a separate build system.
//!
//! ## `root_path_env: "..."`
//!
//! Instead of searching relative to the file invoking the macro, search relative to some other
//! folder specified by this env variable. The intended use case is using `OUT_DIR` to be able to
//! load shaders generated by your build script. You can also set this to `CARGO_MANIFEST_DIR` if
//! you want to search relative to the folder of your `Cargo.toml`.
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
//! An optional `define: [("NAME", "VALUE"), ...]` list sets preprocessor definitions
//! for just this source file.
//!
//! ## `include: ["...", "...", ...]`
//!
//! Specifies the standard include directories to be searched through when using the
//! `#include <...>` directive within a shader source. Include directories can be absolute or
//! relative to your `Cargo.toml`. Relative paths are relative to the directory which contains the
//! source file the `#include "..."` directive is declared in. This includes the Rust files with
//! embedded shader source.
//!
//! ## `define: [("NAME", "VALUE"), ...]`
//!
//! Adds the given macro definitions to the pre-processor. This is equivalent to passing the
//! `-DNAME=VALUE` argument on the command line.
//!
//! ## `lang: "..."`
//!
//! Provides the language of the shader source. Must be either `glsl` or `hlsl` (defaults to
//! `glsl`).
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
//! | Feature         | Description                                      |
//! |-----------------|--------------------------------------------------|
//! | `shaderc-debug` | Compile shaders with debug information included. |
//!
//! [`cargo-env-vars`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html
//! [cargo-expand]: https://github.com/dtolnay/cargo-expand
//! [`ShaderModule`]: vulkano::shader::ShaderModule
//! [`ShaderModule::try_new`]: vulkano::shader::ShaderModule::try_new
//! [pipeline]: vulkano::pipeline
//! [`BufferContents`]: vulkano::buffer::BufferContents

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![recursion_limit = "1024"]

use foldhash::HashMap;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::{
    env, fs,
    mem::{self, ManuallyDrop},
    path::{Path, PathBuf},
    result::Result as StdResult,
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
pub fn include_vulkano_glsl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    parse_macro_input!(input as syn::parse::Nothing);

    let include_string = include_str!("../include/vulkano.glsl");

    quote! { #include_string }.into()
}

#[proc_macro]
pub fn shader(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as MacroInput);

    shader_inner(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn shader_inner(mut input: MacroInput) -> Result<TokenStream> {
    let shaders = mem::take(&mut input.shaders); // yoink

    let mut state = MacroState::new(&input)?;

    match shaders.unwrap() {
        Shaders::Single(shader_fields) => {
            state.process_shader(None, shader_fields)?;
        }
        Shaders::Multiple(shaders) => {
            for (shader_name, shader_fields) in shaders {
                state.process_shader(Some(shader_name), shader_fields)?;
            }
        }
    }

    let result = state.finalize();

    if input.dump.value {
        println!("{}", result);
        bail!(input.dump, "`shader!` Rust codegen dumped");
    }

    Ok(result)
}

struct MacroState<'a> {
    input: &'a MacroInput,
    root_path: PathBuf,
    relative_path_error_message: String,
    shaders_code: TokenStream,
    structs_code: TokenStream,
    type_registry: TypeRegistry,
}

impl<'a> MacroState<'a> {
    fn new(input: &'a MacroInput) -> Result<Self> {
        let (root_path, relative_path_error_message) = match input.root_path_env.as_ref() {
            None => root_path_from_call_site(),
            Some(root_path_env) => root_path_from_env_var(root_path_env),
        }?;

        Ok(MacroState {
            input,
            root_path,
            relative_path_error_message,
            shaders_code: TokenStream::new(),
            structs_code: TokenStream::new(),
            type_registry: TypeRegistry::default(),
        })
    }

    fn process_shader(
        &mut self,
        shader_name: Option<String>,
        shader_fields: ShaderFields,
    ) -> Result<()> {
        let ShaderFields {
            shader_kind,
            source_kind,
            macro_defines,
        } = shader_fields;

        let (lit, words, input_paths) = match source_kind.unwrap() {
            source_kind @ (SourceKind::Src(_) | SourceKind::Path(_)) => {
                let source_path;
                let source_code;
                let working_dir;

                let lit = match source_kind {
                    SourceKind::Path(lit) => {
                        source_path = Some(self.root_path.join(lit.value()));
                        let path = source_path.as_deref().unwrap();

                        self.check_file_exists(&lit, path)?;

                        source_code = read_file_to_string(&lit, path)?;
                        working_dir = path.parent().unwrap();

                        lit
                    }
                    SourceKind::Src(lit) => {
                        source_path = None;
                        source_code = lit.value();
                        working_dir = &self.root_path;

                        lit
                    }
                    SourceKind::Bytes(_) => unreachable!(),
                };

                let (words, mut input_paths) = codegen::compile(
                    self.input,
                    &source_code,
                    working_dir,
                    shader_kind.unwrap(),
                    &macro_defines,
                )
                .map_err(|err| Error::new_spanned(&lit, err))?;

                if let Some(source_path) = source_path {
                    input_paths.push(source_path.into_os_string().into_string().unwrap());
                }

                (lit, words, input_paths)
            }
            SourceKind::Bytes(lit) => {
                let path = self.root_path.join(lit.value());

                self.check_file_exists(&lit, &path)?;

                let bytes = read_file(&lit, &path)?;

                let words = spirv_bytes_to_words(bytes).map_err(|err| {
                    Error::new_spanned(&lit, format!("failed to read source `{path:?}`: {err}"))
                })?;

                let input_paths = vec![path.into_os_string().into_string().unwrap()];

                (lit, words, input_paths)
            }
        };

        let (shaders_code, structs_code) = codegen::reflect(
            self.input,
            lit,
            shader_name,
            &words,
            input_paths,
            &mut self.type_registry,
        )?;

        self.shaders_code.extend(shaders_code);
        self.structs_code.extend(structs_code);

        Ok(())
    }

    fn check_file_exists(&self, lit: &LitStr, path: &Path) -> Result<()> {
        if !path.is_file() {
            let msg = &self.relative_path_error_message;

            bail!(
                lit,
                "file `{path:?}` was not found; note that the path must be relative {msg}",
            );
        }

        Ok(())
    }

    fn finalize(self) -> TokenStream {
        let shaders_code = self.shaders_code;
        let structs_code = self.structs_code;

        quote! {
            #shaders_code
            #structs_code
        }
    }
}

fn root_path_from_call_site() -> Result<(PathBuf, String)> {
    let Some(file_path) = Span::call_site().unwrap().local_file() else {
        bail!(
            "failed to get the file invoking the macro from the compiler; this only happens when \
            using rust-analyzer with the file with this macro invocation open, and you can ignore \
            this error; if you can't live with it, you can use the `root_path_env: \
            \"CARGO_MANIFEST_DIR\"` option to search relative to your Cargo.toml instead",
        );
    };

    // This comes from cargo, so it better be a real directory path we have access to.
    //
    // TODO: Use a variable set by cargo for the working directory of rustc invocations once it's
    // stable such that this is truly bullet-proof.
    let working_dir = env::current_dir().expect("failed to get the current working directory");

    // The file path should be relative to the current working directory that rustc was invoked
    // with. The proc macros rustc invokes should have the same current working directory as that
    // rustc. So this should always give us the right full path.
    let full_path = working_dir
        .join(&file_path)
        .components()
        .collect::<PathBuf>();

    if !full_path.exists() {
        bail!(
            "the file invoking the macro, `{full_path:?}`, doesn't exist; you can use the \
            `root_path_env: \"CARGO_MANIFEST_DIR\"` option to search relative to your Cargo.toml \
            instead",
        );
    }

    // The file path comes from rustc, so it better be a real file path.
    let root = full_path.parent().expect(
        "the file invoking the macro we got from the compiler doesn't reside in a directory",
    );

    let error = format!("to the file `{file_path:?}` invoking the macro");

    Ok((root.into(), error))
}

fn root_path_from_env_var(root_path_env: &LitStr) -> Result<(PathBuf, String)> {
    let env = root_path_env.value();
    let root = env::var(&env).map_err(|e| {
        Error::new_spanned(
            root_path_env,
            format_args!(
                "failed to fetch the environment variable `{env}`: {e}; typical parameters are \
                `OUT_DIR` to gather results from your build script, `CARGO_MANIFEST_DIR` to \
                search relative to your Cargo.toml, or left default to search relative to the \
                file invoking the macro",
            ),
        )
    })?;
    let error = format!("to the path `{root}` specified by the env variable `{env:?}`");

    Ok((root.into(), error))
}

fn read_file_to_string(lit: &LitStr, path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|err| {
        Error::new_spanned(lit, format_args!("failed to read file `{path:?}`: {err}"))
    })
}

fn read_file(lit: &LitStr, path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|err| {
        Error::new_spanned(lit, format_args!("failed to read file `{path:?}`: {err}"))
    })
}

fn spirv_bytes_to_words(bytes: Vec<u8>) -> StdResult<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err("the length of the bytes is not a multiple of 4".into());
    }

    #[cfg(target_endian = "little")]
    if bytes.as_ptr().addr().is_multiple_of(align_of::<u32>()) {
        let mut bytes = ManuallyDrop::new(bytes);
        let len = bytes.len() / size_of::<u32>();
        let cap = bytes.capacity();
        let ptr = bytes.as_mut_ptr().cast::<u32>();

        // SAFETY: We checked that the pointer is 4-byte aligned and that the size divides evenly.
        return Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) });
    }

    // SAFETY: We checked that the size divides evenly.
    let words = unsafe { bytes.as_chunks_unchecked::<{ size_of::<u32>() }>() };

    Ok(words.iter().copied().map(u32::from_le_bytes).collect())
}

enum SourceKind {
    Src(LitStr),
    Path(LitStr),
    Bytes(LitStr),
}

#[derive(Copy, Clone)]
enum SourceLanguage {
    Glsl,
    Hlsl,
    Slang,
}

impl From<SourceLanguage> for &str {
    fn from(lang: SourceLanguage) -> Self {
        match lang {
            SourceLanguage::Glsl => "glsl",
            SourceLanguage::Hlsl => "hlsl",
            SourceLanguage::Slang => "slang",
        }
    }
}

impl std::fmt::Display for SourceLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(<&str>::from(*self))
    }
}

#[derive(Debug, Copy, Clone)]
enum EnvVersion {
    Vulkan1_0,
    Vulkan1_1,
    Vulkan1_2,
    Vulkan1_3,
}

impl From<EnvVersion> for &str {
    fn from(version: EnvVersion) -> Self {
        match version {
            EnvVersion::Vulkan1_0 => "vulkan1.0",
            EnvVersion::Vulkan1_1 => "vulkan1.1",
            EnvVersion::Vulkan1_2 => "vulkan1.2",
            EnvVersion::Vulkan1_3 => "vulkan1.3",
        }
    }
}

impl std::fmt::Display for EnvVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(<&str>::from(*self))
    }
}

#[derive(Copy, Clone, Debug)]
enum SpirvVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
    V1_4,
    V1_5,
    V1_6,
}

impl From<SpirvVersion> for &str {
    fn from(version: SpirvVersion) -> Self {
        match version {
            SpirvVersion::V1_0 => "spv1.0",
            SpirvVersion::V1_1 => "spv1.1",
            SpirvVersion::V1_2 => "spv1.2",
            SpirvVersion::V1_3 => "spv1.3",
            SpirvVersion::V1_4 => "spv1.4",
            SpirvVersion::V1_5 => "spv1.5",
            SpirvVersion::V1_6 => "spv1.6",
        }
    }
}

impl SpirvVersion {
    fn as_slangc_profile(self) -> &'static str {
        match self {
            SpirvVersion::V1_0 => "spirv_1_0",
            SpirvVersion::V1_1 => "spirv_1_1",
            SpirvVersion::V1_2 => "spirv_1_2",
            SpirvVersion::V1_3 => "spirv_1_3",
            SpirvVersion::V1_4 => "spirv_1_4",
            SpirvVersion::V1_5 => "spirv_1_5",
            SpirvVersion::V1_6 => "spirv_1_6",
        }
    }
}

impl std::fmt::Display for SpirvVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(<&str>::from(*self))
    }
}

#[derive(Copy, Clone, Debug)]
enum ShaderKind {
    Vertex,
    TessControl,
    TessEvaluation,
    Geometry,
    Task,
    Mesh,
    Fragment,
    Compute,
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,
}

impl ShaderKind {
    fn as_glslc_stage(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vert",
            ShaderKind::TessControl => "tesc",
            ShaderKind::TessEvaluation => "tese",
            ShaderKind::Geometry => "geom",
            ShaderKind::Task => "task",
            ShaderKind::Mesh => "mesh",
            ShaderKind::Fragment => "frag",
            ShaderKind::Compute => "comp",
            ShaderKind::RayGeneration => "rgen",
            ShaderKind::AnyHit => "rahit",
            ShaderKind::ClosestHit => "rchit",
            ShaderKind::Miss => "rmiss",
            ShaderKind::Intersection => "rint",
            ShaderKind::Callable => "rcall",
        }
    }

    fn as_slangc_stage(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vertex",
            ShaderKind::TessControl => "hull",
            ShaderKind::TessEvaluation => "domain",
            ShaderKind::Geometry => "geometry",
            ShaderKind::Task => "amplification",
            ShaderKind::Mesh => "mesh",
            ShaderKind::Fragment => "fragment",
            ShaderKind::Compute => "compute",
            ShaderKind::RayGeneration => "raygeneration",
            ShaderKind::AnyHit => "anyhit",
            ShaderKind::ClosestHit => "closesthit",
            ShaderKind::Miss => "miss",
            ShaderKind::Intersection => "intersection",
            ShaderKind::Callable => "callable",
        }
    }
}

impl TryFrom<&str> for ShaderKind {
    type Error = String;

    fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
        match s {
            "vertex" => Ok(ShaderKind::Vertex),
            "tess_ctrl" => Ok(ShaderKind::TessControl),
            "tess_eval" => Ok(ShaderKind::TessEvaluation),
            "geometry" => Ok(ShaderKind::Geometry),
            "task" => Ok(ShaderKind::Task),
            "mesh" => Ok(ShaderKind::Mesh),
            "fragment" => Ok(ShaderKind::Fragment),
            "compute" => Ok(ShaderKind::Compute),
            "raygen" => Ok(ShaderKind::RayGeneration),
            "anyhit" => Ok(ShaderKind::AnyHit),
            "closesthit" => Ok(ShaderKind::ClosestHit),
            "miss" => Ok(ShaderKind::Miss),
            "intersection" => Ok(ShaderKind::Intersection),
            "callable" => Ok(ShaderKind::Callable),
            _ => Err(format!(
                "unknown shader kind: {s}; expected `vertex`, `tess_ctrl`, `tess_eval`, \
                `geometry`, `task`, `mesh`, `fragment`, `compute`, `raygen`, `anyhit`, \
                `closesthit`, `miss`, `intersection` or `callable`",
            )),
        }
    }
}

struct MacroInput {
    root_path_env: Option<LitStr>,
    include_directories: Vec<PathBuf>,
    global_macro_defines: Vec<(String, String)>,
    shaders: Option<Shaders>,
    source_language: Option<SourceLanguage>,
    spirv_version: Option<SpirvVersion>,
    vulkan_version: Option<EnvVersion>,
    generate_structs: bool,
    custom_derives: Vec<SynPath>,
    linalg_type: LinAlgType,
    dump: LitBool,
}

enum Shaders {
    Single(ShaderFields),
    Multiple(HashMap<String, ShaderFields>),
}

#[derive(Default)]
struct ShaderFields {
    shader_kind: Option<ShaderKind>,
    source_kind: Option<SourceKind>,
    macro_defines: Vec<(String, String)>,
}

impl MacroInput {
    #[cfg(test)]
    fn empty() -> Self {
        MacroInput {
            root_path_env: None,
            include_directories: Vec::new(),
            global_macro_defines: Vec::new(),
            shaders: None,
            vulkan_version: None,
            spirv_version: None,
            generate_structs: true,
            custom_derives: Vec::new(),
            linalg_type: LinAlgType::default(),
            dump: LitBool::new(false, Span::call_site()),
            source_language: None,
        }
    }
}

impl Parse for MacroInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());

        let mut root_path_env = None;
        let mut include_directories = Vec::new();
        let mut global_macro_defines = Vec::new();
        let mut shaders = None;
        let mut vulkan_version = None;
        let mut spirv_version = None;
        let mut generate_structs = None;
        let mut custom_derives = None;
        let mut linalg_type = None;
        let mut dump = None;
        let mut source_language = None;

        fn parse_shader_fields(
            output: &mut ShaderFields,
            name: &str,
            input: ParseStream<'_>,
        ) -> Result<()> {
            match name {
                "ty" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.shader_kind.is_some() {
                        bail!(lit, "field `ty` is already defined");
                    }

                    output.shader_kind = Some(match ShaderKind::try_from(lit.value().as_str()) {
                        Ok(shader_kind) => shader_kind,
                        Err(msg) => bail!(lit, "{msg}"),
                    });
                }
                "bytes" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.source_kind.is_some() {
                        bail!(
                            lit,
                            "only one of `src`, `path`, or `bytes` can be defined per shader entry",
                        );
                    }

                    output.source_kind = Some(SourceKind::Bytes(lit));
                }
                "path" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.source_kind.is_some() {
                        bail!(
                            lit,
                            "only one of `src`, `path` or `bytes` can be defined per shader entry",
                        );
                    }

                    output.source_kind = Some(SourceKind::Path(lit));
                }
                "src" => {
                    let lit = input.parse::<LitStr>()?;
                    if output.source_kind.is_some() {
                        bail!(
                            lit,
                            "only one of `src`, `path` or `bytes` can be defined per shader entry",
                        );
                    }

                    output.source_kind = Some(SourceKind::Src(lit));
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
                        output.macro_defines.push((name.value(), value.value()));

                        if !array_input.is_empty() {
                            array_input.parse::<Token![,]>()?;
                        }
                    }
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
                    if matches!(&shaders, Some(Shaders::Multiple(_))) {
                        bail!(
                            field_ident,
                            "only one of `src`, `path`, `bytes` or `shaders` can be defined",
                        );
                    }

                    if shaders.is_none() {
                        shaders = Some(Shaders::Single(ShaderFields::default()));
                    }

                    let Some(Shaders::Single(output)) = &mut shaders else {
                        unreachable!();
                    };

                    parse_shader_fields(output, &field, input)?;
                }
                "shaders" => {
                    if !shaders.is_none() {
                        bail!(
                            field_ident,
                            "only one of `src`, `path`, `bytes` or `shaders` can be defined",
                        );
                    }

                    shaders = Some(Shaders::Multiple(HashMap::default()));

                    let Some(Shaders::Multiple(shaders)) = &mut shaders else {
                        unreachable!();
                    };

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
                                "bytes" | "src" | "path" | "ty" | "define" => {
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
                            ShaderFields {
                                shader_kind: None, ..
                            } => bail!(
                                "please specify a type for shader `{name}` e.g. `ty: \"vertex\"`",
                            ),
                            ShaderFields {
                                source_kind: None, ..
                            } => bail!(
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
                        global_macro_defines.push((name.value(), value.value()));

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

                        include_directories
                            .push([&manifest_dir, &path.value()].into_iter().collect());

                        if !in_brackets.is_empty() {
                            in_brackets.parse::<Token![,]>()?;
                        }
                    }
                }
                "lang" => {
                    let lit = input.parse::<LitStr>()?;
                    if source_language.is_some() {
                        bail!(lit, "field `lang` is already defined");
                    }

                    source_language = Some(match lit.value().as_str() {
                        "glsl" => SourceLanguage::Glsl,
                        "hlsl" => SourceLanguage::Hlsl,
                        "slang" => SourceLanguage::Slang,
                        lang => bail!(lit, "expected `glsl`, `hlsl`, or `slang`, found `{lang}`"),
                    })
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
                    if generate_structs.is_some() {
                        bail!(lit, "field `generate_structs` is already defined");
                    }
                    generate_structs = Some(lit.value);
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

        if shaders.is_none()
            || matches!(&shaders, Some(Shaders::Multiple(shaders)) if shaders.is_empty())
        {
            bail!(r#"please specify at least one shader e.g. `ty: "vertex", src: "<GLSL code>"`"#);
        }

        if let Some(Shaders::Single(fields)) = &shaders {
            match fields {
                ShaderFields {
                    shader_kind: None,
                    source_kind: Some(SourceKind::Bytes(_)),
                    ..
                } => {}
                ShaderFields {
                    shader_kind: Some(_),
                    source_kind: Some(SourceKind::Bytes(_)),
                    ..
                } => {
                    bail!(
                        r#"one may not specify a shader type when including precompiled SPIR-V binaries. Please remove the `ty:` declaration"#
                    );
                }
                ShaderFields {
                    shader_kind: None, ..
                } => {
                    bail!(r#"please specify the type of the shader e.g. `ty: "vertex"`"#);
                }
                ShaderFields {
                    source_kind: None, ..
                } => {
                    bail!(r#"please specify the source of the shader e.g. `src: "<GLSLcode>"`"#);
                }
                _ => {}
            }
        }

        Ok(MacroInput {
            root_path_env,
            include_directories,
            global_macro_defines,
            shaders,
            vulkan_version,
            spirv_version,
            generate_structs: generate_structs.unwrap_or(true),
            custom_derives: custom_derives.unwrap_or_else(|| {
                vec![
                    parse_quote! { ::std::clone::Clone },
                    parse_quote! { ::std::marker::Copy },
                ]
            }),
            linalg_type: linalg_type.unwrap_or_default(),
            dump: dump.unwrap_or_else(|| LitBool::new(false, Span::call_site())),
            source_language,
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
            format_args!($msg),
        ))
    };
    ($span:expr, $msg:literal $(,)?) => {
        return Err(syn::Error::new_spanned(&$span, format_args!($msg)))
    };
}
use bail;
