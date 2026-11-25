//! The procedural macro for vulkano's shader system. Manages the compile-time compilation of
//! shader code into SPIR-V and generation of associated Rust code.
//!
//! # Requirements
//!
//! This crate invokes [`glslc`] and/or [`slangc`] at compile time to compile shaders. `glslc`
//! and/or `slangc` must be installed and available on your `PATH`. Both are included in the
//! [Vulkan SDK] or can be installed separately via your package manager.
//!
//! # Basic usage
//!
//! ```
//! mod vs {
//!     # mod vulkano_shaders { macro_rules! shader { ($($tt:tt)*) => {}; } pub(super) use shader; }
//!     #
//!     vulkano_shaders::shader! {
//!         ty: "vertex",
//!         path: "path/to/vertex_shader.glsl",
//!     }
//! }
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
//!   structure has a `Clone` and a `Copy` implementation. This behavior can be customized through
//!   the `custom_derives` macro option (see below for details). Each struct also has an
//!   implementation of [`BufferContents`], so that it can be read from/written to a buffer.
//!
//! All of these generated items will be accessed through the module where the macro was invoked.
//! If you wanted to store the `ShaderModule` in a struct of your own, you could do something like
//! this:
//!
//! ```
//! # mod vs {
//! #     pub use std::sync::Arc;
//! #
//! #     pub struct Device;
//! #     pub struct ShaderModule;
//! #     pub struct VulkanError;
//! #
//! #     pub unsafe fn load(device: &Arc<Device>) -> Result<Arc<ShaderModule>, VulkanError> {
//! #         unimplemented!()
//! #     }
//! # }
//! #
//! # use vs::{Arc, Device, ShaderModule, VulkanError};
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
//! ## `ty`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     ty: "...",
//!     ...
//! }
//! ```
//!
//! This defines what shader type the given shader source will be compiled into. It can be any of
//! the following:
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
//! If unspecified, the behavior is the same as the behavior of the compiler of choice when the
//! shader stage is unspecified. In particular, this means that shaderc will expect a `#pragma
//! shader_stage(<stage>)` directive in the code, and slangc will expect the entry point(s) to be
//! annotated with `[shader("<stage>")]`.
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `path`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     path: "...",
//!     ...
//! }
//! ```
//!
//! Provides the path to the shader source to be compiled, relative to the file invoking the macro.
//!
//! Cannot be defined alongside the [`src`] or [`bytes`] option.
//!
//! ## `src`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     src: "...",
//!     ...
//! }
//! ```
//!
//! Provides the raw shader source to be compiled in the form of a string.
//!
//! Cannot be defined alongside the [`path`] or [`bytes`] option.
//!
//! ## `bytes`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     bytes: "...",
//!     ...
//! }
//! ```
//!
//! Provides the path to precompiled SPIR-V bytecode, relative to the file invoking the macro.
//!
//! Cannot be defined alongside the [`src`] or [`path`] option, and may also not specify the
//! [`ty`], [`lang`], [`entry_point`] or [`compiler`] option. This allows using shaders compiled
//! through a separate build system.
//!
//! ## `lang`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     lang: "...",
//!     ...
//! }
//! ```
//!
//! Provides the language of the shader source. It can be any of the following:
//!
//! - `glsl` (default)
//! - `hlsl`
//! - `slang`
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `entry_point`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     entry_point: "...",
//!     ...
//! }
//! ```
//!
//! Provides the entry point of the shader source to compile.
//!
//! If unspecified, the behavior is the same as the behavior of the compiler of choice when the
//! entry point is unspecified. In particular, this means that the entry point defaults to `main`,
//! except when using slangc and the [`ty`] option is also left out, in which case slangc searches
//! for entry points annotated with `[shader("<stage>")]`.
//!
//! This option cannot be used when compiling with shaderc as it has no support for compiling any
//! entry point other than `main`.
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `compiler`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     compiler: "...",
//!     ...
//! }
//! ```
//!
//! Provides the compiler to invoke. It can be any of the following:
//!
//! - `shaderc` (default for GLSL)
//! - `slangc` (default for HLSL and Slang)
//!
//! Here are the source languages supported for each compiler:
//!
//! - `shaderc`: GLSL
//! - `slangc`: GLSL, HLSL and Slang
//!
//! When using `shaderc`, you must have [`glslc`] available on your `PATH`. When using `slangc`,
//! you must have [`slangc`] available on your `PATH`. The [Vulkan SDK] comes with both.
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `shaders`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     shaders: {
//!         first: {
//!             ...
//!         },
//!         ...
//!     },
//!     ...
//! }
//! ```
//!
//! With this option, the user can compile several shaders in a single macro invocation. Each entry
//! key will be the suffix of the generated `load` function (`load_first` in the above example).
//! However, all other Rust structs translated from the shader source will be shared between
//! shaders. The macro checks that the source structs with the same names between different shaders
//! have the same declaration signature, and throws a compile-time error if they don't.
//!
//! Each entry accepts [`ty`], [`path`], [`src`], [`bytes`] and [`entry_point`] options same as
//! above.
//!
//! If an entry defines the [`lang`] option, it overrides a potential global `lang` option:
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     shaders: {
//!         first: {
//!             ty: "...",
//!             path: "...",
//!         },
//!         second: {
//!             ty: "...",
//!             path: "...",
//!             lang: "slang",
//!         },
//!     },
//!     lang: "hlsl",
//! }
//! ```
//!
//! Above, the shaders named `first` and `second` will be compiled as HLSL and Slang, respectively.
//! Without the global `lang` option present, they will be compiled as GLSL and Slang,
//! respectively.
//!
//! If an entry defines the [`compiler`] option, it overrides a potential global `compiler` option
//! in the same way as the `lang` option does.
//!
//! If an entry defines the [`define`] option, it extends a potential global `define` option,
//! overriding any duplicates:
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     shaders: {
//!         first: {
//!             ty: "...",
//!             path: "...",
//!         },
//!         second: {
//!             ty: "...",
//!             path: "...",
//!             define: [("X", "69"), ("Z", "3")],
//!         },
//!     },
//!     define: [("X", "1"), ("Y", "2")],
//! }
//! ```
//!
//! Above, the shaders named `first` and `second` will be compiled with `-DX=1 -DY=2` and
//! `-DX=69 -DY=2 -DZ=3`, respectively. Without the global `define` option present, they will be
//! compiled with no defines and `-DX=69 -DZ=3`, respectively.
//!
//! ## `root_path_env`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     root_path_env: "...",
//!     ...
//! }
//! ```
//!
//! Instead of searching relative to the file invoking the macro, search relative to some other
//! folder specified by this env variable. The intended use case is using `OUT_DIR` to be able to
//! load shaders generated by your build script. You can also set this to `CARGO_MANIFEST_DIR` if
//! you want to search relative to the folder of your `Cargo.toml`.
//!
//! See [`cargo-env-vars`] for a full set of env variables set by cargo. It is also possible to
//! specify env variables from within the build script using the following:
//!
//! ```
//! # let shader_out_dir = "";
//! println!("cargo:rustc-env=SHADER_OUT_DIR={shader_out_dir}");
//! ```
//!
//! ## `include`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     include: ["...", "...", ...],
//!     ...
//! }
//! ```
//!
//! Specifies the standard include directories to be searched through when using the
//! `#include <...>` directive within a shader source. Include directories can be absolute or
//! relative to your `Cargo.toml`. Relative paths are relative to the directory which contains the
//! source file the `#include "..."` directive is declared in. This includes the Rust files with
//! embedded shader source.
//!
//! ## `define`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     define: [("NAME", "VALUE"), ...],
//!     ...
//! }
//! ```
//!
//! Adds the given macro definitions for the pre-processor. This is equivalent to passing the
//! `-DNAME=VALUE` argument on the command line.
//!
//! ## `vulkan_version`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     vulkan_version: "...",
//!     ...
//! }
//! ```
//!
//! Sets the Vulkan version to compile for. It can be any of the following:
//!
//! - `1.0` (default)
//! - `1.1`
//! - `1.2`
//! - `1.3`
//!
//! The generated code must be supported by the device at runtime. If not, then an error will be
//! returned when calling `load`.
//!
//! ## `spirv_version`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     spirv_version: "...",
//!     ...
//! }
//! ```
//!
//! Sets the SPIR-V version to compile for. It can be any of the following:
//!
//! - `1.0`
//! - `1.1`
//! - `1.2`
//! - `1.3`
//! - `1.4`
//! - `1.5`
//! - `1.6`
//!
//! By default, the minimum required SPIR-V version for the [`vulkan_version`] is used.
//!
//! The generated code must be supported by the device at runtime. If not, then an error will be
//! returned when calling `load`.
//!
//! ## `generate_structs`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     generate_structs: false,
//!     ...
//! }
//! ```
//!
//! Generate Rust structs that represent the structs contained in the shader. They all implement
//! [`BufferContents`], which allows them to be passed to the shader without having to worry about
//! the layout of the struct manually. However, some use cases, such as Rust-GPU, may not have any
//! use for such structs and may choose to disable them.
//!
//! By default, structs are generated.
//!
//! ## `custom_derives`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     custom_derives: [Clone, Default, PartialEq, ...],
//!     ...
//! }
//! ```
//!
//! Extends the list of derive macros that are added to the `derive` attribute of Rust structs that
//! represent shader structs.
//!
//! By default, each generated struct derives `Clone` and `Copy`. If the struct has unsized
//! members, none of the derives are applied on the struct, except [`BufferContents`], which is
//! always derived.
//!
//! ## `linalg_type`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     linalg_type: "...",
//!     ...
//! }
//! ```
//!
//! Specifies the way that linear algebra types should be generated. It can be any of the
//! following:
//!
//! - `std` (default)
//! - `cgmath`
//! - `nalgebra`
//!
//! `std` uses arrays to represent vectors and matrices. Note that if the chosen crate doesn't have
//! a type that represents a certain linear algebra type (e.g., `mat3` or a rectangular matrix),
//! then the macro will default back to arrays for that type.
//!
//! If you use linear algebra types from a third-party crate, then you have to have the crate in
//! your dependencies with the appropriate feature enabled that adds `bytemuck` support.
//!
//! ## `dump`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     dump: true,
//!     ...
//! }
//! ```
//!
//! The crate fails to compile, but prints the generated Rust code to stdout.
//!
//! # Cargo features
//!
//! | Feature         | Description                                      |
//! |-----------------|--------------------------------------------------|
//! | `shaderc-debug` | Compile shaders with debug information included. |
//!
//! [`glslc`]: https://github.com/google/shaderc/tree/main/glslc
//! [`slangc`]: https://github.com/shader-slang/slang
//! [Vulkan SDK]: https://www.lunarg.com/vulkan-sdk/
//! [cargo-expand]: https://github.com/dtolnay/cargo-expand
//! [`ShaderModule`]: https://docs.rs/vulkano/latest/vulkano/shader/struct.ShaderModule.html
//! [`ShaderModule::try_new`]: https://docs.rs/vulkano/latest/vulkano/shader/struct.ShaderModule.html#method.try_new
//! [`ShaderModule::new_unchecked`]: https://docs.rs/vulkano/latest/vulkano/shader/struct.ShaderModule.html#method.new_unchecked
//! [`BufferContents`]: https://docs.rs/vulkano/latest/vulkano/buffer/trait.BufferContents.html
//! [pipeline]: https://docs.rs/vulkano/latest/vulkano/pipeline/index.html
//! [`cargo-env-vars`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html
//! [`ty`]: self#ty
//! [`path`]: self#path
//! [`src`]: self#src
//! [`bytes`]: self#bytes
//! [`lang`]: self#lang
//! [`entry_point`]: self#entry_point
//! [`compiler`]: self#compiler
//! [`define`]: self#define
//! [`vulkan_version`]: self#vulkan_version

#![deprecated(
    since = "0.36.0",
    note = "vulkano-shaders has been merged into vulkano-macros"
)]
#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![recursion_limit = "1024"]

use proc_macro::TokenStream;
use vulkano_codegen::CodegenContext;

#[proc_macro]
pub fn include_vulkano_glsl(input: TokenStream) -> TokenStream {
    vulkano_codegen::include_vulkano_glsl(input.into()).into()
}

#[proc_macro]
pub fn include_vulkano_slang(input: TokenStream) -> TokenStream {
    vulkano_codegen::include_vulkano_slang(input.into()).into()
}

#[proc_macro]
pub fn shader(input: TokenStream) -> TokenStream {
    let ccx = CodegenContext::default();

    vulkano_codegen::shader(&ccx, input.into()).into()
}
