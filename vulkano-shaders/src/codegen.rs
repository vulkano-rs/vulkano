// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::entry_point;
use crate::read_file_to_string;
use crate::structs;
use crate::RegisteredType;
use crate::TypesMeta;
use proc_macro2::TokenStream;
pub use shaderc::{CompilationArtifact, IncludeType, ResolvedInclude, ShaderKind};
use shaderc::{CompileOptions, Compiler, EnvVersion, SpirvVersion, TargetEnv};
use std::collections::HashMap;
use std::iter::Iterator;
use std::path::Path;
use std::{
    cell::{RefCell, RefMut},
    io::Error as IoError,
};
use vulkano::shader::reflect;
use vulkano::shader::spirv::{Spirv, SpirvError};

pub(super) fn path_to_str(path: &Path) -> &str {
    path.to_str().expect(
        "Could not stringify the file to be included. Make sure the path consists of \
                 valid unicode characters.",
    )
}

fn include_callback(
    requested_source_path_raw: &str,
    directive_type: IncludeType,
    contained_within_path_raw: &str,
    recursion_depth: usize,
    include_directories: &[impl AsRef<Path>],
    root_source_has_path: bool,
    base_path: &impl AsRef<Path>,
    mut includes_tracker: RefMut<Vec<String>>,
) -> Result<ResolvedInclude, String> {
    let file_to_include = match directive_type {
        IncludeType::Relative => {
            let requested_source_path = Path::new(requested_source_path_raw);
            // Is embedded current shader source embedded within a rust macro?
            // If so, abort unless absolute path.
            if !root_source_has_path && recursion_depth == 1 && !requested_source_path.is_absolute()
            {
                let requested_source_name = requested_source_path
                    .file_name()
                    .expect("Could not get the name of the requested source file.")
                    .to_string_lossy();
                let requested_source_directory = requested_source_path
                    .parent()
                    .expect("Could not get the directory of the requested source file.")
                    .to_string_lossy();

                return Err(format!(
                    "Usage of relative paths in imports in embedded GLSL is not \
                                    allowed, try using `#include <{}>` and adding the directory \
                                    `{}` to the `include` array in your `shader!` macro call \
                                    instead.",
                    requested_source_name, requested_source_directory
                ));
            }

            let mut resolved_path = if recursion_depth == 1 {
                Path::new(contained_within_path_raw)
                    .parent()
                    .map(|parent| base_path.as_ref().join(parent))
            } else {
                Path::new(contained_within_path_raw)
                    .parent()
                    .map(|parent| parent.to_owned())
            }
            .unwrap_or_else(|| {
                panic!(
                    "The file `{}` does not reside in a directory. This is \
                                        an implementation error.",
                    contained_within_path_raw
                )
            });
            resolved_path.push(requested_source_path);

            if !resolved_path.is_file() {
                return Err(format!(
                    "Invalid inclusion path `{}`, the path does not point to a file.",
                    requested_source_path_raw
                ));
            }

            resolved_path
        }
        IncludeType::Standard => {
            let requested_source_path = Path::new(requested_source_path_raw);

            if requested_source_path.is_absolute() {
                // This message is printed either when using a missing file with an absolute path
                // in the relative include directive or when using absolute paths in a standard
                // include directive.
                return Err(format!(
                    "No such file found, as specified by the absolute path. \
                                    Keep in mind, that absolute paths cannot be used with \
                                    inclusion from standard directories (`#include <...>`), try \
                                    using `#include \"...\"` instead. Requested path: {}",
                    requested_source_path_raw
                ));
            }

            let found_requested_source_path = include_directories
                .iter()
                .map(|include_directory| include_directory.as_ref().join(requested_source_path))
                .find(|resolved_requested_source_path| resolved_requested_source_path.is_file());

            if let Some(found_requested_source_path) = found_requested_source_path {
                found_requested_source_path
            } else {
                return Err(format!(
                    "Could not include the file `{}` from any include directories.",
                    requested_source_path_raw
                ));
            }
        }
    };

    let file_to_include_string = path_to_str(file_to_include.as_path()).to_string();
    let content = read_file_to_string(file_to_include.as_path()).map_err(|_| {
        format!(
            "Could not read the contents of file `{}` to be included in the \
                              shader source.",
            &file_to_include_string
        )
    })?;

    includes_tracker.push(file_to_include_string.clone());

    Ok(ResolvedInclude {
        resolved_name: file_to_include_string,
        content,
    })
}

pub fn compile(
    path: Option<String>,
    base_path: &impl AsRef<Path>,
    code: &str,
    ty: ShaderKind,
    include_directories: &[impl AsRef<Path>],
    macro_defines: &[(impl AsRef<str>, impl AsRef<str>)],
    vulkan_version: Option<EnvVersion>,
    spirv_version: Option<SpirvVersion>,
) -> Result<(CompilationArtifact, Vec<String>), String> {
    let includes_tracker = RefCell::new(Vec::new());
    let mut compiler = Compiler::new().ok_or("failed to create GLSL compiler")?;
    let mut compile_options = CompileOptions::new().ok_or("failed to initialize compile option")?;

    compile_options.set_target_env(
        TargetEnv::Vulkan,
        vulkan_version.unwrap_or(EnvVersion::Vulkan1_0) as u32,
    );

    if let Some(spirv_version) = spirv_version {
        compile_options.set_target_spirv(spirv_version);
    }

    let root_source_path = if let &Some(ref path) = &path {
        path
    } else {
        // An arbitrary placeholder file name for embedded shaders
        "shader.glsl"
    };

    // Specify file resolution callback for the `#include` directive
    compile_options.set_include_callback(
        |requested_source_path, directive_type, contained_within_path, recursion_depth| {
            include_callback(
                requested_source_path,
                directive_type,
                contained_within_path,
                recursion_depth,
                include_directories,
                path.is_some(),
                base_path,
                includes_tracker.borrow_mut(),
            )
        },
    );

    for (macro_name, macro_value) in macro_defines.iter() {
        compile_options.add_macro_definition(macro_name.as_ref(), Some(macro_value.as_ref()));
    }

    #[cfg(feature = "shaderc-debug")]
    compile_options.set_generate_debug_info();

    let content = compiler
        .compile_into_spirv(&code, ty, root_source_path, "main", Some(&compile_options))
        .map_err(|e| e.to_string())?;

    let includes = includes_tracker.borrow().clone();

    Ok((content, includes))
}

pub(super) fn reflect<'a, I>(
    prefix: &'a str,
    words: &[u32],
    types_meta: &TypesMeta,
    input_paths: I,
    shared_constants: bool,
    types_registry: &'a mut HashMap<String, RegisteredType>,
) -> Result<(TokenStream, TokenStream), Error>
where
    I: IntoIterator<Item = &'a str>,
{
    let spirv = Spirv::new(words)?;

    let include_bytes = input_paths.into_iter().map(|s| {
        quote! {
            // using include_bytes here ensures that changing the shader will force recompilation.
            // The bytes themselves can be optimized out by the compiler as they are unused.
            ::std::include_bytes!( #s )
        }
    });

    let spirv_version = {
        let major = spirv.version().major;
        let minor = spirv.version().minor;
        let patch = spirv.version().patch;
        quote! {
            Version {
                major: #major,
                minor: #minor,
                patch: #patch,
            }
        }
    };
    let spirv_capabilities = reflect::spirv_capabilities(&spirv).map(|capability| {
        let name = format_ident!("{}", format!("{:?}", capability));
        quote! { &Capability::#name }
    });
    let spirv_extensions = reflect::spirv_extensions(&spirv);
    let entry_points = reflect::entry_points(&spirv)
        .map(|(name, model, info)| entry_point::write_entry_point(&name, model, &info));

    let specialization_constants = structs::write_specialization_constants(
        prefix,
        &spirv,
        types_meta,
        shared_constants,
        types_registry,
    );

    let load_name = if prefix.is_empty() {
        format_ident!("load")
    } else {
        format_ident!("load_{}", prefix)
    };

    let shader_code = quote! {
        /// Loads the shader in Vulkan as a `ShaderModule`.
        #[inline]
        #[allow(unsafe_code)]
        pub fn #load_name(device: ::std::sync::Arc<::vulkano::device::Device>)
            -> Result<::std::sync::Arc<::vulkano::shader::ShaderModule>, ::vulkano::shader::ShaderCreationError>
        {
            use vulkano::shader::EntryPointInfo;
            use vulkano::shader::GeometryShaderExecution;
            use vulkano::shader::ShaderExecution;
            use vulkano::shader::ShaderModule;
            use vulkano::shader::ShaderStage;
            use vulkano::shader::SpecializationConstantRequirements;
            use vulkano::shader::spirv::Capability;
            use vulkano::Version;

            let _bytes = ( #( #include_bytes),* );

            static WORDS: &[u32] = &[ #( #words ),* ];

            unsafe {
                Ok(ShaderModule::from_words_with_data(
                    device,
                    WORDS,
                    #spirv_version,
                    [#(#spirv_capabilities),*],
                    [#(#spirv_extensions),*],
                    [#(#entry_points),*],
                )?)
            }
        }

        #specialization_constants
    };

    let structs = structs::write_structs(prefix, &spirv, types_meta, types_registry);

    Ok((shader_code, structs))
}

#[derive(Debug)]
pub enum Error {
    IoError(IoError),
    SpirvError(SpirvError),
}

impl From<IoError> for Error {
    #[inline]
    fn from(err: IoError) -> Error {
        Error::IoError(err)
    }
}

impl From<SpirvError> for Error {
    #[inline]
    fn from(err: SpirvError) -> Error {
        Error::SpirvError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::compile;
    use shaderc::ShaderKind;
    use std::path::{Path, PathBuf};
    use vulkano::shader::{reflect, spirv::Spirv};

    #[cfg(not(target_os = "windows"))]
    pub fn path_separator() -> &'static str {
        "/"
    }

    #[cfg(target_os = "windows")]
    pub fn path_separator() -> &'static str {
        "\\"
    }

    fn convert_paths(root_path: &Path, paths: &[String]) -> Vec<String> {
        paths
            .iter()
            .map(|p| path_to_str(root_path.join(p).as_path()).to_owned())
            .collect()
    }

    #[test]
    fn spirv_parse() {
        let data = include_bytes!("../tests/frag.spv");
        let insts: Vec<_> = data
            .chunks(4)
            .map(|c| {
                ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) | c[0] as u32
            })
            .collect();

        Spirv::new(&insts).unwrap();
    }

    #[test]
    fn test_bad_alignment() {
        // vec3/mat3/mat3x* are problematic in arrays since their rust
        // representations don't have the same array stride as the SPIR-V
        // ones. E.g. in a vec3[2], the second element starts on the 16th
        // byte, but in a rust [[f32;3];2], the second element starts on the
        // 12th byte. Since we can't generate code for these types, we should
        // create an error instead of generating incorrect code.
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        struct MyStruct {
            vec3 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        let res = std::panic::catch_unwind(|| {
            structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new())
        });
        assert!(res.is_err());
    }
    #[test]
    fn test_trivial_alignment() {
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        struct MyStruct {
            vec4 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new());
    }
    #[test]
    fn test_wrap_alignment() {
        // This is a workaround suggested in the case of test_bad_alignment,
        // so we should make sure it works.
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        struct Vec3Wrap {
            vec3 v;
        };
        struct MyStruct {
            Vec3Wrap vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new());
    }

    #[test]
    fn test_include_resolution() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let empty_includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (_compile_relative, _) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include \"include_dir_a/target_a.glsl\"
        #include \"include_dir_b/target_b.glsl\"
        void main() {}
        ",
            ShaderKind::Vertex,
            &empty_includes,
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");

        let (_compile_include_paths, includes) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include <target_a.glsl>
        #include <target_b.glsl>
        void main() {}
        ",
            ShaderKind::Vertex,
            &[
                root_path.join("tests").join("include_dir_a"),
                root_path.join("tests").join("include_dir_b"),
            ],
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes,
            convert_paths(
                &root_path,
                &[
                    vec!["tests", "include_dir_a", "target_a.glsl"].join(path_separator()),
                    vec!["tests", "include_dir_b", "target_b.glsl"].join(path_separator()),
                ]
            )
        );

        let (_compile_include_paths_with_relative, includes_with_relative) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include <target_a.glsl>
        #include <../include_dir_b/target_b.glsl>
        void main() {}
        ",
            ShaderKind::Vertex,
            &[root_path.join("tests").join("include_dir_a")],
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes_with_relative,
            convert_paths(
                &root_path,
                &[
                    vec!["tests", "include_dir_a", "target_a.glsl"].join(path_separator()),
                    vec!["tests", "include_dir_a", "../include_dir_b/target_b.glsl"]
                        .join(path_separator()),
                ]
            )
        );

        let absolute_path = root_path
            .join("tests")
            .join("include_dir_a")
            .join("target_a.glsl");
        let absolute_path_str = absolute_path
            .to_str()
            .expect("Cannot run tests in a folder with non unicode characters");
        let (_compile_absolute_path, includes_absolute_path) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            &format!(
                "
        #version 450
        #include \"{}\"
        void main() {{}}
        ",
                absolute_path_str
            ),
            ShaderKind::Vertex,
            &empty_includes,
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes_absolute_path,
            convert_paths(
                &root_path,
                &[vec!["tests", "include_dir_a", "target_a.glsl"].join(path_separator())]
            )
        );

        let (_compile_recursive_, includes_recursive) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include <target_c.glsl>
        void main() {}
        ",
            ShaderKind::Vertex,
            &[
                root_path.join("tests").join("include_dir_b"),
                root_path.join("tests").join("include_dir_c"),
            ],
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes_recursive,
            convert_paths(
                &root_path,
                &[
                    vec!["tests", "include_dir_c", "target_c.glsl"].join(path_separator()),
                    vec!["tests", "include_dir_c", "../include_dir_a/target_a.glsl"]
                        .join(path_separator()),
                    vec!["tests", "include_dir_b", "target_b.glsl"].join(path_separator()),
                ]
            )
        );
    }

    #[test]
    fn test_macros() {
        let empty_includes: [PathBuf; 0] = [];
        let defines = vec![("NAME1", ""), ("NAME2", "58")];
        let no_defines: [(String, String); 0] = [];
        let need_defines = "
        #version 450
        #if defined(NAME1) && NAME2 > 29
        void main() {}
        #endif
        ";
        let compile_no_defines = compile(
            None,
            &Path::new(""),
            need_defines,
            ShaderKind::Vertex,
            &empty_includes,
            &no_defines,
            None,
            None,
        );
        assert!(compile_no_defines.is_err());

        let compile_defines = compile(
            None,
            &Path::new(""),
            need_defines,
            ShaderKind::Vertex,
            &empty_includes,
            &defines,
            None,
            None,
        );
        compile_defines.expect("Setting shader macros did not work");
    }

    /// `entrypoint1.frag.glsl`:
    /// ```glsl
    /// #version 450
    ///
    /// layout(set = 0, binding = 0) uniform Uniform {
    ///     uint data;
    /// } ubo;
    ///
    /// layout(set = 0, binding = 1) buffer Buffer {
    ///     uint data;
    /// } bo;
    ///
    /// layout(set = 0, binding = 2) uniform sampler textureSampler;
    /// layout(set = 0, binding = 3) uniform texture2D imageTexture;
    ///
    /// layout(push_constant) uniform PushConstant {
    ///    uint data;
    /// } push;
    ///
    /// layout(input_attachment_index = 0, set = 0, binding = 4) uniform subpassInput inputAttachment;
    ///
    /// layout(location = 0) out vec4 outColor;
    ///
    /// void entrypoint1() {
    ///     bo.data = 12;
    ///     outColor = vec4(
    ///         float(ubo.data),
    ///         float(push.data),
    ///         texture(sampler2D(imageTexture, textureSampler), vec2(0.0, 0.0)).x,
    ///         subpassLoad(inputAttachment).x
    ///     );
    /// }
    /// ```
    ///
    /// `entrypoint2.frag.glsl`:
    /// ```glsl
    /// #version 450
    ///
    /// layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputAttachment2;
    ///
    /// layout(set = 0, binding = 1) buffer Buffer {
    ///     uint data;
    /// } bo2;
    ///
    /// layout(set = 0, binding = 2) uniform Uniform {
    ///     uint data;
    /// } ubo2;
    ///
    /// layout(push_constant) uniform PushConstant {
    ///    uint data;
    /// } push2;
    ///
    /// void entrypoint2() {
    ///     bo2.data = ubo2.data + push2.data + int(subpassLoad(inputAttachment2).y);
    /// }
    /// ```
    ///
    /// Compiled and linked with:
    /// ```sh
    /// glslangvalidator -e entrypoint1 --source-entrypoint entrypoint1 -V100 entrypoint1.frag.glsl -o entrypoint1.spv
    /// glslangvalidator -e entrypoint2 --source-entrypoint entrypoint2 -V100 entrypoint2.frag.glsl -o entrypoint2.spv
    /// spirv-link entrypoint1.spv entrypoint2.spv -o multiple_entrypoints.spv
    /// ```
    #[test]
    fn test_descriptor_calculation_with_multiple_entrypoints() {
        let data = include_bytes!("../tests/multiple_entrypoints.spv");
        let instructions: Vec<u32> = data
            .chunks(4)
            .map(|c| {
                ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) | c[0] as u32
            })
            .collect();
        let spirv = Spirv::new(&instructions).unwrap();

        let mut descriptors = Vec::new();
        for (_, _, info) in reflect::entry_points(&spirv) {
            descriptors.push(info.descriptor_requirements);
        }

        // Check first entrypoint
        let e1_descriptors = descriptors.get(0).expect("Could not find entrypoint1");
        let mut e1_bindings = Vec::new();
        for (loc, _reqs) in e1_descriptors {
            e1_bindings.push(*loc);
        }
        assert_eq!(e1_bindings.len(), 5);
        assert!(e1_bindings.contains(&(0, 0)));
        assert!(e1_bindings.contains(&(0, 1)));
        assert!(e1_bindings.contains(&(0, 2)));
        assert!(e1_bindings.contains(&(0, 3)));
        assert!(e1_bindings.contains(&(0, 4)));

        // Check second entrypoint
        let e2_descriptors = descriptors.get(1).expect("Could not find entrypoint2");
        let mut e2_bindings = Vec::new();
        for (loc, _reqs) in e2_descriptors {
            e2_bindings.push(*loc);
        }
        assert_eq!(e2_bindings.len(), 3);
        assert!(e2_bindings.contains(&(0, 0)));
        assert!(e2_bindings.contains(&(0, 1)));
        assert!(e2_bindings.contains(&(0, 2)));
    }

    #[test]
    fn test_descriptor_calculation_with_multiple_functions() {
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450

        layout(set = 1, binding = 0) buffer Buffer {
            vec3 data;
        } bo;

        layout(set = 2, binding = 0) uniform Uniform {
            float data;
        } ubo;

        layout(set = 3, binding = 1) uniform sampler textureSampler;
        layout(set = 3, binding = 2) uniform texture2D imageTexture;

        float withMagicSparkles(float data) {
            return texture(sampler2D(imageTexture, textureSampler), vec2(data, data)).x;
        }

        vec3 makeSecretSauce() {
            return vec3(withMagicSparkles(ubo.data));
        }

        void main() {
            bo.data = makeSecretSauce();
        }
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();

        for (_, _, info) in reflect::entry_points(&spirv) {
            let mut bindings = Vec::new();
            for (loc, _reqs) in info.descriptor_requirements {
                bindings.push(loc);
            }
            assert_eq!(bindings.len(), 4);
            assert!(bindings.contains(&(1, 0)));
            assert!(bindings.contains(&(2, 0)));
            assert!(bindings.contains(&(3, 1)));
            assert!(bindings.contains(&(3, 2)));

            return;
        }
        panic!("Could not find entrypoint");
    }
}
