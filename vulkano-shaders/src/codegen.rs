use crate::{
    structs::{self, TypeRegistry},
    MacroInput,
};
use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
pub use shaderc::{CompilationArtifact, IncludeType, ResolvedInclude, ShaderKind};
use shaderc::{CompileOptions, Compiler, EnvVersion, TargetEnv};
use std::{
    cell::RefCell,
    fs,
    iter::Iterator,
    path::{Path, PathBuf},
};
use syn::{Error, LitStr};
use vulkano::shader::spirv::Spirv;

pub struct Shader {
    pub source: LitStr,
    pub name: String,
    pub spirv: Spirv,
}

#[allow(clippy::too_many_arguments)]
fn include_callback(
    requested_source_path_raw: &str,
    directive_type: IncludeType,
    contained_within_path_raw: &str,
    recursion_depth: usize,
    include_directories: &[PathBuf],
    root_source_has_path: bool,
    base_path: &Path,
    includes: &mut Vec<String>,
) -> Result<ResolvedInclude, String> {
    let file_to_include = match directive_type {
        IncludeType::Relative => {
            let requested_source_path = Path::new(requested_source_path_raw);
            // If the shader source is embedded within the macro, abort unless we get an absolute
            // path.
            if !root_source_has_path && recursion_depth == 1 && !requested_source_path.is_absolute()
            {
                let requested_source_name = requested_source_path
                    .file_name()
                    .expect("failed to get the name of the requested source file")
                    .to_string_lossy();
                let requested_source_directory = requested_source_path
                    .parent()
                    .expect("failed to get the directory of the requested source file")
                    .to_string_lossy();

                return Err(format!(
                    "usage of relative paths in imports in embedded GLSL is not allowed, try \
                    using `#include <{}>` and adding the directory `{}` to the `include` array in \
                    your `shader!` macro call instead",
                    requested_source_name, requested_source_directory,
                ));
            }

            let mut resolved_path = if recursion_depth == 1 {
                Path::new(contained_within_path_raw)
                    .parent()
                    .map(|parent| base_path.join(parent))
            } else {
                Path::new(contained_within_path_raw)
                    .parent()
                    .map(|parent| parent.to_owned())
            }
            .unwrap_or_else(|| {
                panic!(
                    "the file `{}` does not reside in a directory, this is an implementation \
                    error",
                    contained_within_path_raw,
                )
            });
            resolved_path.push(requested_source_path);

            if !resolved_path.is_file() {
                return Err(format!(
                    "invalid inclusion path `{}`, the path does not point to a file",
                    requested_source_path_raw,
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
                    "no such file found as specified by the absolute path; keep in mind that \
                    absolute paths cannot be used with inclusion from standard directories \
                    (`#include <...>`), try using `#include \"...\"` instead; requested path: {}",
                    requested_source_path_raw,
                ));
            }

            let found_requested_source_path = include_directories
                .iter()
                .map(|include_directory| include_directory.join(requested_source_path))
                .find(|resolved_requested_source_path| resolved_requested_source_path.is_file());

            if let Some(found_requested_source_path) = found_requested_source_path {
                found_requested_source_path
            } else {
                return Err(format!(
                    "failed to include the file `{}` from any include directories",
                    requested_source_path_raw,
                ));
            }
        }
    };

    let content = fs::read_to_string(file_to_include.as_path()).map_err(|err| {
        format!(
            "failed to read the contents of file `{file_to_include:?}` to be included in the \
            shader source: {err}",
        )
    })?;
    let resolved_name = file_to_include
        .into_os_string()
        .into_string()
        .map_err(|_| {
            "failed to stringify the file to be included; make sure the path consists of valid \
            unicode characters"
        })?;

    includes.push(resolved_name.clone());

    Ok(ResolvedInclude {
        resolved_name,
        content,
    })
}

pub(super) fn compile(
    input: &MacroInput,
    path: Option<String>,
    base_path: &Path,
    code: &str,
    shader_kind: ShaderKind,
) -> Result<(CompilationArtifact, Vec<String>), String> {
    let includes = RefCell::new(Vec::new());
    let compiler = Compiler::new().ok_or("failed to create GLSL compiler")?;
    let mut compile_options =
        CompileOptions::new().ok_or("failed to initialize compile options")?;

    compile_options.set_target_env(
        TargetEnv::Vulkan,
        input.vulkan_version.unwrap_or(EnvVersion::Vulkan1_0) as u32,
    );

    if let Some(spirv_version) = input.spirv_version {
        compile_options.set_target_spirv(spirv_version);
    }

    let root_source_path = path.as_deref().unwrap_or(
        // An arbitrary placeholder file name for embedded shaders.
        "shader.glsl",
    );

    // Specify the file resolution callback for the `#include` directive.
    compile_options.set_include_callback(
        |requested_source_path, directive_type, contained_within_path, recursion_depth| {
            include_callback(
                requested_source_path,
                directive_type,
                contained_within_path,
                recursion_depth,
                &input.include_directories,
                path.is_some(),
                base_path,
                &mut includes.borrow_mut(),
            )
        },
    );

    for (macro_name, macro_value) in &input.macro_defines {
        compile_options.add_macro_definition(macro_name, Some(macro_value));
    }

    #[cfg(feature = "shaderc-debug")]
    compile_options.set_generate_debug_info();

    let content = compiler
        .compile_into_spirv(
            code,
            shader_kind,
            root_source_path,
            "main",
            Some(&compile_options),
        )
        .map_err(|e| e.to_string().replace("(s): ", "(s):\n"))?;

    drop(compile_options);

    Ok((content, includes.into_inner()))
}

pub(super) fn reflect(
    input: &MacroInput,
    source: LitStr,
    name: String,
    words: &[u32],
    input_paths: Vec<String>,
    type_registry: &mut TypeRegistry,
) -> Result<(TokenStream, TokenStream), Error> {
    let spirv = Spirv::new(words).map_err(|err| {
        Error::new_spanned(&source, format!("failed to parse SPIR-V words: {err}"))
    })?;
    let shader = Shader {
        source,
        name,
        spirv,
    };

    let include_bytes = input_paths.into_iter().map(|s| {
        quote! {
            // Using `include_bytes` here ensures that changing the shader will force recompilation.
            // The bytes themselves can be optimized out by the compiler as they are unused.
            ::std::include_bytes!( #s )
        }
    });

    let load_name = if shader.name.is_empty() {
        format_ident!("load")
    } else {
        format_ident!("load_{}", shader.name.to_snake_case())
    };

    let shader_code = quote! {
        /// Loads the shader as a `ShaderModule`.
        #[allow(unsafe_code)]
        #[inline]
        pub fn #load_name(
            device: ::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::Validated<::vulkano::VulkanError>,
        > {
            let _bytes = ( #( #include_bytes ),* );

            static WORDS: &[u32] = &[ #( #words ),* ];

            unsafe {
                ::vulkano::shader::ShaderModule::new(
                    device,
                    ::vulkano::shader::ShaderModuleCreateInfo::new(&WORDS),
                )
            }
        }
    };

    let structs = structs::write_structs(input, &shader, type_registry)?;

    Ok((shader_code, structs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;
    use quote::ToTokens;
    use shaderc::SpirvVersion;
    use syn::{File, Item};
    use vulkano::shader::reflect;

    fn spv_to_words(data: &[u8]) -> Vec<u32> {
        data.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn convert_paths(root_path: &Path, paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|p| root_path.join(p).into_os_string().into_string().unwrap())
            .collect()
    }

    #[test]
    fn spirv_parse() {
        let insts = spv_to_words(include_bytes!("../tests/frag.spv"));
        Spirv::new(&insts).unwrap();
    }

    #[test]
    fn spirv_reflect() {
        let insts = spv_to_words(include_bytes!("../tests/frag.spv"));

        let mut type_registry = TypeRegistry::default();
        let (_shader_code, _structs) = reflect(
            &MacroInput::empty(),
            LitStr::new("../tests/frag.spv", Span::call_site()),
            String::new(),
            &insts,
            Vec::new(),
            &mut type_registry,
        )
        .expect("reflecting spv failed");

        assert_eq!(_structs.to_string(), "", "No structs should be generated");
    }

    #[test]
    fn include_resolution() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let (_compile_relative, _) = compile(
            &MacroInput::empty(),
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            r#"
                #version 450
                #include "include_dir_a/target_a.glsl"
                #include "include_dir_b/target_b.glsl"
                void main() {}
            "#,
            ShaderKind::Vertex,
        )
        .expect("cannot resolve include files");

        let (_compile_include_paths, includes) = compile(
            &MacroInput {
                include_directories: vec![
                    root_path.join("tests").join("include_dir_a"),
                    root_path.join("tests").join("include_dir_b"),
                ],
                ..MacroInput::empty()
            },
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            r#"
                #version 450
                #include <target_a.glsl>
                #include <target_b.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
        )
        .expect("cannot resolve include files");

        assert_eq!(
            includes,
            convert_paths(
                &root_path,
                &[
                    ["tests", "include_dir_a", "target_a.glsl"]
                        .into_iter()
                        .collect(),
                    ["tests", "include_dir_b", "target_b.glsl"]
                        .into_iter()
                        .collect(),
                ],
            ),
        );

        let (_compile_include_paths_with_relative, includes_with_relative) = compile(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_a")],
                ..MacroInput::empty()
            },
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            r#"
                #version 450
                #include <target_a.glsl>
                #include <../include_dir_b/target_b.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
        )
        .expect("cannot resolve include files");

        assert_eq!(
            includes_with_relative,
            convert_paths(
                &root_path,
                &[
                    ["tests", "include_dir_a", "target_a.glsl"]
                        .into_iter()
                        .collect(),
                    ["tests", "include_dir_a", "../include_dir_b/target_b.glsl"]
                        .into_iter()
                        .collect(),
                ],
            ),
        );

        let absolute_path = root_path
            .join("tests")
            .join("include_dir_a")
            .join("target_a.glsl");
        let absolute_path_str = absolute_path
            .to_str()
            .expect("cannot run tests in a folder with non unicode characters");
        let (_compile_absolute_path, includes_absolute_path) = compile(
            &MacroInput::empty(),
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            &format!(
                r#"
                    #version 450
                    #include "{absolute_path_str}"
                    void main() {{}}
                "#,
            ),
            ShaderKind::Vertex,
        )
        .expect("cannot resolve include files");

        assert_eq!(
            includes_absolute_path,
            convert_paths(
                &root_path,
                &[["tests", "include_dir_a", "target_a.glsl"]
                    .into_iter()
                    .collect()],
            ),
        );

        let (_compile_recursive_, includes_recursive) = compile(
            &MacroInput {
                include_directories: vec![
                    root_path.join("tests").join("include_dir_b"),
                    root_path.join("tests").join("include_dir_c"),
                ],
                ..MacroInput::empty()
            },
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            r#"
                #version 450
                #include <target_c.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
        )
        .expect("cannot resolve include files");

        assert_eq!(
            includes_recursive,
            convert_paths(
                &root_path,
                &[
                    ["tests", "include_dir_c", "target_c.glsl"]
                        .into_iter()
                        .collect(),
                    ["tests", "include_dir_c", "../include_dir_a/target_a.glsl"]
                        .into_iter()
                        .collect(),
                    ["tests", "include_dir_b", "target_b.glsl"]
                        .into_iter()
                        .collect(),
                ],
            ),
        );
    }

    #[test]
    fn macros() {
        let need_defines = r#"
            #version 450
            #if defined(NAME1) && NAME2 > 29
            void main() {}
            #endif
        "#;

        let compile_no_defines = compile(
            &MacroInput::empty(),
            None,
            Path::new(""),
            need_defines,
            ShaderKind::Vertex,
        );
        assert!(compile_no_defines.is_err());

        compile(
            &MacroInput {
                macro_defines: vec![("NAME1".into(), "".into()), ("NAME2".into(), "58".into())],
                ..MacroInput::empty()
            },
            None,
            Path::new(""),
            need_defines,
            ShaderKind::Vertex,
        )
        .expect("setting shader macros did not work");
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
    fn descriptor_calculation_with_multiple_entrypoints() {
        let insts = spv_to_words(include_bytes!("../tests/multiple_entrypoints.spv"));
        let spirv = Spirv::new(&insts).unwrap();

        let mut descriptors = Vec::new();
        for (_, info) in reflect::entry_points(&spirv) {
            descriptors.push(info.descriptor_binding_requirements);
        }

        // Check first entrypoint
        let e1_descriptors = &descriptors[0];
        let mut e1_bindings = Vec::new();
        for loc in e1_descriptors.keys() {
            e1_bindings.push(*loc);
        }

        assert_eq!(e1_bindings.len(), 5);
        assert!(e1_bindings.contains(&(0, 0)));
        assert!(e1_bindings.contains(&(0, 1)));
        assert!(e1_bindings.contains(&(0, 2)));
        assert!(e1_bindings.contains(&(0, 3)));
        assert!(e1_bindings.contains(&(0, 4)));

        // Check second entrypoint
        let e2_descriptors = &descriptors[1];
        let mut e2_bindings = Vec::new();
        for loc in e2_descriptors.keys() {
            e2_bindings.push(*loc);
        }

        assert_eq!(e2_bindings.len(), 3);
        assert!(e2_bindings.contains(&(0, 0)));
        assert!(e2_bindings.contains(&(0, 1)));
        assert!(e2_bindings.contains(&(0, 2)));
    }

    #[test]
    fn reflect_descriptor_calculation_with_multiple_entrypoints() {
        let insts = spv_to_words(include_bytes!("../tests/multiple_entrypoints.spv"));

        let mut type_registry = TypeRegistry::default();
        let (_shader_code, _structs) = reflect(
            &MacroInput::empty(),
            LitStr::new("../tests/multiple_entrypoints.spv", Span::call_site()),
            String::new(),
            &insts,
            Vec::new(),
            &mut type_registry,
        )
        .expect("reflecting spv failed");

        let structs = _structs.to_string();
        assert_ne!(structs, "", "Has some structs");

        let file: File = syn::parse2(_structs).unwrap();
        let structs: Vec<_> = file
            .items
            .iter()
            .filter_map(|item| {
                if let Item::Struct(s) = item {
                    Some(s)
                } else {
                    None
                }
            })
            .collect();

        let buffer = structs.iter().find(|s| s.ident == "Buffer").unwrap();
        assert_eq!(
            buffer.fields.to_token_stream().to_string(),
            quote!({pub data: u32,}).to_string()
        );

        let uniform = structs.iter().find(|s| s.ident == "Uniform").unwrap();
        assert_eq!(
            uniform.fields.to_token_stream().to_string(),
            quote!({pub data: u32,}).to_string()
        );
    }

    fn descriptor_calculation_with_multiple_functions_shader() -> (CompilationArtifact, Vec<String>)
    {
        compile(
            &MacroInput {
                spirv_version: Some(SpirvVersion::V1_6),
                vulkan_version: Some(EnvVersion::Vulkan1_3),
                ..MacroInput::empty()
            },
            None,
            Path::new(""),
            r#"
                #version 460

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
            "#,
            ShaderKind::Vertex,
        )
        .unwrap()
    }

    #[test]
    fn descriptor_calculation_with_multiple_functions() {
        let (artifact, _) = descriptor_calculation_with_multiple_functions_shader();
        let spirv = Spirv::new(artifact.as_binary()).unwrap();

        if let Some((_, info)) = reflect::entry_points(&spirv).next() {
            let mut bindings = Vec::new();
            for (loc, _reqs) in info.descriptor_binding_requirements {
                bindings.push(loc);
            }

            assert_eq!(bindings.len(), 4);
            assert!(bindings.contains(&(1, 0)));
            assert!(bindings.contains(&(2, 0)));
            assert!(bindings.contains(&(3, 1)));
            assert!(bindings.contains(&(3, 2)));

            return;
        }
        panic!("could not find entrypoint");
    }

    #[test]
    fn reflect_descriptor_calculation_with_multiple_functions() {
        let (artifact, _) = descriptor_calculation_with_multiple_functions_shader();

        let mut type_registry = TypeRegistry::default();
        let (_shader_code, _structs) = reflect(
            &MacroInput::empty(),
            LitStr::new(
                "descriptor_calculation_with_multiple_functions_shader",
                Span::call_site(),
            ),
            String::new(),
            artifact.as_binary(),
            Vec::new(),
            &mut type_registry,
        )
        .expect("reflecting spv failed");

        let structs = _structs.to_string();
        assert_ne!(structs, "", "Has some structs");

        let file: File = syn::parse2(_structs).unwrap();
        let structs: Vec<_> = file
            .items
            .iter()
            .filter_map(|item| {
                if let Item::Struct(s) = item {
                    Some(s)
                } else {
                    None
                }
            })
            .collect();

        let buffer = structs.iter().find(|s| s.ident == "Buffer").unwrap();
        assert_eq!(
            buffer.fields.to_token_stream().to_string(),
            quote!({pub data: [f32; 3usize],}).to_string()
        );

        let uniform = structs.iter().find(|s| s.ident == "Uniform").unwrap();
        assert_eq!(
            uniform.fields.to_token_stream().to_string(),
            quote!({pub data: f32,}).to_string()
        );
    }
}
