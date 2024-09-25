use crate::MacroOptions;
use ahash::{HashMap, HashSet};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use shaderc::{
    CompilationArtifact, CompileOptions, Compiler, IncludeType, ResolvedInclude, ShaderKind,
    SourceLanguage, TargetEnv,
};
use std::{
    cell::RefCell,
    fs,
    iter::Iterator,
    path::{Path, PathBuf},
};

pub(super) fn compile(
    compiler: &Compiler,
    macro_options: &MacroOptions,
    source_path: Option<&str>,
    shader_name: &str,
    source_code: &str,
    entry_points: HashMap<Option<String>, ShaderKind>,
    base_path: &Path,
    include_paths: &[PathBuf],
    sources_to_include: &RefCell<HashSet<String>>,
) -> Result<Vec<(Option<String>, CompilationArtifact)>, String> {
    let mut compile_options = CompileOptions::new().ok_or("failed to create compile options")?;

    compile_options.set_source_language(macro_options.source_language);

    if let Some(vulkan_version) = macro_options.vulkan_version {
        compile_options.set_target_env(TargetEnv::Vulkan, vulkan_version as u32);
    }

    if let Some(spirv_version) = macro_options.spirv_version {
        compile_options.set_target_spirv(spirv_version);
    }

    compile_options.set_include_callback(
        |requested_source, include_type, containing_source, depth| {
            include_callback(
                Path::new(requested_source),
                include_type,
                Path::new(containing_source),
                depth,
                base_path,
                include_paths,
                source_path.is_none(),
                &mut sources_to_include.borrow_mut(),
            )
        },
    );

    for (name, value) in &macro_options.macro_defines {
        compile_options.add_macro_definition(name, value.as_deref());
    }

    let file_name = match (source_path, macro_options.source_language) {
        (Some(source_path), _) => source_path,
        (None, SourceLanguage::GLSL) => &format!("{shader_name}.glsl"),
        (None, SourceLanguage::HLSL) => &format!("{shader_name}.hlsl"),
    };

    entry_points
        .into_iter()
        .map(|(entry_point, shader_kind)| {
            compiler
                .compile_into_spirv(
                    source_code,
                    shader_kind,
                    file_name,
                    entry_point.as_deref().unwrap_or("main"),
                    Some(&compile_options),
                )
                .map(|artifact| (entry_point, artifact))
                .map_err(|err| err.to_string().replace("\n", " "))
        })
        .collect()
}

fn include_callback(
    requested_source: &Path,
    include_type: IncludeType,
    containing_source: &Path,
    depth: usize,
    base_path: &Path,
    include_paths: &[PathBuf],
    embedded_root_source: bool,
    sources_to_include: &mut HashSet<String>,
) -> Result<ResolvedInclude, String> {
    let resolved_path = match include_type {
        IncludeType::Relative => {
            if depth == 1 && embedded_root_source && !requested_source.is_absolute() {
                return Err(
                    "you cannot use relative include directives in embedded shader source code; \
                    try using `#include <...>` instead"
                        .to_owned(),
                );
            }

            let parent = containing_source.parent().unwrap();

            if depth == 1 {
                [base_path, parent, requested_source].iter().collect()
            } else {
                [parent, requested_source].iter().collect()
            }
        }
        IncludeType::Standard => {
            if requested_source.is_absolute() {
                // This is returned when attempting to include a missing file by an absolute path
                // in a relative include directive and when using an absolute path in a standard
                // include directive.
                return Err(
                    "the specified file was not found; if you're using an absolute path in a \
                    standard include directive (`#include <...>`), try using `#include \"...\"` \
                    instead"
                        .to_owned(),
                );
            }

            include_paths
                .iter()
                .map(|include_path| include_path.join(requested_source))
                .find(|source_path| source_path.is_file())
                .ok_or("the specified file was not found".to_owned())?
        }
    };

    let resolved_name = resolved_path.into_os_string().into_string().unwrap();

    let content = fs::read_to_string(&resolved_name)
        .map_err(|err| format!("failed to read `{resolved_name}`: {err}"))?;

    sources_to_include.insert(resolved_name.clone());

    Ok(ResolvedInclude {
        resolved_name,
        content,
    })
}

pub(super) fn generate_shader_code(
    entry_points: &[(Option<&str>, &[u32])],
    shader_name: &Option<String>,
) -> TokenStream {
    let load_fns = entry_points.iter().map(|(name, words)| {
        let load_name = match name {
            Some(name) => format_ident!("load_{name}"),
            None => format_ident!("load"),
        };

        quote! {
            #[allow(unsafe_code)]
            #[inline]
            pub fn #load_name(
                device: ::std::sync::Arc<::vulkano::device::Device>,
            ) -> ::std::result::Result<
                ::std::sync::Arc<::vulkano::shader::ShaderModule>,
                ::vulkano::Validated<::vulkano::VulkanError>,
            > {
                static WORDS: &[u32] = &[ #( #words ),* ];

                unsafe {
                    ::vulkano::shader::ShaderModule::new(
                        device,
                        ::vulkano::shader::ShaderModuleCreateInfo::new(WORDS),
                    )
                }
            }
        }
    });

    if let Some(shader_name) = shader_name {
        let shader_name = format_ident!("{shader_name}");

        quote! {
            pub mod #shader_name {
                #( #load_fns )*
            }
        }
    } else {
        quote! {
            #( #load_fns )*
        }
    }
}
