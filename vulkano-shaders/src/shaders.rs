use crate::{Compiler, MacroOptions, Result, ShaderKind, SourceLanguage};
use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::{
    fs,
    io::Write,
    iter::Iterator,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU32, Ordering},
};

pub(super) fn compile_shader(
    options: &MacroOptions,
    source: &str,
    working_dir: &Path,
    shader_kind: ShaderKind,
    macro_defines: &[(String, String)],
) -> Result<(Vec<u32>, Vec<String>), String> {
    let compiler = match options.source_language {
        SourceLanguage::Glsl | SourceLanguage::Hlsl => Compiler::Shaderc,
        SourceLanguage::Slang => Compiler::Slangc,
    };
    let entry_point = "main";

    let mut command = Command::new(compiler.as_command());

    command.current_dir(working_dir);

    let vulkano_temp_dir = create_vulkano_dir()?;
    let vulkano_dir = &vulkano_temp_dir.0;
    let dependencies_file = vulkano_dir.join("deps.d");

    match compiler {
        Compiler::Shaderc => {
            command.arg("-x").arg(options.source_language.as_str());
            command.arg(format!("-fshader-stage={}", shader_kind.as_shaderc_stage()));
            command.arg(format!("-fentry-point={}", entry_point));
            let target_env = options.vulkan_version.as_shaderc_target_env();
            command.arg(format!("--target-env={}", target_env));
            let target_spv = options.spirv_version.as_shaderc_target_spv();
            command.arg(format!("--target-spv={}", target_spv));

            // vulkano.glsl dir first, then user include directories.
            command.arg("-I").arg(vulkano_dir);
            set_common_options(&mut command, options, macro_defines);

            command.arg("-MD");
            command.arg("-MF").arg(&dependencies_file);
            command.arg("-o").arg("-");
            command.arg("-");
        }
        Compiler::Slangc => {
            command.arg("-lang").arg(options.source_language.as_str());
            command.arg("-stage").arg(shader_kind.as_slangc_stage());
            command.arg("-entry").arg(entry_point);
            command.arg("-target").arg("spirv");
            let profile = options.spirv_version.as_slangc_profile();
            command.arg("-profile").arg(profile);

            // vulkano.glsl dir first, working dir for module imports, then user include
            // directories.
            command.arg("-I").arg(vulkano_dir);
            command.arg("-I").arg(working_dir);
            set_common_options(&mut command, options, macro_defines);

            command.arg("-depfile").arg(&dependencies_file);
            command.arg("--");
            command.arg("-");
        }
    }

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to call {}: {e}", compiler.as_command()))?;

    child
        .stdin
        .take()
        .ok_or_else(|| format!("failed to open {} stdin", compiler.as_command()))?
        .write_all(source.as_bytes())
        .map_err(|e| format!("failed to write to {} stdin: {e}", compiler.as_command()))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for {}: {e}", compiler.as_command()))?;

    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            compiler.as_command(),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    let content = &fs::read_to_string(&dependencies_file)
        .map_err(|e| format!("failed to read dependencies file: {e}"))?;
    let input_files = parse_deps_file(content, vulkano_dir, working_dir).map_err(|e| {
        let content = content
            .lines()
            .flat_map(|line| ["    ", line])
            .collect::<String>();

        format!("failed to parse dependencies file: {e}\nfile content:\n{content}")
    })?;

    let words =
        crate::spirv_bytes_to_words(output.stdout).map_err(|e| format!("malformed SPIR-V: {e}"))?;

    Ok((words, input_files))
}

fn create_vulkano_dir() -> Result<TempDir, String> {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let vulkano_dir = TempDir(std::env::temp_dir().join(format!(
        "vulkano_shaders_{}_{}",
        std::process::id(),
        id,
    )));

    fs::create_dir_all(&vulkano_dir.0)
        .map_err(|e| format!("failed to create vulkano include dir: {e}"))?;
    fs::write(
        vulkano_dir.0.join("vulkano.glsl"),
        include_str!("../include/vulkano.glsl"),
    )
    .map_err(|e| format!("failed to write vulkano.glsl: {e}"))?;

    Ok(vulkano_dir)
}

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn set_common_options(
    command: &mut Command,
    options: &MacroOptions,
    macro_defines: &[(String, String)],
) {
    if cfg!(feature = "shaderc-debug") {
        command.arg("-g");
    }

    for dir in &options.include_directories {
        command.arg("-I").arg(dir);
    }

    for (name, value) in &options.global_macro_defines {
        command.arg(format!("-D{name}={value}"));
    }

    for (name, value) in macro_defines {
        command.arg(format!("-D{name}={value}"));
    }
}

/// Parses a Makefile-format dependency file produced by glslc `-MF`, returning the list of
/// included file paths. The format is `target: source dep1 dep2 ...`.
fn parse_deps_file(
    content: &str,
    vulkano_dir: &Path,
    working_dir: &Path,
) -> Result<Vec<String>, String> {
    fn take_while(input: &str, predicate: impl FnMut(char) -> bool) -> (&str, &str) {
        let index = input.len() - input.trim_start_matches(predicate).len();

        input.split_at(index)
    }

    fn take_until(input: &str, predicate: impl FnMut(char) -> bool) -> (&str, &str) {
        let Some(index) = input.find(predicate) else {
            return (input, "");
        };

        input.split_at(index)
    }

    fn is_space(c: char) -> bool {
        c == ' '
    }

    fn is_space_or_eol(c: char) -> bool {
        c == ' ' || c == '\n'
    }

    fn found(input: &str) -> String {
        if let Some(c) = input.chars().next() {
            format!("`{}`", c.escape_default())
        } else {
            "EOF".to_owned()
        }
    }

    let column = |input: &str| content.len() - input.len() + 1;

    // Everything after the first `:` is in the format `source dep1 dep2 ...`.
    let Some((start, input)) = content.split_once(':') else {
        return Ok(Vec::new());
    };

    if start.contains('\n') {
        return Err("expected no new lines before the `:`".to_owned());
    }

    // Skip the source.
    let Some(input) = input.strip_prefix(" <stdin>") else {
        let column = column(input);
        let found = found(input);
        return Err(format!("1:{column}: expected ` <stdin>`, found {found}"));
    };

    if input.starts_with('\n') {
        return Ok(Vec::new());
    }

    let Some(input) = input.strip_prefix(' ') else {
        let column = column(input);
        let found = found(input);
        return Err(format!("1:{column}: expected `\\n` or ` `, found {found}"));
    };

    let mut paths = Vec::new();
    let mut path = String::new();
    let mut input = input;

    // HACK: The Makefile format doesn't support path quoting, and shaderc just outputs paths that
    // contain spaces unquoted. Meaning that we don't know if a space is an actual delimiter or
    // part of a path. What we do is that we treat a file extension followed by a single space as
    // an actual delimiter, and treat any other space as part of a path. This rules out file and/or
    // directory names that contain an extension followed by space(s), as well as file names
    // without an extension.
    loop {
        let (spaces, rest) = take_while(input, is_space);
        let (token, rest) = take_until(rest, is_space_or_eol);
        let has_extension = Path::new(token).extension().is_some();

        path.push_str(spaces);
        path.push_str(token);

        if has_extension {
            let normalized = {
                let mut result = normalize_str(path.replace("\\:", ":"));
                if !Path::new(&result).is_absolute() {
                    result =
                        normalize_str(PathBuf::from_iter([working_dir, &PathBuf::from(result)]))
                }
                result
            };

            if !Path::new(&normalized).starts_with(vulkano_dir) {
                if !Path::new(&normalized).is_file() {
                    return Err(format!(
                        "expected `{normalized}` to be a file existing on the file system",
                    ));
                }

                paths.push(normalized);
            }

            path = String::new();
        }

        input = rest;

        if input.starts_with('\n') {
            if !path.is_empty() {
                let column = column(input);
                return Err(format!(
                    "1:{column}: expected a file extension, found `\\n`",
                ));
            }

            break;
        }

        if has_extension {
            input = input.strip_prefix(' ').unwrap();
        }
    }

    Ok(paths)
}

fn normalize_str(path: impl AsRef<Path>) -> String {
    fn inner(path: &Path) -> String {
        path.components()
            .collect::<PathBuf>()
            .into_os_string()
            .into_string()
            .unwrap()
    }

    inner(path.as_ref())
}

pub(super) fn generate_shaders(
    name: Option<&str>,
    words: &[u32],
    input_paths: Vec<String>,
) -> Result<TokenStream> {
    let include_bytes = input_paths.into_iter().map(|s| {
        quote! {
            // Using `include_bytes` here ensures that changing the shader will force recompilation.
            // The bytes themselves can be optimized out by the compiler as they are unused.
            ::std::include_bytes!( #s )
        }
    });

    let load_name = if let Some(name) = &name {
        format_ident!("load_{}", name.to_snake_case())
    } else {
        format_ident!("load")
    };
    let try_load_name = format_ident!("try_{load_name}");
    let load_unchecked_name = format_ident!("{load_name}_unchecked");
    let words_name = format_ident!("{}_WORDS", load_name.to_string().to_uppercase());

    let shaders_code = quote! {
        const _: &[&[u8]] = &[ #( #include_bytes ),* ];

        static #words_name: &[u32] = &[ #( #words ),* ];

        /// Loads the shader as a `ShaderModule`, panicking on a validation error.
        #[allow(unsafe_code)]
        #[inline]
        #[track_caller]
        pub unsafe fn #load_name(
            device: &::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::VulkanError,
        > {
            match unsafe { #try_load_name(device) } {
                ::std::result::Result::Ok(shader_module) => {
                    ::std::result::Result::Ok(shader_module)
                }
                ::std::result::Result::Err(err) => {
                    ::std::result::Result::Err(::vulkano::Validated::unwrap(err))
                }
            }
        }

        /// Loads the shader as a `ShaderModule`.
        #[allow(unsafe_code)]
        #[inline]
        pub unsafe fn #try_load_name(
            device: &::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::Validated<::vulkano::VulkanError>,
        > {
            unsafe {
                ::vulkano::shader::ShaderModule::try_new(
                    device,
                    &::vulkano::shader::ShaderModuleCreateInfo::new(#words_name),
                )
            }
        }

        /// Loads the shader as a `ShaderModule`, skipping validation.
        #[allow(unsafe_code)]
        #[inline]
        pub unsafe fn #load_unchecked_name(
            device: &::std::sync::Arc<::vulkano::device::Device>,
        ) -> ::std::result::Result<
            ::std::sync::Arc<::vulkano::shader::ShaderModule>,
            ::vulkano::VulkanError,
        > {
            unsafe {
                ::vulkano::shader::ShaderModule::new_unchecked(
                    device,
                    &::vulkano::shader::ShaderModuleCreateInfo::new(#words_name),
                )
            }
        }
    };

    Ok(shaders_code)
}
