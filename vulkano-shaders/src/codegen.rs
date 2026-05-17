use crate::{
    structs::{self, TypeRegistry},
    EnvVersion, MacroInput, ShaderKind, SourceLanguage, SpirvVersion,
};
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
use syn::{Error, LitStr};
use vulkano::shader::spirv::Spirv;

pub struct Shader {
    pub source: LitStr,
    pub name: String,
    pub spirv: Spirv,
}

struct CompileOptions {
    source_language: SourceLanguage,
    target_env: EnvVersion,
    target_spirv: Option<SpirvVersion>,
    macro_definitions: Vec<(String, String)>,
    include_directories: Vec<PathBuf>,
    debug: bool,
}

impl CompileOptions {
    pub fn new() -> Self {
        CompileOptions {
            source_language: SourceLanguage::Glsl,
            target_env: EnvVersion::Vulkan1_0,
            target_spirv: None,
            macro_definitions: Vec::new(),
            include_directories: Vec::new(),
            debug: false,
        }
    }
}

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn compile_into_spirv(
    shader_kind: ShaderKind,
    source: &str,
    entry_point_name: &str,
    working_dir: &Path,
    options: &CompileOptions,
) -> Result<(Vec<u32>, Vec<String>), String> {
    // A guard against concurrent compilations using the same temp directory.
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let vulkano_dir = TempDir(std::env::temp_dir().join(format!(
        "vulkano_shaders_{}_{}",
        std::process::id(),
        id,
    )));
    let vulkano_dir = &vulkano_dir.0;

    // Write vulkano.glsl to a temp directory and prepend it as the first include path so glslc
    // can resolve `#include "vulkano.glsl"` or `#include <vulkano.glsl>`.
    fs::create_dir_all(vulkano_dir)
        .map_err(|e| format!("failed to create vulkano include dir: {e}"))?;
    fs::write(
        vulkano_dir.join("vulkano.glsl"),
        include_str!("../include/vulkano.glsl"),
    )
    .map_err(|e| format!("failed to write vulkano.glsl: {e}"))?;

    let dependencies_file = vulkano_dir.join("deps.d");

    let mut cmd = Command::new("glslc");

    cmd.current_dir(working_dir)
        .arg("-x")
        .arg(options.source_language.to_string())
        .arg(format!("--target-env={}", options.target_env));

    if let Some(spirv) = options.target_spirv {
        cmd.arg(format!("--target-spv={}", spirv));
    }

    if options.debug {
        cmd.arg("-g");
    }

    // vulkano.glsl dir first, then user include directories.
    cmd.arg(format!("-I{}", vulkano_dir.display()));
    for dir in &options.include_directories {
        cmd.arg(format!("-I{}", dir.display()));
    }

    for (macro_name, macro_value) in &options.macro_definitions {
        cmd.arg(format!("-D{macro_name}={macro_value}"));
    }

    cmd.arg(format!("-fshader-stage={}", shader_kind.as_glslc_stage()))
        .arg(format!("-fentry-point={}", entry_point_name))
        .arg("-MD")
        .arg("-MF")
        .arg(&dependencies_file)
        .arg("-o")
        .arg("-")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to call glslc: {e}"))?;

    child
        .stdin
        .take()
        .ok_or("failed to open glslc stdin")?
        .write_all(source.as_bytes())
        .map_err(|e| format!("failed to write to glslc stdin: {e}"))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for glslc: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "glslc failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let content = &fs::read_to_string(&dependencies_file)
        .map_err(|e| format!("failed to read dependencies file: {e}"))?;
    let includes = parse_deps_file(content, vulkano_dir, working_dir).map_err(|e| {
        let content = content
            .lines()
            .flat_map(|line| ["    ", line])
            .collect::<String>();

        format!("failed to parse dependencies file: {e}\nfile content:\n{content}")
    })?;

    let words = vulkano::shader::spirv::bytes_to_words(&output.stdout)
        .map(|w| w.into_owned())
        .map_err(|e| format!("malformed SPIR-V: {e}"))?;

    Ok((words, includes))
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
            let normalized = if Path::new(&path).is_absolute() {
                normalize_str(path)
            } else {
                normalize_str(PathBuf::from_iter([working_dir, &PathBuf::from(path)]))
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

pub(super) fn compile(
    input: &MacroInput,
    source: &str,
    working_dir: &Path,
    shader_kind: ShaderKind,
    macro_defines: &[(String, String)],
) -> Result<(Vec<u32>, Vec<String>), String> {
    let mut compile_options = CompileOptions::new();

    compile_options.source_language = input.source_language.unwrap_or(SourceLanguage::Glsl);
    compile_options.target_env = input.vulkan_version.unwrap_or(EnvVersion::Vulkan1_0);
    compile_options.target_spirv = input.spirv_version;
    compile_options.macro_definitions = input
        .global_macro_defines
        .iter()
        .chain(macro_defines.iter())
        .cloned()
        .collect();
    compile_options.include_directories = input.include_directories.clone();
    compile_options.debug = cfg!(feature = "shaderc-debug");

    compile_into_spirv(shader_kind, source, "main", working_dir, &compile_options)
        .map_err(|e| e.replace("(s): ", "(s):\n"))
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
        Error::new_spanned(&source, format_args!("failed to parse SPIR-V words: {err}"))
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
    let try_load_name = format_ident!("try_{load_name}");

    let shader_code = quote! {
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
            let _bytes = ( #( #include_bytes ),* );

            static WORDS: &[u32] = &[ #( #words ),* ];

            unsafe {
                ::vulkano::shader::ShaderModule::try_new(
                    device,
                    &::vulkano::shader::ShaderModuleCreateInfo::new(WORDS),
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
    use crate::EnvVersion;
    use proc_macro2::Span;
    use quote::ToTokens;
    use std::collections::HashSet;
    use syn::{File, Item};
    use vulkano::shader::reflect;

    fn compile_inline(
        input: &MacroInput,
        source: &str,
        shader_kind: ShaderKind,
        macro_defines: &[(String, String)],
    ) -> Result<(Vec<u32>, Vec<String>), String> {
        compile(input, source, Path::new("."), shader_kind, macro_defines)
    }

    fn convert_paths(root_path: &Path, paths: &[PathBuf]) -> HashSet<String> {
        paths
            .iter()
            .map(|p| normalize_str(root_path.join(p)))
            .collect()
    }

    #[test]
    fn spirv_parse() {
        let insts =
            vulkano::shader::spirv::bytes_to_words(include_bytes!("../tests/frag.spv")).unwrap();
        Spirv::new(&insts).unwrap();
    }

    #[test]
    fn spirv_reflect() {
        let insts =
            vulkano::shader::spirv::bytes_to_words(include_bytes!("../tests/frag.spv")).unwrap();

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

        let working_dir = root_path.join("tests");
        let include_test_path = working_dir.join("include_test.glsl");
        let include_test_source = std::fs::read_to_string(&include_test_path).unwrap();
        let (_compile_relative, relative_includes) = compile(
            &MacroInput::empty(),
            &include_test_source,
            &working_dir,
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(relative_includes),
            convert_paths(
                &root_path,
                &[
                    PathBuf::from_iter(["tests", "include_dir_a", "target_a.glsl"]),
                    PathBuf::from_iter(["tests", "include_dir_b", "target_b.glsl"]),
                ],
            ),
        );

        let (_compile_include_paths, includes) = compile_inline(
            &MacroInput {
                include_directories: vec![
                    root_path.join("tests").join("include_dir_a"),
                    root_path.join("tests").join("include_dir_b"),
                ],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <target_a.glsl>
                #include <target_b.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(includes),
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

        let (_compile_include_paths_with_relative, includes_with_relative) = compile_inline(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_a")],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <target_a.glsl>
                #include <../include_dir_b/target_b.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(includes_with_relative),
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
            .expect("cannot run tests in a folder with non unicode characters")
            .replace('\\', "/");
        let (_compile_absolute_path, includes_absolute_path) = compile_inline(
            &MacroInput::empty(),
            &format!(
                r#"
                    #version 450
                    #include "{absolute_path_str}"
                    void main() {{}}
                "#,
            ),
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(includes_absolute_path),
            convert_paths(
                &root_path,
                &[["tests", "include_dir_a", "target_a.glsl"]
                    .into_iter()
                    .collect()],
            ),
        );

        let (_compile_recursive_, includes_recursive) = compile_inline(
            &MacroInput {
                include_directories: vec![
                    root_path.join("tests").join("include_dir_b"),
                    root_path.join("tests").join("include_dir_c"),
                ],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <target_c.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(includes_recursive),
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
    fn include_paths_with_spaces() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let err = compile_inline(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_spaces")],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <foo bar>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .unwrap_err();

        assert!(err.contains("expected a file extension"));

        let (_compile_include_paths, includes) = compile_inline(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_spaces")],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <foo bar.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(includes),
            convert_paths(
                &root_path,
                &[PathBuf::from_iter([
                    "tests",
                    "include_dir_spaces",
                    "foo bar.glsl"
                ])],
            ),
        );

        let err = compile_inline(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_spaces")],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <foo.glsl bar>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .unwrap_err();

        dbg!(&err);
        assert!(err.contains("foo.glsl` to be a file existing on the file system"));

        let err = compile_inline(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_spaces")],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <foo.glsl bar.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .unwrap_err();

        assert!(err.contains("foo.glsl` to be a file existing on the file system"));
    }

    #[test]
    fn include_many_paths() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        let (_compile_include_paths, includes) = compile_inline(
            &MacroInput {
                include_directories: vec![root_path.join("tests").join("include_dir_many")],
                ..MacroInput::empty()
            },
            r#"
                #version 450
                #include <very_long_file_name_01.glsl>
                #include <very_long_file_name_02.glsl>
                #include <very_long_file_name_03.glsl>
                #include <very_long_file_name_04.glsl>
                #include <very_long_file_name_05.glsl>
                #include <very_long_file_name_06.glsl>
                #include <very_long_file_name_07.glsl>
                #include <very_long_file_name_08.glsl>
                #include <very_long_file_name_09.glsl>
                #include <very_long_file_name_10.glsl>
                #include <very_long_file_name_11.glsl>
                #include <very_long_file_name_12.glsl>
                #include <very_long_file_name_13.glsl>
                #include <very_long_file_name_14.glsl>
                #include <very_long_file_name_15.glsl>
                #include <very_long_file_name_16.glsl>
                #include <very_long_file_name_17.glsl>
                #include <very_long_file_name_18.glsl>
                #include <very_long_file_name_19.glsl>
                #include <very_long_file_name_20.glsl>
                void main() {}
            "#,
            ShaderKind::Vertex,
            &[],
        )
        .expect("cannot resolve include files");

        assert_eq!(
            HashSet::from_iter(includes),
            convert_paths(
                &root_path,
                &(1..=20)
                    .map(|i| {
                        PathBuf::from_iter([
                            "tests",
                            "include_dir_many",
                            &format!("very_long_file_name_{i:0>2}.glsl"),
                        ])
                    })
                    .collect::<Vec<_>>(),
            ),
        );
    }

    #[test]
    fn macros() {
        let need_defines = r#"
            #version 450
            #ifndef NAME1
            #error NAME1 must be defined
            #endif
            #if NAME2 <= 29
            #error NAME2 must be greater than 29
            #endif
            void main() {}
        "#;

        let compile_no_defines =
            compile_inline(&MacroInput::empty(), need_defines, ShaderKind::Vertex, &[]);
        assert!(compile_no_defines.is_err());

        compile_inline(
            &MacroInput {
                global_macro_defines: vec![
                    ("NAME1".into(), "".into()),
                    ("NAME2".into(), "58".into()),
                ],
                ..MacroInput::empty()
            },
            need_defines,
            ShaderKind::Vertex,
            &[],
        )
        .expect("setting global shader macros did not work");

        compile_inline(
            &MacroInput {
                global_macro_defines: vec![("NAME1".into(), "".into())],
                ..MacroInput::empty()
            },
            need_defines,
            ShaderKind::Vertex,
            &[("NAME2".into(), "58".into())],
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
        let insts = vulkano::shader::spirv::bytes_to_words(include_bytes!(
            "../tests/multiple_entrypoints.spv"
        ))
        .unwrap();
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
        let insts = vulkano::shader::spirv::bytes_to_words(include_bytes!(
            "../tests/multiple_entrypoints.spv"
        ))
        .unwrap();

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

    fn descriptor_calculation_with_multiple_functions_shader() -> (Vec<u32>, Vec<String>) {
        compile_inline(
            &MacroInput {
                spirv_version: Some(SpirvVersion::V1_6),
                vulkan_version: Some(EnvVersion::Vulkan1_3),
                ..MacroInput::empty()
            },
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
            &[],
        )
        .unwrap()
    }

    #[test]
    fn descriptor_calculation_with_multiple_functions() {
        let (artifact, _) = descriptor_calculation_with_multiple_functions_shader();
        let spirv = Spirv::new(&artifact).unwrap();

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
            &artifact,
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
