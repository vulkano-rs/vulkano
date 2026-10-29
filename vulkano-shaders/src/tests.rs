use crate::{
    shaders::{compile_shader, generate_shaders},
    spirv_bytes_to_words,
    structs::{generate_structs, TypeRegistry},
    MacroOptions, Result, ShaderKind, SourceLanguage,
};
use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};
use syn::{File, Item, LitStr};

fn compile_inline(
    options: &MacroOptions,
    source: &str,
    shader_kind: ShaderKind,
    source_language: Option<SourceLanguage>,
    macro_defines: &[(String, String)],
) -> Result<(Vec<u32>, Vec<String>), String> {
    compile(
        options,
        source,
        Path::new("."),
        shader_kind,
        source_language,
        macro_defines,
    )
}

fn compile(
    options: &MacroOptions,
    source: &str,
    working_dir: &Path,
    shader_kind: ShaderKind,
    source_language: Option<SourceLanguage>,
    macro_defines: &[(String, String)],
) -> Result<(Vec<u32>, Vec<String>), String> {
    let (bytes, includes) = compile_shader(
        options,
        source,
        working_dir,
        Some(shader_kind),
        source_language,
        None,
        None,
        macro_defines,
    )?;
    let words = spirv_bytes_to_words(&bytes).unwrap().into_owned();

    Ok((words, includes))
}

fn convert_paths(root_path: &Path, paths: &[PathBuf]) -> HashSet<String> {
    paths
        .iter()
        .map(|p| normalize_str(root_path.join(p)))
        .collect()
}

fn normalize_str(path: PathBuf) -> String {
    path.components()
        .collect::<PathBuf>()
        .into_os_string()
        .into_string()
        .unwrap()
}

#[test]
fn spirv_reflect() {
    let words = spirv_bytes_to_words(include_bytes!("../tests/frag.spv")).unwrap();

    let mut type_registry = TypeRegistry::default();
    let (_shader_code, _structs) = reflect(
        &MacroOptions::empty(),
        LitStr::new("../tests/frag.spv", Span::call_site()),
        None,
        &words,
        Vec::new(),
        &mut type_registry,
    )
    .expect("reflecting spv failed");

    assert_eq!(_structs.to_string(), "", "No structs should be generated");
}

fn include_resolution(source_language: SourceLanguage, shader_prefix: &str, shader_suffix: &str) {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let (_spirv, includes) = compile_inline(
        &MacroOptions {
            include_directories: vec![
                root_path.join("include_dir_a"),
                root_path.join("include_dir_b"),
            ],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <target_a.glsl>
                #include <target_b.glsl>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .expect("cannot resolve include files");

    assert_eq!(
        HashSet::from_iter(includes),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );
}

#[test]
fn include_resolution_shaderc() {
    include_resolution(SourceLanguage::Glsl, "#version 450", "void main() {}");
}

#[test]
fn include_resolution_slangc() {
    include_resolution(
        SourceLanguage::Slang,
        "",
        "float4 main() : SV_Position { return float4(0, 0, 0, 1); }",
    );
}

#[test]
fn include_resolution_dotdot_shaderc() {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let (_spirv2, includes2) = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_a")],
            ..MacroOptions::empty()
        },
        r#"
            #version 450
            #include <target_a.glsl>
            #include <../include_dir_b/target_b.glsl>
            void main() {}
        "#,
        ShaderKind::Vertex,
        Some(SourceLanguage::Glsl),
        &[],
    )
    .expect("cannot resolve include files with relative path");

    assert_eq!(
        HashSet::from_iter(includes2),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_a", "../include_dir_b/target_b.glsl"]),
            ],
        ),
    );

    let absolute_path = root_path.join("include_dir_a").join("target_a.glsl");
    let absolute_path_str = absolute_path
        .to_str()
        .expect("cannot run tests in a folder with non unicode characters")
        .replace('\\', "/");
    let (_spirv3, includes3) = compile_inline(
        &MacroOptions::empty(),
        &format!(
            "
                #version 450
                #include \"{absolute_path_str}\"
                void main() {{}}
            ",
        ),
        ShaderKind::Vertex,
        Some(SourceLanguage::Glsl),
        &[],
    )
    .expect("cannot resolve absolute include files");

    assert_eq!(
        HashSet::from_iter(includes3),
        convert_paths(
            &root_path,
            &[PathBuf::from_iter(["include_dir_a", "target_a.glsl"])],
        ),
    );

    let (_spirv4, includes4) = compile_inline(
        &MacroOptions {
            include_directories: vec![
                root_path.join("include_dir_b"),
                root_path.join("include_dir_c"),
            ],
            ..MacroOptions::empty()
        },
        r#"
            #version 450
            #include <target_c.glsl>
            void main() {}
        "#,
        ShaderKind::Vertex,
        Some(SourceLanguage::Glsl),
        &[],
    )
    .expect("cannot resolve recursive include files");

    assert_eq!(
        HashSet::from_iter(includes4),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_c", "target_c.glsl"]),
                PathBuf::from_iter(["include_dir_c", "../include_dir_a/target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );
}

#[test]
fn include_resolution_dotdot_slangc() {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let (_spirv2, includes2) = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_a")],
            ..MacroOptions::empty()
        },
        r#"
            #include <target_a.glsl>
            #include <../include_dir_b/target_b.glsl>
            float4 main() : SV_Position { return float4(0, 0, 0, 1); }
        "#,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("cannot resolve include files with relative path");

    assert_eq!(
        HashSet::from_iter(includes2),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );

    let absolute_path = root_path.join("include_dir_a").join("target_a.glsl");
    let absolute_path_str = absolute_path
        .to_str()
        .expect("cannot run tests in a folder with non unicode characters")
        .replace('\\', "/");
    let (_spirv3, includes3) = compile_inline(
        &MacroOptions::empty(),
        &format!(
            "
                #include \"{absolute_path_str}\"
                float4 main() : SV_Position {{ return float4(0, 0, 0, 1); }}
            ",
        ),
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("cannot resolve absolute include files");

    assert_eq!(
        HashSet::from_iter(includes3),
        convert_paths(
            &root_path,
            &[PathBuf::from_iter(["include_dir_a", "target_a.glsl"])],
        ),
    );

    let (_spirv4, includes4) = compile_inline(
        &MacroOptions {
            include_directories: vec![
                root_path.join("include_dir_b"),
                root_path.join("include_dir_c"),
            ],
            ..MacroOptions::empty()
        },
        r#"
            #include <target_c.glsl>
            float4 main() : SV_Position { return float4(0, 0, 0, 1); }
        "#,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("cannot resolve recursive include files");

    assert_eq!(
        HashSet::from_iter(includes4),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_c", "target_c.glsl"]),
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );
}

fn include_inline_relative(
    source_language: SourceLanguage,
    shader_prefix: &str,
    shader_suffix: &str,
) {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let (_spirv, includes) = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_a/target_a.glsl\"
                #include \"include_dir_b/target_b.glsl\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .expect("cannot resolve inline relative include files");

    assert_eq!(
        HashSet::from_iter(includes),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );
}

#[test]
fn include_inline_relative_shaderc() {
    include_inline_relative(SourceLanguage::Glsl, "#version 450", "void main() {}");
}

#[test]
fn include_inline_relative_slangc() {
    include_inline_relative(
        SourceLanguage::Slang,
        "",
        "float4 main() : SV_Position { return float4(0, 0, 0, 1); }",
    );
}

#[test]
fn include_inline_relative_dotdot_shaderc() {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let (_spirv2, includes2) = compile(
        &MacroOptions::empty(),
        r#"
            #version 450
            #include "target_a.glsl"
            #include "../include_dir_b/target_b.glsl"
            void main() {}
        "#,
        &root_path.join("include_dir_a"),
        ShaderKind::Vertex,
        Some(SourceLanguage::Glsl),
        &[],
    )
    .expect("cannot resolve inline relative include files with relative path");

    assert_eq!(
        HashSet::from_iter(includes2),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_a", "../include_dir_b/target_b.glsl"]),
            ],
        ),
    );

    let (_spirv3, includes3) = compile(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_b")],
            ..MacroOptions::empty()
        },
        r#"
            #version 450
            #include "include_dir_c/target_c.glsl"
            void main() {}
        "#,
        &root_path,
        ShaderKind::Vertex,
        Some(SourceLanguage::Glsl),
        &[],
    )
    .expect("cannot resolve recursive inline relative include files");

    assert_eq!(
        HashSet::from_iter(includes3),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_c", "target_c.glsl"]),
                PathBuf::from_iter(["include_dir_c", "../include_dir_a/target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );
}

#[test]
fn include_inline_relative_dotdot_slangc() {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let (_spirv2, includes2) = compile(
        &MacroOptions::empty(),
        r#"
            #include "target_a.glsl"
            #include "../include_dir_b/target_b.glsl"
            float4 main() : SV_Position { return float4(0, 0, 0, 1); }
        "#,
        &root_path.join("include_dir_a"),
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("cannot resolve inline relative include files with relative path");

    assert_eq!(
        HashSet::from_iter(includes2),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );

    let (_spirv3, includes3) = compile(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_b")],
            ..MacroOptions::empty()
        },
        r#"
            #include "include_dir_c/target_c.glsl"
            float4 main() : SV_Position { return float4(0, 0, 0, 1); }
        "#,
        &root_path,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("cannot resolve recursive inline relative include files");

    assert_eq!(
        HashSet::from_iter(includes3),
        convert_paths(
            &root_path,
            &[
                PathBuf::from_iter(["include_dir_c", "target_c.glsl"]),
                PathBuf::from_iter(["include_dir_a", "target_a.glsl"]),
                PathBuf::from_iter(["include_dir_b", "target_b.glsl"]),
            ],
        ),
    );
}

fn include_paths_with_spaces(
    source_language: SourceLanguage,
    shader_prefix: &str,
    shader_suffix: &str,
) {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo bar>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .unwrap_err();
    assert!(err.contains("expected a file extension"));

    let (_spirv, includes) = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo bar.glsl>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .expect("cannot resolve include file with spaces in name");

    assert_eq!(
        HashSet::from_iter(includes),
        convert_paths(
            &root_path,
            &[PathBuf::from_iter(["include_dir_spaces", "foo bar.glsl"])],
        ),
    );

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo.glsl bar>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("foo.glsl` to be a file existing on the file system"));

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo.glsl bar.glsl>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("foo.glsl` to be a file existing on the file system"));

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo bar\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("expected a file extension"));

    let (_spirv2, includes2) = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo bar.glsl\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .expect("cannot resolve include file with spaces in name");

    assert_eq!(
        HashSet::from_iter(includes2),
        convert_paths(
            &root_path,
            &[PathBuf::from_iter(["include_dir_spaces", "foo bar.glsl"])],
        ),
    );

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo.glsl bar\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("foo.glsl` to be a file existing on the file system"));

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo.glsl bar.glsl\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("foo.glsl` to be a file existing on the file system"));
}

#[test]
fn include_paths_with_spaces_shaderc() {
    include_paths_with_spaces(SourceLanguage::Glsl, "#version 450", "void main() {}");
}

#[test]
fn include_paths_with_spaces_slangc() {
    // slangc strips spaces from filenames (e.g. "foo bar.glsl" -> "foobar.glsl"),
    // so all space-containing includes fail with "include file not found".
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let shader_prefix = "";
    let shader_suffix = "float4 main() : SV_Position { return float4(0, 0, 0, 1); }";

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo bar>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("include file not found"));

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo bar.glsl>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("include file not found"));

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo.glsl bar>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("include file not found"));

    let err = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("include_dir_spaces")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                #include <foo.glsl bar.glsl>
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("include file not found"));

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo bar\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    // slangc preserves spaces in quoted include paths using make-escape (\ ) in the depfile,
    assert!(err.contains("failed to parse dependencies file"));

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo bar.glsl\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("failed to parse dependencies file"));

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo.glsl bar\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("failed to parse dependencies file"));

    let err = compile(
        &MacroOptions::empty(),
        &format!(
            "
                {shader_prefix}
                #include \"include_dir_spaces/foo.glsl bar.glsl\"
                {shader_suffix}
            ",
        ),
        &root_path,
        ShaderKind::Vertex,
        Some(SourceLanguage::Slang),
        &[],
    )
    .unwrap_err();

    assert!(err.contains("failed to parse dependencies file"));
}

fn include_many_paths(source_language: SourceLanguage, shader_prefix: &str, shader_suffix: &str) {
    let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let many_includes = (1..=20)
        .map(|i| format!("#include <very_long_file_name_{i:0>2}.glsl>"))
        .collect::<Vec<_>>()
        .join("\n");

    let (_spirv, includes) = compile_inline(
        &MacroOptions {
            include_directories: vec![root_path.join("tests").join("include_dir_many")],
            ..MacroOptions::empty()
        },
        &format!(
            "
                {shader_prefix}
                {many_includes}
                {shader_suffix}
            ",
        ),
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .expect("cannot resolve many include files");

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
fn include_many_paths_shaderc() {
    include_many_paths(SourceLanguage::Glsl, "#version 450", "void main() {}");
}

#[test]
fn include_many_paths_slangc() {
    include_many_paths(
        SourceLanguage::Slang,
        "",
        "float4 main() : SV_Position { return float4(0, 0, 0, 1); }",
    );
}

fn macros(source_language: SourceLanguage, shader_prefix: &str, shader_suffix: &str) {
    let need_defines = format!(
        "
            {shader_prefix}
            #ifndef NAME1
            #error NAME1 must be defined
            #endif
            #if NAME2 <= 29
            #error NAME2 must be greater than 29
            #endif
            {shader_suffix}
        ",
    );

    let compile_no_defines = compile_inline(
        &MacroOptions::empty(),
        &need_defines,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    );
    assert!(compile_no_defines.is_err());

    compile_inline(
        &MacroOptions {
            global_macro_defines: vec![("NAME1".into(), "".into()), ("NAME2".into(), "58".into())],
            ..MacroOptions::empty()
        },
        &need_defines,
        ShaderKind::Vertex,
        Some(source_language),
        &[],
    )
    .expect("setting global shader macros did not work");

    compile_inline(
        &MacroOptions {
            global_macro_defines: vec![("NAME1".into(), "".into())],
            ..MacroOptions::empty()
        },
        &need_defines,
        ShaderKind::Vertex,
        Some(source_language),
        &[("NAME2".into(), "58".into())],
    )
    .expect("setting shader macros did not work");
}

#[test]
fn macros_shaderc() {
    macros(SourceLanguage::Glsl, "#version 450", "void main() {}");
}

#[test]
fn macros_slangc() {
    macros(
        SourceLanguage::Slang,
        "",
        "float4 main() : SV_Position { return float4(0, 0, 0, 1); }",
    );
}

#[test]
fn reflect_slangc_simple_compute() {
    let (words, _includes) = compile_inline(
        &MacroOptions::empty(),
        r#"
            [vk::binding(0, 0)] RWStructuredBuffer<float> output;
            [vk::binding(1, 0)] StructuredBuffer<uint> buffer_1;
            [vk::binding(2, 0)] StructuredBuffer<float> buffer_2;
            [numthreads(1, 1, 1)]
            void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
                output[dispatchThreadID.x] = (float)buffer_1[dispatchThreadID.x]
                    + buffer_2[dispatchThreadID.x];
            }
        "#,
        ShaderKind::Compute,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("slangc failed to compile simple compute shader");

    let mut type_registry = TypeRegistry::default();
    let (_shader_code, _structs) = reflect(
        &MacroOptions::empty(),
        LitStr::new("slangc_simple_compute", Span::call_site()),
        None,
        &words,
        Vec::new(),
        &mut type_registry,
    )
    .expect("reflecting spv failed");
}

#[test]
fn reflect_slangc_multiple_structured_buffers() {
    let (words, _includes) = compile_inline(
        &MacroOptions::empty(),
        r#"
            [vk::binding(0, 0)] RWStructuredBuffer<float> output_1;
            [vk::binding(1, 0)] RWStructuredBuffer<uint> output_2;
            [vk::binding(2, 0)] StructuredBuffer<float> buffer_1;
            [vk::binding(3, 0)] StructuredBuffer<uint> buffer_2;
            [numthreads(1, 1, 1)]
            void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
                output_1[dispatchThreadID.x] = (float)buffer_1[dispatchThreadID.x]
                    + buffer_2[dispatchThreadID.x];
                output_2[dispatchThreadID.x] = buffer_2[dispatchThreadID.x];
            }
        "#,
        ShaderKind::Compute,
        Some(SourceLanguage::Slang),
        &[],
    )
    .expect("slangc failed to compile shader");

    let mut type_registry = TypeRegistry::default();
    let (_shader_code, _structs) = reflect(
        &MacroOptions::empty(),
        LitStr::new("slangc_multiple_structured_buffers", Span::call_site()),
        None,
        &words,
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

    let buffer_1 = structs
        .iter()
        .find(|s| s.ident == "StructuredBuffer_f32")
        .unwrap();
    assert_eq!(
        buffer_1.fields.to_token_stream().to_string(),
        quote!({pub __member0: [f32],}).to_string()
    );

    let buffer_2 = structs
        .iter()
        .find(|s| s.ident == "StructuredBuffer_u32")
        .unwrap();
    assert_eq!(
        buffer_2.fields.to_token_stream().to_string(),
        quote!({pub __member0: [u32],}).to_string()
    );

    let output_1 = structs
        .iter()
        .find(|s| s.ident == "RWStructuredBuffer_f32")
        .unwrap();
    assert_eq!(
        output_1.fields.to_token_stream().to_string(),
        quote!({pub __member0: [f32],}).to_string()
    );

    let output_2 = structs
        .iter()
        .find(|s| s.ident == "RWStructuredBuffer_u32")
        .unwrap();
    assert_eq!(
        output_2.fields.to_token_stream().to_string(),
        quote!({pub __member0: [u32],}).to_string()
    );
}

#[test]
fn reflect_rust_gpu_vertex() {
    let words =
        spirv_bytes_to_words(include_bytes!("../tests/rust-gpu/test_shader-vertex.spv")).unwrap();

    let mut type_registry = TypeRegistry::default();
    let (_shader_code, _structs) = reflect(
        &MacroOptions::empty(),
        LitStr::new("rust_gpu_vertex", Span::call_site()),
        None,
        &words,
        Vec::new(),
        &mut type_registry,
    )
    .expect("reflecting spv failed");
}

#[test]
fn reflect_rust_gpu_fragment() {
    let words =
        spirv_bytes_to_words(include_bytes!("../tests/rust-gpu/test_shader-fragment.spv")).unwrap();

    let mut type_registry = TypeRegistry::default();
    let (_shader_code, _structs) = reflect(
        &MacroOptions::empty(),
        LitStr::new("rust_gpu_fragment", Span::call_site()),
        None,
        &words,
        Vec::new(),
        &mut type_registry,
    )
    .expect("reflecting spv failed");
}

fn reflect(
    options: &MacroOptions,
    source: LitStr,
    name: Option<String>,
    words: &[u32],
    input_paths: Vec<String>,
    type_registry: &mut TypeRegistry,
) -> Result<(TokenStream, TokenStream)> {
    let shaders_code = generate_shaders(name.as_deref(), words, input_paths)?;
    let structs_code = generate_structs(options, source, name, words, type_registry)?;

    Ok((shaders_code, structs_code))
}
