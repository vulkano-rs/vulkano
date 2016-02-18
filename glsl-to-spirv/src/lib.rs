extern crate tempdir;

use std::fs::File;
use std::io::Write;
use std::process::Command;

pub type SpirvOutput = File;

pub fn compile(code: &str, ty: ShaderType) -> Result<SpirvOutput, String> {
    compile_inner(Some((code, ty)))
}

// Eventually the API will look like this, with an iterator for multiple shader stages.
// However for the moment GLSLang doesn't like that, so we only pass one shader at a time.
fn compile_inner<'a, I>(shaders: I) -> Result<SpirvOutput, String>
    where I: IntoIterator<Item = (&'a str, ShaderType)>
{
    let temp_dir = tempdir::TempDir::new("glslang-compile").unwrap();
    let output_file = temp_dir.path().join("compilation_output.spv");

    let mut command = Command::new(concat!(env!("OUT_DIR"), "/glslang_validator"));
    command.arg("-V");
    command.arg("-l");
    command.arg("-o").arg(&output_file);

    for (num, (source, ty)) in shaders.into_iter().enumerate() {
        let extension = match ty {
            ShaderType::Vertex => ".vert",
            ShaderType::Fragment => ".frag",
            ShaderType::Geometry => ".geom",
            ShaderType::TessellationControl => ".tesc",
            ShaderType::TessellationEvaluation => ".tese",
            ShaderType::Compute => ".comp",
        };

        let file_path = temp_dir.path().join(format!("{}{}", num, extension));
        File::create(&file_path).unwrap().write_all(source.as_bytes()).unwrap();
        command.arg(file_path);
    }

    let output = command.output().expect("Failed to execute glslangValidator");

    if output.status.success() {
        let spirv_output = File::open(output_file).expect("failed to open SPIR-V output file");
        return Ok(spirv_output);
    }

    let error1 = String::from_utf8(output.stdout).expect("output of glsl compiler is not UTF-8");
    let error2 = String::from_utf8(output.stderr).expect("output of glsl compiler is not UTF-8");
    return Err(error1 + &error2);
}

/// Type of shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Geometry,
    TessellationControl,
    TessellationEvaluation,
    Compute,
}
