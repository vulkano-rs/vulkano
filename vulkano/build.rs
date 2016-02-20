extern crate glsl_to_spirv;
extern crate vulkano_shaders;
extern crate vk_sys;

use std::env;
use std::fs::File;
use std::path::Path;
use std::io::Write;

fn main() {
    // tell Cargo that this build script never needs to be rerun
    println!("cargo:rerun-if-changed=build.rs");

    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);

    let mut file_output = File::create(&dest.join("vk_bindings.rs")).unwrap();
    vk_sys::write_bindings(&mut file_output).unwrap();

    write_examples();
}

fn write_examples() {
    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);

    let mut file_output = File::create(&dest.join("examples-triangle_vs.rs")).unwrap();
    println!("cargo:rerun-if-changed=examples/triangle_vs.glsl");
    let content = glsl_to_spirv::compile(include_str!("examples/triangle_vs.glsl"), glsl_to_spirv::ShaderType::Vertex).unwrap();
    let output = vulkano_shaders::reflect("TriangleShader", content).unwrap();
    write!(file_output, "{}", output).unwrap();

    let mut file_output = File::create(&dest.join("examples-triangle_fs.rs")).unwrap();
    println!("cargo:rerun-if-changed=examples/triangle_fs.glsl");
    let content = glsl_to_spirv::compile(include_str!("examples/triangle_fs.glsl"), glsl_to_spirv::ShaderType::Fragment).unwrap();
    let output = vulkano_shaders::reflect("TriangleShader", content).unwrap();
    write!(file_output, "{}", output).unwrap();

    let mut file_output = File::create(&dest.join("examples-teapot_vs.rs")).unwrap();
    println!("cargo:rerun-if-changed=examples/teapot_vs.glsl");
    let content = glsl_to_spirv::compile(include_str!("examples/teapot_vs.glsl"), glsl_to_spirv::ShaderType::Vertex).unwrap();
    let output = vulkano_shaders::reflect("TeapotShader", content).unwrap();
    write!(file_output, "{}", output).unwrap();

    let mut file_output = File::create(&dest.join("examples-teapot_fs.rs")).unwrap();
    println!("cargo:rerun-if-changed=examples/teapot_fs.glsl");
    let content = glsl_to_spirv::compile(include_str!("examples/teapot_fs.glsl"), glsl_to_spirv::ShaderType::Fragment).unwrap();
    let output = vulkano_shaders::reflect("TeapotShader", content).unwrap();
    write!(file_output, "{}", output).unwrap();
}
