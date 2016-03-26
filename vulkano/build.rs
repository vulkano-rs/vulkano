extern crate vulkano_shaders;
extern crate vk_sys;

use std::env;
use std::fs::File;
use std::path::Path;

fn main() {
    // tell Cargo that this build script never needs to be rerun
    println!("cargo:rerun-if-changed=build.rs");

    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);

    let mut file_output = File::create(&dest.join("vk_bindings.rs")).unwrap();
    vk_sys::write_bindings(&mut file_output).unwrap();

    // building the shaders used in the examples
    vulkano_shaders::build_glsl_shaders([
        ("examples/triangle_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("examples/triangle_fs.glsl", vulkano_shaders::ShaderType::Fragment),
        ("examples/teapot_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("examples/teapot_fs.glsl", vulkano_shaders::ShaderType::Fragment),
        ("examples/image_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("examples/image_fs.glsl", vulkano_shaders::ShaderType::Fragment),
    ].iter().cloned());
}
