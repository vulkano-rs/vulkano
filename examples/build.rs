extern crate vulkano_shaders;

fn main() {
    println!("cargo:rustc-link-search=framework={}", "/Library/Frameworks");
    // building the shaders used in the examples
    vulkano_shaders::build_glsl_shaders([
        ("src/bin/triangle_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("src/bin/triangle_fs.glsl", vulkano_shaders::ShaderType::Fragment),
        ("src/bin/teapot_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("src/bin/teapot_fs.glsl", vulkano_shaders::ShaderType::Fragment),
        ("src/bin/image_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("src/bin/image_fs.glsl", vulkano_shaders::ShaderType::Fragment),
    ].iter().cloned());
}
