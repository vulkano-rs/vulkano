extern crate vulkano_shaders;

fn main() {
    // building the shaders used in the examples
    vulkano_shaders::build_glsl_shaders([
        ("src/bin/compute_buffer.glsl", vulkano_shaders::ShaderType::Compute),
        ("src/bin/compute_image.glsl", vulkano_shaders::ShaderType::Compute),
        ("src/bin/triangle_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("src/bin/triangle_fs.glsl", vulkano_shaders::ShaderType::Fragment),
        ("src/bin/teapot_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("src/bin/teapot_fs.glsl", vulkano_shaders::ShaderType::Fragment),
        ("src/bin/image_vs.glsl", vulkano_shaders::ShaderType::Vertex),
        ("src/bin/image_fs.glsl", vulkano_shaders::ShaderType::Fragment),
    ].iter().cloned());
}
