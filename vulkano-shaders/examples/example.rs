extern crate glsl_to_spirv;
extern crate vulkano_shaders;

fn main() {
    let shader = r#"
#version 450

struct S {
    vec3 val1;
    bool val2[5];
};

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(set = 0, binding = 1) uniform Block {
    S u_data;
} block;

in vec2 v_texcoords;
out vec4 f_color;

void main() {
    if (block.u_data.val2[3]) {
        f_color = texture(u_texture, v_texcoords);
    } else {
        f_color = vec4(1.0);
    }
}

"#;

    let content = glsl_to_spirv::compile(shader, glsl_to_spirv::ShaderType::Fragment).unwrap();
    let output = vulkano_shaders::reflect("Shader", content).unwrap();
    println!("{}", output);
}
