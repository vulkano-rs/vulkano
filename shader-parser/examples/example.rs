extern crate glsl_to_spirv;
extern crate shader_parser;

fn main() {
    let shader = r#"
#version 450

uniform vec4 u_test;

vec4 f_color;

void main() {
    f_color = u_test;
}

"#;

    let content = glsl_to_spirv::compile(Some((shader, glsl_to_spirv::ShaderType::Fragment))).unwrap();
    let output = shader_parser::reflect(content).unwrap();
    println!("{}", output);
}
