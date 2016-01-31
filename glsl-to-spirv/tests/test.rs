extern crate glsl_to_spirv;

#[test]
fn test1() {
    let shader = r#"
#version 330

out vec4 f_color;

void main() {
    f_color = vec4(1.0);
}
"#;

    glsl_to_spirv::compile(Some((shader, glsl_to_spirv::ShaderType::Fragment))).unwrap();
}

#[test]
fn test2() {
    let vertex = r#"
#version 330

in vec2 i_position;
out vec2 v_texcoords;

void main() {
    v_texcoords = i_position;
    gl_Position = vec4(i_position, 0.0, 1.0);
}
"#;

    let fragment = r#"
#version 330

out vec4 f_color;

void main() {
    f_color = vec4(1.0);
}
"#;

    glsl_to_spirv::compile([
        (vertex, glsl_to_spirv::ShaderType::Vertex),
        (fragment, glsl_to_spirv::ShaderType::Fragment)
    ].iter().cloned()).unwrap();
}
