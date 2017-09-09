// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate glsl_to_spirv;
extern crate vulkano_shaders;

fn main() {
    let shader = r#"
#version 450

layout(constant_id = 5) const int index = 2;

struct S {
    vec3 val1;
    bool val2[5];
};

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(set = 0, binding = 1) uniform Block {
    S u_data;
} block;

layout(location = 0) in vec2 v_texcoords;
layout(location = 0) out vec4 f_color;

void main() {
    if (block.u_data.val2[index]) {
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
