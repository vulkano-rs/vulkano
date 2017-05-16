# Usage

This replaces `vulkano-shaders`.

```rust
#[macro_use]
extern crate vulkano_shader_derive;

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_450pack : enable

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}"]
    struct Dummy;
}

let fs = fs::Shader::load(&device).expect("failed to create shader module");
```
