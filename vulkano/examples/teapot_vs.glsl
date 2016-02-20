#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec2 position;

uniform Data {
    mat4 worldview;
} uniforms;

void main() {
    gl_Position = uniforms.worldview * vec4(position, 0.0, 1.0);
}
