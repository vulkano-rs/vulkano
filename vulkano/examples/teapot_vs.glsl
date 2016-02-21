#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 worldview;
    mat4 proj;
} uniforms;

void main() {
    v_normal = transpose(inverse(mat3(uniforms.worldview))) * normal;
    gl_Position = uniforms.proj * uniforms.worldview * vec4(position, 1.0);
}
