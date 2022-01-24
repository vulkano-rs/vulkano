#version 450 // I don't know why, but everybody seems to be using 450
             // But indeed 460 didn't seem to have added any new features
#include "geometry_subpass.incl.comp" // <-- Push constants are defined in this file!

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// For convenience (and probably performance?)
// we'll just pass homogeneous vectors between shaders.
layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec2 out_uv;

void main() {
    out_position =
        constants.model *
        vec4(
            position,
            1); // When transforming a 3D *point* by 4x4 matrix,
                // you need to set the w to 1:
                // https://en.wikipedia.org/wiki/Homogeneous_coordinates#Use_in_computer_graphics_and_computer_vision

    gl_Position = constants.projection * constants.view * out_position;

    out_normal =
        constants.model * vec4(normal, 0); // For normals though, the w is 0.

    out_uv = vec2(uv.x, 1 - uv.y); // Vulkan origin is top-left
                                   // while blender uv origin is bot-left
}