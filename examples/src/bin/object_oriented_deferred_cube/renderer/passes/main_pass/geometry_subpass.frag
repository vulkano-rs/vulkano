#version 450
#include "geometry_subpass.incl.comp"

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_base_color;
layout(location = 3) out uint out_id;

layout(set = 0, binding = 0) uniform sampler2DArray texture_array;

void main() {
    out_position = in_position;

    out_normal =
        normalize(in_normal); // Dont't forget that as-is normals will become
                              // unnormalized after linear interpolation!

    out_base_color = texture(
        texture_array,
        vec3(in_uv, constants.base_color)); // Note how the layer index is NOT
                                            // normalized, unlike UV.

    out_id = constants.id;
}
