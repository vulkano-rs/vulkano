#version 450

// Each subpass must has their own set,
// but the bindings seems to must be 0, unlike in Sascha Willem's example.
layout(input_attachment_index = 0, set = 0,
       binding = 0) uniform subpassInput position;
layout(input_attachment_index = 1, set = 1,
       binding = 0) uniform subpassInput normal;
layout(input_attachment_index = 2, set = 2,
       binding = 0) uniform subpassInput base_color;
layout(input_attachment_index = 3, set = 3,
       binding = 0) uniform usubpassInput id; // Note the "u" here.

layout(location = 0) out vec4 out_composed;

// The light's info.
layout(push_constant, std140) uniform Constants {
    vec4 position;
    vec4 luminance;
}
constants;

// Lambert diffuse BSDF.
float lambert(vec4 n, vec4 l) { return max(dot(n, l), 0); }

void main() {
    // Use subpassLoad() for loading the pixel.
    vec4 position = subpassLoad(position);
    vec4 normal = subpassLoad(normal);
    vec4 base_color = subpassLoad(base_color);
    uint id =
        subpassLoad(id).x; // Use swizzling for extracting non-vec4 values.

    if (length(normal) < 0.1) { // Empty normal is what we used to clear the
                                // buffer. That means this pixel is part of the
                                // background, and we won't want to light that.
        out_composed = vec4(0);
    } else {

        vec4 l = normalize(constants.position - position);

        out_composed = lambert(normal, l) * base_color * constants.luminance;
    }
}