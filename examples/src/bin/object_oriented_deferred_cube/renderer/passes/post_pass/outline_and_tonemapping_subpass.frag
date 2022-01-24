#version 450

layout(location = 0) in vec2 coords;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D position;
layout(set = 0, binding = 1) uniform sampler2D normal;
layout(set = 0, binding = 2) uniform sampler2D base_color;
layout(set = 0, binding = 3) uniform usampler2D id;
layout(set = 0, binding = 4) uniform sampler2D composite;

// A simplified fit of ACES filmic tone mapping
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 aces_filmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1);
}

void main() {

    vec4 self_position = texture(position, coords);
    vec4 self_normal = texture(normal, coords);
    vec4 self_base_color = texture(base_color, coords);
    uint self_id = texture(id, coords).x;
    vec4 self_composite = texture(composite, coords);

    // Search for neighboring ID buffer texels (!)
    // in a square with a side length of (KERNEL_RADIUS * 2 + 1):
    // if any neighbor has a different ID than this pixel,
    // we'll declare this pixel is part of an outline.
    // Note it's texels, not pixels (as in 1:1 mapped the Window's pixels);
    // if you are using different resolutions for the buffers,
    // you'll have to calculate different coordinates for them.
    const int KERNEL_RADIUS = 1;
    vec2 size = textureSize(id, 0); // Get the ID buffer's size.
    vec2 unit = 1 / size;           // Normalized size of a ID buffer texel.
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            vec2 neighbor_coords = coords + unit * vec2(i, j);
            uint neighbor_id = texture(id, neighbor_coords).x;
            if (neighbor_id != self_id) {
                out_color = vec4(0, 0, 0, 1); // Hard coded outline color.
                return;
            }
        }
    }

    // Or else, we'll return the current pixel's actual color, tone-mapped.
    if (self_id == 0) { // Background, you might replace it with a skybox here
        out_color = vec4(0.39, 0.58, 0.93, 1);
    } else {
        out_color = vec4(aces_filmic(self_composite.xyz), 1);
    }
}