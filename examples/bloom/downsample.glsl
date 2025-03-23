#version 450
#include <vulkano.glsl>
#include <shared_exponent.glsl>

const float EPSILON = 1.19209290e-07;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    SamplerId sampler_id;
    SampledImageId texture_id;
    StorageImageId dst_mip_image_id;
    uint dst_mip_level;
    float threshold;
    float knee;
};

uint src_mip_level = dst_mip_level - 1;

vec3 quadraticThreshold(vec3 color) {
    float brightness = max(color.r, max(color.g, color.b));
    float quadratic_response = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    quadratic_response = (quadratic_response * quadratic_response) / (0.25 / knee);

    color *= max(quadratic_response, brightness - threshold) / max(brightness, EPSILON);

    return color;
}

vec3 prefilter(vec3 color) {
    return quadraticThreshold(color);
}

vec3 sample1(vec2 uv) {
    return textureLod(vko_sampler2D(texture_id, sampler_id), uv, src_mip_level).rgb;
}

// 13-tap box filter.
// ┌───┬───┬───┐
// │ A │ B │ C │
// ├──╴D╶─╴E╶──┤
// │ F │ G │ H │
// ├──╴I╶─╴J╶──┤
// │ K │ L │ M │
// └───┴───┴───┘
vec3 downsampleBox13(vec2 uv, vec2 src_texel_size) {
    vec3 a = sample1(uv + vec2(-2.0, -2.0) * src_texel_size);
    vec3 b = sample1(uv + vec2( 0.0, -2.0) * src_texel_size);
    vec3 c = sample1(uv + vec2( 2.0, -2.0) * src_texel_size);
    vec3 d = sample1(uv + vec2(-1.0, -1.0) * src_texel_size);
    vec3 e = sample1(uv + vec2( 1.0, -1.0) * src_texel_size);
    vec3 f = sample1(uv + vec2(-2.0,  0.0) * src_texel_size);
    vec3 g = sample1(uv + vec2( 0.0,  0.0) * src_texel_size);
    vec3 h = sample1(uv + vec2( 2.0,  0.0) * src_texel_size);
    vec3 i = sample1(uv + vec2(-1.0,  1.0) * src_texel_size);
    vec3 j = sample1(uv + vec2( 1.0,  1.0) * src_texel_size);
    vec3 k = sample1(uv + vec2(-2.0,  2.0) * src_texel_size);
    vec3 l = sample1(uv + vec2( 0.0,  2.0) * src_texel_size);
    vec3 m = sample1(uv + vec2( 2.0,  2.0) * src_texel_size);

    vec3 color;
    color  = (d + e + i + j) * 0.25 * 0.5;
    color += (a + b + f + g) * 0.25 * 0.125;
    color += (b + c + g + h) * 0.25 * 0.125;
    color += (f + g + k + l) * 0.25 * 0.125;
    color += (g + h + l + m) * 0.25 * 0.125;

    return color;
}

void store(ivec2 dst_coord, vec3 color) {
    uint packed = convertToSharedExponent(color);
    imageStore(vko_uimage2D_r32ui(dst_mip_image_id), dst_coord, uvec4(packed, 0, 0, 0));
}

void main() {
    ivec2 dst_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dst_size = imageSize(vko_uimage2D_r32ui(dst_mip_image_id));

    if (dst_coord.x > dst_size.x || dst_coord.y > dst_size.y) {
        return;
    }

    ivec2 src_size = textureSize(vko_texture2D(texture_id), int(src_mip_level));
    vec2 src_texel_size = 1.0 / vec2(src_size);
    vec2 uv = (vec2(dst_coord) + 0.5) / vec2(dst_size);
    vec3 color = downsampleBox13(uv, src_texel_size);

    if (src_mip_level == 0) {
        color = prefilter(color);
    }

    color = max(color, 0.0001);

    store(dst_coord, color);
}
