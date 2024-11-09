#version 450
#include <shared_exponent.glsl>

const uint MAX_BLOOM_MIP_LEVELS = 6;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler bloom_sampler;
layout(set = 0, binding = 1) uniform texture2D bloom_texture;
layout(set = 0, binding = 2, r32ui) uniform uimage2D bloom_mip_chain[MAX_BLOOM_MIP_LEVELS];

layout(push_constant) uniform PushConstants {
    uint dst_mip_level;
    float intensity;
};

uint src_mip_level = dst_mip_level + 1;

vec3 sample1(vec2 uv) {
    return textureLod(sampler2D(bloom_texture, bloom_sampler), uv, src_mip_level).rgb;
}

// 9-tap tent filter.
vec3 upsampleTent9(vec2 uv, vec2 src_texel_size) {
    vec3 color;
    color  = sample1(uv + vec2(-1.0, -1.0) * src_texel_size) * 1.0;
    color += sample1(uv + vec2( 0.0, -1.0) * src_texel_size) * 2.0;
    color += sample1(uv + vec2( 1.0, -1.0) * src_texel_size) * 1.0;
    color += sample1(uv + vec2(-1.0,  0.0) * src_texel_size) * 2.0;
    color += sample1(uv + vec2( 0.0,  0.0) * src_texel_size) * 4.0;
    color += sample1(uv + vec2( 1.0,  0.0) * src_texel_size) * 2.0;
    color += sample1(uv + vec2(-1.0,  1.0) * src_texel_size) * 1.0;
    color += sample1(uv + vec2( 0.0,  1.0) * src_texel_size) * 2.0;
    color += sample1(uv + vec2( 1.0,  1.0) * src_texel_size) * 1.0;

    return color * (1.0 / 16.0);
}

void blend(vec2 uv, ivec2 dst_coord, vec3 color) {
    color += textureLod(sampler2D(bloom_texture, bloom_sampler), uv, dst_mip_level).rgb;
    uint packed = convertToSharedExponent(color);
    imageStore(bloom_mip_chain[dst_mip_level], dst_coord, uvec4(packed, 0, 0, 0));
}

void main() {
    ivec2 dst_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dst_size = imageSize(bloom_mip_chain[dst_mip_level]);

    if (dst_coord.x > dst_size.x || dst_coord.y > dst_size.y) {
        return;
    }

    ivec2 src_size = textureSize(sampler2D(bloom_texture, bloom_sampler), int(src_mip_level));
    vec2 src_texel_size = 1.0 / vec2(src_size);
    vec2 uv = (vec2(dst_coord) + 0.5) / vec2(dst_size);
    vec3 color = upsampleTent9(uv, src_texel_size);

    color *= intensity;

    blend(uv, dst_coord, color);
}
