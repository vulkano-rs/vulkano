// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#version 430

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_image_size : enable
#extension GL_ARB_shader_storage_buffer_object : enable
#extension GL_ARB_shading_language_420pack : enable

layout(binding = 0, std430) writeonly buffer PackedOutput {
    uint packed_output[];
};
layout(binding = 1, rgba8ui) uniform readonly uimage2D input_image;

void main() {
    // We happen to know that our invocation IDs will only be along the X axis
    // and so we use that to know which packed index we should process
    uint index_to_pack = gl_GlobalInvocationID.x;

    // Use the index (of the packed output) to figure out which pixel we grab
    uvec2 size = imageSize(input_image);
    uint x = index_to_pack % size.x;
    uint y = index_to_pack / size.x;
    ivec2 xy = ivec2(x, y);

    // Load one pixel
    uvec4 rgba = imageLoad(input_image, xy);

    // Do some sort of logic to pack the pixel
    uint r = rgba.r;
    uint g = rgba.g << 2;
    uint b = rgba.b << 4;
    uint a = rgba.a << 8;
    uint packed = r | g | b | a;

    // Put the packed data in the output array
    packed_output[index_to_pack] = packed;
}
