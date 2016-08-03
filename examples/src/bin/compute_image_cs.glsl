// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#version 430

layout(set = 0, binding = 0, std430) readonly buffer PackedInputBlock {
    uint packed_input[];
};

layout(set = 1, binding = 0, rgba8ui) writeonly uniform uimage2D output_image;

void main() {
    // We happen to know that our invocation IDs will only be along the X axis
    // and so we use that to know which packed index we should process
    uint index_to_unpack = gl_GlobalInvocationID.x;

    // Use the index (of the packed input) to figure out which pixel we fill
    uvec2 size = imageSize(output_image);
    uint x = index_to_unpack % size.x;
    uint y = index_to_unpack / size.x;
    ivec2 xy = ivec2(x, y);

    // This would be some neato algorithm that required compute shaders in the
    // first place - here we're just pushing image bytes around....
    uint packed = packed_input[index_to_unpack];
    uint r = packed & 0xff;
    uint g = (packed & 0xff00) >> 8;
    uint b = (packed & 0xff0000) >> 16;
    uint a = (packed & 0xff000000) >> 24;	
    uvec4 rgba = uvec4(r, g, b, a);
    
    // Store one pixel
    imageStore(output_image, xy, rgba);
}
