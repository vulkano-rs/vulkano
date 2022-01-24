#version 450

// We don't really need any input or output.
// Fetching a fragment in a subpass doesn't require coordinates.

void main() 
{
    // Generate positions from gl_VertexIndex.
    // https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0f + -1.0f, 0.0f, 1.0f);
}