#version 450

// We will need the coordinates this time, because we're using sampled images.
// Although you can just 
layout (location = 0) out vec2 coords;

void main() 
{
    // https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    coords = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(coords * 2.0f + -1.0f, 0.0f, 1.0f);
}