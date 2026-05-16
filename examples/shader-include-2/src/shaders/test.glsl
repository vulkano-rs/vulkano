#version 450

// Substitutes this line with the contents of the file `common.glsl` found in
// one of the standard `include` directories specified above.
// Note that relative inclusion (`#include "..."`), although it falls back to
// standard inclusion, should not be used for **embedded** shader source, as it
// may be misleading and/or confusing.
#include "standard-shaders/common.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
   uint data[];
};

void main() {
   uint idx = gl_GlobalInvocationID.x;
   data[idx] = multiply_by_12(data[idx]);
}
