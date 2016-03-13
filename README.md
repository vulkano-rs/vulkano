# Vulkano

Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.khronos.org/vulkan/).
It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't
be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code
should always conform to valid API usage.

Vulkano is *not* a high level library. It provides only the minimal features required to make
using the Vulkan API safe.

*Warning: this library breaks every five minutes for the moment.*

## [Documentation](http://tomaka.github.io/vulkano/vulkano/index.html)

## [Differences between Vulkan and vulkano](FEATURES.md)

This repository contains three libraries:

- `vulkano` is the main one.
- `vulkano-shaders` can analyse SPIR-V shaders at compile-time.
- `glsl-to-spirv` can compile GLSL to SPIR-V.
