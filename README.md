# Vulkano

**Note: requires Rust 1.8**

Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.khronos.org/vulkan/).
It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't
be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code
should always conform to valid API usage.

Vulkano is *not* a high level library. It provides only the minimal features required to make
using the Vulkan API safe.

**Warning: this library breaks every five minutes for the moment.**

## [Documentation](http://tomaka.github.io/vulkano/vulkano/index.html)

## [Differences between Vulkan and vulkano](FEATURES.md)

This repository contains three libraries:

- `vulkano` is the main one.
- `vulkano-shaders` can analyse SPIR-V shaders at compile-time.
- `glsl-to-spirv` can compile GLSL to SPIR-V.

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
