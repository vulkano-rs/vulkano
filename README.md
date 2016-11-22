# Vulkano

**Note: requires Rust 1.9**. This library would highly benefit from multiple upcoming features in
Rust. Therefore it is likely that in the future you will need to update your version of Rust to
continue using vulkano.

Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.khronos.org/vulkan/).
It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't
be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code
should always conform to valid API usage.

What does vulkano do?

- Provides a low-levelish API around Vulkan. It doesn't hide what it does, but provides some
  comfort types.
- Plans to prevents all invalid API usages, even the most obscure ones. The purpose of vulkano
  is not to draw a teapot, but to cover all possible usages of Vulkan and detect all the
  possible problems. Invalid API usage is prevented thanks to both compile-time checks and
  runtime checks.
- Handles synchronization on the GPU side for you, as this aspect of Vulkan is both annoying
  to handle and error-prone. Dependencies between submissions are automatically detected, and
  semaphores are managed automatically. The behavior of the library can be customized thanks
  to unsafe trait implementations.
- Tries to be convenient to use. Nobody is going to use a library that requires you to browse
  the documentation for hours for every single operation.

**Warning: this library breaks every five minutes for the moment.**

## [Documentation](https://docs.rs/vulkano)

[![](https://docs.rs/vulkano/badge.svg)](https://docs.rs/vulkano)

To get started you are encouraged to read the examples in `examples/src/bin`, starting with
the `triangle` example.

## Structure

This repository contains four libraries:

- `vulkano` is the main one.
- `vulkano-shaders` can analyse SPIR-V shaders at compile-time.
- `vulkano-win` provides a safe link between vulkano and the `winit` library which can create
  a window where to render to.
- `glsl-to-spirv` can compile GLSL to SPIR-V by wrapping around `glslang`. `glsl-to-spirv` is an
  implementation detail that you don't need to use manually if you use vulkano.

Once procedural macros are stabilized in Rust, the `vulkano-shaders` crate will be merged with the
`vulkano` crate. The `glsl-to-spirv` crate is an implementation detail of vulkano and is not
supposed to be used directly if you use vulkano. You are, however, free to use it if you want to
write an alternative to vulkano.

This crate uses the Cargo workspaces feature that is available only in nightly, and whose purpose
is to make several crates share the same `target/` directory. It is normal to get an error if you
try to run `cargo build` at the root of the directory.

In order to run tests, go to the `vulkano` subdirectory and run `cargo test`. On nVidia GPUs, you
will have to set the `RUST_TEST_THREADS` environment variable to `1` because of
[a bug](https://devtalk.nvidia.com/default/topic/938723/creating-destroying-several-vkdevices-concurrently-sometimes-crashes-or-deadlocks/). 

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
