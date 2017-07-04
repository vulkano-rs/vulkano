# [Vulkano](http://vulkano.rs)

See also [vulkano.rs](http://vulkano.rs).

Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.khronos.org/vulkan/).
It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't
be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code
should always conform to valid API usage.

What does vulkano do?

- Provides a low-levelish API around Vulkan. It doesn't hide what it does, but provides some
  comfort types.
- Plans to prevent all invalid API usages, even the most obscure ones. The purpose of vulkano
  is not to simply let you draw a teapot, but to cover all possible usages of Vulkan and detect all
  the possible problems in order to write robust programs. Invalid API usage is prevented thanks to
  both compile-time checks and runtime checks.
- Can handle synchronization on the GPU side for you (unless you choose do that yourself), as this
  aspect of Vulkan is both annoying to handle and error-prone. Dependencies between submissions are
  automatically detected, and semaphores are managed automatically. The behavior of the library can
  be customized thanks to unsafe trait implementations.
- Tries to be convenient to use. Nobody is going to use a library that requires you to browse
  the documentation for hours for every single operation.

Note that vulkano does **not** require you to install the official Vulkan SDK. This is not
something specific to vulkano (you don't need to SDK to write program that use Vulkan, even
without vulkano), but many people are unaware of that and install the SDK thinking that it is
required.

## Development status

Vulkano is still in heavy development and doesn't yet meet its goals of being very robust. However
the general structure of the library is most likely definitive, and all future breaking changes
will likely be straight-forward to fix in user code.

## [Documentation](https://docs.rs/vulkano)

[![](https://docs.rs/vulkano/badge.svg)](https://docs.rs/vulkano)

To get started you are encouraged to read the examples in `examples/src/bin`, starting with
the `triangle` example.

## Donate

[![Become a patron](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/tomaka)

## Structure

This repository contains six libraries:

- `vulkano` is the main one.
- `vulkano-shaders` can analyse SPIR-V shaders at compile-time.
- `vulkano-shader-derive` provides a custom derive that invokes `vulkano-shaders`. It lets you
  easily integrate your GLSL shaders within the rest of your source code.
- `vulkano-win` provides a safe link between vulkano and the `winit` library which can create
  a window where to render to.
- `glsl-to-spirv` can compile GLSL to SPIR-V by wrapping around `glslang`. `glsl-to-spirv` is an
  implementation detail that you don't need to use manually if you use vulkano.
- `vk-sys` contains raw bindings for Vulkan. You can use it even if you don't care about vulkano.

Once procedural macros are stabilized in Rust, the `vulkano-shaders` and `vulkano-shader-derive`
crates will be merged with the `vulkano` crate. The `glsl-to-spirv` crate is an implementation
detail of vulkano and is not supposed to be used directly if you use vulkano. You are, however,
free to use it if you want to write an alternative to vulkano.

In order to run tests, run `cargo test --all` at the root of the repository. Make sure your Vulkan
driver is up to date before doing so.

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
