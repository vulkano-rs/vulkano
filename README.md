<img align="left" alt="" src="logo.png" height="150" />

# [Vulkano](https://vulkano.rs)

[![Crates.io](https://img.shields.io/crates/v/vulkano.svg)](https://crates.io/crates/vulkano)
[![Docs](https://docs.rs/vulkano/badge.svg)](https://docs.rs/vulkano)
[![Build Status](https://travis-ci.org/vulkano-rs/vulkano.svg?branch=master)](https://travis-ci.org/vulkano-rs/vulkano)
[![Gitter chat](https://badges.gitter.im/vulkano-rs/Lobby.png)](https://gitter.im/vulkano-rs/Lobby)

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

Note that in general vulkano does **not** require you to install the official Vulkan SDK. This is
not something specific to vulkano (you don't need the SDK to write programs that use Vulkan, even
without vulkano), but many people are unaware of that and install the SDK thinking that it is
required. However, macOS and iOS platforms do require a little more Vulkan setup since it is not
natively supported. See below for more details.

## Development status

Vulkano is still in heavy development and doesn't yet meet its goals of being very robust. However
the general structure of the library is most likely definitive, and all future breaking changes
will likely be straight-forward to fix in user code.

## Documentation

To get started you are encouraged to use the following resources:

*   The guide on [vulkano.rs](http://vulkano.rs/guide/introduction) - Starts with trivial compute
    examples (~50 lines of code) then works up to rendering triangles and mandelbrots.
*   [The `vulkano-examples` repository](https://github.com/vulkano-rs/vulkano-examples) - Includes
    examples in the repo and also a list of projects that use vulkano.
*   [docs.rs](https://docs.rs/vulkano) - Full Vulkano API documentation

## macOS and iOS Setup

Vulkan is not natively supported by macOS and iOS. However, there exists [MoltenVK](https://github.com/KhronosGroup/MoltenVK)
a Vulkan implementation on top of Apple's Metal API. This allows vulkano to build and run on macOS
and iOS platforms.

The easiest way to get vulkano up and running on macOS is to install the 
[Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home). To install the SDK so that
Vulkano will find it and dynamically link with `libvulkan.dylib`:

1. Download the latest macOS release and unpack it somewhere, for the next step
we'll assume that's `~/vulkan_sdk`.
2. Modify your environment to contain the SDK bin directory in PATH and the SDK lib directory in 
DYLD_LIBRARY_PATH. We also need to set VK_ICD_FILENAMES and VK_LAYER_PATH. When using the Bash 
shell, which is the default for macOS, it's easiest to do this by appending the following to the
`~/.bash_profile` file and then restarting the terminal.

```sh
export VULKAN_SDK=$HOME/vulkan_sdk/macOS
export PATH=$VULKAN_SDK/bin:$PATH
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH
export VK_ICD_FILENAMES=$VULKAN_SDK/etc/vulkan/icd.d/MoltenVK_icd.json
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
```

It is also possible to link with the MoltenVK framework (as vulkano did in previous versions) by adding the
appropriate cargo output lines to your build script and implementing your own
`vulkano::instance::loader::Loader` that calls the MoltenVK `vkGetInstanceProcAddr` implementation.

On iOS vulkano links directly to the MoltenVK framework. There is nothing else to do besides
installing it. Note that the Vulkan SDK for macOS also comes with the iOS framework.

## Donate

[![Become a patron](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/tomaka)

## Contributing

Contributions are welcome! Feel free to submit pull requests.

Pull requests that fix bugs or improve documentation are likely to be quickly reviewed, while pull
requests that add features or change the API may be more controversial and take more time.

If your change adds, removes or modifies a trait or a function, please add an entry to the
`CHANGELOG.md` file as part of your pull request.

## Structure

This repository contains six libraries:

- `vulkano` is the main one.
- `vulkano-shaders` can analyse SPIR-V shaders at compile-time.
- `vulkano-shader-derive` provides a custom derive that invokes `vulkano-shaders`. It lets you
  easily integrate your GLSL shaders within the rest of your source code.
- `vulkano-win` provides a safe link between vulkano and the `winit` library which can create
  a window to render to.
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
