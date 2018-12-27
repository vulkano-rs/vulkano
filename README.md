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
*   The [vulkano-examples](https://github.com/vulkano-rs/vulkano-examples) repository - Includes
    examples in the repo and also a list of projects that use vulkano.
*   [docs.rs](https://docs.rs/vulkano) - Full Vulkano API documentation

## Setup

Vulkano uses [shaderc-rs](https://github.com/google/shaderc-rs) for shader compilation. In order to
build the shaderc-rs crate the following tools must be installed and available on `PATH`:
- [CMake](https://cmake.org/)
- [Python](https://www.python.org/) (works with both Python 2.x and 3.x)

These requirements can be either installed with your favourite package manager or with installers
from the projects' websites. Below are some example ways to get setup.

### windows-msvc Specific Setup

1. `rustup default stable-x86_64-pc-windows-msvc`
2. Install [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017). If you have already been using this toolchain then its probably already installed.
3.  Install [msys2](http://www.msys2.org/), following ALL of the instructions.
4.  Then in the msys2 terminal run: `pacman --noconfirm -Syu mingw-w64-x86_64-cmake mingw-w64-x86_64-python3`
5.  Add the msys2 mingw64 binary path to the PATH environment variable.

### Windows-gnu Specific Setup

windows-gnu toolchain is not supported but you can instead cross-compile to windows-gnu from windows-msvc

Steps 1 and 2 are to workaround https://github.com/rust-lang/rust/issues/49078 by using the same mingw that rust uses.

1.  Download and extract https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/x86_64-6.3.0-release-posix-seh-rt_v5-rev2.7z
2.  Add the absolute path to mingw64\bin to your PATH environment variable. (This path needs to be before the msys2 path)
3.  Run the command: `rustup default stable-x86_64-pc-windows-msvc`
4.  Run the command: `rustup target install x86_64-pc-windows-gnu`
5.  Install [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017). If you have already been using this toolchain then its probably already installed.
6.  Install [msys2](http://www.msys2.org/), following ALL of the instructions.
7.  Then in the msys2 terminal run: `pacman --noconfirm -Syu mingw64/mingw-w64-x86_64-pkg-config mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make mingw-w64-x86_64-python3`
8.  Add the msys2 mingw64 binary path to the PATH environment variable.
9.  Any cargo command that builds the project needs to include `--target x86_64-pc-windows-gnu` e.g. to run: `cargo run --target x86_64-pc-windows-gnu`

### Linux Specific Setup

Use your package manager to install the required dev-tools and vulkan drivers

For example on ubuntu:
```
sudo apt-get install build-essential git python cmake libvulkan-dev vulkan-utils
```

### macOS and iOS Specific Setup

Vulkan is not natively supported by macOS and iOS. However, there exists [MoltenVK](https://github.com/KhronosGroup/MoltenVK)
a Vulkan implementation on top of Apple's Metal API. This allows vulkano to build and run on macOS
and iOS platforms.

The easiest way to get vulkano up and running on macOS is to install the
[Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home). To install the SDK so that
Vulkano will find it and dynamically link with `libvulkan.dylib`:

Vulkano will attempt to automatically download and install the latest Vulkan SDK for macOS.
If you would prefer a manual installation complete the following instructions before
building vulkano.

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

Note that as of writing, MoltenVK has some bugs that show up in the examples.
Some minor modifications may be required as workarounds: see https://github.com/vulkano-rs/vulkano/pull/1027.
The examples also do not work properly on macOS 10.11 and lower without workarounds due to MoltenVK's Metal backend not getting
the required features until macOS 10.12. See https://github.com/vulkano-rs/vulkano/issues/1075 for workarounds.

## Donate

Rukai (Current maintainer) [![Become a patron](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/rukai)

Tomaka (Original developer) [![Become a patron](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/tomaka)

## Contributing

Contributions are welcome! Feel free to submit pull requests.

Pull requests that fix bugs or improve documentation are likely to be quickly reviewed, while pull
requests that add features or change the API may be more controversial and take more time.

If your change adds, removes or modifies a trait or a function, please add an entry to the
`CHANGELOG.md` file as part of your pull request.

## Structure

This repository contains six libraries:

- `vulkano` is the main one.
- `vulkano-shaders` Provides the `shader!` macro for compiling glsl shaders.
- `vulkano-win` provides a safe link between vulkano and the `winit` library which can create
  a window to render to.
- `vk-sys` contains raw bindings for Vulkan. You can use it even if you don't care about vulkano.

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
