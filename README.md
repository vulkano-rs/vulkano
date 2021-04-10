<img align="left" alt="" src="logo.png" height="150" />

# [Vulkano](https://vulkano.rs)

[![Crates.io](https://img.shields.io/crates/v/vulkano.svg)](https://crates.io/crates/vulkano)
[![Docs](https://docs.rs/vulkano/badge.svg)](https://docs.rs/vulkano)
[![Build Status](https://github.com/vulkano-rs/vulkano/workflows/Rust/badge.svg)](https://github.com/vulkano-rs/vulkano/actions?query=workflow%3ARust)

See also [vulkano.rs](https://vulkano.rs).

Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.khronos.org/vulkan/).
It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't
be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code
should always conform to valid API usage.

What does vulkano do?

- Provides a low-levelish API around Vulkan. It doesn't hide what it does, but provides some
  comfort types.
- Plans to prevent all invalid API usages, even the most obscure ones. The purpose of Vulkano
  is not to simply let you draw a teapot, but to cover all possible usages of Vulkan and detect all
  the possible problems in order to write robust programs. Invalid API usage is prevented thanks to
  both compile-time checks and runtime checks.
- Can handle synchronization on the GPU side for you (unless you choose do that yourself), as this
  aspect of Vulkan is both annoying to handle and error-prone. Dependencies between submissions are
  automatically detected, and semaphores are managed automatically. The behavior of the library can
  be customized thanks to unsafe trait implementations.
- Tries to be convenient to use. Nobody is going to use a library that requires you to browse
  the documentation for hours for every single operation.

### Comparison

Comparison to other well-known Graphics APIs in Rust ecosystem.

| Name | Open-sourced Since | API Level | Notable Features |
| ---- | ------------------ | --------- | ------------- |
| Vulkano | March, 2016 | Fully safe high-level Rust API wrapping Vulkan/Metal APIs. | Type-safe compile-time shaders. Transparent interoperability with glsl and spir-v shader code types in Rust code. Automatically generated safe types for shader's Layout. |
| [Wgpu-rs](https://github.com/gfx-rs/wgpu-rs) | May, 2019 | Partially unsafe high-level Rust API with multiple backends. | Fully safe run-time shaders. Supports multiple backends: Vulkan, Metal, DirectX, WebGPU, and other. Following WebGPU specification. With async/await API. |
| [Miniquad](https://github.com/not-fl3/miniquad) | March, 2020 | Safe high-level minimalistic Rust API with multiple backends. | Relatively minimalistic API well suited for small to medium graphics projects. Supports multiple backends, including browser target. |
| [Sierra](https://github.com/zakarumych/sierra) | March, 2021 | Partially safe high-level Rust API for Vulkan/Metal APIs. | Layouts, Descriptors and shader Types construction in Rust code through the macro system in safe way. Built on top of [Erupt](https://gitlab.com/Friz64/erupt), but fully hiding it's unsafe API. |
| [Glium](https://github.com/glium/glium) | October, 2014 | Safe high-level Rust API wrapping OpenGL | OpenGL only. |
| [Ash](https://github.com/MaikKlein/ash) | August, 2016 | Low-level unsafe API for Vulkan. | Unsafe Vulkan API bindings. |
| [Erupt](https://gitlab.com/Friz64/erupt) | April, 2020 | Low-level unsafe API for Vulkan. | Unsafe Vulkan API bindings. |

Please note that by the current date non of the known projects in the ecosystem(including Vulkano)
reached stable release versions and the final design goals, their APIs are changing from time
to time in breakable way too, and there could be bugs and unfinished features too.

However, most of the projects mentioned above are already established definitive structure, all
feature breaking changes will likely be straight-forward to fix in user code, and most of them
are maintained. As such we can recommend to start using any of them in the 3rd party code.
The choice depends on the end project's goals and requirements, and we recommend examining
actual set of their features and API capabilities beforehand.

### Projects using Vulkano

We started collecting this list just recently, and will be appreciated if you help us by
contributing(opening a PR) into [README.md](https://github.com/vulkano-rs/vulkano/blob/master/README.md).

| Project Name | Description |
| ------------ | ----------- |
| [Basalt](https://github.com/AustinJ235/basalt) | GUI framework for Desktop applications |
| [Ferret](https://github.com/Rua/ferret) | Doom-compatible game engine |

We would love to help you keep your project in sync with the most recent changes in Vulkano
if you give us feedback by adding your project into this list.

Thanks in advance!

## Documentation and Resources

To get started you are encouraged to use the following resources:

 - The [vulkano-examples](https://github.com/vulkano-rs/vulkano-examples) repository - Includes
   examples in the repo and also a list of projects that use vulkano.
 - [docs.rs](https://docs.rs/vulkano) - Full Vulkano API documentation
 - The guide on [vulkano.rs](https://vulkano.rs/guide/introduction) - Starts with trivial compute
   examples (~50 lines of code) then works up to rendering triangles and mandelbrots.
   The guide is currently outdated a little. We are planning to update it in the future, but it's
   a good place to start understanding base building blocks of Vulkano API.
 - Github [Issues](https://github.com/vulkano-rs/vulkano/issues) - Raise a topic, ask a question
   or report a bug. The new topics there are watching regularly by maintainers and other
   community users.
 - Gitter [Chat](https://gitter.im/vulkano-rs/Lobby) - Another place to raise a question. However,
   the chat is not maintained regularly at this moment. Better use Github Issues for this purpose.

## Contributing

Contributions are very welcome! Feel free to submit pull requests, to open questions and topics
in the [Issues](https://github.com/vulkano-rs/vulkano/issues) section.

The project was initially developed by Pierre Krieger(Tomaka), who established Vulkano's base
design goals, and the code structure. In the meantime the development is driven by Vulkano
community members.

**New Pull Requests are usually scheduled for review by the end of each week.**
The older PRs that already in review have priority over the new ones. We are trying to push
development forward as quick as possible, but the review process sometimes may take time,
please be patient as the maintainers need time to check everything properly.

If something needs to get promoted urgently, please ping current Vulkano
maintainer([@Eliah-Lakhin](https://github.com/Eliah-Lakhin/jonathan-hoag)) in the PR's
or Issue's comments.

If your change adds, removes or modifies a trait or a function, please add an entry to the
changelog file(s) as part of your pull request.

Every PR must pass tests in order to be merged to `master`.

Minor releases are usually happening between 1 to 3 months averagely depending on the grow
of unreleased and breaking changes in `master`

### Repository Structure

This repository contains four libraries:

- `vulkano` is the main one.
- `vulkano-shaders` Provides the `shader!` macro for compiling glsl shaders.
- `vulkano-win` provides a safe link between vulkano and the `winit` library which can create
  a window to render to.
- `vk-sys` contains raw bindings for Vulkan. You can use it even if you don't care about vulkano.

In order to run tests, run `cargo test --all` at the root of the repository. Make sure your Vulkan
driver is up to date before doing so.

### Hall of Fame

We would love to mention some members, who put significant contributions to this project:
- Pierre Krieger. The initial developer. [Patron page](https://www.patreon.com/tomaka).
- Lucas Kent. Maintainer. [Patron page](https://www.patreon.com/rukai).
- Austin Johnson. Maintainer. [Patron page](https://www.patreon.com/austinj235).
- Rua. An active developer, who put a lot of efforts to improve Vulkano and constantly keeping it up to date.
- **You!** Thanks to your help, contributions, improvements, bug reports and use experience
  to make this project one of the major Rust graphics API library in Rust!

## Setup and Troubleshooting

Vulkano uses [shaderc-rs](https://github.com/google/shaderc-rs) for shader compilation.  Refer to shaderc-rs documentation to provide a pre-built libshaderc for faster build times.

Note that in general vulkano does **not** require you to install the official Vulkan SDK. This is
not something specific to vulkano (you don't need the SDK to write programs that use Vulkan, even
without vulkano), but many people are unaware of that and install the SDK thinking that it is
required. However, macOS and iOS platforms do require a little more Vulkan setup since it is not
natively supported. See below for more details.

Unless you provide libshaderc, in order to build libshaderc with the shaderc-sys crate, the following tools must be installed and available on `PATH`:
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/) Is optional except when building with MSVC.  It may speed up build time for libshaderc.
- [Python](https://www.python.org/) (works with both Python 2.x and 3.x, on windows the executable must be named `python.exe`)

These requirements can be either installed with your favourite package manager or with installers
from the projects' websites. Below are some example ways to get setup.

### windows-msvc Specific Setup

1. `rustup default stable-x86_64-pc-windows-msvc`
2. Install [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017). If you have already been using this toolchain then its probably already installed.
3.  Install [msys2](https://www.msys2.org/), following ALL of the instructions.
4.  Then in the msys2 terminal run: `pacman --noconfirm -Syu mingw-w64-x86_64-cmake mingw-w64-x86_64-python2 mingw-w64-x86_64-ninja`
5.  Add the msys2 mingw64 binary path to the PATH environment variable.

### Windows-gnu Specific Setup

windows-gnu toolchain is not supported but you can instead cross-compile to windows-gnu from windows-msvc

Steps 1 and 2 are to workaround https://github.com/rust-lang/rust/issues/49078 by using the same mingw that rust uses.

1.  Download and extract https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/x86_64-6.3.0-release-posix-seh-rt_v5-rev2.7z
2.  Add the absolute path to mingw64\bin to your PATH environment variable. (This path needs to be before the msys2 path)
3.  Run the command: `rustup default stable-x86_64-pc-windows-msvc`
4.  Run the command: `rustup target install x86_64-pc-windows-gnu`
5.  Install [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017). If you have already been using this toolchain then its probably already installed.
6.  Install [msys2](https://www.msys2.org/), following ALL of the instructions.
7.  Then in the msys2 terminal run: `pacman --noconfirm -Syu mingw64/mingw-w64-x86_64-pkg-config mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make mingw-w64-x86_64-python2 mingw-w64-x86_64-ninja`
8.  Add the msys2 mingw64 binary path to the PATH environment variable.
9.  Any cargo command that builds the project needs to include `--target x86_64-pc-windows-gnu` e.g. to run: `cargo run --target x86_64-pc-windows-gnu`

### Linux Specific Setup

Use your package manager to install the required dev-tools and vulkan drivers

For example on ubuntu:
```
sudo apt-get install build-essential git python cmake libvulkan-dev vulkan-utils
```
On arch based system
```
sudo pacman -Sy base-devel git python cmake vulkan-devel --noconfirm
```

### macOS and iOS Specific Setup

Vulkan is not natively supported by macOS and iOS. However, there exists [MoltenVK](https://github.com/KhronosGroup/MoltenVK)
an open-source Vulkan implementation on top of Apple's Metal API. This allows vulkano to build and run on macOS
and iOS platforms.

The easiest way to get vulkano up and running with MoltenVK is to install the
[Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home). There are [installation instructions](https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html) on the LunarG website.

On iOS, vulkano links directly to the MoltenVK framework. There is nothing else to do besides
installing it. Note that the Vulkan SDK for macOS also comes with the iOS framework.

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
