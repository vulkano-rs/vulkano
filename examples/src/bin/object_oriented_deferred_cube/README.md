# Object-oriented example of textured cubes ft. texture array, subpass-based deferred shading and screen-space post-processing

Last updated 2022-01 for Vulkano 0.28 (unreleased) and Vulkan 1.2

---

[Start by diving into the main.rs.](main.rs)

It's recommended to clone this repository (it won't take long!) and read it in your preferred IDE instead of web browser. You can get invaluable parameter name hints and the ability to jump to symbol definition this way.

---

That was some long title.

This example assumes you have some idea about how GPU rendering works, but new to Vulkan and Vulkano; If not, I recommend a quick read of [LearnOpenGL](https://learnopengl.com/) minus the non-GLSL code details (they are useless here).

In case you want to modify this example into a basic real-time deferred 3D renderer but don't know how:

- Implement shadow mapping.

  [Sascha Willems has an example on how to use geometry shaders for rendering all the shadow maps in a single pass](https://github.com/SaschaWillems/Vulkan/blob/master/examples/deferredshadows/deferredshadows.cpp), if you can read C++ that is.

- Implement some real PBR direct lighting.

  The current industrial standard is [Principled BSDF](https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf) and its derivatives, but it's too complicated and has a [suspicious license](https://lists.opensource.org/pipermail/license-discuss_lists.opensource.org/2021-February/021896.html), so it's not included.

- Implement a good screen-space ambient occlusion, or even world-space AO if you could.

  I don't have a recommendation on SSAO, but as long as you don't use the ancient ones (like the one presented in LearnOpenGL) you'll be fine.

- Implement anti-aliasing.

  The simplest and slowest method is to render at a higher resolution then scale it back ([SSAA](https://en.wikipedia.org/wiki/Supersampling)), but there are alternatives that trades fidelity for vastly improved performance, like [MSAA](https://docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/d3d_samples/antialiaseddeferredrendering.htm), [FXAA](https://en.wikipedia.org/wiki/Fast_approximate_anti-aliasing) and [TAA](https://en.wikipedia.org/wiki/Temporal_anti-aliasing) (sorted by fidelity in descending order).

- Implement general optimizations.

  You'll quickly run into performance issues if you don't implement e.g. view frustum culling, occlusion culling, level of detail, light/shadow map caching.

- Implement Vulkan-specific optimizations.

  Such as command buffer reusing. [Nvidia has a list covering this](https://developer.nvidia.com/blog/vulkan-dos-donts/), though somewhat confusingly written and doesn't map 1:1 to Vulkano.

## Libraries used

- Windowing: [winit](https://github.com/rust-windowing/winit)

  Although its design is frustrating for Rust beginners (at least for me) and is lacking some features like IME support, it doesn't require any additional build steps unlike SDL.

- Linear algebra: [nalgebra](https://nalgebra.org)

  Seems like the future of Rust linear algebra, and subjectively the most well designed and feature-rich. But [it currently breaks rust-analyzer](https://github.com/rust-analyzer/rust-analyzer/issues/8654).

- Others:
  - [simple-logger](https://github.com/borntyping/rust-simple_logger),
    currently the only logger not using the [known](https://www.reddit.com/r/rust/comments/qamgyh/is_the_chrono_crate_unmaintained) [vulnerable](https://github.com/chronotope/chrono/issues/602) [chrono](https://github.com/chronotope/chrono) crate. Feel free to switch to any other [log-compatible](https://github.com/rust-lang/log) loggers after they resolved the issue.
  - [anyhow](https://github.com/dtolnay/anyhow) to avoid unwrapping everywhere while preventing excess amount of `Error` type boilerplate. If you are seeing `anyhow!()` and `bail!()`, that's an error early-return using anyhow.

## Caveats

Here's a list of common tasks Vulkano currently can't do:

- Ray tracing with VK_KHR_ray_tracing_pipeline, i.e. the thing Nvidia marketed as RTX ([#1350](https://github.com/vulkano-rs/vulkano/pull/1350))

  You can still ray trace with compute shaders. They are just slower.

- Unbounded arrays in shader (`uniform sampler2D[]` without specifying a number between the brackets) ([#1743](https://github.com/vulkano-rs/vulkano/pull/1743))

- Debugging painlessly in RenderDoc ([#1588](https://github.com/vulkano-rs/vulkano/pull/1588))

- 64-bit floats
