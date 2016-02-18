# Vulkano

Safe Rust wrapper around Vulkan.

- Much easier to use than raw Vulkan.

- Any error that the validation layer would trigger is avoided in the first
  place. This is done through a lot of compile-time checks and a few runtime
  checks.

- Anything that is possible to do with Vulkan should be possible with vulkano
  as well. Please open an issue if this is not the case.

- Safety is favored over performances. In particular, compared to raw Vulkan
  vulkano does some runtime checks and wraps most objects around
  reference-counted pointers.

## Usage

Add to your Cargo.toml:

```toml
vulkano = "0.1"
```

Note that this library doesn't handle creating and managing windows. In order
to render to a window, you will have to create that window separately and use
an unsafe function of this library to link to it.

This only concerns windows. Safe fullscreen rendering is possible with this
library alone.

## Shaders handling

The API of vulkano related to shader modules is entirely unsafe. This is
because you're not supposed to use it directly.

Instead, you are encouraged to use the `vulkano-shaders` crate which compiles
and analyses your shader, and generates Rust code that wraps around vulkano's
API.

Thanks to this, vulkano can provide compile-time guarantees about your
specialization constants, push constants, descriptor sets, vertex layouts, and
so on.
