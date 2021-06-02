# vulkano-gen

A tool for Vulkano to automatically generate various lists of Vulkan data, using the [vk-parse](https://github.com/krolli/vk-parse) library. This tool is intended for internal Vulkano use.

## Updating

The tool should be run whenever Vulkano is updated to use a new version of [Ash](https://github.com/MaikKlein/ash). First, replace the `vk.xml` file in the Vulkano root with a new one from [Khronos](https://github.com/KhronosGroup/Vulkan-Headers/commits/master/registry/vk.xml). Make sure to use the revision with the same Vulkan patch version that was used in the Ash version that Vulkano uses. This ensures that the generated code is compatible with Ash. Then run the tool from the Vulkano project root:

```
cargo run --bin vulkano-gen
```

The tool will then write new contents to `vulkano/src/autogen.rs`.