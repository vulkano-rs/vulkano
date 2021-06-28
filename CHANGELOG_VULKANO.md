# Unreleased
<!--
    This file is maintained by the repository organization members. Please, do
    not put changes into changelog files directly, they lead to frequent merging
    conflicts. Instead put incoming changelog entries into the Pull Request
    description. They will be transferred to this file right after the
    Pull Request merge. 
-->
- Examples now enable only the features they need instead of all of them.
- Examples have much smarter device selection logic. In the triangle example this is supplied with comments.
- Errors checking(by unwrapping) in `MappedDeviceMemory::read_write`.
- Add creation of Semaphores with exportable Linux file descriptor on.
- Add method to export file descriptor corresponding to Semaphore.
- `SemaphoreBuilder` introduced. 
- Add DisplayNative enum variant to ColorSpaceEnum (AMD-specific feature).

# Version 0.24.0 (2021-06-20)

- **Breaking** Vulkano-shaders now checks if the device supports the shader's SPIR-V version, when loading the shader.
- **Breaking** (but unlikely) Vulkano-shaders now compiles to SPIR-V 1.0 by default. If your shader needs features only available in a higher version, you can specify the target version on the `shader!` macro with the new `vulkan_version: "major.minor"` and `spirv_version: "major.minor"` arguments.
- **Breaking** Changes to how image sample counts are represented.
  - Instead of an integer, functions with a parameter for number of image samples now take a value of `SampleCount`, an enum with variants named `SampleN`, where `N` is a power-of-two integer. It can be converted to a Vulkan `SampleCountFlags`, and from an integer with `try_from`.
  - `sample_counts` field is originaly represented as u32 type, which is now represented by `SampleCounts` struct-type which is a boolean collection of supported `sample_counts`. It can be converted to and from a Vulkan `SampleCountFlags`.
- **Breaking** Changes to shader interfaces and pipeline layouts.
  - The module `descriptor::pipeline_layout` has been renamed to `pipeline::layout`.
  - The trait `ShaderInterfaceDef` has been replaced by a simple struct `ShaderInterface`, and its `elements` method returns a slice instead of an iterator. This means you no longer need to define a new type for a shader interface. The accompanying type `ShaderInterfaceDefEntry` has been renamed to `ShaderInterfaceEntry` to match. The `ShaderInterfaceDefMatch` trait and `EmptyShaderInterfaceDef` struct have been removed.
  - The trait `PipelineLayoutDesc` has also been converted into a struct, with methods `descriptor_sets` and `push_constants` which return slices. It is functionally equivalent to the old `RuntimePipelineDesc` type, which is now merged into it. The traits `PipelineLayoutSuperset`, `PipelineLayoutSetsCompatible` and `PipelineLayoutPushConstantsCompatible` have been integrated into this struct as well. The `EmptyPipelineDesc` trait has been replaced with an `::empty()` constructor.
  - Consequently, functions that took a value of these traits now take a plain `ShaderInterface` or `PipelineLayoutDesc`. Types that had a type parameter for it no longer have it, e.g. `VertexDefinition`, `GraphicsEntryPoint`, `GraphicsEntryPointAbstract`, `PipelineLayout`.
  - Now that `PipelineLayout` has no more type parameter, the trait `PipelineLayoutAbstract` is removed. The layout type parameter is also removed from `ComputePipeline` and `GraphicsPipeline`.
  - `ComputeEntryPoint` and `GraphicsEntryPoint` now take a value specifying the push constants descriptor, instead of having a type parameter. The corresponding associated type on `EntryPointAbstract` has been removed.
  - The `GraphicsEntryPointAbstract` trait has been removed. `GraphicsPipelineBuilder` now takes a `GraphicsEntryPoint` object directly, and has lifetime parameters for the 5 shader types instead. `EntryPointDummy` is no longer needed and has been removed.
- **Breaking** The constructors of `Instance` now take an additional argument to specify the maximum API version.
- **Breaking** Vulkano is now built on top of [Ash](https://github.com/MaikKlein/ash) instead of vk-sys.
  - The `EntryPoints`, `InstancePointers` and `DevicePointers` types from vk-sys have been replaced with a new module `fns` containing `EntryFunctions`, `InstanceFunctions` and `DeviceFunctions`. Rather than containing the functions directly, there is a member for each Vulkan version and extension, which is loaded from Ash.
  - The functions to retrieve the function pointers have been renamed to `fns`.
  - The `TYPE` associated constant has been removed from the `VulkanObject` trait. This is now provided by the Ash `Handle` trait, which the object returned by `internal_object` must implement.
- **Breaking** `RawInstanceExtensions` and `RawDeviceExtensions` have been removed. The `Instance` and `Device` constructors now take `&InstanceExtensions` and `&DeviceExtensions` respectively. The `loaded_extensions` function returns these reference types as well.
- **Breaking** The restrictions of each enabled extension and feature are checked when creating an instance or device.
- **Breaking** Physical device properties are now retrieved through a new `properties` method, returning a `&Properties` reference.
  - To allow for the possibility that not every property is known by the physical device, fields in `Properties` are wrapped by an `Option`.
  - The previous methods for retrieving properties, `name`, `ty`, `limits`, `driver_version`, `pci_device_id`, `pci_vendor_id`, `uuid` and `extended_properties`, have been removed.
  - The `api_version` method remains, but there is now a semantic difference between it and the version reported by `properties`: The raw property gives the maximum supported version that the driver itself reports, while the method on `PhysicalDevice` returns the version a logical device would support if it were created from this physical device (that is, restricted by the instance's `max_api_version`).
- **Breaking** `ImageAccess` trait method `try_gpu_lock()` now has an additional argument to allow locking the image in an uninitialized state.`
- Improve `ImageLayout` checks to prevent `AccessError::ImageNotInitialized` from occurring where the image is safe to use uninitialized.
- Added `DeviceExtensions::khr_spirv_1_4`, which allows SPIR-V 1.4 shaders in Vulkan 1.1.
- Added `FunctionPointers::api_version` to query the highest supported instance version.
- Added `Instance::api_version` and `Device::api_version` to return the actual supported Vulkan version. These may differ between instance and device, and be lower than what `FunctionPointers::api_version` and `PhysicalDevice::api_version` return.
- Added `Instance::max_api_version`, which returns the maximum version that was specified when creating the instance.
- Fixed the issue when creating a buffer with exportable fd on Linux(see to #1545).
- The `draw_indirect` and `draw_indexed_indirect` commands on `AutoCommandBufferBuilder` now check the draw count against the `max_draw_indirect_count` limit.
- Fixed a few documentation errors.
- It is now possible to construct a graphics pipeline without a fragment shader.
- Added support for all core Vulkan 1.1 and 1.2 device features.
- Updated winit to 0.25.
- Fixed the teapot example on ArchLinux (GTX 1650).
- Added support for the SPIR-V draw parameters capability.
- Added support for the VK_KHR_multiview extension. 
- Vulkano-shaders: Added support for MultiView SPIR-V capability.
- Multiview example added showing how to utilize the VK_KHR_multiview extension to render to multiple layers of a framebuffer at once.
- All Vulkan extensions supported by Ash are now provided in `InstanceExtensions` and `DeviceExtensions`. This includes all but the very newest extensions; new extensions should be added whenever Ash is updated to a new version.
- Every extension is now documented with a link to its Vulkan page and information about requirements, conflicts, promotion and deprecation.
- `InstanceExtensions` and `DeviceExtensions` now have a `From` implementation that takes an iterator of `&CStr`. There is also a `From` implementation for `Vec<CString>` that performs the reverse conversion.
- All Vulkan features supported by Ash are now provided.
- Implement fmt::Display for DisplayMode.
- Clarify return value is in millihertz for DisplayMode.refresh_rate()

# Version 0.23.0 (2021-04-10)

- **Breaking** Changes to command buffers:
  - `AutoCommandBuffer` and the `CommandBuffer` trait have been split in two, one for primary and the other for secondary command buffers. `AutoCommandBufferBuilder` remains one type, but has a type parameter for the level of command buffer it will be creating, and some of its methods are only implemented for builders that create `PrimaryAutoCommandBuffer`.
  - The `Flags` enum is renamed to `CommandBufferUsage`, and is exported from the main `command_buffer` module. The `None` variant is renamed to `MultipleSubmit`.
  - Simplified the number of constructors on `AutoCommandBufferBuilder` by adding a `CommandBufferUsage` parameter.
  - `Kind` has been renamed to `CommandBufferLevel`, and for secondary command buffers it now contains a single `CommandBufferInheritance` value.
  - `CommandBufferInheritance::occlusion_query` and `UnsafeCommandBufferBuilder::begin_query` now take `QueryControlFlags` instead of a boolean.
- **Breaking** The non-default constructors of `ImageView` have been replaced with a builder, created with `ImageView::start(image)`.
- **Breaking** Added support for component mapping/swizzling on image views.
  - `image::Swizzle` is moved and renamed to `image::view::ComponentMapping`. It now has an `is_identity` method.
  - A non-default component mapping can now be specified for image views, via the new builder. A `ComponentMapping` parameter has been added to `UnsafeImageView` as well.
  - The `identity_swizzle` method on the `ImageViewAbstract` trait has been replaced with `component_mapping`, which returns a `ComponentMapping` directly.
  - Storage image and input attachment descriptors now check for identity swizzling when being built.
- **Breaking** Major rearranging of framebuffer and render pass-related types:
  - The `framebuffer` module is renamed to `render_pass`.
  - `RenderPassDesc` is now a struct, not a trait. The methods have been simplified, returning a slice reference to the `attachments`, `subpasses` and `dependencies`.
    - Renamed: `AttachmentDescription` > `AttachmentDesc`, `PassDescription` > `SubpassDesc`, `PassDependencyDescription` > `SubpassDependencyDesc`.
    - `EmptySinglePassRenderPassDesc` is replaced with the `RenderPassDesc::empty` constructor, or its `Default` implementation.
    - The `RenderPassCompatible`, `RenderPassDescClearValues` and `RenderPassSubpassInterface` traits are removed, their functionality is moved to `RenderPassDesc`.
  - `RenderPass` takes a concrete `RenderPassDesc` value for construction, and no longer has a type parameter.
    - The `RenderPassAbstract` trait is removed.
    - `GraphicsPipeline` and `Framebuffer` no longer have a render pass type parameter.
    - `GraphicsPipelineAbstract` and `FramebufferAbstract` have trait methods to retrieve the render pass instead.
  - The `ordered_passes_renderpass!` and `single_pass_renderpass!` macros are unchanged externally.
- Support for queries:
  - **Breaking** `UnsafeQueryPool`, `UnsafeQuery` and `UnsafeQueriesRange` have `Unsafe` removed from their names.
  - **Breaking** `QueriesRange` is now represented with a standard Rust `Range` in its API.
  - **Breaking** The secondary command buffer constructors that have parameters for queries will check if the corresponding features are enabled, and return a different error type.
  - Removed `OcclusionQueriesPool`, which was incomplete and never did anything useful.
  - `get_results` has been added to `QueriesRange`, to copy query results to the CPU.
  - The following functions have been added to both `SyncCommandBufferBuilder` and `AutoCommandBufferBuilder`: `begin_query` (still unsafe), `end_query` (safe), `write_timestamp` (still unsafe), `copy_query_pool_results` (safe), `reset_command_pool` (still unsafe).
  - Better documentation of everything in the `query` module.
  - An example demonstrating occlusion queries.
- **Breaking** Improved the handling of image aspects a little, with the enum `ImageAspect` and the struct `ImageAspects`. `UnsafeCommandBufferBuilderImageAspect` has been removed.
- **Breaking** Removed the separate structs for each image format. Now, only the `Format` enum exists.
  - Traits that no longer make sense in this context have been removed: `FormatDesc`, the `Possible*FormatDesc` traits, `StrongStorage`.
  - In types that had a type parameter for the format type, it has been removed.
  - `AcceptsPixels` has been converted to `Pixel`, which is implemented on the pixel type rather than on the format type.
- **Breaking** `shader!` will generate descriptor information for all variables declared in the shader module, even if they are not used. *This reverts the default behavior from the last release.*
  - **Breaking** Added the `exact_entrypoint_interface` option to `shader!` to force vulkano to only generate descriptor information for variables that are used. (the default behavior from the last release)
- **Breaking** `AccessFlagBits` is renamed to `AccessFlags`.
- **Breaking** Minor refactor of `UnsafeBuffer`:
  - Replaced the various usage functions with a single `usage` function, mirroring the change made earlier to `UnsafeImage`.
  - The `SparseLevel::sparse` member is removed, and `UnsafeBuffer::new` takes `Option<SparseLevel>` instead.
- **Breaking** `Swapchain` is now constructed using a builder. `Swapchain::start` will start building a new blank swapchain. Calling `recreate` on an existing swapchain will create a builder filled with all the properties of the old swapchain.
- Added two methods to `Format`: `planes` to query the number of planes in the format, and `aspects` to query what aspects an image of this type has.
- The deprecated `cause` trait function on Vulkano error types is replaced with `source`.
- Fixed bug in descriptor array layers check when the image is a cubemap.
- Vulkano-shaders: Fixed and refined the generation of the `readonly` descriptor attribute. It should now correctly mark uniforms and sampled images as read-only, but storage buffers and images only if explicitly marked as `readonly` in the shader.
- Vulkano-shaders: Added support for StoragePushConstant8 SPIR-V capability.
- Fixed a bug which caused a segfault when extending memory allocation info in DeviceMemoryBuilder
- `BufferlessDefinition` and `BufferlessVertices` now derive `Copy` and `Clone`. This allows `GraphicsPipelineBuilder`s that have not yet defined a vertex buffer type to be cloned.
- Various functions for converting to/from Vulkan flags have been consolidated into implementations of the standard `From` trait.
- Export root-level `entry_point` method on `loader::FunctionPointers` type.
- Add few more `InstanceExtensions` from KHR and EXT.

# Version 0.22.0 (2021-03-31)

- **Breaking** Updated all code to Rust 2018 edition.
- **Breaking** DeviceMemoryBuilder::new() takes in `memory_index` rather than `MemoryType`.
- Fixed `shader!` generated descriptor set layouts for shader modules with multiple entrypoints.
  - **Breaking** Prefixed `shader!` generated descriptor set `Layout` structs with the name of the entrypoint the layout belongs to. For shaders generated from GLSL source, this means `Layout` has been renamed to `MainLayout`.
  - **Breaking** `shader!` will no longer generate descriptor information for variables that are declared but not used in a shader.
- **Breaking** `shader!` now accepts structs in shader interfaces decorated with `BufferBlock` rather than `Block`.
- Fixed missing barriers in dispatch calls
  - **Breaking** `shader!` no longer marks descriptor sets as readonly as a fallback when it doesn't know
    - **Breaking** The keyword `readonly` might need to be added in front of the `buffer` keyword in GLSL files to get them working again
- **Breaking** Changes to image types:
  - Image types no longer implement `ImageViewAccess`.
  - `Dimensions` is removed. Image constructors now take `ImageDimensions`.
  - `ImageDimensions` no longer has the `cubemap_compatible` member. Instead, several image constructors take `ImageCreateFlags` which specifies this aspect.
  - Replaced the various functions of `UnsafeImage` to query format features and usage with two that simply return `Formatfeatures` and `ImageUsage`.
- **Breaking** Changes to image view handling:
  - Created a new `image::view` module for all image view related things.
  - Introduced a new `ImageView` type, a safe wrapper around `UnsafeImageView`.
  - The `ImageViewAccess` trait is renamed to `ImageViewAbstract`, some methods added, removed or renamed. `ImageView` implements this trait.
  - `UnsafeImageView` no longer holds image usage information, nor does it check for valid usage.
- **Breaking** `UnsafeCommandBuffer` and `SyncCommandBuffer` and their corresponding builders and other related types no longer have a type parameter for the command pool allocation, and no longer keep the command pool alive. Their constructors now take an `&UnsafeCommandPoolAlloc`. Users must now ensure that the pool allocation outlives the command buffers and their builders (`AutoCommandBuffer` does this itself).
- **Breaking** The `CommandBuffer` trait no longer has the `PoolAlloc` associated type, and has four new methods: `num_buffers`, `buffer`, `num_images` and `image`.
- **Breaking** structures passed to `ImmutableBuffer::from_data` and `CpuAccessibleBuffer::from_data` must implement [`Copy`](https://doc.rust-lang.org/std/marker/trait.Copy.html) to ensure soundness of these functions
- Replaced deprecated `compare_and_swap` with `compare_exchange`.
- `UnsafeCommandPoolAlloc` now implements `DeviceOwned`.
- Allow `const` usage of features and `BufferUsage`.
- Opaque fd and dma-buf import support on `Linux`.
- `DeviceMemoryMapping` to separate device memory and mappings.
- Added external memory support for `DeviceLocalBuffer` for Linux.
- Implemented synchronization for `SyncCommandBufferBuilder::execute_commands`.
- `AutoCommandBufferBuilder::execute_commands` is now fully safe to use.
- `SyncCommandBufferBuilder` now becomes poisoned when it returns an error, to prevent using the builder in an inconsistent state.
- Added a `dispatch_indirect` command to `AutoCommandBufferBuilder`.

# Version 0.21.0 (2021-03-05)

- **Breaking** `Message::layer_prefix` turned to Option to prevent segfaults when Vulkan message didn't provide `pMessageIdName` value
- **Breaking** On `AutoCommandBufferBuilder`, methods that bind a descriptor set now take a `dynamic_offsets` parameter
- **Breaking** On `AutoCommandBufferBuilder` and `SyncCommandBufferBuilder`, the `update_buffer` method now takes `data` by reference
- **Breaking** Made `PipelineLayoutDescTweaks` public, for use with compute pipelines
- Added support for `ImageAspect` and YV12/NV12 formats,  for use with the UnsafeImage API.
- Added basic VK_KHR_external_memory, VK_KHR_external_memory_fd, and VK_EXT_external_memory_dma_buf support.
- Fixed potential segmentation fault in `ComputePipeline` when referencing `PipelineCache` objects.
- Fixed race condition in `StandardCommandPool` when allocating buffers.
- Fixed potential stack overflow error in loading large shaders by storing the bytecode as static.
- Fixed descriptor set layouts with arrays containing more than one element triggering unreachable code.
- Fixed panic on some machines when calling `Surface::capabilities`
- Added basic support and safety checks for dynamic uniform/storage buffers
- Updated dependencies:
  - `crossbeam` 0.7 -> 0.8
  - `half` 1.6 -> 1.7
  - `shaderc` 0.6 -> 0.7
  - `smallvec` 1.4 -> 1.6
  - `cgmath` 0.17 -> 0.18 (for examples)
  - `png` 0.15 -> 0.16 (for examples)
  - `time` 0.1 -> 0.2 (for examples)
- Added `VK_KHR_portability_subset` device extension.
- Added `DeviceExtensions::required_extensions` function that returns a set of available extensions required to create `Device` on this platform.
- `FormatFeatures` now implements `Copy`.
- Removed the `AttachmentImageView` trait, which didn't appear to be used for anything anyway.

# Version 0.20.0 (2020-12-26)

- **Breaking** The `ImmutableImage::from_iter` and `ImmutableImage::from_buffer` can build Mipmaps
- **Breaking** `CpuAccessibleBuffer` now uses `RwLock` from `parking_lot`.
- **Breaking** The `Kind` and `SubpassContents` types have been moved to the root of the `command_buffer` module.
- **Breaking** On `AutoCommandBufferBuilder`, the methods `begin_render_pass` and `next_subpass` now take `SubpassContents` instead of a boolean value.
- **Breaking** The `CommandBuffer` trait now has an additional required method, `kind`.
- **Breaking** Update dependency `winit` 0.22 -> 0.24
- **Breaking** Added an Option to the ComputePipeline constructor to enable pipeline caching.
- Add support for 8bit storage.
- Some safety checks have been added to `execute_commands`/`execute_commands_from_vec` on `AutoCommandBufferBuilder`, but not everything is checked yet so it's still unsafe to call.
- Added two vulkano-win features `raw-window-handle_` and `winit_` to be able to use vulkano without winit.
- Added function to create surface from `RawWindowHandle`.
- Added a `properties` method to `Format`.
- Added additional device feature flags for enabling SPIR-V related capabilities.
- Added method `copy_buffer_dimensions` that allows copying parts of buffers containing arrays.
- Added `debug_marker_begin`, `debug_marker_end` and `debug_marker_insert` to `AutoCommandBufferBuilder`.
- Fixed surface creation function on Windows(PR #1410).
- Travis CI Linux Nightly job temporary disabled until #1423 resolved.
- Renamed feature from `shader_f3264` to `shader_float64`.
- Added method `build_with_cache` to the `GraphicsPipelineBuilder` that enables pipeline caching.
- Check usage bits on image when creating image view.
- Fixing an assertion panic in the SyncCommandBuffer. If the buffer encountered an error while locking the necessary resources, it would unlock all previously locked resources. Images were unlocked incorrectly and an assert in the image unlock function would panic.
- Added support for including precompiled shaders in vulkano-shaders using the `bytes` option.
- Added an option for Vulkano-Shaders macro to automatically generate standard
  traits(Default, PartialEq, etc) implementations for Rust types generated from
  the Shader types, and to derive these structs with external traits.
- Fixed clear `Cubemap` and `CubemapArray` framebuffer attachment dimensions; previously only 1/6th of the layers in each of these image layouts were actually considered by the framebuffer (for clearing and rendering). This seems to have been a result of the difference between `Dimensions` and `ImageDimensions`.

# Version 0.19.0 (2020-06-01)

- **Breaking** AutoCommandBufferBuilder methods now take a mutable reference to `self` instead of taking ownership
- Update `smallvec` 1.2 -> 1.4
- Added additional `ImageUsage` constructor functions, `color_attachment`, `depth_stencil_attachment`, `transient_color_attachment` & `transient_depth_stencil_attachment`.
- Update dependency `half` 1.5 -> 1.6
- Update MacOS dependency `metal` 0.17 -> 0.18
- Added additional checks around image/device compatibility.
- Fixed all warnings as of Rust 1.43.1
- Provides new API to fetch additional information of the Physical Device from the `PhysicalDeviceProperties2KHR` structure whenever it possible. In the current implementation only `subgroupSize` property is fetching. This interface can be extended in the future to obtain more metadata depending on community needs.
- `dynamic-local-size` compute shader example added showing how to utilize `subgroupSize` to compute and set shader's local size in run time.
- Fixed Vulkano Shaders bug when compute shader local group layout values bound to specialization constants. Now it is possible to define the layout in form of `layout(local_size_x_id = 12, local_size_y_id = 13, local_size_z = 1) in;` and then set the values as `SpecializationConstants {constant_12: 8, constant_13: 4, ...}`.
- Allow applications to access the instance and device pointers
- Add a helper function '.boxed()' on the `GpuFuture` that is short for `Box::new(yourFuture) as Box<dyn GpuFuture>`

# Version 0.18.0 (2020-03-11)

- **Breaking** Update dependency `winit` 0.21 -> 0.22
- Update dependency `half` 1.4 -> 1.5
- Update dependency `smallvec` 0.6 -> 1.2

# Version 0.17.0 (2020-02-09)

- **Breaking** Swapchain::recreate_with_dimension() is now Swapchain::recreate_with_dimensions()
- **Breaking** Sync::SharingMode::Exclusive(queue_family: u32) is now Sync::SharingMode::Exclusive.
- **Breaking** Added Swapchain::with_old_swapchain() - same as previous Swapchain::new(), if an oldswapchain needs to be used
- **Breaking** Swapchain::new() now doesnt need to have the old_swapchain parameter anymore but requires the ColorSpace
- **Breaking** Decouple descriptor sets from pipeline
- **Breaking** Update Winit to 0.21.0
- **Breaking** Add `host_cached` field to all `CpuAccessibleBuffer` initializers to allow the user to perfer host cached memory.
- **Breaking** Added `fullscreen_exclusive` field to `Swapchain` initializers to allow the user to specify how fullscreen exclusivity should be handled.
    + Swapchain methods added: `Swapchain::acquire_fullscreen_exclusive()`, `Swapchain::release_fullscreen_exclusive()`, and `Swapchain::is_fullscreen_exclusive()`
- Add function `execute_commands_from_vec` to handle submission of multiple secondary command buffers.
- Allow `DebugCallback` to be sent between threads
- Pipeline barriers are now correctly inserted when a resource is used more than two times.
- Update MacOS dependency cocoa to 0.20
- Fixed code generated by `shader!` macro so that SSBO's are supported again (broken in 0.16.0).
- Added Swapchain::surface() - which returns the saved surface
- Propogate new lines correctly in shader compile errors.
- `Queue` and `QueueFamily` now implement `PartialEq` and `Eq`
- `Swapchain::acquire_next_image()`` now returns ``(image_id, suboptimal, aquire_future)``
    + *suboptimal indicates that the swapchain is usable, but should be recreated*
- Fixed Join Future implementation to not submit joined command buffers twice.
- The traits `GraphicsPipelineAbstract` and `DescriptorSet` now require `DeviceOwned`.
- Added `PartialEq`, `Eq` and `Hash` implementations to all types involved in a draw call, including:
  - `Instance`, `Device`, `GraphicsPipeline` and `dyn GraphicsPipelineAbstract`
  - `UnsafeBuffer` and all types implementing `BufferAccess`
  - `UnsafeImage`, `UnsafeImageView` and all types implementing `ImageAccess` or `ImageViewAccess`
  - All types implementing `DescriptorSet`

# Version 0.16.0 (2019-11-01)

- Fixed bug in examples causing OutOfHostMemory errors
- Replaced `VK_EXT_debug_report` `VK_EXT_debug_marker` with `VK_EXT_debug_utils`.
- Update MacOS dependencies metal to 0.17 and cocoa to 0.19
- Added dynamic stencil elements to `DynamicState`
- Fixed `ImageDimensions::mipmap_dimensions` and `max_mipmaps` in cases where the original size is not a power of two.
- Shader includes now work on Windows.
- **Breaking Change** Shader include directories passed to the `shader!` macro are now relative to the crates `Cargo.toml`
- Add support for `VK_KHR_16bit_storage` and `VK_KHR_storage_buffer_storage_class` device extensions.
- Update dependencies: lazy_static, half, syn, quote & proc-macro2
- Swapchain can now be recreated with dimensions of corresponding surface using `recreate()`.
- Added `raw_loaded_extensions()` to `Instance` to allow querying of all extensions, not just known ones.
- **Breaking Change** `loaded_extensions()` on `Instance` no longer returns a reference.
- Add support for GLSL macro defines to the `shader!` macro.
- Switch to Vulkan 1.1 and inherently SpirV 1.3 (shaderc default version for vulkan 1.1)

# Version 0.15.0 (2019-10-18)

*Yanked*

# Version 0.14.0 (2019-08-17)

- Update shaderc to 0.6. This again allows to use locally installed libraries which reduces the build-time significantly on Arch/Voidlinux (see https://github.com/google/shaderc-rs/issues/58)
- Removed faulty debug_assert in `SwapchainAcquireFuture::drop`.
- Compressed texture formats can now be uploaded using `ImmutableImage::from_iter`, `ImmutableImage::from_buffer`, `AutoCommandBuilder::copy_buffer_to_image_dimensions`,
  and `AutoCommandBuilder::copy_buffer_to_image`.

# Version 0.13.0 (2019-07-02)

This is an emergency breaking breaking change. It fixes Undefined Behaviour that was preventing the compilation of Vulkano on the latest rust nightlies.

- Structs that have the `impl_vertex` macro applied to them, now also need to `#[derive(Default)]`.

# Version 0.12.0 (2019-05-24)

- Update shaderc to 0.5.  New shaderc has improved pre-built options for libshaderc that significantly reduce package build time and are appropriate for use in CI
- `QueueFamily::explicitly_supports_tranfers` only returns true if `vk::QUEUE_TRANSFER_BIT` is set instead of also always returning true.  Removed `supports_transfers`.
- Update to winit 0.19
- Add support for `#include "..."` and `#include <...>` directives within source
  files.
- Add a `union` method for the extensions types.
- Make `BufferUsage` derive `PartialEq` and `Eq`
- Fix a compiler error in struct code generated by the `shader!` macro.
- Improve inference of image layout transitions and in turn image memory barrier
  behaviour.

# Version 0.11.1 (2018-11-16)

- Expose `CopyImageError` and `DrawIndexedIndirectError`.

# Version 0.11.0 (2018-11-08) (**yanked** because vk-sys 0.3.4 was accidentally breaking vulkano 0.10)

- Update to winit 0.18
- Export features and device extensions from the device module instead of the instance module
    +   `instance::Features` -> `device::Features`
    +   `instance::DeviceExtensions` -> `device::DeviceExtensions`
    +   `instance::RawDeviceExtensions` -> `device::RawDeviceExtensions`
- Added `vulkano_shaders::shader!` proc macro, use this instead of `vulkano_shader_deriver::VulkanoShaders`.
- The entire `vulkano_shader_derive` crate is deprecated.
- `vulkano_shaders::{reflect, compile, Error}` are no longer public.
- Remove vulkano_shaders::build_glsl_shaders
- Removed mir support, as it is being removed from the vulkan spec.
- Split `PersistentDescriptorSetError::MissingUsage` into `MissingImageUsage` and `MissingBufferUsage`
  each with a matching enum indicating the usage that was missing.
- Fix instance_count when using draw_index with instance buffers
- Added a `reinterpret` function to `BufferSlice`
- Made `AttributeInfo` derive `Copy`, `Clone` and `Debug`
- Use [google/shaderc](https://github.com/google/shaderc-rs) for shader compilation
- Reject generation of rust types for SPIR-V arrays that would have incorrect array stride.
- Removed the `Layout` prefix of the descriptions used for a render pass.
- Implemented DescriptorSetCollection for `Vec<T>` which allows easier use of construction them for usage when drawing.

# Version 0.10.0 (2018-08-10)

- Use dynamically loaded `libvulkan` like on other platforms instead of linking to MoltenVK on macOS
- Updated winit to version 0.17.
- Allow custom implementations of `RenderPassDesc` to specify `VK_SUBPASS_EXTERNAL` as a dependency source or destination
- Added `vulkano_win::create_vk_surface` which allows creating a surface safely without taking ownership of
  the window.
- `AutoCommandBufferBuilder::draw` and friends no longer consume the `DynamicState` argument, allowing reuse between calls.
- `Instance::new` and `Instance::with_loader` now take in the layers as an iterator of borrowed `str`s, not of references to
borrowed `str`s.

# Version 0.9.0 (2018-03-13)

- Updated winit to version 0.11.

# Version 0.8.0 (2018-03-11)

- Changed `ShaderInterfaceMismatchError` to be more verbose.
- Allow depth/stencil images to be used with `AutoCommandBufferBuilder::copy_image_to_buffer()`
- Allow `Surface` to own the window it references.
- Clear value validation for `AutoCommandBufferBuilder::begin_render_pass()`
- Fix occasional truncation of glslang_validator when glsl-to-spirv is rebuilt
- Fix linking against MoltenVK >= 0.19.0
- Fix panic on DeviceLost error after resizing swapchain on nvidia/amd hardware
- Added `AutoCommandBufferBuilder::copy_image`
- Added `VulkanObject::TYPE` to look up the `DebugReportObjectTypeEXT` of an object
- Added `Device::set_object_name` and `Device::set_object_name_raw`
- Added `GraphicsPipelineBuilder::with_auto_layout` to simplify use of dynamic buffers.

# Version 0.7.3 (2018-02-10)

*Yanked*

# Version 0.7.2 (2017-10-09)

- Allow `impl_vertex!` to support generic structs.
- Added `GraphicsPipelineBuilder::with_pipeline_layout`.
- Fixed creating a buffer view not checking the `min_texel_buffer_offset_alignment` limit.
- Added support for loading the `VK_EXT_debug_marker` extension and adding debug markers to
  `UnsafeCommandBufferBuilder`
- Changed `GraphicsPipelineBuilder` to default to a buffer-less vertex input.
- Deprecated `pipeline::input_assembly::InputAssembly`.

# Version 0.7.1 (2017-09-28)

- Fixed an HiDPI scaling issue on MacOS in vulkano-win.
- Fixed `le()` and `ge()` for `DescriptorsCount`, which also fixes a potential memory leak when
  allocating descriptor sets.
- Fixed the `ordered_passes_renderpass!` macro not working properly when using `resolve`.
- Added an optional `resolve` entry in `pass` for `single_pass_renderpass!`, for resolving
  multisampled attachments into non-multisampled attachments.
- Fixed the value of `rasterizationSamples` not corresponding to the render pass.
- Added `GraphicsPipelineBuilder::sample_shading_disabled`, `sample_shading_enabled`,
  `alpha_to_coverage_disabled`, `alpha_to_coverage_enabled`, `alpha_to_one_disabled` and
  `alpha_to_one_enabled`. Added `GraphicsPipelineCreationError::SampleRateShadingFeatureNotEnabled`
  and `AlphaToOneFeatureNotEnabled`.
- Deprecated `pipeline::multisample::Multisample`.

# Version 0.7.0 (2017-09-21)

- Added `RuntimePipelineDesc`, an implementation of `PipelineLayoutDesc` that makes creating custom
  layouts easier.
- Changed `CpuBufferPool::next()` and `chunk()` to return a `Result` in case of an error when
  allocating or mapping memory.
- Changed `CommandBufferExecError::AccessError` to provide a hint of where the error occurs.
- Added `vulkano::pipeline::vertex::BufferlessDefinition` and `BufferlessVertices` to enable
  bufferless drawing.
- Changed `ImageAccess::try_gpu_lock` and `unlock()` to verify whether the image layout is correct,
  especially at the first usage of an image.
- Changed `BufferAccess::conflict_*` and `ImageAccess::conflict_*` to forbid querying a specific
  range of the resource.
- Fixed `layers` argument validation in `Swapchain::new_inner`.
- Added a 32-bit word constructor for `ShaderModule` (`ShaderModule::from_words`).
- Changed the various `is_superset_of` functions to return a `Result` instead of a `bool`.

# Version 0.6.2 (2017-09-06)

- Added checking compatibility between the descriptor sets and the pipeline object when adding a
  draw or compute command.
- Fixed several bugs in the validation checks of `blit_image`.
- Fixed `blit_image`, `copy_buffer_to_image` and `copy_image_to_buffer` not taking the mipmap level
  into account when checking for the correct image dimensions.
- Added `ImageDimensions::mipmap_dimensions()` and `ImageDimensions::num_mipmaps()`.
- Added an implementation of `DeviceOwned` for `QueuesIter` and `Queue`.
- Fixed the standard command pool and descriptor pool being destroyed too often.
- Added `#[allow(non_snake_case)]` to structs generated by vulkano-shaders.
