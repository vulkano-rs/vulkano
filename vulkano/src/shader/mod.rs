// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A program that is run on the device.
//!
//! In Vulkan, shaders are grouped in *shader modules*. Each shader module is built from SPIR-V
//! code and can contain one or more entry points. Note that for the moment the official
//! GLSL-to-SPIR-V compiler does not support multiple entry points.
//!
//! The vulkano library can parse and introspect SPIR-V code, but it does not fully validate the
//! code. You are encouraged to use the `vulkano-shaders` crate that will generate Rust code that
//! wraps around vulkano's shaders API.
//!
//! # Shader interface
//!
//! Vulkan has specific rules for interfacing shaders with each other, and with other parts
//! of a program.
//!
//! ## Endianness
//!
//! The Vulkan specification requires that a Vulkan implementation has runtime support for the
//! types [`u8`], [`u16`], [`u32`], [`u64`] as well as their signed versions, as well as [`f32`]
//! and [`f64`] on the host, and that the representation and endianness of these types matches
//! those on the device. This means that if you have for example a `Subbuffer<u32>`, you can be
//! sure that it is represented the same way on the host as it is on the device, and you don't need
//! to worry about converting the endianness.
//!
//! ## Layout of data
//!
//! When buffers, push constants or other user-provided data are accessed in shaders,
//! the shader expects the values inside to be laid out in a specific way. For every uniform buffer,
//! storage buffer or push constant block, the SPIR-V specification requires the SPIR-V code to
//! provide the `Offset` decoration for every member of a struct, indicating where it is placed
//! relative to the start of the struct. If there are arrays or matrices among the variables, the
//! SPIR-V code must also provide an `ArrayStride` or `MatrixStride` decoration for them,
//! indicating the number of bytes between the start of each element in the array or column in the
//! matrix. When providing data to shaders, you must make sure that your data is placed at the
//! locations indicated within the SPIR-V code, or the shader will read the wrong data and produce
//! nonsense.
//!
//! GLSL does not require you to give explicit offsets and/or strides to your variables (although
//! it has the option to provide them if you wish). Instead, the shader compiler automatically
//! assigns every variable an offset, increasing in the order you declare them in.
//! To know the exact offsets that will be used, so that you can lay out your data appropriately,
//! you must know the alignment rules that the shader compiler uses. The shader compiler will
//! always give a variable the smallest offset that fits the alignment rules and doesn't overlap
//! with the previous variable. The shader compiler uses default alignment rules depending on the
//! type of block, but you can specify another layout by using the `layout` qualifier.
//!
//! ## Alignment rules
//!
//! The offset of each variable from the start of a block, matrix or array must be a
//! multiple of a certain number, which is called its *alignment*. The stride of an array or matrix
//! must likewise be a multiple of this number. An alignment is always a power-of-two value.
//! Regardless of whether the offset/stride is provided manually in the compiled SPIR-V code,
//! or assigned automatically by the shader compiler, all variable offsets/strides in a shader must
//! follow these alignment rules.
//!
//! Three sets of [alignment rules] are supported by Vulkan. Each one has a GLSL qualifier that
//! you can place in front of a block, to make the shader compiler use that layout for the block.
//! If you don't provide this qualifier, it will use a default alignment.
//!
//! - **Scalar alignment** (GLSL qualifier: `layout(scalar)`, requires the
//!   [`GL_EXT_scalar_block_layout`] GLSL extension). This is the same as the C alignment,
//!   expressed in Rust with the
//!   [`#[repr(C)]`](https://doc.rust-lang.org/nomicon/other-reprs.html#reprc) attribute.
//!   The shader compiler does not use this alignment by default, so you must use the GLSL
//!   qualifier. You must also enable the [`scalar_block_layout`] feature in Vulkan.
//! - **Base alignment**, also known as **std430** (GLSL qualifier: `layout(std430)`).
//!   The shader compiler uses this alignment by default for all shader data except uniform buffers.
//!   If you use the base alignment for a uniform buffer, you must also enable the
//!   [`uniform_buffer_standard_layout`] feature in Vulkan.
//! - **Extended alignment**, also known as **std140** (GLSL qualifier: `layout(std140)`).
//!   The shader compiler uses this alignment by default for uniform buffers.
//!
//! Each alignment type is a subset of the ones above it, so if something adheres to the extended
//! alignment rules, it also follows the rules for the base and scalar alignments.
//!
//! In all three of these alignment rules, a primitive/scalar value with a size of N bytes has an
//! alignment of N, meaning that it must have an offset that is a multiple of its size,
//! like in C or Rust. For example, a `float` (like a Rust `f32`) has a size of 4 bytes,
//! and an alignment of 4.
//!
//! The differences between the alignment rules are in how compound types (vectors, matrices,
//! arrays and structs) are expected to be laid out. For a compound type with an element whose
//! alignment is N, the scalar alignment considers the alignment of the compound type to be also N.
//! However, the base and extended alignments are stricter:
//!
//! | GLSL type | Scalar          | Base            | Extended                 |
//! |-----------|-----------------|-----------------|--------------------------|
//! | primitive | N               | N               | N                        |
//! | `vec2`    | N               | N * 2           | N * 2                    |
//! | `vec3`    | N               | N * 4           | N * 4                    |
//! | `vec4`    | N               | N * 4           | N * 4                    |
//! | array     | N               | N               | max(N, 16)               |
//! | `struct`  | N<sub>max</sub> | N<sub>max</sub> | max(N<sub>max</sub>, 16) |
//!
//! In the base and extended alignment, the alignment of a vector is the size of the whole vector,
//! rather than the size of its individual elements as is the case in the scalar alignment.
//! But note that, because alignment must be a power of two, the alignment of `vec3` cannot be
//! N * 3; it must be N * 4, the same alignment as `vec4`. This means that it is not possible to
//! tightly pack multiple `vec3` values (e.g. in an array); there will always be empty padding
//! between them.
//!
//! In both the scalar and base alignment, the alignment of arrays and their elements is equal to
//! the alignment of the contained type. In the extended alignment, however, the alignment is
//! always at least 16 (the size of a `vec4`). Therefore, the minimum stride of the array can be
//! much greater than the element size. For example, in an array of `float`, the stride must be at
//! least 16, even though a `float` itself is only 4 bytes in size. Every `float` element will be
//! followed by at least 12 bytes of unused space.
//!
//! A matrix `matCxR` is considered equivalent to an array of column vectors `vecR[C]`.
//! In the base and extended alignments, that means that if the matrix has 3 rows, there will be
//! one element's worth of padding between the column vectors. In the extended alignment,
//! the alignment is also at least 16, further increasing the amount of padding between the
//! column vectors.
//!
//! The rules for `struct`s are similar to those of arrays. When the members of the struct have
//! different alignment requirements, the alignment of the struct as a whole is the maximum
//! of the alignments of its members. As with arrays, in the extended alignment, the alignment
//! of a struct is at least 16.
//!
//! [alignment rules]: <https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-layout>
//! [`GL_EXT_scalar_block_layout`]: <https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_scalar_block_layout.txt>
//! [`scalar_block_layout`]: crate::device::Features::scalar_block_layout
//! [`uniform_buffer_standard_layout`]: crate::device::Features::uniform_buffer_standard_layout

use crate::{
    descriptor_set::layout::DescriptorType,
    device::{Device, DeviceOwned},
    format::{Format, NumericType},
    image::view::ImageViewType,
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags_enum},
    pipeline::{graphics::input_assembly::PrimitiveTopology, layout::PushConstantRange},
    shader::spirv::{Capability, Spirv},
    sync::PipelineStages,
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ahash::{HashMap, HashSet};
use bytemuck::bytes_of;
use half::f16;
use spirv::ExecutionModel;
use std::{
    borrow::Cow,
    collections::hash_map::Entry,
    mem::{discriminant, size_of_val, MaybeUninit},
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

pub mod reflect;
pub mod spirv;

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/spirv_reqs.rs"));

/// Contains SPIR-V code with one or more entry points.
#[derive(Debug)]
pub struct ShaderModule {
    handle: ash::vk::ShaderModule,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,
    entry_point_map: HashMap<String, HashMap<ExecutionModel, usize>>,
    entry_point_infos: Vec<EntryPointInfo>,
}

impl ShaderModule {
    /// Creates a new shader module.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code in `create_info.code` must be valid.
    #[inline]
    pub unsafe fn new(
        device: Arc<Device>,
        create_info: ShaderModuleCreateInfo<'_>,
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        let spirv = Spirv::new(create_info.code).map_err(|err| {
            Box::new(ValidationError {
                context: "create_info.code".into(),
                problem: format!("error while parsing: {}", err).into(),
                ..Default::default()
            })
        })?;

        Self::new_with_data(
            device,
            create_info,
            reflect::entry_points(&spirv),
            spirv.version(),
            reflect::spirv_capabilities(&spirv),
            reflect::spirv_extensions(&spirv),
        )
    }

    // This is public only for vulkano-shaders, do not use otherwise.
    #[doc(hidden)]
    pub unsafe fn new_with_data<'a>(
        device: Arc<Device>,
        create_info: ShaderModuleCreateInfo<'_>,
        entry_points: impl IntoIterator<Item = EntryPointInfo>,
        spirv_version: Version,
        spirv_capabilities: impl IntoIterator<Item = &'a Capability>,
        spirv_extensions: impl IntoIterator<Item = &'a str>,
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        Self::validate_new(
            &device,
            &create_info,
            spirv_version,
            spirv_capabilities,
            spirv_extensions,
        )?;

        Ok(Self::new_with_data_unchecked(
            device,
            create_info,
            entry_points,
        )?)
    }

    fn validate_new<'a>(
        device: &Device,
        create_info: &ShaderModuleCreateInfo<'_>,
        spirv_version: Version,
        spirv_capabilities: impl IntoIterator<Item = &'a Capability>,
        spirv_extensions: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device, spirv_version, spirv_capabilities, spirv_extensions)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: ShaderModuleCreateInfo<'_>,
    ) -> Result<Arc<ShaderModule>, VulkanError> {
        let spirv = Spirv::new(create_info.code).unwrap();

        Self::new_with_data_unchecked(device, create_info, reflect::entry_points(&spirv))
    }

    unsafe fn new_with_data_unchecked(
        device: Arc<Device>,
        create_info: ShaderModuleCreateInfo<'_>,
        entry_points: impl IntoIterator<Item = EntryPointInfo>,
    ) -> Result<Arc<ShaderModule>, VulkanError> {
        let &ShaderModuleCreateInfo { code, _ne: _ } = &create_info;

        let handle = {
            let infos = ash::vk::ShaderModuleCreateInfo {
                flags: ash::vk::ShaderModuleCreateFlags::empty(),
                code_size: size_of_val(code),
                p_code: code.as_ptr(),
                ..Default::default()
            };

            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_shader_module)(
                device.handle(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle_with_data(
            device,
            handle,
            create_info,
            entry_points,
        ))
    }

    /// Creates a new `ShaderModule` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::ShaderModule,
        create_info: ShaderModuleCreateInfo<'_>,
    ) -> Arc<ShaderModule> {
        let spirv = Spirv::new(create_info.code).unwrap();

        Self::from_handle_with_data(device, handle, create_info, reflect::entry_points(&spirv))
    }

    unsafe fn from_handle_with_data(
        device: Arc<Device>,
        handle: ash::vk::ShaderModule,
        create_info: ShaderModuleCreateInfo<'_>,
        entry_points: impl IntoIterator<Item = EntryPointInfo>,
    ) -> Arc<ShaderModule> {
        let ShaderModuleCreateInfo { code: _, _ne: _ } = create_info;

        let mut entry_point_map: HashMap<String, HashMap<ExecutionModel, usize>> =
            HashMap::default();
        let entry_point_infos: Vec<_> = entry_points
            .into_iter()
            .enumerate()
            .map(|(index, info)| {
                entry_point_map
                    .entry(info.name.clone())
                    .or_default()
                    .insert(ExecutionModel::from(&info.execution), index);
                info
            })
            .collect();

        Arc::new(ShaderModule {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),
            entry_point_map,
            entry_point_infos,
        })
    }

    /// Builds a new shader module from SPIR-V 32-bit words. The shader code is parsed and the
    /// necessary information is extracted from it.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated beyond the minimum needed to extract the information.
    #[deprecated(since = "0.34.0", note = "use `new` instead")]
    #[inline]
    pub unsafe fn from_words(
        device: Arc<Device>,
        words: &[u32],
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        Self::new(device, ShaderModuleCreateInfo::new(words))
    }

    /// As `from_words`, but takes a slice of bytes.
    ///
    /// # Panics
    ///
    /// - Panics if `bytes` is not aligned to 4.
    /// - Panics if the length of `bytes` is not a multiple of 4.
    #[deprecated(
        since = "0.34.0",
        note = "use `shader::spirv::bytes_to_words`, and then use `new` instead"
    )]
    #[inline]
    pub unsafe fn from_bytes(
        device: Arc<Device>,
        bytes: &[u8],
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        let words = spirv::bytes_to_words(bytes).unwrap();
        Self::new(device, ShaderModuleCreateInfo::new(&words))
    }

    /// Returns information about the entry point with the provided name. Returns `None` if no entry
    /// point with that name exists in the shader module or if multiple entry points with the same
    /// name exist.
    #[inline]
    pub fn entry_point(self: &Arc<Self>, name: &str) -> Option<EntryPoint> {
        self.entry_point_map.get(name).and_then(|infos| {
            if infos.len() == 1 {
                infos.iter().next().map(|(_, &info_index)| EntryPoint {
                    module: self.clone(),
                    info_index,
                })
            } else {
                None
            }
        })
    }

    /// Returns information about the entry point with the provided name and execution model.
    /// Returns `None` if no entry and execution model exists in the shader module.
    #[inline]
    pub fn entry_point_with_execution(
        self: &Arc<Self>,
        name: &str,
        execution: ExecutionModel,
    ) -> Option<EntryPoint> {
        self.entry_point_map.get(name).and_then(|infos| {
            infos.get(&execution).map(|&info_index| EntryPoint {
                module: self.clone(),
                info_index,
            })
        })
    }
}

impl Drop for ShaderModule {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_shader_module)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for ShaderModule {
    type Handle = ash::vk::ShaderModule;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for ShaderModule {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(ShaderModule);

pub struct ShaderModuleCreateInfo<'a> {
    /// The SPIR-V code, in the form of 32-bit words.
    ///
    /// There is no default value.
    pub code: &'a [u32],

    pub _ne: crate::NonExhaustive,
}

impl<'a> ShaderModuleCreateInfo<'a> {
    /// Returns a `ShaderModuleCreateInfo` with the specified `code`.
    #[inline]
    pub fn new(code: &'a [u32]) -> Self {
        Self {
            code,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate<'b>(
        &self,
        device: &Device,
        mut spirv_version: Version,
        spirv_capabilities: impl IntoIterator<Item = &'b Capability>,
        spirv_extensions: impl IntoIterator<Item = &'b str>,
    ) -> Result<(), Box<ValidationError>> {
        let &Self { code, _ne: _ } = self;

        if code.is_empty() {
            return Err(Box::new(ValidationError {
                context: "code".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkShaderModuleCreateInfo-codeSize-01085"],
                ..Default::default()
            }));
        }

        {
            spirv_version.patch = 0; // Ignore the patch version

            match spirv_version {
                Version::V1_0 => None,
                Version::V1_1 | Version::V1_2 | Version::V1_3 => {
                    (!(device.api_version() >= Version::V1_1)).then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    ]))
                }
                Version::V1_4 => (!(device.api_version() >= Version::V1_2
                    || device.enabled_extensions().khr_spirv_1_4))
                    .then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_spirv_1_4")]),
                    ])),
                Version::V1_5 => {
                    (!(device.api_version() >= Version::V1_2)).then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                    ]))
                }
                Version::V1_6 => {
                    (!(device.api_version() >= Version::V1_3)).then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    ]))
                }
                _ => {
                    return Err(Box::new(ValidationError {
                        context: "code".into(),
                        problem: format!(
                            "uses SPIR-V version {}.{}, which is not supported by Vulkan",
                            spirv_version.major, spirv_version.minor
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
        }
        .map_or(Ok(()), |requires_one_of| {
            Err(Box::new(ValidationError {
                context: "code".into(),
                problem: format!(
                    "uses SPIR-V version {}.{}",
                    spirv_version.major, spirv_version.minor
                )
                .into(),
                requires_one_of,
                ..Default::default()
            }))
        })?;

        for &capability in spirv_capabilities {
            validate_spirv_capability(device, capability).map_err(|err| err.add_context("code"))?;
        }

        for extension in spirv_extensions {
            validate_spirv_extension(device, extension).map_err(|err| err.add_context("code"))?;
        }

        // VUID-VkShaderModuleCreateInfo-pCode-08736
        // VUID-VkShaderModuleCreateInfo-pCode-08737
        // VUID-VkShaderModuleCreateInfo-pCode-08738
        // Unsafe

        Ok(())
    }
}

/// The information associated with a single entry point in a shader.
#[derive(Clone, Debug)]
pub struct EntryPointInfo {
    pub name: String,
    pub execution: ShaderExecution,
    pub descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    pub push_constant_requirements: Option<PushConstantRange>,
    pub specialization_constants: HashMap<u32, SpecializationConstant>,
    pub input_interface: ShaderInterface,
    pub output_interface: ShaderInterface,
}

/// Represents a shader entry point in a shader module.
///
/// Can be obtained by calling [`entry_point`](ShaderModule::entry_point) on the shader module.
#[derive(Clone, Debug)]
pub struct EntryPoint {
    module: Arc<ShaderModule>,
    info_index: usize,
}

impl EntryPoint {
    /// Returns the module this entry point comes from.
    #[inline]
    pub fn module(&self) -> &Arc<ShaderModule> {
        &self.module
    }

    /// Returns information about the entry point.
    #[inline]
    pub fn info(&self) -> &EntryPointInfo {
        &self.module.entry_point_infos[self.info_index]
    }
}

/// The mode in which a shader executes. This includes both information about the shader type/stage,
/// and additional data relevant to particular shader types.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderExecution {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry(GeometryShaderExecution),
    Fragment(FragmentShaderExecution),
    Compute,
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,
    Task,
    Mesh,
    SubpassShading,
}

impl From<&ShaderExecution> for ExecutionModel {
    fn from(value: &ShaderExecution) -> Self {
        match value {
            ShaderExecution::Vertex => Self::Vertex,
            ShaderExecution::TessellationControl => Self::TessellationControl,
            ShaderExecution::TessellationEvaluation => Self::TessellationEvaluation,
            ShaderExecution::Geometry(_) => Self::Geometry,
            ShaderExecution::Fragment(_) => Self::Fragment,
            ShaderExecution::Compute => Self::GLCompute,
            ShaderExecution::RayGeneration => Self::RayGenerationKHR,
            ShaderExecution::AnyHit => Self::AnyHitKHR,
            ShaderExecution::ClosestHit => Self::ClosestHitKHR,
            ShaderExecution::Miss => Self::MissKHR,
            ShaderExecution::Intersection => Self::IntersectionKHR,
            ShaderExecution::Callable => Self::CallableKHR,
            ShaderExecution::Task => Self::TaskNV,
            ShaderExecution::Mesh => Self::MeshNV,
            ShaderExecution::SubpassShading => todo!(),
        }
    }
}

/*#[derive(Clone, Copy, Debug)]
pub struct TessellationShaderExecution {
    pub num_output_vertices: u32,
    pub point_mode: bool,
    pub subdivision: TessellationShaderSubdivision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TessellationShaderSubdivision {
    Triangles,
    Quads,
    Isolines,
}*/

/// The mode in which a geometry shader executes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryShaderExecution {
    pub input: GeometryShaderInput,
    /*pub max_output_vertices: u32,
    pub num_invocations: u32,
    pub output: GeometryShaderOutput,*/
}

/// The input primitive type that is expected by a geometry shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryShaderInput {
    Points,
    Lines,
    LinesWithAdjacency,
    Triangles,
    TrianglesWithAdjacency,
}

impl GeometryShaderInput {
    /// Returns true if the given primitive topology can be used as input for this geometry shader.
    #[inline]
    pub fn is_compatible_with(self, topology: PrimitiveTopology) -> bool {
        match self {
            Self::Points => matches!(topology, PrimitiveTopology::PointList),
            Self::Lines => matches!(
                topology,
                PrimitiveTopology::LineList | PrimitiveTopology::LineStrip
            ),
            Self::LinesWithAdjacency => matches!(
                topology,
                PrimitiveTopology::LineListWithAdjacency
                    | PrimitiveTopology::LineStripWithAdjacency
            ),
            Self::Triangles => matches!(
                topology,
                PrimitiveTopology::TriangleList
                    | PrimitiveTopology::TriangleStrip
                    | PrimitiveTopology::TriangleFan,
            ),
            Self::TrianglesWithAdjacency => matches!(
                topology,
                PrimitiveTopology::TriangleListWithAdjacency
                    | PrimitiveTopology::TriangleStripWithAdjacency,
            ),
        }
    }
}

/*#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryShaderOutput {
    Points,
    LineStrip,
    TriangleStrip,
}*/

/// The mode in which a fragment shader executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentShaderExecution {
    pub fragment_tests_stages: FragmentTestsStages,
}

/// The fragment tests stages that will be executed in a fragment shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FragmentTestsStages {
    Early,
    Late,
    EarlyAndLate,
}

/// The requirements imposed by a shader on a binding within a descriptor set layout, and on any
/// resource that is bound to that binding.
#[derive(Clone, Debug, Default)]
pub struct DescriptorBindingRequirements {
    /// The descriptor types that are allowed.
    pub descriptor_types: Vec<DescriptorType>,

    /// The number of descriptors (array elements) that the shader requires. The descriptor set
    /// layout can declare more than this, but never less.
    ///
    /// `None` means that the shader declares this as a runtime-sized array, and could potentially
    /// access every array element provided in the descriptor set.
    pub descriptor_count: Option<u32>,

    /// The image format that is required for image views bound to this binding. If this is
    /// `None`, then any image format is allowed.
    pub image_format: Option<Format>,

    /// Whether image views bound to this binding must have multisampling enabled or disabled.
    pub image_multisampled: bool,

    /// The base scalar type required for the format of image views bound to this binding.
    /// This is `None` for non-image bindings.
    pub image_scalar_type: Option<NumericType>,

    /// The view type that is required for image views bound to this binding.
    /// This is `None` for non-image bindings.
    pub image_view_type: Option<ImageViewType>,

    /// The shader stages that the binding must be declared for.
    pub stages: ShaderStages,

    /// The requirements for individual descriptors within a binding.
    ///
    /// Keys with `Some` hold requirements for a specific descriptor index, if it is statically
    /// known in the shader (a constant). The key `None` holds requirements for indices that are
    /// not statically known, but determined only at runtime (calculated from an input variable).
    pub descriptors: HashMap<Option<u32>, DescriptorRequirements>,
}

/// The requirements imposed by a shader on resources bound to a descriptor.
#[derive(Clone, Debug, Default)]
pub struct DescriptorRequirements {
    /// For buffers and images, which shader stages perform read operations.
    pub memory_read: ShaderStages,

    /// For buffers and images, which shader stages perform write operations.
    pub memory_write: ShaderStages,

    /// For sampler bindings, whether the shader performs depth comparison operations.
    pub sampler_compare: bool,

    /// For sampler bindings, whether the shader performs sampling operations that are not
    /// permitted with unnormalized coordinates. This includes sampling with `ImplicitLod`,
    /// `Dref` or `Proj` SPIR-V instructions or with an LOD bias or offset.
    pub sampler_no_unnormalized_coordinates: bool,

    /// For sampler bindings, whether the shader performs sampling operations that are not
    /// permitted with a sampler YCbCr conversion. This includes sampling with `Gather` SPIR-V
    /// instructions or with an offset.
    pub sampler_no_ycbcr_conversion: bool,

    /// For sampler bindings, the sampled image descriptors that are used in combination with this
    /// sampler.
    pub sampler_with_images: HashSet<DescriptorIdentifier>,

    /// For storage image bindings, whether the shader performs atomic operations.
    pub storage_image_atomic: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DescriptorIdentifier {
    pub set: u32,
    pub binding: u32,
    pub index: u32,
}

impl DescriptorBindingRequirements {
    /// Merges `other` into `self`, so that `self` satisfies the requirements of both.
    /// An error is returned if the requirements conflict.
    #[inline]
    pub fn merge(&mut self, other: &Self) -> Result<(), Box<ValidationError>> {
        let Self {
            descriptor_types,
            descriptor_count,
            image_format,
            image_multisampled,
            image_scalar_type,
            image_view_type,
            stages,
            descriptors,
        } = self;

        /* Checks */

        if !descriptor_types
            .iter()
            .any(|ty| other.descriptor_types.contains(ty))
        {
            return Err(Box::new(ValidationError {
                problem: "the allowed descriptor types of the two descriptors do not overlap"
                    .into(),
                ..Default::default()
            }));
        }

        if let (Some(first), Some(second)) = (*image_format, other.image_format) {
            if first != second {
                return Err(Box::new(ValidationError {
                    problem: "the descriptors require different formats".into(),
                    ..Default::default()
                }));
            }
        }

        if let (Some(first), Some(second)) = (*image_scalar_type, other.image_scalar_type) {
            if first != second {
                return Err(Box::new(ValidationError {
                    problem: "the descriptors require different scalar types".into(),
                    ..Default::default()
                }));
            }
        }

        if let (Some(first), Some(second)) = (*image_view_type, other.image_view_type) {
            if first != second {
                return Err(Box::new(ValidationError {
                    problem: "the descriptors require different image view types".into(),
                    ..Default::default()
                }));
            }
        }

        if *image_multisampled != other.image_multisampled {
            return Err(Box::new(ValidationError {
                problem: "the multisampling requirements of the descriptors differ".into(),
                ..Default::default()
            }));
        }

        /* Merge */

        descriptor_types.retain(|ty| other.descriptor_types.contains(ty));

        *descriptor_count = (*descriptor_count).max(other.descriptor_count);
        *image_format = image_format.or(other.image_format);
        *image_scalar_type = image_scalar_type.or(other.image_scalar_type);
        *image_view_type = image_view_type.or(other.image_view_type);
        *stages |= other.stages;

        for (&index, other) in &other.descriptors {
            match descriptors.entry(index) {
                Entry::Vacant(entry) => {
                    entry.insert(other.clone());
                }
                Entry::Occupied(entry) => {
                    entry.into_mut().merge(other);
                }
            }
        }

        Ok(())
    }
}

impl DescriptorRequirements {
    /// Merges `other` into `self`, so that `self` satisfies the requirements of both.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        let Self {
            memory_read,
            memory_write,
            sampler_compare,
            sampler_no_unnormalized_coordinates,
            sampler_no_ycbcr_conversion,
            sampler_with_images,
            storage_image_atomic,
        } = self;

        *memory_read |= other.memory_read;
        *memory_write |= other.memory_write;
        *sampler_compare |= other.sampler_compare;
        *sampler_no_unnormalized_coordinates |= other.sampler_no_unnormalized_coordinates;
        *sampler_no_ycbcr_conversion |= other.sampler_no_ycbcr_conversion;
        sampler_with_images.extend(&other.sampler_with_images);
        *storage_image_atomic |= other.storage_image_atomic;
    }
}

/// The value to provide for a specialization constant, when creating a pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecializationConstant {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(f16),
    F32(f32),
    F64(f64),
}

impl SpecializationConstant {
    /// Returns the value as a byte slice. Booleans are expanded to a `VkBool32` value.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Bool(false) => bytes_of(&ash::vk::FALSE),
            Self::Bool(true) => bytes_of(&ash::vk::TRUE),
            Self::I8(value) => bytes_of(value),
            Self::I16(value) => bytes_of(value),
            Self::I32(value) => bytes_of(value),
            Self::I64(value) => bytes_of(value),
            Self::U8(value) => bytes_of(value),
            Self::U16(value) => bytes_of(value),
            Self::U32(value) => bytes_of(value),
            Self::U64(value) => bytes_of(value),
            Self::F16(value) => bytes_of(value),
            Self::F32(value) => bytes_of(value),
            Self::F64(value) => bytes_of(value),
        }
    }

    /// Returns whether `self` and `other` have the same type, ignoring the value.
    #[inline]
    pub fn eq_type(&self, other: &Self) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl From<bool> for SpecializationConstant {
    #[inline]
    fn from(value: bool) -> Self {
        SpecializationConstant::Bool(value)
    }
}

impl From<i8> for SpecializationConstant {
    #[inline]
    fn from(value: i8) -> Self {
        SpecializationConstant::I8(value)
    }
}

impl From<i16> for SpecializationConstant {
    #[inline]
    fn from(value: i16) -> Self {
        SpecializationConstant::I16(value)
    }
}

impl From<i32> for SpecializationConstant {
    #[inline]
    fn from(value: i32) -> Self {
        SpecializationConstant::I32(value)
    }
}

impl From<i64> for SpecializationConstant {
    #[inline]
    fn from(value: i64) -> Self {
        SpecializationConstant::I64(value)
    }
}

impl From<u8> for SpecializationConstant {
    #[inline]
    fn from(value: u8) -> Self {
        SpecializationConstant::U8(value)
    }
}

impl From<u16> for SpecializationConstant {
    #[inline]
    fn from(value: u16) -> Self {
        SpecializationConstant::U16(value)
    }
}

impl From<u32> for SpecializationConstant {
    #[inline]
    fn from(value: u32) -> Self {
        SpecializationConstant::U32(value)
    }
}

impl From<u64> for SpecializationConstant {
    #[inline]
    fn from(value: u64) -> Self {
        SpecializationConstant::U64(value)
    }
}

impl From<f16> for SpecializationConstant {
    #[inline]
    fn from(value: f16) -> Self {
        SpecializationConstant::F16(value)
    }
}

impl From<f32> for SpecializationConstant {
    #[inline]
    fn from(value: f32) -> Self {
        SpecializationConstant::F32(value)
    }
}

impl From<f64> for SpecializationConstant {
    #[inline]
    fn from(value: f64) -> Self {
        SpecializationConstant::F64(value)
    }
}

/// Type that contains the definition of an interface between two shader stages, or between
/// the outside and a shader stage.
#[derive(Clone, Debug)]
pub struct ShaderInterface {
    elements: Vec<ShaderInterfaceEntry>,
}

impl ShaderInterface {
    /// Constructs a new `ShaderInterface`.
    ///
    /// # Safety
    ///
    /// - Must only provide one entry per location.
    /// - The format of each element must not be larger than 128 bits.
    // TODO: 4x64 bit formats are possible, but they require special handling.
    // TODO: could this be made safe?
    #[inline]
    pub unsafe fn new_unchecked(elements: Vec<ShaderInterfaceEntry>) -> ShaderInterface {
        ShaderInterface { elements }
    }

    /// Creates a description of an empty shader interface.
    #[inline]
    pub const fn empty() -> ShaderInterface {
        ShaderInterface {
            elements: Vec::new(),
        }
    }

    /// Returns a slice containing the elements of the interface.
    #[inline]
    pub fn elements(&self) -> &[ShaderInterfaceEntry] {
        self.elements.as_ref()
    }

    /// Checks whether the interface is potentially compatible with another one.
    ///
    /// Returns `Ok` if the two interfaces are compatible.
    #[inline]
    pub fn matches(&self, other: &ShaderInterface) -> Result<(), Box<ValidationError>> {
        if self.elements().len() != other.elements().len() {
            return Err(Box::new(ValidationError {
                problem: "the number of elements in the shader interfaces are not equal".into(),
                ..Default::default()
            }));
        }

        for a in self.elements() {
            let location_range = a.location..a.location + a.ty.num_locations();
            for loc in location_range {
                let b = match other
                    .elements()
                    .iter()
                    .find(|e| loc >= e.location && loc < e.location + e.ty.num_locations())
                {
                    None => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the second shader is missing an interface element at location {}",
                                loc
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }
                    Some(b) => b,
                };

                if a.ty != b.ty {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "the interface element at location {} does not have the same type \
                            in both shaders",
                            loc
                        )
                        .into(),
                        ..Default::default()
                    }));
                }

                // TODO: enforce this?
                /*match (a.name, b.name) {
                    (Some(ref an), Some(ref bn)) => if an != bn { return false },
                    _ => ()
                };*/
            }
        }

        // NOTE: since we check that the number of elements is the same, we don't need to iterate
        // over b's elements.

        Ok(())
    }
}

/// Entry of a shader interface definition.
#[derive(Debug, Clone)]
pub struct ShaderInterfaceEntry {
    /// The location slot that the variable starts at.
    pub location: u32,

    /// The component slot that the variable starts at. Must be in the range 0..=3.
    pub component: u32,

    /// Name of the element, or `None` if the name is unknown.
    pub name: Option<Cow<'static, str>>,

    /// The type of the variable.
    pub ty: ShaderInterfaceEntryType,
}

/// The type of a variable in a shader interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderInterfaceEntryType {
    /// The base numeric type.
    pub base_type: NumericType,

    /// The number of vector components. Must be in the range 1..=4.
    pub num_components: u32,

    /// The number of array elements or matrix columns.
    pub num_elements: u32,

    /// Whether the base type is 64 bits wide. If true, each item of the base type takes up two
    /// component slots instead of one.
    pub is_64bit: bool,
}

impl ShaderInterfaceEntryType {
    pub(crate) fn num_locations(&self) -> u32 {
        assert!(!self.is_64bit); // TODO: implement
        self.num_elements
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`ShaderStage`] values.
    ShaderStages impl {
        /// Creates a `ShaderStages` struct with all graphics stages set to `true`.
        #[inline]
        pub const fn all_graphics() -> ShaderStages {
            ShaderStages::VERTEX
                .union(ShaderStages::TESSELLATION_CONTROL)
                .union(ShaderStages::TESSELLATION_EVALUATION)
                .union(ShaderStages::GEOMETRY)
                .union(ShaderStages::FRAGMENT)
        }
    },

    /// A shader stage within a pipeline.
    ShaderStage,

    = ShaderStageFlags(u32);

    // TODO: document
    VERTEX, Vertex = VERTEX,

    // TODO: document
    TESSELLATION_CONTROL, TessellationControl = TESSELLATION_CONTROL,

    // TODO: document
    TESSELLATION_EVALUATION, TessellationEvaluation = TESSELLATION_EVALUATION,

    // TODO: document
    GEOMETRY, Geometry = GEOMETRY,

    // TODO: document
    FRAGMENT, Fragment = FRAGMENT,

    // TODO: document
    COMPUTE, Compute = COMPUTE,

    // TODO: document
    RAYGEN, Raygen = RAYGEN_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    ANY_HIT, AnyHit = ANY_HIT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    CLOSEST_HIT, ClosestHit = CLOSEST_HIT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    MISS, Miss = MISS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    INTERSECTION, Intersection = INTERSECTION_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    CALLABLE, Callable = CALLABLE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    TASK, Task = TASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    // TODO: document
    MESH, Mesh = MESH_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    // TODO: document
    SUBPASS_SHADING, SubpassShading = SUBPASS_SHADING_HUAWEI
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(huawei_subpass_shading)]),
    ]),
}

impl From<&ShaderExecution> for ShaderStage {
    #[inline]
    fn from(value: &ShaderExecution) -> Self {
        match value {
            ShaderExecution::Vertex => Self::Vertex,
            ShaderExecution::TessellationControl => Self::TessellationControl,
            ShaderExecution::TessellationEvaluation => Self::TessellationEvaluation,
            ShaderExecution::Geometry(_) => Self::Geometry,
            ShaderExecution::Fragment(_) => Self::Fragment,
            ShaderExecution::Compute => Self::Compute,
            ShaderExecution::RayGeneration => Self::Raygen,
            ShaderExecution::AnyHit => Self::AnyHit,
            ShaderExecution::ClosestHit => Self::ClosestHit,
            ShaderExecution::Miss => Self::Miss,
            ShaderExecution::Intersection => Self::Intersection,
            ShaderExecution::Callable => Self::Callable,
            ShaderExecution::Task => Self::Task,
            ShaderExecution::Mesh => Self::Mesh,
            ShaderExecution::SubpassShading => Self::SubpassShading,
        }
    }
}

impl From<ShaderStages> for PipelineStages {
    #[inline]
    fn from(stages: ShaderStages) -> PipelineStages {
        let mut result = PipelineStages::empty();

        if stages.intersects(ShaderStages::VERTEX) {
            result |= PipelineStages::VERTEX_SHADER
        }

        if stages.intersects(ShaderStages::TESSELLATION_CONTROL) {
            result |= PipelineStages::TESSELLATION_CONTROL_SHADER
        }

        if stages.intersects(ShaderStages::TESSELLATION_EVALUATION) {
            result |= PipelineStages::TESSELLATION_EVALUATION_SHADER
        }

        if stages.intersects(ShaderStages::GEOMETRY) {
            result |= PipelineStages::GEOMETRY_SHADER
        }

        if stages.intersects(ShaderStages::FRAGMENT) {
            result |= PipelineStages::FRAGMENT_SHADER
        }

        if stages.intersects(ShaderStages::COMPUTE) {
            result |= PipelineStages::COMPUTE_SHADER
        }

        if stages.intersects(
            ShaderStages::RAYGEN
                | ShaderStages::ANY_HIT
                | ShaderStages::CLOSEST_HIT
                | ShaderStages::MISS
                | ShaderStages::INTERSECTION
                | ShaderStages::CALLABLE,
        ) {
            result |= PipelineStages::RAY_TRACING_SHADER
        }

        if stages.intersects(ShaderStages::TASK) {
            result |= PipelineStages::TASK_SHADER;
        }

        if stages.intersects(ShaderStages::MESH) {
            result |= PipelineStages::MESH_SHADER;
        }

        if stages.intersects(ShaderStages::SUBPASS_SHADING) {
            result |= PipelineStages::SUBPASS_SHADING;
        }

        result
    }
}
