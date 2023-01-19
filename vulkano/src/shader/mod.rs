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
//! provide an explicit *offset* for every member of a struct, indicating where it is placed
//! relative to the start of the struct. If there are arrays or matrices among the variables, the
//! SPIR-V code must also provide an explicit *stride* (the number of bytes between the start of
//! each value) for them. When providing data to shaders, you must make sure that your data is
//! placed at the locations indicated within the SPIR-V code, or the shader will read the wrong
//! data and produce nonsense.
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
//! must likewise be a multiple of this number. Regardless of whether the offset/stride is provided
//! manually in the compiled SPIR-V code, or assigned automatically by the shader compiler, all
//! variable offsets/strides in a shader must follow these alignment rules.
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
//! | GLSL type | Scalar | Base   | Extended                             |
//! |-----------|--------|--------|--------------------------------------|
//! | primitive | N      | N      | N                                    |
//! | `vec2`    | N      | N * 2  | N * 2                                |
//! | `vec3`    | N      | N * 4  | N * 4                                |
//! | `vec4`    | N      | N * 4  | N * 4                                |
//! | array     | N      | N      | N, rounded up to multiple of 16      |
//! | `struct`  | max(N) | max(N) | max(N), rounded up to multiple of 16 |
//!
//! In the base and extended alignment, the alignment of a vector is the size of the whole vector,
//! rather than the size of its individual elements as is the case in the scalar alignment.
//! But note that `vec3` has the same alignment as `vec4`, so it is not possible to tightly
//! pack multiple `vec3` values (e.g. in an array); there will always be empty padding between them.
//!
//! For arrays, in both the scalar and base alignment, the offset and stride must be a multiple of
//! the alignment of the type of element. In the extended alignment, however, the offset and stride
//! must also be a multiple of 16 (the size of a `vec4`). Therefore, the stride of the array can
//! be greater than the element size. For example, in an array of `float`, the stride will be 16,
//! even though a `float` itself is only 4 bytes in size. Every `float` element will be followed
//! by 12 bytes of unused space.
//!
//! A matrix `matCxR` is considered equivalent to an array of column vectors `vecR[C]`.
//! In the base and extended alignments, that means that if the matrix has 3 rows, there will be
//! one element's worth of padding between the column vectors. In the extended alignment,
//! the stride between column vectors of the matrix must also be a multiple of 16,
//! further increasing the amount of padding between the column vectors.
//!
//! The rules for `struct`s are similar to those of arrays. When the members of the struct have
//! different alignment requirements, the alignment of the struct as a whole is the maximum
//! of the alignments of its members. As with arrays, in the extended alignment, the alignment
//! of a struct must be a multiple of 16.
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
    macros::vulkan_bitflags_enum,
    pipeline::{graphics::input_assembly::PrimitiveTopology, layout::PushConstantRange},
    shader::spirv::{Capability, Spirv, SpirvError},
    sync::PipelineStages,
    DeviceSize, OomError, Version, VulkanError, VulkanObject,
};
use ahash::{HashMap, HashSet};
use std::{
    borrow::Cow,
    collections::hash_map::Entry,
    error::Error,
    ffi::{CStr, CString},
    fmt::{Display, Error as FmtError, Formatter},
    mem,
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

pub mod reflect;
pub mod spirv;

use spirv::ExecutionModel;

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/spirv_reqs.rs"));

/// Contains SPIR-V code with one or more entry points.
#[derive(Debug)]
pub struct ShaderModule {
    handle: ash::vk::ShaderModule,
    device: Arc<Device>,
    id: NonZeroU64,
    entry_points: HashMap<String, HashMap<ExecutionModel, EntryPointInfo>>,
}

impl ShaderModule {
    /// Builds a new shader module from SPIR-V 32-bit words. The shader code is parsed and the
    /// necessary information is extracted from it.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated beyond the minimum needed to extract the information.
    #[inline]
    pub unsafe fn from_words(
        device: Arc<Device>,
        words: &[u32],
    ) -> Result<Arc<ShaderModule>, ShaderCreationError> {
        let spirv = Spirv::new(words)?;

        Self::from_words_with_data(
            device,
            words,
            spirv.version(),
            reflect::spirv_capabilities(&spirv),
            reflect::spirv_extensions(&spirv),
            reflect::entry_points(&spirv),
        )
    }

    /// As `from_words`, but takes a slice of bytes.
    ///
    /// # Panics
    ///
    /// - Panics if the length of `bytes` is not a multiple of 4.
    #[inline]
    pub unsafe fn from_bytes(
        device: Arc<Device>,
        bytes: &[u8],
    ) -> Result<Arc<ShaderModule>, ShaderCreationError> {
        assert!((bytes.len() % 4) == 0);

        Self::from_words(
            device,
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const _,
                bytes.len() / mem::size_of::<u32>(),
            ),
        )
    }

    /// As `from_words`, but does not parse the code. Instead, you must provide the needed
    /// information yourself. This can be useful if you've already done parsing yourself and
    /// want to prevent Vulkano from doing it a second time.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated at all.
    /// - The provided information must match what the SPIR-V code contains.
    pub unsafe fn from_words_with_data<'a>(
        device: Arc<Device>,
        words: &[u32],
        spirv_version: Version,
        spirv_capabilities: impl IntoIterator<Item = &'a Capability>,
        spirv_extensions: impl IntoIterator<Item = &'a str>,
        entry_points: impl IntoIterator<Item = (String, ExecutionModel, EntryPointInfo)>,
    ) -> Result<Arc<ShaderModule>, ShaderCreationError> {
        if let Err(reason) = check_spirv_version(&device, spirv_version) {
            return Err(ShaderCreationError::SpirvVersionNotSupported {
                version: spirv_version,
                reason,
            });
        }

        for &capability in spirv_capabilities {
            if let Err(reason) = check_spirv_capability(&device, capability) {
                return Err(ShaderCreationError::SpirvCapabilityNotSupported {
                    capability,
                    reason,
                });
            }
        }

        for extension in spirv_extensions {
            if let Err(reason) = check_spirv_extension(&device, extension) {
                return Err(ShaderCreationError::SpirvExtensionNotSupported {
                    extension: extension.to_owned(),
                    reason,
                });
            }
        }

        let handle = {
            let infos = ash::vk::ShaderModuleCreateInfo {
                flags: ash::vk::ShaderModuleCreateFlags::empty(),
                code_size: words.len() * mem::size_of::<u32>(),
                p_code: words.as_ptr(),
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

        let entries = entry_points.into_iter().collect::<Vec<_>>();
        let entry_points = entries
            .iter()
            .map(|(name, _, _)| name)
            .collect::<HashSet<_>>()
            .iter()
            .map(|name| {
                (
                    (*name).clone(),
                    entries
                        .iter()
                        .filter_map(|(entry_name, entry_model, info)| {
                            if &entry_name == name {
                                Some((*entry_model, info.clone()))
                            } else {
                                None
                            }
                        })
                        .collect::<HashMap<_, _>>(),
                )
            })
            .collect();

        Ok(Arc::new(ShaderModule {
            handle,
            device,
            id: Self::next_id(),
            entry_points,
        }))
    }

    /// As `from_words_with_data`, but takes a slice of bytes.
    ///
    /// # Panics
    ///
    /// - Panics if the length of `bytes` is not a multiple of 4.
    pub unsafe fn from_bytes_with_data<'a>(
        device: Arc<Device>,
        bytes: &[u8],
        spirv_version: Version,
        spirv_capabilities: impl IntoIterator<Item = &'a Capability>,
        spirv_extensions: impl IntoIterator<Item = &'a str>,
        entry_points: impl IntoIterator<Item = (String, ExecutionModel, EntryPointInfo)>,
    ) -> Result<Arc<ShaderModule>, ShaderCreationError> {
        assert!((bytes.len() % 4) == 0);

        Self::from_words_with_data(
            device,
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const _,
                bytes.len() / mem::size_of::<u32>(),
            ),
            spirv_version,
            spirv_capabilities,
            spirv_extensions,
            entry_points,
        )
    }

    /// Returns information about the entry point with the provided name. Returns `None` if no entry
    /// point with that name exists in the shader module or if multiple entry points with the same
    /// name exist.
    #[inline]
    pub fn entry_point<'a>(&'a self, name: &str) -> Option<EntryPoint<'a>> {
        self.entry_points.get(name).and_then(|infos| {
            if infos.len() == 1 {
                infos.iter().next().map(|(_, info)| EntryPoint {
                    module: self,
                    name: CString::new(name).unwrap(),
                    info,
                })
            } else {
                None
            }
        })
    }

    /// Returns information about the entry point with the provided name and execution model.
    /// Returns `None` if no entry and execution model exists in the shader module.
    #[inline]
    pub fn entry_point_with_execution<'a>(
        &'a self,
        name: &str,
        execution: ExecutionModel,
    ) -> Option<EntryPoint<'a>> {
        self.entry_points.get(name).and_then(|infos| {
            infos.get(&execution).map(|info| EntryPoint {
                module: self,
                name: CString::new(name).unwrap(),
                info,
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

crate::impl_id_counter!(ShaderModule);

/// Error that can happen when creating a new shader module.
#[derive(Clone, Debug)]
pub enum ShaderCreationError {
    OomError(OomError),
    SpirvCapabilityNotSupported {
        capability: Capability,
        reason: ShaderSupportError,
    },
    SpirvError(SpirvError),
    SpirvExtensionNotSupported {
        extension: String,
        reason: ShaderSupportError,
    },
    SpirvVersionNotSupported {
        version: Version,
        reason: ShaderSupportError,
    },
}

impl Error for ShaderCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            Self::SpirvCapabilityNotSupported { reason, .. } => Some(reason),
            Self::SpirvError(err) => Some(err),
            Self::SpirvExtensionNotSupported { reason, .. } => Some(reason),
            Self::SpirvVersionNotSupported { reason, .. } => Some(reason),
        }
    }
}

impl Display for ShaderCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::SpirvCapabilityNotSupported { capability, .. } => write!(
                f,
                "the SPIR-V capability {:?} enabled by the shader is not supported by the device",
                capability,
            ),
            Self::SpirvError(_) => write!(f, "the SPIR-V module could not be read"),
            Self::SpirvExtensionNotSupported { extension, .. } => write!(
                f,
                "the SPIR-V extension {} enabled by the shader is not supported by the device",
                extension,
            ),
            Self::SpirvVersionNotSupported { version, .. } => write!(
                f,
                "the shader uses SPIR-V version {}.{}, which is not supported by the device",
                version.major, version.minor,
            ),
        }
    }
}

impl From<VulkanError> for ShaderCreationError {
    fn from(err: VulkanError) -> Self {
        Self::OomError(err.into())
    }
}

impl From<SpirvError> for ShaderCreationError {
    fn from(err: SpirvError) -> Self {
        Self::SpirvError(err)
    }
}

/// Error that can happen when checking whether a shader is supported by a device.
#[derive(Clone, Copy, Debug)]
pub enum ShaderSupportError {
    NotSupportedByVulkan,
    RequirementsNotMet(&'static [&'static str]),
}

impl Error for ShaderSupportError {}

impl Display for ShaderSupportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::NotSupportedByVulkan => write!(f, "not supported by Vulkan"),
            Self::RequirementsNotMet(requirements) => write!(
                f,
                "at least one of the following must be available/enabled on the device: {}",
                requirements.join(", "),
            ),
        }
    }
}

/// The information associated with a single entry point in a shader.
#[derive(Clone, Debug)]
pub struct EntryPointInfo {
    pub execution: ShaderExecution,
    pub descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    pub push_constant_requirements: Option<PushConstantRange>,
    pub specialization_constant_requirements: HashMap<u32, SpecializationConstantRequirements>,
    pub input_interface: ShaderInterface,
    pub output_interface: ShaderInterface,
}

/// Represents a shader entry point in a shader module.
///
/// Can be obtained by calling [`entry_point`](ShaderModule::entry_point) on the shader module.
#[derive(Clone, Debug)]
pub struct EntryPoint<'a> {
    module: &'a ShaderModule,
    name: CString,
    info: &'a EntryPointInfo,
}

impl<'a> EntryPoint<'a> {
    /// Returns the module this entry point comes from.
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    /// Returns the name of the entry point.
    #[inline]
    pub fn name(&self) -> &CStr {
        &self.name
    }

    /// Returns the execution model of the shader.
    #[inline]
    pub fn execution(&self) -> &ShaderExecution {
        &self.info.execution
    }

    /// Returns the descriptor binding requirements.
    #[inline]
    pub fn descriptor_binding_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = ((u32, u32), &DescriptorBindingRequirements)> {
        self.info
            .descriptor_binding_requirements
            .iter()
            .map(|(k, v)| (*k, v))
    }

    /// Returns the push constant requirements.
    #[inline]
    pub fn push_constant_requirements(&self) -> Option<&PushConstantRange> {
        self.info.push_constant_requirements.as_ref()
    }

    /// Returns the specialization constant requirements.
    #[inline]
    pub fn specialization_constant_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = (u32, &SpecializationConstantRequirements)> {
        self.info
            .specialization_constant_requirements
            .iter()
            .map(|(k, v)| (*k, v))
    }

    /// Returns the input attributes used by the shader stage.
    #[inline]
    pub fn input_interface(&self) -> &ShaderInterface {
        &self.info.input_interface
    }

    /// Returns the output attributes used by the shader stage.
    #[inline]
    pub fn output_interface(&self) -> &ShaderInterface {
        &self.info.output_interface
    }
}

/// The mode in which a shader executes. This includes both information about the shader type/stage,
/// and additional data relevant to particular shader types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    pub image_scalar_type: Option<ShaderScalarType>,

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
    pub fn merge(&mut self, other: &Self) -> Result<(), DescriptorBindingRequirementsIncompatible> {
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
            return Err(DescriptorBindingRequirementsIncompatible::DescriptorType);
        }

        if let (Some(first), Some(second)) = (*image_format, other.image_format) {
            if first != second {
                return Err(DescriptorBindingRequirementsIncompatible::ImageFormat);
            }
        }

        if let (Some(first), Some(second)) = (*image_scalar_type, other.image_scalar_type) {
            if first != second {
                return Err(DescriptorBindingRequirementsIncompatible::ImageScalarType);
            }
        }

        if let (Some(first), Some(second)) = (*image_view_type, other.image_view_type) {
            if first != second {
                return Err(DescriptorBindingRequirementsIncompatible::ImageViewType);
            }
        }

        if *image_multisampled != other.image_multisampled {
            return Err(DescriptorBindingRequirementsIncompatible::ImageMultisampled);
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

/// An error that can be returned when trying to create the intersection of two
/// `DescriptorBindingRequirements` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorBindingRequirementsIncompatible {
    /// The allowed descriptor types of the descriptors do not overlap.
    DescriptorType,
    /// The descriptors require different formats.
    ImageFormat,
    /// The descriptors require different scalar types.
    ImageScalarType,
    /// The multisampling requirements of the descriptors differ.
    ImageMultisampled,
    /// The descriptors require different image view types.
    ImageViewType,
}

impl Error for DescriptorBindingRequirementsIncompatible {}

impl Display for DescriptorBindingRequirementsIncompatible {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            DescriptorBindingRequirementsIncompatible::DescriptorType => write!(
                f,
                "the allowed descriptor types of the two descriptors do not overlap",
            ),
            DescriptorBindingRequirementsIncompatible::ImageFormat => {
                write!(f, "the descriptors require different formats",)
            }
            DescriptorBindingRequirementsIncompatible::ImageMultisampled => write!(
                f,
                "the multisampling requirements of the descriptors differ",
            ),
            DescriptorBindingRequirementsIncompatible::ImageScalarType => {
                write!(f, "the descriptors require different scalar types",)
            }
            DescriptorBindingRequirementsIncompatible::ImageViewType => {
                write!(f, "the descriptors require different image view types",)
            }
        }
    }
}

/// The requirements imposed by a shader on a specialization constant.
#[derive(Clone, Copy, Debug)]
pub struct SpecializationConstantRequirements {
    pub size: DeviceSize,
}

/// Trait for types that contain specialization data for shaders.
///
/// Shader modules can contain what is called *specialization constants*. They are the same as
/// constants except that their values can be defined when you create a compute pipeline or a
/// graphics pipeline. Doing so is done by passing a type that implements the
/// `SpecializationConstants` trait and that stores the values in question. The `descriptors()`
/// method of this trait indicates how to grab them.
///
/// Boolean specialization constants must be stored as 32bits integers, where `0` means `false` and
/// any non-zero value means `true`. Integer and floating-point specialization constants are
/// stored as their Rust equivalent.
///
/// This trait is implemented on `()` for shaders that don't have any specialization constant.
///
/// # Examples
///
/// ```rust
/// use vulkano::shader::SpecializationConstants;
/// use vulkano::shader::SpecializationMapEntry;
///
/// #[repr(C)]      // `#[repr(C)]` guarantees that the struct has a specific layout
/// struct MySpecConstants {
///     my_integer_constant: i32,
///     a_boolean: u32,
///     floating_point: f32,
/// }
///
/// unsafe impl SpecializationConstants for MySpecConstants {
///     fn descriptors() -> &'static [SpecializationMapEntry] {
///         static DESCRIPTORS: [SpecializationMapEntry; 3] = [
///             SpecializationMapEntry {
///                 constant_id: 0,
///                 offset: 0,
///                 size: 4,
///             },
///             SpecializationMapEntry {
///                 constant_id: 1,
///                 offset: 4,
///                 size: 4,
///             },
///             SpecializationMapEntry {
///                 constant_id: 2,
///                 offset: 8,
///                 size: 4,
///             },
///         ];
///
///         &DESCRIPTORS
///     }
/// }
/// ```
///
/// # Safety
///
/// - The `SpecializationMapEntry` returned must contain valid offsets and sizes.
/// - The size of each `SpecializationMapEntry` must match the size of the corresponding constant
///   (`4` for booleans).
pub unsafe trait SpecializationConstants {
    /// Returns descriptors of the struct's layout.
    fn descriptors() -> &'static [SpecializationMapEntry];
}

unsafe impl SpecializationConstants for () {
    #[inline]
    fn descriptors() -> &'static [SpecializationMapEntry] {
        &[]
    }
}

/// Describes an individual constant to set in the shader. Also a field in the struct.
// Implementation note: has the same memory representation as a `VkSpecializationMapEntry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SpecializationMapEntry {
    /// Identifier of the constant in the shader that corresponds to this field.
    ///
    /// For SPIR-V, this must be the value of the `SpecId` decoration applied to the specialization
    /// constant.
    /// For GLSL, this must be the value of `N` in the `layout(constant_id = N)` attribute applied
    /// to a constant.
    pub constant_id: u32,

    /// Offset within the struct where the data can be found.
    pub offset: u32,

    /// Size of the data in bytes. Must match the size of the constant (`4` for booleans).
    pub size: usize,
}

impl From<SpecializationMapEntry> for ash::vk::SpecializationMapEntry {
    #[inline]
    fn from(val: SpecializationMapEntry) -> Self {
        Self {
            constant_id: val.constant_id,
            offset: val.offset,
            size: val.size,
        }
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
    pub fn matches(&self, other: &ShaderInterface) -> Result<(), ShaderInterfaceMismatchError> {
        if self.elements().len() != other.elements().len() {
            return Err(ShaderInterfaceMismatchError::ElementsCountMismatch {
                self_elements: self.elements().len() as u32,
                other_elements: other.elements().len() as u32,
            });
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
                        return Err(ShaderInterfaceMismatchError::MissingElement { location: loc })
                    }
                    Some(b) => b,
                };

                if a.ty != b.ty {
                    return Err(ShaderInterfaceMismatchError::TypeMismatch {
                        location: loc,
                        self_ty: a.ty,
                        other_ty: b.ty,
                    });
                }

                // TODO: enforce this?
                /*match (a.name, b.name) {
                    (Some(ref an), Some(ref bn)) => if an != bn { return false },
                    _ => ()
                };*/
            }
        }

        // Note: since we check that the number of elements is the same, we don't need to iterate
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
    pub base_type: ShaderScalarType,

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

/// The numeric base type of a shader variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderScalarType {
    Float,
    Sint,
    Uint,
}

// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap43.html#formats-numericformat
impl From<NumericType> for ShaderScalarType {
    #[inline]
    fn from(val: NumericType) -> Self {
        match val {
            NumericType::SFLOAT => Self::Float,
            NumericType::UFLOAT => Self::Float,
            NumericType::SINT => Self::Sint,
            NumericType::UINT => Self::Uint,
            NumericType::SNORM => Self::Float,
            NumericType::UNORM => Self::Float,
            NumericType::SSCALED => Self::Float,
            NumericType::USCALED => Self::Float,
            NumericType::SRGB => Self::Float,
        }
    }
}

/// Error that can happen when the interface mismatches between two shader stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderInterfaceMismatchError {
    /// The number of elements is not the same between the two shader interfaces.
    ElementsCountMismatch {
        /// Number of elements in the first interface.
        self_elements: u32,
        /// Number of elements in the second interface.
        other_elements: u32,
    },

    /// An element is missing from one of the interfaces.
    MissingElement {
        /// Location of the missing element.
        location: u32,
    },

    /// The type of an element does not match.
    TypeMismatch {
        /// Location of the element that mismatches.
        location: u32,
        /// Type in the first interface.
        self_ty: ShaderInterfaceEntryType,
        /// Type in the second interface.
        other_ty: ShaderInterfaceEntryType,
    },
}

impl Error for ShaderInterfaceMismatchError {}

impl Display for ShaderInterfaceMismatchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                ShaderInterfaceMismatchError::ElementsCountMismatch { .. } => {
                    "the number of elements mismatches"
                }
                ShaderInterfaceMismatchError::MissingElement { .. } => "an element is missing",
                ShaderInterfaceMismatchError::TypeMismatch { .. } => {
                    "the type of an element does not match"
                }
            }
        )
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
    RAYGEN, Raygen = RAYGEN_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    ANY_HIT, AnyHit = ANY_HIT_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    CLOSEST_HIT, ClosestHit = CLOSEST_HIT_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    MISS, Miss = MISS_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    INTERSECTION, Intersection = INTERSECTION_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    CALLABLE, Callable = CALLABLE_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    TASK, Task = TASK_EXT {
        device_extensions: [ext_mesh_shader, nv_mesh_shader],
    },

    // TODO: document
    MESH, Mesh = MESH_EXT {
        device_extensions: [ext_mesh_shader, nv_mesh_shader],
    },

    // TODO: document
    SUBPASS_SHADING, SubpassShading = SUBPASS_SHADING_HUAWEI {
        device_extensions: [huawei_subpass_shading],
    },
}

impl From<ShaderExecution> for ShaderStage {
    #[inline]
    fn from(val: ShaderExecution) -> Self {
        match val {
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

fn check_spirv_version(device: &Device, mut version: Version) -> Result<(), ShaderSupportError> {
    version.patch = 0; // Ignore the patch version

    match version {
        Version::V1_0 => {}
        Version::V1_1 | Version::V1_2 | Version::V1_3 => {
            if !(device.api_version() >= Version::V1_1) {
                return Err(ShaderSupportError::RequirementsNotMet(&[
                    "Vulkan API version 1.1",
                ]));
            }
        }
        Version::V1_4 => {
            if !(device.api_version() >= Version::V1_2 || device.enabled_extensions().khr_spirv_1_4)
            {
                return Err(ShaderSupportError::RequirementsNotMet(&[
                    "Vulkan API version 1.2",
                    "extension `khr_spirv_1_4`",
                ]));
            }
        }
        Version::V1_5 => {
            if !(device.api_version() >= Version::V1_2) {
                return Err(ShaderSupportError::RequirementsNotMet(&[
                    "Vulkan API version 1.2",
                ]));
            }
        }
        Version::V1_6 => {
            if !(device.api_version() >= Version::V1_3) {
                return Err(ShaderSupportError::RequirementsNotMet(&[
                    "Vulkan API version 1.3",
                ]));
            }
        }
        _ => return Err(ShaderSupportError::NotSupportedByVulkan),
    }
    Ok(())
}
