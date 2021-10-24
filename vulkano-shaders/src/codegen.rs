// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::entry_point;
use crate::read_file_to_string;
use crate::spec_consts;
use crate::structs;
use crate::RegisteredType;
use crate::TypesMeta;
use proc_macro2::{Span, TokenStream};
pub use shaderc::{CompilationArtifact, IncludeType, ResolvedInclude, ShaderKind};
use shaderc::{CompileOptions, Compiler, EnvVersion, SpirvVersion, TargetEnv};
use std::collections::HashMap;
use std::iter::Iterator;
use std::path::Path;
use std::{
    cell::{RefCell, RefMut},
    io::Error as IoError,
};
use syn::Ident;
use vulkano::{
    spirv::{Capability, Instruction, Spirv, SpirvError, StorageClass},
    Version,
};

pub(super) fn path_to_str(path: &Path) -> &str {
    path.to_str().expect(
        "Could not stringify the file to be included. Make sure the path consists of \
                 valid unicode characters.",
    )
}

fn include_callback(
    requested_source_path_raw: &str,
    directive_type: IncludeType,
    contained_within_path_raw: &str,
    recursion_depth: usize,
    include_directories: &[impl AsRef<Path>],
    root_source_has_path: bool,
    base_path: &impl AsRef<Path>,
    mut includes_tracker: RefMut<Vec<String>>,
) -> Result<ResolvedInclude, String> {
    let file_to_include = match directive_type {
        IncludeType::Relative => {
            let requested_source_path = Path::new(requested_source_path_raw);
            // Is embedded current shader source embedded within a rust macro?
            // If so, abort unless absolute path.
            if !root_source_has_path && recursion_depth == 1 && !requested_source_path.is_absolute()
            {
                let requested_source_name = requested_source_path
                    .file_name()
                    .expect("Could not get the name of the requested source file.")
                    .to_string_lossy();
                let requested_source_directory = requested_source_path
                    .parent()
                    .expect("Could not get the directory of the requested source file.")
                    .to_string_lossy();

                return Err(format!(
                    "Usage of relative paths in imports in embedded GLSL is not \
                                    allowed, try using `#include <{}>` and adding the directory \
                                    `{}` to the `include` array in your `shader!` macro call \
                                    instead.",
                    requested_source_name, requested_source_directory
                ));
            }

            let mut resolved_path = if recursion_depth == 1 {
                Path::new(contained_within_path_raw)
                    .parent()
                    .map(|parent| base_path.as_ref().join(parent))
            } else {
                Path::new(contained_within_path_raw)
                    .parent()
                    .map(|parent| parent.to_owned())
            }
            .unwrap_or_else(|| {
                panic!(
                    "The file `{}` does not reside in a directory. This is \
                                        an implementation error.",
                    contained_within_path_raw
                )
            });
            resolved_path.push(requested_source_path);

            if !resolved_path.is_file() {
                return Err(format!(
                    "Invalid inclusion path `{}`, the path does not point to a file.",
                    requested_source_path_raw
                ));
            }

            resolved_path
        }
        IncludeType::Standard => {
            let requested_source_path = Path::new(requested_source_path_raw);

            if requested_source_path.is_absolute() {
                // This message is printed either when using a missing file with an absolute path
                // in the relative include directive or when using absolute paths in a standard
                // include directive.
                return Err(format!(
                    "No such file found, as specified by the absolute path. \
                                    Keep in mind, that absolute paths cannot be used with \
                                    inclusion from standard directories (`#include <...>`), try \
                                    using `#include \"...\"` instead. Requested path: {}",
                    requested_source_path_raw
                ));
            }

            let found_requested_source_path = include_directories
                .iter()
                .map(|include_directory| include_directory.as_ref().join(requested_source_path))
                .find(|resolved_requested_source_path| resolved_requested_source_path.is_file());

            if let Some(found_requested_source_path) = found_requested_source_path {
                found_requested_source_path
            } else {
                return Err(format!(
                    "Could not include the file `{}` from any include directories.",
                    requested_source_path_raw
                ));
            }
        }
    };

    let file_to_include_string = path_to_str(file_to_include.as_path()).to_string();
    let content = read_file_to_string(file_to_include.as_path()).map_err(|_| {
        format!(
            "Could not read the contents of file `{}` to be included in the \
                              shader source.",
            &file_to_include_string
        )
    })?;

    includes_tracker.push(file_to_include_string.clone());

    Ok(ResolvedInclude {
        resolved_name: file_to_include_string,
        content,
    })
}

pub fn compile(
    path: Option<String>,
    base_path: &impl AsRef<Path>,
    code: &str,
    ty: ShaderKind,
    include_directories: &[impl AsRef<Path>],
    macro_defines: &[(impl AsRef<str>, impl AsRef<str>)],
    vulkan_version: Option<EnvVersion>,
    spirv_version: Option<SpirvVersion>,
) -> Result<(CompilationArtifact, Vec<String>), String> {
    let includes_tracker = RefCell::new(Vec::new());
    let mut compiler = Compiler::new().ok_or("failed to create GLSL compiler")?;
    let mut compile_options = CompileOptions::new().ok_or("failed to initialize compile option")?;

    compile_options.set_target_env(
        TargetEnv::Vulkan,
        vulkan_version.unwrap_or(EnvVersion::Vulkan1_0) as u32,
    );

    if let Some(spirv_version) = spirv_version {
        compile_options.set_target_spirv(spirv_version);
    }

    let root_source_path = if let &Some(ref path) = &path {
        path
    } else {
        // An arbitrary placeholder file name for embedded shaders
        "shader.glsl"
    };

    // Specify file resolution callback for the `#include` directive
    compile_options.set_include_callback(
        |requested_source_path, directive_type, contained_within_path, recursion_depth| {
            include_callback(
                requested_source_path,
                directive_type,
                contained_within_path,
                recursion_depth,
                include_directories,
                path.is_some(),
                base_path,
                includes_tracker.borrow_mut(),
            )
        },
    );

    for (macro_name, macro_value) in macro_defines.iter() {
        compile_options.add_macro_definition(macro_name.as_ref(), Some(macro_value.as_ref()));
    }

    let content = compiler
        .compile_into_spirv(&code, ty, root_source_path, "main", Some(&compile_options))
        .map_err(|e| e.to_string())?;

    let includes = includes_tracker.borrow().clone();

    Ok((content, includes))
}

pub(super) fn reflect<'a, I>(
    prefix: &'a str,
    words: &[u32],
    types_meta: &TypesMeta,
    input_paths: I,
    exact_entrypoint_interface: bool,
    shared_constants: bool,
    types_registry: &'a mut HashMap<String, RegisteredType>,
) -> Result<(TokenStream, TokenStream), Error>
where
    I: IntoIterator<Item = &'a str>,
{
    let struct_name = Ident::new(&format!("{}Shader", prefix), Span::call_site());
    let spirv = Spirv::new(words)?;

    // checking whether each required capability is enabled in the Vulkan device
    let mut cap_checks: Vec<TokenStream> = vec![];
    match spirv.version() {
        Version::V1_0 => {}
        Version::V1_1 | Version::V1_2 | Version::V1_3 => {
            cap_checks.push(quote! {
                if device.api_version() < Version::V1_1 {
                    panic!("Device API version 1.1 required");
                }
            });
        }
        Version::V1_4 => {
            cap_checks.push(quote! {
                if device.api_version() < Version::V1_2
                    && !device.enabled_extensions().khr_spirv_1_4 {
                    panic!("Device API version 1.2 or extension VK_KHR_spirv_1_4 required");
                }
            });
        }
        Version::V1_5 => {
            cap_checks.push(quote! {
                if device.api_version() < Version::V1_2 {
                    panic!("Device API version 1.2 required");
                }
            });
        }
        _ => return Err(Error::UnsupportedSpirvVersion),
    }

    for i in spirv.instructions() {
        let dev_req = {
            match i {
                Instruction::Variable {
                    result_type_id: _,
                    result_id: _,
                    storage_class,
                    initializer: _,
                } => storage_class_requirement(storage_class),
                Instruction::TypePointer {
                    result_id: _,
                    storage_class,
                    ty: _,
                } => storage_class_requirement(storage_class),
                Instruction::Capability { capability } => capability_requirement(capability),
                _ => &[],
            }
        };

        if dev_req.len() == 0 {
            continue;
        }

        let (conditions, messages): (Vec<_>, Vec<_>) = dev_req
            .iter()
            .map(|req| match req {
                DeviceRequirement::Extension(extension) => {
                    let ident = Ident::new(extension, Span::call_site());
                    (
                        quote! { device.enabled_extensions().#ident },
                        format!("extension {}", extension),
                    )
                }
                DeviceRequirement::Feature(feature) => {
                    let ident = Ident::new(feature, Span::call_site());
                    (
                        quote! { device.enabled_features().#ident },
                        format!("feature {}", feature),
                    )
                }
                DeviceRequirement::Version(major, minor) => {
                    let ident = format_ident!("V{}_{}", major, minor);
                    (
                        quote! { device.api_version() >= crate::Version::#ident },
                        format!("API version {}.{}", major, minor),
                    )
                }
            })
            .unzip();
        let messages = messages.join(", ");

        cap_checks.push(quote! {
            if !std::array::IntoIter::new([#(#conditions),*]).all(|x| x) {
                panic!("One of the following must be enabled on the device: {}", #messages);
            }
        });
    }

    // writing one method for each entry point of this module
    let mut entry_points_inside_impl: Vec<TokenStream> = vec![];
    for instruction in spirv
        .iter_entry_point()
        .filter(|instruction| matches!(instruction, Instruction::EntryPoint { .. }))
    {
        let entry_point = entry_point::write_entry_point(
            prefix,
            &spirv,
            instruction,
            types_meta,
            exact_entrypoint_interface,
            shared_constants,
        );
        entry_points_inside_impl.push(entry_point);
    }

    let include_bytes = input_paths.into_iter().map(|s| {
        quote! {
            // using include_bytes here ensures that changing the shader will force recompilation.
            // The bytes themselves can be optimized out by the compiler as they are unused.
            ::std::include_bytes!( #s )
        }
    });

    let structs = structs::write_structs(prefix, &spirv, types_meta, types_registry);
    let specialization_constants = spec_consts::write_specialization_constants(
        prefix,
        &spirv,
        types_meta,
        shared_constants,
        types_registry,
    );
    let shader_code = quote! {
        pub struct #struct_name {
            shader: ::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule>,
        }

        impl #struct_name {
            /// Loads the shader in Vulkan as a `ShaderModule`.
            #[inline]
            #[allow(unsafe_code)]
            pub fn load(device: ::std::sync::Arc<::vulkano::device::Device>)
                        -> Result<#struct_name, ::vulkano::OomError>
            {
                let _bytes = ( #( #include_bytes),* );

                #( #cap_checks )*
                static WORDS: &[u32] = &[ #( #words ),* ];

                unsafe {
                    Ok(#struct_name {
                        shader: ::vulkano::pipeline::shader::ShaderModule::from_words(device, WORDS)?
                    })
                }
            }

            /// Returns the module that was created.
            #[allow(dead_code)]
            #[inline]
            pub fn module(&self) -> &::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule> {
                &self.shader
            }

            #( #entry_points_inside_impl )*
        }

        #specialization_constants
    };

    Ok((shader_code, structs))
}

#[derive(Debug)]
pub enum Error {
    UnsupportedSpirvVersion,
    IoError(IoError),
    SpirvError(SpirvError),
}

impl From<IoError> for Error {
    #[inline]
    fn from(err: IoError) -> Error {
        Error::IoError(err)
    }
}

impl From<SpirvError> for Error {
    #[inline]
    fn from(err: SpirvError) -> Error {
        Error::SpirvError(err)
    }
}

/// Returns the Vulkan device requirement for a SPIR-V `OpCapability`.
#[rustfmt::skip]
fn capability_requirement(cap: &Capability) -> &'static [DeviceRequirement] {
    match *cap {
        Capability::Matrix => &[],
        Capability::Shader => &[],
        Capability::InputAttachment => &[],
        Capability::Sampled1D => &[],
        Capability::Image1D => &[],
        Capability::SampledBuffer => &[],
        Capability::ImageBuffer => &[],
        Capability::ImageQuery => &[],
        Capability::DerivativeControl => &[],
        Capability::Geometry => &[DeviceRequirement::Feature("geometry_shader")],
        Capability::Tessellation => &[DeviceRequirement::Feature("tessellation_shader")],
        Capability::Float64 => &[DeviceRequirement::Feature("shader_float64")],
        Capability::Int64 => &[DeviceRequirement::Feature("shader_int64")],
        Capability::Int64Atomics => &[
            DeviceRequirement::Feature("shader_buffer_int64_atomics"),
            DeviceRequirement::Feature("shader_shared_int64_atomics"),
            DeviceRequirement::Feature("shader_image_int64_atomics"),
        ],
        /* Capability::AtomicFloat16AddEXT => &[
            DeviceRequirement::Feature("shader_buffer_float16_atomic_add"),
            DeviceRequirement::Feature("shader_shared_float16_atomic_add"),
        ], */
        Capability::AtomicFloat32AddEXT => &[
            DeviceRequirement::Feature("shader_buffer_float32_atomic_add"),
            DeviceRequirement::Feature("shader_shared_float32_atomic_add"),
            DeviceRequirement::Feature("shader_image_float32_atomic_add"),
        ],
        Capability::AtomicFloat64AddEXT => &[
            DeviceRequirement::Feature("shader_buffer_float64_atomic_add"),
            DeviceRequirement::Feature("shader_shared_float64_atomic_add"),
        ],
        /* Capability::AtomicFloat16MinMaxEXT => &[
            DeviceRequirement::Feature("shader_buffer_float16_atomic_min_max"),
            DeviceRequirement::Feature("shader_shared_float16_atomic_min_max"),
        ], */
        /* Capability::AtomicFloat32MinMaxEXT => &[
            DeviceRequirement::Feature("shader_buffer_float32_atomic_min_max"),
            DeviceRequirement::Feature("shader_shared_float32_atomic_min_max"),
            DeviceRequirement::Feature("shader_image_float32_atomic_min_max"),
        ], */
        /* Capability::AtomicFloat64MinMaxEXT => &[
            DeviceRequirement::Feature("shader_buffer_float64_atomic_min_max"),
            DeviceRequirement::Feature("shader_shared_float64_atomic_min_max"),
        ], */
        Capability::Int64ImageEXT => &[DeviceRequirement::Feature("shader_image_int64_atomics")],
        Capability::Int16 => &[DeviceRequirement::Feature("shader_int16")],
        Capability::TessellationPointSize => &[DeviceRequirement::Feature(
            "shader_tessellation_and_geometry_point_size",
        )],
        Capability::GeometryPointSize => &[DeviceRequirement::Feature(
            "shader_tessellation_and_geometry_point_size",
        )],
        Capability::ImageGatherExtended => {
            &[DeviceRequirement::Feature("shader_image_gather_extended")]
        }
        Capability::StorageImageMultisample => &[DeviceRequirement::Feature(
            "shader_storage_image_multisample",
        )],
        Capability::UniformBufferArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_uniform_buffer_array_dynamic_indexing",
        )],
        Capability::SampledImageArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_sampled_image_array_dynamic_indexing",
        )],
        Capability::StorageBufferArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_storage_buffer_array_dynamic_indexing",
        )],
        Capability::StorageImageArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_storage_image_array_dynamic_indexing",
        )],
        Capability::ClipDistance => &[DeviceRequirement::Feature("shader_clip_distance")],
        Capability::CullDistance => &[DeviceRequirement::Feature("shader_cull_distance")],
        Capability::ImageCubeArray => &[DeviceRequirement::Feature("image_cube_array")],
        Capability::SampleRateShading => &[DeviceRequirement::Feature("sample_rate_shading")],
        Capability::SparseResidency => &[DeviceRequirement::Feature("shader_resource_residency")],
        Capability::MinLod => &[DeviceRequirement::Feature("shader_resource_min_lod")],
        Capability::SampledCubeArray => &[DeviceRequirement::Feature("image_cube_array")],
        Capability::ImageMSArray => &[DeviceRequirement::Feature(
            "shader_storage_image_multisample",
        )],
        Capability::StorageImageExtendedFormats => &[],
        Capability::InterpolationFunction => &[DeviceRequirement::Feature("sample_rate_shading")],
        Capability::StorageImageReadWithoutFormat => &[DeviceRequirement::Feature(
            "shader_storage_image_read_without_format",
        )],
        Capability::StorageImageWriteWithoutFormat => &[DeviceRequirement::Feature(
            "shader_storage_image_write_without_format",
        )],
        Capability::MultiViewport => &[DeviceRequirement::Feature("multi_viewport")],
        Capability::DrawParameters => &[
            DeviceRequirement::Feature("shader_draw_parameters"),
            DeviceRequirement::Extension("khr_shader_draw_parameters"),
        ],
        Capability::MultiView => &[DeviceRequirement::Feature("multiview")],
        Capability::DeviceGroup => &[
            DeviceRequirement::Version(1, 1),
            DeviceRequirement::Extension("khr_device_group"),
        ],
        Capability::VariablePointersStorageBuffer => &[DeviceRequirement::Feature(
            "variable_pointers_storage_buffer",
        )],
        Capability::VariablePointers => &[DeviceRequirement::Feature("variable_pointers")],
        Capability::ShaderClockKHR => &[DeviceRequirement::Extension("khr_shader_clock")],
        Capability::StencilExportEXT => {
            &[DeviceRequirement::Extension("ext_shader_stencil_export")]
        }
        Capability::SubgroupBallotKHR => {
            &[DeviceRequirement::Extension("ext_shader_subgroup_ballot")]
        }
        Capability::SubgroupVoteKHR => &[DeviceRequirement::Extension("ext_shader_subgroup_vote")],
        Capability::ImageReadWriteLodAMD => &[DeviceRequirement::Extension(
            "amd_shader_image_load_store_lod",
        )],
        Capability::ImageGatherBiasLodAMD => {
            &[DeviceRequirement::Extension("amd_texture_gather_bias_lod")]
        }
        Capability::FragmentMaskAMD => &[DeviceRequirement::Extension("amd_shader_fragment_mask")],
        Capability::SampleMaskOverrideCoverageNV => &[DeviceRequirement::Extension(
            "nv_sample_mask_override_coverage",
        )],
        Capability::GeometryShaderPassthroughNV => &[DeviceRequirement::Extension(
            "nv_geometry_shader_passthrough",
        )],
        Capability::ShaderViewportIndex => {
            &[DeviceRequirement::Feature("shader_output_viewport_index")]
        }
        Capability::ShaderLayer => &[DeviceRequirement::Feature("shader_output_layer")],
        Capability::ShaderViewportIndexLayerEXT => &[
            DeviceRequirement::Extension("ext_shader_viewport_index_layer"),
            DeviceRequirement::Extension("nv_viewport_array2"),
        ],
        Capability::ShaderViewportMaskNV => &[DeviceRequirement::Extension("nv_viewport_array2")],
        Capability::PerViewAttributesNV => &[DeviceRequirement::Extension(
            "nvx_multiview_per_view_attributes",
        )],
        Capability::StorageBuffer16BitAccess => {
            &[DeviceRequirement::Feature("storage_buffer16_bit_access")]
        }
        Capability::UniformAndStorageBuffer16BitAccess => &[DeviceRequirement::Feature(
            "uniform_and_storage_buffer16_bit_access",
        )],
        Capability::StoragePushConstant16 => {
            &[DeviceRequirement::Feature("storage_push_constant16")]
        }
        Capability::StorageInputOutput16 => &[DeviceRequirement::Feature("storage_input_output16")],
        Capability::GroupNonUniform => todo!(),
        Capability::GroupNonUniformVote => todo!(),
        Capability::GroupNonUniformArithmetic => todo!(),
        Capability::GroupNonUniformBallot => todo!(),
        Capability::GroupNonUniformShuffle => todo!(),
        Capability::GroupNonUniformShuffleRelative => todo!(),
        Capability::GroupNonUniformClustered => todo!(),
        Capability::GroupNonUniformQuad => todo!(),
        Capability::GroupNonUniformPartitionedNV => todo!(),
        Capability::SampleMaskPostDepthCoverage => {
            &[DeviceRequirement::Extension("ext_post_depth_coverage")]
        }
        Capability::ShaderNonUniform => &[
            DeviceRequirement::Version(1, 2),
            DeviceRequirement::Extension("ext_descriptor_indexing"),
        ],
        Capability::RuntimeDescriptorArray => {
            &[DeviceRequirement::Feature("runtime_descriptor_array")]
        }
        Capability::InputAttachmentArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_input_attachment_array_dynamic_indexing",
        )],
        Capability::UniformTexelBufferArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_uniform_texel_buffer_array_dynamic_indexing",
        )],
        Capability::StorageTexelBufferArrayDynamicIndexing => &[DeviceRequirement::Feature(
            "shader_storage_texel_buffer_array_dynamic_indexing",
        )],
        Capability::UniformBufferArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_uniform_buffer_array_non_uniform_indexing",
        )],
        Capability::SampledImageArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_sampled_image_array_non_uniform_indexing",
        )],
        Capability::StorageBufferArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_storage_buffer_array_non_uniform_indexing",
        )],
        Capability::StorageImageArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_storage_image_array_non_uniform_indexing",
        )],
        Capability::InputAttachmentArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_input_attachment_array_non_uniform_indexing",
        )],
        Capability::UniformTexelBufferArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_uniform_texel_buffer_array_non_uniform_indexing",
        )],
        Capability::StorageTexelBufferArrayNonUniformIndexing => &[DeviceRequirement::Feature(
            "shader_storage_texel_buffer_array_non_uniform_indexing",
        )],
        Capability::Float16 => &[
            DeviceRequirement::Feature("shader_float16"),
            DeviceRequirement::Extension("amd_gpu_shader_half_float"),
        ],
        Capability::Int8 => &[DeviceRequirement::Feature("shader_int8")],
        Capability::StorageBuffer8BitAccess => {
            &[DeviceRequirement::Feature("storage_buffer8_bit_access")]
        }
        Capability::UniformAndStorageBuffer8BitAccess => &[DeviceRequirement::Feature(
            "uniform_and_storage_buffer8_bit_access",
        )],
        Capability::StoragePushConstant8 => &[DeviceRequirement::Feature("storage_push_constant8")],
        Capability::VulkanMemoryModel => &[DeviceRequirement::Feature("vulkan_memory_model")],
        Capability::VulkanMemoryModelDeviceScope => &[DeviceRequirement::Feature(
            "vulkan_memory_model_device_scope",
        )],
        Capability::DenormPreserve => todo!(),
        Capability::DenormFlushToZero => todo!(),
        Capability::SignedZeroInfNanPreserve => todo!(),
        Capability::RoundingModeRTE => todo!(),
        Capability::RoundingModeRTZ => todo!(),
        Capability::ComputeDerivativeGroupQuadsNV => {
            &[DeviceRequirement::Feature("compute_derivative_group_quads")]
        }
        Capability::ComputeDerivativeGroupLinearNV => &[DeviceRequirement::Feature(
            "compute_derivative_group_linear",
        )],
        Capability::FragmentBarycentricNV => {
            &[DeviceRequirement::Feature("fragment_shader_barycentric")]
        }
        Capability::ImageFootprintNV => &[DeviceRequirement::Feature("image_footprint")],
        Capability::MeshShadingNV => &[DeviceRequirement::Extension("nv_mesh_shader")],
        Capability::RayTracingKHR | Capability::RayTracingProvisionalKHR => {
            &[DeviceRequirement::Feature("ray_tracing_pipeline")]
        }
        Capability::RayQueryKHR | Capability::RayQueryProvisionalKHR => &[DeviceRequirement::Feature("ray_query")],
        Capability::RayTraversalPrimitiveCullingKHR => &[DeviceRequirement::Feature(
            "ray_traversal_primitive_culling",
        )],
        Capability::RayTracingNV => &[DeviceRequirement::Extension("nv_ray_tracing")],
        // Capability::RayTracingMotionBlurNV => &[DeviceRequirement::Feature("ray_tracing_motion_blur")],
        Capability::TransformFeedback => &[DeviceRequirement::Feature("transform_feedback")],
        Capability::GeometryStreams => &[DeviceRequirement::Feature("geometry_streams")],
        Capability::FragmentDensityEXT => &[
            DeviceRequirement::Feature("fragment_density_map"),
            DeviceRequirement::Feature("shading_rate_image"),
        ],
        Capability::PhysicalStorageBufferAddresses => {
            &[DeviceRequirement::Feature("buffer_device_address")]
        }
        Capability::CooperativeMatrixNV => &[DeviceRequirement::Feature("cooperative_matrix")],
        Capability::IntegerFunctions2INTEL => {
            &[DeviceRequirement::Feature("shader_integer_functions2")]
        }
        Capability::ShaderSMBuiltinsNV => &[DeviceRequirement::Feature("shader_sm_builtins")],
        Capability::FragmentShaderSampleInterlockEXT => &[DeviceRequirement::Feature(
            "fragment_shader_sample_interlock",
        )],
        Capability::FragmentShaderPixelInterlockEXT => &[DeviceRequirement::Feature(
            "fragment_shader_pixel_interlock",
        )],
        Capability::FragmentShaderShadingRateInterlockEXT => &[
            DeviceRequirement::Feature("fragment_shader_shading_rate_interlock"),
            DeviceRequirement::Feature("shading_rate_image"),
        ],
        Capability::DemoteToHelperInvocationEXT => &[DeviceRequirement::Feature(
            "shader_demote_to_helper_invocation",
        )],
        Capability::FragmentShadingRateKHR => &[
            DeviceRequirement::Feature("pipeline_fragment_shading_rate"),
            DeviceRequirement::Feature("primitive_fragment_shading_rate"),
            DeviceRequirement::Feature("attachment_fragment_shading_rate"),
        ],
        // Capability::WorkgroupMemoryExplicitLayoutKHR => &[DeviceRequirement::Feature("workgroup_memory_explicit_layout")],
        // Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR => &[DeviceRequirement::Feature("workgroup_memory_explicit_layout8_bit_access")],
        // Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR => &[DeviceRequirement::Feature("workgroup_memory_explicit_layout16_bit_access")],
        Capability::Addresses => panic!(),        // not supported
        Capability::Linkage => panic!(),          // not supported
        Capability::Kernel => panic!(),           // not supported
        Capability::Vector16 => panic!(),         // not supported
        Capability::Float16Buffer => panic!(),    // not supported
        Capability::ImageBasic => panic!(),       // not supported
        Capability::ImageReadWrite => panic!(),   // not supported
        Capability::ImageMipmap => panic!(),      // not supported
        Capability::Pipes => panic!(),            // not supported
        Capability::Groups => panic!(),           // not supported
        Capability::DeviceEnqueue => panic!(),    // not supported
        Capability::LiteralSampler => panic!(),   // not supported
        Capability::AtomicStorage => panic!(),    // not supported
        Capability::ImageRect => panic!(),        // not supported
        Capability::SampledRect => panic!(),      // not supported
        Capability::GenericPointer => panic!(),   // not supported
        Capability::SubgroupDispatch => panic!(), // not supported
        Capability::NamedBarrier => panic!(),     // not supported
        Capability::PipeStorage => panic!(),      // not supported
        Capability::AtomicStorageOps => panic!(), // not supported
        Capability::Float16ImageAMD => panic!(),  // not supported
        Capability::ShaderStereoViewNV => panic!(), // not supported
        Capability::FragmentFullyCoveredEXT => panic!(), // not supported
        Capability::SubgroupShuffleINTEL => panic!(), // not supported
        Capability::SubgroupBufferBlockIOINTEL => panic!(), // not supported
        Capability::SubgroupImageBlockIOINTEL => panic!(), // not supported
        Capability::SubgroupImageMediaBlockIOINTEL => panic!(), // not supported
        Capability::SubgroupAvcMotionEstimationINTEL => panic!(), // not supported
        Capability::SubgroupAvcMotionEstimationIntraINTEL => panic!(), // not supported
        Capability::SubgroupAvcMotionEstimationChromaINTEL => panic!(), // not supported
        Capability::FunctionPointersINTEL => panic!(), // not supported
        Capability::IndirectReferencesINTEL => panic!(), // not supported
        Capability::FPGAKernelAttributesINTEL => panic!(), // not supported
        Capability::FPGALoopControlsINTEL => panic!(), // not supported
        Capability::FPGAMemoryAttributesINTEL => panic!(), // not supported
        Capability::FPGARegINTEL => panic!(), // not supported
        Capability::UnstructuredLoopControlsINTEL => panic!(), // not supported
        Capability::KernelAttributesINTEL => panic!(), // not supported
        Capability::BlockingPipesINTEL => panic!(), // not supported
    }
}

/// Returns the Vulkan device requirement for a SPIR-V storage class.
fn storage_class_requirement(storage_class: &StorageClass) -> &'static [DeviceRequirement] {
    match *storage_class {
        StorageClass::UniformConstant => &[],
        StorageClass::Input => &[],
        StorageClass::Uniform => &[],
        StorageClass::Output => &[],
        StorageClass::Workgroup => &[],
        StorageClass::CrossWorkgroup => &[],
        StorageClass::Private => &[],
        StorageClass::Function => &[],
        StorageClass::Generic => &[],
        StorageClass::PushConstant => &[],
        StorageClass::AtomicCounter => &[],
        StorageClass::Image => &[],
        StorageClass::StorageBuffer => &[DeviceRequirement::Extension(
            "khr_storage_buffer_storage_class",
        )],
        StorageClass::CallableDataKHR => todo!(),
        StorageClass::IncomingCallableDataKHR => todo!(),
        StorageClass::RayPayloadKHR => todo!(),
        StorageClass::HitAttributeKHR => todo!(),
        StorageClass::IncomingRayPayloadKHR => todo!(),
        StorageClass::ShaderRecordBufferKHR => todo!(),
        StorageClass::PhysicalStorageBuffer => todo!(),
        StorageClass::CodeSectionINTEL => todo!(),
    }
}

enum DeviceRequirement {
    Feature(&'static str),
    Extension(&'static str),
    Version(u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[cfg(not(target_os = "windows"))]
    pub fn path_separator() -> &'static str {
        "/"
    }

    #[cfg(target_os = "windows")]
    pub fn path_separator() -> &'static str {
        "\\"
    }

    fn convert_paths(root_path: &Path, paths: &[String]) -> Vec<String> {
        paths
            .iter()
            .map(|p| path_to_str(root_path.join(p).as_path()).to_owned())
            .collect()
    }

    #[test]
    fn test_bad_alignment() {
        // vec3/mat3/mat3x* are problematic in arrays since their rust
        // representations don't have the same array stride as the SPIR-V
        // ones. E.g. in a vec3[2], the second element starts on the 16th
        // byte, but in a rust [[f32;3];2], the second element starts on the
        // 12th byte. Since we can't generate code for these types, we should
        // create an error instead of generating incorrect code.
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        struct MyStruct {
            vec3 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        let res = std::panic::catch_unwind(|| {
            structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new())
        });
        assert!(res.is_err());
    }
    #[test]
    fn test_trivial_alignment() {
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        struct MyStruct {
            vec4 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new());
    }
    #[test]
    fn test_wrap_alignment() {
        // This is a workaround suggested in the case of test_bad_alignment,
        // so we should make sure it works.
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        struct Vec3Wrap {
            vec3 v;
        };
        struct MyStruct {
            Vec3Wrap vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new());
    }
    #[test]
    fn test_vector_double_attributes() {
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450
        layout( location = 0 ) in dvec4 d4v;
        layout( location = 2 ) in double d4a[4];
        void main() {}
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let spirv = Spirv::new(comp.as_binary()).unwrap();
        structs::write_structs("", &spirv, &TypesMeta::default(), &mut HashMap::new());
    }

    #[test]
    fn test_include_resolution() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let empty_includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (_compile_relative, _) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include \"include_dir_a/target_a.glsl\"
        #include \"include_dir_b/target_b.glsl\"
        void main() {}
        ",
            ShaderKind::Vertex,
            &empty_includes,
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");

        let (_compile_include_paths, includes) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include <target_a.glsl>
        #include <target_b.glsl>
        void main() {}
        ",
            ShaderKind::Vertex,
            &[
                root_path.join("tests").join("include_dir_a"),
                root_path.join("tests").join("include_dir_b"),
            ],
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes,
            convert_paths(
                &root_path,
                &[
                    vec!["tests", "include_dir_a", "target_a.glsl"].join(path_separator()),
                    vec!["tests", "include_dir_b", "target_b.glsl"].join(path_separator()),
                ]
            )
        );

        let (_compile_include_paths_with_relative, includes_with_relative) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include <target_a.glsl>
        #include <../include_dir_b/target_b.glsl>
        void main() {}
        ",
            ShaderKind::Vertex,
            &[root_path.join("tests").join("include_dir_a")],
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes_with_relative,
            convert_paths(
                &root_path,
                &[
                    vec!["tests", "include_dir_a", "target_a.glsl"].join(path_separator()),
                    vec!["tests", "include_dir_a", "../include_dir_b/target_b.glsl"]
                        .join(path_separator()),
                ]
            )
        );

        let absolute_path = root_path
            .join("tests")
            .join("include_dir_a")
            .join("target_a.glsl");
        let absolute_path_str = absolute_path
            .to_str()
            .expect("Cannot run tests in a folder with non unicode characters");
        let (_compile_absolute_path, includes_absolute_path) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            &format!(
                "
        #version 450
        #include \"{}\"
        void main() {{}}
        ",
                absolute_path_str
            ),
            ShaderKind::Vertex,
            &empty_includes,
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes_absolute_path,
            convert_paths(
                &root_path,
                &[vec!["tests", "include_dir_a", "target_a.glsl"].join(path_separator())]
            )
        );

        let (_compile_recursive_, includes_recursive) = compile(
            Some(String::from("tests/include_test.glsl")),
            &root_path,
            "
        #version 450
        #include <target_c.glsl>
        void main() {}
        ",
            ShaderKind::Vertex,
            &[
                root_path.join("tests").join("include_dir_b"),
                root_path.join("tests").join("include_dir_c"),
            ],
            &defines,
            None,
            None,
        )
        .expect("Cannot resolve include files");
        assert_eq!(
            includes_recursive,
            convert_paths(
                &root_path,
                &[
                    vec!["tests", "include_dir_c", "target_c.glsl"].join(path_separator()),
                    vec!["tests", "include_dir_c", "../include_dir_a/target_a.glsl"]
                        .join(path_separator()),
                    vec!["tests", "include_dir_b", "target_b.glsl"].join(path_separator()),
                ]
            )
        );
    }

    #[test]
    fn test_macros() {
        let empty_includes: [PathBuf; 0] = [];
        let defines = vec![("NAME1", ""), ("NAME2", "58")];
        let no_defines: [(String, String); 0] = [];
        let need_defines = "
        #version 450
        #if defined(NAME1) && NAME2 > 29
        void main() {}
        #endif
        ";
        let compile_no_defines = compile(
            None,
            &Path::new(""),
            need_defines,
            ShaderKind::Vertex,
            &empty_includes,
            &no_defines,
            None,
            None,
        );
        assert!(compile_no_defines.is_err());

        let compile_defines = compile(
            None,
            &Path::new(""),
            need_defines,
            ShaderKind::Vertex,
            &empty_includes,
            &defines,
            None,
            None,
        );
        compile_defines.expect("Setting shader macros did not work");
    }
}
