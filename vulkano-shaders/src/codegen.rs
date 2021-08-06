// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::entry_point;
use crate::parse;
use crate::parse::Instruction;
pub use crate::parse::ParseError;
use crate::read_file_to_string;
use crate::spec_consts;
use crate::structs;
use crate::TypesMeta;
use proc_macro2::{Span, TokenStream};
pub use shaderc::{CompilationArtifact, IncludeType, ResolvedInclude, ShaderKind};
use shaderc::{CompileOptions, Compiler, EnvVersion, SpirvVersion, TargetEnv};
use spirv_headers::{Capability, StorageClass};
use std::iter::Iterator;
use std::path::Path;
use std::{
    cell::{RefCell, RefMut},
    io::Error as IoError,
};
use syn::Ident;

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
    name: &str,
    spirv: &[u32],
    types_meta: TypesMeta,
    input_paths: I,
    exact_entrypoint_interface: bool,
    dump: bool,
) -> Result<TokenStream, Error>
where
    I: Iterator<Item = &'a str>,
{
    let struct_name = Ident::new(&name, Span::call_site());
    let doc = parse::parse_spirv(spirv)?;

    // checking whether each required capability is enabled in the Vulkan device
    let mut cap_checks: Vec<TokenStream> = vec![];
    match doc.version {
        (1, 0) => {}
        (1, 1) | (1, 2) | (1, 3) => {
            cap_checks.push(quote! {
                if device.api_version() < Version::V1_1 {
                    panic!("Device API version 1.1 required");
                }
            });
        }
        (1, 4) => {
            cap_checks.push(quote! {
                if device.api_version() < Version::V1_2
                    && !device.enabled_extensions().khr_spirv_1_4 {
                    panic!("Device API version 1.2 or extension VK_KHR_spirv_1_4 required");
                }
            });
        }
        (1, 5) => {
            cap_checks.push(quote! {
                if device.api_version() < Version::V1_2 {
                    panic!("Device API version 1.2 required");
                }
            });
        }
        _ => return Err(Error::UnsupportedSpirvVersion),
    }

    for i in doc.instructions.iter() {
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
                    type_id: _,
                } => storage_class_requirement(storage_class),
                Instruction::Capability(cap) => capability_requirement(cap),
                _ => &[],
            }
        };

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
            if !std::array::IntoIter::new([#(#conditions),*]).any(|x| x) {
                panic!("One of the following must be enabled on the device: {}", #messages);
            }
        });
    }

    // writing one method for each entry point of this module
    let mut entry_points_inside_impl: Vec<TokenStream> = vec![];
    for instruction in doc.instructions.iter() {
        if let &Instruction::EntryPoint { .. } = instruction {
            let entry_point = entry_point::write_entry_point(
                &doc,
                instruction,
                &types_meta,
                exact_entrypoint_interface,
            );
            entry_points_inside_impl.push(entry_point);
        }
    }

    let include_bytes = input_paths.map(|s| {
        quote! {
            // using include_bytes here ensures that changing the shader will force recompilation.
            // The bytes themselves can be optimized out by the compiler as they are unused.
            ::std::include_bytes!( #s )
        }
    });

    let structs = structs::write_structs(&doc, &types_meta);
    let specialization_constants = spec_consts::write_specialization_constants(&doc, &types_meta);
    let uses = &types_meta.uses;
    let ast = quote! {
        #[allow(unused_imports)]
        use std::sync::Arc;
        #[allow(unused_imports)]
        use std::vec::IntoIter as VecIntoIter;

        #[allow(unused_imports)]
        use vulkano::device::Device;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorDescTy;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorBufferDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorImageDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorImageDescDimensions;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorImageDescArray;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorSetDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::layout::DescriptorSetLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor_set::DescriptorSet;
        #[allow(unused_imports)]
        use vulkano::format::Format;
        #[allow(unused_imports)]
        use vulkano::pipeline::layout::PipelineLayout;
        #[allow(unused_imports)]
        use vulkano::pipeline::layout::PipelineLayoutPcRange;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::ShaderStages;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationConstants as SpecConstsTrait;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationMapEntry;
        #[allow(unused_imports)]
        use vulkano::Version;

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
                static WORDS: &[u32] = &[ #( #spirv ),* ];

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

        pub mod ty {
            #( #uses )*
            #structs
        }

        #specialization_constants
    };

    if dump {
        println!("{}", ast.to_string());
        panic!("`shader!` rust codegen dumped") // TODO: use span from dump
    }

    Ok(ast)
}

#[derive(Debug)]
pub enum Error {
    UnsupportedSpirvVersion,
    IoError(IoError),
    ParseError(ParseError),
}

impl From<IoError> for Error {
    #[inline]
    fn from(err: IoError) -> Error {
        Error::IoError(err)
    }
}

impl From<ParseError> for Error {
    #[inline]
    fn from(err: ParseError) -> Error {
        Error::ParseError(err)
    }
}

/// Returns the Vulkan device requirement for a SPIR-V `OpCapability`.
// TODO: this function is a draft, as the actual names may not be the same
fn capability_requirement(cap: &Capability) -> &'static [DeviceRequirement] {
    match *cap {
        Capability::Matrix => &[],
        Capability::Shader => &[],
        Capability::Geometry => &[DeviceRequirement::Feature("geometry_shader")],
        Capability::Tessellation => &[DeviceRequirement::Feature("tessellation_shader")],
        Capability::Addresses => panic!(),     // not supported
        Capability::Linkage => panic!(),       // not supported
        Capability::Kernel => panic!(),        // not supported
        Capability::Vector16 => panic!(),      // not supported
        Capability::Float16Buffer => panic!(), // not supported
        Capability::Float16 => panic!(),       // not supported
        Capability::Float64 => &[DeviceRequirement::Feature("shader_float64")],
        Capability::Int64 => &[DeviceRequirement::Feature("shader_int64")],
        Capability::Int64Atomics => panic!(),   // not supported
        Capability::ImageBasic => panic!(),     // not supported
        Capability::ImageReadWrite => panic!(), // not supported
        Capability::ImageMipmap => panic!(),    // not supported
        Capability::Pipes => panic!(),          // not supported
        Capability::Groups => panic!(),         // not supported
        Capability::DeviceEnqueue => panic!(),  // not supported
        Capability::LiteralSampler => panic!(), // not supported
        Capability::AtomicStorage => panic!(),  // not supported
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
        Capability::ImageRect => panic!(),      // not supported
        Capability::SampledRect => panic!(),    // not supported
        Capability::GenericPointer => panic!(), // not supported
        Capability::Int8 => &[DeviceRequirement::Extension("khr_8bit_storage")],
        Capability::InputAttachment => &[],
        Capability::SparseResidency => &[DeviceRequirement::Feature("shader_resource_residency")],
        Capability::MinLod => &[DeviceRequirement::Feature("shader_resource_min_lod")],
        Capability::Sampled1D => &[],
        Capability::Image1D => &[],
        Capability::SampledCubeArray => &[DeviceRequirement::Feature("image_cube_array")],
        Capability::SampledBuffer => &[],
        Capability::ImageBuffer => &[],
        Capability::ImageMSArray => &[DeviceRequirement::Feature(
            "shader_storage_image_multisample",
        )],
        Capability::StorageImageExtendedFormats => &[DeviceRequirement::Feature(
            "shader_storage_image_extended_formats",
        )],
        Capability::ImageQuery => &[],
        Capability::DerivativeControl => &[],
        Capability::InterpolationFunction => &[DeviceRequirement::Feature("sample_rate_shading")],
        Capability::TransformFeedback => panic!(), // not supported
        Capability::GeometryStreams => panic!(),   // not supported
        Capability::StorageImageReadWithoutFormat => &[DeviceRequirement::Feature(
            "shader_storage_image_read_without_format",
        )],
        Capability::StorageImageWriteWithoutFormat => &[DeviceRequirement::Feature(
            "shader_storage_image_write_without_format",
        )],
        Capability::MultiViewport => &[DeviceRequirement::Feature("multi_viewport")],
        Capability::DrawParameters => &[DeviceRequirement::Feature("shader_draw_parameters")],
        Capability::StorageBuffer16BitAccess => {
            &[DeviceRequirement::Extension("khr_16bit_storage")]
        }
        Capability::UniformAndStorageBuffer16BitAccess => {
            &[DeviceRequirement::Extension("khr_16bit_storage")]
        }
        Capability::StoragePushConstant16 => &[DeviceRequirement::Extension("khr_16bit_storage")],
        Capability::StorageInputOutput16 => &[DeviceRequirement::Extension("khr_16bit_storage")],
        Capability::MultiView => &[DeviceRequirement::Feature("multiview")],
        Capability::StoragePushConstant8 => &[DeviceRequirement::Extension("khr_8bit_storage")],
        Capability::SubgroupDispatch => todo!(),
        Capability::NamedBarrier => todo!(),
        Capability::PipeStorage => todo!(),
        Capability::GroupNonUniform => todo!(),
        Capability::GroupNonUniformVote => todo!(),
        Capability::GroupNonUniformArithmetic => todo!(),
        Capability::GroupNonUniformBallot => todo!(),
        Capability::GroupNonUniformShuffle => todo!(),
        Capability::GroupNonUniformShuffleRelative => todo!(),
        Capability::GroupNonUniformClustered => todo!(),
        Capability::GroupNonUniformQuad => todo!(),
        Capability::ShaderLayer => todo!(),
        Capability::ShaderViewportIndex => todo!(),
        Capability::SubgroupBallotKHR => todo!(),
        Capability::SubgroupVoteKHR => todo!(),
        Capability::DeviceGroup => todo!(),
        Capability::VariablePointersStorageBuffer => todo!(),
        Capability::VariablePointers => todo!(),
        Capability::AtomicStorageOps => todo!(),
        Capability::SampleMaskPostDepthCoverage => todo!(),
        Capability::StorageBuffer8BitAccess => todo!(),
        Capability::UniformAndStorageBuffer8BitAccess => {
            &[DeviceRequirement::Extension("khr_8bit_storage")]
        }
        Capability::DenormPreserve => todo!(),
        Capability::DenormFlushToZero => todo!(),
        Capability::SignedZeroInfNanPreserve => todo!(),
        Capability::RoundingModeRTE => todo!(),
        Capability::RoundingModeRTZ => todo!(),
        Capability::RayQueryProvisionalKHR => todo!(),
        Capability::RayTraversalPrimitiveCullingProvisionalKHR => todo!(),
        Capability::Float16ImageAMD => todo!(),
        Capability::ImageGatherBiasLodAMD => todo!(),
        Capability::FragmentMaskAMD => todo!(),
        Capability::StencilExportEXT => todo!(),
        Capability::ImageReadWriteLodAMD => todo!(),
        Capability::ShaderClockKHR => todo!(),
        Capability::SampleMaskOverrideCoverageNV => todo!(),
        Capability::GeometryShaderPassthroughNV => todo!(),
        Capability::ShaderViewportIndexLayerEXT => todo!(),
        Capability::ShaderViewportMaskNV => todo!(),
        Capability::ShaderStereoViewNV => todo!(),
        Capability::PerViewAttributesNV => todo!(),
        Capability::FragmentFullyCoveredEXT => todo!(),
        Capability::MeshShadingNV => todo!(),
        Capability::ImageFootprintNV => todo!(),
        Capability::FragmentBarycentricNV => todo!(),
        Capability::ComputeDerivativeGroupQuadsNV => todo!(),
        Capability::FragmentDensityEXT => todo!(),
        Capability::GroupNonUniformPartitionedNV => todo!(),
        Capability::ShaderNonUniform => todo!(),
        Capability::RuntimeDescriptorArray => todo!(),
        Capability::InputAttachmentArrayDynamicIndexing => todo!(),
        Capability::UniformTexelBufferArrayDynamicIndexing => todo!(),
        Capability::StorageTexelBufferArrayDynamicIndexing => todo!(),
        Capability::UniformBufferArrayNonUniformIndexing => todo!(),
        Capability::SampledImageArrayNonUniformIndexing => todo!(),
        Capability::StorageBufferArrayNonUniformIndexing => todo!(),
        Capability::StorageImageArrayNonUniformIndexing => todo!(),
        Capability::InputAttachmentArrayNonUniformIndexing => todo!(),
        Capability::UniformTexelBufferArrayNonUniformIndexing => todo!(),
        Capability::StorageTexelBufferArrayNonUniformIndexing => todo!(),
        Capability::RayTracingNV => todo!(),
        Capability::VulkanMemoryModel => todo!(),
        Capability::VulkanMemoryModelDeviceScope => todo!(),
        Capability::PhysicalStorageBufferAddresses => todo!(),
        Capability::ComputeDerivativeGroupLinearNV => todo!(),
        Capability::RayTracingProvisionalKHR => todo!(),
        Capability::CooperativeMatrixNV => todo!(),
        Capability::FragmentShaderSampleInterlockEXT => todo!(),
        Capability::FragmentShaderShadingRateInterlockEXT => todo!(),
        Capability::ShaderSMBuiltinsNV => todo!(),
        Capability::FragmentShaderPixelInterlockEXT => todo!(),
        Capability::DemoteToHelperInvocationEXT => todo!(),
        Capability::SubgroupShuffleINTEL => todo!(),
        Capability::SubgroupBufferBlockIOINTEL => todo!(),
        Capability::SubgroupImageBlockIOINTEL => todo!(),
        Capability::SubgroupImageMediaBlockIOINTEL => todo!(),
        Capability::IntegerFunctions2INTEL => todo!(),
        Capability::SubgroupAvcMotionEstimationINTEL => todo!(),
        Capability::SubgroupAvcMotionEstimationIntraINTEL => todo!(),
        Capability::SubgroupAvcMotionEstimationChromaINTEL => todo!(),
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
        StorageClass::CallableDataNV => todo!(),
        StorageClass::IncomingCallableDataNV => todo!(),
        StorageClass::RayPayloadNV => todo!(),
        StorageClass::HitAttributeNV => todo!(),
        StorageClass::IncomingRayPayloadNV => todo!(),
        StorageClass::ShaderRecordBufferNV => todo!(),
        StorageClass::PhysicalStorageBuffer => todo!(),
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
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        let res = std::panic::catch_unwind(|| structs::write_structs(&doc, &TypesMeta::default()));
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
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc, &TypesMeta::default());
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
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc, &TypesMeta::default());
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
