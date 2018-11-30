// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::io::Error as IoError;
use std::path::Path;

use syn::Ident;
use proc_macro2::{Span, TokenStream};
use shaderc::{Compiler, CompileOptions};

pub use shaderc::{CompilationArtifact, ShaderKind, IncludeType, ResolvedInclude};
pub use parse::ParseError;

use parse::Instruction;
use enums::Capability;

use parse;
use entry_point;
use structs;
use descriptor_sets;
use spec_consts;
use read_file_to_string;

fn include_callback(requested_source_path_raw: &str, directive_type: IncludeType,
                    contained_within_path_raw: &str, recursion_depth: usize,
                    include_directories: &[String], root_source_has_path: bool) -> Result<ResolvedInclude, String> {
    let file_to_include = match directive_type {
        IncludeType::Relative => {
            let requested_source_path = Path::new(requested_source_path_raw);

            // Is embedded current shader source embedded within a rust macro?
            // If so, abort.
            if !root_source_has_path && recursion_depth == 1 {
                let requested_source_name = requested_source_path.file_name()
                    .expect("Could not get the name of the requested source file.")
                    .to_string_lossy();
                let requested_source_directory = requested_source_path.parent()
                    .expect("Could not get the directory of the requested source file.")
                    .to_string_lossy();

                return Err(format!("Usage of relative paths in imports in embedded GLSL is not \
                                    allowed, try using `#include <{}>` and adding the directory \
                                    `{}` to the `include` array in your `shader!` macro call \
                                    instead.",
                                   requested_source_name, requested_source_directory));
            }

            let parent_of_current_source = Path::new(contained_within_path_raw).parent()
                .unwrap_or_else(|| panic!("The file `{}` does not reside in a directory. This is \
                                           an implementation error.",
                                          contained_within_path_raw));
            let resolved_requested_source_path = parent_of_current_source.join(requested_source_path);

            if !resolved_requested_source_path.is_file() {
                return Err(format!("Invalid inclusion path `{}`, the path does not point to a file.",
                                   requested_source_path_raw));
            }

            resolved_requested_source_path
        },
        IncludeType::Standard => {
            let requested_source_path = Path::new(requested_source_path_raw);

            if requested_source_path.is_absolute() {
                // This message is printed either when using a missing file with an absolute path
                // in the relative include directive or when using absolute paths in a standard
                // include directive.
                return Err(format!("No such file found, as specified by the absolute path. \
                                    Keep in mind, that absolute paths cannot be used with \
                                    inclusion from standard directories (`#include <...>`), try \
                                    using `#include \"...\"` instead. Requested path: {}",
                                   requested_source_path_raw));
            }

            let mut found_requested_source_path = None;

            for include_directory in include_directories {
                let include_directory_path = Path::new(include_directory).canonicalize()
                    .unwrap_or_else(|_| panic!("Invalid standard shader inclusion directory `{}`.",
                                               include_directory));
                let resolved_requested_source_path_rel = include_directory_path
                    .join(requested_source_path);
                let resolved_requested_source_path = resolved_requested_source_path_rel
                    .canonicalize()
                    .map_err(|_| format!("Invalid inclusion path `{}`.",
                                         resolved_requested_source_path_rel.to_string_lossy()))?;

                if !resolved_requested_source_path.starts_with(include_directory_path) {
                    return Err(format!("Cannot use `..` with inclusion from standard directories \
                                        (`#include <...>`), try using `#include \"...\"` instead. \
                                        Requested path: {}", requested_source_path.to_string_lossy()));
                }

                if resolved_requested_source_path.is_file() {
                    found_requested_source_path = Some(resolved_requested_source_path);
                    break;
                }
            }

            if found_requested_source_path.is_none() {
                return Err(format!("Could not include the file `{}` from any include directories.",
                                   requested_source_path_raw));
            }

            found_requested_source_path.unwrap()
        },
    };

    let canonical_file_to_include = file_to_include.canonicalize()
        .unwrap_or_else(|_| file_to_include);
    let canonical_file_to_include_string = canonical_file_to_include.to_str()
        .expect("Could not stringify the file to be included. Make sure the path consists of \
                 valid unicode characters.")
        .to_string();
    let content = read_file_to_string(canonical_file_to_include.as_path())
        .map_err(|_| format!("Could not read the contents of file `{}` to be included in the \
                              shader source.",
                              &canonical_file_to_include_string))?;

    Ok(ResolvedInclude {
        resolved_name: canonical_file_to_include_string,
        content,
    })
}

pub fn compile(path: Option<String>, code: &str, ty: ShaderKind, include_directories: &[String]) -> Result<CompilationArtifact, String> {
    let mut compiler = Compiler::new().ok_or("failed to create GLSL compiler")?;
    let mut compile_options = CompileOptions::new()
        .ok_or("failed to initialize compile option")?;
    let root_source_path = if let &Some(ref path) = &path {
        path
    } else {
        // An arbitrary placeholder file name for embedded shaders
        "shader.glsl"
    };

    // Specify file resolution callback for the `#include` directive
    compile_options.set_include_callback(|requested_source_path, directive_type,
                                          contained_within_path, recursion_depth| {
        include_callback(requested_source_path, directive_type, contained_within_path,
                         recursion_depth, include_directories, path.is_some())
    });

    let content = compiler
        .compile_into_spirv(&code, ty, root_source_path, "main", Some(&compile_options))
        .map_err(|e| e.to_string())?;

    Ok(content)
}

pub fn reflect(name: &str, spirv: &[u32], dump: bool) -> Result<TokenStream, Error> {
    let struct_name = Ident::new(&name, Span::call_site());
    let doc = parse::parse_spirv(spirv)?;

    // checking whether each required capability is enabled in the Vulkan device
    let mut cap_checks: Vec<TokenStream> = vec!();
    for i in doc.instructions.iter() {
        if let &Instruction::Capability(ref cap) = i {
            if let Some(cap_string) = capability_name(cap) {
                let cap = Ident::new(cap_string, Span::call_site());
                cap_checks.push(quote!{
                    if !device.enabled_features().#cap {
                        panic!("capability {:?} not enabled", #cap_string); // TODO: error
                        //return Err(CapabilityNotEnabled);
                    }
                });
            }
        }
    }

    // writing one method for each entry point of this module
    let mut entry_points_inside_impl: Vec<TokenStream> = vec!();
    let mut entry_points_outside_impl: Vec<TokenStream> = vec!();
    for instruction in doc.instructions.iter() {
        if let &Instruction::EntryPoint { .. } = instruction {
            let (outside, entry_point) = entry_point::write_entry_point(&doc, instruction);
            entry_points_inside_impl.push(entry_point);
            entry_points_outside_impl.push(outside);
        }
    }

    let structs = structs::write_structs(&doc);
    let descriptor_sets = descriptor_sets::write_descriptor_sets(&doc);
    let specialization_constants = spec_consts::write_specialization_constants(&doc);
    let ast = quote!{
        #[allow(unused_imports)]
        use std::sync::Arc;
        #[allow(unused_imports)]
        use std::vec::IntoIter as VecIntoIter;

        #[allow(unused_imports)]
        use vulkano::device::Device;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorDescTy;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorBufferDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::DescriptorImageDescArray;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor::ShaderStages;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::DescriptorSet;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::UnsafeDescriptorSet;
        #[allow(unused_imports)]
        use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayout;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
        #[allow(unused_imports)]
        use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationConstants as SpecConstsTrait;
        #[allow(unused_imports)]
        use vulkano::pipeline::shader::SpecializationMapEntry;

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
                #( #cap_checks )*
                let words = [ #( #spirv ),* ];

                unsafe {
                    Ok(#struct_name {
                        shader: try!(::vulkano::pipeline::shader::ShaderModule::from_words(device, &words))
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

        #( #entry_points_outside_impl )*

        pub mod ty {
            #structs
        }

        #descriptor_sets
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

/// Returns the name of the Vulkan something that corresponds to an `OpCapability`.
///
/// Returns `None` if irrelevant.
// TODO: this function is a draft, as the actual names may not be the same
fn capability_name(cap: &Capability) -> Option<&'static str> {
    match *cap {
        Capability::CapabilityMatrix => None,        // always supported
        Capability::CapabilityShader => None,        // always supported
        Capability::CapabilityGeometry => Some("geometry_shader"),
        Capability::CapabilityTessellation => Some("tessellation_shader"),
        Capability::CapabilityAddresses => panic!(), // not supported
        Capability::CapabilityLinkage => panic!(),   // not supported
        Capability::CapabilityKernel => panic!(),    // not supported
        Capability::CapabilityVector16 => panic!(),  // not supported
        Capability::CapabilityFloat16Buffer => panic!(), // not supported
        Capability::CapabilityFloat16 => panic!(),   // not supported
        Capability::CapabilityFloat64 => Some("shader_f3264"),
        Capability::CapabilityInt64 => Some("shader_int64"),
        Capability::CapabilityInt64Atomics => panic!(),  // not supported
        Capability::CapabilityImageBasic => panic!(),    // not supported
        Capability::CapabilityImageReadWrite => panic!(),    // not supported
        Capability::CapabilityImageMipmap => panic!(),   // not supported
        Capability::CapabilityPipes => panic!(), // not supported
        Capability::CapabilityGroups => panic!(),    // not supported
        Capability::CapabilityDeviceEnqueue => panic!(), // not supported
        Capability::CapabilityLiteralSampler => panic!(),    // not supported
        Capability::CapabilityAtomicStorage => panic!(), // not supported
        Capability::CapabilityInt16 => Some("shader_int16"),
        Capability::CapabilityTessellationPointSize =>
            Some("shader_tessellation_and_geometry_point_size"),
        Capability::CapabilityGeometryPointSize =>
            Some("shader_tessellation_and_geometry_point_size"),
        Capability::CapabilityImageGatherExtended => Some("shader_image_gather_extended"),
        Capability::CapabilityStorageImageMultisample =>
            Some("shader_storage_image_multisample"),
        Capability::CapabilityUniformBufferArrayDynamicIndexing =>
            Some("shader_uniform_buffer_array_dynamic_indexing"),
        Capability::CapabilitySampledImageArrayDynamicIndexing =>
            Some("shader_sampled_image_array_dynamic_indexing"),
        Capability::CapabilityStorageBufferArrayDynamicIndexing =>
            Some("shader_storage_buffer_array_dynamic_indexing"),
        Capability::CapabilityStorageImageArrayDynamicIndexing =>
            Some("shader_storage_image_array_dynamic_indexing"),
        Capability::CapabilityClipDistance => Some("shader_clip_distance"),
        Capability::CapabilityCullDistance => Some("shader_cull_distance"),
        Capability::CapabilityImageCubeArray => Some("image_cube_array"),
        Capability::CapabilitySampleRateShading => Some("sample_rate_shading"),
        Capability::CapabilityImageRect => panic!(), // not supported
        Capability::CapabilitySampledRect => panic!(),   // not supported
        Capability::CapabilityGenericPointer => panic!(),    // not supported
        Capability::CapabilityInt8 => panic!(),  // not supported
        Capability::CapabilityInputAttachment => None,       // always supported
        Capability::CapabilitySparseResidency => Some("shader_resource_residency"),
        Capability::CapabilityMinLod => Some("shader_resource_min_lod"),
        Capability::CapabilitySampled1D => None,        // always supported
        Capability::CapabilityImage1D => None,        // always supported
        Capability::CapabilitySampledCubeArray => Some("image_cube_array"),
        Capability::CapabilitySampledBuffer => None,         // always supported
        Capability::CapabilityImageBuffer => None,        // always supported
        Capability::CapabilityImageMSArray => Some("shader_storage_image_multisample"),
        Capability::CapabilityStorageImageExtendedFormats =>
            Some("shader_storage_image_extended_formats"),
        Capability::CapabilityImageQuery => None,        // always supported
        Capability::CapabilityDerivativeControl => None,        // always supported
        Capability::CapabilityInterpolationFunction => Some("sample_rate_shading"),
        Capability::CapabilityTransformFeedback => panic!(), // not supported
        Capability::CapabilityGeometryStreams => panic!(),   // not supported
        Capability::CapabilityStorageImageReadWithoutFormat =>
            Some("shader_storage_image_read_without_format"),
        Capability::CapabilityStorageImageWriteWithoutFormat =>
            Some("shader_storage_image_write_without_format"),
        Capability::CapabilityMultiViewport => Some("multi_viewport"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bad_alignment() {
        // vec3/mat3/mat3x* are problematic in arrays since their rust
        // representations don't have the same array stride as the SPIR-V
        // ones. E.g. in a vec3[2], the second element starts on the 16th
        // byte, but in a rust [[f32;3];2], the second element starts on the
        // 12th byte. Since we can't generate code for these types, we should
        // create an error instead of generating incorrect code.
        let comp = compile(None, "
        #version 450
        struct MyStruct {
            vec3 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ", ShaderKind::Vertex, &[]).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        let res = std::panic::catch_unwind(|| structs::write_structs(&doc));
        assert!(res.is_err());
    }
    #[test]
    fn test_trivial_alignment() {
        let comp = compile(None, "
        #version 450
        struct MyStruct {
            vec4 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ", ShaderKind::Vertex, &[]).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc);
    }
    #[test]
    fn test_wrap_alignment() {
        // This is a workaround suggested in the case of test_bad_alignment,
        // so we should make sure it works.
        let comp = compile(None, "
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
        ", ShaderKind::Vertex, &[]).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc);
    }
}
