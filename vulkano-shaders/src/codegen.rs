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
use shaderc::{Compiler, CompileOptions, TargetEnv};

pub use shaderc::{CompilationArtifact, ShaderKind, IncludeType, ResolvedInclude};
pub use crate::parse::ParseError;

use crate::parse::Instruction;
use crate::enums::Capability;
use crate::enums::StorageClass;

use crate::parse;
use crate::entry_point;
use crate::structs;
use crate::descriptor_sets;
use crate::spec_consts;
use crate::read_file_to_string;

fn include_callback(requested_source_path_raw: &str, directive_type: IncludeType,
                    contained_within_path_raw: &str, recursion_depth: usize,
                    include_directories: &[impl AsRef<Path>], root_source_has_path: bool,
                    base_path: &impl AsRef<Path>) -> Result<ResolvedInclude, String> {
    let file_to_include = match directive_type {
        IncludeType::Relative => {
            let requested_source_path = Path::new(requested_source_path_raw);
            // Is embedded current shader source embedded within a rust macro?
            // If so, abort unless absolute path.
            if !root_source_has_path && recursion_depth == 1 && !requested_source_path.is_absolute() {
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

            let mut resolved_path = if recursion_depth == 1 {
                Path::new(contained_within_path_raw).parent().map(|parent| base_path.as_ref().join(parent))
            } else {
                Path::new(contained_within_path_raw).parent().map(|parent| parent.to_owned())
            }.unwrap_or_else(|| panic!("The file `{}` does not reside in a directory. This is \
                                        an implementation error.",
                                        contained_within_path_raw));
            resolved_path.push(requested_source_path);

            if !resolved_path.is_file() {
                return Err(format!("Invalid inclusion path `{}`, the path does not point to a file.",
                                   requested_source_path_raw));
            }

            resolved_path
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

            let found_requested_source_path = include_directories.iter().map(|include_directory| include_directory.as_ref().join(requested_source_path)).find(
                |resolved_requested_source_path| resolved_requested_source_path.is_file()
            );

            if let Some(found_requested_source_path) = found_requested_source_path {
                found_requested_source_path
            } else {
                return Err(format!("Could not include the file `{}` from any include directories.",
                                   requested_source_path_raw));
            }
        },
    };

    let file_to_include_string = file_to_include.to_str()
        .expect("Could not stringify the file to be included. Make sure the path consists of \
                 valid unicode characters.")
        .to_string();
    let content = read_file_to_string(file_to_include.as_path())
        .map_err(|_| format!("Could not read the contents of file `{}` to be included in the \
                              shader source.",
                              &file_to_include_string))?;

    Ok(ResolvedInclude {
        resolved_name: file_to_include_string,
        content,
    })
}

pub fn compile(path: Option<String>, base_path: &impl AsRef<Path>, code: &str, ty: ShaderKind, include_directories: &[impl AsRef<Path>], macro_defines: &[(impl AsRef<str>, impl AsRef<str>)]) -> Result<CompilationArtifact, String> {
    let mut compiler = Compiler::new().ok_or("failed to create GLSL compiler")?;
    let mut compile_options = CompileOptions::new()
        .ok_or("failed to initialize compile option")?;
    const ENV_VULKAN_VERSION: u32 = ((1 << 22) | (1 << 12));
    compile_options.set_target_env(TargetEnv::Vulkan, ENV_VULKAN_VERSION);
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
                         recursion_depth, include_directories, path.is_some(), base_path)
    });

    for (macro_name, macro_value) in macro_defines.iter() {
        compile_options.add_macro_definition(macro_name.as_ref(), Some(macro_value.as_ref()));
    }

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
        let dev_req = {
            match i {
                Instruction::Variable{result_type_id: _, result_id: _, storage_class, initializer: _}
                    => storage_class_requirement(storage_class),
                Instruction::TypePointer{result_id: _, storage_class, type_id: _}
                    => storage_class_requirement(storage_class),
                Instruction::Capability(cap) => capability_requirement(cap),
                _ => DeviceRequirement::None,
            }
        };

        match dev_req {
            DeviceRequirement::None => continue,
            DeviceRequirement::Features(features) => {
                for feature in features {
                    let ident = Ident::new(feature, Span::call_site());
                    cap_checks.push(quote! {
                        if !device.enabled_features().#ident {
                            panic!("Device feature {:?} required", #feature);
                        }
                    });
                }
            },
            DeviceRequirement::Extensions(extensions) => {
                for extension in extensions {
                    let ident = Ident::new(extension, Span::call_site());
                    cap_checks.push(quote! {
                        if !device.loaded_extensions().#ident {
                            panic!("Device extension {:?} required", #extension);
                        }
                    });
                }
            },
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
                        shader: ::vulkano::pipeline::shader::ShaderModule::from_words(device, &words)?
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

/// Returns the Vulkan device requirement for a SPIR-V `OpCapability`.
// TODO: this function is a draft, as the actual names may not be the same
fn capability_requirement(cap: &Capability) -> DeviceRequirement {
    match *cap {
        Capability::CapabilityMatrix => DeviceRequirement::None,
        Capability::CapabilityShader => DeviceRequirement::None,
        Capability::CapabilityGeometry =>
            DeviceRequirement::Features(&["geometry_shader"]),
        Capability::CapabilityTessellation =>
            DeviceRequirement::Features(&["tessellation_shader"]),
        Capability::CapabilityAddresses => panic!(), // not supported
        Capability::CapabilityLinkage => panic!(), // not supported
        Capability::CapabilityKernel => panic!(), // not supported
        Capability::CapabilityVector16 => panic!(), // not supported
        Capability::CapabilityFloat16Buffer => panic!(), // not supported
        Capability::CapabilityFloat16 => panic!(), // not supported
        Capability::CapabilityFloat64 =>
            DeviceRequirement::Features(&["shader_f3264"]),
        Capability::CapabilityInt64 =>
            DeviceRequirement::Features(&["shader_int64"]),
        Capability::CapabilityInt64Atomics => panic!(), // not supported
        Capability::CapabilityImageBasic => panic!(), // not supported
        Capability::CapabilityImageReadWrite => panic!(), // not supported
        Capability::CapabilityImageMipmap => panic!(), // not supported
        Capability::CapabilityPipes => panic!(), // not supported
        Capability::CapabilityGroups => panic!(), // not supported
        Capability::CapabilityDeviceEnqueue => panic!(), // not supported
        Capability::CapabilityLiteralSampler => panic!(), // not supported
        Capability::CapabilityAtomicStorage => panic!(), // not supported
        Capability::CapabilityInt16 =>
            DeviceRequirement::Features(&["shader_int16"]),
        Capability::CapabilityTessellationPointSize =>
            DeviceRequirement::Features(&["shader_tessellation_and_geometry_point_size"]),
        Capability::CapabilityGeometryPointSize =>
            DeviceRequirement::Features(&["shader_tessellation_and_geometry_point_size"]),
        Capability::CapabilityImageGatherExtended =>
            DeviceRequirement::Features(&["shader_image_gather_extended"]),
        Capability::CapabilityStorageImageMultisample =>
            DeviceRequirement::Features(&["shader_storage_image_multisample"]),
        Capability::CapabilityUniformBufferArrayDynamicIndexing =>
            DeviceRequirement::Features(&["shader_uniform_buffer_array_dynamic_indexing"]),
        Capability::CapabilitySampledImageArrayDynamicIndexing =>
            DeviceRequirement::Features(&["shader_sampled_image_array_dynamic_indexing"]),
        Capability::CapabilityStorageBufferArrayDynamicIndexing =>
            DeviceRequirement::Features(&["shader_storage_buffer_array_dynamic_indexing"]),
        Capability::CapabilityStorageImageArrayDynamicIndexing =>
            DeviceRequirement::Features(&["shader_storage_image_array_dynamic_indexing"]),
        Capability::CapabilityClipDistance =>
            DeviceRequirement::Features(&["shader_clip_distance"]),
        Capability::CapabilityCullDistance =>
            DeviceRequirement::Features(&["shader_cull_distance"]),
        Capability::CapabilityImageCubeArray =>
            DeviceRequirement::Features(&["image_cube_array"]),
        Capability::CapabilitySampleRateShading =>
            DeviceRequirement::Features(&["sample_rate_shading"]),
        Capability::CapabilityImageRect => panic!(), // not supported
        Capability::CapabilitySampledRect => panic!(), // not supported
        Capability::CapabilityGenericPointer => panic!(), // not supported
        Capability::CapabilityInt8 => panic!(), // not supported
        Capability::CapabilityInputAttachment => DeviceRequirement::None,
        Capability::CapabilitySparseResidency =>
            DeviceRequirement::Features(&["shader_resource_residency"]),
        Capability::CapabilityMinLod =>
            DeviceRequirement::Features(&["shader_resource_min_lod"]),
        Capability::CapabilitySampled1D => DeviceRequirement::None,
        Capability::CapabilityImage1D => DeviceRequirement::None,
        Capability::CapabilitySampledCubeArray =>
            DeviceRequirement::Features(&["image_cube_array"]),
        Capability::CapabilitySampledBuffer => DeviceRequirement::None,
        Capability::CapabilityImageBuffer => DeviceRequirement::None,
        Capability::CapabilityImageMSArray =>
            DeviceRequirement::Features(&["shader_storage_image_multisample"]),
        Capability::CapabilityStorageImageExtendedFormats =>
            DeviceRequirement::Features(&["shader_storage_image_extended_formats"]),
        Capability::CapabilityImageQuery => DeviceRequirement::None,
        Capability::CapabilityDerivativeControl => DeviceRequirement::None,
        Capability::CapabilityInterpolationFunction =>
            DeviceRequirement::Features(&["sample_rate_shading"]),
        Capability::CapabilityTransformFeedback => panic!(), // not supported
        Capability::CapabilityGeometryStreams => panic!(), // not supported
        Capability::CapabilityStorageImageReadWithoutFormat =>
            DeviceRequirement::Features(&["shader_storage_image_read_without_format"]),
        Capability::CapabilityStorageImageWriteWithoutFormat =>
            DeviceRequirement::Features(&["shader_storage_image_write_without_format"]),
        Capability::CapabilityMultiViewport =>
            DeviceRequirement::Features(&["multi_viewport"]),
        Capability::CapabilityStorageUniformBufferBlock16 =>
            DeviceRequirement::Extensions(&["khr_16bit_storage"]),
        Capability::CapabilityStorageUniform16 =>
            DeviceRequirement::Extensions(&["khr_16bit_storage"]),
        Capability::CapabilityStoragePushConstant16 =>
            DeviceRequirement::Extensions(&["khr_16bit_storage"]),
        Capability::CapabilityStorageInputOutput16 =>
            DeviceRequirement::Extensions(&["khr_16bit_storage"]),
    }
}

/// Returns the Vulkan device requirement for a SPIR-V storage class.
fn storage_class_requirement(storage_class: &StorageClass) -> DeviceRequirement {
    match *storage_class {
        StorageClass::StorageClassUniformConstant => DeviceRequirement::None,
        StorageClass::StorageClassInput => DeviceRequirement::None,
        StorageClass::StorageClassUniform => DeviceRequirement::None,
        StorageClass::StorageClassOutput => DeviceRequirement::None,
        StorageClass::StorageClassWorkgroup => DeviceRequirement::None,
        StorageClass::StorageClassCrossWorkgroup => DeviceRequirement::None,
        StorageClass::StorageClassPrivate => DeviceRequirement::None,
        StorageClass::StorageClassFunction => DeviceRequirement::None,
        StorageClass::StorageClassGeneric => DeviceRequirement::None,
        StorageClass::StorageClassPushConstant => DeviceRequirement::None,
        StorageClass::StorageClassAtomicCounter => DeviceRequirement::None,
        StorageClass::StorageClassImage => DeviceRequirement::None,
        StorageClass::StorageClassStorageBuffer =>
            DeviceRequirement::Extensions(&["khr_storage_buffer_storage_class"]),
    }
}

enum DeviceRequirement {
    None,
    Features(&'static[&'static str]),
    Extensions(&'static[&'static str]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_bad_alignment() {
        // vec3/mat3/mat3x* are problematic in arrays since their rust
        // representations don't have the same array stride as the SPIR-V
        // ones. E.g. in a vec3[2], the second element starts on the 16th
        // byte, but in a rust [[f32;3];2], the second element starts on the
        // 12th byte. Since we can't generate code for these types, we should
        // create an error instead of generating incorrect code.
        let includes: [PathBuf;0] = [];
        let defines: [(String, String);0] = [];
        let comp = compile(None, &Path::new(""), "
        #version 450
        struct MyStruct {
            vec3 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ", ShaderKind::Vertex, &includes, &defines).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        let res = std::panic::catch_unwind(|| structs::write_structs(&doc));
        assert!(res.is_err());
    }
    #[test]
    fn test_trivial_alignment() {
        let includes: [PathBuf;0] = [];
        let defines: [(String, String);0] = [];
        let comp = compile(None, &Path::new(""), "
        #version 450
        struct MyStruct {
            vec4 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ", ShaderKind::Vertex, &includes, &defines).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc);
    }
    #[test]
    fn test_wrap_alignment() {
        // This is a workaround suggested in the case of test_bad_alignment,
        // so we should make sure it works.
        let includes: [PathBuf;0] = [];
        let defines: [(String, String);0] = [];
        let comp = compile(None, &Path::new(""), "
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
        ", ShaderKind::Vertex, &includes, &defines).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc);
    }

    #[test]
    fn test_include_resolution() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let empty_includes: [PathBuf;0] = [];
        let defines: [(String, String);0] = [];
        let _compile_relative = compile(Some(String::from("tests/include_test.glsl")), &root_path, "
        #version 450
        #include \"include_dir_a/target_a.glsl\"
        #include \"include_dir_b/target_b.glsl\"
        void main() {}
        ", ShaderKind::Vertex, &empty_includes, &defines).expect("Cannot resolve include files");

        let _compile_include_paths = compile(Some(String::from("tests/include_test.glsl")), &root_path, "
        #version 450
        #include <target_a.glsl>
        #include <target_b.glsl>
        void main() {}
        ", ShaderKind::Vertex,&[root_path.join("tests/include_dir_a"), root_path.join("tests/include_dir_b")],
        &defines).expect("Cannot resolve include files");

        let _compile_include_paths_with_relative = compile(Some(String::from("tests/include_test.glsl")), &root_path, "
        #version 450
        #include <target_a.glsl>
        #include <../include_dir_b/target_b.glsl>
        void main() {}
        ", ShaderKind::Vertex,&[root_path.join("tests/include_dir_a")],
        &defines).expect("Cannot resolve include files");

        let absolute_path = root_path.join("tests/include_dir_a/target_a.glsl");
        let absolute_path_str = absolute_path.to_str().expect("Cannot run tests in a folder with non unicode characters");
        let _compile_absolute_path = compile(Some(String::from("tests/include_test.glsl")), &root_path, &format!("
        #version 450
        #include \"{}\"
        void main() {{}}
        ", absolute_path_str), ShaderKind::Vertex, &empty_includes, &defines).expect("Cannot resolve include files");

        let _compile_recursive = compile(Some(String::from("tests/include_test.glsl")), &root_path, "
        #version 450
        #include <target_c.glsl>
        void main() {}
        ", ShaderKind::Vertex,&[root_path.join("tests/include_dir_b"), root_path.join("tests/include_dir_c")],
        &defines).expect("Cannot resolve include files");
    }

    #[test]
    fn test_macros() {
        let empty_includes: [PathBuf;0] = [];
        let defines= vec![("NAME1", ""), ("NAME2", "58")];
        let no_defines: [(String, String);0] = [];
        let need_defines = "
        #version 450
        #if defined(NAME1) && NAME2 > 29
        void main() {}
        #endif
        ";
        let compile_no_defines = compile(None, &Path::new(""), need_defines, ShaderKind::Vertex, &empty_includes, &no_defines);
        assert!(compile_no_defines.is_err());

        let compile_defines = compile(None, &Path::new(""), need_defines, ShaderKind::Vertex, &empty_includes, &defines);
        compile_defines.expect("Setting shader macros did not work");
    }
}
