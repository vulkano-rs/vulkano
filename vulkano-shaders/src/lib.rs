// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]

#![recursion_limit = "1024"]
#[macro_use] extern crate quote;
             extern crate shaderc;
             extern crate proc_macro2;
             extern crate syn;

use std::io::Error as IoError;

use syn::Ident;
use proc_macro2::{Span, TokenStream};
use shaderc::{Compiler, CompileOptions};

pub use shaderc::{CompilationArtifact, ShaderKind};
pub use parse::ParseError;

use parse::Instruction;
use enums::Capability;

mod descriptor_sets;
mod entry_point;
mod enums;
mod parse;
mod spec_consts;
mod structs;
mod spirv_search;

pub fn compile(code: &str, ty: ShaderKind) -> Result<CompilationArtifact, String> {
    let mut compiler = Compiler::new().ok_or("failed to create GLSL compiler")?;
    let compile_options = CompileOptions::new().ok_or("failed to initialize compile option")?;

    let content = compiler
        .compile_into_spirv(&code, ty, "shader.glsl", "main", Some(&compile_options))
        .map_err(|e| e.to_string())?;

    Ok(content)
}

pub fn reflect(name: &str, spirv: &[u32], mod_name: &Ident, dump: bool) -> Result<TokenStream, Error> {
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
        mod #mod_name {
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
        }
    };

    if dump {
        println!("{}", ast.to_string());
        panic!("vulkano_shader! rust codegen dumped") // TODO: use span from dump
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
        let comp = compile("
        #version 450
        struct MyStruct {
            vec3 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ", ShaderKind::Vertex).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        let res = std::panic::catch_unwind(|| structs::write_structs(&doc));
        assert!(res.is_err());
    }
    #[test]
    fn test_trivial_alignment() {
        let comp = compile("
        #version 450
        struct MyStruct {
            vec4 vs[2];
        };
        layout(binding=0) uniform UBO {
            MyStruct s;
        };
        void main() {}
        ", ShaderKind::Vertex).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc);
    }
    #[test]
    fn test_wrap_alignment() {
        // This is a workaround suggested in the case of test_bad_alignment,
        // so we should make sure it works.
        let comp = compile("
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
        ", ShaderKind::Vertex).unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();
        structs::write_structs(&doc);
    }
}
