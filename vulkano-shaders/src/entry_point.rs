// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use fnv::FnvHashMap;
use proc_macro2::TokenStream;
use vulkano::pipeline::layout::PipelineLayoutPcRange;
use vulkano::shader::{
    DescriptorRequirements, GeometryShaderExecution, ShaderExecution, ShaderInterfaceEntry,
    ShaderInterfaceEntryType, SpecializationConstantRequirements,
};
use vulkano::shader::{EntryPointInfo, ShaderInterface, ShaderStages};
use vulkano::shader::spirv::ExecutionModel;

pub(super) fn write_entry_point(name: &str, model: ExecutionModel, info: &EntryPointInfo) -> TokenStream {
    let execution = write_shader_execution(&info.execution);
    let model = syn::parse_str::<syn::Path>(&format!("vulkano::shader::spirv::ExecutionModel::{:?}", model)).unwrap();
    let descriptor_requirements = write_descriptor_requirements(&info.descriptor_requirements);
    let push_constant_requirements =
        write_push_constant_requirements(&info.push_constant_requirements);
    let specialization_constant_requirements =
        write_specialization_constant_requirements(&info.specialization_constant_requirements);
    let input_interface = write_interface(&info.input_interface);
    let output_interface = write_interface(&info.output_interface);

    quote! {
        (
            #name.to_owned(),
            #model,
            EntryPointInfo {
                execution: #execution,
                descriptor_requirements: std::array::IntoIter::new(#descriptor_requirements).collect(),
                push_constant_requirements: #push_constant_requirements,
                specialization_constant_requirements: std::array::IntoIter::new(#specialization_constant_requirements).collect(),
                input_interface: #input_interface,
                output_interface: #output_interface,
            },
        ),
    }
}

fn write_shader_execution(execution: &ShaderExecution) -> TokenStream {
    match execution {
        ShaderExecution::Vertex => quote! { ShaderExecution::Vertex },
        ShaderExecution::TessellationControl => quote! { ShaderExecution::TessellationControl },
        ShaderExecution::TessellationEvaluation => {
            quote! { ShaderExecution::TessellationEvaluation }
        }
        ShaderExecution::Geometry(GeometryShaderExecution { input }) => {
            let input = format_ident!("{}", format!("{:?}", input));
            quote! {
                ShaderExecution::Geometry {
                    input: GeometryShaderInput::#input,
                }
            }
        }
        ShaderExecution::Fragment => quote! { ShaderExecution::Fragment },
        ShaderExecution::Compute => quote! { ShaderExecution::Compute },
    }
}

fn write_descriptor_requirements(
    descriptor_requirements: &FnvHashMap<(u32, u32), DescriptorRequirements>,
) -> TokenStream {
    let descriptor_requirements = descriptor_requirements.into_iter().map(|(loc, reqs)| {
        let (set_num, binding_num) = loc;
        let DescriptorRequirements {
            descriptor_types,
            descriptor_count,
            format,
            image_view_type,
            multisampled,
            mutable,
            stages,
        } = reqs;

        let descriptor_types = descriptor_types.into_iter().map(|ty| {
            let ident = format_ident!("{}", format!("{:?}", ty));
            quote! { DescriptorType::#ident }
        });
        let format = match format {
            Some(format) => {
                let ident = format_ident!("{}", format!("{:?}", format));
                quote! { Some(Format::#ident) }
            }
            None => quote! { None },
        };
        let image_view_type = match image_view_type {
            Some(image_view_type) => {
                let ident = format_ident!("{}", format!("{:?}", image_view_type));
                quote! { Some(ImageViewType::#ident) }
            }
            None => quote! { None },
        };
        let stages = {
            let ShaderStages {
                vertex,
                tessellation_control,
                tessellation_evaluation,
                geometry,
                fragment,
                compute,
                raygen,
                any_hit,
                closest_hit,
                miss,
                intersection,
                callable,
            } = stages;

            quote! {
                ShaderStages {
                    vertex: #vertex,
                    tessellation_control: #tessellation_control,
                    tessellation_evaluation: #tessellation_evaluation,
                    geometry: #geometry,
                    fragment: #fragment,
                    compute: #compute,
                    raygen: #raygen,
                    any_hit: #any_hit,
                    closest_hit: #closest_hit,
                    miss: #miss,
                    intersection: #intersection,
                    callable: #callable,
                }
            }
        };

        quote! {
            (
                (#set_num, #binding_num),
                DescriptorRequirements {
                    descriptor_types: vec![#(#descriptor_types),*],
                    descriptor_count: #descriptor_count,
                    format: #format,
                    image_view_type: #image_view_type,
                    multisampled: #multisampled,
                    mutable: #mutable,
                    stages: #stages,
                },
            ),
        }
    });

    quote! {
        [
            #( #descriptor_requirements )*
        ]
    }
}

fn write_push_constant_requirements(
    push_constant_requirements: &Option<PipelineLayoutPcRange>,
) -> TokenStream {
    match push_constant_requirements {
        Some(PipelineLayoutPcRange {
            offset,
            size,
            stages,
        }) => {
            let stages = {
                let ShaderStages {
                    vertex,
                    tessellation_control,
                    tessellation_evaluation,
                    geometry,
                    fragment,
                    compute,
                    raygen,
                    any_hit,
                    closest_hit,
                    miss,
                    intersection,
                    callable,
                } = stages;

                quote! {
                    ShaderStages {
                        vertex: #vertex,
                        tessellation_control: #tessellation_control,
                        tessellation_evaluation: #tessellation_evaluation,
                        geometry: #geometry,
                        fragment: #fragment,
                        compute: #compute,
                        raygen: #raygen,
                        any_hit: #any_hit,
                        closest_hit: #closest_hit,
                        miss: #miss,
                        intersection: #intersection,
                        callable: #callable,
                    }
                }
            };

            quote! {
                Some(PipelineLayoutPcRange {
                    offset: #offset,
                    size: #size,
                    stages: #stages,
                })
            }
        }
        None => quote! {
            None
        },
    }
}

fn write_specialization_constant_requirements(
    specialization_constant_requirements: &FnvHashMap<u32, SpecializationConstantRequirements>,
) -> TokenStream {
    let specialization_constant_requirements = specialization_constant_requirements
        .into_iter()
        .map(|(&constant_id, reqs)| {
            let SpecializationConstantRequirements { size } = reqs;
            quote! {
                (
                    #constant_id,
                    SpecializationConstantRequirements {
                        size: #size,
                    },
                ),
            }
        });

    quote! {
        [
            #( #specialization_constant_requirements )*
        ]
    }
}

fn write_interface(interface: &ShaderInterface) -> TokenStream {
    let items = interface.elements().iter().map(
        |ShaderInterfaceEntry {
             location,
             component,
             ty:
                 ShaderInterfaceEntryType {
                     base_type,
                     num_components,
                     num_elements,
                     is_64bit,
                 },
             name,
         }| {
            let base_type = format_ident!("{}", format!("{:?}", base_type));

            quote! {
                ::vulkano::shader::ShaderInterfaceEntry {
                    location: #location,
                    component: #component,
                    ty: ::vulkano::shader::ShaderInterfaceEntryType {
                        base_type: ::vulkano::shader::ShaderScalarType::#base_type,
                        num_components: #num_components,
                        num_elements: #num_elements,
                        is_64bit: #is_64bit,
                    },
                    name: Some(::std::borrow::Cow::Borrowed(#name))
                },
            }
        },
    );

    quote! {
        unsafe {
            ::vulkano::shader::ShaderInterface::new_unchecked(vec![
                #( #items )*
            ])
        }
    }
}
