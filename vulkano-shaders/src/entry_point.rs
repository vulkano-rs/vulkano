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
use vulkano::pipeline::layout::PushConstantRange;
use vulkano::shader::spirv::ExecutionModel;
use vulkano::shader::{
    DescriptorIdentifier, DescriptorRequirements, ShaderExecution, ShaderInterfaceEntry,
    ShaderInterfaceEntryType, SpecializationConstantRequirements,
};
use vulkano::shader::{EntryPointInfo, ShaderInterface, ShaderStages};

pub(super) fn write_entry_point(
    name: &str,
    model: ExecutionModel,
    info: &EntryPointInfo,
) -> TokenStream {
    let execution = write_shader_execution(&info.execution);
    let model = syn::parse_str::<syn::Path>(&format!(
        "::vulkano::shader::spirv::ExecutionModel::{:?}",
        model
    ))
    .unwrap();
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
            ::vulkano::shader::EntryPointInfo {
                execution: #execution,
                descriptor_requirements: #descriptor_requirements.into_iter().collect(),
                push_constant_requirements: #push_constant_requirements,
                specialization_constant_requirements: #specialization_constant_requirements.into_iter().collect(),
                input_interface: #input_interface,
                output_interface: #output_interface,
            },
        ),
    }
}

fn write_shader_execution(execution: &ShaderExecution) -> TokenStream {
    match execution {
        ShaderExecution::Vertex => quote! { ::vulkano::shader::ShaderExecution::Vertex },
        ShaderExecution::TessellationControl => {
            quote! { ::vulkano::shader::ShaderExecution::TessellationControl }
        }
        ShaderExecution::TessellationEvaluation => {
            quote! { ::vulkano::shader::ShaderExecution::TessellationEvaluation }
        }
        ShaderExecution::Geometry(::vulkano::shader::GeometryShaderExecution { input }) => {
            let input = format_ident!("{}", format!("{:?}", input));
            quote! {
                ::vulkano::shader::ShaderExecution::Geometry(
                    ::vulkano::shader::GeometryShaderExecution {
                        input: ::vulkano::shader::GeometryShaderInput::#input
                    }
                )
            }
        }
        ShaderExecution::Fragment => quote! { ::vulkano::shader::ShaderExecution::Fragment },
        ShaderExecution::Compute => quote! { ::vulkano::shader::ShaderExecution::Compute },
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
            image_format,
            image_multisampled,
            image_scalar_type,
            image_view_type,
            sampler_compare,
            sampler_no_unnormalized_coordinates,
            sampler_no_ycbcr_conversion,
            sampler_with_images,
            stages,
            storage_image_atomic,
            storage_read,
            storage_write,
        } = reqs;

        let descriptor_types = descriptor_types.into_iter().map(|ty| {
            let ident = format_ident!("{}", format!("{:?}", ty));
            quote! { ::vulkano::descriptor_set::layout::DescriptorType::#ident }
        });
        let image_format = match image_format {
            Some(image_format) => {
                let ident = format_ident!("{}", format!("{:?}", image_format));
                quote! { Some(::vulkano::format::Format::#ident) }
            }
            None => quote! { None },
        };
        let image_scalar_type = match image_scalar_type {
            Some(image_scalar_type) => {
                let ident = format_ident!("{}", format!("{:?}", image_scalar_type));
                quote! { Some(::vulkano::shader::ShaderScalarType::#ident) }
            }
            None => quote! { None },
        };
        let image_view_type = match image_view_type {
            Some(image_view_type) => {
                let ident = format_ident!("{}", format!("{:?}", image_view_type));
                quote! { Some(::vulkano::image::view::ImageViewType::#ident) }
            }
            None => quote! { None },
        };
        let sampler_compare = sampler_compare.iter();
        let sampler_no_unnormalized_coordinates = sampler_no_unnormalized_coordinates.iter();
        let sampler_no_ycbcr_conversion = sampler_no_ycbcr_conversion.iter();
        let sampler_with_images = {
            sampler_with_images.iter().map(|(&index, identifiers)| {
                let identifiers = identifiers.iter().map(
                    |DescriptorIdentifier {
                         set,
                         binding,
                         index,
                     }| {
                        quote! {
                            ::vulkano::shader::DescriptorIdentifier {
                                set: #set,
                                binding: #binding,
                                index: #index,
                            }
                        }
                    },
                );
                quote! {
                    (
                        #index,
                        [#(#identifiers),*].into_iter().collect(),
                    )
                }
            })
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
                ::vulkano::shader::ShaderStages {
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
        let storage_image_atomic = storage_image_atomic.iter();
        let storage_read = storage_read.iter();
        let storage_write = storage_write.iter();

        quote! {
            (
                (#set_num, #binding_num),
                ::vulkano::shader::DescriptorRequirements {
                    descriptor_types: vec![#(#descriptor_types),*],
                    descriptor_count: #descriptor_count,
                    image_format: #image_format,
                    image_multisampled: #image_multisampled,
                    image_scalar_type: #image_scalar_type,
                    image_view_type: #image_view_type,
                    sampler_compare: [#(#sampler_compare),*].into_iter().collect(),
                    sampler_no_unnormalized_coordinates: [#(#sampler_no_unnormalized_coordinates),*].into_iter().collect(),
                    sampler_no_ycbcr_conversion: [#(#sampler_no_ycbcr_conversion),*].into_iter().collect(),
                    sampler_with_images: [#(#sampler_with_images),*].into_iter().collect(),
                    stages: #stages,
                    storage_image_atomic: [#(#storage_image_atomic),*].into_iter().collect(),
                    storage_read: [#(#storage_read),*].into_iter().collect(),
                    storage_write: [#(#storage_write),*].into_iter().collect(),
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
    push_constant_requirements: &Option<PushConstantRange>,
) -> TokenStream {
    match push_constant_requirements {
        Some(PushConstantRange {
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
                    ::vulkano::shader::ShaderStages {
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
                Some(::vulkano::pipeline::layout::PushConstantRange {
                    stages: #stages,
                    offset: #offset,
                    size: #size,
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
                    ::vulkano::shader::SpecializationConstantRequirements {
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
