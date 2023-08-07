// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use ahash::HashMap;
use proc_macro2::TokenStream;
use vulkano::{
    pipeline::layout::PushConstantRange,
    shader::{
        DescriptorBindingRequirements, DescriptorIdentifier, DescriptorRequirements,
        EntryPointInfo, ShaderExecution, ShaderInterface, ShaderInterfaceEntry,
        ShaderInterfaceEntryType, ShaderStages, SpecializationConstant,
    },
};

pub(super) fn write_entry_point(info: &EntryPointInfo) -> TokenStream {
    let name = &info.name;
    let execution = write_shader_execution(&info.execution);
    let descriptor_binding_requirements =
        write_descriptor_binding_requirements(&info.descriptor_binding_requirements);
    let push_constant_requirements =
        write_push_constant_requirements(&info.push_constant_requirements);
    let specialization_constants = write_specialization_constants(&info.specialization_constants);
    let input_interface = write_interface(&info.input_interface);
    let output_interface = write_interface(&info.output_interface);

    quote! {
        ::vulkano::shader::EntryPointInfo {
            name: #name.to_owned(),
            execution: #execution,
            descriptor_binding_requirements: #descriptor_binding_requirements.into_iter().collect(),
            push_constant_requirements: #push_constant_requirements,
            specialization_constants: #specialization_constants.into_iter().collect(),
            input_interface: #input_interface,
            output_interface: #output_interface,
        },
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
        ShaderExecution::Fragment(::vulkano::shader::FragmentShaderExecution {
            fragment_tests_stages,
        }) => {
            let fragment_tests_stages = format_ident!("{}", format!("{:?}", fragment_tests_stages));
            quote! {
                ::vulkano::shader::ShaderExecution::Fragment(
                    ::vulkano::shader::FragmentShaderExecution {
                        fragment_tests_stages: ::vulkano::shader::FragmentTestsStages::#fragment_tests_stages,
                    }
                )
            }
        }
        ShaderExecution::Compute(execution) => {
            use ::quote::ToTokens;
            use ::vulkano::shader::{ComputeShaderExecution, LocalSize};

            struct LocalSizeToTokens(LocalSize);

            impl ToTokens for LocalSizeToTokens {
                fn to_tokens(&self, tokens: &mut TokenStream) {
                    match self.0 {
                        LocalSize::Literal(literal) => quote! {
                            ::vulkano::shader::LocalSize::Literal(#literal)
                        },
                        LocalSize::SpecId(id) => quote! {
                            ::vulkano::shader::LocalSize::SpecId(#id)
                        },
                    }
                    .to_tokens(tokens);
                }
            }

            match execution {
                ComputeShaderExecution::LocalSize([x, y, z]) => {
                    let [x, y, z] = [
                        LocalSizeToTokens(*x),
                        LocalSizeToTokens(*y),
                        LocalSizeToTokens(*z),
                    ];
                    quote! { ::vulkano::shader::ShaderExecution::Compute(
                        ::vulkano::shader::ComputeShaderExecution::LocalSize([#x, #y, #z])
                    ) }
                }
                ComputeShaderExecution::LocalSizeId([x, y, z]) => {
                    let [x, y, z] = [
                        LocalSizeToTokens(*x),
                        LocalSizeToTokens(*y),
                        LocalSizeToTokens(*z),
                    ];
                    quote! { ::vulkano::shader::ShaderExecution::Compute(
                        ::vulkano::shader::ComputeShaderExecution::LocalSizeId([#x, #y, #z])
                    ) }
                }
            }
        }
        ShaderExecution::RayGeneration => {
            quote! { ::vulkano::shader::ShaderExecution::RayGeneration }
        }
        ShaderExecution::AnyHit => quote! { ::vulkano::shader::ShaderExecution::AnyHit },
        ShaderExecution::ClosestHit => quote! { ::vulkano::shader::ShaderExecution::ClosestHit },
        ShaderExecution::Miss => quote! { ::vulkano::shader::ShaderExecution::Miss },
        ShaderExecution::Intersection => {
            quote! { ::vulkano::shader::ShaderExecution::Intersection }
        }
        ShaderExecution::Callable => quote! { ::vulkano::shader::ShaderExecution::Callable },
        ShaderExecution::Task => quote! { ::vulkano::shader::ShaderExecution::Task },
        ShaderExecution::Mesh => quote! { ::vulkano::shader::ShaderExecution::Mesh },
        ShaderExecution::SubpassShading => {
            quote! { ::vulkano::shader::ShaderExecution::SubpassShading }
        }
    }
}

fn write_descriptor_binding_requirements(
    descriptor_binding_requirements: &HashMap<(u32, u32), DescriptorBindingRequirements>,
) -> TokenStream {
    let descriptor_binding_requirements =
        descriptor_binding_requirements
            .iter()
            .map(|(loc, binding_reqs)| {
                let (set_num, binding_num) = loc;
                let DescriptorBindingRequirements {
                    descriptor_types,
                    descriptor_count,
                    image_format,
                    image_multisampled,
                    image_scalar_type,
                    image_view_type,
                    stages,
                    descriptors,
                } = binding_reqs;

                let descriptor_types_items = descriptor_types.iter().map(|ty| {
                    let ident = format_ident!("{}", format!("{:?}", ty));
                    quote! { ::vulkano::descriptor_set::layout::DescriptorType::#ident }
                });
                let descriptor_count = match descriptor_count {
                    Some(descriptor_count) => quote! { Some(#descriptor_count) },
                    None => quote! { None },
                };
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
                let stages = stages_to_items(*stages);
                let descriptor_items = descriptors.iter().map(|(index, desc_reqs)| {
                    let DescriptorRequirements {
                        memory_read,
                        memory_write,
                        sampler_compare,
                        sampler_no_unnormalized_coordinates,
                        sampler_no_ycbcr_conversion,
                        sampler_with_images,
                        storage_image_atomic,
                    } = desc_reqs;

                    let index = match index {
                        Some(index) => quote! { Some(#index) },
                        None => quote! { None },
                    };
                    let memory_read = stages_to_items(*memory_read);
                    let memory_write = stages_to_items(*memory_write);
                    let sampler_with_images_items = sampler_with_images.iter().map(|DescriptorIdentifier {
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
                    });

                    quote! {
                        (
                            #index,
                            ::vulkano::shader::DescriptorRequirements {
                                memory_read: #memory_read,
                                memory_write: #memory_write,
                                sampler_compare: #sampler_compare,
                                sampler_no_unnormalized_coordinates: #sampler_no_unnormalized_coordinates,
                                sampler_no_ycbcr_conversion: #sampler_no_ycbcr_conversion,
                                sampler_with_images: [ #( #sampler_with_images_items ),* ]
                                    .into_iter()
                                    .collect(),
                                storage_image_atomic: #storage_image_atomic,
                            }
                        )
                    }
                });

                quote! {
                    (
                        (#set_num, #binding_num),
                        ::vulkano::shader::DescriptorBindingRequirements {
                            descriptor_types: vec![ #( #descriptor_types_items ),* ],
                            descriptor_count: #descriptor_count,
                            image_format: #image_format,
                            image_multisampled: #image_multisampled,
                            image_scalar_type: #image_scalar_type,
                            image_view_type: #image_view_type,
                            stages: #stages,
                            descriptors: [ #( #descriptor_items ),* ].into_iter().collect(),
                        },
                    )
                }
            });

    quote! {
        [
            #( #descriptor_binding_requirements ),*
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
            let stages = stages_to_items(*stages);

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

fn write_specialization_constants(
    specialization_constants: &HashMap<u32, SpecializationConstant>,
) -> TokenStream {
    let specialization_constants = specialization_constants
        .iter()
        .map(|(&constant_id, value)| {
            let value = match value {
                SpecializationConstant::Bool(value) => quote! { Bool(#value) },
                SpecializationConstant::I8(value) => quote! { I8(#value) },
                SpecializationConstant::I16(value) => quote! { I16(#value) },
                SpecializationConstant::I32(value) => quote! { I32(#value) },
                SpecializationConstant::I64(value) => quote! { I64(#value) },
                SpecializationConstant::U8(value) => quote! { U8(#value) },
                SpecializationConstant::U16(value) => quote! { U16(#value) },
                SpecializationConstant::U32(value) => quote! { U32(#value) },
                SpecializationConstant::U64(value) => quote! { U64(#value) },
                SpecializationConstant::F16(value) => {
                    let bits = value.to_bits();
                    quote! { F16(f16::from_bits(#bits)) }
                }
                SpecializationConstant::F32(value) => {
                    let bits = value.to_bits();
                    quote! { F32(f32::from_bits(#bits)) }
                }
                SpecializationConstant::F64(value) => {
                    let bits = value.to_bits();
                    quote! { F64(f64::from_bits(#bits)) }
                }
            };

            quote! {
                (
                    #constant_id,
                    ::vulkano::shader::SpecializationConstant::#value,
                )
            }
        });

    quote! {
        [
            #( #specialization_constants ),*
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
                    name: ::std::option::Option::Some(::std::borrow::Cow::Borrowed(#name)),
                }
            }
        },
    );

    quote! {
        ::vulkano::shader::ShaderInterface::new_unchecked(vec![
            #( #items ),*
        ])
    }
}

fn stages_to_items(stages: ShaderStages) -> TokenStream {
    if stages.is_empty() {
        quote! { ::vulkano::shader::ShaderStages::empty() }
    } else {
        let stages_items = [
            stages.intersects(ShaderStages::VERTEX).then(|| {
                quote! { ::vulkano::shader::ShaderStages::VERTEX }
            }),
            stages
                .intersects(ShaderStages::TESSELLATION_CONTROL)
                .then(|| {
                    quote! { ::vulkano::shader::ShaderStages::TESSELLATION_CONTROL }
                }),
            stages
                .intersects(ShaderStages::TESSELLATION_EVALUATION)
                .then(|| {
                    quote! { ::vulkano::shader::ShaderStages::TESSELLATION_EVALUATION }
                }),
            stages.intersects(ShaderStages::GEOMETRY).then(|| {
                quote! { ::vulkano::shader::ShaderStages::GEOMETRY }
            }),
            stages.intersects(ShaderStages::FRAGMENT).then(|| {
                quote! { ::vulkano::shader::ShaderStages::FRAGMENT }
            }),
            stages.intersects(ShaderStages::COMPUTE).then(|| {
                quote! { ::vulkano::shader::ShaderStages::COMPUTE }
            }),
            stages.intersects(ShaderStages::RAYGEN).then(|| {
                quote! { ::vulkano::shader::ShaderStages::RAYGEN }
            }),
            stages.intersects(ShaderStages::ANY_HIT).then(|| {
                quote! { ::vulkano::shader::ShaderStages::ANY_HIT }
            }),
            stages.intersects(ShaderStages::CLOSEST_HIT).then(|| {
                quote! { ::vulkano::shader::ShaderStages::CLOSEST_HIT }
            }),
            stages.intersects(ShaderStages::MISS).then(|| {
                quote! { ::vulkano::shader::ShaderStages::MISS }
            }),
            stages.intersects(ShaderStages::INTERSECTION).then(|| {
                quote! { ::vulkano::shader::ShaderStages::INTERSECTION }
            }),
            stages.intersects(ShaderStages::CALLABLE).then(|| {
                quote! { ::vulkano::shader::ShaderStages::CALLABLE }
            }),
        ]
        .into_iter()
        .flatten();

        quote! { #( #stages_items )|* }
    }
}
