// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor_sets::{write_descriptor_set_layout_descs, write_push_constant_ranges};
use crate::{spirv_search, TypesMeta};
use proc_macro2::{Span, TokenStream};
use syn::Ident;
use vulkano::spirv::{
    Decoration, ExecutionMode, ExecutionModel, Id, Instruction, Spirv, StorageClass,
};

pub(super) fn write_entry_point(
    shader: &str,
    spirv: &Spirv,
    instruction: &Instruction,
    types_meta: &TypesMeta,
    exact_entrypoint_interface: bool,
    shared_constants: bool,
) -> TokenStream {
    let (execution, id, ep_name, interface) = match instruction {
        &Instruction::EntryPoint {
            ref execution_model,
            entry_point,
            ref name,
            ref interface,
            ..
        } => (execution_model, entry_point, name, interface),
        _ => unreachable!(),
    };

    let ignore_first_array_in = match *execution {
        ExecutionModel::TessellationControl => true,
        ExecutionModel::TessellationEvaluation => true,
        ExecutionModel::Geometry => true,
        _ => false,
    };
    let ignore_first_array_out = match *execution {
        ExecutionModel::TessellationControl => true,
        _ => false,
    };

    let (input_interface, output_interface) = write_interfaces(
        spirv,
        interface,
        ignore_first_array_in,
        ignore_first_array_out,
    );

    let stage = if let ExecutionModel::GLCompute = *execution {
        quote! { ShaderStages { compute: true, ..ShaderStages::none() } }
    } else {
        match *execution {
            ExecutionModel::Vertex => {
                quote! { ShaderStages { vertex: true, ..ShaderStages::none() } }
            }
            ExecutionModel::TessellationControl => {
                quote! { ShaderStages { tessellation_control: true, ..ShaderStages::none() } }
            }
            ExecutionModel::TessellationEvaluation => {
                quote! { ShaderStages { tessellation_evaluation: true, ..ShaderStages::none() } }
            }
            ExecutionModel::Geometry => {
                quote! { ShaderStages { geometry: true, ..ShaderStages::none() } }
            }
            ExecutionModel::Fragment => {
                quote! { ShaderStages { fragment: true, ..ShaderStages::none() } }
            }
            ExecutionModel::GLCompute
            | ExecutionModel::Kernel
            | ExecutionModel::TaskNV
            | ExecutionModel::MeshNV
            | ExecutionModel::RayGenerationKHR
            | ExecutionModel::IntersectionKHR
            | ExecutionModel::AnyHitKHR
            | ExecutionModel::ClosestHitKHR
            | ExecutionModel::MissKHR
            | ExecutionModel::CallableKHR => unreachable!(),
        }
    };

    let descriptor_set_layout_descs = write_descriptor_set_layout_descs(
        &spirv,
        id,
        interface,
        exact_entrypoint_interface,
        &stage,
    );
    let push_constant_ranges = write_push_constant_ranges(shader, &spirv, &stage, &types_meta);

    let spec_consts_struct = if crate::spec_consts::has_specialization_constants(spirv) {
        let spec_consts_struct_name = Ident::new(
            &format!(
                "{}SpecializationConstants",
                if shared_constants { "" } else { shader }
            ),
            Span::call_site(),
        );
        quote! { #spec_consts_struct_name }
    } else {
        quote! { () }
    };

    let (ty, f_call) = {
        if let ExecutionModel::GLCompute = *execution {
            (
                quote! { ::vulkano::pipeline::shader::ComputeEntryPoint },
                quote! { compute_entry_point(
                    ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                    #descriptor_set_layout_descs,
                    #push_constant_ranges,
                    <#spec_consts_struct>::descriptors(),
                )},
            )
        } else {
            let entry_ty = match *execution {
                ExecutionModel::Vertex => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::Vertex }
                }

                ExecutionModel::TessellationControl => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::TessellationControl }
                }

                ExecutionModel::TessellationEvaluation => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::TessellationEvaluation }
                }

                ExecutionModel::Geometry => {
                    let execution_mode =
                        spirv
                            .iter_execution_mode()
                            .find_map(|instruction| match instruction {
                                &Instruction::ExecutionMode {
                                    entry_point,
                                    ref mode,
                                    ..
                                } if entry_point == id => match mode {
                                    &ExecutionMode::InputPoints => Some(quote! { Points }),
                                    &ExecutionMode::InputLines => Some(quote! { Lines }),
                                    &ExecutionMode::InputLinesAdjacency => {
                                        Some(quote! { LinesWithAdjacency })
                                    }
                                    &ExecutionMode::Triangles => Some(quote! { Triangles }),
                                    &ExecutionMode::InputTrianglesAdjacency => {
                                        Some(quote! { TrianglesWithAdjacency })
                                    }
                                    _ => None,
                                },
                                _ => None,
                            });

                    quote! {
                        ::vulkano::pipeline::shader::GraphicsShaderType::Geometry(
                            ::vulkano::pipeline::shader::GeometryShaderExecutionMode::#execution_mode
                        )
                    }
                }

                ExecutionModel::Fragment => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::Fragment }
                }

                ExecutionModel::GLCompute => unreachable!(),

                ExecutionModel::Kernel
                | ExecutionModel::TaskNV
                | ExecutionModel::MeshNV
                | ExecutionModel::RayGenerationKHR
                | ExecutionModel::IntersectionKHR
                | ExecutionModel::AnyHitKHR
                | ExecutionModel::ClosestHitKHR
                | ExecutionModel::MissKHR
                | ExecutionModel::CallableKHR => {
                    panic!("Shaders with {:?} are not supported", execution)
                }
            };

            let ty = quote! { ::vulkano::pipeline::shader::GraphicsEntryPoint };
            let f_call = quote! {
                graphics_entry_point(
                    ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                    #descriptor_set_layout_descs,
                    #push_constant_ranges,
                    <#spec_consts_struct>::descriptors(),
                    #input_interface,
                    #output_interface,
                    #entry_ty
                )
            };

            (ty, f_call)
        }
    };

    let mut method_name = ep_name.clone();
    method_name.push_str("_entry_point");
    let method_ident = Ident::new(&method_name, Span::call_site());

    let ep_name_lenp1 = ep_name.chars().count() + 1;
    let encoded_ep_name = ep_name.chars().map(|c| (c as u8)).collect::<Vec<_>>();

    let entry_point = quote! {
        /// Returns a logical struct describing the entry point named `{ep_name}`.
        #[inline]
        #[allow(unsafe_code)]
        pub fn #method_ident(&self) -> #ty {
            unsafe {
                #[allow(dead_code)]
                static NAME: [u8; #ep_name_lenp1] = [ #( #encoded_ep_name ),* , 0];
                self.shader.#f_call
            }
        }
    };

    entry_point
}

struct Element {
    location: u32,
    name: String,
    format: String,
    location_len: usize,
    // The component mask represents which components in each of up to eight locations (dmat4 aka matrix 4 by 4 of double)
    // consumed by the element in the vertex input.
    // The order of the mask is such that the least significant bit corresponds to the first
    // component in the first location, the next bit corresponds to the second component, etc.
    // For example, if the element is a vec4, it consumes all four components of one location, and the
    // mask would be 0x000F or 0b0000_0000_0000_1111.
    // This is a mask because the components consumed are not always consecutive. Otherwise,
    // a length would be sufficient.
    // A None value means that the element consumes all components of every location for some length N, N may be large,
    // and is not able to be bound per-component. This is distinct from a filled mask, which means that the element 
    // consumes for at most eight locations.
    component_mask: Option<u32>,
}

fn write_interfaces(
    spirv: &Spirv,
    interface: &[Id],
    ignore_first_array_in: bool,
    ignore_first_array_out: bool,
) -> (TokenStream, TokenStream) {
    let mut input_elements = vec![];
    let mut output_elements = vec![];

    // Filling `input_elements` and `output_elements`.
    for &interface in interface.iter() {
        let interface_info = spirv.id(interface);

        match interface_info.instruction() {
            &Instruction::Variable {
                result_type_id,
                result_id,
                ref storage_class,
                ..
            } => {
                if spirv_search::is_builtin(spirv, result_id) {
                    continue;
                }

                let id_info = spirv.id(result_id);

                let (to_write, ignore_first_array) = match storage_class {
                    &StorageClass::Input => (&mut input_elements, ignore_first_array_in),
                    &StorageClass::Output => (&mut output_elements, ignore_first_array_out),
                    _ => continue,
                };

                let name = match id_info
                    .iter_name()
                    .find_map(|instruction| match instruction {
                        Instruction::Name { name, .. } => Some(name.as_str()),
                        _ => None,
                    }) {
                    Some(name) => name,
                    None => continue,
                };

                let location = id_info
                    .iter_decoration()
                    .find_map(|instruction| match instruction {
                        Instruction::Decorate {
                            decoration: Decoration::Location { location },
                            ..
                        } => Some(*location),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "Attribute `{}` (id {}) is missing a location",
                            name, result_id
                        )
                    });

                let component = id_info
                    .iter_decoration()
                    .find_map(|instruction| match instruction {
                        Instruction::Decorate {
                            decoration: Decoration::Component { component },
                            ..
                        } => Some(*component),
                        _ => None,
                    });
                // this assertion is insufficient by itself, format_from_id() will panic if the
                // component is not valid for the type.
                assert!(component.unwrap_or(0) < 4, "There are only four components in a location.");

                let (format, location_len, component_mask) =
                    spirv_search::format_from_id(spirv, result_type_id, ignore_first_array, component);
                to_write.push(Element {
                    location,
                    name: name.to_owned(),
                    format,
                    location_len,
                    component_mask
                });
            }
            _ => (),
        }
    }

    (
        write_interface(&input_elements),
        write_interface(&output_elements),
    )
}

fn write_interface(attributes: &[Element]) -> TokenStream {
    // Checking for overlapping elements.
    for (offset, element1) in attributes.iter().enumerate() {
        for element2 in attributes.iter().skip(offset + 1) {
            if element1.location == element2.location
                || (element1.location < element2.location
                    && element1.location + element1.location_len as u32 > element2.location)
                || (element2.location < element1.location
                    && element2.location + element2.location_len as u32 > element1.location)
            {
                if element1.component_mask.is_none() || element1.component_mask.is_none() {
                    panic!(
                        "The locations of attributes `{}` (start={}, len={}) \
                        and `{}` (start={}, len={}) overlap",
                        element1.name,
                        element1.location,
                        element1.location_len,
                        element2.name,
                        element2.location,
                        element2.location_len
                    );
                }
                else {

                    let mask1_at_loc = element1.component_mask.unwrap() >> (element2.location.checked_sub(element1.location).unwrap_or(0)) * 4;
                    let mask2_at_loc = element2.component_mask.unwrap() >> (element1.location.checked_sub(element2.location).unwrap_or(0)) * 4;
                    if mask1_at_loc & mask2_at_loc != 0 {
                        panic!(
                            "The components of attributes `{}` (loc={}, mask at loc={:#06x}) \
                            and `{}` (loc={}, mask at loc={:#06x}) overlap",
                            element1.name,
                            element1.location,
                            mask1_at_loc,
                            element2.name,
                            element2.location,
                            mask2_at_loc
                        );
                    }
                    else {
                        println!(
                            "The components of attributes `{}` (start={}, mask={:#08x}) \
                            and `{}` (start={}, mask={:#08x}) do not overlap",
                            element1.name,
                            element1.location,
                            mask1_at_loc,
                            element2.name,
                            element2.location,
                            mask2_at_loc
                        );
                    }
                }
            }
        }
    }

    let body = attributes
        .iter()
        .map(|element| {
            assert!(element.location_len >= 1);
            let loc = element.location;
            let loc_end = element.location + element.location_len as u32;
            let format = Ident::new(&element.format, Span::call_site());
            let name = &element.name;

            quote! {
                ::vulkano::pipeline::shader::ShaderInterfaceEntry {
                    location: #loc .. #loc_end,
                    format: ::vulkano::format::Format::#format,
                    name: Some(::std::borrow::Cow::Borrowed(#name))
                },
            }
        })
        .collect::<Vec<_>>();

    quote! {
        #[allow(unsafe_code)]
        unsafe {
            ::vulkano::pipeline::shader::ShaderInterface::new_unchecked(vec![
                #( #body )*
            ])
        }
    }
}
