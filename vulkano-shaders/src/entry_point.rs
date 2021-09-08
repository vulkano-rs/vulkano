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
    doc: &Spirv,
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
        doc,
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

    let descriptor_set_layout_descs =
        write_descriptor_set_layout_descs(&doc, id, interface, exact_entrypoint_interface, &stage);
    let push_constant_ranges = write_push_constant_ranges(shader, &doc, &stage, &types_meta);

    let spec_consts_struct = if crate::spec_consts::has_specialization_constants(doc) {
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
                    let mut execution_mode = None;

                    for instruction in doc.instructions() {
                        if let &Instruction::ExecutionMode {
                            entry_point,
                            ref mode,
                            ..
                        } = instruction
                        {
                            if entry_point == id {
                                execution_mode = match mode {
                                    &ExecutionMode::InputPoints => Some(quote! { Points }),
                                    &ExecutionMode::InputLines => Some(quote! { Lines }),
                                    &ExecutionMode::InputLinesAdjacency => {
                                        Some(quote! { LinesWithAdjacency })
                                    }
                                    &ExecutionMode::Triangles => Some(quote! { Triangles }),
                                    &ExecutionMode::InputTrianglesAdjacency => {
                                        Some(quote! { TrianglesWithAdjacency })
                                    }
                                    _ => continue,
                                };
                                break;
                            }
                        }
                    }

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
}

fn write_interfaces(
    doc: &Spirv,
    interface: &[Id],
    ignore_first_array_in: bool,
    ignore_first_array_out: bool,
) -> (TokenStream, TokenStream) {
    let mut input_elements = vec![];
    let mut output_elements = vec![];

    // Filling `input_elements` and `output_elements`.
    for interface in interface.iter() {
        for i in doc.instructions() {
            match i {
                &Instruction::Variable {
                    result_type_id,
                    result_id,
                    ref storage_class,
                    ..
                } if &result_id == interface => {
                    if spirv_search::is_builtin(doc, result_id) {
                        continue;
                    }

                    let (to_write, ignore_first_array) = match storage_class {
                        &StorageClass::Input => (&mut input_elements, ignore_first_array_in),
                        &StorageClass::Output => (&mut output_elements, ignore_first_array_out),
                        _ => continue,
                    };

                    let name = spirv_search::name_from_id(doc, result_id);
                    if name == "__unnamed" {
                        continue;
                    } // FIXME: hack

                    let location = match doc.get_decoration_params(result_id, |d| {
                        matches!(d, Decoration::Location { .. })
                    }) {
                        Some(Decoration::Location { location }) => *location,
                        None => panic!(
                            "Attribute `{}` (id {}) is missing a location",
                            name, result_id
                        ),
                        _ => unreachable!(),
                    };

                    let (format, location_len) =
                        spirv_search::format_from_id(doc, result_type_id, ignore_first_array);
                    to_write.push(Element {
                        location,
                        name,
                        format,
                        location_len,
                    });
                }
                _ => (),
            }
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
                panic!(
                    "The locations of attributes `{}` (start={}, size={}) \
                    and `{}` (start={}, size={}) overlap",
                    element1.name,
                    element1.location,
                    element1.location_len,
                    element2.name,
                    element2.location,
                    element2.location_len
                );
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
