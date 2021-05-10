// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor_sets::write_descriptor_sets;
use crate::enums::{Decoration, ExecutionMode, ExecutionModel, StorageClass};
use crate::parse::{Instruction, Spirv};
use crate::{spirv_search, TypesMeta};
use proc_macro2::{Span, TokenStream};
use syn::Ident;

pub(super) fn write_entry_point(
    doc: &Spirv,
    instruction: &Instruction,
    types_meta: &TypesMeta,
    exact_entrypoint_interface: bool,
) -> (TokenStream, TokenStream) {
    let (execution, id, ep_name, interface) = match instruction {
        &Instruction::EntryPoint {
            ref execution,
            id,
            ref name,
            ref interface,
            ..
        } => (execution, id, name, interface),
        _ => unreachable!(),
    };

    let capitalized_ep_name: String = ep_name
        .chars()
        .take(1)
        .flat_map(|c| c.to_uppercase())
        .chain(ep_name.chars().skip(1))
        .collect();

    let ignore_first_array_in = match *execution {
        ExecutionModel::ExecutionModelTessellationControl => true,
        ExecutionModel::ExecutionModelTessellationEvaluation => true,
        ExecutionModel::ExecutionModelGeometry => true,
        _ => false,
    };
    let ignore_first_array_out = match *execution {
        ExecutionModel::ExecutionModelTessellationControl => true,
        _ => false,
    };

    let (input_interface, output_interface) = write_interfaces(
        doc,
        interface,
        ignore_first_array_in,
        ignore_first_array_out,
    );

    let descriptor_sets_layout_name = Ident::new(
        format!("{}Layout", capitalized_ep_name).as_str(),
        Span::call_site(),
    );
    let descriptor_sets_layout_struct = write_descriptor_sets(
        &doc,
        &descriptor_sets_layout_name,
        id,
        interface,
        &types_meta,
        exact_entrypoint_interface,
    );

    let spec_consts_struct = if crate::spec_consts::has_specialization_constants(doc) {
        quote! { SpecializationConstants }
    } else {
        quote! { () }
    };

    let (ty, f_call) = {
        if let ExecutionModel::ExecutionModelGLCompute = *execution {
            (
                quote! { ::vulkano::pipeline::shader::ComputeEntryPoint<#spec_consts_struct, #descriptor_sets_layout_name> },
                quote! { compute_entry_point(
                    ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                    #descriptor_sets_layout_name(ShaderStages { compute: true, .. ShaderStages::none() })
                )},
            )
        } else {
            let entry_ty = match *execution {
                ExecutionModel::ExecutionModelVertex => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::Vertex }
                }

                ExecutionModel::ExecutionModelTessellationControl => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::TessellationControl }
                }

                ExecutionModel::ExecutionModelTessellationEvaluation => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::TessellationEvaluation }
                }

                ExecutionModel::ExecutionModelGeometry => {
                    let mut execution_mode = None;

                    for instruction in doc.instructions.iter() {
                        if let &Instruction::ExecutionMode {
                            target_id,
                            ref mode,
                            ..
                        } = instruction
                        {
                            if target_id == id {
                                execution_mode = match mode {
                                    &ExecutionMode::ExecutionModeInputPoints => {
                                        Some(quote! { Points })
                                    }
                                    &ExecutionMode::ExecutionModeInputLines => {
                                        Some(quote! { Lines })
                                    }
                                    &ExecutionMode::ExecutionModeInputLinesAdjacency => {
                                        Some(quote! { LinesWithAdjacency })
                                    }
                                    &ExecutionMode::ExecutionModeTriangles => {
                                        Some(quote! { Triangles })
                                    }
                                    &ExecutionMode::ExecutionModeInputTrianglesAdjacency => {
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

                ExecutionModel::ExecutionModelFragment => {
                    quote! { ::vulkano::pipeline::shader::GraphicsShaderType::Fragment }
                }

                ExecutionModel::ExecutionModelGLCompute => unreachable!(),

                ExecutionModel::ExecutionModelKernel => panic!("Kernels are not supported"),
            };

            let stage = match *execution {
                ExecutionModel::ExecutionModelVertex => {
                    quote! { ShaderStages { vertex: true, .. ShaderStages::none() } }
                }

                ExecutionModel::ExecutionModelTessellationControl => {
                    quote! { ShaderStages { tessellation_control: true, .. ShaderStages::none() } }
                }

                ExecutionModel::ExecutionModelTessellationEvaluation => {
                    quote! { ShaderStages { tessellation_evaluation: true, .. ShaderStages::none() } }
                }

                ExecutionModel::ExecutionModelGeometry => {
                    quote! { ShaderStages { geometry: true, .. ShaderStages::none() } }
                }

                ExecutionModel::ExecutionModelFragment => {
                    quote! { ShaderStages { fragment: true, .. ShaderStages::none() } }
                }

                ExecutionModel::ExecutionModelGLCompute => unreachable!(),
                ExecutionModel::ExecutionModelKernel => unreachable!(),
            };

            let ty = quote! {
                ::vulkano::pipeline::shader::GraphicsEntryPoint<
                    #spec_consts_struct,
                    #descriptor_sets_layout_name>
            };
            let f_call = quote! {
                graphics_entry_point(
                    ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                    #input_interface,
                    #output_interface,
                    #descriptor_sets_layout_name(#stage),
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

    (entry_point, descriptor_sets_layout_struct)
}

struct Element {
    location: u32,
    name: String,
    format: String,
    location_len: usize,
}

fn write_interfaces(
    doc: &Spirv,
    interface: &[u32],
    ignore_first_array_in: bool,
    ignore_first_array_out: bool,
) -> (TokenStream, TokenStream) {
    let mut input_elements = vec![];
    let mut output_elements = vec![];

    // Filling `input_elements` and `output_elements`.
    for interface in interface.iter() {
        for i in doc.instructions.iter() {
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
                        &StorageClass::StorageClassInput => {
                            (&mut input_elements, ignore_first_array_in)
                        }
                        &StorageClass::StorageClassOutput => {
                            (&mut output_elements, ignore_first_array_out)
                        }
                        _ => continue,
                    };

                    let name = spirv_search::name_from_id(doc, result_id);
                    if name == "__unnamed" {
                        continue;
                    } // FIXME: hack

                    let location = match doc
                        .get_decoration_params(result_id, Decoration::DecorationLocation)
                    {
                        Some(l) => l[0],
                        None => panic!(
                            "Attribute `{}` (id {}) is missing a location",
                            name, result_id
                        ),
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
            ::vulkano::pipeline::shader::ShaderInterface::new(::std::borrow::Cow::Borrowed(&[
                #( #body )*
            ]))
        }
    }
}
