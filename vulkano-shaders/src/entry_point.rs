// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use syn::Ident;
use proc_macro2::{Span, TokenStream};

use crate::enums::{StorageClass, ExecutionModel, ExecutionMode, Decoration};
use crate::parse::{Instruction, Spirv};
use crate::spirv_search;

pub fn write_entry_point(doc: &Spirv, instruction: &Instruction) -> (TokenStream, TokenStream) {
    let (execution, id, ep_name, interface) = match instruction {
        &Instruction::EntryPoint { ref execution, id, ref name, ref interface, .. } =>
            (execution, id, name, interface),
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

    let interface_structs = write_interface_structs(
        doc,
        &capitalized_ep_name,
        interface,
        ignore_first_array_in,
        ignore_first_array_out
    );

    let spec_consts_struct = if crate::spec_consts::has_specialization_constants(doc) {
        quote!{ SpecializationConstants }
    } else {
        quote!{ () }
    };

    let (ty, f_call) = {
        if let ExecutionModel::ExecutionModelGLCompute = *execution {
            (
                quote!{ ::vulkano::pipeline::shader::ComputeEntryPoint<#spec_consts_struct, Layout> },
                quote!{ compute_entry_point(
                    ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                    Layout(ShaderStages { compute: true, .. ShaderStages::none() })
                )}
            )
        } else {
            let entry_ty = match *execution {
                ExecutionModel::ExecutionModelVertex =>
                    quote!{ ::vulkano::pipeline::shader::GraphicsShaderType::Vertex },

                ExecutionModel::ExecutionModelTessellationControl =>
                    quote!{ ::vulkano::pipeline::shader::GraphicsShaderType::TessellationControl },

                ExecutionModel::ExecutionModelTessellationEvaluation =>
                    quote!{ ::vulkano::pipeline::shader::GraphicsShaderType::TessellationEvaluation },

                ExecutionModel::ExecutionModelGeometry => {
                    let mut execution_mode = None;

                    for instruction in doc.instructions.iter() {
                        if let &Instruction::ExecutionMode { target_id, ref mode, .. } = instruction {
                            if target_id == id {
                                execution_mode = match mode {
                                    &ExecutionMode::ExecutionModeInputPoints => Some(quote!{ Points }),
                                    &ExecutionMode::ExecutionModeInputLines => Some(quote!{ Lines }),
                                    &ExecutionMode::ExecutionModeInputLinesAdjacency =>
                                        Some(quote!{ LinesWithAdjacency }),
                                    &ExecutionMode::ExecutionModeTriangles => Some(quote!{ Triangles }),
                                    &ExecutionMode::ExecutionModeInputTrianglesAdjacency =>
                                        Some(quote!{ TrianglesWithAdjacency }),
                                    _ => continue,
                                };
                                break;
                            }
                        }
                    }

                    quote!{
                        ::vulkano::pipeline::shader::GraphicsShaderType::Geometry(
                            ::vulkano::pipeline::shader::GeometryShaderExecutionMode::#execution_mode
                        )
                    }
                }

                ExecutionModel::ExecutionModelFragment =>
                    quote!{ ::vulkano::pipeline::shader::GraphicsShaderType::Fragment },

                ExecutionModel::ExecutionModelGLCompute => unreachable!(),

                ExecutionModel::ExecutionModelKernel => panic!("Kernels are not supported"),
            };

            let stage = match *execution {
                ExecutionModel::ExecutionModelVertex =>
                    quote!{ ShaderStages { vertex: true, .. ShaderStages::none() } },

                ExecutionModel::ExecutionModelTessellationControl =>
                    quote!{ ShaderStages { tessellation_control: true, .. ShaderStages::none() } },

                ExecutionModel::ExecutionModelTessellationEvaluation =>
                    quote!{ ShaderStages { tessellation_evaluation: true, .. ShaderStages::none() } },

                ExecutionModel::ExecutionModelGeometry =>
                    quote!{ ShaderStages { geometry: true, .. ShaderStages::none() } },

                ExecutionModel::ExecutionModelFragment =>
                    quote!{ ShaderStages { fragment: true, .. ShaderStages::none() } },

                ExecutionModel::ExecutionModelGLCompute => unreachable!(),
                ExecutionModel::ExecutionModelKernel => unreachable!(),
            };

            let mut capitalized_ep_name_input = capitalized_ep_name.clone();
            capitalized_ep_name_input.push_str("Input");
            let capitalized_ep_name_input = Ident::new(&capitalized_ep_name_input, Span::call_site());

            let mut capitalized_ep_name_output = capitalized_ep_name.clone();
            capitalized_ep_name_output.push_str("Output");
            let capitalized_ep_name_output = Ident::new(&capitalized_ep_name_output, Span::call_site());

            let ty = quote!{
                ::vulkano::pipeline::shader::GraphicsEntryPoint<
                    #spec_consts_struct,
                    #capitalized_ep_name_input,
                    #capitalized_ep_name_output,
                    Layout>
            };
            let f_call = quote!{
                graphics_entry_point(
                    ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                    #capitalized_ep_name_input,
                    #capitalized_ep_name_output,
                    Layout(#stage),
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

    let entry_point = quote!{
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

    (interface_structs, entry_point)
}

struct Element {
    location: u32,
    name: String,
    format: String,
    location_len: usize,
}

fn write_interface_structs(doc: &Spirv, capitalized_ep_name: &str, interface: &[u32],
                           ignore_first_array_in: bool, ignore_first_array_out: bool)
                           -> TokenStream {
    let mut input_elements = vec!();
    let mut output_elements = vec!();

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
                        &StorageClass::StorageClassInput =>
                            (&mut input_elements, ignore_first_array_in),
                        &StorageClass::StorageClassOutput =>
                            (&mut output_elements, ignore_first_array_out),
                        _ => continue,
                    };

                    let name = spirv_search::name_from_id(doc, result_id);
                    if name == "__unnamed" {
                        continue;
                    } // FIXME: hack

                    let location = match doc.get_decoration_params(result_id, Decoration::DecorationLocation) {
                        Some(l) => l[0],
                        None => panic!("Attribute `{}` (id {}) is missing a location", name, result_id),
                    };

                    let (format, location_len) = spirv_search::format_from_id(doc, result_type_id, ignore_first_array);
                    to_write.push(Element { location, name, format, location_len });
                },
                _ => (),
            }
        }
    }

    let input: TokenStream = write_interface_struct(&format!("{}Input", capitalized_ep_name), &input_elements);
    let output: TokenStream = write_interface_struct(&format!("{}Output", capitalized_ep_name), &output_elements);
    quote!{ #input #output }
}

fn write_interface_struct(struct_name_str: &str, attributes: &[Element]) -> TokenStream {
    // Checking for overlapping elements.
    for (offset, element1) in attributes.iter().enumerate() {
        for element2 in attributes.iter().skip(offset + 1) {
            if element1.location == element2.location ||
                (element1.location < element2.location && element1.location + element1.location_len as u32 > element2.location) ||
                (element2.location < element1.location && element2.location + element2.location_len as u32 > element1.location)
            {
                panic!("The locations of attributes `{}` (start={}, size={}) \
                    and `{}` (start={}, size={}) overlap",
                    element1.name,
                    element1.location,
                    element1.location_len,
                    element2.name,
                    element2.location,
                    element2.location_len);
            }
        }
    }

    let body = attributes
        .iter()
        .enumerate()
        .map(|(num, element)| {
            assert!(element.location_len >= 1);
            let loc = element.location;
            let loc_end = element.location + element.location_len as u32;
            let format = Ident::new(&element.format, Span::call_site());
            let name = &element.name;
            let num = num as u16;

            quote!{
                if self.num == #num {
                    self.num += 1;

                    return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry {
                        location: #loc .. #loc_end,
                        format: ::vulkano::format::Format::#format,
                        name: Some(::std::borrow::Cow::Borrowed(#name))
                    });
                }
            }
        })
        .collect::<Vec<_>>();

    let struct_name = Ident::new(struct_name_str, Span::call_site());

    let mut iter_name = struct_name.to_string();
    iter_name.push_str("Iter");
    let iter_name = Ident::new(&iter_name, Span::call_site());

    let len = attributes.len();

    quote!{
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        pub struct #struct_name;

        #[allow(unsafe_code)]
        unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for #struct_name {
            type Iter = #iter_name;
            fn elements(&self) -> #iter_name {
                 #iter_name { num: 0 }
            }
        }

        #[derive(Debug, Copy, Clone)]
        pub struct #iter_name { num: u16 }

        impl Iterator for #iter_name {
            type Item = ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                #( #body )*
                None
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = #len - self.num as usize;
                (len, Some(len))
            }
         }

        impl ExactSizeIterator for #iter_name {}
    }
}
