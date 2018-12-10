// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;

use syn::Ident;
use proc_macro2::{Span, TokenStream};

use crate::enums::Decoration;
use crate::parse::{Instruction, Spirv};
use crate::spirv_search;
use crate::structs;

/// Returns true if the document has specialization constants.
pub fn has_specialization_constants(doc: &Spirv) -> bool {
    for instruction in doc.instructions.iter() {
        match instruction {
            &Instruction::SpecConstantTrue { .. } => return true,
            &Instruction::SpecConstantFalse { .. } => return true,
            &Instruction::SpecConstant { .. } => return true,
            &Instruction::SpecConstantComposite { .. } => return true,
            _ => (),
        }
    }

    false
}

/// Writes the `SpecializationConstants` struct that contains the specialization constants and
/// implements the `Default` and the `vulkano::pipeline::shader::SpecializationConstants` traits.
pub fn write_specialization_constants(doc: &Spirv) -> TokenStream {
    struct SpecConst {
        name: String,
        constant_id: u32,
        rust_ty: TokenStream,
        rust_size: usize,
        rust_alignment: u32,
        default_value: TokenStream,
    }

    let mut spec_consts = Vec::new();

    for instruction in doc.instructions.iter() {
        let (type_id, result_id, default_value) = match instruction {
            &Instruction::SpecConstantTrue { result_type_id, result_id } =>
                (result_type_id, result_id, quote!{1u32}),

            &Instruction::SpecConstantFalse { result_type_id, result_id } =>
                (result_type_id, result_id, quote!{0u32}),

            &Instruction::SpecConstant { result_type_id, result_id, ref data } => {
                let def_val = quote!{
                    unsafe {{ ::std::mem::transmute([ #( #data ),* ]) }}
                };
                (result_type_id, result_id, def_val)
            }
            &Instruction::SpecConstantComposite { result_type_id, result_id, ref data } => {
                let def_val = quote!{
                    unsafe {{ ::std::mem::transmute([ #( #data ),* ]) }}
                };
                (result_type_id, result_id, def_val)
            }
            _ => continue,
        };

        let (rust_ty, rust_size, rust_alignment) = spec_const_type_from_id(doc, type_id);
        let rust_size = rust_size.expect("Found runtime-sized specialization constant");

        let constant_id = doc.get_decoration_params(result_id, Decoration::DecorationSpecId)
            .unwrap()[0];

        spec_consts.push(SpecConst {
            name: spirv_search::name_from_id(doc, result_id),
            constant_id,
            rust_ty,
            rust_size,
            rust_alignment: rust_alignment as u32,
            default_value,
        });
    }

    let map_entries = {
        let mut map_entries = Vec::new();
        let mut curr_offset = 0;
        for spec_const in &spec_consts {
            let constant_id = spec_const.constant_id;
            let rust_size = spec_const.rust_size;
            map_entries.push(quote!{
                SpecializationMapEntry {
                    constant_id: #constant_id,
                    offset: #curr_offset,
                    size: #rust_size,
                }
            });

            assert_ne!(spec_const.rust_size, 0);
            curr_offset += spec_const.rust_size as u32;
            curr_offset = spec_const.rust_alignment * (1 + (curr_offset - 1) / spec_const.rust_alignment);
        }
        map_entries
    };

    let num_map_entries = map_entries.len();

    let mut struct_members = vec!();
    let mut struct_member_defaults = vec!();
    for spec_const in spec_consts {
        let name = Ident::new(&spec_const.name, Span::call_site());
        let rust_ty = spec_const.rust_ty;
        let default_value = spec_const.default_value;
        struct_members.push(quote!{ pub #name: #rust_ty });
        struct_member_defaults.push(quote!{ #name: #default_value });
    }

    quote!{
        #[derive(Debug, Copy, Clone)]
        #[allow(non_snake_case)]
        #[repr(C)]
        pub struct SpecializationConstants {
            #( #struct_members ),*
        }

        impl Default for SpecializationConstants {
            fn default() -> SpecializationConstants {
                SpecializationConstants {
                    #( #struct_member_defaults ),*
                }
            }
        }

        unsafe impl SpecConstsTrait for SpecializationConstants {
            fn descriptors() -> &'static [SpecializationMapEntry] {
                static DESCRIPTORS: [SpecializationMapEntry; #num_map_entries] = [
                    #( #map_entries ),*
                ];
                &DESCRIPTORS
            }
        }
    }
}

// Wrapper around `type_from_id` that also handles booleans.
fn spec_const_type_from_id(doc: &Spirv, searched: u32) -> (TokenStream, Option<usize>, usize) {
    for instruction in doc.instructions.iter() {
        match instruction {
            &Instruction::TypeBool { result_id } if result_id == searched => {
                return (quote!{u32}, Some(mem::size_of::<u32>()), mem::align_of::<u32>());
            },
            _ => (),
        }
    }

    structs::type_from_id(doc, searched)
}
