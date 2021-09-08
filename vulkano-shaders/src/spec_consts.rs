// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{spirv_search, TypesMeta};
use crate::{structs, RegisteredType};
use proc_macro2::{Span, TokenStream};
use std::borrow::Cow;
use std::collections::HashMap;
use std::mem;
use syn::Ident;
use vulkano::spirv::{Decoration, Id, Instruction, Spirv};

/// Returns true if the document has specialization constants.
pub fn has_specialization_constants(doc: &Spirv) -> bool {
    for instruction in doc.instructions() {
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
pub(super) fn write_specialization_constants<'a>(
    shader: &'a str,
    doc: &Spirv,
    types_meta: &TypesMeta,
    shared_constants: bool,
    types_registry: &'a mut HashMap<String, RegisteredType>,
) -> TokenStream {
    struct SpecConst {
        name: String,
        constant_id: u32,
        rust_ty: TokenStream,
        rust_signature: Cow<'static, str>,
        rust_size: usize,
        rust_alignment: u32,
        default_value: TokenStream,
    }

    let mut spec_consts = Vec::new();

    for instruction in doc.instructions() {
        let (type_id, result_id, default_value) = match instruction {
            &Instruction::SpecConstantTrue {
                result_type_id,
                result_id,
            } => (result_type_id, result_id, quote! {1u32}),

            &Instruction::SpecConstantFalse {
                result_type_id,
                result_id,
            } => (result_type_id, result_id, quote! {0u32}),

            &Instruction::SpecConstant {
                result_type_id,
                result_id,
                ref value,
            } => {
                let def_val = quote! {
                    unsafe {{ ::std::mem::transmute([ #( #value ),* ]) }}
                };
                (result_type_id, result_id, def_val)
            }
            &Instruction::SpecConstantComposite {
                result_type_id,
                result_id,
                ref constituents,
            } => {
                let constituents = constituents.iter().map(|&id| u32::from(id));
                let def_val = quote! {
                    unsafe {{ ::std::mem::transmute([ #( #constituents ),* ]) }}
                };
                (result_type_id, result_id, def_val)
            }
            _ => continue,
        };

        let (rust_ty, rust_signature, rust_size, rust_alignment) =
            spec_const_type_from_id(shader, doc, type_id, types_meta);
        let rust_size = rust_size.expect("Found runtime-sized specialization constant");

        let constant_id =
            doc.get_decoration_params(result_id, |d| matches!(d, Decoration::SpecId { .. }));

        if let Some(&Decoration::SpecId {
            specialization_constant_id: constant_id,
        }) = constant_id
        {
            let mut name = spirv_search::name_from_id(doc, result_id);

            if name == "__unnamed".to_owned() {
                name = String::from(format!("constant_{}", constant_id));
            }

            spec_consts.push(SpecConst {
                name,
                constant_id,
                rust_ty,
                rust_signature,
                rust_size,
                rust_alignment: rust_alignment as u32,
                default_value,
            });
        }
    }

    let struct_name = Ident::new(
        &format!(
            "{}SpecializationConstants",
            if shared_constants { "" } else { shader }
        ),
        Span::call_site(),
    );

    // For multi-constants mode registration mechanism skipped
    if shared_constants {
        let target_type = RegisteredType {
            shader: shader.to_string(),
            signature: spec_consts
                .iter()
                .map(|member| (member.name.to_string(), member.rust_signature.clone()))
                .collect(),
        };

        let name = struct_name.to_string();

        // Checking with Registry if this struct already registered by another shader, and if their
        // signatures match.
        if let Some(registered) = types_registry.get(name.as_str()) {
            registered.assert_signatures(name.as_str(), &target_type);

            // If the struct already registered and matches this one, skip duplicate.
            return quote! {};
        }

        debug_assert!(types_registry.insert(name, target_type).is_none());
    }

    let map_entries = {
        let mut map_entries = Vec::new();
        let mut curr_offset = 0;
        for spec_const in &spec_consts {
            let constant_id = spec_const.constant_id;
            let rust_size = spec_const.rust_size;
            map_entries.push(quote! {
                SpecializationMapEntry {
                    constant_id: #constant_id,
                    offset: #curr_offset,
                    size: #rust_size,
                }
            });

            assert_ne!(spec_const.rust_size, 0);
            curr_offset += spec_const.rust_size as u32;
            curr_offset =
                spec_const.rust_alignment * (1 + (curr_offset - 1) / spec_const.rust_alignment);
        }
        map_entries
    };

    let num_map_entries = map_entries.len();

    let mut struct_members = vec![];
    let mut struct_member_defaults = vec![];
    for spec_const in spec_consts {
        let name = Ident::new(&spec_const.name, Span::call_site());
        let rust_ty = spec_const.rust_ty;
        let default_value = spec_const.default_value;
        struct_members.push(quote! { pub #name: #rust_ty });
        struct_member_defaults.push(quote! { #name: #default_value });
    }

    quote! {
        #[derive(Debug, Copy, Clone)]
        #[allow(non_snake_case)]
        #[repr(C)]
        pub struct #struct_name {
            #( #struct_members ),*
        }

        impl Default for #struct_name {
            fn default() -> #struct_name {
                #struct_name {
                    #( #struct_member_defaults ),*
                }
            }
        }

        unsafe impl SpecConstsTrait for #struct_name {
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
fn spec_const_type_from_id(
    shader: &str,
    doc: &Spirv,
    searched: Id,
    types_meta: &TypesMeta,
) -> (TokenStream, Cow<'static, str>, Option<usize>, usize) {
    for instruction in doc.instructions() {
        match instruction {
            &Instruction::TypeBool { result_id } if result_id == searched => {
                return (
                    quote! {u32},
                    Cow::from("u32"),
                    Some(mem::size_of::<u32>()),
                    mem::align_of::<u32>(),
                );
            }
            _ => (),
        }
    }

    structs::type_from_id(shader, doc, searched, types_meta)
}
