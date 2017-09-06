// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;

use enums;
use parse;

/// Returns true if the document has specialization constants.
pub fn has_specialization_constants(doc: &parse::Spirv) -> bool {
    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::SpecConstantTrue { .. } => return true,
            &parse::Instruction::SpecConstantFalse { .. } => return true,
            &parse::Instruction::SpecConstant { .. } => return true,
            &parse::Instruction::SpecConstantComposite { .. } => return true,
            _ => (),
        }
    }

    false
}

/// Writes the `SpecializationConstants` struct that contains the specialization constants and
/// implements the `Default` and the `vulkano::pipeline::shader::SpecializationConstants` traits.
pub fn write_specialization_constants(doc: &parse::Spirv) -> String {
    struct SpecConst {
        name: String,
        constant_id: u32,
        rust_ty: String,
        rust_size: usize,
        rust_alignment: usize,
        default_value: String,
    }

    let mut spec_consts = Vec::new();

    for instruction in doc.instructions.iter() {
        let (type_id, result_id, default_value) = match instruction {
            &parse::Instruction::SpecConstantTrue {
                result_type_id,
                result_id,
            } => {
                (result_type_id, result_id, "1u32".to_string())
            },
            &parse::Instruction::SpecConstantFalse {
                result_type_id,
                result_id,
            } => {
                (result_type_id, result_id, "0u32".to_string())
            },
            &parse::Instruction::SpecConstant {
                result_type_id,
                result_id,
                ref data,
            } => {
                let data = data.iter()
                    .map(|d| d.to_string() + "u32")
                    .collect::<Vec<_>>()
                    .join(", ");
                let def_val = format!("unsafe {{ ::std::mem::transmute([{}]) }}", data);
                (result_type_id, result_id, def_val)
            },
            &parse::Instruction::SpecConstantComposite {
                result_type_id,
                result_id,
                ref data,
            } => {
                let data = data.iter()
                    .map(|d| d.to_string() + "u32")
                    .collect::<Vec<_>>()
                    .join(", ");
                let def_val = format!("unsafe {{ ::std::mem::transmute([{}]) }}", data);
                (result_type_id, result_id, def_val)
            },
            _ => continue,
        };

        let (rust_ty, rust_size, rust_alignment) = spec_const_type_from_id(doc, type_id);
        let rust_size = rust_size.expect("Found runtime-sized specialization constant");

        let constant_id = doc.instructions
            .iter()
            .filter_map(|i| match i {
                            &parse::Instruction::Decorate {
                                target_id,
                                decoration: enums::Decoration::DecorationSpecId,
                                ref params,
                            } if target_id == result_id => {
                                Some(params[0])
                            },
                            _ => None,
                        })
            .next()
            .expect("Found a specialization constant with no SpecId decoration");

        spec_consts.push(SpecConst {
                             name: ::name_from_id(doc, result_id),
                             constant_id,
                             rust_ty,
                             rust_size,
                             rust_alignment,
                             default_value,
                         });
    }

    let map_entries = {
        let mut map_entries = Vec::new();
        let mut curr_offset = 0;
        for c in &spec_consts {
            map_entries.push(format!(
                "SpecializationMapEntry {{
                constant_id: \
                 {},
                offset: {},
                size: {},
            \
                 }}",
                c.constant_id,
                curr_offset,
                c.rust_size
            ));

            assert_ne!(c.rust_size, 0);
            curr_offset += c.rust_size;
            curr_offset = c.rust_alignment * (1 + (curr_offset - 1) / c.rust_alignment);
        }
        map_entries
    };

    format!(
        r#"

#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
#[repr(C)]
pub struct SpecializationConstants {{
    {struct_def}
}}

impl Default for SpecializationConstants {{
    fn default() -> SpecializationConstants {{
        SpecializationConstants {{
            {def_vals}
        }}
    }}
}}

unsafe impl SpecConstsTrait for SpecializationConstants {{
    fn descriptors() -> &'static [SpecializationMapEntry] {{
        static DESCRIPTORS: [SpecializationMapEntry; {num_map_entries}] = [
            {map_entries}
        ];
        &DESCRIPTORS
    }}
}}

    "#,
        struct_def = spec_consts
            .iter()
            .map(|c| format!("pub {}: {}", c.name, c.rust_ty))
            .collect::<Vec<_>>()
            .join(", "),
        def_vals = spec_consts
            .iter()
            .map(|c| format!("{}: {}", c.name, c.default_value))
            .collect::<Vec<_>>()
            .join(", "),
        num_map_entries = map_entries.len(),
        map_entries = map_entries.join(", ")
    )
}

// Wrapper around `type_from_id` that also handles booleans.
fn spec_const_type_from_id(doc: &parse::Spirv, searched: u32) -> (String, Option<usize>, usize) {
    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::TypeBool { result_id } if result_id == searched => {
                return ("u32".to_owned(), Some(mem::size_of::<u32>()), mem::align_of::<u32>());
            },
            _ => (),
        }
    }

    ::structs::type_from_id(doc, searched)
}
