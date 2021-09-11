// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::spirv::{Decoration, Id, Instruction, Spirv};

/// Returns the vulkano `Format` and number of occupied locations from an id.
///
/// If `ignore_first_array` is true, the function expects the outermost instruction to be
/// `OpTypeArray`. If it's the case, the OpTypeArray will be ignored. If not, the function will
/// panic.
pub fn format_from_id(spirv: &Spirv, searched: Id, ignore_first_array: bool) -> (String, usize) {
    let id_info = spirv.id(searched);

    match id_info.instruction() {
        &Instruction::TypeInt {
            width, signedness, ..
        } => {
            assert!(!ignore_first_array);
            let format = match (width, signedness) {
                (8, 1) => "R8_SINT",
                (8, 0) => "R8_UINT",
                (16, 1) => "R16_SINT",
                (16, 0) => "R16_UINT",
                (32, 1) => "R32_SINT",
                (32, 0) => "R32_UINT",
                (64, 1) => "R64_SINT",
                (64, 0) => "R64_UINT",
                _ => panic!(),
            };
            (format.to_string(), 1)
        }
        &Instruction::TypeFloat { width, .. } => {
            assert!(!ignore_first_array);
            let format = match width {
                32 => "R32_SFLOAT",
                64 => "R64_SFLOAT",
                _ => panic!(),
            };
            (format.to_string(), 1)
        }
        &Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => {
            assert!(!ignore_first_array);
            let (format, sz) = format_from_id(spirv, component_type, false);
            assert!(format.starts_with("R32"));
            assert_eq!(sz, 1);
            let format = match component_count {
                1 => format,
                2 => format!("R32G32{}", &format[3..]),
                3 => format!("R32G32B32{}", &format[3..]),
                4 => format!("R32G32B32A32{}", &format[3..]),
                _ => panic!("Found vector type with more than 4 elements"),
            };
            (format, sz)
        }
        &Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            assert!(!ignore_first_array);
            let (format, sz) = format_from_id(spirv, column_type, false);
            (format, sz * column_count as usize)
        }
        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            if ignore_first_array {
                format_from_id(spirv, element_type, false)
            } else {
                let (format, sz) = format_from_id(spirv, element_type, false);
                let len = spirv
                    .instructions()
                    .iter()
                    .filter_map(|e| match e {
                        &Instruction::Constant {
                            result_id,
                            ref value,
                            ..
                        } if result_id == length => Some(value.clone()),
                        _ => None,
                    })
                    .next()
                    .expect("failed to find array length");
                let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
                (format, sz * len as usize)
            }
        }
        &Instruction::TypePointer { ty, .. } => format_from_id(spirv, ty, ignore_first_array),
        _ => panic!("Type #{} not found or invalid", searched),
    }
}

/// Returns true if a `BuiltIn` decorator is applied on an id.
pub fn is_builtin(spirv: &Spirv, id: Id) -> bool {
    let id_info = spirv.id(id);

    if id_info.iter_decoration().any(|instruction| {
        matches!(
            instruction,
            Instruction::Decorate {
                decoration: Decoration::BuiltIn { .. },
                ..
            }
        )
    }) {
        return true;
    }

    if id_info
        .iter_members()
        .flat_map(|member_info| member_info.iter_decoration())
        .any(|instruction| {
            matches!(
                instruction,
                Instruction::MemberDecorate {
                    decoration: Decoration::BuiltIn { .. },
                    ..
                }
            )
        })
    {
        return true;
    }

    match id_info.instruction() {
        Instruction::Variable { result_type_id, .. } => {
            return is_builtin(spirv, *result_type_id);
        }
        Instruction::TypeArray { element_type, .. } => {
            return is_builtin(spirv, *element_type);
        }
        Instruction::TypeRuntimeArray { element_type, .. } => {
            return is_builtin(spirv, *element_type);
        }
        Instruction::TypeStruct { member_types, .. } => {
            if member_types.iter().any(|ty| is_builtin(spirv, *ty)) {
                return true;
            }
        }
        Instruction::TypePointer { ty, .. } => {
            return is_builtin(spirv, *ty);
        }
        _ => (),
    }

    false
}
