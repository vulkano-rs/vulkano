// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::parse::{Instruction, Spirv};
use crate::enums::Decoration;

/// Returns the vulkano `Format` and number of occupied locations from an id.
///
/// If `ignore_first_array` is true, the function expects the outermost instruction to be
/// `OpTypeArray`. If it's the case, the OpTypeArray will be ignored. If not, the function will
/// panic.
pub fn format_from_id(doc: &Spirv, searched: u32, ignore_first_array: bool) -> (String, usize) {
    for instruction in doc.instructions.iter() {
        match instruction {
            &Instruction::TypeInt {
                result_id,
                width,
                signedness,
            } if result_id == searched => {
                assert!(!ignore_first_array);
                let format = match (width, signedness) {
                    (8, true) => "R8Sint",
                    (8, false) => "R8Uint",
                    (16, true) => "R16Sint",
                    (16, false) => "R16Uint",
                    (32, true) => "R32Sint",
                    (32, false) => "R32Uint",
                    (64, true) => "R64Sint",
                    (64, false) => "R64Uint",
                    _ => panic!(),
                };
                return (format.to_string(), 1);
            },
            &Instruction::TypeFloat { result_id, width } if result_id == searched => {
                assert!(!ignore_first_array);
                let format = match width {
                    32 => "R32Sfloat",
                    64 => "R64Sfloat",
                    _ => panic!(),
                };
                return (format.to_string(), 1);
            }
            &Instruction::TypeVector {
                result_id,
                component_id,
                count,
            } if result_id == searched => {
                assert!(!ignore_first_array);
                let (format, sz) = format_from_id(doc, component_id, false);
                assert!(format.starts_with("R32"));
                assert_eq!(sz, 1);
                let format = match count {
                    1 => format,
                    2 => format!("R32G32{}", &format[3 ..]),
                    3 => format!("R32G32B32{}", &format[3 ..]),
                    4 => format!("R32G32B32A32{}", &format[3 ..]),
                    _ => panic!("Found vector type with more than 4 elements")
                };
                return (format, sz);
            },
            &Instruction::TypeMatrix {
                result_id,
                column_type_id,
                column_count,
            } if result_id == searched => {
                assert!(!ignore_first_array);
                let (format, sz) = format_from_id(doc, column_type_id, false);
                return (format, sz * column_count as usize);
            },
            &Instruction::TypeArray {
                result_id,
                type_id,
                length_id,
            } if result_id == searched => {
                if ignore_first_array {
                    return format_from_id(doc, type_id, false);
                }

                let (format, sz) = format_from_id(doc, type_id, false);
                let len = doc.instructions
                    .iter()
                    .filter_map(|e| match e {
                        &Instruction::Constant { result_id, ref data, .. }
                            if result_id == length_id => Some(data.clone()),
                        _ => None,
                    })
                    .next()
                    .expect("failed to find array length");
                let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
                return (format, sz * len as usize);
            },
            &Instruction::TypePointer { result_id, type_id, .. }
                if result_id == searched => {
                return format_from_id(doc, type_id, ignore_first_array);
            },
            _ => (),
        }
    }

    panic!("Type #{} not found or invalid", searched)
}

pub fn name_from_id(doc: &Spirv, searched: u32) -> String {
    for instruction in &doc.instructions {
        if let &Instruction::Name { target_id, ref name } = instruction {
            if target_id == searched {
                return name.clone()
            }
        }
    }

    String::from("__unnamed")
}

pub fn member_name_from_id(doc: &Spirv, searched: u32, searched_member: u32) -> String {
    for instruction in &doc.instructions {
        if let &Instruction::MemberName { target_id, member, ref name } = instruction {
            if target_id == searched && member == searched_member {
                return name.clone()
            }
        }
    }

    String::from("__unnamed")
}

/// Returns true if a `BuiltIn` decorator is applied on an id.
pub fn is_builtin(doc: &Spirv, id: u32) -> bool {
    if doc.get_decoration_params(id, Decoration::DecorationBuiltIn).is_some() {
        return true
    }
    if doc.get_member_decoration_builtin_params(id).is_some() {
        return true
    }

    for instruction in &doc.instructions {
        match *instruction {
            Instruction::Variable {
                result_type_id,
                result_id,
                ..
            } if result_id == id => {
                return is_builtin(doc, result_type_id);
            }
            Instruction::TypeArray { result_id, type_id, .. } if result_id == id => {
                return is_builtin(doc, type_id);
            }
            Instruction::TypeRuntimeArray { result_id, type_id } if result_id == id => {
                return is_builtin(doc, type_id);
            }
            Instruction::TypeStruct {
                result_id,
                ref member_types,
            } if result_id == id => {
                for &mem in member_types {
                    if is_builtin(doc, mem) {
                        return true;
                    }
                }
            }
            Instruction::TypePointer { result_id, type_id, .. } if result_id == id => {
                return is_builtin(doc, type_id);
            }
            _ => ()
        }
    }

    false
}
