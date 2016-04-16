// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;

use parse;
use enums;

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub fn write_structs(doc: &parse::Spirv) -> String {
    let mut result = String::new();

    for instruction in &doc.instructions {
        match *instruction {
            parse::Instruction::TypeStruct { result_id, ref member_types } => {
                result.push_str(&write_struct(doc, result_id, member_types));
                result.push_str("\n");
            },
            _ => ()
        }
    }

    result
}

/// Writes a single struct.
fn write_struct(doc: &parse::Spirv, struct_id: u32, members: &[u32]) -> String {
    let name = ::name_from_id(doc, struct_id);

    let mut members_defs = Vec::with_capacity(members.len());

    // Contains the offset of the next field.
    // Equals to `None` if there's a runtime-sized field in there.
    let mut current_rust_offset = Some(0);

    for (num, &member) in members.iter().enumerate() {
        let (ty, rust_size, rust_align) = type_from_id(doc, member);
        let member_name = ::member_name_from_id(doc, struct_id, num as u32);

        // Ignore the whole struct is a member is built in, which includes
        // `gl_Position` for example.
        if is_builtin_member(doc, struct_id, num as u32) {
            return String::new();
        }

        let spirv_offset = doc.instructions.iter().filter_map(|i| {
            match *i {
                parse::Instruction::MemberDecorate { target_id, member,
                                                   decoration: enums::Decoration::DecorationOffset,
                                                   ref params } if target_id == struct_id &&
                                                                   member as usize == num =>
                {
                    return Some(params[0]);
                },
                _ => ()
            };

            None
        }).next();

        let spirv_stride = doc.instructions.iter().filter_map(|i| {
            match *i {
                parse::Instruction::Decorate { target_id,
                                               decoration: enums::Decoration::DecorationArrayStride,
                                               ref params } if target_id == member =>
                {
                    return Some(params[0]);
                },
                _ => ()
            };

            None
        }).next();

        // Some structs don't have `Offset` decorations, in the case they are used as local
        // variables only. Ignoring these.
        let spirv_offset = match spirv_offset {
            Some(o) => o as usize,
            None => return String::new()
        };

        // We need to add a dummy field if necessary.
        {
            let current_rust_offset = current_rust_offset.as_mut().expect("Found runtime-sized member in non-final position");

            // Updating current_rust_offset to take the alignment of the next field into account
            *current_rust_offset = if *current_rust_offset == 0 {
                0
            } else {
                (1 + (*current_rust_offset - 1) / rust_align) * rust_align
            };

            if spirv_offset != *current_rust_offset {
                let diff = spirv_offset.checked_sub(*current_rust_offset).unwrap();
                members_defs.push(format!("_dummy: [u8; {}]", diff));       // FIXME: fix name if there are multiple dummies
                *current_rust_offset += diff;
            }
        }

        // Updating `current_rust_offset`.
        if let Some(s) = rust_size {
            *current_rust_offset.as_mut().unwrap() += s;
        } else {
            current_rust_offset = None;
        }

        members_defs.push(format!("pub {name}: {ty}  /* offset: {offset}, stride: {stride:?} */",
                                  name = member_name, ty = ty, offset = spirv_offset, stride = spirv_stride));
    }

    // We can only derive common traits if there's no unsized member in the struct.
    let derive = if current_rust_offset.is_some() {
        "#[derive(Copy, Clone, Debug, Default)]\n"
    } else {
        ""
    };

    format!("#[repr(C)]\n{derive}\
             pub struct {name} {{\n\t{members}\n}}\n",
            derive = derive, name = name, members = members_defs.join(",\n\t"))
}

/// Returns true if a `BuiltIn` decorator is applied on a struct member.
fn is_builtin_member(doc: &parse::Spirv, id: u32, member_id: u32) -> bool {
    for instruction in &doc.instructions {
        match *instruction {
            parse::Instruction::MemberDecorate { target_id, member,
                                                 decoration: enums::Decoration::DecorationBuiltIn,
                                                 .. } if target_id == id && member == member_id =>
            {
                return true;
            },
            _ => ()
        }
    }

    false
}

/// Returns the type name to put in the Rust struct, and its size and alignment.
///
/// The size can be `None` if it's only known at runtime.
fn type_from_id(doc: &parse::Spirv, searched: u32) -> (String, Option<usize>, usize) {
    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::TypeBool { result_id } if result_id == searched => {
                return ("bool".to_owned(), Some(mem::size_of::<bool>()), mem::align_of::<bool>())
            },
            &parse::Instruction::TypeInt { result_id, width, signedness } if result_id == searched => {
                // FIXME: width
                return ("i32".to_owned(), Some(mem::size_of::<i32>()), mem::align_of::<i32>())
            },
            &parse::Instruction::TypeFloat { result_id, width } if result_id == searched => {
                // FIXME: width
                return ("f32".to_owned(), Some(mem::size_of::<f32>()), mem::align_of::<f32>())
            },
            &parse::Instruction::TypeVector { result_id, component_id, count } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, t_size, t_align) = type_from_id(doc, component_id);
                return (format!("[{}; {}]", t, count), t_size.map(|s| s * count as usize), t_align);
            },
            &parse::Instruction::TypeMatrix { result_id, column_type_id, column_count } if result_id == searched => {
                // FIXME: row-major or column-major
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, t_size, t_align) = type_from_id(doc, column_type_id);
                return (format!("[{}; {}]", t, column_count), t_size.map(|s| s * column_count as usize), t_align);
            },
            &parse::Instruction::TypeArray { result_id, type_id, length_id } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, t_size, t_align) = type_from_id(doc, type_id);
                let len = doc.instructions.iter().filter_map(|e| {
                    match e { &parse::Instruction::Constant { result_id, ref data, .. } if result_id == length_id => Some(data.clone()), _ => None }
                }).next().expect("failed to find array length");
                let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
                return (format!("[{}; {}]", t, len), t_size.map(|s| s * len as usize), t_align);       // FIXME:
            },
            &parse::Instruction::TypeRuntimeArray { result_id, type_id } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, _, t_align) = type_from_id(doc, type_id);
                return (format!("[{}]", t), None, t_align);
            },
            &parse::Instruction::TypeStruct { result_id, ref member_types } if result_id == searched => {
                let name = ::name_from_id(doc, result_id);
                let size = member_types.iter().filter_map(|&t| type_from_id(doc, t).1).fold(0, |a,b|a+b);
                let align = member_types.iter().map(|&t| type_from_id(doc, t).2).max().unwrap_or(1);
                return (name, Some(size), align);
            },
            _ => ()
        }
    }

    panic!("Type #{} not found", searched)
}
