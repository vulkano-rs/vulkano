// Copyright (c) 2016 The vulkano developers
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

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub fn write_structs(doc: &parse::Spirv) -> String {
    let mut result = String::new();

    for instruction in &doc.instructions {
        match *instruction {
            parse::Instruction::TypeStruct {
                result_id,
                ref member_types,
            } => {
                let (s, _) = write_struct(doc, result_id, member_types);
                result.push_str(&s);
                result.push_str("\n");
            },
            _ => (),
        }
    }

    result
}

/// Represents a rust struct member
struct Member {
    name: String,
    value: String,
    offset: Option<usize>,
}

impl Member {
    fn declaration_text(&self) -> String {
        let offset = match self.offset {
            Some(o) => format!("/* offset: {} */", o),
            _ => "".to_owned(),
        };
        format!("    pub {}: {} {}", self.name, self.value, offset)
    }
    fn copy_text(&self) -> String {
        format!("            {name}: self.{name}", name = self.name)
    }
}

/// Analyzes a single struct, returns a string containing its Rust definition, plus its size.
fn write_struct(doc: &parse::Spirv, struct_id: u32, members: &[u32]) -> (String, Option<usize>) {
    let name = ::name_from_id(doc, struct_id);

    // The members of this struct.
    let mut rust_members = Vec::with_capacity(members.len());

    // Padding structs will be named `_paddingN` where `N` is determined by this variable.
    let mut next_padding_num = 0;

    // Contains the offset of the next field.
    // Equals to `None` if there's a runtime-sized field in there.
    let mut current_rust_offset = Some(0);

    for (num, &member) in members.iter().enumerate() {
        // Compute infos about the member.
        let (ty, rust_size, rust_align) = type_from_id(doc, member);
        let member_name = ::member_name_from_id(doc, struct_id, num as u32);

        // Ignore the whole struct is a member is built in, which includes
        // `gl_Position` for example.
        if is_builtin_member(doc, struct_id, num as u32) {
            return (String::new(), None); // TODO: is this correct? shouldn't it return a correct struct but with a flag or something?
        }

        // Finding offset of the current member, as requested by the SPIR-V code.
        let spirv_offset = doc.instructions
            .iter()
            .filter_map(|i| {
                match *i {
                    parse::Instruction::MemberDecorate {
                        target_id,
                        member,
                        decoration: enums::Decoration::DecorationOffset,
                        ref params,
                    } if target_id == struct_id && member as usize == num => {
                        return Some(params[0]);
                    },
                    _ => (),
                };

                None
            })
            .next();

        // Some structs don't have `Offset` decorations, in the case they are used as local
        // variables only. Ignoring these.
        let spirv_offset = match spirv_offset {
            Some(o) => o as usize,
            None => return (String::new(), None),        // TODO: shouldn't we return and let the caller ignore it instead?
        };

        // We need to add a dummy field if necessary.
        {
            let current_rust_offset =
                current_rust_offset
                    .as_mut()
                    .expect("Found runtime-sized member in non-final position");

            // Updating current_rust_offset to take the alignment of the next field into account
            *current_rust_offset = if *current_rust_offset == 0 {
                0
            } else {
                (1 + (*current_rust_offset - 1) / rust_align) * rust_align
            };

            if spirv_offset != *current_rust_offset {
                let diff = spirv_offset.checked_sub(*current_rust_offset).unwrap();
                let padding_num = next_padding_num;
                next_padding_num += 1;
                rust_members.push(Member {
                                      name: format!("_dummy{}", padding_num),
                                      value: format!("[u8; {}]", diff),
                                      offset: None,
                                  });
                *current_rust_offset += diff;
            }
        }

        // Updating `current_rust_offset`.
        if let Some(s) = rust_size {
            *current_rust_offset.as_mut().unwrap() += s;
        } else {
            current_rust_offset = None;
        }

        rust_members.push(Member {
                              name: member_name.to_owned(),
                              value: ty,
                              offset: Some(spirv_offset),
                          });
    }

    // Try determine the total size of the struct in order to add padding at the end of the struct.
    let spirv_req_total_size = doc.instructions
        .iter()
        .filter_map(|i| match *i {
                        parse::Instruction::Decorate {
                            target_id,
                            decoration: enums::Decoration::DecorationArrayStride,
                            ref params,
                        } => {
                            for inst in doc.instructions.iter() {
                                match *inst {
                                    parse::Instruction::TypeArray {
                                        result_id, type_id, ..
                                    } if result_id == target_id && type_id == struct_id => {
                                        return Some(params[0]);
                                    },
                                    parse::Instruction::TypeRuntimeArray { result_id, type_id }
                                        if result_id == target_id && type_id == struct_id => {
                                        return Some(params[0]);
                                    },
                                    _ => (),
                                }
                            }

                            None
                        },
                        _ => None,
                    })
        .fold(None, |a, b| if let Some(a) = a {
            assert_eq!(a, b);
            Some(a)
        } else {
            Some(b)
        });

    // Adding the final padding members.
    if let (Some(cur_size), Some(req_size)) = (current_rust_offset, spirv_req_total_size) {
        let diff = req_size.checked_sub(cur_size as u32).unwrap();
        if diff >= 1 {
            rust_members.push(Member {
                                  name: format!("_dummy{}", next_padding_num),
                                  value: format!("[u8; {}]", diff),
                                  offset: None,
                              });
        }
    }

    // We can only implement Clone if there's no unsized member in the struct.
    let (impl_text, derive_text) = if current_rust_offset.is_some() {
        let i = format!("\nimpl Clone for {name} {{\n    fn clone(&self) -> Self {{\n        \
                         {name} {{\n{copies}\n        }}\n    }}\n}}\n",
                        name = name,
                        copies = rust_members
                            .iter()
                            .map(Member::copy_text)
                            .collect::<Vec<_>>()
                            .join(",\n"));
        (i, "#[derive(Copy)]")
    } else {
        ("".to_owned(), "")
    };

    let s =
        format!("#[repr(C)]\n{derive_text}\n#[allow(non_snake_case)]\npub struct {name} \
                 {{\n{members}\n}} /* total_size: {t:?} */\n{impl_text}",
                name = name,
                members = rust_members
                    .iter()
                    .map(Member::declaration_text)
                    .collect::<Vec<_>>()
                    .join(",\n"),
                t = spirv_req_total_size,
                impl_text = impl_text,
                derive_text = derive_text);
    (s,
     spirv_req_total_size
         .map(|sz| sz as usize)
         .or(current_rust_offset))
}

/// Returns true if a `BuiltIn` decorator is applied on a struct member.
fn is_builtin_member(doc: &parse::Spirv, id: u32, member_id: u32) -> bool {
    for instruction in &doc.instructions {
        match *instruction {
            parse::Instruction::MemberDecorate {
                target_id,
                member,
                decoration: enums::Decoration::DecorationBuiltIn,
                ..
            } if target_id == id && member == member_id => {
                return true;
            },
            _ => (),
        }
    }

    false
}

/// Returns the type name to put in the Rust struct, and its size and alignment.
///
/// The size can be `None` if it's only known at runtime.
pub fn type_from_id(doc: &parse::Spirv, searched: u32) -> (String, Option<usize>, usize) {
    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::TypeBool { result_id } if result_id == searched => {
                panic!("Can't put booleans in structs")
            },
            &parse::Instruction::TypeInt {
                result_id,
                width,
                signedness,
            } if result_id == searched => {
                match (width, signedness) {
                    (8, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i8,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("i8".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (8, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u8,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("u8".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (16, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i16,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("i16".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (16, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u16,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("u16".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (32, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i32,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("i32".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (32, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u32,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("u32".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (64, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i64,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("i64".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    (64, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u64,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("u64".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    _ => panic!("No Rust equivalent for an integer of width {}", width),
                }
            },
            &parse::Instruction::TypeFloat { result_id, width } if result_id == searched => {
                match width {
                    32 => {
                        #[repr(C)]
                        struct Foo {
                            data: f32,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("f32".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    64 => {
                        #[repr(C)]
                        struct Foo {
                            data: f64,
                            after: u8,
                        }
                        let size = unsafe { (&(&*(0 as *const Foo)).after) as *const u8 as usize };
                        return ("f64".to_owned(), Some(size), mem::align_of::<Foo>());
                    },
                    _ => panic!("No Rust equivalent for a floating-point of width {}", width),
                }
            },
            &parse::Instruction::TypeVector {
                result_id,
                component_id,
                count,
            } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, t_size, t_align) = type_from_id(doc, component_id);
                return (format!("[{}; {}]", t, count), t_size.map(|s| s * count as usize), t_align);
            },
            &parse::Instruction::TypeMatrix {
                result_id,
                column_type_id,
                column_count,
            } if result_id == searched => {
                // FIXME: row-major or column-major
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, t_size, t_align) = type_from_id(doc, column_type_id);
                return (format!("[{}; {}]", t, column_count),
                        t_size.map(|s| s * column_count as usize),
                        t_align);
            },
            &parse::Instruction::TypeArray {
                result_id,
                type_id,
                length_id,
            } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, t_size, t_align) = type_from_id(doc, type_id);
                let len = doc.instructions
                    .iter()
                    .filter_map(|e| match e {
                                    &parse::Instruction::Constant {
                                        result_id,
                                        ref data,
                                        ..
                                    } if result_id == length_id => Some(data.clone()),
                                    _ => None,
                                })
                    .next()
                    .expect("failed to find array length");
                let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
                return (format!("[{}; {}]", t, len), t_size.map(|s| s * len as usize), t_align); // FIXME:
            },
            &parse::Instruction::TypeRuntimeArray { result_id, type_id }
                if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (t, _, t_align) = type_from_id(doc, type_id);
                return (format!("[{}]", t), None, t_align);
            },
            &parse::Instruction::TypeStruct {
                result_id,
                ref member_types,
            } if result_id == searched => {
                // TODO: take the Offset member decorate into account?
                let name = ::name_from_id(doc, result_id);
                let (_, size) = write_struct(doc, result_id, member_types);
                let align = member_types
                    .iter()
                    .map(|&t| type_from_id(doc, t).2)
                    .max()
                    .unwrap_or(1);
                return (name, size, align);
            },
            _ => (),
        }
    }

    panic!("Type #{} not found", searched)
}
