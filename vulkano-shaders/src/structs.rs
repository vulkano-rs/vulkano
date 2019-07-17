// Copyright (c) 2016 The vulkano developers
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

use crate::parse::{Instruction, Spirv};
use crate::enums::Decoration;
use crate::spirv_search;

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub fn write_structs(doc: &Spirv) -> TokenStream {
    let mut structs = vec!();
    for instruction in &doc.instructions {
        match *instruction {
            Instruction::TypeStruct { result_id, ref member_types } =>
                structs.push(write_struct(doc, result_id, member_types).0),
            _ => ()
        }
    }

    quote!{
        #( #structs )*
    }
}

/// Analyzes a single struct, returns a string containing its Rust definition, plus its size.
fn write_struct(doc: &Spirv, struct_id: u32, members: &[u32]) -> (TokenStream, Option<usize>) {
    let name = Ident::new(&spirv_search::name_from_id(doc, struct_id), Span::call_site());

    // The members of this struct.
    struct Member {
        pub name: Ident,
        pub ty: TokenStream,
    }
    let mut rust_members = Vec::with_capacity(members.len());

    // Padding structs will be named `_paddingN` where `N` is determined by this variable.
    let mut next_padding_num = 0;

    // Contains the offset of the next field.
    // Equals to `None` if there's a runtime-sized field in there.
    let mut current_rust_offset = Some(0);

    for (num, &member) in members.iter().enumerate() {
        // Compute infos about the member.
        let (ty, rust_size, rust_align) = type_from_id(doc, member);
        let member_name = spirv_search::member_name_from_id(doc, struct_id, num as u32);

        // Ignore the whole struct is a member is built in, which includes
        // `gl_Position` for example.
        if doc.get_member_decoration_params(struct_id, num as u32, Decoration::DecorationBuiltIn).is_some() {
            return (quote!{}, None); // TODO: is this correct? shouldn't it return a correct struct but with a flag or something?
        }

        // Finding offset of the current member, as requested by the SPIR-V code.
        let spirv_offset = doc.get_member_decoration_params(struct_id, num as u32, Decoration::DecorationOffset)
            .map(|x| x[0]);

        // Some structs don't have `Offset` decorations, in the case they are used as local
        // variables only. Ignoring these.
        let spirv_offset = match spirv_offset {
            Some(o) => o as usize,
            None => return (quote!{}, None), // TODO: shouldn't we return and let the caller ignore it instead?
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
                    name: Ident::new(&format!("_dummy{}", padding_num), Span::call_site()),
                    ty: quote!{ [u8; #diff] },
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
            name: Ident::new(&member_name, Span::call_site()),
            ty,
        });
    }

    // Try determine the total size of the struct in order to add padding at the end of the struct.
    let mut spirv_req_total_size = None;
    for inst in doc.instructions.iter() {
        match *inst {
            Instruction::TypeArray { result_id, type_id, .. } if type_id == struct_id => {
                if let Some(params) = doc.get_decoration_params(result_id, Decoration::DecorationArrayStride) {
                    spirv_req_total_size = Some(params[0]);
                }
            }
            Instruction::TypeRuntimeArray { result_id, type_id } if type_id == struct_id => {
                if let Some(params) = doc.get_decoration_params(result_id, Decoration::DecorationArrayStride) {
                    spirv_req_total_size = Some(params[0]);
                }
            }
            _ => ()
        }
    }

    // Adding the final padding members.
    if let (Some(cur_size), Some(req_size)) = (current_rust_offset, spirv_req_total_size) {
        let diff = req_size.checked_sub(cur_size as u32).unwrap();
        if diff >= 1 {
            rust_members.push(Member {
                name: Ident::new(&format!("_dummy{}", next_padding_num), Span::call_site()),
                ty: quote!{ [u8; #diff as usize] },
            });
        }
    }

    // We can only implement Clone if there's no unsized member in the struct.
    let (clone_impl, copy_derive) = if current_rust_offset.is_some() {
        let mut copies = vec!();
        for member in &rust_members {
            let name = &member.name;
            copies.push(quote!{ #name: self.#name, });
        }
        (
            // Clone is implemented manually because members can be large arrays
            // that do not implement Clone, but do implement Copy
            quote!{
                impl Clone for #name {
                    fn clone(&self) -> Self {
                        #name {
                            #( #copies )*
                        }
                    }
                }
            },
            quote!{ #[derive(Copy)] }
        )
    } else {
        (quote!{}, quote!{})
    };

    let mut members = vec!();
    for member in &rust_members {
        let name = &member.name;
        let ty = &member.ty;
        members.push(quote!(pub #name: #ty,));
    }

    let ast = quote! {
        #[repr(C)]
        #copy_derive
        #[allow(non_snake_case)]
        pub struct #name {
            #( #members )*
        }
        #clone_impl
    };

    (ast, spirv_req_total_size.map(|sz| sz as usize).or(current_rust_offset))
}

/// Returns the type name to put in the Rust struct, and its size and alignment.
///
/// The size can be `None` if it's only known at runtime.
pub fn type_from_id(doc: &Spirv, searched: u32) -> (TokenStream, Option<usize>, usize) {
    for instruction in doc.instructions.iter() {
        match instruction {
            &Instruction::TypeBool { result_id } if result_id == searched => {
                panic!("Can't put booleans in structs")
            }
            &Instruction::TypeInt { result_id, width, signedness } if result_id == searched => {
                match (width, signedness) {
                    (8, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i8,
                            after: u8,
                        }
                        return (quote!{i8}, Some(std::mem::size_of::<i8>()), mem::align_of::<Foo>());
                    },
                    (8, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u8,
                            after: u8,
                        }
                        return (quote!{u8}, Some(std::mem::size_of::<u8>()), mem::align_of::<Foo>());
                    },
                    (16, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i16,
                            after: u8,
                        }
                        return (quote!{i16}, Some(std::mem::size_of::<i16>()), mem::align_of::<Foo>());
                    },
                    (16, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u16,
                            after: u8,
                        }
                        return (quote!{u16}, Some(std::mem::size_of::<u16>()), mem::align_of::<Foo>());
                    },
                    (32, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i32,
                            after: u8,
                        }
                        return (quote!{i32}, Some(std::mem::size_of::<i32>()), mem::align_of::<Foo>());
                    },
                    (32, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u32,
                            after: u8,
                        }
                        return (quote!{u32}, Some(std::mem::size_of::<u32>()), mem::align_of::<Foo>());
                    },
                    (64, true) => {
                        #[repr(C)]
                        struct Foo {
                            data: i64,
                            after: u8,
                        }
                        return (quote!{i64}, Some(std::mem::size_of::<i64>()), mem::align_of::<Foo>());
                    },
                    (64, false) => {
                        #[repr(C)]
                        struct Foo {
                            data: u64,
                            after: u8,
                        }
                        return (quote!{u64}, Some(std::mem::size_of::<u64>()), mem::align_of::<Foo>());
                    },
                    _ => panic!("No Rust equivalent for an integer of width {}", width),
                }
            }
            &Instruction::TypeFloat { result_id, width } if result_id == searched => {
                match width {
                    32 => {
                        #[repr(C)]
                        struct Foo {
                            data: f32,
                            after: u8,
                        }
                        return (quote!{f32}, Some(std::mem::size_of::<f32>()), mem::align_of::<Foo>());
                    },
                    64 => {
                        #[repr(C)]
                        struct Foo {
                            data: f64,
                            after: u8,
                        }
                        return (quote!{f64}, Some(std::mem::size_of::<f64>()), mem::align_of::<Foo>());
                    },
                    _ => panic!("No Rust equivalent for a floating-point of width {}", width),
                }
            }
            &Instruction::TypeVector {
                result_id,
                component_id,
                count,
            } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (ty, t_size, t_align) = type_from_id(doc, component_id);
                let array_length = count as usize;
                let size = t_size.map(|s| s * count as usize);
                return (quote!{ [#ty; #array_length] }, size, t_align);
            }
            &Instruction::TypeMatrix {
                result_id,
                column_type_id,
                column_count,
            } if result_id == searched => {
                // FIXME: row-major or column-major
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (ty, t_size, t_align) = type_from_id(doc, column_type_id);
                let array_length = column_count as usize;
                let size = t_size.map(|s| s * column_count as usize);
                return (quote!{ [#ty; #array_length] }, size, t_align);
            }
            &Instruction::TypeArray {
                result_id,
                type_id,
                length_id,
            } if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (ty, t_size, t_align) = type_from_id(doc, type_id);
                let t_size = t_size.expect("array components must be sized");
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
                let stride = doc.get_decoration_params(searched, Decoration::DecorationArrayStride).unwrap()[0];
                if stride as usize > t_size {
                    panic!("Not possible to generate a rust array with the correct alignment since the SPIR-V \
                            ArrayStride is larger than the size of the array element in rust. Try wrapping \
                            the array element in a struct or rounding up the size of a vector or matrix \
                            (e.g. increase a vec3 to a vec4)")
                }
                let array_length = len as usize;
                let size = Some(t_size * len as usize);
                return (quote!{ [#ty; #array_length] }, size, t_align);
            }
            &Instruction::TypeRuntimeArray { result_id, type_id }
                if result_id == searched => {
                debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
                let (ty, _, t_align) = type_from_id(doc, type_id);
                return (quote!{ [#ty] }, None, t_align);
            }
            &Instruction::TypeStruct {
                result_id,
                ref member_types,
            } if result_id == searched => {
                // TODO: take the Offset member decorate into account?
                let name = Ident::new(&spirv_search::name_from_id(doc, result_id), Span::call_site());
                let ty = quote!{ #name };
                let (_, size) = write_struct(doc, result_id, member_types);
                let align = member_types
                    .iter()
                    .map(|&t| type_from_id(doc, t).2)
                    .max()
                    .unwrap_or(1);
                return (ty, size, align);
            },
            _ => (),
        }
    }

    panic!("Type #{} not found", searched)
}
