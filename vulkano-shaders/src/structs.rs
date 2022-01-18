// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{RegisteredType, TypesMeta};
use heck::ToUpperCamelCase;
use proc_macro2::{Span, TokenStream};
use std::borrow::Cow;
use std::collections::HashMap;
use std::mem;
use syn::Ident;
use syn::LitStr;
use vulkano::shader::spirv::{Decoration, Id, Instruction, Spirv};

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub(super) fn write_structs<'a>(
    shader: &'a str,
    spirv: &Spirv,
    types_meta: &TypesMeta,
    types_registry: &'a mut HashMap<String, RegisteredType>,
) -> TokenStream {
    let structs = spirv
        .iter_global()
        .filter_map(|instruction| match instruction {
            Instruction::TypeStruct {
                result_id,
                member_types,
            } => Some(
                write_struct(
                    shader,
                    spirv,
                    *result_id,
                    member_types,
                    types_meta,
                    Some(types_registry),
                )
                .0,
            ),
            _ => None,
        });

    quote! {
        #( #structs )*
    }
}

/// Analyzes a single struct, returns a string containing its Rust definition, plus its size.
fn write_struct<'a>(
    shader: &'a str,
    spirv: &Spirv,
    struct_id: Id,
    members: &[Id],
    types_meta: &TypesMeta,
    types_registry: Option<&'a mut HashMap<String, RegisteredType>>,
) -> (TokenStream, Option<usize>) {
    let id_info = spirv.id(struct_id);
    let name = Ident::new(
        id_info
            .iter_name()
            .find_map(|instruction| match instruction {
                Instruction::Name { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .unwrap_or("__unnamed"),
        Span::call_site(),
    );

    // The members of this struct.
    struct Member {
        name: Ident,
        dummy: bool,
        ty: TokenStream,
        signature: Cow<'static, str>,
    }
    let mut rust_members = Vec::with_capacity(members.len());

    // Padding structs will be named `_paddingN` where `N` is determined by this variable.
    let mut next_padding_num = 0;

    // Contains the offset of the next field.
    // Equals to `None` if there's a runtime-sized field in there.
    let mut current_rust_offset = Some(0);

    for (&member, member_info) in members.iter().zip(id_info.iter_members()) {
        // Compute infos about the member.
        let (ty, signature, rust_size, rust_align) =
            type_from_id(shader, spirv, member, types_meta);
        let member_name = member_info
            .iter_name()
            .find_map(|instruction| match instruction {
                Instruction::MemberName { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .unwrap_or("__unnamed");

        // Ignore the whole struct is a member is built in, which includes
        // `gl_Position` for example.
        if member_info.iter_decoration().any(|instruction| {
            matches!(
                instruction,
                Instruction::MemberDecorate {
                    decoration: Decoration::BuiltIn { .. },
                    ..
                }
            )
        }) {
            return (quote! {}, None); // TODO: is this correct? shouldn't it return a correct struct but with a flag or something?
        }

        // Finding offset of the current member, as requested by the SPIR-V code.
        let spirv_offset =
            member_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::MemberDecorate {
                        decoration: Decoration::Offset { byte_offset },
                        ..
                    } => Some(*byte_offset),
                    _ => None,
                });

        // Some structs don't have `Offset` decorations, in the case they are used as local
        // variables only. Ignoring these.
        let spirv_offset = match spirv_offset {
            Some(o) => o as usize,
            None => return (quote! {}, None), // TODO: shouldn't we return and let the caller ignore it instead?
        };

        // We need to add a dummy field if necessary.
        {
            let current_rust_offset = current_rust_offset
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
                    dummy: true,
                    ty: quote! { [u8; #diff] },
                    signature: Cow::from(format!("[u8; {}]", diff)),
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
            dummy: false,
            ty,
            signature,
        });
    }

    // Try determine the total size of the struct in order to add padding at the end of the struct.
    let mut spirv_req_total_size = None;
    for inst in spirv.iter_global() {
        match inst {
            Instruction::TypeArray {
                result_id,
                element_type,
                ..
            }
            | Instruction::TypeRuntimeArray {
                result_id,
                element_type,
            } if *element_type == struct_id => {
                spirv_req_total_size =
                    spirv
                        .id(*result_id)
                        .iter_decoration()
                        .find_map(|instruction| match instruction {
                            Instruction::Decorate {
                                decoration: Decoration::ArrayStride { array_stride },
                                ..
                            } => Some(*array_stride),
                            _ => None,
                        })
            }
            _ => (),
        }
    }

    // Adding the final padding members.
    if let (Some(cur_size), Some(req_size)) = (current_rust_offset, spirv_req_total_size) {
        let diff = req_size.checked_sub(cur_size as u32).unwrap();
        if diff >= 1 {
            rust_members.push(Member {
                name: Ident::new(&format!("_dummy{}", next_padding_num), Span::call_site()),
                dummy: true,
                ty: quote! { [u8; #diff as usize] },
                signature: Cow::from(format!("[u8; {}]", diff)),
            });
        }
    }

    let total_size = spirv_req_total_size
        .map(|sz| sz as usize)
        .or(current_rust_offset);

    // For single shader-mode registration mechanism skipped.
    if let Some(types_registry) = types_registry {
        let target_type = RegisteredType {
            shader: shader.to_string(),
            signature: rust_members
                .iter()
                .map(|member| (member.name.to_string(), member.signature.clone()))
                .collect(),
        };

        let name = name.to_string();

        // Checking with Registry if this struct already registered by another shader, and if their
        // signatures match.
        if let Some(registered) = types_registry.get(name.as_str()) {
            registered.assert_signatures(name.as_str(), &target_type);

            // If the struct already registered and matches this one, skip duplicate.
            return (quote! {}, total_size);
        }

        assert!(types_registry.insert(name, target_type).is_none());
    }

    // We can only implement Clone if there's no unsized member in the struct.
    let (clone_impl, copy_derive) =
        if current_rust_offset.is_some() && (types_meta.clone || types_meta.copy) {
            (
                if types_meta.clone {
                    let mut copies = vec![];
                    for member in &rust_members {
                        let name = &member.name;
                        copies.push(quote! { #name: self.#name, });
                    }

                    // Clone is implemented manually because members can be large arrays
                    // that do not implement Clone, but do implement Copy
                    quote! {
                        impl Clone for #name {
                            fn clone(&self) -> Self {
                                #name {
                                    #( #copies )*
                                }
                            }
                        }
                    }
                } else {
                    quote! {}
                },
                if types_meta.copy {
                    quote! { #[derive(Copy)] }
                } else {
                    quote! {}
                },
            )
        } else {
            (quote! {}, quote! {})
        };

    let partial_eq_impl = if current_rust_offset.is_some() && types_meta.partial_eq {
        let mut fields = vec![];
        for member in &rust_members {
            if !member.dummy {
                let name = &member.name;
                fields.push(quote! {
                    if self.#name != other.#name {
                        return false
                    }
                });
            }
        }

        quote! {
            impl PartialEq for #name {
                fn eq(&self, other: &Self) -> bool {
                    #( #fields )*
                    true
                }
            }
        }
    } else {
        quote! {}
    };

    let (debug_impl, display_impl) = if current_rust_offset.is_some()
        && (types_meta.debug || types_meta.display)
    {
        let mut fields = vec![];
        for member in &rust_members {
            if !member.dummy {
                let name = &member.name;
                let name_string = LitStr::new(name.to_string().as_ref(), name.span());

                fields.push(quote! {.field(#name_string, &self.#name)});
            }
        }

        let name_string = LitStr::new(name.to_string().as_ref(), name.span());

        (
            if types_meta.debug {
                quote! {
                    impl std::fmt::Debug for #name {
                        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                            formatter
                                .debug_struct(#name_string)
                                #( #fields )*
                                .finish()
                        }
                    }
                }
            } else {
                quote! {}
            },
            if types_meta.display {
                quote! {
                    impl std::fmt::Display for #name {
                        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                            formatter
                                .debug_struct(#name_string)
                                #( #fields )*
                                .finish()
                        }
                    }
                }
            } else {
                quote! {}
            },
        )
    } else {
        (quote! {}, quote! {})
    };

    let default_impl = if current_rust_offset.is_some() && types_meta.default {
        quote! {
            impl Default for #name {
                fn default() -> Self {
                    unsafe {
                        std::mem::MaybeUninit::<Self>::zeroed().assume_init()
                    }
                }
            }
        }
    } else {
        quote! {}
    };

    // If the struct has unsized members none of custom impls applied.
    let custom_impls = if current_rust_offset.is_some() {
        let impls = &types_meta.impls;

        quote! {
            #( #impls for #name {} )*
        }
    } else {
        quote! {}
    };

    // If the struct has unsized members none of custom derives applied.
    let custom_derives = if current_rust_offset.is_some() && !types_meta.custom_derives.is_empty() {
        let derive_list = &types_meta.custom_derives;
        quote! { #[derive(#( #derive_list ),*)] }
    } else {
        quote! {}
    };

    let mut members = vec![];
    for member in &rust_members {
        let name = &member.name;
        let ty = &member.ty;
        members.push(quote!(pub #name: #ty,));
    }

    let ast = quote! {
        #[repr(C)]
        #copy_derive
        #custom_derives
        #[allow(non_snake_case)]
        pub struct #name {
            #( #members )*
        }
        #clone_impl
        #partial_eq_impl
        #debug_impl
        #display_impl
        #default_impl
        #custom_impls
    };

    (ast, total_size)
}

/// Returns the type name to put in the Rust struct, and its size and alignment.
///
/// The size can be `None` if it's only known at runtime.
pub(super) fn type_from_id(
    shader: &str,
    spirv: &Spirv,
    searched: Id,
    types_meta: &TypesMeta,
) -> (TokenStream, Cow<'static, str>, Option<usize>, usize) {
    let id_info = spirv.id(searched);

    match id_info.instruction() {
        Instruction::TypeBool { .. } => {
            panic!("Can't put booleans in structs")
        }
        Instruction::TypeInt {
            width, signedness, ..
        } => match (width, signedness) {
            (8, 1) => {
                #[repr(C)]
                struct Foo {
                    data: i8,
                    after: u8,
                }
                return (
                    quote! {i8},
                    Cow::from("i8"),
                    Some(std::mem::size_of::<i8>()),
                    mem::align_of::<Foo>(),
                );
            }
            (8, 0) => {
                #[repr(C)]
                struct Foo {
                    data: u8,
                    after: u8,
                }
                return (
                    quote! {u8},
                    Cow::from("u8"),
                    Some(std::mem::size_of::<u8>()),
                    mem::align_of::<Foo>(),
                );
            }
            (16, 1) => {
                #[repr(C)]
                struct Foo {
                    data: i16,
                    after: u8,
                }
                return (
                    quote! {i16},
                    Cow::from("i16"),
                    Some(std::mem::size_of::<i16>()),
                    mem::align_of::<Foo>(),
                );
            }
            (16, 0) => {
                #[repr(C)]
                struct Foo {
                    data: u16,
                    after: u8,
                }
                return (
                    quote! {u16},
                    Cow::from("u16"),
                    Some(std::mem::size_of::<u16>()),
                    mem::align_of::<Foo>(),
                );
            }
            (32, 1) => {
                #[repr(C)]
                struct Foo {
                    data: i32,
                    after: u8,
                }
                return (
                    quote! {i32},
                    Cow::from("i32"),
                    Some(std::mem::size_of::<i32>()),
                    mem::align_of::<Foo>(),
                );
            }
            (32, 0) => {
                #[repr(C)]
                struct Foo {
                    data: u32,
                    after: u8,
                }
                return (
                    quote! {u32},
                    Cow::from("u32"),
                    Some(std::mem::size_of::<u32>()),
                    mem::align_of::<Foo>(),
                );
            }
            (64, 1) => {
                #[repr(C)]
                struct Foo {
                    data: i64,
                    after: u8,
                }
                return (
                    quote! {i64},
                    Cow::from("i64"),
                    Some(std::mem::size_of::<i64>()),
                    mem::align_of::<Foo>(),
                );
            }
            (64, 0) => {
                #[repr(C)]
                struct Foo {
                    data: u64,
                    after: u8,
                }
                return (
                    quote! {u64},
                    Cow::from("u64"),
                    Some(std::mem::size_of::<u64>()),
                    mem::align_of::<Foo>(),
                );
            }
            _ => panic!("No Rust equivalent for an integer of width {}", width),
        },
        Instruction::TypeFloat { width, .. } => match width {
            32 => {
                #[repr(C)]
                struct Foo {
                    data: f32,
                    after: u8,
                }
                return (
                    quote! {f32},
                    Cow::from("f32"),
                    Some(std::mem::size_of::<f32>()),
                    mem::align_of::<Foo>(),
                );
            }
            64 => {
                #[repr(C)]
                struct Foo {
                    data: f64,
                    after: u8,
                }
                return (
                    quote! {f64},
                    Cow::from("f64"),
                    Some(std::mem::size_of::<f64>()),
                    mem::align_of::<Foo>(),
                );
            }
            _ => panic!("No Rust equivalent for a floating-point of width {}", width),
        },
        &Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
            let (ty, item, t_size, t_align) =
                type_from_id(shader, spirv, component_type, types_meta);
            let array_length = component_count as usize;
            let size = t_size.map(|s| s * component_count as usize);
            return (
                quote! { [#ty; #array_length] },
                Cow::from(format!("[{}; {}]", item, array_length)),
                size,
                t_align,
            );
        }
        &Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            // FIXME: row-major or column-major
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
            let (ty, item, t_size, t_align) = type_from_id(shader, spirv, column_type, types_meta);
            let array_length = column_count as usize;
            let size = t_size.map(|s| s * column_count as usize);
            return (
                quote! { [#ty; #array_length] },
                Cow::from(format!("[{}; {}]", item, array_length)),
                size,
                t_align,
            );
        }
        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
            let (ty, item, t_size, t_align) = type_from_id(shader, spirv, element_type, types_meta);
            let t_size = t_size.expect("array components must be sized");
            let len = match spirv.id(length).instruction() {
                &Instruction::Constant { ref value, .. } => value,
                _ => panic!("failed to find array length"),
            };
            let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
            let stride = id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::ArrayStride { array_stride },
                        ..
                    } => Some(*array_stride),
                    _ => None,
                })
                .unwrap();
            if stride as usize > t_size {
                panic!("Not possible to generate a rust array with the correct alignment since the SPIR-V \
                            ArrayStride is larger than the size of the array element in rust. Try wrapping \
                            the array element in a struct or rounding up the size of a vector or matrix \
                            (e.g. increase a vec3 to a vec4)")
            }
            let array_length = len as usize;
            let size = Some(t_size * len as usize);
            return (
                quote! { [#ty; #array_length] },
                Cow::from(format!("[{}; {}]", item, array_length)),
                size,
                t_align,
            );
        }
        &Instruction::TypeRuntimeArray { element_type, .. } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
            let (ty, name, _, t_align) = type_from_id(shader, spirv, element_type, types_meta);
            return (
                quote! { [#ty] },
                Cow::from(format!("[{}]", name)),
                None,
                t_align,
            );
        }
        Instruction::TypeStruct { member_types, .. } => {
            // TODO: take the Offset member decorate into account?
            let name_string = id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .unwrap_or("__unnamed");
            let name = Ident::new(&name_string, Span::call_site());
            let ty = quote! { #name };
            let (_, size) = write_struct(shader, spirv, searched, member_types, types_meta, None);
            let align = member_types
                .iter()
                .map(|&t| type_from_id(shader, spirv, t, types_meta).3)
                .max()
                .unwrap_or(1);
            return (ty, Cow::from(name_string.to_owned()), size, align);
        }
        _ => panic!("Type #{} not found", searched),
    }
}

/// Writes the `SpecializationConstants` struct that contains the specialization constants and
/// implements the `Default` and the `vulkano::shader::SpecializationConstants` traits.
pub(super) fn write_specialization_constants<'a>(
    shader: &'a str,
    spirv: &Spirv,
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

    for instruction in spirv.iter_global() {
        let (result_type_id, result_id, default_value) = match instruction {
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

        // Translate bool to u32
        let (rust_ty, rust_signature, rust_size, rust_alignment) =
            match spirv.id(result_type_id).instruction() {
                Instruction::TypeBool { .. } => (
                    quote! {u32},
                    Cow::from("u32"),
                    Some(mem::size_of::<u32>()),
                    mem::align_of::<u32>(),
                ),
                _ => type_from_id(shader, spirv, result_type_id, types_meta),
            };
        let rust_size = rust_size.expect("Found runtime-sized specialization constant");

        let id_info = spirv.id(result_id);

        let constant_id = id_info
            .iter_decoration()
            .find_map(|instruction| match instruction {
                Instruction::Decorate {
                    decoration:
                        Decoration::SpecId {
                            specialization_constant_id,
                        },
                    ..
                } => Some(*specialization_constant_id),
                _ => None,
            });

        if let Some(constant_id) = constant_id {
            let name = match id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(name.as_str()),
                    _ => None,
                }) {
                Some(name) => name.to_owned(),
                None => format!("constant_{}", constant_id),
            };

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

    let struct_name = if shared_constants {
        format_ident!("SpecializationConstants")
    } else {
        format_ident!("{}SpecializationConstants", shader.to_upper_camel_case())
    };

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

        assert!(types_registry.insert(name, target_type).is_none());
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
