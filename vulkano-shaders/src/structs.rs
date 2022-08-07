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
use std::{borrow::Cow, collections::HashMap, mem};
use syn::{Ident, LitStr};
use vulkano::shader::spirv::{Decoration, Id, Instruction, Spirv};

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub(super) fn write_structs<'a>(
    shader: &'a str,
    spirv: &'a Spirv,
    types_meta: &'a TypesMeta,
    types_registry: &'a mut HashMap<String, RegisteredType>,
) -> TokenStream {
    spirv
        .iter_global()
        .filter_map(|instruction| match instruction {
            &Instruction::TypeStruct {
                result_id,
                ref member_types,
            } => Some((result_id, member_types)),
            _ => None,
        })
        .filter(|&(struct_id, _member_types)| has_defined_layout(spirv, struct_id))
        .filter_map(|(struct_id, member_types)| {
            let (rust_members, is_sized) =
                write_struct_members(shader, spirv, struct_id, member_types);

            let struct_name = spirv
                .id(struct_id)
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .unwrap_or("__unnamed");

            // Register the type if needed
            if !register_struct(types_registry, shader, &rust_members, struct_name) {
                return None;
            }

            let struct_ident = format_ident!("{}", struct_name);
            let members = rust_members
                .iter()
                .map(|Member { name, ty, .. }| quote!(pub #name: #ty,));

            let struct_body = quote! {
                #[repr(C)]
                #[allow(non_snake_case)]
                pub struct #struct_ident {
                    #( #members )*
                }
            };

            Some(if is_sized {
                let derives = write_derives(types_meta);
                let impls = write_impls(types_meta, &struct_name, &rust_members);
                quote! {
                    #derives
                    #struct_body
                    #(#impls)*
                }
            } else {
                struct_body
            })
        })
        .collect()
}

// The members of this struct.
struct Member {
    name: Ident,
    is_dummy: bool,
    ty: TokenStream,
    signature: Cow<'static, str>,
}

fn write_struct_members<'a>(
    shader: &'a str,
    spirv: &Spirv,
    struct_id: Id,
    members: &[Id],
) -> (Vec<Member>, bool) {
    let mut rust_members = Vec::with_capacity(members.len());

    // Dummy members will be named `_dummyN` where `N` is determined by this variable.
    let mut next_dummy_num = 0;

    // Contains the offset of the next field.
    // Equals to `None` if there's a runtime-sized field in there.
    let mut current_rust_offset = Some(0);

    for (member_index, (&member, member_info)) in members
        .iter()
        .zip(spirv.id(struct_id).iter_members())
        .enumerate()
    {
        // Compute infos about the member.
        let (ty, signature, rust_size, rust_align) = type_from_id(shader, spirv, member);
        let member_name = member_info
            .iter_name()
            .find_map(|instruction| match instruction {
                Instruction::MemberName { name, .. } => Some(Cow::from(name.as_str())),
                _ => None,
            })
            .unwrap_or_else(|| Cow::from(format!("__unnamed{}", member_index)));

        // Finding offset of the current member, as requested by the SPIR-V code.
        let spirv_offset = member_info
            .iter_decoration()
            .find_map(|instruction| match instruction {
                Instruction::MemberDecorate {
                    decoration: Decoration::Offset { byte_offset },
                    ..
                } => Some(*byte_offset as usize),
                _ => None,
            })
            .unwrap();

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
                rust_members.push(Member {
                    name: format_ident!("_dummy{}", next_dummy_num.to_string()),
                    is_dummy: true,
                    ty: quote! { [u8; #diff] },
                    signature: Cow::from(format!("[u8; {}]", diff)),
                });
                next_dummy_num += 1;
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
            is_dummy: false,
            ty,
            signature,
        });
    }

    // Adding the final padding members, if the struct is sized.
    if let Some(cur_size) = current_rust_offset {
        // Try to determine the total size of the struct.
        if let Some(req_size) = struct_size_from_array_stride(spirv, struct_id) {
            let diff = req_size.checked_sub(cur_size as u32).unwrap();

            if diff >= 1 {
                rust_members.push(Member {
                    name: Ident::new(&format!("_dummy{}", next_dummy_num), Span::call_site()),
                    is_dummy: true,
                    ty: quote! { [u8; #diff as usize] },
                    signature: Cow::from(format!("[u8; {}]", diff)),
                });
            }
        }
    }

    (rust_members, current_rust_offset.is_some())
}

fn register_struct(
    types_registry: &mut HashMap<String, RegisteredType>,
    shader: &str,
    rust_members: &[Member],
    struct_name: &str,
) -> bool {
    let target_type = RegisteredType {
        shader: shader.to_string(),
        signature: rust_members
            .iter()
            .map(|member| (member.name.to_string(), member.signature.clone()))
            .collect(),
    };

    // Checking with Registry if this struct already registered by another shader, and if their
    // signatures match.
    if let Some(registered) = types_registry.get(struct_name) {
        registered.assert_signatures(struct_name, &target_type);

        // If the struct already registered and matches this one, skip duplicate.
        false
    } else {
        assert!(types_registry
            .insert(struct_name.to_owned(), target_type)
            .is_none());
        true
    }
}

fn write_derives(types_meta: &TypesMeta) -> TokenStream {
    let mut derives = vec![];

    if types_meta.clone {
        derives.push(quote! { Clone });
    }

    if types_meta.copy {
        derives.push(quote! { Copy });
    }

    derives.extend(
        types_meta
            .custom_derives
            .iter()
            .map(|derive| quote! { #derive }),
    );

    if !derives.is_empty() {
        quote! {
            #[derive(#(#derives),*)]
        }
    } else {
        quote! {}
    }
}

fn write_impls<'a>(
    types_meta: &'a TypesMeta,
    struct_name: &'a str,
    rust_members: &'a [Member],
) -> impl Iterator<Item = TokenStream> + 'a {
    let struct_ident = format_ident!("{}", struct_name);

    (types_meta.partial_eq.then(|| {
        let fields = rust_members
            .iter()
            .filter(|Member { is_dummy, .. }| !is_dummy)
            .map(|Member { name, .. }| {
                quote! {
                    if self.#name != other.#name {
                        return false
                    }
                }
            });

        quote! {
            impl PartialEq for #struct_ident {
                fn eq(&self, other: &Self) -> bool {
                    #( #fields )*
                    true
                }
            }
        }
    }).into_iter())
    .chain(types_meta.debug.then(|| {
        let fields = rust_members
            .iter()
            .filter(|Member { is_dummy, .. }| !is_dummy)
            .map(|Member { name, .. }| {
                let name_string = LitStr::new(name.to_string().as_ref(), name.span());
                quote! { .field(#name_string, &self.#name) }
            });

        quote! {
            impl std::fmt::Debug for #struct_ident {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                    formatter
                        .debug_struct(#struct_name)
                        #( #fields )*
                        .finish()
                }
            }
        }
    }))
    .chain(types_meta.display.then(|| {
        let fields = rust_members
            .iter()
            .filter(|Member { is_dummy, .. }| !is_dummy)
            .map(|Member { name, .. }| {
                let name_string = LitStr::new(name.to_string().as_ref(), name.span());
                quote! { .field(#name_string, &self.#name) }
            });

        quote! {
            impl std::fmt::Display for #struct_ident {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                    formatter
                        .debug_struct(#struct_name)
                        #( #fields )*
                        .finish()
                }
            }
        }
    }))
    .chain(types_meta.default.then(|| {
        quote! {
            impl Default for #struct_ident {
                fn default() -> Self {
                    unsafe {
                        std::mem::MaybeUninit::<Self>::zeroed().assume_init()
                    }
                }
            }
        }
    }))
    .chain(types_meta.impls.iter().map(move |i| quote!{ #i for #struct_ident {} }))
}

fn has_defined_layout(spirv: &Spirv, struct_id: Id) -> bool {
    for member_info in spirv.id(struct_id).iter_members() {
        let mut offset_found = false;

        for instruction in member_info.iter_decoration() {
            match instruction {
                Instruction::MemberDecorate {
                    decoration: Decoration::BuiltIn { .. },
                    ..
                } => {
                    // Ignore the whole struct if a member is built in, which includes
                    // `gl_Position` for example.
                    return false;
                }
                Instruction::MemberDecorate {
                    decoration: Decoration::Offset { .. },
                    ..
                } => {
                    offset_found = true;
                }
                _ => (),
            }
        }

        // Some structs don't have `Offset` decorations, in the case they are used as local
        // variables only. Ignoring these.
        if !offset_found {
            return false;
        }
    }

    true
}

fn struct_size_from_array_stride(spirv: &Spirv, type_id: Id) -> Option<u32> {
    let mut iter = spirv.iter_global().filter_map(|inst| match inst {
        Instruction::TypeArray {
            result_id,
            element_type,
            ..
        }
        | Instruction::TypeRuntimeArray {
            result_id,
            element_type,
        } if *element_type == type_id => {
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
        _ => None,
    });
    iter.next().map(|first_stride| {
        // Ensure that all strides we find match the first one.
        debug_assert!(iter.all(|array_stride| array_stride == first_stride));
        first_stride
    })
}

/// Returns the type name to put in the Rust struct, and its size and alignment.
///
/// The size can be `None` if it's only known at runtime.
pub(super) fn type_from_id(
    shader: &str,
    spirv: &Spirv,
    type_id: Id,
) -> (TokenStream, Cow<'static, str>, Option<usize>, usize) {
    let id_info = spirv.id(type_id);

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
            let (ty, item, t_size, t_align) = type_from_id(shader, spirv, component_type);
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
            let (ty, item, t_size, t_align) = type_from_id(shader, spirv, column_type);
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

            let (element_type, element_type_string, element_size, element_align) =
                type_from_id(shader, spirv, element_type);

            let element_size = element_size.expect("array components must be sized");
            let array_length = match spirv.id(length).instruction() {
                &Instruction::Constant { ref value, .. } => {
                    value.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64)
                }
                _ => panic!("failed to find array length"),
            } as usize;

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
            if stride as usize > element_size {
                panic!("Not possible to generate a rust array with the correct alignment since the SPIR-V \
                            ArrayStride is larger than the size of the array element in rust. Try wrapping \
                            the array element in a struct or rounding up the size of a vector or matrix \
                            (e.g. increase a vec3 to a vec4)")
            }

            return (
                quote! { [#element_type; #array_length] },
                Cow::from(format!("[{}; {}]", element_type_string, array_length)),
                Some(element_size * array_length as usize),
                element_align,
            );
        }
        &Instruction::TypeRuntimeArray { element_type, .. } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());

            let (element_type, element_type_string, _, element_align) =
                type_from_id(shader, spirv, element_type);

            return (
                quote! { [#element_type] },
                Cow::from(format!("[{}]", element_type_string)),
                None,
                element_align,
            );
        }
        Instruction::TypeStruct { member_types, .. } => {
            // TODO: take the Offset member decorate into account?
            let size = if !has_defined_layout(spirv, type_id) {
                None
            } else {
                // If the struct appears in an array, then first try to get the size from the
                // array stride.
                struct_size_from_array_stride(spirv, type_id)
                    .map(|size| size as usize)
                    .or_else(|| {
                        // We haven't found any strides, so we have to calculate the size based
                        // on the offset and size of the last member.
                        member_types
                            .iter()
                            .zip(spirv.id(type_id).iter_members())
                            .last()
                            .map_or(Some(0), |(&member, member_info)| {
                                let spirv_offset = member_info
                                    .iter_decoration()
                                    .find_map(|instruction| match instruction {
                                        Instruction::MemberDecorate {
                                            decoration: Decoration::Offset { byte_offset },
                                            ..
                                        } => Some(*byte_offset as usize),
                                        _ => None,
                                    })
                                    .unwrap();
                                let (_, _, rust_size, _) = type_from_id(shader, spirv, member);
                                rust_size.map(|rust_size| spirv_offset + rust_size)
                            })
                    })
            };

            let align = member_types
                .iter()
                .map(|&t| type_from_id(shader, spirv, t).3)
                .max()
                .unwrap_or(1);

            let name_string = id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(Cow::from(name.clone())),
                    _ => None,
                })
                .unwrap_or(Cow::from("__unnamed"));
            let name = {
                let name = format_ident!("{}", name_string);
                quote! { #name }
            };

            return (name, name_string, size, align);
        }
        _ => panic!("Type #{} not found", type_id),
    }
}

/// Writes the `SpecializationConstants` struct that contains the specialization constants and
/// implements the `Default` and the `vulkano::shader::SpecializationConstants` traits.
pub(super) fn write_specialization_constants<'a>(
    shader: &'a str,
    spirv: &Spirv,
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
                _ => type_from_id(shader, spirv, result_type_id),
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
                ::vulkano::shader::SpecializationMapEntry {
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

        unsafe impl ::vulkano::shader::SpecializationConstants for #struct_name {
            fn descriptors() -> &'static [::vulkano::shader::SpecializationMapEntry] {
                static DESCRIPTORS: [::vulkano::shader::SpecializationMapEntry; #num_map_entries] = [
                    #( #map_entries ),*
                ];
                &DESCRIPTORS
            }
        }
    }
}
