// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{LinAlgType, RegisteredType, TypesMeta};
use ahash::HashMap;
use heck::ToUpperCamelCase;
use proc_macro2::{Span, TokenStream};
use std::{borrow::Cow, mem};
use syn::{Ident, LitStr};
use vulkano::shader::spirv::{Decoration, Id, Instruction, Spirv};

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub(super) fn write_structs<'a, L: LinAlgType>(
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
                write_struct_members::<L>(spirv, struct_id, member_types);

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
                let impls = write_impls(types_meta, struct_name, &rust_members);
                let bounds = rust_members
                    .iter()
                    .map(|Member { ty, .. }| quote!(#ty: ::vulkano::bytemuck::AnyBitPattern));

                quote! {
                    #derives
                    #struct_body
                    #(#impls)*

                    // SAFETY: All that's required for deriving `AnyBitPattern` is that all the
                    // fields are `AnyBitPattern`, which we enforce with the bounds.
                    #[allow(unsafe_code)]
                    unsafe impl ::vulkano::bytemuck::AnyBitPattern for #struct_ident
                    where
                        #(#bounds,)*
                    {
                    }

                    // SAFETY: `AnyBitPattern` implies `Zeroable`.
                    #[allow(unsafe_code)]
                    unsafe impl ::vulkano::bytemuck::Zeroable for #struct_ident
                    where
                        Self: ::vulkano::bytemuck::AnyBitPattern
                    {
                    }
                }
            } else {
                quote! {
                    #[derive(::vulkano::buffer::BufferContents)]
                    #struct_body
                }
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

fn write_struct_members<L: LinAlgType>(
    spirv: &Spirv,
    struct_id: Id,
    members: &[Id],
) -> (Vec<Member>, bool) {
    let mut rust_members = Vec::with_capacity(members.len());

    let mut is_sized = true;

    for (member_index, (&member, member_info)) in members
        .iter()
        .zip(spirv.id(struct_id).iter_members())
        .enumerate()
    {
        // Compute infos about the member.
        let (ty, signature, rust_size, _) = type_from_id::<L>(spirv, member);
        let member_name = member_info
            .iter_name()
            .find_map(|instruction| match instruction {
                Instruction::MemberName { name, .. } => Some(Cow::from(name.as_str())),
                _ => None,
            })
            .unwrap_or_else(|| Cow::from(format!("__unnamed{}", member_index)));

        is_sized = rust_size.is_some();

        rust_members.push(Member {
            name: Ident::new(&member_name, Span::call_site()),
            is_dummy: false,
            ty,
            signature,
        });
    }

    (rust_members, is_sized)
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

    (types_meta
        .partial_eq
        .then(|| {
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
                impl ::std::cmp::PartialEq for #struct_ident {
                    fn eq(&self, other: &Self) -> bool {
                        #( #fields )*
                        true
                    }
                }
            }
        })
        .into_iter())
    .chain(types_meta.debug.then(|| {
        let fields = rust_members
            .iter()
            .filter(|Member { is_dummy, .. }| !is_dummy)
            .map(|Member { name, .. }| {
                let name_string = LitStr::new(name.to_string().as_ref(), name.span());
                quote! { .field(#name_string, &self.#name) }
            });

        quote! {
            impl ::std::fmt::Debug for #struct_ident {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::result::Result<(), ::std::fmt::Error> {
                    f.debug_struct(#struct_name)
                        #( #fields )*
                        .finish()
                }
            }
        }
    }))
    .chain(types_meta.display.then(|| {
        quote! {
            impl ::std::fmt::Display for #struct_ident {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::result::Result<(), ::std::fmt::Error> {
                    ::std::fmt::Debug::fmt(self, f)
                }
            }
        }
    }))
    .chain(types_meta.default.then(|| {
        quote! {
            #[allow(unsafe_code)]
            impl ::std::default::Default for #struct_ident {
                fn default() -> Self {
                    unsafe {
                        ::std::mem::MaybeUninit::<Self>::zeroed().assume_init()
                    }
                }
            }
        }
    }))
    .chain(
        types_meta
            .impls
            .iter()
            .map(move |i| quote! { #i for #struct_ident {} }),
    )
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
pub(super) fn type_from_id<L: LinAlgType>(
    spirv: &Spirv,
    type_id: Id,
) -> (TokenStream, Cow<'static, str>, Option<usize>, usize) {
    let id_info = spirv.id(type_id);

    match id_info.instruction() {
        Instruction::TypeBool { .. } => {
            panic!("can't put booleans in structs")
        }
        Instruction::TypeInt {
            width, signedness, ..
        } => match (width, signedness) {
            (8, 1) => (
                quote! {i8},
                Cow::from("i8"),
                Some(mem::size_of::<i8>()),
                mem::align_of::<i8>(),
            ),
            (8, 0) => (
                quote! {u8},
                Cow::from("u8"),
                Some(mem::size_of::<u8>()),
                mem::align_of::<u8>(),
            ),
            (16, 1) => (
                quote! {i16},
                Cow::from("i16"),
                Some(mem::size_of::<i16>()),
                mem::align_of::<i16>(),
            ),
            (16, 0) => (
                quote! {u16},
                Cow::from("u16"),
                Some(mem::size_of::<u16>()),
                mem::align_of::<u16>(),
            ),
            (32, 1) => (
                quote! {i32},
                Cow::from("i32"),
                Some(mem::size_of::<i32>()),
                mem::align_of::<i32>(),
            ),
            (32, 0) => (
                quote! {u32},
                Cow::from("u32"),
                Some(mem::size_of::<u32>()),
                mem::align_of::<u32>(),
            ),
            (64, 1) => (
                quote! {i64},
                Cow::from("i64"),
                Some(mem::size_of::<i64>()),
                mem::align_of::<i64>(),
            ),
            (64, 0) => (
                quote! {u64},
                Cow::from("u64"),
                Some(mem::size_of::<u64>()),
                mem::align_of::<u64>(),
            ),
            _ => panic!("no Rust equivalent for an integer of width {}", width),
        },
        Instruction::TypeFloat { width, .. } => match width {
            32 => (
                quote! {f32},
                Cow::from("f32"),
                Some(mem::size_of::<f32>()),
                mem::align_of::<f32>(),
            ),
            64 => (
                quote! {f64},
                Cow::from("f64"),
                Some(mem::size_of::<f64>()),
                mem::align_of::<f64>(),
            ),
            _ => panic!("no Rust equivalent for a floating-point of width {}", width),
        },
        &Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
            let component_count = component_count as usize;
            let (element_ty, element_item, element_size, align) =
                type_from_id::<L>(spirv, component_type);

            let ty = L::vector(&element_ty, component_count);
            let item = Cow::from(format!("[{}; {}]", element_item, component_count));
            let size = element_size.map(|s| s * component_count);

            (ty, item, size, align)
        }
        &Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());
            let column_count = column_count as usize;

            // FIXME: row-major or column-major
            let (row_count, element, element_item, element_size, align) =
                match spirv.id(column_type).instruction() {
                    &Instruction::TypeVector {
                        component_type,
                        component_count,
                        ..
                    } => {
                        let (element, element_item, element_size, align) =
                            type_from_id::<L>(spirv, component_type);
                        (
                            component_count as usize,
                            element,
                            element_item,
                            element_size,
                            align,
                        )
                    }
                    _ => unreachable!(),
                };

            let ty = L::matrix(&element, row_count, column_count);
            let size = element_size.map(|s| s * row_count * column_count);
            let item = Cow::from(format!(
                "[[{}; {}]; {}]",
                element_item, row_count, column_count,
            ));

            (ty, item, size, align)
        }
        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());

            let (element_type, element_type_string, element_size, element_align) =
                type_from_id::<L>(spirv, element_type);

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
                panic!(
                    "not possible to generate a Rust array with the correct alignment since the \
                    SPIR-V ArrayStride is larger than the size of the array element in Rust. Try \
                    wrapping the array element in a struct or rounding up the size of a vector or \
                    matrix (e.g. increase a vec3 to a vec4)",
                );
            }

            (
                quote! { [#element_type; #array_length] },
                Cow::from(format!("[{}; {}]", element_type_string, array_length)),
                Some(element_size * array_length),
                element_align,
            )
        }
        &Instruction::TypeRuntimeArray { element_type, .. } => {
            debug_assert_eq!(mem::align_of::<[u32; 3]>(), mem::align_of::<u32>());

            let (element_type, element_type_string, _, element_align) =
                type_from_id::<L>(spirv, element_type);

            (
                quote! { [#element_type] },
                Cow::from(format!("[{}]", element_type_string)),
                None,
                element_align,
            )
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
                                let (_, _, rust_size, _) = type_from_id::<L>(spirv, member);
                                rust_size.map(|rust_size| spirv_offset + rust_size)
                            })
                    })
            };

            let align = member_types
                .iter()
                .map(|&t| type_from_id::<L>(spirv, t).3)
                .max()
                .unwrap_or(1);

            let name_string = id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(Cow::from(name.clone())),
                    _ => None,
                })
                .unwrap_or_else(|| Cow::from("__unnamed"));
            let name = {
                let name = format_ident!("{}", name_string);
                quote! { #name }
            };

            (name, name_string, size, align)
        }
        _ => panic!("type #{} not found", type_id),
    }
}

/// Writes the `SpecializationConstants` struct that contains the specialization constants and
/// implements the `Default` and the `vulkano::shader::SpecializationConstants` traits.
pub(super) fn write_specialization_constants<'a, L: LinAlgType>(
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
        let (result_type_id, result_id, default_value) = match *instruction {
            Instruction::SpecConstantTrue {
                result_type_id,
                result_id,
            } => (result_type_id, result_id, quote! {1u32}),

            Instruction::SpecConstantFalse {
                result_type_id,
                result_id,
            } => (result_type_id, result_id, quote! {0u32}),

            Instruction::SpecConstant {
                result_type_id,
                result_id,
                ref value,
            } => {
                let def_val = quote! {
                    unsafe {{ ::std::mem::transmute([ #( #value ),* ]) }}
                };
                (result_type_id, result_id, def_val)
            }

            Instruction::SpecConstantComposite {
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
                _ => type_from_id::<L>(spirv, result_type_id),
            };
        let rust_size = rust_size.expect("found runtime-sized specialization constant");

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
        #[derive(::std::clone::Clone, ::std::marker::Copy, ::std::fmt::Debug)]
        #[allow(non_snake_case)]
        #[repr(C)]
        pub struct #struct_name {
            #( #struct_members ),*
        }

        impl ::std::default::Default for #struct_name {
            #[inline]
            fn default() -> #struct_name {
                #struct_name {
                    #( #struct_member_defaults ),*
                }
            }
        }

        #[allow(unsafe_code)]
        unsafe impl ::vulkano::shader::SpecializationConstants for #struct_name {
            #[inline(always)]
            fn descriptors() -> &'static [::vulkano::shader::SpecializationMapEntry] {
                static DESCRIPTORS: [::vulkano::shader::SpecializationMapEntry; #num_map_entries] = [
                    #( #map_entries ),*
                ];
                &DESCRIPTORS
            }
        }
    }
}
