// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;

use proc_macro2::{Ident, TokenStream};

use crate::enums::{Decoration, Dim, ImageFormat, StorageClass};
use crate::parse::{Instruction, Spirv};
use crate::{spirv_search, TypesMeta};
use std::collections::HashSet;

pub(super) fn write_descriptor_sets(
    doc: &Spirv,
    entry_point_layout_name: &Ident,
    entrypoint_id: u32,
    interface: &[u32],
    types_meta: &TypesMeta,
) -> TokenStream {
    // TODO: somewhat implemented correctly

    // Finding all the descriptors.
    let mut descriptors = Vec::new();
    struct Descriptor {
        set: u32,
        binding: u32,
        desc_ty: TokenStream,
        array_count: u64,
        readonly: bool,
    }

    // For SPIR-V 1.4+, the entrypoint interface can specify variables of all storage classes,
    // and most tools will put all used variables in the entrypoint interface. However,
    // SPIR-V 1.0-1.3 do not specify variables other than Input/Output ones in the interface,
    // and instead the function itself must be inspected.
    let variables = {
        let mut found_variables: HashSet<u32> = interface.iter().cloned().collect();
        let mut inspected_functions: HashSet<u32> = HashSet::new();
        find_variables_in_function(&doc, entrypoint_id, &mut inspected_functions, &mut found_variables);
        found_variables
    };

    // Looping to find all the interface elements that have the `DescriptorSet` decoration.
    for set_decoration in doc.get_decorations(Decoration::DecorationDescriptorSet) {
        let variable_id = set_decoration.target_id;

        if !variables.contains(&variable_id) {
            continue;
        }

        let set = set_decoration.params[0];

        // Find which type is pointed to by this variable.
        let (pointed_ty, storage_class) = pointer_variable_ty(doc, variable_id);
        // Name of the variable.
        let name = spirv_search::name_from_id(doc, variable_id);

        // Find the binding point of this descriptor.
        // TODO: There was a previous todo here, I think it was asking for this to be implemented for member decorations? check git history
        let binding = doc
            .get_decoration_params(variable_id, Decoration::DecorationBinding)
            .unwrap()[0];

        // Find information about the kind of binding for this descriptor.
        let (desc_ty, readonly, array_count) =
            descriptor_infos(doc, pointed_ty, storage_class, false).expect(&format!(
                "Couldn't find relevant type for uniform `{}` (type {}, maybe unimplemented)",
                name, pointed_ty
            ));
        descriptors.push(Descriptor {
            desc_ty,
            set,
            binding,
            array_count,
            readonly,
        });
    }

    // Looping to find all the push constant structs.
    let mut push_constants_size = 0;
    for instruction in doc.instructions.iter() {
        let type_id = match instruction {
            &Instruction::TypePointer {
                type_id,
                storage_class: StorageClass::StorageClassPushConstant,
                ..
            } => type_id,
            _ => continue,
        };

        let (_, size, _) = crate::structs::type_from_id(doc, type_id, types_meta);
        let size = size.expect("Found runtime-sized push constants");
        push_constants_size = cmp::max(push_constants_size, size);
    }

    // Writing the body of the `descriptor` method.
    let descriptor_body = descriptors
        .iter()
        .map(|d| {
            let set = d.set as usize;
            let binding = d.binding as usize;
            let desc_ty = &d.desc_ty;
            let array_count = d.array_count as u32;
            let readonly = d.readonly;
            quote! {
                (#set, #binding) => Some(DescriptorDesc {
                    ty: #desc_ty,
                    array_count: #array_count,
                    stages: self.0.clone(),
                    readonly: #readonly,
                }),
            }
        })
        .collect::<Vec<_>>();

    let num_sets = descriptors.iter().fold(0, |s, d| cmp::max(s, d.set + 1)) as usize;

    // Writing the body of the `num_bindings_in_set` method.
    let num_bindings_in_set_body = (0..num_sets)
        .map(|set| {
            let num = descriptors
                .iter()
                .filter(|d| d.set == set as u32)
                .fold(0, |s, d| cmp::max(s, 1 + d.binding)) as usize;
            quote! { #set => Some(#num), }
        })
        .collect::<Vec<_>>();

    // Writing the body of the `num_push_constants_ranges` method.
    let num_push_constants_ranges_body = if push_constants_size == 0 { 0 } else { 1 } as usize;

    // Writing the body of the `push_constants_range` method.
    let push_constants_range_body = quote!(
        if num != 0 || #push_constants_size == 0 {
            None
        } else {
            Some(PipelineLayoutDescPcRange {
                offset: 0,                   // FIXME: not necessarily true
                size: #push_constants_size,
                stages: ShaderStages::all(), // FIXME: wrong
            })
        }
    );

    quote! {
        #[derive(Debug, Clone)]
        pub struct #entry_point_layout_name(pub ShaderStages);

        #[allow(unsafe_code)]
        unsafe impl PipelineLayoutDesc for #entry_point_layout_name {
            fn num_sets(&self) -> usize {
                #num_sets
            }

            fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                match set {
                    #( #num_bindings_in_set_body )*
                    _ => None
                }
            }

            fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                match (set, binding) {
                    #( #descriptor_body )*
                    _ => None
                }
            }

            fn num_push_constants_ranges(&self) -> usize {
                #num_push_constants_ranges_body
            }

            fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                #push_constants_range_body
            }
        }
    }
}

// Recursively finds every pointer variable used in the execution of a function.
fn find_variables_in_function(
    doc: &Spirv,
    function: u32,
    inspected_functions: &mut HashSet<u32>,
    found_variables: &mut HashSet<u32>,
) {
    inspected_functions.insert(function);
    let mut in_function = false;
    for instruction in &doc.instructions {
        if !in_function {
            match instruction {
                Instruction::Function {
                    result_id,
                    ..
                } if result_id == &function => {
                    in_function = true;
                },
                _ => {},
            }
        } else {
            // We only care about instructions that accept pointers.
            // https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_universal_validation_rules
            match instruction {
                Instruction::Load { pointer, .. }
                | Instruction::Store { pointer, .. } => {
                    found_variables.insert(*pointer);
                },
                Instruction::AccessChain { base_id, .. }
                | Instruction::InBoundsAccessChain { base_id, .. } => {
                    found_variables.insert(*base_id);
                },
                Instruction::FunctionCall { function_id, args, .. } => {
                    args.iter().for_each(|&x| {
                        found_variables.insert(x);
                    });
                    if !inspected_functions.contains(function_id) {
                        find_variables_in_function(doc, *function_id, inspected_functions, found_variables);
                    }
                },
                Instruction::ImageTexelPointer { image, coordinate, sample, .. } => {
                    found_variables.insert(*image);
                    found_variables.insert(*coordinate);
                    found_variables.insert(*sample);
                },
                Instruction::CopyMemory { target_id, source_id, .. } => {
                    found_variables.insert(*target_id);
                    found_variables.insert(*source_id);
                },
                Instruction::CopyObject { operand_id, .. } => {
                    found_variables.insert(*operand_id);
                },
                Instruction::AtomicLoad { pointer, .. }
                | Instruction::AtomicIIncrement { pointer, .. }
                | Instruction::AtomicIDecrement { pointer, .. }
                | Instruction::AtomicFlagTestAndSet { pointer, .. }
                | Instruction::AtomicFlagClear { pointer, .. } => {
                    found_variables.insert(*pointer);
                },
                Instruction::AtomicStore { pointer, value_id, .. }
                | Instruction::AtomicExchange { pointer, value_id, .. }
                | Instruction::AtomicIAdd { pointer, value_id, .. }
                | Instruction::AtomicISub { pointer, value_id, .. }
                | Instruction::AtomicSMin { pointer, value_id, .. }
                | Instruction::AtomicUMin { pointer, value_id, .. }
                | Instruction::AtomicSMax { pointer, value_id, .. }
                | Instruction::AtomicUMax { pointer, value_id, .. }
                | Instruction::AtomicAnd { pointer, value_id, .. }
                | Instruction::AtomicOr { pointer, value_id, .. }
                | Instruction::AtomicXor { pointer, value_id, .. } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value_id);
                },
                Instruction::AtomicCompareExchange { pointer, value_id, comparator_id, .. }
                | Instruction::AtomicCompareExchangeWeak { pointer, value_id, comparator_id, .. } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value_id);
                    found_variables.insert(*comparator_id);
                },
                Instruction::ExtInst { operands, .. } => {
                    // We don't know which extended instructions take pointers,
                    // so we must interpret every operand as a pointer.
                    operands.iter().for_each(|&o| {
                        found_variables.insert(o);
                    });
                },
                Instruction::FunctionEnd => return,
                _ => {},
            }
        }
    }
}

/// Assumes that `variable` is a variable with a `TypePointer` and returns the id of the pointed
/// type and the storage class.
fn pointer_variable_ty(doc: &Spirv, variable: u32) -> (u32, StorageClass) {
    let var_ty = doc
        .instructions
        .iter()
        .filter_map(|i| match i {
            &Instruction::Variable {
                result_type_id,
                result_id,
                ..
            } if result_id == variable => Some(result_type_id),
            _ => None,
        })
        .next()
        .unwrap();

    doc.instructions
        .iter()
        .filter_map(|i| match i {
            &Instruction::TypePointer {
                result_id,
                type_id,
                ref storage_class,
                ..
            } if result_id == var_ty => Some((type_id, storage_class.clone())),
            _ => None,
        })
        .next()
        .unwrap()
}

/// Returns a `DescriptorDescTy` constructor, a bool indicating whether the descriptor is
/// read-only, and the number of array elements.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_infos(
    doc: &Spirv,
    pointed_ty: u32,
    pointer_storage: StorageClass,
    force_combined_image_sampled: bool,
) -> Option<(TokenStream, bool, u64)> {
    doc.instructions
        .iter()
        .filter_map(|i| {
            match i {
                &Instruction::TypeStruct { result_id, .. } if result_id == pointed_ty => {
                    let decoration_block = doc
                        .get_decoration_params(pointed_ty, Decoration::DecorationBlock)
                        .is_some();
                    assert!(
                        decoration_block,
                        "Structs in shader interface are expected to be decorated with Block"
                    );
                    let is_ssbo = pointer_storage == StorageClass::StorageClassStorageBuffer;

                    // Determine whether there's a NonWritable decoration.
                    //let non_writable = false;       // TODO: tricky because the decoration is on struct members

                    let desc = quote! {
                        DescriptorDescTy::Buffer(DescriptorBufferDesc {
                            dynamic: None,
                            storage: #is_ssbo,
                        })
                    };

                    Some((desc, true, 1))
                }
                &Instruction::TypeImage {
                    result_id,
                    ref dim,
                    arrayed,
                    ms,
                    sampled,
                    ref format,
                    ..
                } if result_id == pointed_ty => {
                    let sampled = sampled.expect(
                        "Vulkan requires that variables of type OpTypeImage \
                                              have a Sampled operand of 1 or 2",
                    );

                    let arrayed = match arrayed {
                        true => quote! { DescriptorImageDescArray::Arrayed { max_layers: None } },
                        false => quote! { DescriptorImageDescArray::NonArrayed },
                    };

                    match dim {
                        Dim::DimSubpassData => {
                            // We are an input attachment.
                            assert!(
                                !force_combined_image_sampled,
                                "An OpTypeSampledImage can't point to \
                                                                an OpTypeImage whose dimension is \
                                                                SubpassData"
                            );
                            assert!(
                                if let &ImageFormat::ImageFormatUnknown = format {
                                    true
                                } else {
                                    false
                                },
                                "If Dim is SubpassData, Image Format must be Unknown"
                            );
                            assert!(!sampled, "If Dim is SubpassData, Sampled must be 2");

                            let desc = quote! {
                                DescriptorDescTy::InputAttachment {
                                    multisampled: #ms,
                                    array_layers: #arrayed
                                }
                            };

                            Some((desc, true, 1))
                        }
                        Dim::DimBuffer => {
                            // We are a texel buffer.
                            let not_sampled = !sampled;
                            let desc = quote! {
                                DescriptorDescTy::TexelBuffer {
                                    storage: #not_sampled,
                                    format: None, // TODO: specify format if known
                                }
                            };

                            Some((desc, true, 1))
                        }
                        _ => {
                            // We are a sampled or storage image.
                            let ty = match force_combined_image_sampled {
                                true => quote! { DescriptorDescTy::CombinedImageSampler },
                                false => quote! { DescriptorDescTy::Image },
                            };
                            let dim = match *dim {
                                Dim::Dim1D => {
                                    quote! { DescriptorImageDescDimensions::OneDimensional }
                                }
                                Dim::Dim2D => {
                                    quote! { DescriptorImageDescDimensions::TwoDimensional }
                                }
                                Dim::Dim3D => {
                                    quote! { DescriptorImageDescDimensions::ThreeDimensional }
                                }
                                Dim::DimCube => quote! { DescriptorImageDescDimensions::Cube },
                                Dim::DimRect => panic!("Vulkan doesn't support rectangle textures"),
                                _ => unreachable!(),
                            };

                            let desc = quote! {
                                #ty(DescriptorImageDesc {
                                    sampled: #sampled,
                                    dimensions: #dim,
                                    format: None,       // TODO: specify format if known
                                    multisampled: #ms,
                                    array_layers: #arrayed,
                                })
                            };

                            Some((desc, true, 1))
                        }
                    }
                }

                &Instruction::TypeSampledImage {
                    result_id,
                    image_type_id,
                } if result_id == pointed_ty => {
                    descriptor_infos(doc, image_type_id, pointer_storage.clone(), true)
                }

                &Instruction::TypeSampler { result_id } if result_id == pointed_ty => {
                    let desc = quote! { DescriptorDescTy::Sampler };
                    Some((desc, true, 1))
                }
                &Instruction::TypeArray {
                    result_id,
                    type_id,
                    length_id,
                } if result_id == pointed_ty => {
                    let (desc, readonly, arr) =
                        match descriptor_infos(doc, type_id, pointer_storage.clone(), false) {
                            None => return None,
                            Some(v) => v,
                        };
                    assert_eq!(arr, 1); // TODO: implement?
                    let len = doc
                        .instructions
                        .iter()
                        .filter_map(|e| match e {
                            &Instruction::Constant {
                                result_id,
                                ref data,
                                ..
                            } if result_id == length_id => Some(data.clone()),
                            _ => None,
                        })
                        .next()
                        .expect("failed to find array length");
                    let len = len.iter().rev().fold(0, |a, &b| (a << 32) | b as u64);
                    Some((desc, readonly, len))
                }
                _ => None, // TODO: other types
            }
        })
        .next()
}
