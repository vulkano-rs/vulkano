// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::parse::{Instruction, Spirv};
use crate::{spirv_search, TypesMeta};
use proc_macro2::TokenStream;
use spirv_headers::{Decoration, Dim, ImageFormat, StorageClass};
use std::cmp;
use std::collections::HashSet;

#[derive(Debug)]
struct Descriptor {
    set: u32,
    binding: u32,
    desc_ty: TokenStream,
    array_count: u64,
    readonly: bool,
}

pub(super) fn write_descriptor_set_layout_descs(
    doc: &Spirv,
    entrypoint_id: u32,
    interface: &[u32],
    exact_entrypoint_interface: bool,
    stages: &TokenStream,
) -> TokenStream {
    // TODO: somewhat implemented correctly

    // Finding all the descriptors.
    let descriptors = find_descriptors(doc, entrypoint_id, interface, exact_entrypoint_interface);
    let num_sets = descriptors.iter().map(|d| d.set + 1).max().unwrap_or(0);
    let sets: Vec<_> = (0..num_sets)
        .map(|set_num| {
            let num_bindings = descriptors
                .iter()
                .filter(|d| d.set == set_num)
                .map(|d| d.binding + 1)
                .max()
                .unwrap_or(0);
            let bindings: Vec<_> = (0..num_bindings)
                .map(|binding_num| {
                    match descriptors
                        .iter()
                        .find(|d| d.set == set_num && d.binding == binding_num)
                    {
                        Some(d) => {
                            let desc_ty = &d.desc_ty;
                            let array_count = d.array_count as u32;
                            let readonly = d.readonly;
                            quote! {
                                Some(DescriptorDesc {
                                    ty: #desc_ty,
                                    array_count: #array_count,
                                    stages: #stages,
                                    readonly: #readonly,
                                }),
                            }
                        }
                        None => quote! {
                            None,
                        },
                    }
                })
                .collect();

            quote! {
                DescriptorSetDesc::new(
                    [#( #bindings )*]
                ),
            }
        })
        .collect();

    quote! {
        [
            #( #sets )*
        ]
    }
}

pub(super) fn write_push_constant_ranges(doc: &Spirv, stage: &TokenStream, types_meta: &TypesMeta) -> TokenStream {
    // TODO: somewhat implemented correctly

    // Looping to find all the push constant structs.
    let mut push_constants_size = 0;
    for instruction in doc.instructions.iter() {
        let type_id = match instruction {
            &Instruction::TypePointer {
                type_id,
                storage_class: StorageClass::PushConstant,
                ..
            } => type_id,
            _ => continue,
        };

        let (_, size, _) = crate::structs::type_from_id(doc, type_id, types_meta);
        let size = size.expect("Found runtime-sized push constants");
        push_constants_size = cmp::max(push_constants_size, size);
    }

    if push_constants_size == 0 {
        quote! {
            None
        }
    } else {
        quote! {
            Some(
                PipelineLayoutPcRange {
                    offset: 0,                   // FIXME: not necessarily true
                    size: #push_constants_size,
                    stages: #stage,
                }
            )
        }
    }
}

fn find_descriptors(
    doc: &Spirv,
    entrypoint_id: u32,
    interface: &[u32],
    exact: bool,
) -> Vec<Descriptor> {
    let mut descriptors = Vec::new();

    // For SPIR-V 1.4+, the entrypoint interface can specify variables of all storage classes,
    // and most tools will put all used variables in the entrypoint interface. However,
    // SPIR-V 1.0-1.3 do not specify variables other than Input/Output ones in the interface,
    // and instead the function itself must be inspected.
    let variables = if exact {
        let mut found_variables: HashSet<u32> = interface.iter().cloned().collect();
        let mut inspected_functions: HashSet<u32> = HashSet::new();
        find_variables_in_function(
            &doc,
            entrypoint_id,
            &mut inspected_functions,
            &mut found_variables,
        );
        Some(found_variables)
    } else {
        None
    };

    // Looping to find all the interface elements that have the `DescriptorSet` decoration.
    for set_decoration in doc.get_decorations(Decoration::DescriptorSet) {
        let variable_id = set_decoration.target_id;

        if exact && !variables.as_ref().unwrap().contains(&variable_id) {
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
            .get_decoration_params(variable_id, Decoration::Binding)
            .unwrap()[0];

        let nonwritable = doc
            .get_decoration_params(variable_id, Decoration::NonWritable)
            .is_some();

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
            readonly: nonwritable || readonly,
        });
    }

    descriptors
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
                Instruction::Function { result_id, .. } if result_id == &function => {
                    in_function = true;
                }
                _ => {}
            }
        } else {
            // We only care about instructions that accept pointers.
            // https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_universal_validation_rules
            match instruction {
                Instruction::Load { pointer, .. } | Instruction::Store { pointer, .. } => {
                    found_variables.insert(*pointer);
                }
                Instruction::AccessChain { base_id, .. }
                | Instruction::InBoundsAccessChain { base_id, .. } => {
                    found_variables.insert(*base_id);
                }
                Instruction::FunctionCall {
                    function_id, args, ..
                } => {
                    args.iter().for_each(|&x| {
                        found_variables.insert(x);
                    });
                    if !inspected_functions.contains(function_id) {
                        find_variables_in_function(
                            doc,
                            *function_id,
                            inspected_functions,
                            found_variables,
                        );
                    }
                }
                Instruction::ImageTexelPointer {
                    image,
                    coordinate,
                    sample,
                    ..
                } => {
                    found_variables.insert(*image);
                    found_variables.insert(*coordinate);
                    found_variables.insert(*sample);
                }
                Instruction::CopyMemory {
                    target_id,
                    source_id,
                    ..
                } => {
                    found_variables.insert(*target_id);
                    found_variables.insert(*source_id);
                }
                Instruction::CopyObject { operand_id, .. } => {
                    found_variables.insert(*operand_id);
                }
                Instruction::AtomicLoad { pointer, .. }
                | Instruction::AtomicIIncrement { pointer, .. }
                | Instruction::AtomicIDecrement { pointer, .. }
                | Instruction::AtomicFlagTestAndSet { pointer, .. }
                | Instruction::AtomicFlagClear { pointer, .. } => {
                    found_variables.insert(*pointer);
                }
                Instruction::AtomicStore {
                    pointer, value_id, ..
                }
                | Instruction::AtomicExchange {
                    pointer, value_id, ..
                }
                | Instruction::AtomicIAdd {
                    pointer, value_id, ..
                }
                | Instruction::AtomicISub {
                    pointer, value_id, ..
                }
                | Instruction::AtomicSMin {
                    pointer, value_id, ..
                }
                | Instruction::AtomicUMin {
                    pointer, value_id, ..
                }
                | Instruction::AtomicSMax {
                    pointer, value_id, ..
                }
                | Instruction::AtomicUMax {
                    pointer, value_id, ..
                }
                | Instruction::AtomicAnd {
                    pointer, value_id, ..
                }
                | Instruction::AtomicOr {
                    pointer, value_id, ..
                }
                | Instruction::AtomicXor {
                    pointer, value_id, ..
                } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value_id);
                }
                Instruction::AtomicCompareExchange {
                    pointer,
                    value_id,
                    comparator_id,
                    ..
                }
                | Instruction::AtomicCompareExchangeWeak {
                    pointer,
                    value_id,
                    comparator_id,
                    ..
                } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value_id);
                    found_variables.insert(*comparator_id);
                }
                Instruction::ExtInst { operands, .. } => {
                    // We don't know which extended instructions take pointers,
                    // so we must interpret every operand as a pointer.
                    operands.iter().for_each(|&o| {
                        found_variables.insert(o);
                    });
                }
                Instruction::FunctionEnd => return,
                _ => {}
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
                Instruction::TypeStruct { result_id, member_types } if *result_id == pointed_ty => {
                    let decoration_block = doc
                        .get_decoration_params(pointed_ty, Decoration::Block)
                        .is_some();
                    let decoration_buffer_block = doc
                        .get_decoration_params(pointed_ty, Decoration::BufferBlock)
                        .is_some();
                    assert!(
                        decoration_block ^ decoration_buffer_block,
                        "Structs in shader interface are expected to be decorated with one of Block or BufferBlock"
                    );

                    // false -> VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                    // true -> VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    let storage = decoration_buffer_block || decoration_block && pointer_storage == StorageClass::StorageBuffer;

                    // Determine whether all members have a NonWritable decoration.
                    let nonwritable = (0..member_types.len() as u32).all(|i| {
                        doc.get_member_decoration_params(pointed_ty, i, Decoration::NonWritable).is_some()
                    });

                    // Uniforms are never writable.
                    let readonly = !storage || nonwritable;

                    let desc = quote! {
                        DescriptorDescTy::Buffer(DescriptorBufferDesc {
                            dynamic: None,
                            storage: #storage,
                        })
                    };

                    Some((desc, readonly, 1))
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

                    let vulkan_format = to_vulkan_format(*format);

                    let arrayed = match arrayed {
                        true => quote! { DescriptorImageDescArray::Arrayed { max_layers: None } },
                        false => quote! { DescriptorImageDescArray::NonArrayed },
                    };

                    match dim {
                        Dim::DimSubpassData => {
                            // VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
                            assert!(
                                !force_combined_image_sampled,
                                "An OpTypeSampledImage can't point to \
                                                                an OpTypeImage whose dimension is \
                                                                SubpassData"
                            );
                            assert!(
                                *format == ImageFormat::Unknown,
                                "If Dim is SubpassData, Image Format must be Unknown"
                            );
                            assert!(!sampled, "If Dim is SubpassData, Sampled must be 2");

                            let desc = quote! {
                                DescriptorDescTy::InputAttachment {
                                    multisampled: #ms,
                                    array_layers: #arrayed
                                }
                            };

                            Some((desc, true, 1)) // Never writable.
                        }
                        Dim::DimBuffer => {
                            // false -> VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                            // true -> VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                            let storage = !sampled;
                            let desc = quote! {
                                DescriptorDescTy::TexelBuffer {
                                    storage: #storage,
                                    format: #vulkan_format,
                                }
                            };

                            Some((desc, !storage, 1)) // Uniforms are never writable.
                        }
                        _ => {
                            let (ty, readonly) = match force_combined_image_sampled {
                                // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                // Never writable.
                                true => (quote! { DescriptorDescTy::CombinedImageSampler }, true),
                                false => {
                                    // false -> VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                    // true -> VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                    let storage = !sampled;
                                    (quote! { DescriptorDescTy::Image }, !storage) // Sampled images are never writable.
                                },
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
                                    format: #vulkan_format,
                                    multisampled: #ms,
                                    array_layers: #arrayed,
                                })
                            };

                            Some((desc, readonly, 1))
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

                &Instruction::TypeRuntimeArray {
                    result_id,
                    type_id,
                } if result_id == pointed_ty => {
                    let (desc, readonly, arr) =
                        match descriptor_infos(doc, type_id, pointer_storage.clone(), false) {
                            None => return None,
                            Some(v) => v,
                        };
                    assert_eq!(arr, 1); // TODO: implement?
                    Some((desc, readonly, 0))
                }

                _ => None, // TODO: other types
            }
        })
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::compile;
    use crate::parse;
    use shaderc::ShaderKind;
    use std::path::{Path, PathBuf};

    /// `entrypoint1.frag.glsl`:
    /// ```glsl
    /// #version 450
    ///
    /// layout(set = 0, binding = 0) uniform Uniform {
    ///     uint data;
    /// } ubo;
    ///
    /// layout(set = 0, binding = 1) buffer Buffer {
    ///     uint data;
    /// } bo;
    ///
    /// layout(set = 0, binding = 2) uniform sampler textureSampler;
    /// layout(set = 0, binding = 3) uniform texture2D imageTexture;
    ///
    /// layout(push_constant) uniform PushConstant {
    ///    uint data;
    /// } push;
    ///
    /// layout(input_attachment_index = 0, set = 0, binding = 4) uniform subpassInput inputAttachment;
    ///
    /// layout(location = 0) out vec4 outColor;
    ///
    /// void entrypoint1() {
    ///     bo.data = 12;
    ///     outColor = vec4(
    ///         float(ubo.data),
    ///         float(push.data),
    ///         texture(sampler2D(imageTexture, textureSampler), vec2(0.0, 0.0)).x,
    ///         subpassLoad(inputAttachment).x
    ///     );
    /// }
    /// ```
    ///
    /// `entrypoint2.frag.glsl`:
    /// ```glsl
    /// #version 450
    ///
    /// layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputAttachment2;
    ///
    /// layout(set = 0, binding = 1) buffer Buffer {
    ///     uint data;
    /// } bo2;
    ///
    /// layout(set = 0, binding = 2) uniform Uniform {
    ///     uint data;
    /// } ubo2;
    ///
    /// layout(push_constant) uniform PushConstant {
    ///    uint data;
    /// } push2;
    ///
    /// void entrypoint2() {
    ///     bo2.data = ubo2.data + push2.data + int(subpassLoad(inputAttachment2).y);
    /// }
    /// ```
    ///
    /// Compiled and linked with:
    /// ```sh
    /// glslangvalidator -e entrypoint1 --source-entrypoint entrypoint1 -V100 entrypoint1.frag.glsl -o entrypoint1.spv
    /// glslangvalidator -e entrypoint2 --source-entrypoint entrypoint2 -V100 entrypoint2.frag.glsl -o entrypoint2.spv
    /// spirv-link entrypoint1.spv entrypoint2.spv -o multiple_entrypoints.spv
    /// ```
    #[test]
    fn test_descriptor_calculation_with_multiple_entrypoints() {
        let data = include_bytes!("../tests/multiple_entrypoints.spv");
        let instructions: Vec<u32> = data
            .chunks(4)
            .map(|c| {
                ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) | c[0] as u32
            })
            .collect();
        let doc = parse::parse_spirv(&instructions).unwrap();

        let mut descriptors = Vec::new();
        for instruction in doc.instructions.iter() {
            if let &Instruction::EntryPoint {
                id, ref interface, ..
            } = instruction
            {
                descriptors.push(find_descriptors(&doc, id, interface, true));
            }
        }

        // Check first entrypoint
        let e1_descriptors = descriptors.get(0).expect("Could not find entrypoint1");
        let mut e1_bindings = Vec::new();
        for d in e1_descriptors {
            e1_bindings.push((d.set, d.binding));
        }
        assert_eq!(e1_bindings.len(), 5);
        assert!(e1_bindings.contains(&(0, 0)));
        assert!(e1_bindings.contains(&(0, 1)));
        assert!(e1_bindings.contains(&(0, 2)));
        assert!(e1_bindings.contains(&(0, 3)));
        assert!(e1_bindings.contains(&(0, 4)));

        // Check second entrypoint
        let e2_descriptors = descriptors.get(1).expect("Could not find entrypoint2");
        let mut e2_bindings = Vec::new();
        for d in e2_descriptors {
            e2_bindings.push((d.set, d.binding));
        }
        assert_eq!(e2_bindings.len(), 3);
        assert!(e2_bindings.contains(&(0, 0)));
        assert!(e2_bindings.contains(&(0, 1)));
        assert!(e2_bindings.contains(&(0, 2)));
    }

    #[test]
    fn test_descriptor_calculation_with_multiple_functions() {
        let includes: [PathBuf; 0] = [];
        let defines: [(String, String); 0] = [];
        let (comp, _) = compile(
            None,
            &Path::new(""),
            "
        #version 450

        layout(set = 1, binding = 0) buffer Buffer {
            vec3 data;
        } bo;

        layout(set = 2, binding = 0) uniform Uniform {
            float data;
        } ubo;

        layout(set = 3, binding = 1) uniform sampler textureSampler;
        layout(set = 3, binding = 2) uniform texture2D imageTexture;

        float withMagicSparkles(float data) {
            return texture(sampler2D(imageTexture, textureSampler), vec2(data, data)).x;
        }

        vec3 makeSecretSauce() {
            return vec3(withMagicSparkles(ubo.data));
        }

        void main() {
            bo.data = makeSecretSauce();
        }
        ",
            ShaderKind::Vertex,
            &includes,
            &defines,
            None,
            None,
        )
        .unwrap();
        let doc = parse::parse_spirv(comp.as_binary()).unwrap();

        for instruction in doc.instructions.iter() {
            if let &Instruction::EntryPoint {
                id, ref interface, ..
            } = instruction
            {
                let descriptors = find_descriptors(&doc, id, interface, true);
                let mut bindings = Vec::new();
                for d in descriptors {
                    bindings.push((d.set, d.binding));
                }
                assert_eq!(bindings.len(), 4);
                assert!(bindings.contains(&(1, 0)));
                assert!(bindings.contains(&(2, 0)));
                assert!(bindings.contains(&(3, 1)));
                assert!(bindings.contains(&(3, 2)));

                return;
            }
        }
        panic!("Could not find entrypoint");
    }
}

fn to_vulkan_format(spirv_format: ImageFormat) -> TokenStream {
    match spirv_format {
        ImageFormat::Unknown => quote! { None },
        ImageFormat::Rgba32f => quote! { Some(Format::R32G32B32A32Sfloat) },
        ImageFormat::Rgba16f => quote! { Some(Format::R16G16B16A16Sfloat) },
        ImageFormat::R32f => quote! { Some(Format::R32Sfloat) },
        ImageFormat::Rgba8 => quote! { Some(Format::R8G8B8A8Unorm) },
        ImageFormat::Rgba8Snorm => quote! { Some(Format::R8G8B8A8Snorm) },
        ImageFormat::Rg32f => quote! { Some(Format::R32G32Sfloat) },
        ImageFormat::Rg16f => quote! { Some(Format::R16G16Sfloat) },
        ImageFormat::R11fG11fB10f => quote! { Some(Format::B10G11R11UfloatPack32) },
        ImageFormat::R16f => quote! { Some(Format::R16Sfloat) },
        ImageFormat::Rgba16 => quote! { Some(Format::R16G16B16A16Unorm) },
        ImageFormat::Rgb10A2 => quote! { Some(Format::A2B10G10R10UnormPack32) },
        ImageFormat::Rg16 => quote! { Some(Format::R16G16Unorm) },
        ImageFormat::Rg8 => quote! { Some(Format::R8G8Unorm) },
        ImageFormat::R16 => quote! { Some(Format::R16Unorm) },
        ImageFormat::R8 => quote! { Some(Format::R8Unorm) },
        ImageFormat::Rgba16Snorm => quote! { Some(Format::R16G16B16A16Snorm) },
        ImageFormat::Rg16Snorm => quote! { Some(Format::R16G16Snorm) },
        ImageFormat::Rg8Snorm => quote! { Some(Format::R8G8Snorm) },
        ImageFormat::R16Snorm => quote! { Some(Format::R16Snorm) },
        ImageFormat::R8Snorm => quote! { Some(Format::R8Snorm) },
        ImageFormat::Rgba32i => quote! { Some(Format::R32G32B32A32Sint) },
        ImageFormat::Rgba16i => quote! { Some(Format::R16G16B16A16Sint) },
        ImageFormat::Rgba8i => quote! { Some(Format::R8G8B8A8Sint) },
        ImageFormat::R32i => quote! { Some(Format::R32Sint) },
        ImageFormat::Rg32i => quote! { Some(Format::R32G32Sint) },
        ImageFormat::Rg16i => quote! { Some(Format::R16G16Sint) },
        ImageFormat::Rg8i => quote! { Some(Format::R8G8Sint) },
        ImageFormat::R16i => quote! { Some(Format::R16Sint) },
        ImageFormat::R8i => quote! { Some(Format::R8Sint) },
        ImageFormat::Rgba32ui => quote! { Some(Format::R32G32B32A32Uint) },
        ImageFormat::Rgba16ui => quote! { Some(Format::R16G16B16A16Uint) },
        ImageFormat::Rgba8ui => quote! { Some(Format::R8G8B8A8Uint) },
        ImageFormat::R32ui => quote! { Some(Format::R32Uint) },
        ImageFormat::Rgb10a2ui => quote! { Some(Format::A2B10G10R10UintPack32) },
        ImageFormat::Rg32ui => quote! { Some(Format::R32G32Uint) },
        ImageFormat::Rg16ui => quote! { Some(Format::R16G16Uint) },
        ImageFormat::Rg8ui => quote! { Some(Format::R8G8Uint) },
        ImageFormat::R16ui => quote! { Some(Format::R16Uint) },
        ImageFormat::R8ui => quote! { Some(Format::R8Uint) },
    }
}
