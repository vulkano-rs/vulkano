// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::TypesMeta;
use proc_macro2::TokenStream;
use std::cmp;
use std::collections::HashSet;
use vulkano::spirv::{Decoration, Dim, Id, ImageFormat, Instruction, Spirv, StorageClass};

#[derive(Debug)]
struct Descriptor {
    set_num: u32,
    binding_num: u32,
    desc_ty: TokenStream,
    descriptor_count: u64,
    variable_count: bool,
    mutable: bool,
}

pub(super) fn write_descriptor_set_layout_descs(
    spirv: &Spirv,
    entrypoint_id: Id,
    interface: &[Id],
    exact_entrypoint_interface: bool,
    stages: &TokenStream,
) -> TokenStream {
    // TODO: somewhat implemented correctly

    // Finding all the descriptors.
    let descriptors = find_descriptors(spirv, entrypoint_id, interface, exact_entrypoint_interface);
    let num_sets = descriptors.iter().map(|d| d.set_num + 1).max().unwrap_or(0);
    let sets: Vec<_> = (0..num_sets)
        .map(|set_num| {
            let num_bindings = descriptors
                .iter()
                .filter(|d| d.set_num == set_num)
                .map(|d| d.binding_num + 1)
                .max()
                .unwrap_or(0);
            let bindings: Vec<_> = (0..num_bindings)
                .map(|binding_num| {
                    match descriptors
                        .iter()
                        .find(|d| d.set_num == set_num && d.binding_num == binding_num)
                    {
                        Some(d) => {
                            let desc_ty = &d.desc_ty;
                            let descriptor_count = d.descriptor_count as u32;
                            let mutable = d.mutable;
                            let variable_count = d.variable_count;
                            quote! {
                                Some(DescriptorDesc {
                                    ty: #desc_ty,
                                    descriptor_count: #descriptor_count,
                                    stages: #stages,
                                    variable_count: #variable_count,
                                    mutable: #mutable,
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

pub(super) fn write_push_constant_ranges(
    shader: &str,
    spirv: &Spirv,
    stage: &TokenStream,
    types_meta: &TypesMeta,
) -> TokenStream {
    // TODO: somewhat implemented correctly

    // Looping to find all the push constant structs.
    let mut push_constants_size = 0;
    for type_id in spirv
        .iter_global()
        .filter_map(|instruction| match instruction {
            &Instruction::TypePointer {
                ty,
                storage_class: StorageClass::PushConstant,
                ..
            } => Some(ty),
            _ => None,
        })
    {
        let (_, _, size, _) = crate::structs::type_from_id(shader, spirv, type_id, types_meta);
        let size = size.expect("Found runtime-sized push constants") as u32;
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
    spirv: &Spirv,
    entrypoint_id: Id,
    interface: &[Id],
    exact: bool,
) -> Vec<Descriptor> {
    // For SPIR-V 1.4+, the entrypoint interface can specify variables of all storage classes,
    // and most tools will put all used variables in the entrypoint interface. However,
    // SPIR-V 1.0-1.3 do not specify variables other than Input/Output ones in the interface,
    // and instead the function itself must be inspected.
    let variables = if exact {
        let mut found_variables: HashSet<Id> = interface.iter().cloned().collect();
        let mut inspected_functions: HashSet<Id> = HashSet::new();
        find_variables_in_function(
            &spirv,
            entrypoint_id,
            &mut inspected_functions,
            &mut found_variables,
        );
        Some(found_variables)
    } else {
        None
    };

    // Looping to find all the global variables that have the `DescriptorSet` decoration.
    spirv
        .iter_global()
        .filter_map(|instruction| {
            let (variable_id, variable_type_id, storage_class) = match instruction {
                Instruction::Variable {
                    result_id,
                    result_type_id,
                    ..
                } => {
                    let (real_type, storage_class) = match spirv
                        .id(*result_type_id)
                        .instruction()
                    {
                        Instruction::TypePointer {
                            ty, storage_class, ..
                        } => (ty, storage_class),
                        _ => panic!(
                            "Variable {} result_type_id does not refer to a TypePointer instruction", result_id
                        ),
                    };

                    (*result_id, *real_type, storage_class)
                }
                _ => return None,
            };

            if exact && !variables.as_ref().unwrap().contains(&variable_id) {
                return None;
            }

            let variable_id_info = spirv.id(variable_id);
            let set_num = match variable_id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::DescriptorSet { descriptor_set },
                        ..
                    } => Some(*descriptor_set),
                    _ => None,
                }) {
                Some(x) => x,
                None => return None,
            };

            let binding_num = variable_id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::Binding { binding_point },
                        ..
                    } => Some(*binding_point),
                    _ => None,
                })
                .unwrap();

            let name = variable_id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .unwrap_or("__unnamed");

            let nonwritable = variable_id_info.iter_decoration().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Decorate {
                        decoration: Decoration::NonWritable,
                        ..
                    }
                )
            });

            // Find information about the kind of binding for this descriptor.
            let (desc_ty, mutable, descriptor_count, variable_count) =
                descriptor_infos(spirv, variable_type_id, storage_class, false).expect(&format!(
                "Couldn't find relevant type for uniform `{}` (type {}, maybe unimplemented)",
                name, variable_type_id,
            ));

            Some(Descriptor {
                desc_ty,
                set_num,
                binding_num,
                descriptor_count,
                mutable: !nonwritable && mutable,
                variable_count: variable_count,
            })
        })
        .collect()
}

// Recursively finds every pointer variable used in the execution of a function.
fn find_variables_in_function(
    spirv: &Spirv,
    function: Id,
    inspected_functions: &mut HashSet<Id>,
    found_variables: &mut HashSet<Id>,
) {
    inspected_functions.insert(function);
    let mut in_function = false;
    for instruction in spirv.instructions() {
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
                Instruction::AccessChain { base, .. }
                | Instruction::InBoundsAccessChain { base, .. } => {
                    found_variables.insert(*base);
                }
                Instruction::FunctionCall {
                    function,
                    arguments,
                    ..
                } => {
                    arguments.iter().for_each(|&x| {
                        found_variables.insert(x);
                    });
                    if !inspected_functions.contains(function) {
                        find_variables_in_function(
                            spirv,
                            *function,
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
                Instruction::CopyMemory { target, source, .. } => {
                    found_variables.insert(*target);
                    found_variables.insert(*source);
                }
                Instruction::CopyObject { operand, .. } => {
                    found_variables.insert(*operand);
                }
                Instruction::AtomicLoad { pointer, .. }
                | Instruction::AtomicIIncrement { pointer, .. }
                | Instruction::AtomicIDecrement { pointer, .. }
                | Instruction::AtomicFlagTestAndSet { pointer, .. }
                | Instruction::AtomicFlagClear { pointer, .. } => {
                    found_variables.insert(*pointer);
                }
                Instruction::AtomicStore { pointer, value, .. }
                | Instruction::AtomicExchange { pointer, value, .. }
                | Instruction::AtomicIAdd { pointer, value, .. }
                | Instruction::AtomicISub { pointer, value, .. }
                | Instruction::AtomicSMin { pointer, value, .. }
                | Instruction::AtomicUMin { pointer, value, .. }
                | Instruction::AtomicSMax { pointer, value, .. }
                | Instruction::AtomicUMax { pointer, value, .. }
                | Instruction::AtomicAnd { pointer, value, .. }
                | Instruction::AtomicOr { pointer, value, .. }
                | Instruction::AtomicXor { pointer, value, .. } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value);
                }
                Instruction::AtomicCompareExchange {
                    pointer,
                    value,
                    comparator,
                    ..
                }
                | Instruction::AtomicCompareExchangeWeak {
                    pointer,
                    value,
                    comparator,
                    ..
                } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value);
                    found_variables.insert(*comparator);
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

/// Returns a `DescriptorDescTy` constructor, a bool indicating whether the descriptor is
/// read-only, and the number of array elements.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_infos(
    spirv: &Spirv,
    pointed_ty: Id,
    pointer_storage: &StorageClass,
    force_combined_image_sampled: bool,
) -> Option<(TokenStream, bool, u64, bool)> {
    let id_info = spirv.id(pointed_ty);

    match id_info.instruction() {
        Instruction::TypeStruct { .. } => {
            let decoration_block = id_info.iter_decoration().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Decorate {
                        decoration: Decoration::Block,
                        ..
                    }
                )
            });

            let decoration_buffer_block = id_info.iter_decoration().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Decorate {
                        decoration: Decoration::BufferBlock,
                        ..
                    }
                )
            });

            assert!(
                decoration_block ^ decoration_buffer_block,
                "Structs in shader interface are expected to be decorated with one of Block or BufferBlock"
            );

            let (ty, mutable) = if decoration_buffer_block
                || decoration_block && *pointer_storage == StorageClass::StorageBuffer
            {
                // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                // Determine whether all members have a NonWritable decoration.
                let nonwritable = id_info.iter_members().all(|member_info| {
                    member_info.iter_decoration().any(|instruction| {
                        matches!(
                            instruction,
                            Instruction::MemberDecorate {
                                decoration: Decoration::NonWritable,
                                ..
                            }
                        )
                    })
                });

                (quote! { DescriptorDescTy::StorageBuffer }, !nonwritable)
            } else {
                // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                (quote! { DescriptorDescTy::UniformBuffer }, false) // Uniforms are never mutable.
            };

            Some((ty, mutable, 1, false))
        }
        &Instruction::TypeImage {
            ref dim,
            arrayed,
            ms,
            sampled,
            ref image_format,
            ..
        } => {
            let ms = ms != 0;
            assert!(sampled != 0, "Vulkan requires that variables of type OpTypeImage have a Sampled operand of 1 or 2");
            let vulkan_format = to_vulkan_format(image_format);

            match dim {
                Dim::SubpassData => {
                    // VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT
                    assert!(
                        !force_combined_image_sampled,
                        "An OpTypeSampledImage can't point to \
                                                                an OpTypeImage whose dimension is \
                                                                SubpassData"
                    );
                    assert!(
                        *image_format == ImageFormat::Unknown,
                        "If Dim is SubpassData, Image Format must be Unknown"
                    );
                    assert!(sampled == 2, "If Dim is SubpassData, Sampled must be 2");
                    assert!(arrayed == 0, "If Dim is SubpassData, Arrayed must be 0");

                    let desc = quote! {
                        DescriptorDescTy::InputAttachment {
                            multisampled: #ms,
                        }
                    };

                    Some((desc, true, 1, false)) // Never writable.
                }
                Dim::Buffer => {
                    let (ty, mutable) = if sampled == 1 {
                        // VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                        (quote! { DescriptorDescTy::UniformTexelBuffer }, false)
                    // Uniforms are never mutable.
                    } else {
                        // VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                        (quote! { DescriptorDescTy::StorageTexelBuffer }, true)
                    };

                    let desc = quote! {
                        #ty {
                            format: #vulkan_format,
                        }
                    };

                    Some((desc, mutable, 1, false))
                }
                _ => {
                    let view_type = match (dim, arrayed) {
                        (Dim::Dim1D, 0) => quote! { ImageViewType::Dim1d },
                        (Dim::Dim1D, 1) => quote! { ImageViewType::Dim1dArray },
                        (Dim::Dim2D, 0) => quote! { ImageViewType::Dim2d },
                        (Dim::Dim2D, 1) => quote! { ImageViewType::Dim2dArray },
                        (Dim::Dim3D, 0) => quote! { ImageViewType::Dim3d },
                        (Dim::Dim3D, 1) => panic!("Vulkan doesn't support arrayed 3D textures"),
                        (Dim::Cube, 0) => quote! { ImageViewType::Cube },
                        (Dim::Cube, 1) => quote! { ImageViewType::CubeArray },
                        (Dim::Rect, _) => panic!("Vulkan doesn't support rectangle textures"),
                        _ => unreachable!(),
                    };

                    let image_desc = quote! {
                        DescriptorDescImage {
                            format: #vulkan_format,
                            multisampled: #ms,
                            view_type: #view_type,
                        }
                    };

                    let (desc, mutable) = if force_combined_image_sampled {
                        // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                        // Never writable.
                        assert!(
                            sampled == 1,
                            "A combined image sampler must not reference a storage image"
                        );

                        (
                            quote! {
                                DescriptorDescTy::CombinedImageSampler {
                                    image_desc: #image_desc,
                                    immutable_samplers: Vec::new(),
                                }
                            },
                            false, // Sampled images are never mutable.
                        )
                    } else {
                        let (ty, mutable) = if sampled == 1 {
                            // VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                            (quote! { DescriptorDescTy::SampledImage }, false) // Sampled images are never mutable.
                        } else {
                            // VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                            (quote! { DescriptorDescTy::StorageImage }, true)
                        };

                        (
                            quote! {
                                #ty {
                                    image_desc: #image_desc,
                                }
                            },
                            mutable,
                        )
                    };

                    Some((desc, mutable, 1, false))
                }
            }
        }

        &Instruction::TypeSampledImage { image_type, .. } => {
            descriptor_infos(spirv, image_type, pointer_storage, true)
        }

        &Instruction::TypeSampler { .. } => {
            let desc = quote! { DescriptorDescTy::Sampler { immutable_samplers: Vec::new() } };
            Some((desc, false, 1, false))
        }

        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            let (desc, mutable, arr, variable_count) =
                match descriptor_infos(spirv, element_type, pointer_storage, false) {
                    None => return None,
                    Some(v) => v,
                };
            assert_eq!(arr, 1); // TODO: implement?
            assert!(!variable_count); // TODO: Is this even a thing?
            let len = match spirv.id(length).instruction() {
                &Instruction::Constant { ref value, .. } => value,
                _ => panic!("failed to find array length"),
            };
            let len = len.iter().rev().fold(0, |a, &b| (a << 32) | b as u64);

            Some((desc, mutable, len, false))
        }

        &Instruction::TypeRuntimeArray { element_type, .. } => {
            let (desc, mutable, arr, variable_count) =
                match descriptor_infos(spirv, element_type, pointer_storage, false) {
                    None => return None,
                    Some(v) => v,
                };
            assert_eq!(arr, 1); // TODO: implement?
            assert!(!variable_count); // TODO: Don't think this is possible?

            Some((desc, mutable, 1, true))
        }

        _ => None, // TODO: other types
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::compile;
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
        let spirv = Spirv::new(&instructions).unwrap();

        let mut descriptors = Vec::new();
        for instruction in spirv.instructions() {
            if let &Instruction::EntryPoint {
                entry_point,
                ref interface,
                ..
            } = instruction
            {
                descriptors.push(find_descriptors(&spirv, entry_point, interface, true));
            }
        }

        // Check first entrypoint
        let e1_descriptors = descriptors.get(0).expect("Could not find entrypoint1");
        let mut e1_bindings = Vec::new();
        for d in e1_descriptors {
            e1_bindings.push((d.set_num, d.binding_num));
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
            e2_bindings.push((d.set_num, d.binding_num));
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
        let spirv = Spirv::new(comp.as_binary()).unwrap();

        for instruction in spirv.instructions() {
            if let &Instruction::EntryPoint {
                entry_point,
                ref interface,
                ..
            } = instruction
            {
                let descriptors = find_descriptors(&spirv, entry_point, interface, true);
                let mut bindings = Vec::new();
                for d in descriptors {
                    bindings.push((d.set_num, d.binding_num));
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

fn to_vulkan_format(spirv_format: &ImageFormat) -> TokenStream {
    match spirv_format {
        ImageFormat::Unknown => quote! { None },
        ImageFormat::Rgba32f => quote! { Some(Format::R32G32B32A32_SFLOAT) },
        ImageFormat::Rgba16f => quote! { Some(Format::R16G16B16A16_SFLOAT) },
        ImageFormat::R32f => quote! { Some(Format::R32_SFLOAT) },
        ImageFormat::Rgba8 => quote! { Some(Format::R8G8B8A8_UNORM) },
        ImageFormat::Rgba8Snorm => quote! { Some(Format::R8G8B8A8_SNORM) },
        ImageFormat::Rg32f => quote! { Some(Format::R32G32_SFLOAT) },
        ImageFormat::Rg16f => quote! { Some(Format::R16G16_SFLOAT) },
        ImageFormat::R11fG11fB10f => quote! { Some(Format::B10G11R11_UFLOAT_PACK32) },
        ImageFormat::R16f => quote! { Some(Format::R16_SFLOAT) },
        ImageFormat::Rgba16 => quote! { Some(Format::R16G16B16A16_UNORM) },
        ImageFormat::Rgb10A2 => quote! { Some(Format::A2B10G10R10_UNORMPack32) },
        ImageFormat::Rg16 => quote! { Some(Format::R16G16_UNORM) },
        ImageFormat::Rg8 => quote! { Some(Format::R8G8_UNORM) },
        ImageFormat::R16 => quote! { Some(Format::R16_UNORM) },
        ImageFormat::R8 => quote! { Some(Format::R8_UNORM) },
        ImageFormat::Rgba16Snorm => quote! { Some(Format::R16G16B16A16_SNORM) },
        ImageFormat::Rg16Snorm => quote! { Some(Format::R16G16_SNORM) },
        ImageFormat::Rg8Snorm => quote! { Some(Format::R8G8_SNORM) },
        ImageFormat::R16Snorm => quote! { Some(Format::R16_SNORM) },
        ImageFormat::R8Snorm => quote! { Some(Format::R8_SNORM) },
        ImageFormat::Rgba32i => quote! { Some(Format::R32G32B32A32_SINT) },
        ImageFormat::Rgba16i => quote! { Some(Format::R16G16B16A16_SINT) },
        ImageFormat::Rgba8i => quote! { Some(Format::R8G8B8A8_SINT) },
        ImageFormat::R32i => quote! { Some(Format::R32_SINT) },
        ImageFormat::Rg32i => quote! { Some(Format::R32G32_SINT) },
        ImageFormat::Rg16i => quote! { Some(Format::R16G16_SINT) },
        ImageFormat::Rg8i => quote! { Some(Format::R8G8_SINT) },
        ImageFormat::R16i => quote! { Some(Format::R16_SINT) },
        ImageFormat::R8i => quote! { Some(Format::R8_SINT) },
        ImageFormat::Rgba32ui => quote! { Some(Format::R32G32B32A32_UINT) },
        ImageFormat::Rgba16ui => quote! { Some(Format::R16G16B16A16_UINT) },
        ImageFormat::Rgba8ui => quote! { Some(Format::R8G8B8A8_UINT) },
        ImageFormat::R32ui => quote! { Some(Format::R32_UINT) },
        ImageFormat::Rgb10a2ui => quote! { Some(Format::A2B10G10R10_UINT_PACK32) },
        ImageFormat::Rg32ui => quote! { Some(Format::R32G32_UINT) },
        ImageFormat::Rg16ui => quote! { Some(Format::R16G16_UINT) },
        ImageFormat::Rg8ui => quote! { Some(Format::R8G8_UINT) },
        ImageFormat::R16ui => quote! { Some(Format::R16_UINT) },
        ImageFormat::R8ui => quote! { Some(Format::R8_UINT) },
        ImageFormat::R64ui => quote! { Some(Format::R64_UINT) },
        ImageFormat::R64i => quote! { Some(Format::R64_SINT) },
    }
}
