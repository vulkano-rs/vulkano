// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashSet;

use enums;
use parse;

pub fn write_descriptor_sets(doc: &parse::Spirv) -> String {
    // TODO: not implemented correctly

    // Finding all the descriptors.
    let mut descriptors = Vec::new();
    struct Descriptor {
        set: u32,
        binding: u32,
        desc_ty: String,
        readonly: bool,
    }

    // Looping to find all the elements that have the `DescriptorSet` decoration.
    for instruction in doc.instructions.iter() {
        let (variable_id, descriptor_set) = match instruction {
            &parse::Instruction::Decorate { target_id, decoration: enums::Decoration::DecorationDescriptorSet, ref params } => {
                (target_id, params[0])
            },
            _ => continue
        };

        // Find which type is pointed to by this variable.
        let pointed_ty = pointer_variable_ty(doc, variable_id);
        // Name of the variable.
        let name = ::name_from_id(doc, variable_id);

        // Find the binding point of this descriptor.
        let binding = doc.instructions.iter().filter_map(|i| {
            match i {
                &parse::Instruction::Decorate { target_id, decoration: enums::Decoration::DecorationBinding, ref params } if target_id == variable_id => {
                    Some(params[0])
                },
                _ => None,      // TODO: other types
            }
        }).next().expect(&format!("Uniform `{}` is missing a binding", name));

        // Find informations about the kind of binding for this descriptor.
        let (desc_ty, readonly) = descriptor_infos(doc, pointed_ty, false).expect(&format!("Couldn't find relevant type for uniform `{}` (type {}, maybe unimplemented)", name, pointed_ty));

        descriptors.push(Descriptor {
            desc_ty: desc_ty,
            set: descriptor_set,
            binding: binding,
            readonly: readonly,
        });
    }

    // Sorting descriptors by binding in order to make sure we're in the right order.
    descriptors.sort_by(|a, b| a.binding.cmp(&b.binding));

    // Computing the list of sets that are needed.
    let sets_list = descriptors.iter().map(|d| d.set).collect::<HashSet<u32>>();

    let mut output = String::new();

    // Iterate once per set.
    for &set in sets_list.iter() {
        let descr = descriptors.iter().enumerate().filter(|&(_, d)| d.set == set)
                               .map(|(_, d)| {
                                   format!("DescriptorDesc {{
                                                binding: {binding},
                                                ty: {desc_ty},
                                                array_count: 1,
                                                stages: stages.clone(),
                                                readonly: {readonly},
                                            }}", binding = d.binding, desc_ty = d.desc_ty,
                                                 readonly = if d.readonly { "true" } else { "false" })
                               })
                               .collect::<Vec<_>>();

        output.push_str(&format!(r#"
            fn set{set}_layout(stages: ShaderStages) -> VecIntoIter<DescriptorDesc> {{
                vec![
                    {descr}
                ].into_iter()
            }}
        "#, set = set, descr = descr.join(",")));
    }

    let max_set = sets_list.iter().cloned().max().map(|v| v + 1).unwrap_or(0);

    output.push_str(&format!(r#"
        pub struct Layout(ShaderStages);

        #[allow(unsafe_code)]
        unsafe impl PipelineLayoutDesc for Layout {{
            type SetsIter = VecIntoIter<Self::DescIter>;
            type DescIter = VecIntoIter<DescriptorDesc>;

            fn descriptors_desc(&self) -> Self::SetsIter {{
                vec![
                    {layouts}
                ].into_iter()
            }}
        }}

        "#, layouts = (0 .. max_set).map(|n| format!("set{}_layout(self.0)", n)).collect::<Vec<_>>().join(",")));

    output
}

/// Assumes that `variable` is a variable with a `TypePointer` and returns the id of the pointed
/// type.
fn pointer_variable_ty(doc: &parse::Spirv, variable: u32) -> u32 {
    let var_ty = doc.instructions.iter().filter_map(|i| {
        match i {
            &parse::Instruction::Variable { result_type_id, result_id, .. } if result_id == variable => {
                Some(result_type_id)
            },
            _ => None
        }
    }).next().unwrap();

    doc.instructions.iter().filter_map(|i| {
        match i {
            &parse::Instruction::TypePointer { result_id, type_id, .. } if result_id == var_ty => {
                Some(type_id)
            },
            _ => None
        }
    }).next().unwrap()
}

/// Returns a `DescriptorDescTy` constructor and a bool indicating whether the descriptor is
/// read-only.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_infos(doc: &parse::Spirv, pointed_ty: u32, force_combined_image_sampled: bool)
                    -> Option<(String, bool)>
{
    doc.instructions.iter().filter_map(|i| {
        match i {
            &parse::Instruction::TypeStruct { result_id, .. } if result_id == pointed_ty => {
                // Determine whether there's a Block or BufferBlock decoration.
                let is_ssbo = doc.instructions.iter().filter_map(|i| {
                    match i {
                        &parse::Instruction::Decorate
                            { target_id, decoration: enums::Decoration::DecorationBufferBlock, .. }
                            if target_id == pointed_ty =>
                        {
                            Some(true)
                        },
                        &parse::Instruction::Decorate
                            { target_id, decoration: enums::Decoration::DecorationBlock, .. }
                            if target_id == pointed_ty =>
                        {
                            Some(false)
                        },
                        _ => None,
                    }
                }).next().expect("Found a buffer uniform with neither the Block nor BufferBlock \
                                  decorations");

                // Determine whether there's a NonWritable decoration.
                //let non_writable = false;       // TODO: tricky because the decoration is on struct members

                let desc = format!("DescriptorDescTy::Buffer(DescriptorBufferDesc {{
                    dynamic: Some(false),
                    storage: {}
                }})", if is_ssbo { "true" } else { "false "});

                Some((desc, true))
            },

            &parse::Instruction::TypeImage { result_id, ref dim, arrayed, ms, sampled,
                                             ref format, .. } if result_id == pointed_ty =>
            {
                let sampled = sampled.expect("Vulkan requires that variables of type OpTypeImage \
                                              have a Sampled operand of 1 or 2");

                let ms = if ms { "true" } else { "false" };
                let arrayed = if arrayed {
                    "DescriptorImageDescArray::Arrayed { max_layers: None }"
                } else {
                    "DescriptorImageDescArray::NonArrayed"
                };

                if let &enums::Dim::DimSubpassData = dim {
                    // We are an input attachment.
                    assert!(!force_combined_image_sampled, "An OpTypeSampledImage can't point to \
                                                            an OpTypeImage whose dimension is \
                                                            SubpassData");
                    assert!(if let &enums::ImageFormat::ImageFormatUnknown = format { true }
                            else { false }, "If Dim is SubpassData, Image Format must be Unknown");
                    assert!(!sampled, "If Dim is SubpassData, Sampled must be 2");

                    let desc = format!("DescriptorDescTy::InputAttachment {{
                                            multisampled: {},
                                            array_layers: {}
                                        }}", ms, arrayed);

                    Some((desc, true))

                } else if let &enums::Dim::DimBuffer = dim {
                    // We are a texel buffer.
                    let desc = format!("DescriptorDescTy::TexelBuffer {{
                        storage: {},
                        format: None,       // TODO: specify format if known
                    }}", !sampled);

                    Some((desc, true))

                } else {
                    // We are a sampled or storage image.
                    let sampled = if sampled { "true" } else { "false" };
                    let ty = if force_combined_image_sampled { "CombinedImageSampler" }
                             else { "Image" };
                    let dim = match *dim {
                        enums::Dim::Dim1D => "DescriptorImageDescDimensions::OneDimensional",
                        enums::Dim::Dim2D => "DescriptorImageDescDimensions::TwoDimensional",
                        enums::Dim::Dim3D => "DescriptorImageDescDimensions::ThreeDimensional",
                        enums::Dim::DimCube => "DescriptorImageDescDimensions::Cube",
                        enums::Dim::DimRect => panic!("Vulkan doesn't support rectangle textures"),
                        _ => unreachable!()
                    };

                    let desc = format!("DescriptorDescTy::{}(DescriptorImageDesc {{
                        sampled: {},
                        dimensions: {},
                        format: None,       // TODO: specify format if known
                        multisampled: {},
                        array_layers: {},
                    }})", ty, sampled, dim, ms, arrayed);

                    Some((desc, true))
                }
            },

            &parse::Instruction::TypeSampledImage { result_id, image_type_id }
                                                                if result_id == pointed_ty =>
            {
                descriptor_infos(doc, image_type_id, true)
            },

            &parse::Instruction::TypeSampler { result_id } if result_id == pointed_ty => {
                let desc = format!("DescriptorDescTy::Sampler");
                Some((desc, true))
            },

            _ => None,      // TODO: other types
        }
    }).next()
}
