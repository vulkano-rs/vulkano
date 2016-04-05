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
        name: String,
        set: u32,
        binding: u32,
        desc_ty: String,
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
        let desc_ty = doc.instructions.iter().filter_map(|i| {
            match i {
                &parse::Instruction::TypeStruct { result_id, .. } if result_id == pointed_ty => {
                    // Determine whether there's a Block or BufferBlock decoration.
                    let is_ssbo = doc.instructions.iter().filter_map(|i| {
                        match i {
                            &parse::Instruction::Decorate { target_id, decoration: enums::Decoration::DecorationBufferBlock, .. } if target_id == pointed_ty => {
                                Some(true)
                            },
                            &parse::Instruction::Decorate { target_id, decoration: enums::Decoration::DecorationBlock, .. } if target_id == pointed_ty => {
                                Some(false)
                            },
                            _ => None,
                        }
                    }).next().expect("Found a buffer uniform with neither the Block nor BufferBlock decorations");

                    Some(if !is_ssbo {
                        "DescriptorType::UniformBuffer"
                    } else {
                        "DescriptorType::StorageBuffer"
                    })
                },
                &parse::Instruction::TypeImage { result_id, sampled_type_id, ref dim, arrayed, ms,
                                                 sampled, ref format, ref access, .. }
                                        if result_id == pointed_ty && sampled == Some(true) =>
                {
                    Some("DescriptorType::SampledImage")
                },
                &parse::Instruction::TypeImage { result_id, sampled_type_id, ref dim, arrayed, ms,
                                                 sampled, ref format, ref access, .. }
                                        if result_id == pointed_ty && sampled == Some(false) =>
                {
                    Some("DescriptorType::InputAttachment")       // FIXME: can be `StorageImage`
                },
                &parse::Instruction::TypeSampledImage { result_id, image_type_id }
                                                                    if result_id == pointed_ty =>
                {
                    Some("DescriptorType::CombinedImageSampler")
                },
                _ => None,      // TODO: other types
            }
        }).next().expect(&format!("Couldn't find relevant type for uniform `{}` (type {}, maybe unimplemented)", name, pointed_ty));

        descriptors.push(Descriptor {
            name: name,
            desc_ty: desc_ty.to_owned(),
            set: descriptor_set,
            binding: binding,
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
                               .map(|(entry, d)| {
                                   format!("DescriptorDesc {{
                                                binding: {binding},
                                                ty: {desc_ty},
                                                array_count: 1,
                                                stages: ShaderStages::all(),        // TODO:
                                            }}", binding = d.binding, desc_ty = d.desc_ty)
                               })
                               .collect::<Vec<_>>();

        output.push_str(&format!(r#"
            fn set{set}_layout() -> VecIntoIter<DescriptorDesc> {{
                vec![
                    {descr}
                ].into_iter()
            }}
        "#, set = set, descr = descr.join(",")));
    }

    let max_set = sets_list.iter().cloned().max().map(|v| v + 1).unwrap_or(0);

    output.push_str(&format!(r#"
        #[derive(Default)]
        pub struct Layout;

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

        "#, layouts = (0 .. max_set).map(|n| format!("set{}_layout()", n)).collect::<Vec<_>>().join(",")));

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
