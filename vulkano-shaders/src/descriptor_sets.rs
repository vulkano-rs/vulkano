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
    // TODO: not implemented

    // finding all the descriptors
    let mut descriptors = Vec::new();
    struct Descriptor {
        name: String,
        desc_ty: String,
        bind_ty: String,
        bind_template_params: Vec<String>,
        bind_where_clauses: String,
        bind: String,
        set: u32,
        binding: u32,
    }

    // template parameter names that are available for the purpose of the code below
    // TODO: better system
    let mut template_params = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"].into_iter();

    // looping to find all the elements that have the `DescriptorSet` decoration
    for instruction in doc.instructions.iter() {
        let (variable_id, descriptor_set) = match instruction {
            &parse::Instruction::Decorate { target_id, decoration: enums::Decoration::DecorationDescriptorSet, ref params } => {
                (target_id, params[0])
            },
            _ => continue
        };

        // find which type is pointed to by this variable
        let pointed_ty = pointer_variable_ty(doc, variable_id);
        // name of the variable
        let name = ::name_from_id(doc, variable_id);

        // find the binding point of this descriptor
        let binding = doc.instructions.iter().filter_map(|i| {
            match i {
                &parse::Instruction::Decorate { target_id, decoration: enums::Decoration::DecorationBinding, ref params } if target_id == variable_id => {
                    Some(params[0])
                },
                _ => None,      // TODO: other types
            }
        }).next().expect(&format!("Uniform `{}` is missing a binding", name));

        // find informations about the kind of binding for this descriptor
        let (desc_ty, bind_template_params, bind_where_clauses, bind_ty, bind) = doc.instructions.iter().filter_map(|i| {
            match i {
                &parse::Instruction::TypeStruct { result_id, .. } if result_id == pointed_ty => {
                    let tp_buffer = template_params.next().unwrap();

                    Some((
                        "::vulkano::descriptor_set::DescriptorType::UniformBuffer",
                        vec![tp_buffer.to_owned()],
                        format!("{}: 'static + ::vulkano::buffer::TypedBuffer", tp_buffer),
                        format!("&'a ::std::sync::Arc<{}>", tp_buffer),
                        "::vulkano::descriptor_set::DescriptorBind::uniform_buffer(data)"
                    ))
                },
                &parse::Instruction::TypeImage { result_id, sampled_type_id, ref dim, arrayed, ms,
                                                 sampled, ref format, ref access, .. }
                                        if result_id == pointed_ty && sampled == Some(true) =>
                {
                    let img = template_params.next().unwrap();

                    Some((
                        "::vulkano::descriptor_set::DescriptorType::SampledImage",
                        vec![img.to_owned()],
                        format!("{}: 'static + ::vulkano::image::ImageView", img),
                        format!("&'a ::std::sync::Arc<{}>", img),
                        "::vulkano::descriptor_set::DescriptorBind::sampled_image(data)"
                    ))
                },
                &parse::Instruction::TypeImage { result_id, sampled_type_id, ref dim, arrayed, ms,
                                                 sampled, ref format, ref access, .. }
                                        if result_id == pointed_ty && sampled == Some(false) =>
                {
                    let img = template_params.next().unwrap();

                    Some((
                        "::vulkano::descriptor_set::DescriptorType::InputAttachment",       // FIXME: can be `StorageImage`
                        vec![img.to_owned()],
                        format!("{}: 'static + ::vulkano::image::ImageView", img),
                        format!("&'a ::std::sync::Arc<{}>", img),
                        "::vulkano::descriptor_set::DescriptorBind::input_attachment(data)"
                    ))
                },
                &parse::Instruction::TypeSampledImage { result_id, image_type_id }
                                                                    if result_id == pointed_ty =>
                {
                    let img = template_params.next().unwrap();

                    Some((
                        "::vulkano::descriptor_set::DescriptorType::CombinedImageSampler",
                        vec![img.to_owned()],
                        format!("{}: 'static + ::vulkano::image::ImageView", img),
                        format!("(&'a ::std::sync::Arc<::vulkano::sampler::Sampler>, &'a ::std::sync::Arc<{}>)", img),
                        "::vulkano::descriptor_set::DescriptorBind::combined_image_sampler(data.0, data.1)"
                    ))
                },
                _ => None,      // TODO: other types
            }
        }).next().expect(&format!("Couldn't find relevant type for uniform `{}` (type {}, maybe unimplemented)", name, pointed_ty));

        descriptors.push(Descriptor {
            name: name,
            desc_ty: desc_ty.to_owned(),
            bind_ty: bind_ty,
            bind_template_params: bind_template_params,
            bind_where_clauses: bind_where_clauses,
            bind: bind.to_owned(),
            set: descriptor_set,
            binding: binding,
        });
    }

    let sets_list = descriptors.iter().map(|d| d.set).collect::<HashSet<u32>>();

    let mut output = String::new();

    // iterate once per set that is defined somewhere
    for set in sets_list.iter() {
        let write_ty = descriptors.iter().filter(|d| d.set == *set)
                                  .map(|d| d.bind_ty.clone())
                                  .collect::<Vec<_>>();

        let write_tp = {
            let v = descriptors.iter().filter(|d| d.set == *set)
                               .map(|d| d.bind_template_params.join(", "))
                               .collect::<Vec<_>>().join(", ");
            if v.is_empty() { v } else { ", ".to_owned() + &v }
        };

        let write_where = {
            let v = descriptors.iter().filter(|d| d.set == *set)
                               .map(|d| d.bind_where_clauses.clone())
                               .collect::<Vec<_>>().join(", ");
            if v.is_empty() { v } else { "where ".to_owned() + &v }
        };

        let writes = descriptors.iter().enumerate().filter(|&(_, d)| d.set == *set)
                                .map(|(entry, d)| {
                                    let entry = if write_ty.len() == 1 {
                                        "".to_owned()
                                    } else {
                                        format!(".{}", entry)
                                    };

                                    format!("{{ let data = data{entry};
                                            ::vulkano::descriptor_set::DescriptorWrite {{
                                                 binding: {binding},
                                                 array_element: 0,
                                                 content: {bind},
                                             }} }}", binding = d.binding, bind = d.bind,
                                                     entry = entry)
                                })
                                .collect::<Vec<_>>();

        let write_ty = if write_ty.len() == 1 {
            write_ty.into_iter().next().unwrap()
        } else {
            format!("({})", write_ty.join(", "))
        };

        let descr = descriptors.iter().enumerate().filter(|&(_, d)| d.set == *set)
                               .map(|(entry, d)| {
                                   format!("::vulkano::descriptor_set::DescriptorDesc {{
                                                binding: {binding},
                                                ty: {desc_ty},
                                                array_count: 1,
                                                stages: ::vulkano::descriptor_set::ShaderStages::all(),        // TODO:
                                            }}", binding = d.binding, desc_ty = d.desc_ty)
                               })
                               .collect::<Vec<_>>();

        output.push_str(&format!(r#"
#[derive(Default)]
pub struct Set{set};

unsafe impl ::vulkano::descriptor_set::SetLayout for Set{set} {{
    fn descriptors(&self) -> Vec<::vulkano::descriptor_set::DescriptorDesc> {{
        vec![
            {descr}
        ]
    }}
}}

unsafe impl<'a {write_tp}> ::vulkano::descriptor_set::SetLayoutWrite<{write_ty}> for Set{set}
    {write_where}
{{
    fn decode(&self, data: {write_ty}) -> Vec<::vulkano::descriptor_set::DescriptorWrite> {{
        vec![
            {writes}
        ]
    }}
}}

unsafe impl<'a {write_tp}> ::vulkano::descriptor_set::SetLayoutInit<{write_ty}> for Set{set}
    {write_where}
{{
    fn decode(&self, data: {write_ty}) -> Vec<::vulkano::descriptor_set::DescriptorWrite> {{
        ::vulkano::descriptor_set::SetLayoutWrite::decode(self, data)
    }}
}}

"#, set = set, write_ty = write_ty, writes = writes.join(","), write_tp = write_tp, 
    write_where = write_where, descr = descr.join(",")));
    }

    let max_set = sets_list.iter().cloned().max().map(|v| v + 1).unwrap_or(0);

    let sets_defs = (0 .. max_set).map(|num| {
        if sets_list.contains(&num) {
            format!("::std::sync::Arc<::vulkano::descriptor_set::DescriptorSet<Set{}>>", num)
        } else {
            "()".to_owned()
        }
    }).collect::<Vec<_>>();

    let sets = (0 .. max_set).map(|num| {
        if sets_list.contains(&num) {
            if sets_defs.len() == 1 {
                format!("sets")
            } else {
                format!("sets.{}", num)
            }
        } else {
            "()".to_owned()     // FIXME: wrong
        }
    }).collect::<Vec<_>>();

    let layouts_defs = (0 .. max_set).map(|num| {
        if sets_list.contains(&num) {
            format!("::std::sync::Arc<::vulkano::descriptor_set::DescriptorSetLayout<Set{}>>", num)
        } else {
            "()".to_owned()
        }
    }).collect::<Vec<_>>();

    let layouts = (0 .. max_set).map(|num| {
        if sets_list.contains(&num) {
            if layouts_defs.len() == 1 {
                format!("layouts")
            } else {
                format!("layouts.{}", num)
            }
        } else {
            "()".to_owned()     // FIXME: wrong
        }
    }).collect::<Vec<_>>();

    output.push_str(&format!(r#"
#[derive(Default)]
pub struct Layout;

unsafe impl ::vulkano::descriptor_set::Layout for Layout {{
    type DescriptorSets = ({sets_defs});
    type DescriptorSetLayouts = ({layouts_defs});
    type PushConstants = ();

    fn decode_descriptor_set_layouts(&self, layouts: Self::DescriptorSetLayouts)
        -> Vec<::std::sync::Arc<::vulkano::descriptor_set::AbstractDescriptorSetLayout>>
    {{
        vec![
            {layouts}
        ]
    }}

    fn decode_descriptor_sets(&self, sets: Self::DescriptorSets)
        -> Vec<::std::sync::Arc<::vulkano::descriptor_set::AbstractDescriptorSet>>
    {{
        vec![
            {sets}
        ]
    }}
}}
"#, sets_defs = sets_defs.join(","), layouts_defs = layouts_defs.join(","),
    layouts = layouts.join(","), sets = sets.join(",")));

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
