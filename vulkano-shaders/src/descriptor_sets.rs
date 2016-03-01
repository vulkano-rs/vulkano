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
        bind_start: String,
        bind_end: String,
        set: u32,
        binding: u32,
    }

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
        }).next().expect("A uniform is missing a binding");

        // find informations about the kind of binding for this descriptor
        let (desc_ty, bind_ty, bind_start, bind_end) = doc.instructions.iter().filter_map(|i| {
            match i {
                &parse::Instruction::TypeStruct { result_id, .. } if result_id == pointed_ty => {
                    Some((
                        "::vulkano::descriptor_set::DescriptorType::UniformBuffer",
                        "::vulkano::buffer::AbstractBuffer",
                        "::vulkano::descriptor_set::DescriptorBind::UniformBuffer { buffer: ",
                        ", offset: 0, size: 128 /* FIXME */ }"
                    ))
                },
                &parse::Instruction::TypeImage { result_id, sampled_type_id, ref dim, arrayed, ms,
                                                 sampled, ref format, ref access, .. }
                                        if result_id == pointed_ty && sampled == Some(true) =>
                {
                    Some((
                        "::vulkano::descriptor_set::DescriptorType::SampledImage",
                        "::vulkano::image::AbstractImageView",
                        "::vulkano::descriptor_set::DescriptorBind::UniformBuffer",      // FIXME:
                        ""
                    ))
                },
                &parse::Instruction::TypeSampledImage { result_id, image_type_id }
                                                                    if result_id == pointed_ty =>
                {
                    Some((
                        "::vulkano::descriptor_set::DescriptorType::SampledImage",
                        "::vulkano::image::AbstractImageView",
                        "::vulkano::descriptor_set::DescriptorBind::UniformBuffer",      // FIXME:
                        ""
                    ))
                },
                _ => None,      // TODO: other types
            }
        }).next().expect(&format!("Couldn't find relevant type for uniform `{}` (maybe unimplemented)", name));

        descriptors.push(Descriptor {
            name: name,
            desc_ty: desc_ty.to_owned(),
            bind_ty: bind_ty.to_owned(),
            bind_start: bind_start.to_owned(),
            bind_end: bind_end.to_owned(),
            set: descriptor_set,
            binding: binding,
        });
    }

    let sets_list = descriptors.iter().map(|d| d.set).collect::<HashSet<u32>>();

    let mut output = String::new();

    // iterate once per set that is defined somewhere
    for set in sets_list.iter() {
        let write_ty = descriptors.iter().filter(|d| d.set == *set)
                                  .map(|d| format!("::std::sync::Arc<{}>", d.bind_ty))
                                  .collect::<Vec<_>>();

        let writes = descriptors.iter().enumerate().filter(|&(_, d)| d.set == *set)
                                .map(|(entry, d)| {
                                    let entry = if write_ty.len() == 1 {
                                        "".to_owned()
                                    } else {
                                        format!(".{}", entry)
                                    };

                                    format!("::vulkano::descriptor_set::DescriptorWrite {{
                                                 binding: {binding},
                                                 array_element: 0,
                                                 content: {bind_start} data{entry} {bind_end},
                                             }}", binding = d.binding, bind_start = d.bind_start,
                                                  bind_end = d.bind_end,
                                                  entry = entry)
                                })
                                .collect::<Vec<_>>();

        let descr = descriptors.iter().enumerate().filter(|&(_, d)| d.set == *set)
                               .map(|(entry, d)| {
                                   format!("::vulkano::descriptor_set::DescriptorDesc {{
                                                binding: {binding},
                                                ty: {desc_ty},
                                                array_count: 1,
                                                stages: ::vulkano::descriptor_set::ShaderStages::all_graphics(),        // TODO:
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

unsafe impl ::vulkano::descriptor_set::SetLayoutWrite<{write_ty}> for Set{set} {{
    fn decode(&self, data: {write_ty}) -> Vec<::vulkano::descriptor_set::DescriptorWrite> {{
        vec![
            {writes}
        ]
    }}
}}

unsafe impl ::vulkano::descriptor_set::SetLayoutInit<{write_ty}> for Set{set} {{
    fn decode(&self, data: {write_ty}) -> Vec<::vulkano::descriptor_set::DescriptorWrite> {{
        ::vulkano::descriptor_set::SetLayoutWrite::decode(self, data)
    }}
}}

"#, set = set, write_ty = write_ty.join(","), writes = writes.join(","), descr = descr.join(",")));
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
