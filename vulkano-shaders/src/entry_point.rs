// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use enums;
use parse;

use is_builtin;
use name_from_id;
use location_decoration;
use type_from_id;
use format_from_id;

pub fn write_entry_point(doc: &parse::Spirv, instruction: &parse::Instruction) -> String {
    let (execution, ep_name, interface) = match instruction {
        &parse::Instruction::EntryPoint { ref execution, id, ref name, ref interface } => {
            (execution, name, interface)
        },
        _ => unreachable!()
    };

    let (ty, f_call) = match *execution {
        enums::ExecutionModel::ExecutionModelVertex => {
            let mut attributes = Vec::new();

            for interface in interface.iter() {
                for i in doc.instructions.iter() {
                    match i {
                        &parse::Instruction::Variable { result_type_id, result_id,
                                    storage_class: enums::StorageClass::StorageClassInput, .. }
                                    if &result_id == interface =>
                        {
                            if is_builtin(doc, result_id) {
                                continue;
                            }

                            let name = name_from_id(doc, result_id);
                            let loc = match location_decoration(doc, result_id) {
                                Some(l) => l,
                                None => panic!("vertex attribute `{}` is missing a location", name)
                            };
                            attributes.push((loc, name, format_from_id(doc, result_type_id)));
                        },
                        _ => ()
                    }
                }
            }

            // Checking for overlapping attributes.
            for (offset, &(loc, ref name, (_, loc_len))) in attributes.iter().enumerate() {
                for &(loc2, ref name2, (_, loc_len2)) in attributes.iter().skip(offset + 1) {
                    if loc == loc2 || (loc < loc2 && loc + loc_len as u32 > loc2) ||
                       (loc2 < loc && loc2 + loc_len2 as u32 > loc)
                    {
                        panic!("The locations of vertex attributes `{}` and `{}` overlap",
                               name, name2);
                    }
                }
            }

            let attributes = attributes.iter().map(|&(loc, ref name, (ref ty, num_locs))| {
                assert!(num_locs >= 1);

                format!("::vulkano::pipeline::shader::ShaderInterfaceDefEntry {{
                    location: {} .. {},
                    format: ::vulkano::format::Format::{},
                    name: Some(::std::borrow::Cow::Borrowed(\"{}\"))
                }}", loc, loc as usize + num_locs, ty, name)
            }).collect::<Vec<_>>().join(", ");

            let t = "::vulkano::pipeline::shader::VertexShaderEntryPoint<(), Vec<::vulkano::pipeline::shader::ShaderInterfaceDefEntry>, Layout>".to_owned();
            let f = format!("vertex_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), Layout, vec![{}])", attributes);
            (t, f)
        },

        enums::ExecutionModel::ExecutionModelTessellationControl => {
            (format!("::vulkano::pipeline::shader::TessControlShaderEntryPoint"), String::new())
        },

        enums::ExecutionModel::ExecutionModelTessellationEvaluation => {
            (format!("::vulkano::pipeline::shader::TessEvaluationShaderEntryPoint"), String::new())
        },

        enums::ExecutionModel::ExecutionModelGeometry => {
            (format!("::vulkano::pipeline::shader::GeometryShaderEntryPoint"), String::new())
        },

        enums::ExecutionModel::ExecutionModelFragment => {
            let mut output_types = Vec::new();

            for interface in interface.iter() {
                for i in doc.instructions.iter() {
                    match i {
                        &parse::Instruction::Variable { result_type_id, result_id,
                                    storage_class: enums::StorageClass::StorageClassOutput, .. }
                                    if &result_id == interface =>
                        {
                            output_types.push(type_from_id(doc, result_type_id));
                        },
                        _ => ()
                    }
                }
            }

            let output = {
                let output = output_types.join(", ");
                if output.is_empty() { output } else { output + "," }
            };

            let t = format!("::vulkano::pipeline::shader::FragmentShaderEntryPoint<(), ({output}), Layout>",
                            output = output);
            (t, format!("fragment_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), Layout)"))
        },

        enums::ExecutionModel::ExecutionModelGLCompute => {
            (format!("::vulkano::pipeline::shader::ComputeShaderEntryPoint<(), Layout>"),
             format!("compute_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), Layout)"))
        },

        enums::ExecutionModel::ExecutionModelKernel => panic!("Kernels are not supported"),
    };

    format!(r#"
    /// Returns a logical struct describing the entry point named `{ep_name}`.
    #[inline]
    pub fn {ep_name}_entry_point(&self) -> {ty} {{
        unsafe {{
            #[allow(dead_code)]
            static NAME: [u8; {ep_name_lenp1}] = [{encoded_ep_name}, 0];     // "{ep_name}"
            self.shader.{f_call}
        }}
    }}
            "#, ep_name = ep_name, ep_name_lenp1 = ep_name.chars().count() + 1, ty = ty,
                encoded_ep_name = ep_name.chars().map(|c| (c as u32).to_string())
                                         .collect::<Vec<String>>().join(", "),
                f_call = f_call)
}
