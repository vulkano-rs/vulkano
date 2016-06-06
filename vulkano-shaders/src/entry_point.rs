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

pub fn write_entry_point(doc: &parse::Spirv, instruction: &parse::Instruction) -> (String, String) {
    let (execution, ep_name, interface) = match instruction {
        &parse::Instruction::EntryPoint { ref execution, id, ref name, ref interface } => {
            (execution, name, interface)
        },
        _ => unreachable!()
    };

    let capitalized_ep_name: String = ep_name.chars().take(1).flat_map(|c| c.to_uppercase())
                                             .chain(ep_name.chars().skip(1)).collect();

    let interface_structs = write_interface_structs(doc, &capitalized_ep_name, interface);

    let (ty, f_call) = match *execution {
        enums::ExecutionModel::ExecutionModelVertex => {
            let t = format!("::vulkano::pipeline::shader::VertexShaderEntryPoint<(), {0}Input, {0}Output, Layout>", capitalized_ep_name);
            let f = format!("vertex_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), {0}Input, {0}Output, Layout(ShaderStages {{ vertex: true, .. ShaderStages::none() }}))", capitalized_ep_name);
            (t, f)
        },

        enums::ExecutionModel::ExecutionModelTessellationControl => {
            let t = format!("::vulkano::pipeline::shader::TessControlShaderEntryPoint<(), {0}Input, {0}Output, Layout>", capitalized_ep_name);
            let f = format!("tess_control_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), {0}Input, {0}Output, Layout(ShaderStages {{ tessellation_control: true, .. ShaderStages::none() }}))", capitalized_ep_name);
            (t, f)
        },

        enums::ExecutionModel::ExecutionModelTessellationEvaluation => {
            let t = format!("::vulkano::pipeline::shader::TessEvaluationShaderEntryPoint<(), {0}Input, {0}Output, Layout>", capitalized_ep_name);
            let f = format!("tess_evaluation_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), {0}Input, {0}Output, Layout(ShaderStages {{ tessellation_evaluation: true, .. ShaderStages::none() }}))", capitalized_ep_name);
            (t, f)
        },

        enums::ExecutionModel::ExecutionModelGeometry => {
            let t = format!("::vulkano::pipeline::shader::GeometryShaderEntryPoint<(), {0}Input, {0}Output, Layout>", capitalized_ep_name);
            let f = format!("geometry_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), {0}Input, {0}Output, Layout(ShaderStages {{ geometry: true, .. ShaderStages::none() }}))", capitalized_ep_name);
            (t, f)
        },

        enums::ExecutionModel::ExecutionModelFragment => {
            let t = format!("::vulkano::pipeline::shader::FragmentShaderEntryPoint<(), {0}Input, {0}Output, Layout>", capitalized_ep_name);
            let f = format!("fragment_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), {0}Input, {0}Output, Layout(ShaderStages {{ fragment: true, .. ShaderStages::none() }}))", capitalized_ep_name);
            (t, f)
        },

        enums::ExecutionModel::ExecutionModelGLCompute => {
            (format!("::vulkano::pipeline::shader::ComputeShaderEntryPoint<(), Layout>"),
             format!("compute_shader_entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _), Layout(ShaderStages {{ compute: true, .. ShaderStages::none() }}))"))
        },

        enums::ExecutionModel::ExecutionModelKernel => panic!("Kernels are not supported"),
    };

    let entry_point = format!(r#"
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
                f_call = f_call);

    (interface_structs, entry_point)
}

fn write_interface_structs(doc: &parse::Spirv, capitalized_ep_name: &str, interface: &[u32])
                           -> String
{
    let mut input_elements = Vec::new();
    let mut output_elements = Vec::new();

    // Filling `input_elements` and `output_elements`.
    for interface in interface.iter() {
        for i in doc.instructions.iter() {
            match i {
                &parse::Instruction::Variable { result_type_id, result_id, ref storage_class, .. }
                                              if &result_id == interface =>
                {
                    if is_builtin(doc, result_id) {
                        continue;
                    }

                    if is_builtin(doc, result_type_id) {
                        continue;
                    }

                    let to_write = match storage_class {
                        &enums::StorageClass::StorageClassInput => &mut input_elements,
                        &enums::StorageClass::StorageClassOutput => &mut output_elements,
                        _ => continue
                    };

                    let name = name_from_id(doc, result_id);
                    if name == "__unnamed" { continue; }        // FIXME: hack

                    let loc = match location_decoration(doc, result_id) {
                        Some(l) => l,
                        None => panic!("Attribute `{}` (id {}) is missing a location", name, result_id)
                    };

                    to_write.push((loc, name, format_from_id(doc, result_type_id)));
                },
                _ => ()
            }
        }
    }

    write_interface_struct(&format!("{}Input", capitalized_ep_name), &input_elements) +
        &write_interface_struct(&format!("{}Output", capitalized_ep_name), &output_elements)
}

fn write_interface_struct(struct_name: &str, attributes: &[(u32, String, (String, usize))]) -> String {
    // Checking for overlapping elements.
    for (offset, &(loc, ref name, (_, loc_len))) in attributes.iter().enumerate() {
        for &(loc2, ref name2, (_, loc_len2)) in attributes.iter().skip(offset + 1) {
            if loc == loc2 || (loc < loc2 && loc + loc_len as u32 > loc2) ||
               (loc2 < loc && loc2 + loc_len2 as u32 > loc)
            {
                panic!("The locations of attributes `{}` (start={}, size={}) \
                        and `{}` (start={}, size={}) overlap",
                        name, loc, loc_len, name2, loc2, loc_len2);
            }
        }
    }

    let body = attributes.iter().enumerate().map(|(num, &(loc, ref name, (ref ty, num_locs)))| {
        assert!(num_locs >= 1);

        format!("if self.num == {} {{
            self.num += 1;

            return Some(::vulkano::pipeline::shader::ShaderInterfaceDefEntry {{
                location: {} .. {},
                format: ::vulkano::format::Format::{},
                name: Some(::std::borrow::Cow::Borrowed(\"{}\"))
            }});
        }}", num, loc, loc as usize + num_locs, ty, name)
    }).collect::<Vec<_>>().join("");

    format!("
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        pub struct {name};

        unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for {name} {{
            type Iter = {name}Iter;
            fn elements(&self) -> {name}Iter {{
                {name}Iter {{ num: 0 }}
            }}
        }}

        #[derive(Debug, Copy, Clone)]
        pub struct {name}Iter {{ num: u16 }}
        impl Iterator for {name}Iter {{
            type Item = ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {{
                {body}
                None
            }}

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {{
                let len = ({len} - self.num) as usize;
                (len, Some(len))
            }}
        }}

        impl ExactSizeIterator for {name}Iter {{}}
    ", name = struct_name, body = body, len = attributes.len())
}
