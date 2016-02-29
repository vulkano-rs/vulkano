use parse;
use enums;

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub fn write_structs(doc: &parse::Spirv) -> String {
    let mut result = String::new();

    for instruction in &doc.instructions {
        match *instruction {
            parse::Instruction::TypeStruct { result_id, ref member_types } => {
                result.push_str(&write_struct(doc, result_id, member_types));
                result.push_str("\n");
            },
            _ => ()
        }
    }

    result
}

/// Writes a single struct.
fn write_struct(doc: &parse::Spirv, struct_id: u32, members: &[u32]) -> String {
    let name = ::name_from_id(doc, struct_id);

    let mut members_defs = Vec::with_capacity(members.len());

    for (num, &member) in members.iter().enumerate() {
        let ty = ::type_from_id(doc, member);
        let member_name = ::member_name_from_id(doc, struct_id, num as u32);

        // Ignore the field if it is built in, which includes `gl_Position` for example.
        if is_builtin_member(doc, struct_id, num as u32) {
            continue;
        }

        let offset = doc.instructions.iter().filter_map(|i| {
            match *i {
                parse::Instruction::MemberDecorate { target_id, member,
                                                   decoration: enums::Decoration::DecorationOffset,
                                                   ref params } if target_id == struct_id &&
                                                                   member as usize == num =>
                {
                    return Some(params[0]);
                },
                _ => ()
            };

            None
        }).next().expect(&format!("Struct member `{}` of `{}` is missing an `Offset` decoration",
                                  member_name, name));

        members_defs.push(format!("{name}: {ty} /* {off} */", name = member_name, ty = ty, off = offset));
    }

    // GLSL doesn't allow empty structures. However it is possible to have structures entirely
    // made of built-in members, hence this check.
    if !members_defs.is_empty() {
        format!("#[repr(C)]\n#[derive(Copy, Clone, Debug, Default)]\n\
                 pub struct {name} {{\n\t{members}\n}}\n",
                name = name, members = members_defs.join(",\n\t"))
    } else {
        String::new()
    }
}

/// Returns true if a `BuiltIn` decorator is applied on a struct member.
fn is_builtin_member(doc: &parse::Spirv, id: u32, member_id: u32) -> bool {
    let mut result = String::new();

    for instruction in &doc.instructions {
        match *instruction {
            parse::Instruction::MemberDecorate { target_id, member,
                                                 decoration: enums::Decoration::DecorationBuiltIn,
                                                 .. } if target_id == id && member == member_id =>
            {
                return true;
            },
            _ => ()
        }
    }

    false
}
