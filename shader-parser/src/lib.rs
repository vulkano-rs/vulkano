use std::io::Error as IoError;
use std::io::Read;

pub use parse::ParseError;

mod parse;

pub fn reflect<R>(mut spirv: R) -> Result<String, Error>
    where R: Read
{
    let mut data = Vec::new();
    try!(spirv.read_to_end(&mut data));

    // now parsing the document
    let doc = try!(parse::parse_spirv(&data));

    let mut output = String::new();

    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::Variable { result_type_id, result_id, ref storage_class, .. } => {
                match *storage_class {
                    parse::StorageClass::UniformConstant => (),
                    parse::StorageClass::Uniform => (),
                    parse::StorageClass::PushConstant => (),
                    _ => continue
                };

                let name = name_from_id(&doc, result_id);
                output.push_str(&format!("{}: {};", name, type_from_id(&doc, result_type_id)));
            },
            _ => ()
        }
    }

    Ok(output)
}

fn name_from_id(doc: &parse::Spirv, searched: u32) -> String {
    doc.instructions.iter().filter_map(|i| {
        if let &parse::Instruction::Name { target_id, ref name } = i {
            if target_id == searched {
                Some(name.clone())
            } else {
                None
            }
        } else {
            None
        }
    }).next().and_then(|n| if !n.is_empty() { Some(n) } else { None })
      .unwrap_or("__unnamed".to_owned())
}

fn member_name_from_id(doc: &parse::Spirv, searched: u32, searched_member: u32) -> String {
    doc.instructions.iter().filter_map(|i| {
        if let &parse::Instruction::MemberName { target_id, member, ref name } = i {
            if target_id == searched && member == searched_member {
                Some(name.clone())
            } else {
                None
            }
        } else {
            None
        }
    }).next().and_then(|n| if !n.is_empty() { Some(n) } else { None })
      .unwrap_or("__unnamed".to_owned())
}

fn type_from_id(doc: &parse::Spirv, searched: u32) -> String {
    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::TypeVoid { result_id } if result_id == searched => {
                return "()".to_owned()
            },
            &parse::Instruction::TypeBool { result_id } if result_id == searched => {
                return "bool".to_owned()
            },
            &parse::Instruction::TypeInt { result_id, width, signedness } if result_id == searched => {
                return "i32".to_owned()
            },
            &parse::Instruction::TypeFloat { result_id, width } if result_id == searched => {
                return "f32".to_owned()
            },
            &parse::Instruction::TypeVector { result_id, component_id, count } if result_id == searched => {
                let t = type_from_id(doc, component_id);
                return format!("[{}; {}]", t, count);
            },
            &parse::Instruction::TypeArray { result_id, type_id, length_id } if result_id == searched => {
                let t = type_from_id(doc, type_id);
                return format!("[{}; {}]", t, length_id);       // FIXME:
            },
            &parse::Instruction::TypeRuntimeArray { result_id, type_id } if result_id == searched => {
                let t = type_from_id(doc, type_id);
                return format!("[{}]", t);
            },
            &parse::Instruction::TypeStruct { result_id, ref member_types } if result_id == searched => {
                let name = name_from_id(doc, result_id);
                let members = member_types.iter().enumerate().map(|(offset, &member)| {
                    let ty = type_from_id(doc, member);
                    let name = member_name_from_id(doc, result_id, offset as u32);
                    format!("\t{}: {}", name, ty)
                }).collect::<Vec<_>>();
                return format!("struct {} {{\n{}\n}}", name, members.join(",\n"));
            },
            &parse::Instruction::TypeOpaque { result_id, ref name } if result_id == searched => {
            },
            &parse::Instruction::TypePointer { result_id, type_id, .. } if result_id == searched => {
                let t = type_from_id(doc, type_id);
                return format!("*const {}", t);
            },
            _ => ()
        }
    }

    panic!("Type #{} not found", searched)
}

#[derive(Debug)]
pub enum Error {
    IoError(IoError),
    ParseError(ParseError),
}

impl From<IoError> for Error {
    #[inline]
    fn from(err: IoError) -> Error {
        Error::IoError(err)
    }
}

impl From<ParseError> for Error {
    #[inline]
    fn from(err: ParseError) -> Error {
        Error::ParseError(err)
    }
}
