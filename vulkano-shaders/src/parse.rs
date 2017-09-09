// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use enums::*;

/// Parses a SPIR-V document.
pub fn parse_spirv(data: &[u8]) -> Result<Spirv, ParseError> {
    if data.len() < 20 {
        return Err(ParseError::MissingHeader);
    }

    // we need to determine whether we are in big endian order or little endian order depending
    // on the magic number at the start of the file
    let data = if data[0] == 0x07 && data[1] == 0x23 && data[2] == 0x02 && data[3] == 0x03 {
        // big endian
        data.chunks(4)
            .map(|c| {
                     ((c[0] as u32) << 24) | ((c[1] as u32) << 16) | ((c[2] as u32) << 8) |
                         c[3] as u32
                 })
            .collect::<Vec<_>>()

    } else if data[3] == 0x07 && data[2] == 0x23 && data[1] == 0x02 && data[0] == 0x03 {
        // little endian
        data.chunks(4)
            .map(|c| {
                     ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) |
                         c[0] as u32
                 })
            .collect::<Vec<_>>()

    } else {
        return Err(ParseError::MissingHeader);
    };

    parse_u32s(&data)
}

/// Parses a SPIR-V document from a list of u32s.
///
/// Endianess has already been handled.
fn parse_u32s(i: &[u32]) -> Result<Spirv, ParseError> {
    if i.len() < 5 {
        return Err(ParseError::MissingHeader);
    }

    if i[0] != 0x07230203 {
        return Err(ParseError::WrongHeader);
    }

    let version = (((i[1] & 0x00ff0000) >> 16) as u8, ((i[1] & 0x0000ff00) >> 8) as u8);

    let instructions = {
        let mut ret = Vec::new();
        let mut i = &i[5 ..];
        while i.len() >= 1 {
            let (instruction, rest) = parse_instruction(i)?;
            ret.push(instruction);
            i = rest;
        }
        ret
    };

    Ok(Spirv {
           version: version,
           bound: i[3],
           instructions: instructions,
       })
}

/// Error that can happen when parsing.
#[derive(Debug, Clone)]
pub enum ParseError {
    MissingHeader,
    WrongHeader,
    IncompleteInstruction,
    UnknownConstant(&'static str, u32),
}

#[derive(Debug, Clone)]
pub struct Spirv {
    pub version: (u8, u8),
    pub bound: u32,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Unknown(u16, Vec<u32>),
    Nop,
    Name { target_id: u32, name: String },
    MemberName {
        target_id: u32,
        member: u32,
        name: String,
    },
    ExtInstImport { result_id: u32, name: String },
    MemoryModel(AddressingModel, MemoryModel),
    EntryPoint {
        execution: ExecutionModel,
        id: u32,
        name: String,
        interface: Vec<u32>,
    },
    ExecutionMode {
        target_id: u32,
        mode: ExecutionMode,
        optional_literals: Vec<u32>,
    },
    Capability(Capability),
    TypeVoid { result_id: u32 },
    TypeBool { result_id: u32 },
    TypeInt {
        result_id: u32,
        width: u32,
        signedness: bool,
    },
    TypeFloat { result_id: u32, width: u32 },
    TypeVector {
        result_id: u32,
        component_id: u32,
        count: u32,
    },
    TypeMatrix {
        result_id: u32,
        column_type_id: u32,
        column_count: u32,
    },
    TypeImage {
        result_id: u32,
        sampled_type_id: u32,
        dim: Dim,
        depth: Option<bool>,
        arrayed: bool,
        ms: bool,
        sampled: Option<bool>,
        format: ImageFormat,
        access: Option<AccessQualifier>,
    },
    TypeSampler { result_id: u32 },
    TypeSampledImage { result_id: u32, image_type_id: u32 },
    TypeArray {
        result_id: u32,
        type_id: u32,
        length_id: u32,
    },
    TypeRuntimeArray { result_id: u32, type_id: u32 },
    TypeStruct {
        result_id: u32,
        member_types: Vec<u32>,
    },
    TypeOpaque { result_id: u32, name: String },
    TypePointer {
        result_id: u32,
        storage_class: StorageClass,
        type_id: u32,
    },
    Constant {
        result_type_id: u32,
        result_id: u32,
        data: Vec<u32>,
    },
    SpecConstantTrue { result_type_id: u32, result_id: u32 },
    SpecConstantFalse { result_type_id: u32, result_id: u32 },
    SpecConstant {
        result_type_id: u32,
        result_id: u32,
        data: Vec<u32>,
    },
    SpecConstantComposite {
        result_type_id: u32,
        result_id: u32,
        data: Vec<u32>,
    },
    FunctionEnd,
    Variable {
        result_type_id: u32,
        result_id: u32,
        storage_class: StorageClass,
        initializer: Option<u32>,
    },
    Decorate {
        target_id: u32,
        decoration: Decoration,
        params: Vec<u32>,
    },
    MemberDecorate {
        target_id: u32,
        member: u32,
        decoration: Decoration,
        params: Vec<u32>,
    },
    Label { result_id: u32 },
    Branch { result_id: u32 },
    Kill,
    Return,
}

fn parse_instruction(i: &[u32]) -> Result<(Instruction, &[u32]), ParseError> {
    assert!(i.len() >= 1);

    let word_count = (i[0] >> 16) as usize;
    assert!(word_count >= 1);
    let opcode = (i[0] & 0xffff) as u16;

    if i.len() < word_count {
        return Err(ParseError::IncompleteInstruction);
    }

    let opcode = decode_instruction(opcode, &i[1 .. word_count])?;
    Ok((opcode, &i[word_count ..]))
}

fn decode_instruction(opcode: u16, operands: &[u32]) -> Result<Instruction, ParseError> {
    Ok(match opcode {
           0 => Instruction::Nop,
           5 => Instruction::Name {
               target_id: operands[0],
               name: parse_string(&operands[1 ..]).0,
           },
           6 => Instruction::MemberName {
               target_id: operands[0],
               member: operands[1],
               name: parse_string(&operands[2 ..]).0,
           },
           11 => Instruction::ExtInstImport {
               result_id: operands[0],
               name: parse_string(&operands[1 ..]).0,
           },
           14 => Instruction::MemoryModel(AddressingModel::from_num(operands[0])?,
                                          MemoryModel::from_num(operands[1])?),
           15 => {
               let (n, r) = parse_string(&operands[2 ..]);
               Instruction::EntryPoint {
                   execution: ExecutionModel::from_num(operands[0])?,
                   id: operands[1],
                   name: n,
                   interface: r.to_owned(),
               }
           },
           16 => {
               Instruction::ExecutionMode {
                   target_id: operands[0],
                   mode: ExecutionMode::from_num(operands[1])?,
                   optional_literals: operands[2 ..].to_vec(),
               }
           },
           17 => Instruction::Capability(Capability::from_num(operands[0])?),
           19 => Instruction::TypeVoid { result_id: operands[0] },
           20 => Instruction::TypeBool { result_id: operands[0] },
           21 => Instruction::TypeInt {
               result_id: operands[0],
               width: operands[1],
               signedness: operands[2] != 0,
           },
           22 => Instruction::TypeFloat {
               result_id: operands[0],
               width: operands[1],
           },
           23 => Instruction::TypeVector {
               result_id: operands[0],
               component_id: operands[1],
               count: operands[2],
           },
           24 => Instruction::TypeMatrix {
               result_id: operands[0],
               column_type_id: operands[1],
               column_count: operands[2],
           },
           25 => Instruction::TypeImage {
               result_id: operands[0],
               sampled_type_id: operands[1],
               dim: Dim::from_num(operands[2])?,
               depth: match operands[3] {
                   0 => Some(false),
                   1 => Some(true),
                   2 => None,
                   _ => unreachable!(),
               },
               arrayed: operands[4] != 0,
               ms: operands[5] != 0,
               sampled: match operands[6] {
                   0 => None,
                   1 => Some(true),
                   2 => Some(false),
                   _ => unreachable!(),
               },
               format: ImageFormat::from_num(operands[7])?,
               access: if operands.len() >= 9 {
                   Some(AccessQualifier::from_num(operands[8])?)
               } else {
                   None
               },
           },
           26 => Instruction::TypeSampler { result_id: operands[0] },
           27 => Instruction::TypeSampledImage {
               result_id: operands[0],
               image_type_id: operands[1],
           },
           28 => Instruction::TypeArray {
               result_id: operands[0],
               type_id: operands[1],
               length_id: operands[2],
           },
           29 => Instruction::TypeRuntimeArray {
               result_id: operands[0],
               type_id: operands[1],
           },
           30 => Instruction::TypeStruct {
               result_id: operands[0],
               member_types: operands[1 ..].to_owned(),
           },
           31 => Instruction::TypeOpaque {
               result_id: operands[0],
               name: parse_string(&operands[1 ..]).0,
           },
           32 => Instruction::TypePointer {
               result_id: operands[0],
               storage_class: StorageClass::from_num(operands[1])?,
               type_id: operands[2],
           },
           43 => Instruction::Constant {
               result_type_id: operands[0],
               result_id: operands[1],
               data: operands[2 ..].to_owned(),
           },
           48 => Instruction::SpecConstantTrue {
               result_type_id: operands[0],
               result_id: operands[1],
           },
           49 => Instruction::SpecConstantFalse {
               result_type_id: operands[0],
               result_id: operands[1],
           },
           50 => Instruction::SpecConstant {
               result_type_id: operands[0],
               result_id: operands[1],
               data: operands[2 ..].to_owned(),
           },
           51 => Instruction::SpecConstantComposite {
               result_type_id: operands[0],
               result_id: operands[1],
               data: operands[2 ..].to_owned(),
           },
           56 => Instruction::FunctionEnd,
           59 => Instruction::Variable {
               result_type_id: operands[0],
               result_id: operands[1],
               storage_class: StorageClass::from_num(operands[2])?,
               initializer: operands.get(3).map(|&v| v),
           },
           71 => Instruction::Decorate {
               target_id: operands[0],
               decoration: Decoration::from_num(operands[1])?,
               params: operands[2 ..].to_owned(),
           },
           72 => Instruction::MemberDecorate {
               target_id: operands[0],
               member: operands[1],
               decoration: Decoration::from_num(operands[2])?,
               params: operands[3 ..].to_owned(),
           },
           248 => Instruction::Label { result_id: operands[0] },
           249 => Instruction::Branch { result_id: operands[0] },
           252 => Instruction::Kill,
           253 => Instruction::Return,
           _ => Instruction::Unknown(opcode, operands.to_owned()),
       })
}

fn parse_string(data: &[u32]) -> (String, &[u32]) {
    let bytes = data.iter()
        .flat_map(|&n| {
                      let b1 = (n & 0xff) as u8;
                      let b2 = ((n >> 8) & 0xff) as u8;
                      let b3 = ((n >> 16) & 0xff) as u8;
                      let b4 = ((n >> 24) & 0xff) as u8;
                      vec![b1, b2, b3, b4].into_iter()
                  })
        .take_while(|&b| b != 0)
        .collect::<Vec<u8>>();

    let r = 1 + bytes.len() / 4;
    let s = String::from_utf8(bytes).expect("Shader content is not UTF-8");

    (s, &data[r ..])
}

#[cfg(test)]
mod test {
    use parse;

    #[test]
    fn test() {
        let data = include_bytes!("../tests/frag.spv");
        println!("{:#?}", parse::parse_spirv(data).unwrap());
    }
}
