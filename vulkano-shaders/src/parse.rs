// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::enums::*;

/// Parses a SPIR-V document from a list of words.
pub fn parse_spirv(i: &[u32]) -> Result<Spirv, ParseError> {
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
    DecorationGroup {
        result_id: u32,
    },
    GroupDecorate {
        decoration_group: u32,
        targets: Vec<u32>,
    },
    GroupMemberDecorate {
        decoration_group: u32,
        targets: Vec<(u32, u32)>,
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
           73 => Instruction::DecorationGroup {
                result_id: operands[0],
           },
           74 => Instruction::GroupDecorate {
               decoration_group: operands[0],
               targets: operands[1 ..].to_owned(),
           },
           75 => Instruction::GroupMemberDecorate {
               decoration_group: operands[0],
               targets: operands.chunks(2).map(|x| (x[0], x[1])).collect(),
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

pub(crate) struct FoundDecoration {
    pub target_id: u32,
    pub params: Vec<u32>
}

impl Spirv {
    /// Returns the params and the id of all decorations that match the passed Decoration type
    ///
    /// for each matching OpDecorate:
    ///     if it points at a regular target:
    ///         creates a FoundDecoration with its params and target_id
    ///     if it points at a group:
    ///         the OpDecorate's target_id is ignored and a seperate FoundDecoration is created only for each target_id given in matching OpGroupDecorate instructions.
    pub(crate) fn get_decorations(&self, find_decoration: Decoration) -> Vec<FoundDecoration> {
        let mut decorations = vec!();
        for instruction in &self.instructions {
            if let Instruction::Decorate { target_id, ref decoration, ref params } = instruction {
                if *decoration == find_decoration {
                    // assume by default it is just pointing at the target_id
                    let mut target_ids = vec!(*target_id);

                    // however it might be pointing at a group, which can have multiple target_ids
                    for inner_instruction in &self.instructions {
                        if let Instruction::DecorationGroup { result_id } = inner_instruction {
                            if *result_id == *target_id {
                                target_ids.clear();

                                for inner_instruction in &self.instructions {
                                    if let Instruction::GroupDecorate { decoration_group, targets } = inner_instruction {
                                        if *decoration_group == *target_id {
                                            target_ids.extend(targets);
                                        }
                                    }
                                }

                                // result_id must be unique so we can safely break here
                                break
                            }
                        }
                    }

                    // create for all target_ids found
                    for target_id in target_ids {
                        decorations.push(FoundDecoration {
                            target_id,
                            params: params.clone()
                        });
                    }
                }
            }
        }
        decorations
    }

    /// Returns the params held by the decoration for the specified id and type
    /// Searches OpDecorate and OpGroupMemberDecorate
    /// Returns None if such a decoration does not exist
    pub(crate) fn get_decoration_params(&self, id: u32, find_decoration: Decoration) -> Option<Vec<u32>> {
        for instruction in &self.instructions {
            match instruction {
                Instruction::Decorate { target_id, ref decoration, ref params }
                if *target_id == id && *decoration == find_decoration => {
                    return Some(params.clone());
                }
                Instruction::GroupDecorate { decoration_group, ref targets } => {
                    for group_target_id in targets {
                        if *group_target_id == id {
                            for instruction in &self.instructions {
                                if let Instruction::Decorate { target_id, ref decoration, ref params } = instruction {
                                    if target_id == decoration_group && *decoration == find_decoration {
                                        return Some(params.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                _ => (),
            };
        }
        None
    }

    /// Returns the params held by the decoration for the member specified by id, member and type
    /// Searches OpMemberDecorate and OpGroupMemberDecorate
    /// Returns None if such a decoration does not exist
    pub(crate) fn get_member_decoration_params(&self, struct_id: u32, member_literal: u32, find_decoration: Decoration) -> Option<Vec<u32>> {
        for instruction in &self.instructions {
            match instruction {
                Instruction::MemberDecorate { target_id, member, ref decoration, ref params }
                if *target_id == struct_id && *member == member_literal && *decoration == find_decoration => {
                    return Some(params.clone());
                }
                Instruction::GroupMemberDecorate { decoration_group, ref targets } => {
                    for (group_target_struct_id, group_target_member_literal) in targets {
                        if *group_target_struct_id == struct_id && *group_target_member_literal == member_literal {
                            for instruction in &self.instructions {
                                if let Instruction::Decorate { target_id, ref decoration, ref params } = instruction {
                                    if target_id == decoration_group && *decoration == find_decoration {
                                        return Some(params.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                _ => (),
            };
        }
        None
    }

    /// Returns the params held by the Decoration::DecorationBuiltIn for the specified struct id
    /// Searches OpMemberDecorate and OpGroupMemberDecorate
    /// Returns None if such a decoration does not exist
    ///
    /// This function does not need a member_literal argument because the spirv spec requires that a
    /// struct must contain either all builtin or all non-builtin members.
    pub(crate) fn get_member_decoration_builtin_params(&self, struct_id: u32) -> Option<Vec<u32>> {
        for instruction in &self.instructions {
            match instruction {
                Instruction::MemberDecorate { target_id, decoration: Decoration::DecorationBuiltIn, ref params, .. }
                if *target_id == struct_id => {
                    return Some(params.clone());
                }
                Instruction::GroupMemberDecorate { decoration_group, ref targets } => {
                    for (group_target_struct_id, _) in targets {
                        if *group_target_struct_id == struct_id {
                            for instruction in &self.instructions {
                                if let Instruction::Decorate { target_id, decoration: Decoration::DecorationBuiltIn, ref params } = instruction {
                                    if target_id == decoration_group {
                                        return Some(params.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                _ => (),
            };
        }
        None
    }
}

#[cfg(test)]
mod test {
    use crate::parse;

    #[test]
    fn test() {
        let data = include_bytes!("../tests/frag.spv");
        let insts: Vec<_> = data.chunks(4)
            .map(|c| {
                ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) | c[0] as u32
            })
            .collect();

        parse::parse_spirv(&insts).unwrap();
    }
}
