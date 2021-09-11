// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, SpirvGrammar};
use heck::SnakeCase;
use lazy_static::lazy_static;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::{
    collections::{HashMap, HashSet},
    iter::FromIterator,
};

lazy_static! {
    static ref SPEC_CONSTANT_OP: HashSet<&'static str> = {
        HashSet::from_iter([
            "SConvert",
            "FConvert",
            "SNegate",
            "Not",
            "IAdd",
            "ISub",
            "IMul",
            "UDiv",
            "SDiv",
            "UMod",
            "SRem",
            "SMod",
            "ShiftRightLogical",
            "ShiftRightArithmetic",
            "ShiftLeftLogical",
            "BitwiseOr",
            "BitwiseXor",
            "BitwiseAnd",
            "VectorShuffle",
            "CompositeExtract",
            "CompositeInsert",
            "LogicalOr",
            "LogicalAnd",
            "LogicalNot",
            "LogicalEqual",
            "LogicalNotEqual",
            "Select",
            "IEqual",
            "INotEqual",
            "ULessThan",
            "SLessThan",
            "UGreaterThan",
            "SGreaterThan",
            "ULessThanEqual",
            "SLessThanEqual",
            "UGreaterThanEqual",
            "SGreaterThanEqual",
            "QuantizeToF16",
            "ConvertFToS",
            "ConvertSToF",
            "ConvertFToU",
            "ConvertUToF",
            "UConvert",
            "ConvertPtrToU",
            "ConvertUToPtr",
            "GenericCastToPtr",
            "PtrCastToGeneric",
            "Bitcast",
            "FNegate",
            "FAdd",
            "FSub",
            "FMul",
            "FDiv",
            "FRem",
            "FMod",
            "AccessChain",
            "InBoundsAccessChain",
            "PtrAccessChain",
            "InBoundsPtrAccessChain",
        ])
    };
}

pub fn write(grammar: &SpirvGrammar) {
    let mut instr_members = instruction_members(grammar);
    let instr_output = instruction_output(&instr_members, false);

    instr_members.retain(|member| SPEC_CONSTANT_OP.contains(member.name.to_string().as_str()));
    instr_members.iter_mut().for_each(|member| {
        if member.has_result_type_id {
            member.operands.remove(0);
        }
        if member.has_result_id {
            member.operands.remove(0);
        }
    });
    let spec_constant_instr_output = instruction_output(&instr_members, true);

    let bit_enum_output = bit_enum_output(&bit_enum_members(grammar));
    let value_enum_output = value_enum_output(&value_enum_members(grammar));

    write_file(
        "spirv.rs",
        format!(
            "SPIR-V grammar version {}.{}.{}",
            grammar.major_version, grammar.minor_version, grammar.revision
        ),
        quote! {
            #instr_output
            #spec_constant_instr_output
            #bit_enum_output
            #value_enum_output
        },
    );
}

#[derive(Clone, Debug)]
struct InstructionMember {
    name: Ident,
    has_result_id: bool,
    has_result_type_id: bool,
    opcode: u16,
    operands: Vec<OperandMember>,
}

#[derive(Clone, Debug)]
struct OperandMember {
    name: Ident,
    ty: TokenStream,
    parse: TokenStream,
}

fn instruction_output(members: &[InstructionMember], spec_constant: bool) -> TokenStream {
    let struct_items = members
        .iter()
        .map(|InstructionMember { name, operands, .. }| {
            if operands.is_empty() {
                quote! { #name, }
            } else {
                let operands = operands.iter().map(|OperandMember { name, ty, .. }| {
                    quote! { #name: #ty, }
                });
                quote! {
                    #name {
                        #(#operands)*
                    },
                }
            }
        });
    let parse_items = members.iter().map(
        |InstructionMember {
             name,
             opcode,
             operands,
             ..
         }| {
            if operands.is_empty() {
                quote! {
                    #opcode => Self::#name,
                }
            } else {
                let operands_items =
                    operands.iter().map(|OperandMember { name, parse, .. }| {
                        quote! {
                            #name: #parse,
                        }
                    });

                quote! {
                    #opcode => Self::#name {
                        #(#operands_items)*
                    },
                }
            }
        },
    );

    let doc = if spec_constant {
        "An instruction that is used as the operand of the `SpecConstantOp` instruction."
    } else {
        "A parsed SPIR-V instruction."
    };

    let enum_name = if spec_constant {
        format_ident!("SpecConstantInstruction")
    } else {
        format_ident!("Instruction")
    };

    let result_fns = if spec_constant {
        quote! {}
    } else {
        let result_id_items = members.iter().filter_map(
            |InstructionMember {
                 name,
                 has_result_id,
                 ..
             }| {
                if *has_result_id {
                    Some(quote! { Self::#name { result_id, .. } })
                } else {
                    None
                }
            },
        );

        quote! {
            /// Returns the `Id` that is assigned by this instruction, if any.
            pub fn result_id(&self) -> Option<Id> {
                match self {
                    #(#result_id_items)|* => Some(*result_id),
                    _ => None
                }
            }
        }
    };

    let opcode_error = if spec_constant {
        format_ident!("UnknownSpecConstantOpcode")
    } else {
        format_ident!("UnknownOpcode")
    };

    quote! {
        #[derive(Clone, Debug, PartialEq)]
        #[doc=#doc]
        pub enum #enum_name {
            #(#struct_items)*
        }

        impl #enum_name {
            fn parse(reader: &mut InstructionReader) -> Result<Self, ParseError> {
                let opcode = (reader.next_u32()? & 0xffff) as u16;

                Ok(match opcode {
                    #(#parse_items)*
                    opcode => return Err(reader.map_err(ParseErrors::#opcode_error(opcode))),
                })
            }

            #result_fns
        }
    }
}

fn instruction_members(grammar: &SpirvGrammar) -> Vec<InstructionMember> {
    let operand_kinds = kinds_to_types(grammar);
    grammar
        .instructions
        .iter()
        .map(|instruction| {
            let name = format_ident!("{}", instruction.opname.strip_prefix("Op").unwrap());
            let mut has_result_id = false;
            let mut has_result_type_id = false;
            let mut operand_names = HashMap::new();

            let mut operands = instruction
                .operands
                .iter()
                .map(|operand| {
                    let name = if operand.kind == "IdResult" {
                        has_result_id = true;
                        format_ident!("result_id")
                    } else if operand.kind == "IdResultType" {
                        has_result_type_id = true;
                        format_ident!("result_type_id")
                    } else {
                        to_member_name(&operand.kind, operand.name.as_ref().map(|x| x.as_str()))
                    };

                    *operand_names.entry(name.clone()).or_insert(0) += 1;

                    let (ty, parse) = &operand_kinds[operand.kind.as_str()];
                    let ty = match operand.quantifier {
                        Some('?') => quote! { Option<#ty> },
                        Some('*') => quote! { Vec<#ty> },
                        _ => ty.clone(),
                    };
                    let parse = match operand.quantifier {
                        Some('?') => quote! {
                            if !reader.is_empty() {
                                Some(#parse)
                            } else {
                                None
                            }
                        },
                        Some('*') => quote! {{
                            let mut vec = Vec::new();
                            while !reader.is_empty() {
                                vec.push(#parse);
                            }
                            vec
                        }},
                        _ => parse.clone(),
                    };

                    OperandMember {
                        name,
                        ty,
                        parse: parse.clone(),
                    }
                })
                .collect::<Vec<_>>();

            // Add number to operands with identical names
            for name in operand_names
                .into_iter()
                .filter_map(|(n, c)| if c > 1 { Some(n) } else { None })
            {
                let mut num = 1;

                for operand in operands.iter_mut().filter(|o| o.name == name) {
                    operand.name = format_ident!("{}{}", name, format!("{}", num));
                    num += 1;
                }
            }

            InstructionMember {
                name,
                has_result_id,
                has_result_type_id,
                opcode: instruction.opcode,
                operands,
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct KindEnumMember {
    name: Ident,
    value: u32,
    parameters: Vec<OperandMember>,
}

fn bit_enum_output(enums: &[(Ident, Vec<KindEnumMember>)]) -> TokenStream {
    let enum_items = enums.iter().map(|(name, members)| {
        let members_items = members.iter().map(
            |KindEnumMember {
                 name, parameters, ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        pub #name: bool,
                    }
                } else if let [OperandMember { ty, .. }] = parameters.as_slice() {
                    quote! {
                        pub #name: Option<#ty>,
                    }
                } else {
                    let params = parameters.iter().map(|OperandMember { ty, .. }| {
                        quote! { #ty }
                    });
                    quote! {
                        pub #name: Option<(#(#params),*)>,
                    }
                }
            },
        );
        let parse_items = members.iter().map(
            |KindEnumMember {
                 name,
                 value,
                 parameters,
                 ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        #name: value & #value != 0,
                    }
                } else {
                    let some = if let [OperandMember { parse, .. }] = parameters.as_slice() {
                        quote! { #parse }
                    } else {
                        let parse = parameters.iter().map(|OperandMember { parse, .. }| parse);
                        quote! { (#(#parse),*) }
                    };

                    quote! {
                        #name: if value & #value != 0 {
                            Some(#some)
                        } else {
                            None
                        },
                    }
                }
            },
        );

        quote! {
            #[derive(Clone, Debug, PartialEq)]
            #[allow(non_camel_case_types)]
            pub struct #name {
                #(#members_items)*
            }

            impl #name {
                fn parse(reader: &mut InstructionReader) -> Result<#name, ParseError> {
                    let value = reader.next_u32()?;

                    Ok(Self {
                        #(#parse_items)*
                    })
                }
            }
        }
    });

    quote! {
        #(#enum_items)*
    }
}

fn bit_enum_members(grammar: &SpirvGrammar) -> Vec<(Ident, Vec<KindEnumMember>)> {
    let parameter_kinds = kinds_to_types(grammar);

    grammar
        .operand_kinds
        .iter()
        .filter(|operand_kind| operand_kind.category == "BitEnum")
        .map(|operand_kind| {
            let members = operand_kind
                .enumerants
                .iter()
                .filter_map(|enumerant| {
                    let value = enumerant
                        .value
                        .as_str()
                        .unwrap()
                        .strip_prefix("0x")
                        .unwrap();
                    let value = u32::from_str_radix(value, 16).unwrap();

                    if value == 0 {
                        return None;
                    }

                    let name = match enumerant.enumerant.to_snake_case().as_str() {
                        "const" => format_ident!("constant"),
                        "not_na_n" => format_ident!("not_nan"),
                        name => format_ident!("{}", name),
                    };

                    let parameters = enumerant
                        .parameters
                        .iter()
                        .map(|param| {
                            let name = to_member_name(
                                &param.kind,
                                param.name.as_ref().map(|x| x.as_str()),
                            );
                            let (ty, parse) = parameter_kinds[param.kind.as_str()].clone();

                            OperandMember { name, ty, parse }
                        })
                        .collect();

                    Some(KindEnumMember {
                        name,
                        value,
                        parameters,
                    })
                })
                .collect();

            (format_ident!("{}", operand_kind.kind), members)
        })
        .collect()
}

fn value_enum_output(enums: &[(Ident, Vec<KindEnumMember>)]) -> TokenStream {
    let enum_items = enums.iter().map(|(name, members)| {
        let members_items = members.iter().map(
            |KindEnumMember {
                 name, parameters, ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        #name,
                    }
                } else {
                    let params = parameters.iter().map(|OperandMember { name, ty, .. }| {
                        quote! { #name: #ty, }
                    });
                    quote! {
                        #name {
                            #(#params)*
                        },
                    }
                }
            },
        );
        let parse_items = members.iter().map(
            |KindEnumMember {
                 name,
                 value,
                 parameters,
                 ..
             }| {
                if parameters.is_empty() {
                    quote! {
                        #value => Self::#name,
                    }
                } else {
                    let params_items =
                        parameters.iter().map(|OperandMember { name, parse, .. }| {
                            quote! {
                                #name: #parse,
                            }
                        });

                    quote! {
                        #value => Self::#name {
                            #(#params_items)*
                        },
                    }
                }
            },
        );
        let name_string = name.to_string();

        quote! {
            #[derive(Clone, Debug, PartialEq)]
            #[allow(non_camel_case_types)]
            pub enum #name {
                #(#members_items)*
            }

            impl #name {
                fn parse(reader: &mut InstructionReader) -> Result<#name, ParseError> {
                    Ok(match reader.next_u32()? {
                        #(#parse_items)*
                        value => return Err(reader.map_err(ParseErrors::UnknownEnumerant(#name_string, value))),
                    })
                }
            }
        }
    });

    quote! {
        #(#enum_items)*
    }
}

fn value_enum_members(grammar: &SpirvGrammar) -> Vec<(Ident, Vec<KindEnumMember>)> {
    let parameter_kinds = kinds_to_types(grammar);

    grammar
        .operand_kinds
        .iter()
        .filter(|operand_kind| operand_kind.category == "ValueEnum")
        .map(|operand_kind| {
            let members = operand_kind
                .enumerants
                .iter()
                .map(|enumerant| {
                    let name = match enumerant.enumerant.as_str() {
                        "1D" => format_ident!("Dim1D"),
                        "2D" => format_ident!("Dim2D"),
                        "3D" => format_ident!("Dim3D"),
                        name => format_ident!("{}", name),
                    };
                    let parameters = enumerant
                        .parameters
                        .iter()
                        .map(|param| {
                            let name = to_member_name(
                                &param.kind,
                                param.name.as_ref().map(|x| x.as_str()),
                            );
                            let (ty, parse) = parameter_kinds[param.kind.as_str()].clone();

                            OperandMember { name, ty, parse }
                        })
                        .collect();

                    KindEnumMember {
                        name,
                        value: enumerant.value.as_u64().unwrap() as u32,
                        parameters,
                    }
                })
                .collect();

            (format_ident!("{}", operand_kind.kind), members)
        })
        .collect()
}

fn to_member_name(kind: &str, name: Option<&str>) -> Ident {
    if let Some(name) = name {
        let name = name.to_snake_case();

        // Fix some weird names
        match name.as_str() {
            "argument_0_argument_1" => format_ident!("arguments"),
            "member_0_type_member_1_type" => format_ident!("member_types"),
            "operand_1_operand_2" => format_ident!("operands"),
            "parameter_0_type_parameter_1_type" => format_ident!("parameter_types"),
            "the_name_of_the_opaque_type" => format_ident!("name"),
            "d_ref" => format_ident!("dref"),
            "type" => format_ident!("ty"), // type is a keyword
            _ => format_ident!("{}", name.replace("operand_", "operand")),
        }
    } else {
        format_ident!("{}", kind.to_snake_case())
    }
}

fn kinds_to_types(grammar: &SpirvGrammar) -> HashMap<&str, (TokenStream, TokenStream)> {
    grammar
        .operand_kinds
        .iter()
        .map(|k| {
            let (ty, parse) = match k.kind.as_str() {
                "LiteralContextDependentNumber" => {
                    (quote! { Vec<u32> }, quote! { reader.remainder() })
                }
                "LiteralExtInstInteger" | "LiteralInteger" | "LiteralInt32" => {
                    (quote! { u32 }, quote! { reader.next_u32()? })
                }
                "LiteralInt64" => (quote! { u64 }, quote! { reader.next_u64()? }),
                "LiteralFloat32" => (
                    quote! { f32 },
                    quote! { f32::from_bits(reader.next_u32()?) },
                ),
                "LiteralFloat64" => (
                    quote! { f64 },
                    quote! { f64::from_bits(reader.next_u64()?) },
                ),
                "LiteralSpecConstantOpInteger" => (
                    quote! { SpecConstantInstruction },
                    quote! { SpecConstantInstruction::parse(reader)? },
                ),
                "LiteralString" => (quote! { String }, quote! { reader.next_string()? }),
                "PairIdRefIdRef" => (
                    quote! { (Id, Id) },
                    quote! {
                        (
                            Id(reader.next_u32()?),
                            Id(reader.next_u32()?),
                        )
                    },
                ),
                "PairIdRefLiteralInteger" => (
                    quote! { (Id, u32) },
                    quote! {
                        (
                            Id(reader.next_u32()?),
                            reader.next_u32()?
                        )
                    },
                ),
                "PairLiteralIntegerIdRef" => (
                    quote! { (u32, Id) },
                    quote! {
                    (
                        reader.next_u32()?,
                        Id(reader.next_u32()?)),
                    },
                ),
                _ if k.kind.starts_with("Id") => (quote! { Id }, quote! { Id(reader.next_u32()?) }),
                ident => {
                    let ident = format_ident!("{}", ident);
                    (quote! { #ident }, quote! { #ident::parse(reader)? })
                }
            };

            (k.kind.as_str(), (ty, parse))
        })
        .collect()
}
