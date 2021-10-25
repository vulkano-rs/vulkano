// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::spirv::{Decoration, Id, Instruction, Spirv};

/// Returns the vulkano `Format`, the number of occupied locations, and the number of components from an id.
///
/// If `ignore_first_array` is true, the function expects the outermost instruction to be
/// `OpTypeArray`. If it's the case, the OpTypeArray will be ignored. If not, the function will
/// panic.
pub fn format_from_id(spirv: &Spirv, searched: Id, ignore_first_array: bool, component: Option<u32>) -> (String, usize, Option<u32>) {
    let id_info = spirv.id(searched);
    let component_idx = component.unwrap_or(0);

    match id_info.instruction() {
        &Instruction::TypeInt {
            width, signedness, ..
        } => {
            assert!(!ignore_first_array);
            assert!([0u32, 1u32].contains(&signedness));
            assert!([8u32, 16u32, 32u32, 64u32].contains(&width));
            let format_sign = if signedness == 1 {
                "S"
            } else {
                "U"
            };
            let format = format!("R{}_{}INT", width, format_sign);
            let components = if width == 64 {
                assert_eq!(component_idx, 0);
                0b1111
            } else {
                0b0001 << component_idx
            };
            (format.to_string(), 1, Some(components))
        }
        &Instruction::TypeFloat { width, .. } => {
            assert!(!ignore_first_array);
            assert!([32u32, 64u32].contains(&width));
            let format = format!("R{}_SFLOAT", width);
            let components = if width == 64 {
                assert_eq!(component_idx, 0);
                0b1111
            } else {
                0b0001 << component_idx
            };
            (format.to_string(), 1,  Some(components))
        }
        &Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => {
            assert!(!ignore_first_array);
            let (component_format, component_location_size, component_mask) = format_from_id(spirv, component_type, false, None);
            assert_eq!(component_location_size, 1);
            assert!(component_mask.unwrap() <= 0b1111);
            let is_single = component_format.starts_with("R32");
            let is_double = component_format.starts_with("R64");
            assert!(is_single || is_double);
            let format_bits = (&component_format[1..3]).clone();
            let format = match component_count {
                1 => component_format,
                2 => format!("R{0}G{0}{1}", format_bits, &component_format[3..]),
                3 => format!("R{0}G{0}B{0}{1}", format_bits, &component_format[3..]),
                4 => format!("R{0}G{0}B{0}A{0}{1}", format_bits, &component_format[3..]),
                _ => panic!("Found vector type with more than 4 elements"),
            };
            let components_used: u32 = if is_single {
                assert!(component_idx <= (4 - component_count), "Single precision vectors cannot be shifted into more than one location.");
                ((2 << (component_count-1)) - 1) << component_idx
            } else {
                match component_count {
                    1 => {
                        assert!(component_idx == 0 || component_idx == 2);
                        0b0000_0011 << component_idx
                    },
                    2 => {
                        assert_eq!(component_idx, 0);
                        0b0000_1111
                    },
                    3 => {
                        assert_eq!(component, None);
                        0b0011_1111
                    },
                    4 => {
                        assert_eq!(component, None);
                        0b1111_1111
                    },
                    _ => panic!("Found vector type with more than 4 elements"),
                }
            };
            let locations_used =
                if is_single || component_count < 3  {
                    assert!(components_used <= 0x0000_1111);
                    1
                } else {
                    assert!(components_used <= 0x1111_1111);
                    2
                };
            (format, locations_used, Some(components_used))
        }
        &Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            assert!(!ignore_first_array);
            assert_eq!(component, None, "Matricies cannot have a specified component.");
            let (format, column_locations, column_mask) = format_from_id(spirv, column_type, false, None);
            let locations = column_locations*column_count as usize;
            assert!(column_locations <= 2); // the maximum number of locations in a column is a dvec4 (2 locations)
            // matrix columns take up consecutive locations
            let components = 
                (0..column_count).into_iter().fold(0, |acc: u32, i: u32| {
                    acc | column_mask.unwrap() << (i*4*column_locations as u32)
                });
            (format, locations, Some(components))
        }
        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            if ignore_first_array {
                format_from_id(spirv, element_type, false, None)
            } else {
                let (format, elem_locations, _) = format_from_id(spirv, element_type, false, None);
                let len = spirv
                    .instructions()
                    .iter()
                    .filter_map(|e| match e {
                        &Instruction::Constant {
                            result_id,
                            ref value,
                            ..
                        } if result_id == length => Some(value.clone()),
                        _ => None,
                    })
                    .next()
                    .expect("failed to find array length");
                let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
                // array elements, like matricies, take up consecutive locations
                // unlike matricies or vectors, we can't use component binding.
                // This return value represents a full mask of length `len`, which may be large.
                (format, elem_locations * len as usize, None)
            }
        }
        &Instruction::TypePointer { ty, .. } => format_from_id(spirv, ty, ignore_first_array, component),
        _ => panic!("Type #{} not found or invalid", searched),
    }
}

/// Returns true if a `BuiltIn` decorator is applied on an id.
pub fn is_builtin(spirv: &Spirv, id: Id) -> bool {
    let id_info = spirv.id(id);

    if id_info.iter_decoration().any(|instruction| {
        matches!(
            instruction,
            Instruction::Decorate {
                decoration: Decoration::BuiltIn { .. },
                ..
            }
        )
    }) {
        return true;
    }

    if id_info
        .iter_members()
        .flat_map(|member_info| member_info.iter_decoration())
        .any(|instruction| {
            matches!(
                instruction,
                Instruction::MemberDecorate {
                    decoration: Decoration::BuiltIn { .. },
                    ..
                }
            )
        })
    {
        return true;
    }

    match id_info.instruction() {
        Instruction::Variable { result_type_id, .. } => {
            return is_builtin(spirv, *result_type_id);
        }
        Instruction::TypeArray { element_type, .. } => {
            return is_builtin(spirv, *element_type);
        }
        Instruction::TypeRuntimeArray { element_type, .. } => {
            return is_builtin(spirv, *element_type);
        }
        Instruction::TypeStruct { member_types, .. } => {
            if member_types.iter().any(|ty| is_builtin(spirv, *ty)) {
                return true;
            }
        }
        Instruction::TypePointer { ty, .. } => {
            return is_builtin(spirv, *ty);
        }
        _ => (),
    }

    false
}
