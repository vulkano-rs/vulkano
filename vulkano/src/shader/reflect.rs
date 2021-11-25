// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Extraction of information from SPIR-V modules, that is needed by the rest of Vulkano.

use crate::descriptor_set::layout::DescriptorType;
use crate::image::view::ImageViewType;
use crate::shader::ShaderScalarType;
use crate::DeviceSize;
use crate::{
    format::Format,
    pipeline::layout::PipelineLayoutPcRange,
    shader::{
        spirv::{
            Capability, Decoration, Dim, ExecutionMode, ExecutionModel, Id, ImageFormat,
            Instruction, Spirv, StorageClass,
        },
        DescriptorRequirements, EntryPointInfo, GeometryShaderExecution, GeometryShaderInput,
        ShaderExecution, ShaderInterface, ShaderInterfaceEntry, ShaderInterfaceEntryType,
        ShaderStage, SpecializationConstantRequirements,
    },
};
use fnv::FnvHashMap;
use std::borrow::Cow;
use std::collections::HashSet;

/// Returns an iterator of the capabilities used by `spirv`.
pub fn spirv_capabilities<'a>(spirv: &'a Spirv) -> impl Iterator<Item = &'a Capability> {
    spirv
        .iter_capability()
        .filter_map(|instruction| match instruction {
            Instruction::Capability { capability } => Some(capability),
            _ => None,
        })
}

/// Returns an iterator of the extensions used by `spirv`.
pub fn spirv_extensions<'a>(spirv: &'a Spirv) -> impl Iterator<Item = &'a str> {
    spirv
        .iter_extension()
        .filter_map(|instruction| match instruction {
            Instruction::Extension { name } => Some(name.as_str()),
            _ => None,
        })
}

/// Returns an iterator over all entry points in `spirv`, with information about the entry point.
pub fn entry_points<'a>(
    spirv: &'a Spirv,
    exact_interface: bool,
) -> impl Iterator<Item = (String, EntryPointInfo)> + 'a {
    spirv.iter_entry_point().filter_map(move |instruction| {
        let (execution_model, function_id, entry_point_name, interface) = match instruction {
            &Instruction::EntryPoint {
                ref execution_model,
                entry_point,
                ref name,
                ref interface,
                ..
            } => (execution_model, entry_point, name, interface),
            _ => return None,
        };

        let execution = shader_execution(&spirv, execution_model, function_id);
        let stage = ShaderStage::from(execution);
        let descriptor_requirements =
            descriptor_requirements(&spirv, function_id, stage, interface, exact_interface);
        let push_constant_requirements = push_constant_requirements(&spirv, stage);
        let specialization_constant_requirements = specialization_constant_requirements(&spirv);
        let input_interface = shader_interface(
            &spirv,
            interface,
            StorageClass::Input,
            matches!(
                execution_model,
                ExecutionModel::TessellationControl
                    | ExecutionModel::TessellationEvaluation
                    | ExecutionModel::Geometry
            ),
        );
        let output_interface = shader_interface(
            &spirv,
            interface,
            StorageClass::Output,
            matches!(execution_model, ExecutionModel::TessellationControl),
        );

        Some((
            entry_point_name.clone(),
            EntryPointInfo {
                execution,
                descriptor_requirements,
                push_constant_requirements,
                specialization_constant_requirements,
                input_interface,
                output_interface,
            },
        ))
    })
}

/// Extracts the `ShaderExecution` for the entry point `function_id` from `spirv`.
fn shader_execution(
    spirv: &Spirv,
    execution_model: &ExecutionModel,
    function_id: Id,
) -> ShaderExecution {
    match execution_model {
        ExecutionModel::Vertex => ShaderExecution::Vertex,

        ExecutionModel::TessellationControl => ShaderExecution::TessellationControl,

        ExecutionModel::TessellationEvaluation => ShaderExecution::TessellationEvaluation,

        ExecutionModel::Geometry => {
            let input = spirv
                .iter_execution_mode()
                .into_iter()
                .find_map(|instruction| match instruction {
                    Instruction::ExecutionMode {
                        entry_point, mode, ..
                    } if *entry_point == function_id => match mode {
                        ExecutionMode::InputPoints => Some(GeometryShaderInput::Points),
                        ExecutionMode::InputLines => Some(GeometryShaderInput::Lines),
                        ExecutionMode::InputLinesAdjacency => {
                            Some(GeometryShaderInput::LinesWithAdjacency)
                        }
                        ExecutionMode::Triangles => Some(GeometryShaderInput::Triangles),
                        ExecutionMode::InputTrianglesAdjacency => {
                            Some(GeometryShaderInput::TrianglesWithAdjacency)
                        }
                        _ => todo!(),
                    },
                    _ => None,
                })
                .expect("Geometry shader does not have an input primitive ExecutionMode");

            ShaderExecution::Geometry(GeometryShaderExecution { input })
        }

        ExecutionModel::Fragment => ShaderExecution::Fragment,

        ExecutionModel::GLCompute => ShaderExecution::Compute,

        ExecutionModel::Kernel
        | ExecutionModel::TaskNV
        | ExecutionModel::MeshNV
        | ExecutionModel::RayGenerationKHR
        | ExecutionModel::IntersectionKHR
        | ExecutionModel::AnyHitKHR
        | ExecutionModel::ClosestHitKHR
        | ExecutionModel::MissKHR
        | ExecutionModel::CallableKHR => {
            todo!()
        }
    }
}

/// Extracts the `DescriptorRequirements` for the entry point `function_id` from `spirv`.
fn descriptor_requirements(
    spirv: &Spirv,
    function_id: Id,
    stage: ShaderStage,
    interface: &[Id],
    exact: bool,
) -> FnvHashMap<(u32, u32), DescriptorRequirements> {
    // For SPIR-V 1.4+, the entrypoint interface can specify variables of all storage classes,
    // and most tools will put all used variables in the entrypoint interface. However,
    // SPIR-V 1.0-1.3 do not specify variables other than Input/Output ones in the interface,
    // and instead the function itself must be inspected.
    let variables = if exact {
        let mut found_variables: HashSet<Id> = interface.iter().cloned().collect();
        let mut inspected_functions: HashSet<Id> = HashSet::new();
        find_variables_in_function(
            &spirv,
            function_id,
            &mut inspected_functions,
            &mut found_variables,
        );
        Some(found_variables)
    } else {
        None
    };

    // Looping to find all the global variables that have the `DescriptorSet` decoration.
    spirv
        .iter_global()
        .filter_map(|instruction| {
            let (variable_id, variable_type_id, storage_class) = match instruction {
                Instruction::Variable {
                    result_id,
                    result_type_id,
                    ..
                } => {
                    let (real_type, storage_class) = match spirv
                        .id(*result_type_id)
                        .instruction()
                    {
                        Instruction::TypePointer {
                            ty, storage_class, ..
                        } => (ty, storage_class),
                        _ => panic!(
                            "Variable {} result_type_id does not refer to a TypePointer instruction", result_id
                        ),
                    };

                    (*result_id, *real_type, storage_class)
                }
                _ => return None,
            };

            if exact && !variables.as_ref().unwrap().contains(&variable_id) {
                return None;
            }

            let variable_id_info = spirv.id(variable_id);
            let set_num = match variable_id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::DescriptorSet { descriptor_set },
                        ..
                    } => Some(*descriptor_set),
                    _ => None,
                }) {
                Some(x) => x,
                None => return None,
            };

            let binding_num = variable_id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::Binding { binding_point },
                        ..
                    } => Some(*binding_point),
                    _ => None,
                })
                .unwrap();

            let name = variable_id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .unwrap_or("__unnamed");

            let nonwritable = variable_id_info.iter_decoration().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Decorate {
                        decoration: Decoration::NonWritable,
                        ..
                    }
                )
            });

            // Find information about the kind of binding for this descriptor.
            let mut reqs =
                descriptor_requirements_of(spirv, variable_type_id, storage_class, false).expect(&format!(
                "Couldn't find relevant type for global variable `{}` (type {}, maybe unimplemented)",
                name, variable_type_id,
            ));

            reqs.stages = stage.into();
            reqs.mutable &= !nonwritable;

            Some(((set_num, binding_num), reqs))
        })
        .collect()
}

// Recursively finds every pointer variable used in the execution of a function.
fn find_variables_in_function(
    spirv: &Spirv,
    function: Id,
    inspected_functions: &mut HashSet<Id>,
    found_variables: &mut HashSet<Id>,
) {
    inspected_functions.insert(function);
    let mut in_function = false;
    for instruction in spirv.instructions() {
        if !in_function {
            match instruction {
                Instruction::Function { result_id, .. } if result_id == &function => {
                    in_function = true;
                }
                _ => {}
            }
        } else {
            // We only care about instructions that accept pointers.
            // https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_universal_validation_rules
            match instruction {
                Instruction::Load { pointer, .. } | Instruction::Store { pointer, .. } => {
                    found_variables.insert(*pointer);
                }
                Instruction::AccessChain { base, .. }
                | Instruction::InBoundsAccessChain { base, .. } => {
                    found_variables.insert(*base);
                }
                Instruction::FunctionCall {
                    function,
                    arguments,
                    ..
                } => {
                    arguments.iter().for_each(|&x| {
                        found_variables.insert(x);
                    });
                    if !inspected_functions.contains(function) {
                        find_variables_in_function(
                            spirv,
                            *function,
                            inspected_functions,
                            found_variables,
                        );
                    }
                }
                Instruction::ImageTexelPointer {
                    image,
                    coordinate,
                    sample,
                    ..
                } => {
                    found_variables.insert(*image);
                    found_variables.insert(*coordinate);
                    found_variables.insert(*sample);
                }
                Instruction::CopyMemory { target, source, .. } => {
                    found_variables.insert(*target);
                    found_variables.insert(*source);
                }
                Instruction::CopyObject { operand, .. } => {
                    found_variables.insert(*operand);
                }
                Instruction::AtomicLoad { pointer, .. }
                | Instruction::AtomicIIncrement { pointer, .. }
                | Instruction::AtomicIDecrement { pointer, .. }
                | Instruction::AtomicFlagTestAndSet { pointer, .. }
                | Instruction::AtomicFlagClear { pointer, .. } => {
                    found_variables.insert(*pointer);
                }
                Instruction::AtomicStore { pointer, value, .. }
                | Instruction::AtomicExchange { pointer, value, .. }
                | Instruction::AtomicIAdd { pointer, value, .. }
                | Instruction::AtomicISub { pointer, value, .. }
                | Instruction::AtomicSMin { pointer, value, .. }
                | Instruction::AtomicUMin { pointer, value, .. }
                | Instruction::AtomicSMax { pointer, value, .. }
                | Instruction::AtomicUMax { pointer, value, .. }
                | Instruction::AtomicAnd { pointer, value, .. }
                | Instruction::AtomicOr { pointer, value, .. }
                | Instruction::AtomicXor { pointer, value, .. } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value);
                }
                Instruction::AtomicCompareExchange {
                    pointer,
                    value,
                    comparator,
                    ..
                }
                | Instruction::AtomicCompareExchangeWeak {
                    pointer,
                    value,
                    comparator,
                    ..
                } => {
                    found_variables.insert(*pointer);
                    found_variables.insert(*value);
                    found_variables.insert(*comparator);
                }
                Instruction::ExtInst { operands, .. } => {
                    // We don't know which extended instructions take pointers,
                    // so we must interpret every operand as a pointer.
                    operands.iter().for_each(|&o| {
                        found_variables.insert(o);
                    });
                }
                Instruction::FunctionEnd => return,
                _ => {}
            }
        }
    }
}

/// Returns a `DescriptorRequirements` value for the pointed type.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_requirements_of(
    spirv: &Spirv,
    pointed_ty: Id,
    pointer_storage: &StorageClass,
    force_combined_image_sampled: bool,
) -> Option<DescriptorRequirements> {
    let id_info = spirv.id(pointed_ty);

    match id_info.instruction() {
        Instruction::TypeStruct { .. } => {
            let decoration_block = id_info.iter_decoration().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Decorate {
                        decoration: Decoration::Block,
                        ..
                    }
                )
            });

            let decoration_buffer_block = id_info.iter_decoration().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Decorate {
                        decoration: Decoration::BufferBlock,
                        ..
                    }
                )
            });

            assert!(
                decoration_block ^ decoration_buffer_block,
                "Structs in shader interface are expected to be decorated with one of Block or BufferBlock"
            );

            let mut reqs = DescriptorRequirements {
                descriptor_count: 1,
                ..Default::default()
            };

            if decoration_buffer_block
                || decoration_block && *pointer_storage == StorageClass::StorageBuffer
            {
                // Determine whether all members have a NonWritable decoration.
                let nonwritable = id_info.iter_members().all(|member_info| {
                    member_info.iter_decoration().any(|instruction| {
                        matches!(
                            instruction,
                            Instruction::MemberDecorate {
                                decoration: Decoration::NonWritable,
                                ..
                            }
                        )
                    })
                });

                reqs.descriptor_types = vec![
                    DescriptorType::StorageBuffer,
                    DescriptorType::StorageBufferDynamic,
                ];
                reqs.mutable = !nonwritable;
            } else {
                reqs.descriptor_types = vec![
                    DescriptorType::UniformBuffer,
                    DescriptorType::UniformBufferDynamic,
                ];
            };

            Some(reqs)
        }
        &Instruction::TypeImage {
            ref dim,
            arrayed,
            ms,
            sampled,
            ref image_format,
            ..
        } => {
            let multisampled = ms != 0;
            assert!(sampled != 0, "Vulkan requires that variables of type OpTypeImage have a Sampled operand of 1 or 2");
            let format: Option<Format> = image_format.clone().into();

            match dim {
                Dim::SubpassData => {
                    assert!(
                        !force_combined_image_sampled,
                        "An OpTypeSampledImage can't point to \
                                                                an OpTypeImage whose dimension is \
                                                                SubpassData"
                    );
                    assert!(
                        *image_format == ImageFormat::Unknown,
                        "If Dim is SubpassData, Image Format must be Unknown"
                    );
                    assert!(sampled == 2, "If Dim is SubpassData, Sampled must be 2");
                    assert!(arrayed == 0, "If Dim is SubpassData, Arrayed must be 0");

                    Some(DescriptorRequirements {
                        descriptor_types: vec![DescriptorType::InputAttachment],
                        descriptor_count: 1,
                        multisampled,
                        ..Default::default()
                    })
                }
                Dim::Buffer => {
                    let mut reqs = DescriptorRequirements {
                        descriptor_count: 1,
                        format,
                        ..Default::default()
                    };

                    if sampled == 1 {
                        reqs.descriptor_types = vec![DescriptorType::UniformTexelBuffer];
                    } else {
                        reqs.descriptor_types = vec![DescriptorType::StorageTexelBuffer];
                        reqs.mutable = true;
                    }

                    Some(reqs)
                }
                _ => {
                    let image_view_type = Some(match (dim, arrayed) {
                        (Dim::Dim1D, 0) => ImageViewType::Dim1d,
                        (Dim::Dim1D, 1) => ImageViewType::Dim1dArray,
                        (Dim::Dim2D, 0) => ImageViewType::Dim2d,
                        (Dim::Dim2D, 1) => ImageViewType::Dim2dArray,
                        (Dim::Dim3D, 0) => ImageViewType::Dim3d,
                        (Dim::Dim3D, 1) => panic!("Vulkan doesn't support arrayed 3D textures"),
                        (Dim::Cube, 0) => ImageViewType::Cube,
                        (Dim::Cube, 1) => ImageViewType::CubeArray,
                        (Dim::Rect, _) => panic!("Vulkan doesn't support rectangle textures"),
                        _ => unreachable!(),
                    });

                    let mut reqs = DescriptorRequirements {
                        descriptor_count: 1,
                        format,
                        multisampled,
                        image_view_type,
                        ..Default::default()
                    };

                    if force_combined_image_sampled {
                        assert!(
                            sampled == 1,
                            "A combined image sampler must not reference a storage image"
                        );

                        reqs.descriptor_types = vec![DescriptorType::CombinedImageSampler];
                    } else {
                        if sampled == 1 {
                            reqs.descriptor_types = vec![DescriptorType::SampledImage];
                        } else {
                            reqs.descriptor_types = vec![DescriptorType::StorageImage];
                            reqs.mutable = true;
                        }
                    };

                    Some(reqs)
                }
            }
        }

        &Instruction::TypeSampledImage { image_type, .. } => {
            descriptor_requirements_of(spirv, image_type, pointer_storage, true)
        }

        &Instruction::TypeSampler { .. } => Some(DescriptorRequirements {
            descriptor_types: vec![DescriptorType::Sampler],
            descriptor_count: 1,
            ..Default::default()
        }),

        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            let reqs = match descriptor_requirements_of(spirv, element_type, pointer_storage, false)
            {
                None => return None,
                Some(v) => v,
            };
            assert_eq!(reqs.descriptor_count, 1); // TODO: implement?
            let len = match spirv.id(length).instruction() {
                &Instruction::Constant { ref value, .. } => value,
                _ => panic!("failed to find array length"),
            };
            let len = len.iter().rev().fold(0, |a, &b| (a << 32) | b as u64);

            Some(DescriptorRequirements {
                descriptor_count: len as u32,
                ..reqs
            })
        }

        &Instruction::TypeRuntimeArray { element_type, .. } => {
            let reqs = match descriptor_requirements_of(spirv, element_type, pointer_storage, false)
            {
                None => return None,
                Some(v) => v,
            };
            assert_eq!(reqs.descriptor_count, 1); // TODO: implement?

            Some(DescriptorRequirements {
                descriptor_count: 0,
                ..reqs
            })
        }

        _ => None,
    }
}

/// Extracts the `PipelineLayoutPcRange` from `spirv`.
fn push_constant_requirements(spirv: &Spirv, stage: ShaderStage) -> Option<PipelineLayoutPcRange> {
    spirv
        .iter_global()
        .find_map(|instruction| match instruction {
            &Instruction::TypePointer {
                ty,
                storage_class: StorageClass::PushConstant,
                ..
            } => {
                let id_info = spirv.id(ty);
                assert!(matches!(
                    id_info.instruction(),
                    Instruction::TypeStruct { .. }
                ));
                let start = offset_of_struct(spirv, ty);
                let end =
                    size_of_type(spirv, ty).expect("Found runtime-sized push constants") as u32;
                Some(PipelineLayoutPcRange {
                    offset: start,
                    size: end - start,
                    stages: stage.into(),
                })
            }
            _ => None,
        })
}

/// Extracts the `SpecializationConstantRequirements` from `spirv`.
fn specialization_constant_requirements(
    spirv: &Spirv,
) -> FnvHashMap<u32, SpecializationConstantRequirements> {
    spirv
        .iter_global()
        .filter_map(|instruction| {
            match instruction {
                &Instruction::SpecConstantTrue {
                    result_type_id,
                    result_id,
                }
                | &Instruction::SpecConstantFalse {
                    result_type_id,
                    result_id,
                }
                | &Instruction::SpecConstant {
                    result_type_id,
                    result_id,
                    ..
                }
                | &Instruction::SpecConstantComposite {
                    result_type_id,
                    result_id,
                    ..
                } => spirv
                    .id(result_id)
                    .iter_decoration()
                    .find_map(|instruction| match instruction {
                        Instruction::Decorate {
                            decoration:
                                Decoration::SpecId {
                                    specialization_constant_id,
                                },
                            ..
                        } => Some(*specialization_constant_id),
                        _ => None,
                    })
                    .and_then(|constant_id| {
                        let size = match spirv.id(result_type_id).instruction() {
                            Instruction::TypeBool { .. } => {
                                // Translate bool to Bool32
                                std::mem::size_of::<ash::vk::Bool32>() as DeviceSize
                            }
                            _ => size_of_type(spirv, result_type_id)
                                .expect("Found runtime-sized specialization constant"),
                        };
                        Some((constant_id, SpecializationConstantRequirements { size }))
                    }),
                _ => None,
            }
        })
        .collect()
}

/// Extracts the `ShaderInterface` with the given storage class from `spirv`.
fn shader_interface(
    spirv: &Spirv,
    interface: &[Id],
    filter_storage_class: StorageClass,
    ignore_first_array: bool,
) -> ShaderInterface {
    let elements: Vec<_> = interface
        .iter()
        .filter_map(|&id| {
            let (result_type_id, result_id) = match spirv.id(id).instruction() {
                &Instruction::Variable {
                    result_type_id,
                    result_id,
                    ref storage_class,
                    ..
                } if storage_class == &filter_storage_class => (result_type_id, result_id),
                _ => return None,
            };

            if is_builtin(spirv, result_id) {
                return None;
            }

            let id_info = spirv.id(result_id);

            let name = id_info
                .iter_name()
                .find_map(|instruction| match instruction {
                    Instruction::Name { name, .. } => Some(Cow::Owned(name.to_owned())),
                    _ => None,
                });

            let location = id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::Location { location },
                        ..
                    } => Some(*location),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    panic!(
                        "Input/output variable with id {} (name {:?}) is missing a location",
                        result_id, name,
                    )
                });
            let component = id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::Component { component },
                        ..
                    } => Some(*component),
                    _ => None,
                })
                .unwrap_or(0);

            let ty = shader_interface_type_of(spirv, result_type_id, ignore_first_array);
            assert!(ty.num_elements >= 1);
            Some(ShaderInterfaceEntry {
                location,
                component,
                ty,
                name,
            })
        })
        .collect();

    // Checking for overlapping elements.
    for (offset, element1) in elements.iter().enumerate() {
        for element2 in elements.iter().skip(offset + 1) {
            if element1.location == element2.location
                || (element1.location < element2.location
                    && element1.location + element1.ty.num_locations() > element2.location)
                || (element2.location < element1.location
                    && element2.location + element2.ty.num_locations() > element1.location)
            {
                panic!(
                    "The locations of attributes `{:?}` ({}..{}) and `{:?}` ({}..{}) overlap",
                    element1.name,
                    element1.location,
                    element1.location + element1.ty.num_locations(),
                    element2.name,
                    element2.location,
                    element2.location + element2.ty.num_locations(),
                );
            }
        }
    }

    ShaderInterface { elements }
}

/// Returns the size of a type, or `None` if its size cannot be determined.
fn size_of_type(spirv: &Spirv, id: Id) -> Option<DeviceSize> {
    let id_info = spirv.id(id);

    match id_info.instruction() {
        Instruction::TypeBool { .. } => {
            panic!("Can't put booleans in structs")
        }
        Instruction::TypeInt { width, .. } | Instruction::TypeFloat { width, .. } => {
            assert!(width % 8 == 0);
            Some(*width as DeviceSize / 8)
        }
        &Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => size_of_type(spirv, component_type)
            .map(|component_size| component_size * component_count as DeviceSize),
        &Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            // FIXME: row-major or column-major
            size_of_type(spirv, column_type)
                .map(|column_size| column_size * column_count as DeviceSize)
        }
        &Instruction::TypeArray { length, .. } => {
            let stride = id_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::Decorate {
                        decoration: Decoration::ArrayStride { array_stride },
                        ..
                    } => Some(*array_stride),
                    _ => None,
                })
                .unwrap();
            let length = match spirv.id(length).instruction() {
                &Instruction::Constant { ref value, .. } => Some(
                    value
                        .iter()
                        .rev()
                        .fold(0u64, |a, &b| (a << 32) | b as DeviceSize),
                ),
                _ => None,
            }
            .unwrap();

            Some(stride as DeviceSize * length)
        }
        Instruction::TypeRuntimeArray { .. } => None,
        Instruction::TypeStruct { member_types, .. } => {
            let mut end_of_struct = 0;

            for (&member, member_info) in member_types.iter().zip(id_info.iter_members()) {
                // Built-ins have an unknown size.
                if member_info.iter_decoration().any(|instruction| {
                    matches!(
                        instruction,
                        Instruction::MemberDecorate {
                            decoration: Decoration::BuiltIn { .. },
                            ..
                        }
                    )
                }) {
                    return None;
                }

                // Some structs don't have `Offset` decorations, in the case they are used as local
                // variables only. Ignoring these.
                let offset =
                    member_info
                        .iter_decoration()
                        .find_map(|instruction| match instruction {
                            Instruction::MemberDecorate {
                                decoration: Decoration::Offset { byte_offset },
                                ..
                            } => Some(*byte_offset),
                            _ => None,
                        })?;
                let size = size_of_type(spirv, member)?;
                end_of_struct = end_of_struct.max(offset as DeviceSize + size);
            }

            Some(end_of_struct)
        }
        _ => panic!("Type {} not found", id),
    }
}

/// Returns the smallest offset of all members of a struct, or 0 if `id` is not a struct.
fn offset_of_struct(spirv: &Spirv, id: Id) -> u32 {
    spirv
        .id(id)
        .iter_members()
        .map(|member_info| {
            member_info
                .iter_decoration()
                .find_map(|instruction| match instruction {
                    Instruction::MemberDecorate {
                        decoration: Decoration::Offset { byte_offset },
                        ..
                    } => Some(*byte_offset),
                    _ => None,
                })
        })
        .flatten()
        .min()
        .unwrap_or(0)
}

/// If `ignore_first_array` is true, the function expects the outermost instruction to be
/// `OpTypeArray`. If it's the case, the OpTypeArray will be ignored. If not, the function will
/// panic.
fn shader_interface_type_of(
    spirv: &Spirv,
    id: Id,
    ignore_first_array: bool,
) -> ShaderInterfaceEntryType {
    match spirv.id(id).instruction() {
        &Instruction::TypeInt {
            width, signedness, ..
        } => {
            assert!(!ignore_first_array);
            ShaderInterfaceEntryType {
                base_type: match signedness {
                    0 => ShaderScalarType::Uint,
                    1 => ShaderScalarType::Sint,
                    _ => unreachable!(),
                },
                num_components: 1,
                num_elements: 1,
                is_64bit: match width {
                    8 | 16 | 32 => false,
                    64 => true,
                    _ => unimplemented!(),
                },
            }
        }
        &Instruction::TypeFloat { width, .. } => {
            assert!(!ignore_first_array);
            ShaderInterfaceEntryType {
                base_type: ShaderScalarType::Float,
                num_components: 1,
                num_elements: 1,
                is_64bit: match width {
                    16 | 32 => false,
                    64 => true,
                    _ => unimplemented!(),
                },
            }
        }
        &Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => {
            assert!(!ignore_first_array);
            ShaderInterfaceEntryType {
                num_components: component_count,
                ..shader_interface_type_of(spirv, component_type, false)
            }
        }
        &Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            assert!(!ignore_first_array);
            ShaderInterfaceEntryType {
                num_elements: column_count,
                ..shader_interface_type_of(spirv, column_type, false)
            }
        }
        &Instruction::TypeArray {
            element_type,
            length,
            ..
        } => {
            if ignore_first_array {
                shader_interface_type_of(spirv, element_type, false)
            } else {
                let mut ty = shader_interface_type_of(spirv, element_type, false);
                let num_elements = spirv
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
                    .expect("failed to find array length")
                    .iter()
                    .rev()
                    .fold(0u64, |a, &b| (a << 32) | b as u64)
                    as u32;
                ty.num_elements *= num_elements;
                ty
            }
        }
        &Instruction::TypePointer { ty, .. } => {
            shader_interface_type_of(spirv, ty, ignore_first_array)
        }
        _ => panic!("Type {} not found or invalid", id),
    }
}

/// Returns true if a `BuiltIn` decorator is applied on an id.
fn is_builtin(spirv: &Spirv, id: Id) -> bool {
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
