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
    pipeline::layout::PipelineLayoutPcRange,
    shader::{
        spirv::{
            Capability, Decoration, Dim, ExecutionMode, ExecutionModel, Id, Instruction, Spirv,
            StorageClass,
        },
        DescriptorIdentifier, DescriptorRequirements, EntryPointInfo, GeometryShaderExecution,
        GeometryShaderInput, ShaderExecution, ShaderInterface, ShaderInterfaceEntry,
        ShaderInterfaceEntryType, ShaderStage, SpecializationConstantRequirements,
    },
};
use fnv::{FnvHashMap, FnvHashSet};
use std::borrow::Cow;

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
) -> impl Iterator<Item = (String, ExecutionModel, EntryPointInfo)> + 'a {
    let interface_variables = interface_variables(spirv);

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

        let mut descriptor_requirements =
            inspect_entry_point(&interface_variables.descriptor, spirv, function_id);

        for reqs in descriptor_requirements.values_mut() {
            reqs.stages = stage.into();
        }

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
            *execution_model,
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

#[derive(Clone, Debug, Default)]
struct InterfaceVariables {
    descriptor: FnvHashMap<Id, DescriptorVariable>,
}

// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface.
#[derive(Clone, Debug)]
struct DescriptorVariable {
    set: u32,
    binding: u32,
    reqs: DescriptorRequirements,
}

fn interface_variables(spirv: &Spirv) -> InterfaceVariables {
    let mut variables = InterfaceVariables::default();

    for instruction in spirv.iter_global() {
        match instruction {
            Instruction::Variable {
                result_id,
                result_type_id,
                storage_class,
                ..
            } => match storage_class {
                StorageClass::StorageBuffer
                | StorageClass::Uniform
                | StorageClass::UniformConstant => {
                    variables
                        .descriptor
                        .insert(*result_id, descriptor_requirements_of(spirv, *result_id));
                }
                _ => (),
            },
            _ => (),
        }
    }

    variables
}

fn inspect_entry_point(
    global: &FnvHashMap<Id, DescriptorVariable>,
    spirv: &Spirv,
    entry_point: Id,
) -> FnvHashMap<(u32, u32), DescriptorRequirements> {
    #[inline]
    fn instruction_chain<'a, const N: usize>(
        result: &'a mut FnvHashMap<Id, DescriptorVariable>,
        global: &FnvHashMap<Id, DescriptorVariable>,
        spirv: &Spirv,
        chain: [fn(&Spirv, Id) -> Option<Id>; N],
        id: Id,
    ) -> Option<(&'a mut DescriptorVariable, Option<u32>)> {
        let id = chain.into_iter().try_fold(id, |id, func| func(spirv, id))?;

        if let Some(variable) = global.get(&id) {
            // Variable was accessed without an access chain, return with index 0.
            let variable = result.entry(id).or_insert_with(|| variable.clone());
            return Some((variable, Some(0)));
        }

        let (id, indexes) = match spirv.id(id).instruction() {
            &Instruction::AccessChain {
                base, ref indexes, ..
            } => (base, indexes),
            _ => return None,
        };

        if let Some(variable) = global.get(&id) {
            // Variable was accessed with an access chain.
            // Retrieve index from instruction if it's a constant value.
            // TODO: handle a `None` index too?
            let index = match spirv.id(*indexes.first().unwrap()).instruction() {
                &Instruction::Constant { ref value, .. } => Some(value[0]),
                _ => None,
            };
            let variable = result.entry(id).or_insert_with(|| variable.clone());
            return Some((variable, index));
        }

        None
    }

    #[inline]
    fn inst_image_texel_pointer(spirv: &Spirv, id: Id) -> Option<Id> {
        match spirv.id(id).instruction() {
            &Instruction::ImageTexelPointer { image, .. } => Some(image),
            _ => None,
        }
    }

    #[inline]
    fn inst_load(spirv: &Spirv, id: Id) -> Option<Id> {
        match spirv.id(id).instruction() {
            &Instruction::Load { pointer, .. } => Some(pointer),
            _ => None,
        }
    }

    #[inline]
    fn inst_sampled_image(spirv: &Spirv, id: Id) -> Option<Id> {
        match spirv.id(id).instruction() {
            &Instruction::SampledImage { sampler, .. } => Some(sampler),
            _ => Some(id),
        }
    }

    fn inspect_entry_point_r(
        result: &mut FnvHashMap<Id, DescriptorVariable>,
        inspected_functions: &mut FnvHashSet<Id>,
        global: &FnvHashMap<Id, DescriptorVariable>,
        spirv: &Spirv,
        function: Id,
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
                match instruction {
                    &Instruction::AtomicLoad { pointer, .. }
                    | &Instruction::AtomicStore { pointer, .. }
                    | &Instruction::AtomicExchange { pointer, .. }
                    | &Instruction::AtomicCompareExchange { pointer, .. }
                    | &Instruction::AtomicCompareExchangeWeak { pointer, .. }
                    | &Instruction::AtomicIIncrement { pointer, .. }
                    | &Instruction::AtomicIDecrement { pointer, .. }
                    | &Instruction::AtomicIAdd { pointer, .. }
                    | &Instruction::AtomicISub { pointer, .. }
                    | &Instruction::AtomicSMin { pointer, .. }
                    | &Instruction::AtomicUMin { pointer, .. }
                    | &Instruction::AtomicSMax { pointer, .. }
                    | &Instruction::AtomicUMax { pointer, .. }
                    | &Instruction::AtomicAnd { pointer, .. }
                    | &Instruction::AtomicOr { pointer, .. }
                    | &Instruction::AtomicXor { pointer, .. }
                    | &Instruction::AtomicFlagTestAndSet { pointer, .. }
                    | &Instruction::AtomicFlagClear { pointer, .. }
                    | &Instruction::AtomicFMinEXT { pointer, .. }
                    | &Instruction::AtomicFMaxEXT { pointer, .. }
                    | &Instruction::AtomicFAddEXT { pointer, .. } => {
                        // Storage buffer
                        instruction_chain(result, global, spirv, [], pointer);

                        // Storage image
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_image_texel_pointer],
                            pointer,
                        ) {
                            variable.reqs.storage_image_atomic.insert(index);
                        }
                    }

                    &Instruction::CopyMemory { target, source, .. } => {
                        instruction_chain(result, global, spirv, [], target);
                        instruction_chain(result, global, spirv, [], source);
                    }

                    &Instruction::CopyObject { operand, .. } => {
                        instruction_chain(result, global, spirv, [], operand);
                    }

                    &Instruction::ExtInst { ref operands, .. } => {
                        // We don't know which extended instructions take pointers,
                        // so we must interpret every operand as a pointer.
                        for &operand in operands {
                            instruction_chain(result, global, spirv, [], operand);
                        }
                    }

                    &Instruction::FunctionCall {
                        function,
                        ref arguments,
                        ..
                    } => {
                        // Rather than trying to figure out the type of each argument, we just
                        // try all of them as pointers.
                        for &argument in arguments {
                            instruction_chain(result, global, spirv, [], argument);
                        }

                        if !inspected_functions.contains(&function) {
                            inspect_entry_point_r(
                                result,
                                inspected_functions,
                                global,
                                spirv,
                                function,
                            );
                        }
                    }

                    &Instruction::FunctionEnd => return,

                    &Instruction::ImageGather {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseGather {
                        sampled_image,
                        ref image_operands,
                        ..
                    } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            variable.reqs.sampler_no_ycbcr_conversion.insert(index);

                            if image_operands.as_ref().map_or(false, |image_operands| {
                                image_operands.bias.is_some()
                                    || image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                            }) {
                                variable
                                    .reqs
                                    .sampler_no_unnormalized_coordinates
                                    .insert(index);
                            }
                        }
                    }

                    &Instruction::ImageDrefGather { sampled_image, .. }
                    | &Instruction::ImageSparseDrefGather { sampled_image, .. } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            variable
                                .reqs
                                .sampler_no_unnormalized_coordinates
                                .insert(index);
                            variable.reqs.sampler_no_ycbcr_conversion.insert(index);
                        }
                    }

                    &Instruction::ImageSampleImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSampleProjImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleProjImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            variable
                                .reqs
                                .sampler_no_unnormalized_coordinates
                                .insert(index);

                            if image_operands.as_ref().map_or(false, |image_operands| {
                                image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                            }) {
                                variable.reqs.sampler_no_ycbcr_conversion.insert(index);
                            }
                        }
                    }

                    &Instruction::ImageSampleProjExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleProjExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            variable
                                .reqs
                                .sampler_no_unnormalized_coordinates
                                .insert(index);

                            if image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                variable.reqs.sampler_no_ycbcr_conversion.insert(index);
                            }
                        }
                    }

                    &Instruction::ImageSampleDrefImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSampleProjDrefImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleDrefImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleProjDrefImplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            variable
                                .reqs
                                .sampler_no_unnormalized_coordinates
                                .insert(index);
                            variable.reqs.sampler_compare.insert(index);

                            if image_operands.as_ref().map_or(false, |image_operands| {
                                image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                            }) {
                                variable.reqs.sampler_no_ycbcr_conversion.insert(index);
                            }
                        }
                    }

                    &Instruction::ImageSampleDrefExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSampleProjDrefExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleDrefExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleProjDrefExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            variable
                                .reqs
                                .sampler_no_unnormalized_coordinates
                                .insert(index);
                            variable.reqs.sampler_compare.insert(index);

                            if image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                variable.reqs.sampler_no_ycbcr_conversion.insert(index);
                            }
                        }
                    }

                    &Instruction::ImageSampleExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    }
                    | &Instruction::ImageSparseSampleExplicitLod {
                        sampled_image,
                        ref image_operands,
                        ..
                    } => {
                        if let Some((variable, Some(index))) = instruction_chain(
                            result,
                            global,
                            spirv,
                            [inst_sampled_image, inst_load],
                            sampled_image,
                        ) {
                            if image_operands.bias.is_some()
                                || image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                variable
                                    .reqs
                                    .sampler_no_unnormalized_coordinates
                                    .insert(index);
                            }

                            if image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                variable.reqs.sampler_no_ycbcr_conversion.insert(index);
                            }
                        }
                    }

                    &Instruction::ImageTexelPointer {
                        result_id, image, ..
                    } => {
                        instruction_chain(result, global, spirv, [], image);
                    }

                    &Instruction::ImageRead { image, .. } => {
                        if let Some((variable, Some(index))) =
                            instruction_chain(result, global, spirv, [inst_load], image)
                        {
                            variable.reqs.storage_read.insert(index);
                        }
                    }

                    &Instruction::ImageWrite { image, .. } => {
                        if let Some((variable, Some(index))) =
                            instruction_chain(result, global, spirv, [inst_load], image)
                        {
                            variable.reqs.storage_write.insert(index);
                        }
                    }

                    &Instruction::Load { pointer, .. } => {
                        instruction_chain(result, global, spirv, [], pointer);
                    }

                    &Instruction::SampledImage { image, sampler, .. } => {
                        let identifier =
                            match instruction_chain(result, global, spirv, [inst_load], image) {
                                Some((variable, Some(index))) => DescriptorIdentifier {
                                    set: variable.set,
                                    binding: variable.binding,
                                    index,
                                },
                                _ => continue,
                            };

                        if let Some((variable, Some(index))) =
                            instruction_chain(result, global, spirv, [inst_load], sampler)
                        {
                            variable
                                .reqs
                                .sampler_with_images
                                .entry(index)
                                .or_default()
                                .insert(identifier);
                        }
                    }

                    &Instruction::Store { pointer, .. } => {
                        if let Some((variable, Some(index))) =
                            instruction_chain(result, global, spirv, [], pointer)
                        {
                            variable.reqs.storage_write.insert(index);
                        }
                    }

                    _ => (),
                }
            }
        }
    }

    let mut result = FnvHashMap::default();
    let mut inspected_functions = FnvHashSet::default();
    inspect_entry_point_r(
        &mut result,
        &mut inspected_functions,
        global,
        spirv,
        entry_point,
    );
    result
        .into_iter()
        .map(|(variable_id, variable)| ((variable.set, variable.binding), variable.reqs))
        .collect()
}

/// Returns a `DescriptorRequirements` value for the pointed type.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_requirements_of(spirv: &Spirv, variable_id: Id) -> DescriptorVariable {
    let variable_id_info = spirv.id(variable_id);

    let mut reqs = DescriptorRequirements {
        descriptor_count: 1,
        ..Default::default()
    };

    let (mut next_type_id, is_storage_buffer) = {
        let variable_type_id = match variable_id_info.instruction() {
            Instruction::Variable { result_type_id, .. } => *result_type_id,
            _ => panic!("Id {} is not a variable", variable_id),
        };

        match spirv.id(variable_type_id).instruction() {
            Instruction::TypePointer {
                ty, storage_class, ..
            } => (Some(*ty), *storage_class == StorageClass::StorageBuffer),
            _ => panic!(
                "Variable {} result_type_id does not refer to a TypePointer instruction",
                variable_id
            ),
        }
    };

    while let Some(id) = next_type_id {
        let id_info = spirv.id(id);

        next_type_id = match id_info.instruction() {
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

                if decoration_buffer_block || decoration_block && is_storage_buffer {
                    reqs.descriptor_types = vec![
                        DescriptorType::StorageBuffer,
                        DescriptorType::StorageBufferDynamic,
                    ];
                } else {
                    reqs.descriptor_types = vec![
                        DescriptorType::UniformBuffer,
                        DescriptorType::UniformBufferDynamic,
                    ];
                };

                None
            }

            &Instruction::TypeImage {
                sampled_type,
                ref dim,
                arrayed,
                ms,
                sampled,
                ref image_format,
                ..
            } => {
                assert!(sampled != 0, "Vulkan requires that variables of type OpTypeImage have a Sampled operand of 1 or 2");
                reqs.image_format = image_format.clone().into();
                reqs.image_multisampled = ms != 0;
                reqs.image_scalar_type = Some(match spirv.id(sampled_type).instruction() {
                    &Instruction::TypeInt {
                        width, signedness, ..
                    } => {
                        assert!(width == 32); // TODO: 64-bit components
                        match signedness {
                            0 => ShaderScalarType::Uint,
                            1 => ShaderScalarType::Sint,
                            _ => unreachable!(),
                        }
                    }
                    &Instruction::TypeFloat { width, .. } => {
                        assert!(width == 32); // TODO: 64-bit components
                        ShaderScalarType::Float
                    }
                    _ => unreachable!(),
                });

                match dim {
                    Dim::SubpassData => {
                        assert!(
                            reqs.image_format.is_none(),
                            "If Dim is SubpassData, Image Format must be Unknown"
                        );
                        assert!(sampled == 2, "If Dim is SubpassData, Sampled must be 2");
                        assert!(arrayed == 0, "If Dim is SubpassData, Arrayed must be 0");

                        reqs.descriptor_types = vec![DescriptorType::InputAttachment];
                    }
                    Dim::Buffer => {
                        if sampled == 1 {
                            reqs.descriptor_types = vec![DescriptorType::UniformTexelBuffer];
                        } else {
                            reqs.descriptor_types = vec![DescriptorType::StorageTexelBuffer];
                        }
                    }
                    _ => {
                        reqs.image_view_type = Some(match (dim, arrayed) {
                            (Dim::Dim1D, 0) => ImageViewType::Dim1d,
                            (Dim::Dim1D, 1) => ImageViewType::Dim1dArray,
                            (Dim::Dim2D, 0) => ImageViewType::Dim2d,
                            (Dim::Dim2D, 1) => ImageViewType::Dim2dArray,
                            (Dim::Dim3D, 0) => ImageViewType::Dim3d,
                            (Dim::Dim3D, 1) => {
                                panic!("Vulkan doesn't support arrayed 3D textures")
                            }
                            (Dim::Cube, 0) => ImageViewType::Cube,
                            (Dim::Cube, 1) => ImageViewType::CubeArray,
                            (Dim::Rect, _) => {
                                panic!("Vulkan doesn't support rectangle textures")
                            }
                            _ => unreachable!(),
                        });

                        if reqs.descriptor_types.is_empty() {
                            if sampled == 1 {
                                reqs.descriptor_types = vec![DescriptorType::SampledImage];
                            } else {
                                reqs.descriptor_types = vec![DescriptorType::StorageImage];
                            }
                        }
                    }
                }

                None
            }

            &Instruction::TypeSampler { .. } => {
                reqs.descriptor_types = vec![DescriptorType::Sampler];
                None
            }

            &Instruction::TypeSampledImage { image_type, .. } => {
                reqs.descriptor_types = vec![DescriptorType::CombinedImageSampler];
                Some(image_type)
            }

            &Instruction::TypeArray {
                element_type,
                length,
                ..
            } => {
                let len = match spirv.id(length).instruction() {
                    &Instruction::Constant { ref value, .. } => {
                        value.iter().rev().fold(0, |a, &b| (a << 32) | b as u64)
                    }
                    _ => panic!("failed to find array length"),
                };

                reqs.descriptor_count *= len as u32;
                Some(element_type)
            }

            &Instruction::TypeRuntimeArray { element_type, .. } => {
                reqs.descriptor_count = 0;
                Some(element_type)
            }

            _ => {
                let name = variable_id_info
                    .iter_name()
                    .find_map(|instruction| match instruction {
                        Instruction::Name { name, .. } => Some(name.as_str()),
                        _ => None,
                    })
                    .unwrap_or("__unnamed");

                panic!("Couldn't find relevant type for global variable `{}` (id {}, maybe unimplemented)", name, variable_id);
            }
        };
    }

    DescriptorVariable {
        set: variable_id_info
            .iter_decoration()
            .find_map(|instruction| match instruction {
                Instruction::Decorate {
                    decoration: Decoration::DescriptorSet { descriptor_set },
                    ..
                } => Some(*descriptor_set),
                _ => None,
            })
            .unwrap(),
        binding: variable_id_info
            .iter_decoration()
            .find_map(|instruction| match instruction {
                Instruction::Decorate {
                    decoration: Decoration::Binding { binding_point },
                    ..
                } => Some(*binding_point),
                _ => None,
            })
            .unwrap(),
        reqs,
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
