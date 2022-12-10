// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Extraction of information from SPIR-V modules, that is needed by the rest of Vulkano.

use super::{DescriptorBindingRequirements, FragmentShaderExecution, FragmentTestsStages};
use crate::{
    descriptor_set::layout::DescriptorType,
    image::view::ImageViewType,
    pipeline::layout::PushConstantRange,
    shader::{
        spirv::{
            Capability, Decoration, Dim, ExecutionMode, ExecutionModel, Id, Instruction, Spirv,
            StorageClass,
        },
        DescriptorIdentifier, DescriptorRequirements, EntryPointInfo, GeometryShaderExecution,
        GeometryShaderInput, ShaderExecution, ShaderInterface, ShaderInterfaceEntry,
        ShaderInterfaceEntryType, ShaderScalarType, ShaderStage,
        SpecializationConstantRequirements,
    },
    DeviceSize,
};
use ahash::{HashMap, HashSet};
use std::borrow::Cow;

/// Returns an iterator of the capabilities used by `spirv`.
#[inline]
pub fn spirv_capabilities(spirv: &Spirv) -> impl Iterator<Item = &Capability> {
    spirv
        .iter_capability()
        .filter_map(|instruction| match instruction {
            Instruction::Capability { capability } => Some(capability),
            _ => None,
        })
}

/// Returns an iterator of the extensions used by `spirv`.
#[inline]
pub fn spirv_extensions(spirv: &Spirv) -> impl Iterator<Item = &str> {
    spirv
        .iter_extension()
        .filter_map(|instruction| match instruction {
            Instruction::Extension { name } => Some(name.as_str()),
            _ => None,
        })
}

/// Returns an iterator over all entry points in `spirv`, with information about the entry point.
#[inline]
pub fn entry_points(
    spirv: &Spirv,
) -> impl Iterator<Item = (String, ExecutionModel, EntryPointInfo)> + '_ {
    let interface_variables = interface_variables(spirv);

    spirv.iter_entry_point().filter_map(move |instruction| {
        let (execution_model, function_id, entry_point_name, interface) = match instruction {
            Instruction::EntryPoint {
                execution_model,
                entry_point,
                name,
                interface,
                ..
            } => (*execution_model, *entry_point, name, interface),
            _ => return None,
        };

        let execution = shader_execution(spirv, execution_model, function_id);
        let stage = ShaderStage::from(execution);

        let descriptor_binding_requirements = inspect_entry_point(
            &interface_variables.descriptor_binding,
            spirv,
            stage,
            function_id,
        );
        let push_constant_requirements = push_constant_requirements(spirv, stage);
        let specialization_constant_requirements = specialization_constant_requirements(spirv);
        let input_interface = shader_interface(
            spirv,
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
            spirv,
            interface,
            StorageClass::Output,
            matches!(execution_model, ExecutionModel::TessellationControl),
        );

        Some((
            entry_point_name.clone(),
            execution_model,
            EntryPointInfo {
                execution,
                descriptor_binding_requirements,
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
    execution_model: ExecutionModel,
    function_id: Id,
) -> ShaderExecution {
    match execution_model {
        ExecutionModel::Vertex => ShaderExecution::Vertex,

        ExecutionModel::TessellationControl => ShaderExecution::TessellationControl,

        ExecutionModel::TessellationEvaluation => ShaderExecution::TessellationEvaluation,

        ExecutionModel::Geometry => {
            let mut input = None;

            for instruction in spirv.iter_execution_mode() {
                let mode = match instruction {
                    Instruction::ExecutionMode {
                        entry_point, mode, ..
                    } if *entry_point == function_id => mode,
                    _ => continue,
                };

                match mode {
                    ExecutionMode::InputPoints => {
                        input = Some(GeometryShaderInput::Points);
                    }
                    ExecutionMode::InputLines => {
                        input = Some(GeometryShaderInput::Lines);
                    }
                    ExecutionMode::InputLinesAdjacency => {
                        input = Some(GeometryShaderInput::LinesWithAdjacency);
                    }
                    ExecutionMode::Triangles => {
                        input = Some(GeometryShaderInput::Triangles);
                    }
                    ExecutionMode::InputTrianglesAdjacency => {
                        input = Some(GeometryShaderInput::TrianglesWithAdjacency);
                    }
                    _ => (),
                }
            }

            ShaderExecution::Geometry(GeometryShaderExecution {
                input: input
                    .expect("Geometry shader does not have an input primitive ExecutionMode"),
            })
        }

        ExecutionModel::Fragment => {
            let mut fragment_tests_stages = FragmentTestsStages::Late;

            for instruction in spirv.iter_execution_mode() {
                let mode = match instruction {
                    Instruction::ExecutionMode {
                        entry_point, mode, ..
                    } if *entry_point == function_id => mode,
                    _ => continue,
                };

                #[allow(clippy::single_match)]
                match mode {
                    ExecutionMode::EarlyFragmentTests => {
                        fragment_tests_stages = FragmentTestsStages::Early;
                    }
                    /*ExecutionMode::EarlyAndLateFragmentTestsAMD => {
                        fragment_tests_stages = FragmentTestsStages::EarlyAndLate;
                    }*/
                    _ => (),
                }
            }

            ShaderExecution::Fragment(FragmentShaderExecution {
                fragment_tests_stages,
            })
        }

        ExecutionModel::GLCompute => ShaderExecution::Compute,

        ExecutionModel::RayGenerationKHR => ShaderExecution::RayGeneration,
        ExecutionModel::IntersectionKHR => ShaderExecution::Intersection,
        ExecutionModel::AnyHitKHR => ShaderExecution::AnyHit,
        ExecutionModel::ClosestHitKHR => ShaderExecution::ClosestHit,
        ExecutionModel::MissKHR => ShaderExecution::Miss,
        ExecutionModel::CallableKHR => ShaderExecution::Callable,

        ExecutionModel::TaskNV => ShaderExecution::Task,
        ExecutionModel::MeshNV => ShaderExecution::Mesh,

        ExecutionModel::Kernel => todo!(),
    }
}

#[derive(Clone, Debug, Default)]
struct InterfaceVariables {
    descriptor_binding: HashMap<Id, DescriptorBindingVariable>,
}

// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface.
#[derive(Clone, Debug)]
struct DescriptorBindingVariable {
    set: u32,
    binding: u32,
    reqs: DescriptorBindingRequirements,
}

fn interface_variables(spirv: &Spirv) -> InterfaceVariables {
    let mut variables = InterfaceVariables::default();

    for instruction in spirv.iter_global() {
        if let Instruction::Variable {
            result_id,
            result_type_id: _,
            storage_class,
            ..
        } = instruction
        {
            match storage_class {
                StorageClass::StorageBuffer
                | StorageClass::Uniform
                | StorageClass::UniformConstant => {
                    variables.descriptor_binding.insert(
                        *result_id,
                        descriptor_binding_requirements_of(spirv, *result_id),
                    );
                }
                _ => (),
            }
        }
    }

    variables
}

fn inspect_entry_point(
    global: &HashMap<Id, DescriptorBindingVariable>,
    spirv: &Spirv,
    stage: ShaderStage,
    entry_point: Id,
) -> HashMap<(u32, u32), DescriptorBindingRequirements> {
    struct Context<'a> {
        global: &'a HashMap<Id, DescriptorBindingVariable>,
        spirv: &'a Spirv,
        stage: ShaderStage,
        inspected_functions: HashSet<Id>,
        result: HashMap<Id, DescriptorBindingVariable>,
    }

    impl<'a> Context<'a> {
        fn instruction_chain<const N: usize>(
            &mut self,
            chain: [fn(&Spirv, Id) -> Option<Id>; N],
            id: Id,
        ) -> Option<(&mut DescriptorBindingVariable, Option<u32>)> {
            let id = chain
                .into_iter()
                .try_fold(id, |id, func| func(self.spirv, id))?;

            if let Some(variable) = self.global.get(&id) {
                // Variable was accessed without an access chain, return with index 0.
                let variable = self.result.entry(id).or_insert_with(|| variable.clone());
                variable.reqs.stages = self.stage.into();
                return Some((variable, Some(0)));
            }

            let (id, indexes) = match *self.spirv.id(id).instruction() {
                Instruction::AccessChain {
                    base, ref indexes, ..
                } => (base, indexes),
                _ => return None,
            };

            if let Some(variable) = self.global.get(&id) {
                // Variable was accessed with an access chain.
                // Retrieve index from instruction if it's a constant value.
                // TODO: handle a `None` index too?
                let index = match *self.spirv.id(*indexes.first().unwrap()).instruction() {
                    Instruction::Constant { ref value, .. } => Some(value[0]),
                    _ => None,
                };
                let variable = self.result.entry(id).or_insert_with(|| variable.clone());
                variable.reqs.stages = self.stage.into();
                return Some((variable, index));
            }

            None
        }

        fn inspect_entry_point_r(&mut self, function: Id) {
            fn desc_reqs(
                descriptor_variable: Option<(&mut DescriptorBindingVariable, Option<u32>)>,
            ) -> Option<&mut DescriptorRequirements> {
                descriptor_variable
                    .map(|(variable, index)| variable.reqs.descriptors.entry(index).or_default())
            }

            fn inst_image_texel_pointer(spirv: &Spirv, id: Id) -> Option<Id> {
                match *spirv.id(id).instruction() {
                    Instruction::ImageTexelPointer { image, .. } => Some(image),
                    _ => None,
                }
            }

            fn inst_load(spirv: &Spirv, id: Id) -> Option<Id> {
                match *spirv.id(id).instruction() {
                    Instruction::Load { pointer, .. } => Some(pointer),
                    _ => None,
                }
            }

            fn inst_sampled_image(spirv: &Spirv, id: Id) -> Option<Id> {
                match *spirv.id(id).instruction() {
                    Instruction::SampledImage { sampler, .. } => Some(sampler),
                    _ => Some(id),
                }
            }

            self.inspected_functions.insert(function);
            let mut in_function = false;

            for instruction in self.spirv.instructions() {
                if !in_function {
                    match *instruction {
                        Instruction::Function { result_id, .. } if result_id == function => {
                            in_function = true;
                        }
                        _ => {}
                    }
                } else {
                    let stage = self.stage;

                    match *instruction {
                        Instruction::AtomicLoad { pointer, .. } => {
                            // Storage buffer
                            if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer))
                            {
                                desc_reqs.memory_read = stage.into();
                            }

                            // Storage image
                            if let Some(desc_reqs) = desc_reqs(
                                self.instruction_chain([inst_image_texel_pointer], pointer),
                            ) {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.storage_image_atomic = true;
                            }
                        }

                        Instruction::AtomicStore { pointer, .. } => {
                            // Storage buffer
                            if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer))
                            {
                                desc_reqs.memory_write = stage.into();
                            }

                            // Storage image
                            if let Some(desc_reqs) = desc_reqs(
                                self.instruction_chain([inst_image_texel_pointer], pointer),
                            ) {
                                desc_reqs.memory_write = stage.into();
                                desc_reqs.storage_image_atomic = true;
                            }
                        }

                        Instruction::AtomicExchange { pointer, .. }
                        | Instruction::AtomicCompareExchange { pointer, .. }
                        | Instruction::AtomicCompareExchangeWeak { pointer, .. }
                        | Instruction::AtomicIIncrement { pointer, .. }
                        | Instruction::AtomicIDecrement { pointer, .. }
                        | Instruction::AtomicIAdd { pointer, .. }
                        | Instruction::AtomicISub { pointer, .. }
                        | Instruction::AtomicSMin { pointer, .. }
                        | Instruction::AtomicUMin { pointer, .. }
                        | Instruction::AtomicSMax { pointer, .. }
                        | Instruction::AtomicUMax { pointer, .. }
                        | Instruction::AtomicAnd { pointer, .. }
                        | Instruction::AtomicOr { pointer, .. }
                        | Instruction::AtomicXor { pointer, .. }
                        | Instruction::AtomicFlagTestAndSet { pointer, .. }
                        | Instruction::AtomicFlagClear { pointer, .. }
                        | Instruction::AtomicFMinEXT { pointer, .. }
                        | Instruction::AtomicFMaxEXT { pointer, .. }
                        | Instruction::AtomicFAddEXT { pointer, .. } => {
                            // Storage buffer
                            if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.memory_write = stage.into();
                            }

                            // Storage image
                            if let Some(desc_reqs) = desc_reqs(
                                self.instruction_chain([inst_image_texel_pointer], pointer),
                            ) {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.memory_write = stage.into();
                                desc_reqs.storage_image_atomic = true;
                            }
                        }

                        Instruction::CopyMemory { target, source, .. } => {
                            self.instruction_chain([], target);
                            self.instruction_chain([], source);
                        }

                        Instruction::CopyObject { operand, .. } => {
                            self.instruction_chain([], operand);
                        }

                        Instruction::ExtInst { ref operands, .. } => {
                            // We don't know which extended instructions take pointers,
                            // so we must interpret every operand as a pointer.
                            for &operand in operands {
                                self.instruction_chain([], operand);
                            }
                        }

                        Instruction::FunctionCall {
                            function,
                            ref arguments,
                            ..
                        } => {
                            // Rather than trying to figure out the type of each argument, we just
                            // try all of them as pointers.
                            for &argument in arguments {
                                self.instruction_chain([], argument);
                            }

                            if !self.inspected_functions.contains(&function) {
                                self.inspect_entry_point_r(function);
                            }
                        }

                        Instruction::FunctionEnd => return,

                        Instruction::ImageGather {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseGather {
                            sampled_image,
                            image_operands,
                            ..
                        } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.sampler_no_ycbcr_conversion = true;

                                if image_operands.as_ref().map_or(false, |image_operands| {
                                    image_operands.bias.is_some()
                                        || image_operands.const_offset.is_some()
                                        || image_operands.offset.is_some()
                                }) {
                                    desc_reqs.sampler_no_unnormalized_coordinates = true;
                                }
                            }
                        }

                        Instruction::ImageDrefGather { sampled_image, .. }
                        | Instruction::ImageSparseDrefGather { sampled_image, .. } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.sampler_no_unnormalized_coordinates = true;
                                desc_reqs.sampler_no_ycbcr_conversion = true;
                            }
                        }

                        Instruction::ImageSampleImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSampleProjImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleProjImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.sampler_no_unnormalized_coordinates = true;

                                if image_operands.as_ref().map_or(false, |image_operands| {
                                    image_operands.const_offset.is_some()
                                        || image_operands.offset.is_some()
                                }) {
                                    desc_reqs.sampler_no_ycbcr_conversion = true;
                                }
                            }
                        }

                        Instruction::ImageSampleProjExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleProjExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.sampler_no_unnormalized_coordinates = true;

                                if image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                                {
                                    desc_reqs.sampler_no_ycbcr_conversion = true;
                                }
                            }
                        }

                        Instruction::ImageSampleDrefImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSampleProjDrefImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleDrefImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleProjDrefImplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.sampler_no_unnormalized_coordinates = true;
                                desc_reqs.sampler_compare = true;

                                if image_operands.as_ref().map_or(false, |image_operands| {
                                    image_operands.const_offset.is_some()
                                        || image_operands.offset.is_some()
                                }) {
                                    desc_reqs.sampler_no_ycbcr_conversion = true;
                                }
                            }
                        }

                        Instruction::ImageSampleDrefExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSampleProjDrefExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleDrefExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleProjDrefExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();
                                desc_reqs.sampler_no_unnormalized_coordinates = true;
                                desc_reqs.sampler_compare = true;

                                if image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                                {
                                    desc_reqs.sampler_no_ycbcr_conversion = true;
                                }
                            }
                        }

                        Instruction::ImageSampleExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        }
                        | Instruction::ImageSparseSampleExplicitLod {
                            sampled_image,
                            image_operands,
                            ..
                        } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain(
                                    [inst_sampled_image, inst_load],
                                    sampled_image,
                                ))
                            {
                                desc_reqs.memory_read = stage.into();

                                if image_operands.bias.is_some()
                                    || image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                                {
                                    desc_reqs.sampler_no_unnormalized_coordinates = true;
                                }

                                if image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                                {
                                    desc_reqs.sampler_no_ycbcr_conversion = true;
                                }
                            }
                        }

                        Instruction::ImageTexelPointer { image, .. } => {
                            self.instruction_chain([], image);
                        }

                        Instruction::ImageRead { image, .. } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain([inst_load], image))
                            {
                                desc_reqs.memory_read = stage.into();
                            }
                        }

                        Instruction::ImageWrite { image, .. } => {
                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain([inst_load], image))
                            {
                                desc_reqs.memory_write = stage.into();
                            }
                        }

                        Instruction::Load { pointer, .. } => {
                            if let Some((binding_variable, index)) =
                                self.instruction_chain([], pointer)
                            {
                                // Only loads on buffers access memory directly.
                                // Loads on images load the image object itself, but don't touch
                                // the texels in memory yet.
                                if binding_variable.reqs.descriptor_types.iter().any(|ty| {
                                    matches!(
                                        ty,
                                        DescriptorType::UniformBuffer
                                            | DescriptorType::UniformBufferDynamic
                                            | DescriptorType::StorageBuffer
                                            | DescriptorType::StorageBufferDynamic
                                    )
                                }) {
                                    if let Some(desc_reqs) =
                                        desc_reqs(Some((binding_variable, index)))
                                    {
                                        desc_reqs.memory_read = stage.into();
                                    }
                                }
                            }
                        }

                        Instruction::SampledImage { image, sampler, .. } => {
                            let identifier = match self.instruction_chain([inst_load], image) {
                                Some((variable, Some(index))) => DescriptorIdentifier {
                                    set: variable.set,
                                    binding: variable.binding,
                                    index,
                                },
                                _ => continue,
                            };

                            if let Some(desc_reqs) =
                                desc_reqs(self.instruction_chain([inst_load], sampler))
                            {
                                desc_reqs.sampler_with_images.insert(identifier);
                            }
                        }

                        Instruction::Store { pointer, .. } => {
                            // This can only apply to buffers, right?
                            if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer))
                            {
                                desc_reqs.memory_write = stage.into();
                            }
                        }

                        _ => (),
                    }
                }
            }
        }
    }

    let mut context = Context {
        global,
        spirv,
        stage,
        inspected_functions: HashSet::default(),
        result: HashMap::default(),
    };
    context.inspect_entry_point_r(entry_point);

    context
        .result
        .into_iter()
        .map(|(_, variable)| ((variable.set, variable.binding), variable.reqs))
        .collect()
}

/// Returns a `DescriptorBindingRequirements` value for the pointed type.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_binding_requirements_of(spirv: &Spirv, variable_id: Id) -> DescriptorBindingVariable {
    let variable_id_info = spirv.id(variable_id);

    let mut reqs = DescriptorBindingRequirements {
        descriptor_count: Some(1),
        ..Default::default()
    };

    let (mut next_type_id, is_storage_buffer) = {
        let variable_type_id = match *variable_id_info.instruction() {
            Instruction::Variable { result_type_id, .. } => result_type_id,
            _ => panic!("Id {} is not a variable", variable_id),
        };

        match *spirv.id(variable_type_id).instruction() {
            Instruction::TypePointer {
                ty, storage_class, ..
            } => (Some(ty), storage_class == StorageClass::StorageBuffer),
            _ => panic!(
                "Variable {} result_type_id does not refer to a TypePointer instruction",
                variable_id
            ),
        }
    };

    while let Some(id) = next_type_id {
        let id_info = spirv.id(id);

        next_type_id = match *id_info.instruction() {
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
                    "Structs in shader interface are expected to be decorated with one of Block or \
                    BufferBlock",
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

            Instruction::TypeImage {
                sampled_type,
                dim,
                arrayed,
                ms,
                sampled,
                image_format,
                ..
            } => {
                assert!(
                    sampled != 0,
                    "Vulkan requires that variables of type OpTypeImage have a Sampled operand of \
                    1 or 2",
                );
                reqs.image_format = image_format.into();
                reqs.image_multisampled = ms != 0;
                reqs.image_scalar_type = Some(match *spirv.id(sampled_type).instruction() {
                    Instruction::TypeInt {
                        width, signedness, ..
                    } => {
                        assert!(width == 32); // TODO: 64-bit components
                        match signedness {
                            0 => ShaderScalarType::Uint,
                            1 => ShaderScalarType::Sint,
                            _ => unreachable!(),
                        }
                    }
                    Instruction::TypeFloat { width, .. } => {
                        assert!(width == 32); // TODO: 64-bit components
                        ShaderScalarType::Float
                    }
                    _ => unreachable!(),
                });

                match dim {
                    Dim::SubpassData => {
                        assert!(
                            reqs.image_format.is_none(),
                            "If Dim is SubpassData, Image Format must be Unknown",
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

            Instruction::TypeSampler { .. } => {
                reqs.descriptor_types = vec![DescriptorType::Sampler];

                None
            }

            Instruction::TypeSampledImage { image_type, .. } => {
                reqs.descriptor_types = vec![DescriptorType::CombinedImageSampler];

                Some(image_type)
            }

            Instruction::TypeArray {
                element_type,
                length,
                ..
            } => {
                let len = match spirv.id(length).instruction() {
                    Instruction::Constant { value, .. } => {
                        value.iter().rev().fold(0, |a, &b| (a << 32) | b as u64)
                    }
                    _ => panic!("failed to find array length"),
                };

                if let Some(count) = reqs.descriptor_count.as_mut() {
                    *count *= len as u32
                }

                Some(element_type)
            }

            Instruction::TypeRuntimeArray { element_type, .. } => {
                reqs.descriptor_count = None;

                Some(element_type)
            }

            Instruction::TypeAccelerationStructureKHR { .. } => None, // FIXME temporary workaround

            _ => {
                let name = variable_id_info
                    .iter_name()
                    .find_map(|instruction| match *instruction {
                        Instruction::Name { ref name, .. } => Some(name.as_str()),
                        _ => None,
                    })
                    .unwrap_or("__unnamed");

                panic!(
                    "Couldn't find relevant type for global variable `{}` (id {}, maybe \
                    unimplemented)",
                    name, variable_id,
                );
            }
        };
    }

    DescriptorBindingVariable {
        set: variable_id_info
            .iter_decoration()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    decoration: Decoration::DescriptorSet { descriptor_set },
                    ..
                } => Some(descriptor_set),
                _ => None,
            })
            .unwrap(),
        binding: variable_id_info
            .iter_decoration()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    decoration: Decoration::Binding { binding_point },
                    ..
                } => Some(binding_point),
                _ => None,
            })
            .unwrap(),
        reqs,
    }
}

/// Extracts the `PushConstantRange` from `spirv`.
fn push_constant_requirements(spirv: &Spirv, stage: ShaderStage) -> Option<PushConstantRange> {
    spirv
        .iter_global()
        .find_map(|instruction| match *instruction {
            Instruction::TypePointer {
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

                Some(PushConstantRange {
                    stages: stage.into(),
                    offset: start,
                    size: end - start,
                })
            }
            _ => None,
        })
}

/// Extracts the `SpecializationConstantRequirements` from `spirv`.
fn specialization_constant_requirements(
    spirv: &Spirv,
) -> HashMap<u32, SpecializationConstantRequirements> {
    spirv
        .iter_global()
        .filter_map(|instruction| {
            match *instruction {
                Instruction::SpecConstantTrue {
                    result_type_id,
                    result_id,
                }
                | Instruction::SpecConstantFalse {
                    result_type_id,
                    result_id,
                }
                | Instruction::SpecConstant {
                    result_type_id,
                    result_id,
                    ..
                }
                | Instruction::SpecConstantComposite {
                    result_type_id,
                    result_id,
                    ..
                } => spirv
                    .id(result_id)
                    .iter_decoration()
                    .find_map(|instruction| match *instruction {
                        Instruction::Decorate {
                            decoration:
                                Decoration::SpecId {
                                    specialization_constant_id,
                                },
                            ..
                        } => Some(specialization_constant_id),
                        _ => None,
                    })
                    .map(|constant_id| {
                        let size = match *spirv.id(result_type_id).instruction() {
                            Instruction::TypeBool { .. } => {
                                // Translate bool to Bool32
                                std::mem::size_of::<ash::vk::Bool32>() as DeviceSize
                            }
                            _ => size_of_type(spirv, result_type_id)
                                .expect("Found runtime-sized specialization constant"),
                        };
                        (constant_id, SpecializationConstantRequirements { size })
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
            let (result_type_id, result_id) = match *spirv.id(id).instruction() {
                Instruction::Variable {
                    result_type_id,
                    result_id,
                    storage_class,
                    ..
                } if storage_class == filter_storage_class => (result_type_id, result_id),
                _ => return None,
            };

            if is_builtin(spirv, result_id) {
                return None;
            }

            let id_info = spirv.id(result_id);

            let name = id_info
                .iter_name()
                .find_map(|instruction| match *instruction {
                    Instruction::Name { ref name, .. } => Some(Cow::Owned(name.clone())),
                    _ => None,
                });

            let location = id_info
                .iter_decoration()
                .find_map(|instruction| match *instruction {
                    Instruction::Decorate {
                        decoration: Decoration::Location { location },
                        ..
                    } => Some(location),
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
                .find_map(|instruction| match *instruction {
                    Instruction::Decorate {
                        decoration: Decoration::Component { component },
                        ..
                    } => Some(component),
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

    match *id_info.instruction() {
        Instruction::TypeBool { .. } => {
            panic!("Can't put booleans in structs")
        }
        Instruction::TypeInt { width, .. } | Instruction::TypeFloat { width, .. } => {
            assert!(width % 8 == 0);
            Some(width as DeviceSize / 8)
        }
        Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => size_of_type(spirv, component_type)
            .map(|component_size| component_size * component_count as DeviceSize),
        Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            // FIXME: row-major or column-major
            size_of_type(spirv, column_type)
                .map(|column_size| column_size * column_count as DeviceSize)
        }
        Instruction::TypeArray { length, .. } => {
            let stride = id_info
                .iter_decoration()
                .find_map(|instruction| match *instruction {
                    Instruction::Decorate {
                        decoration: Decoration::ArrayStride { array_stride },
                        ..
                    } => Some(array_stride),
                    _ => None,
                })
                .unwrap();
            let length = match spirv.id(length).instruction() {
                Instruction::Constant { value, .. } => Some(
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
        Instruction::TypeStruct {
            ref member_types, ..
        } => {
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
                        .find_map(|instruction| match *instruction {
                            Instruction::MemberDecorate {
                                decoration: Decoration::Offset { byte_offset },
                                ..
                            } => Some(byte_offset),
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
        .filter_map(|member_info| {
            member_info
                .iter_decoration()
                .find_map(|instruction| match *instruction {
                    Instruction::MemberDecorate {
                        decoration: Decoration::Offset { byte_offset },
                        ..
                    } => Some(byte_offset),
                    _ => None,
                })
        })
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
    match *spirv.id(id).instruction() {
        Instruction::TypeInt {
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
        Instruction::TypeFloat { width, .. } => {
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
        Instruction::TypeVector {
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
        Instruction::TypeMatrix {
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
        Instruction::TypeArray {
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
                    .find_map(|instruction| match *instruction {
                        Instruction::Constant {
                            result_id,
                            ref value,
                            ..
                        } if result_id == length => Some(value.clone()),
                        _ => None,
                    })
                    .expect("failed to find array length")
                    .iter()
                    .rev()
                    .fold(0u64, |a, &b| (a << 32) | b as u64)
                    as u32;
                ty.num_elements *= num_elements;
                ty
            }
        }
        Instruction::TypePointer { ty, .. } => {
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
        Instruction::Variable {
            result_type_id: ty, ..
        }
        | Instruction::TypeArray {
            element_type: ty, ..
        }
        | Instruction::TypeRuntimeArray {
            element_type: ty, ..
        }
        | Instruction::TypePointer { ty, .. } => is_builtin(spirv, *ty),
        Instruction::TypeStruct { member_types, .. } => {
            member_types.iter().any(|ty| is_builtin(spirv, *ty))
        }
        _ => false,
    }
}
