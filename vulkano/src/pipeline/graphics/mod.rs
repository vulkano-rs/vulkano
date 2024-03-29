//! A pipeline that performs graphics processing operations.
//!
//! Unlike a compute pipeline, which performs general-purpose work, a graphics pipeline is geared
//! specifically towards doing graphical processing. To that end, it consists of several shaders,
//! with additional state and glue logic in between, known as the pipeline's *state*.
//! The state often directly corresponds to one or more steps in the graphics pipeline. Each
//! state collection has a dedicated submodule.
//!
//! # Processing steps
//!
//! A graphics pipeline performs many separate steps, that execute more or less in sequence.
//! Due to the parallel nature of a GPU, no strict ordering guarantees may exist.
//!
//! Graphics pipelines come in two different forms:
//! - *Primitive shading* graphics pipelines, which contain a vertex shader, vertex input and input
//!   assembly state, and optionally tessellation shaders and/or a geometry shader.
//! - *Mesh shading* graphics pipelines, which contain a mesh shader, and optionally a task shader.
//!
//! These types differ in the operations that are performed in the first half of the pipeline,
//! but share a common second half. The type of a graphics pipeline is determined by whether
//! it contains a vertex or a mesh shader (it cannot contain both).
//!
//! ## Primitive shading
//!
//! 1. Vertex input and assembly: vertex input data is read from data buffers and then assembled
//!    into primitives (points, lines, triangles etc.).
//! 2. Vertex shader invocations: the vertex data of each primitive is fed as input to the vertex
//!    shader, which performs transformations on the data and generates new data as output.
//! 3. (Optional) Tessellation: primitives are subdivided by the operations of two shaders, the
//!    tessellation control and tessellation evaluation shaders. The control shader produces the
//!    tessellation level to apply for the primitive, while the evaluation shader postprocesses the
//!    newly created vertices.
//! 4. (Optional) Geometry shading: whole primitives are fed as input and processed into a new set
//!    of output primitives.
//!
//! ## Mesh shading
//!
//! 1. (Optional) Task shader invocations: the task shader is run once for each workgroup in the
//!    draw command. The task shader then spawns one or more mesh shader invocations.
//! 2. Mesh shader invocations: the mesh shader is run, either once each time it is spawned by a
//!    task shader, or if there is no task shader, once for each workgroup in the draw command. The
//!    mesh shader outputs a list of primitives (triangles etc).
//!
//! Mesh shading pipelines do not receive any vertex input; their input data is supplied entirely
//! from resources bound via descriptor sets, in combination with the x, y and z coordinates of
//! the current workgroup.
//!
//! ## Rasterization, fragment processing and output
//!
//! These steps are shared by all graphics pipeline types.
//!
//! 1. Vertex post-processing, including:
//!    - Clipping primitives to the view frustum and user-defined clipping planes.
//!    - Perspective division.
//!    - Viewport mapping.
//! 2. Rasterization: converting primitives into a two-dimensional representation. Primitives may
//!    be discarded depending on their orientation, and are then converted into a collection of
//!    fragments that are processed further.
//! 3. Fragment operations. These include invocations of the fragment shader, which generates the
//!    values to be written to the color attachment. Various testing and discarding operations can
//!    be performed both before and after the fragment shader ("early" and "late" fragment tests),
//!    including:
//!    - Discard rectangle test
//!    - Scissor test
//!    - Sample mask test
//!    - Depth bounds test
//!    - Stencil test
//!    - Depth test
//! 4. Color attachment output: the final pixel data is written to a framebuffer. Blending and
//!    logical operations can be applied to combine incoming pixel data with data already present
//!    in the framebuffer.
//!
//! # Using a graphics pipeline
//!
//! Once a graphics pipeline has been created, you can execute it by first *binding* it in a
//! command buffer, binding the necessary vertex buffers, binding any descriptor sets, setting push
//! constants, and setting any dynamic state that the pipeline may need. Then you issue a `draw`
//! command.

use self::{
    color_blend::ColorBlendState,
    conservative_rasterization::ConservativeRasterizationMode,
    depth_stencil::{DepthState, DepthStencilState},
    discard_rectangle::DiscardRectangleState,
    input_assembly::{InputAssemblyState, PrimitiveTopology},
    multisample::MultisampleState,
    rasterization::RasterizationState,
    subpass::PipelineSubpassType,
    tessellation::TessellationState,
    vertex_input::{RequiredVertexInputsVUIDs, VertexInputState},
    viewport::ViewportState,
};
use super::{
    cache::PipelineCache,
    inout_interface::{shader_interface_location_info, ShaderInterfaceLocationInfo},
    shader::inout_interface::validate_interfaces_compatible,
    DynamicState, Pipeline, PipelineBindPoint, PipelineCreateFlags, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    format::FormatFeatures,
    image::{ImageAspect, ImageAspects},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    pipeline::graphics::{
        color_blend::ColorBlendAttachmentState,
        conservative_rasterization::ConservativeRasterizationState,
        depth_stencil::{StencilOpState, StencilState},
        rasterization::{CullMode, DepthBiasState},
        subpass::PipelineRenderingCreateInfo,
        tessellation::TessellationDomainOrigin,
        vertex_input::{
            VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
        },
    },
    shader::{
        spirv::{ExecutionMode, ExecutionModel, Instruction, StorageClass},
        DescriptorBindingRequirements, ShaderStage, ShaderStages,
    },
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError, VulkanObject,
};
use ahash::{HashMap, HashSet};
use smallvec::SmallVec;
use std::{
    collections::hash_map::Entry, ffi::CString, fmt::Debug, mem::MaybeUninit, num::NonZeroU64, ptr,
    sync::Arc,
};

pub mod color_blend;
pub mod conservative_rasterization;
pub mod depth_stencil;
pub mod discard_rectangle;
pub mod input_assembly;
pub mod multisample;
pub mod rasterization;
pub mod subpass;
pub mod tessellation;
pub mod vertex_input;
pub mod viewport;
// FIXME: restore
//mod tests;

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
#[derive(Debug)]
pub struct GraphicsPipeline {
    handle: ash::vk::Pipeline,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: PipelineCreateFlags,
    shader_stages: ShaderStages,
    vertex_input_state: Option<VertexInputState>,
    input_assembly_state: Option<InputAssemblyState>,
    tessellation_state: Option<TessellationState>,
    viewport_state: Option<ViewportState>,
    rasterization_state: RasterizationState,
    multisample_state: Option<MultisampleState>,
    depth_stencil_state: Option<DepthStencilState>,
    color_blend_state: Option<ColorBlendState>,
    dynamic_state: HashSet<DynamicState>,
    layout: DeviceOwnedDebugWrapper<Arc<PipelineLayout>>,
    subpass: PipelineSubpassType,

    discard_rectangle_state: Option<DiscardRectangleState>,
    conservative_rasterization_state: Option<ConservativeRasterizationState>,

    descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    num_used_descriptor_sets: u32,
    fixed_state: HashSet<DynamicState>,
    fragment_tests_stages: Option<FragmentTestsStages>,
    mesh_is_nv: bool,
    // Note: this is only `Some` if `vertex_input_state` is `None`.
    required_vertex_inputs: Option<HashMap<u32, ShaderInterfaceLocationInfo>>,
}

impl GraphicsPipeline {
    /// Creates a new `GraphicsPipeline`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: GraphicsPipelineCreateInfo,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_new(&device, cache.as_ref().map(AsRef::as_ref), &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, cache, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        _cache: Option<&PipelineCache>,
        create_info: &GraphicsPipelineCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: GraphicsPipelineCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        let &GraphicsPipelineCreateInfo {
            flags,
            ref stages,

            ref vertex_input_state,
            ref input_assembly_state,
            ref tessellation_state,
            ref viewport_state,
            ref rasterization_state,
            ref multisample_state,
            ref depth_stencil_state,
            ref color_blend_state,
            ref dynamic_state,

            ref layout,
            ref subpass,
            ref base_pipeline,

            ref discard_rectangle_state,
            ref conservative_rasterization_state,
            _ne: _,
        } = &create_info;

        struct PerPipelineShaderStageCreateInfo {
            name_vk: CString,
            specialization_info_vk: ash::vk::SpecializationInfo,
            specialization_map_entries_vk: Vec<ash::vk::SpecializationMapEntry>,
            specialization_data_vk: Vec<u8>,
            required_subgroup_size_create_info:
                Option<ash::vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo>,
        }

        let (mut stages_vk, mut per_stage_vk): (SmallVec<[_; 5]>, SmallVec<[_; 5]>) = stages
            .iter()
            .map(|stage| {
                let &PipelineShaderStageCreateInfo {
                    flags,
                    ref entry_point,
                    ref required_subgroup_size,
                    _ne: _,
                } = stage;

                let entry_point_info = entry_point.info();
                let stage = ShaderStage::from(entry_point_info.execution_model);

                let mut specialization_data_vk: Vec<u8> = Vec::new();
                let specialization_map_entries_vk: Vec<_> = entry_point
                    .module()
                    .specialization_info()
                    .iter()
                    .map(|(&constant_id, value)| {
                        let data = value.as_bytes();
                        let offset = specialization_data_vk.len() as u32;
                        let size = data.len();
                        specialization_data_vk.extend(data);

                        ash::vk::SpecializationMapEntry {
                            constant_id,
                            offset,
                            size,
                        }
                    })
                    .collect();
                let required_subgroup_size_create_info =
                    required_subgroup_size.map(|required_subgroup_size| {
                        ash::vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
                            required_subgroup_size,
                            ..Default::default()
                        }
                    });
                (
                    ash::vk::PipelineShaderStageCreateInfo {
                        flags: flags.into(),
                        stage: stage.into(),
                        module: entry_point.module().handle(),
                        p_name: ptr::null(),
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    },
                    PerPipelineShaderStageCreateInfo {
                        name_vk: CString::new(entry_point_info.name.as_str()).unwrap(),
                        specialization_info_vk: ash::vk::SpecializationInfo {
                            map_entry_count: specialization_map_entries_vk.len() as u32,
                            p_map_entries: ptr::null(),
                            data_size: specialization_data_vk.len(),
                            p_data: ptr::null(),
                        },
                        specialization_map_entries_vk,
                        specialization_data_vk,
                        required_subgroup_size_create_info,
                    },
                )
            })
            .unzip();

        for (
            stage_vk,
            PerPipelineShaderStageCreateInfo {
                name_vk,
                specialization_info_vk,
                specialization_map_entries_vk,
                specialization_data_vk,
                required_subgroup_size_create_info,
            },
        ) in stages_vk.iter_mut().zip(per_stage_vk.iter_mut())
        {
            *stage_vk = ash::vk::PipelineShaderStageCreateInfo {
                p_next: required_subgroup_size_create_info.as_ref().map_or(
                    ptr::null(),
                    |required_subgroup_size_create_info| {
                        required_subgroup_size_create_info as *const _ as _
                    },
                ),
                p_name: name_vk.as_ptr(),
                p_specialization_info: specialization_info_vk,
                ..*stage_vk
            };

            *specialization_info_vk = ash::vk::SpecializationInfo {
                p_map_entries: specialization_map_entries_vk.as_ptr(),
                p_data: specialization_data_vk.as_ptr() as _,
                ..*specialization_info_vk
            };
        }

        let mut vertex_input_state_vk = None;
        let mut vertex_binding_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_attribute_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_binding_divisor_state_vk = None;
        let mut vertex_binding_divisor_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();

        if let Some(vertex_input_state) = vertex_input_state {
            let VertexInputState {
                bindings,
                attributes,
                _ne: _,
            } = vertex_input_state;

            vertex_binding_descriptions_vk.extend(bindings.iter().map(
                |(&binding, binding_desc)| {
                    let &VertexInputBindingDescription {
                        stride,
                        input_rate,
                        _ne: _,
                    } = binding_desc;

                    ash::vk::VertexInputBindingDescription {
                        binding,
                        stride,
                        input_rate: input_rate.into(),
                    }
                },
            ));

            vertex_attribute_descriptions_vk.extend(attributes.iter().map(
                |(&location, attribute_desc)| {
                    let &VertexInputAttributeDescription {
                        binding,
                        format,
                        offset,
                        _ne: _,
                    } = attribute_desc;

                    ash::vk::VertexInputAttributeDescription {
                        location,
                        binding,
                        format: format.into(),
                        offset,
                    }
                },
            ));

            let vertex_input_state =
                vertex_input_state_vk.insert(ash::vk::PipelineVertexInputStateCreateInfo {
                    flags: ash::vk::PipelineVertexInputStateCreateFlags::empty(),
                    vertex_binding_description_count: vertex_binding_descriptions_vk.len() as u32,
                    p_vertex_binding_descriptions: vertex_binding_descriptions_vk.as_ptr(),
                    vertex_attribute_description_count: vertex_attribute_descriptions_vk.len()
                        as u32,
                    p_vertex_attribute_descriptions: vertex_attribute_descriptions_vk.as_ptr(),
                    ..Default::default()
                });

            {
                vertex_binding_divisor_descriptions_vk.extend(
                    bindings
                        .iter()
                        .filter_map(|(&binding, binding_desc)| match binding_desc.input_rate {
                            VertexInputRate::Instance { divisor } if divisor != 1 => {
                                Some((binding, divisor))
                            }
                            _ => None,
                        })
                        .map(|(binding, divisor)| {
                            ash::vk::VertexInputBindingDivisorDescriptionEXT { binding, divisor }
                        }),
                );

                // VUID-VkPipelineVertexInputDivisorStateCreateInfoEXT-vertexBindingDivisorCount-arraylength
                if !vertex_binding_divisor_descriptions_vk.is_empty() {
                    vertex_input_state.p_next = vertex_binding_divisor_state_vk.insert(
                        ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT {
                            vertex_binding_divisor_count: vertex_binding_divisor_descriptions_vk
                                .len()
                                as u32,
                            p_vertex_binding_divisors: vertex_binding_divisor_descriptions_vk
                                .as_ptr(),
                            ..Default::default()
                        },
                    ) as *const _ as *const _;
                }
            }
        }

        let mut input_assembly_state_vk = None;

        if let Some(input_assembly_state) = input_assembly_state {
            let &InputAssemblyState {
                topology,
                primitive_restart_enable,
                _ne: _,
            } = input_assembly_state;

            let _ = input_assembly_state_vk.insert(ash::vk::PipelineInputAssemblyStateCreateInfo {
                flags: ash::vk::PipelineInputAssemblyStateCreateFlags::empty(),
                topology: topology.into(),
                primitive_restart_enable: primitive_restart_enable as ash::vk::Bool32,
                ..Default::default()
            });
        }

        let mut tessellation_state_vk = None;
        let mut tessellation_domain_origin_state_vk = None;

        if let Some(tessellation_state) = tessellation_state {
            let &TessellationState {
                patch_control_points,
                domain_origin,
                _ne: _,
            } = tessellation_state;

            let tessellation_state_vk =
                tessellation_state_vk.insert(ash::vk::PipelineTessellationStateCreateInfo {
                    flags: ash::vk::PipelineTessellationStateCreateFlags::empty(),
                    patch_control_points,
                    ..Default::default()
                });

            if domain_origin != TessellationDomainOrigin::default() {
                let tessellation_domain_origin_state_vk = tessellation_domain_origin_state_vk
                    .insert(ash::vk::PipelineTessellationDomainOriginStateCreateInfo {
                        domain_origin: domain_origin.into(),
                        ..Default::default()
                    });

                tessellation_domain_origin_state_vk.p_next = tessellation_state_vk.p_next;
                tessellation_state_vk.p_next =
                    tessellation_domain_origin_state_vk as *const _ as *const _;
            }
        }

        let mut viewport_state_vk = None;
        let mut viewports_vk: SmallVec<[_; 2]> = SmallVec::new();
        let mut scissors_vk: SmallVec<[_; 2]> = SmallVec::new();

        if let Some(viewport_state) = viewport_state {
            let ViewportState {
                viewports,
                scissors,
                _ne: _,
            } = viewport_state;

            viewports_vk.extend(viewports.iter().map(Into::into));
            scissors_vk.extend(scissors.iter().map(Into::into));

            let _ = viewport_state_vk.insert(ash::vk::PipelineViewportStateCreateInfo {
                flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
                viewport_count: viewports_vk.len() as u32,
                p_viewports: if viewports_vk.is_empty() {
                    ptr::null()
                } else {
                    viewports_vk.as_ptr()
                },
                scissor_count: scissors_vk.len() as u32,
                p_scissors: if scissors_vk.is_empty() {
                    ptr::null()
                } else {
                    scissors_vk.as_ptr()
                },
                ..Default::default()
            });
        }

        let mut rasterization_state_vk = None;
        let mut rasterization_line_state_vk = None;

        if let Some(rasterization_state) = rasterization_state {
            let &RasterizationState {
                depth_clamp_enable,
                rasterizer_discard_enable,
                polygon_mode,
                cull_mode,
                front_face,
                ref depth_bias,
                line_width,
                line_rasterization_mode,
                line_stipple,
                _ne: _,
            } = rasterization_state;

            let (
                depth_bias_enable,
                depth_bias_constant_factor,
                depth_bias_clamp,
                depth_bias_slope_factor,
            ) = if let Some(depth_bias_state) = depth_bias {
                let &DepthBiasState {
                    constant_factor,
                    clamp,
                    slope_factor,
                } = depth_bias_state;

                (ash::vk::TRUE, constant_factor, clamp, slope_factor)
            } else {
                (ash::vk::FALSE, 0.0, 0.0, 0.0)
            };

            let rasterization_state =
                rasterization_state_vk.insert(ash::vk::PipelineRasterizationStateCreateInfo {
                    flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
                    depth_clamp_enable: depth_clamp_enable as ash::vk::Bool32,
                    rasterizer_discard_enable: rasterizer_discard_enable as ash::vk::Bool32,
                    polygon_mode: polygon_mode.into(),
                    cull_mode: cull_mode.into(),
                    front_face: front_face.into(),
                    depth_bias_enable,
                    depth_bias_constant_factor,
                    depth_bias_clamp,
                    depth_bias_slope_factor,
                    line_width,
                    ..Default::default()
                });

            if device.enabled_extensions().ext_line_rasterization {
                let (stippled_line_enable, line_stipple_factor, line_stipple_pattern) =
                    if let Some(line_stipple) = line_stipple {
                        (ash::vk::TRUE, line_stipple.factor, line_stipple.pattern)
                    } else {
                        (ash::vk::FALSE, 1, 0)
                    };

                rasterization_state.p_next = rasterization_line_state_vk.insert(
                    ash::vk::PipelineRasterizationLineStateCreateInfoEXT {
                        line_rasterization_mode: line_rasterization_mode.into(),
                        stippled_line_enable,
                        line_stipple_factor,
                        line_stipple_pattern,
                        ..Default::default()
                    },
                ) as *const _ as *const _;
            }
        }

        let mut multisample_state_vk = None;

        if let Some(multisample_state) = multisample_state {
            let &MultisampleState {
                rasterization_samples,
                sample_shading,
                ref sample_mask,
                alpha_to_coverage_enable,
                alpha_to_one_enable,
                _ne: _,
            } = multisample_state;

            let (sample_shading_enable, min_sample_shading) =
                if let Some(min_sample_shading) = sample_shading {
                    (ash::vk::TRUE, min_sample_shading)
                } else {
                    (ash::vk::FALSE, 0.0)
                };

            let _ = multisample_state_vk.insert(ash::vk::PipelineMultisampleStateCreateInfo {
                flags: ash::vk::PipelineMultisampleStateCreateFlags::empty(),
                rasterization_samples: rasterization_samples.into(),
                sample_shading_enable,
                min_sample_shading,
                p_sample_mask: sample_mask as _,
                alpha_to_coverage_enable: alpha_to_coverage_enable as ash::vk::Bool32,
                alpha_to_one_enable: alpha_to_one_enable as ash::vk::Bool32,
                ..Default::default()
            });
        }

        let mut depth_stencil_state_vk = None;

        if let Some(depth_stencil_state) = depth_stencil_state {
            let &DepthStencilState {
                flags,
                ref depth,
                ref depth_bounds,
                ref stencil,
                _ne: _,
            } = depth_stencil_state;

            let (depth_test_enable, depth_write_enable, depth_compare_op) =
                if let Some(depth_state) = depth {
                    let &DepthState {
                        write_enable,
                        compare_op,
                    } = depth_state;

                    (
                        ash::vk::TRUE,
                        write_enable as ash::vk::Bool32,
                        compare_op.into(),
                    )
                } else {
                    (ash::vk::FALSE, ash::vk::FALSE, ash::vk::CompareOp::ALWAYS)
                };

            let (depth_bounds_test_enable, min_depth_bounds, max_depth_bounds) =
                if let Some(depth_bounds) = depth_bounds {
                    (ash::vk::TRUE, *depth_bounds.start(), *depth_bounds.end())
                } else {
                    (ash::vk::FALSE, 0.0, 1.0)
                };

            let (stencil_test_enable, front, back) = if let Some(stencil_state) = stencil {
                let StencilState { front, back } = stencil_state;

                let [front, back] = [front, back].map(|stencil_op_state| {
                    let &StencilOpState {
                        ops,
                        compare_mask,
                        write_mask,
                        reference,
                    } = stencil_op_state;

                    ash::vk::StencilOpState {
                        fail_op: ops.fail_op.into(),
                        pass_op: ops.pass_op.into(),
                        depth_fail_op: ops.depth_fail_op.into(),
                        compare_op: ops.compare_op.into(),
                        compare_mask,
                        write_mask,
                        reference,
                    }
                });

                (ash::vk::TRUE, front, back)
            } else {
                (ash::vk::FALSE, Default::default(), Default::default())
            };

            let _ = depth_stencil_state_vk.insert(ash::vk::PipelineDepthStencilStateCreateInfo {
                flags: flags.into(),
                depth_test_enable,
                depth_write_enable,
                depth_compare_op,
                depth_bounds_test_enable,
                stencil_test_enable,
                front,
                back,
                min_depth_bounds,
                max_depth_bounds,
                ..Default::default()
            });
        }

        let mut color_blend_state_vk = None;
        let mut color_blend_attachments_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut color_write_vk = None;
        let mut color_write_enables_vk: SmallVec<[_; 4]> = SmallVec::new();

        if let Some(color_blend_state) = color_blend_state {
            let &ColorBlendState {
                flags,
                logic_op,
                ref attachments,
                blend_constants,
                _ne: _,
            } = color_blend_state;

            color_blend_attachments_vk.extend(attachments.iter().map(
                |color_blend_attachment_state| {
                    let &ColorBlendAttachmentState {
                        blend,
                        color_write_mask,
                        color_write_enable: _,
                    } = color_blend_attachment_state;

                    let blend = if let Some(blend) = blend {
                        blend.into()
                    } else {
                        Default::default()
                    };

                    ash::vk::PipelineColorBlendAttachmentState {
                        color_write_mask: color_write_mask.into(),
                        ..blend
                    }
                },
            ));

            let (logic_op_enable, logic_op) = if let Some(logic_op) = logic_op {
                (ash::vk::TRUE, logic_op.into())
            } else {
                (ash::vk::FALSE, Default::default())
            };

            let color_blend_state_vk =
                color_blend_state_vk.insert(ash::vk::PipelineColorBlendStateCreateInfo {
                    flags: flags.into(),
                    logic_op_enable,
                    logic_op,
                    attachment_count: color_blend_attachments_vk.len() as u32,
                    p_attachments: color_blend_attachments_vk.as_ptr(),
                    blend_constants,
                    ..Default::default()
                });

            if device.enabled_extensions().ext_color_write_enable {
                color_write_enables_vk.extend(attachments.iter().map(
                    |color_blend_attachment_state| {
                        let &ColorBlendAttachmentState {
                            blend: _,
                            color_write_mask: _,
                            color_write_enable,
                        } = color_blend_attachment_state;

                        color_write_enable as ash::vk::Bool32
                    },
                ));

                color_blend_state_vk.p_next =
                    color_write_vk.insert(ash::vk::PipelineColorWriteCreateInfoEXT {
                        attachment_count: color_write_enables_vk.len() as u32,
                        p_color_write_enables: color_write_enables_vk.as_ptr(),
                        ..Default::default()
                    }) as *const _ as *const _;
            }
        }

        let dynamic_state_list_vk: SmallVec<[_; 4]> =
            dynamic_state.iter().copied().map(Into::into).collect();
        let dynamic_state_vk =
            (!dynamic_state_list_vk.is_empty()).then(|| ash::vk::PipelineDynamicStateCreateInfo {
                flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                dynamic_state_count: dynamic_state_list_vk.len() as u32,
                p_dynamic_states: dynamic_state_list_vk.as_ptr(),
                ..Default::default()
            });

        let render_pass = subpass.as_ref().unwrap();
        let mut render_pass_vk = ash::vk::RenderPass::null();
        let mut subpass_vk = 0;
        let mut color_attachment_formats_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut rendering_create_info_vk = None;

        match render_pass {
            PipelineSubpassType::BeginRenderPass(subpass) => {
                render_pass_vk = subpass.render_pass().handle();
                subpass_vk = subpass.index();
            }
            PipelineSubpassType::BeginRendering(rendering_info) => {
                let &PipelineRenderingCreateInfo {
                    view_mask,
                    ref color_attachment_formats,
                    depth_attachment_format,
                    stencil_attachment_format,
                    _ne: _,
                } = rendering_info;

                color_attachment_formats_vk.extend(
                    color_attachment_formats
                        .iter()
                        .map(|format| format.map_or(ash::vk::Format::UNDEFINED, Into::into)),
                );

                let _ = rendering_create_info_vk.insert(ash::vk::PipelineRenderingCreateInfo {
                    view_mask,
                    color_attachment_count: color_attachment_formats_vk.len() as u32,
                    p_color_attachment_formats: color_attachment_formats_vk.as_ptr(),
                    depth_attachment_format: depth_attachment_format
                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                    stencil_attachment_format: stencil_attachment_format
                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                    ..Default::default()
                });
            }
        }

        let mut discard_rectangle_state_vk = None;
        let mut discard_rectangles_vk: SmallVec<[_; 2]> = SmallVec::new();

        if let Some(discard_rectangle_state) = discard_rectangle_state {
            let DiscardRectangleState {
                mode,
                rectangles,
                _ne: _,
            } = discard_rectangle_state;

            discard_rectangles_vk.extend(rectangles.iter().map(|rect| rect.into()));

            let _ = discard_rectangle_state_vk.insert(
                ash::vk::PipelineDiscardRectangleStateCreateInfoEXT {
                    flags: ash::vk::PipelineDiscardRectangleStateCreateFlagsEXT::empty(),
                    discard_rectangle_mode: (*mode).into(),
                    discard_rectangle_count: discard_rectangles_vk.len() as u32,
                    p_discard_rectangles: discard_rectangles_vk.as_ptr(),
                    ..Default::default()
                },
            );
        }

        let mut conservative_rasterization_state_vk = None;

        if let Some(conservative_rasterization_state) = conservative_rasterization_state {
            let ConservativeRasterizationState {
                mode,
                overestimation_size,
                _ne: _,
            } = conservative_rasterization_state;

            let _ = conservative_rasterization_state_vk.insert(
                ash::vk::PipelineRasterizationConservativeStateCreateInfoEXT {
                    flags: ash::vk::PipelineRasterizationConservativeStateCreateFlagsEXT::empty(),
                    conservative_rasterization_mode: (*mode).into(),
                    extra_primitive_overestimation_size: *overestimation_size,
                    ..Default::default()
                },
            );
        }

        /*
            Create
        */

        let mut create_info_vk = ash::vk::GraphicsPipelineCreateInfo {
            flags: flags.into(),
            stage_count: stages_vk.len() as u32,
            p_stages: stages_vk.as_ptr(),
            p_vertex_input_state: vertex_input_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_input_assembly_state: input_assembly_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_tessellation_state: tessellation_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_viewport_state: viewport_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_rasterization_state: rasterization_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_multisample_state: multisample_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_depth_stencil_state: depth_stencil_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_color_blend_state: color_blend_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_dynamic_state: dynamic_state_vk
                .as_ref()
                .map(|s| s as *const _)
                .unwrap_or(ptr::null()),
            layout: layout.handle(),
            render_pass: render_pass_vk,
            subpass: subpass_vk,
            base_pipeline_handle: base_pipeline
                .as_ref()
                .map_or(ash::vk::Pipeline::null(), VulkanObject::handle),
            base_pipeline_index: -1,
            ..Default::default()
        };

        if let Some(info) = discard_rectangle_state_vk.as_mut() {
            info.p_next = create_info_vk.p_next;
            create_info_vk.p_next = info as *const _ as *const _;
        }

        if let Some(info) = conservative_rasterization_state_vk.as_mut() {
            info.p_next = create_info_vk.p_next;
            create_info_vk.p_next = info as *const _ as *const _;
        }

        if let Some(info) = rendering_create_info_vk.as_mut() {
            info.p_next = create_info_vk.p_next;
            create_info_vk.p_next = info as *const _ as *const _;
        }

        let cache_handle = match cache.as_ref() {
            Some(cache) => cache.handle(),
            None => ash::vk::PipelineCache::null(),
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_graphics_pipelines)(
                device.handle(),
                cache_handle,
                1,
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;

            output.assume_init()
        };

        // Some drivers return `VK_SUCCESS` but provide a null handle if they
        // fail to create the pipeline (due to invalid shaders, etc)
        // This check ensures that we don't create an invalid `GraphicsPipeline` instance
        if handle == ash::vk::Pipeline::null() {
            panic!("vkCreateGraphicsPipelines provided a NULL handle");
        }

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Creates a new `GraphicsPipeline` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Pipeline,
        create_info: GraphicsPipelineCreateInfo,
    ) -> Arc<Self> {
        let GraphicsPipelineCreateInfo {
            flags,
            stages,

            vertex_input_state,
            input_assembly_state,
            tessellation_state,
            viewport_state,
            rasterization_state,
            multisample_state,
            depth_stencil_state,
            color_blend_state,
            dynamic_state,

            layout,
            subpass,
            base_pipeline: _,

            discard_rectangle_state,
            conservative_rasterization_state,

            _ne: _,
        } = create_info;

        let mut shader_stages = ShaderStages::empty();
        let mut mesh_is_nv = false;
        let mut descriptor_binding_requirements: HashMap<
            (u32, u32),
            DescriptorBindingRequirements,
        > = HashMap::default();
        let mut fragment_tests_stages = None;
        let mut required_vertex_inputs = None;

        for stage in &stages {
            let &PipelineShaderStageCreateInfo {
                ref entry_point, ..
            } = stage;

            let entry_point_info = entry_point.info();
            let stage = ShaderStage::from(entry_point_info.execution_model);
            shader_stages |= stage.into();

            let spirv = entry_point.module().spirv();
            let entry_point_function = spirv.function(entry_point.id());

            match entry_point_info.execution_model {
                ExecutionModel::Vertex => {
                    if vertex_input_state.is_none() {
                        required_vertex_inputs = Some(shader_interface_location_info(
                            entry_point.module().spirv(),
                            entry_point.id(),
                            StorageClass::Input,
                        ));
                    }
                }
                ExecutionModel::MeshNV | ExecutionModel::TaskNV => mesh_is_nv = true,
                ExecutionModel::Fragment => {
                    fragment_tests_stages = Some(FragmentTestsStages::Late);

                    for instruction in entry_point_function.execution_modes() {
                        if let Instruction::ExecutionMode { mode, .. } = *instruction {
                            match mode {
                                ExecutionMode::EarlyFragmentTests => {
                                    fragment_tests_stages = Some(FragmentTestsStages::Early);
                                }
                                ExecutionMode::EarlyAndLateFragmentTestsAMD => {
                                    fragment_tests_stages = Some(FragmentTestsStages::EarlyAndLate);
                                }
                                _ => (),
                            }
                        }
                    }
                }
                _ => (),
            }

            for (&loc, reqs) in &entry_point_info.descriptor_binding_requirements {
                match descriptor_binding_requirements.entry(loc) {
                    Entry::Occupied(entry) => {
                        entry.into_mut().merge(reqs).expect("Could not produce an intersection of the shader descriptor requirements");
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(reqs.clone());
                    }
                }
            }
        }

        let num_used_descriptor_sets = descriptor_binding_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        let mut fixed_state: HashSet<DynamicState> = Default::default();

        if input_assembly_state.is_some() {
            fixed_state.extend([
                DynamicState::PrimitiveTopology,
                DynamicState::PrimitiveRestartEnable,
            ]);
        }

        if tessellation_state.is_some() {
            fixed_state.extend([DynamicState::PatchControlPoints]);
        }

        if viewport_state.is_some() {
            fixed_state.extend([
                DynamicState::Viewport,
                DynamicState::ViewportWithCount,
                DynamicState::Scissor,
                DynamicState::ScissorWithCount,
            ]);
        }

        if rasterization_state.is_some() {
            fixed_state.extend([
                DynamicState::RasterizerDiscardEnable,
                DynamicState::CullMode,
                DynamicState::FrontFace,
                DynamicState::DepthBiasEnable,
                DynamicState::DepthBias,
                DynamicState::LineWidth,
                DynamicState::LineStipple,
            ]);
        }

        if depth_stencil_state.is_some() {
            fixed_state.extend([
                DynamicState::DepthTestEnable,
                DynamicState::DepthWriteEnable,
                DynamicState::DepthCompareOp,
                DynamicState::DepthBoundsTestEnable,
                DynamicState::DepthBounds,
                DynamicState::StencilTestEnable,
                DynamicState::StencilOp,
                DynamicState::StencilCompareMask,
                DynamicState::StencilWriteMask,
                DynamicState::StencilReference,
            ]);
        }

        if color_blend_state.is_some() {
            fixed_state.extend([
                DynamicState::LogicOp,
                DynamicState::BlendConstants,
                DynamicState::ColorWriteEnable,
            ]);
        }

        if discard_rectangle_state.is_some() {
            fixed_state.extend([DynamicState::DiscardRectangle]);
        }

        if conservative_rasterization_state.is_some() {
            fixed_state.extend([
                DynamicState::ConservativeRasterizationMode,
                DynamicState::ExtraPrimitiveOverestimationSize,
            ]);
        }

        fixed_state.retain(|state| !dynamic_state.contains(state));

        Arc::new(Self {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            shader_stages,
            vertex_input_state,
            input_assembly_state,
            tessellation_state,
            viewport_state,
            // Can be None for pipeline libraries, but we don't support that yet
            rasterization_state: rasterization_state.unwrap(),
            multisample_state,
            depth_stencil_state,
            color_blend_state,
            dynamic_state,
            layout: DeviceOwnedDebugWrapper(layout),
            subpass: subpass.unwrap(),

            discard_rectangle_state,
            conservative_rasterization_state,

            descriptor_binding_requirements,
            num_used_descriptor_sets,
            fixed_state,
            fragment_tests_stages,
            mesh_is_nv,
            required_vertex_inputs,
        })
    }

    /// Returns the device used to create this pipeline.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the flags that the pipeline was created with.
    #[inline]
    pub fn flags(&self) -> PipelineCreateFlags {
        self.flags
    }

    /// Returns the shader stages that this pipeline contains.
    #[inline]
    pub fn shader_stages(&self) -> ShaderStages {
        self.shader_stages
    }

    #[inline]
    pub(crate) fn mesh_is_nv(&self) -> bool {
        self.mesh_is_nv
    }

    /// Returns the vertex input state used to create this pipeline.
    #[inline]
    pub fn vertex_input_state(&self) -> Option<&VertexInputState> {
        self.vertex_input_state.as_ref()
    }

    /// Returns the input assembly state used to create this pipeline.
    #[inline]
    pub fn input_assembly_state(&self) -> Option<&InputAssemblyState> {
        self.input_assembly_state.as_ref()
    }

    /// Returns the tessellation state used to create this pipeline.
    #[inline]
    pub fn tessellation_state(&self) -> Option<&TessellationState> {
        self.tessellation_state.as_ref()
    }

    /// Returns the viewport state used to create this pipeline.
    #[inline]
    pub fn viewport_state(&self) -> Option<&ViewportState> {
        self.viewport_state.as_ref()
    }

    /// Returns the rasterization state used to create this pipeline.
    #[inline]
    pub fn rasterization_state(&self) -> &RasterizationState {
        &self.rasterization_state
    }

    /// Returns the multisample state used to create this pipeline.
    #[inline]
    pub fn multisample_state(&self) -> Option<&MultisampleState> {
        self.multisample_state.as_ref()
    }

    /// Returns the depth/stencil state used to create this pipeline.
    #[inline]
    pub fn depth_stencil_state(&self) -> Option<&DepthStencilState> {
        self.depth_stencil_state.as_ref()
    }

    /// Returns the color blend state used to create this pipeline.
    #[inline]
    pub fn color_blend_state(&self) -> Option<&ColorBlendState> {
        self.color_blend_state.as_ref()
    }

    /// Returns the subpass this graphics pipeline is rendering to.
    #[inline]
    pub fn subpass(&self) -> &PipelineSubpassType {
        &self.subpass
    }

    /// Returns the dynamic states of the pipeline.
    #[inline]
    pub fn dynamic_state(&self) -> &HashSet<DynamicState> {
        &self.dynamic_state
    }

    /// Returns the discard rectangle state used to create this pipeline.
    #[inline]
    pub fn discard_rectangle_state(&self) -> Option<&DiscardRectangleState> {
        self.discard_rectangle_state.as_ref()
    }

    /// Returns the conservative rasterization state used to create this pipeline.
    #[inline]
    pub fn conservative_rasterization_state(&self) -> Option<&ConservativeRasterizationState> {
        self.conservative_rasterization_state.as_ref()
    }

    /// If the pipeline has a fragment shader, returns the fragment tests stages used.
    #[inline]
    pub fn fragment_tests_stages(&self) -> Option<FragmentTestsStages> {
        self.fragment_tests_stages
    }

    /// Returns the dynamic states that are not dynamic in this pipeline.
    #[inline]
    pub(crate) fn fixed_state(&self) -> &HashSet<DynamicState> {
        &self.fixed_state
    }

    /// Returns the required vertex inputs.
    #[inline]
    pub(crate) fn required_vertex_inputs(
        &self,
    ) -> Option<&HashMap<u32, ShaderInterfaceLocationInfo>> {
        self.required_vertex_inputs.as_ref()
    }
}

impl Pipeline for GraphicsPipeline {
    #[inline]
    fn bind_point(&self) -> PipelineBindPoint {
        PipelineBindPoint::Graphics
    }

    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }

    #[inline]
    fn num_used_descriptor_sets(&self) -> u32 {
        self.num_used_descriptor_sets
    }

    #[inline]
    fn descriptor_binding_requirements(
        &self,
    ) -> &HashMap<(u32, u32), DescriptorBindingRequirements> {
        &self.descriptor_binding_requirements
    }
}

unsafe impl DeviceOwned for GraphicsPipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for GraphicsPipeline {
    type Handle = ash::vk::Pipeline;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

impl Drop for GraphicsPipeline {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_pipeline)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

impl_id_counter!(GraphicsPipeline);

/// Parameters to create a new `GraphicsPipeline`.
#[derive(Clone, Debug)]
pub struct GraphicsPipelineCreateInfo {
    /// Additional properties of the pipeline.
    ///
    /// The default value is empty.
    pub flags: PipelineCreateFlags,

    /// The shader stages to use.
    ///
    /// Either a vertex shader or mesh shader must always be included. Other stages are optional.
    ///
    /// The default value is empty.
    pub stages: SmallVec<[PipelineShaderStageCreateInfo; 5]>,

    /// The vertex input state.
    ///
    /// This must be `Some` if `stages` contains a vertex shader.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    pub vertex_input_state: Option<VertexInputState>,

    /// The input assembly state.
    ///
    /// This must be `Some` if `stages` contains a vertex shader.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    pub input_assembly_state: Option<InputAssemblyState>,

    /// The tessellation state.
    ///
    /// This must be `Some` if `stages` contains tessellation shaders.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    pub tessellation_state: Option<TessellationState>,

    /// The viewport state.
    ///
    /// This must be `Some` if [rasterizer discarding] is not enabled.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub viewport_state: Option<ViewportState>,

    /// The rasterization state.
    ///
    /// This must always be `Some`.
    ///
    /// The default value is `None`.
    pub rasterization_state: Option<RasterizationState>,

    /// The multisample state.
    ///
    /// This must be `Some` if [rasterizer discarding] is not enabled.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub multisample_state: Option<MultisampleState>,

    /// The depth/stencil state.
    ///
    /// This must be `Some` if `render_pass` has depth/stencil attachments, or if
    /// [rasterizer discarding] is enabled.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub depth_stencil_state: Option<DepthStencilState>,

    /// The color blend state.
    ///
    /// This must be `Some` if `render_pass` has color attachments, and [rasterizer discarding] is
    /// not enabled.
    /// It must be `None` otherwise.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub color_blend_state: Option<ColorBlendState>,

    /// The state(s) that will be set dynamically when recording a command buffer.
    ///
    /// The default value is empty.
    pub dynamic_state: HashSet<DynamicState>,

    /// The pipeline layout to use for the pipeline.
    ///
    /// There is no default value.
    pub layout: Arc<PipelineLayout>,

    /// The render subpass to use.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    pub subpass: Option<PipelineSubpassType>,

    /// The pipeline to use as a base when creating this pipeline.
    ///
    /// If this is `Some`, then `flags` must contain [`PipelineCreateFlags::DERIVATIVE`],
    /// and the `flags` of the provided pipeline must contain
    /// [`PipelineCreateFlags::ALLOW_DERIVATIVES`].
    ///
    /// The default value is `None`.
    pub base_pipeline: Option<Arc<GraphicsPipeline>>,

    /// The discard rectangle state.
    ///
    /// The default value is `None`.
    pub discard_rectangle_state: Option<DiscardRectangleState>,

    /// The conservative rasterization state.
    ///
    /// The default value is `None`.
    pub conservative_rasterization_state: Option<ConservativeRasterizationState>,

    pub _ne: crate::NonExhaustive,
}

impl GraphicsPipelineCreateInfo {
    /// Returns a `GraphicsPipelineCreateInfo` with the specified `layout`.
    #[inline]
    pub fn layout(layout: Arc<PipelineLayout>) -> Self {
        Self {
            flags: PipelineCreateFlags::empty(),
            stages: SmallVec::new(),

            vertex_input_state: None,
            input_assembly_state: None,
            tessellation_state: None,
            viewport_state: None,
            rasterization_state: None,
            multisample_state: None,
            depth_stencil_state: None,
            color_blend_state: None,
            dynamic_state: Default::default(),

            layout,
            subpass: None,
            base_pipeline: None,

            discard_rectangle_state: None,
            conservative_rasterization_state: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref stages,

            ref vertex_input_state,
            ref input_assembly_state,
            ref tessellation_state,
            ref viewport_state,
            ref rasterization_state,
            ref multisample_state,
            ref depth_stencil_state,
            ref color_blend_state,
            ref dynamic_state,

            ref layout,
            ref subpass,
            ref base_pipeline,

            ref discard_rectangle_state,
            ref conservative_rasterization_state,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkGraphicsPipelineCreateInfo-flags-parameter"])
        })?;

        if flags.intersects(PipelineCreateFlags::DERIVATIVE) {
            let base_pipeline = base_pipeline.as_ref().ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "`flags` contains `PipelineCreateFlags::DERIVATIVE`, but \
                        `base_pipeline` is `None`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-flags-07984"],
                    ..Default::default()
                })
            })?;

            if !base_pipeline
                .flags()
                .intersects(PipelineCreateFlags::ALLOW_DERIVATIVES)
            {
                return Err(Box::new(ValidationError {
                    context: "base_pipeline.flags()".into(),
                    problem: "does not contain `PipelineCreateFlags::ALLOW_DERIVATIVES`".into(),
                    vuids: &["VUID-vkCreateGraphicsPipelines-flags-00721"],
                    ..Default::default()
                }));
            }
        } else if base_pipeline.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`flags` does not contain `PipelineCreateFlags::DERIVATIVE`, but \
                    `base_pipeline` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        /*
            Gather shader stages
        */

        const PRIMITIVE_SHADING_STAGES: ShaderStages = ShaderStages::VERTEX
            .union(ShaderStages::TESSELLATION_CONTROL)
            .union(ShaderStages::TESSELLATION_CONTROL)
            .union(ShaderStages::GEOMETRY);
        const MESH_SHADING_STAGES: ShaderStages = ShaderStages::MESH.union(ShaderStages::TASK);

        let mut stages_present = ShaderStages::empty();

        for stage in stages {
            let stage_flag =
                ShaderStages::from(ShaderStage::from(stage.entry_point.info().execution_model));

            if stages_present.intersects(stage_flag) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: format!(
                        "contains more than one element whose stage is \
                        `ShaderStage::{:?}`",
                        stage_flag
                    )
                    .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-stage-06897"],
                    ..Default::default()
                }));
            }

            stages_present |= stage_flag;
        }

        /*
            Validate needed/unused state
        */

        let need_pre_rasterization_shader_state = true;

        // Check this first because everything else depends on it.
        match (
            rasterization_state.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization shader state, but \
                        `rasterization_state` is `Some`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        pre-rasterization shader state, but \
                        `rasterization_state` is `None`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
            _ => (),
        }

        let need_vertex_input_state =
            need_pre_rasterization_shader_state && stages_present.intersects(ShaderStages::VERTEX);
        let need_fragment_shader_state = need_pre_rasterization_shader_state
            && (!rasterization_state
                .as_ref()
                .unwrap()
                .rasterizer_discard_enable
                || dynamic_state.contains(&DynamicState::RasterizerDiscardEnable));
        let need_fragment_output_state = need_pre_rasterization_shader_state
            && (!rasterization_state
                .as_ref()
                .unwrap()
                .rasterizer_discard_enable
                || dynamic_state.contains(&DynamicState::RasterizerDiscardEnable));

        if need_pre_rasterization_shader_state {
            if !stages_present.intersects(ShaderStages::VERTEX | ShaderStages::MESH) {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        pre-rasterization shader state, but `stages` does not contain a \
                        `ShaderStage::Vertex` or `ShaderStage::Mesh` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-stage-02096"],
                    ..Default::default()
                }));
            }
        } else {
            if stages_present.intersects(PRIMITIVE_SHADING_STAGES | MESH_SHADING_STAGES) {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization shader state, but `stages` contains a \
                        pre-rasterization shader stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06895"],
                    ..Default::default()
                }));
            }
        }

        match (
            stages_present.intersects(ShaderStages::FRAGMENT),
            need_fragment_shader_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        fragment shader state, but `stages` contains a \
                        `ShaderStage::Geometry` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06894"],
                    ..Default::default()
                }));
            }
            (false, true) => (),
            _ => (),
        }

        match (
            vertex_input_state.is_some(),
            need_vertex_input_state && !dynamic_state.contains(&DynamicState::VertexInput),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with vertex input state, or \
                        `dynamic_state` includes `DynamicState::VertexInput`, but \
                        `vertex_input_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with vertex input state, and \
                        `dynamic_state` does not include `DynamicState::VertexInput`, but \
                        `vertex_input_state` is `None`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-02097"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (input_assembly_state.is_some(), need_vertex_input_state) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        vertex input state, but \
                        `input_assembly_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        vertex input state, but \
                        `input_assembly_state` is `None`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-02098"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            tessellation_state.is_some(),
            need_pre_rasterization_shader_state
                && stages_present.contains(
                    ShaderStages::TESSELLATION_CONTROL | ShaderStages::TESSELLATION_EVALUATION,
                ),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization state, or \
                        `stages` does not contain tessellation shader stages, but \
                        `tessellation_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        pre-rasterization state, and \
                        `stages` contains tessellation shader stages, but \
                        `tessellation_state` is `None`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-00731"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            viewport_state.is_some(),
            need_pre_rasterization_shader_state
                && (!rasterization_state
                    .as_ref()
                    .unwrap()
                    .rasterizer_discard_enable
                    || dynamic_state.contains(&DynamicState::RasterizerDiscardEnable)),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization state, or \
                        `rasterization_state.rasterization_discard_enable` is `true`, but \
                        `viewport_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        pre-rasterization state, and \
                        `rasterization_state.rasterization_discard_enable` is `false` \
                        or dynamic, but `viewport_state` is `None`"
                        .into(),
                    vuids: &[
                        "VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00750",
                        "VUID-VkGraphicsPipelineCreateInfo-pViewportState-04892",
                    ],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (multisample_state.is_some(), need_fragment_output_state) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        fragment output state, but \
                        `multisample_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        fragment output state, but \
                        `multisample_state` is `None`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00751"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            depth_stencil_state.is_some(),
            !need_fragment_output_state
                || match subpass {
                    Some(PipelineSubpassType::BeginRenderPass(subpass)) => {
                        subpass.subpass_desc().depth_stencil_attachment.is_some()
                    }
                    Some(PipelineSubpassType::BeginRendering(rendering_info)) => {
                        rendering_info.depth_attachment_format.is_some()
                            || rendering_info.stencil_attachment_format.is_some()
                    }
                    None => false,
                },
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        fragment output state, and \
                        `subpass` does not have a depth/stencil attachment, but \
                        `depth_stencil_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        fragment output state, or \
                        `subpass` has a depth/stencil attachment, but \
                        `depth_stencil_state` is `None`"
                        .into(),
                    vuids: &[
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06590",
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06043",
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06053",
                    ],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            color_blend_state.is_some(),
            need_fragment_output_state
                && match subpass {
                    Some(PipelineSubpassType::BeginRenderPass(subpass)) => {
                        !subpass.subpass_desc().color_attachments.is_empty()
                    }
                    Some(PipelineSubpassType::BeginRendering(rendering_info)) => {
                        !rendering_info.color_attachment_formats.is_empty()
                    }
                    None => false,
                },
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        fragment output state, or \
                        `subpass` does not have any color attachments, but \
                        `color_blend_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        fragment output state, and \
                        `subpass` has a color attachment, but \
                        `color_blend_state` is `None`"
                        .into(),
                    vuids: &[
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06044",
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06054",
                    ],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            subpass.is_some(),
            need_pre_rasterization_shader_state
                || need_fragment_shader_state
                || need_fragment_output_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization, fragment shader or fragment output state, but \
                        `subpass` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        pre-rasterization, fragment shader or fragment output state, but \
                        `subpass` is `None`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06575"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            discard_rectangle_state.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization state, but \
                        `discard_rectangle_state` is `Some`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04058"],
                    ..Default::default()
                }));
            }
            (false, true) => (),
            _ => (),
        }

        /*
            Validate shader stages individually
        */

        let mut has_mesh_ext = false;
        let mut has_mesh_nv = false;
        let mut vertex_stage = None;
        let mut tessellation_control_stage = None;
        let mut tessellation_evaluation_stage = None;
        let mut geometry_stage = None;
        let mut task_stage = None;
        let mut mesh_stage = None;
        let mut fragment_stage = None;

        for (stage_index, stage) in stages.iter().enumerate() {
            stage
                .validate(device)
                .map_err(|err| err.add_context(format!("stages[{}]", stage_index)))?;

            let &PipelineShaderStageCreateInfo {
                flags: _,
                ref entry_point,
                required_subgroup_size: _vk,
                _ne: _,
            } = stage;

            let entry_point_info = entry_point.info();
            let execution_model = entry_point_info.execution_model;

            match execution_model {
                ExecutionModel::TaskEXT | ExecutionModel::MeshEXT => {
                    has_mesh_ext = true;
                }
                ExecutionModel::TaskNV | ExecutionModel::MeshNV => {
                    has_mesh_nv = true;
                }
                _ => (),
            }

            let stage_enum = ShaderStage::from(execution_model);
            let stage_slot = match stage_enum {
                ShaderStage::Vertex => &mut vertex_stage,
                ShaderStage::TessellationControl => &mut tessellation_control_stage,
                ShaderStage::TessellationEvaluation => &mut tessellation_evaluation_stage,
                ShaderStage::Geometry => &mut geometry_stage,
                ShaderStage::Task => &mut task_stage,
                ShaderStage::Mesh => &mut mesh_stage,
                ShaderStage::Fragment => &mut fragment_stage,
                _ => {
                    return Err(Box::new(ValidationError {
                        context: format!("stages[{}]", stage_index).into(),
                        problem: "is not a pre-rasterization or fragment shader stage".into(),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06896"],
                        ..Default::default()
                    }));
                }
            };
            *stage_slot = Some(stage);

            layout
                .ensure_compatible_with_shader(
                    entry_point_info
                        .descriptor_binding_requirements
                        .iter()
                        .map(|(k, v)| (*k, v)),
                    entry_point_info.push_constant_requirements.as_ref(),
                )
                .map_err(|err| {
                    Box::new(ValidationError {
                        context: format!("stages[{}].entry_point", stage_index).into(),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-layout-00756"],
                        ..ValidationError::from_error(err)
                    })
                })?;
        }

        if stages_present.intersects(PRIMITIVE_SHADING_STAGES)
            && stages_present.intersects(MESH_SHADING_STAGES)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains both primitive shading stages and mesh shading stages".into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-02095"],
                ..Default::default()
            }));
        }

        if stages_present.intersects(ShaderStages::TESSELLATION_CONTROL)
            && !stages_present.intersects(ShaderStages::TESSELLATION_EVALUATION)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains a `ShaderStage::TessellationControl` stage, but not a \
                    `ShaderStage::TessellationEvaluation` stage"
                    .into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-00729"],
                ..Default::default()
            }));
        } else if stages_present.intersects(ShaderStages::TESSELLATION_EVALUATION)
            && !stages_present.intersects(ShaderStages::TESSELLATION_CONTROL)
        {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains a `ShaderStage::TessellationEvaluation` stage, but not a \
                    `ShaderStage::TessellationControl` stage"
                    .into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-00730"],
                ..Default::default()
            }));
        }

        if has_mesh_ext && has_mesh_nv {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains mesh shader stages from both the EXT and the NV version".into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-TaskNV-07063"],
                ..Default::default()
            }));
        }

        // VUID-VkGraphicsPipelineCreateInfo-layout-01688
        // Checked at pipeline layout creation time.

        /*
            Check compatibility between shader interfaces
        */

        let ordered_stages: SmallVec<[_; 7]> = [
            vertex_stage,
            tessellation_control_stage,
            tessellation_evaluation_stage,
            geometry_stage,
            task_stage,
            mesh_stage,
            fragment_stage,
        ]
        .into_iter()
        .flatten()
        .collect();

        for (output, input) in ordered_stages.iter().zip(ordered_stages.iter().skip(1)) {
            let out_spirv = output.entry_point.module().spirv();
            let (out_execution_model, out_interface) =
                match out_spirv.function(output.entry_point.id()).entry_point() {
                    Some(&Instruction::EntryPoint {
                        execution_model,
                        ref interface,
                        ..
                    }) => (execution_model, interface),
                    _ => unreachable!(),
                };

            let in_spirv = input.entry_point.module().spirv();
            let (in_execution_model, in_interface) =
                match in_spirv.function(input.entry_point.id()).entry_point() {
                    Some(&Instruction::EntryPoint {
                        execution_model,
                        ref interface,
                        ..
                    }) => (execution_model, interface),
                    _ => unreachable!(),
                };

            validate_interfaces_compatible(
                out_spirv,
                out_execution_model,
                out_interface,
                in_spirv,
                in_execution_model,
                in_interface,
                device.enabled_features().maintenance4,
            )
            .map_err(|mut err| {
                err.context = "stages".into();
                err.problem = format!(
                    "the output interface of the `{:?}` stage is not compatible with \
                    the input interface of the `{:?}` stage: {}",
                    ShaderStage::from(out_execution_model),
                    ShaderStage::from(in_execution_model),
                    err.problem
                )
                .into();
                err
            })?;
        }

        /*
            Validate states individually
        */

        if let Some(vertex_input_state) = vertex_input_state {
            vertex_input_state
                .validate(device)
                .map_err(|err| err.add_context("vertex_input_state"))?;
        }

        if let Some(input_assembly_state) = input_assembly_state {
            input_assembly_state
                .validate(device)
                .map_err(|err| err.add_context("input_assembly_state"))?;
        }

        if let Some(tessellation_state) = tessellation_state {
            tessellation_state
                .validate(device)
                .map_err(|err| err.add_context("tessellation_state"))?;
        }

        if let Some(viewport_state) = viewport_state {
            viewport_state
                .validate(device)
                .map_err(|err| err.add_context("viewport_state"))?;
        }

        if let Some(rasterization_state) = rasterization_state {
            rasterization_state
                .validate(device)
                .map_err(|err| err.add_context("rasterization_state"))?;
        }

        if let Some(multisample_state) = multisample_state {
            multisample_state
                .validate(device)
                .map_err(|err| err.add_context("multisample_state"))?;
        }

        if let Some(depth_stencil_state) = depth_stencil_state {
            depth_stencil_state
                .validate(device)
                .map_err(|err| err.add_context("depth_stencil_state"))?;
        }

        if let Some(color_blend_state) = color_blend_state {
            color_blend_state
                .validate(device)
                .map_err(|err| err.add_context("color_blend_state"))?;
        }

        if let Some(subpass) = subpass {
            match subpass {
                PipelineSubpassType::BeginRenderPass(subpass) => {
                    // VUID-VkGraphicsPipelineCreateInfo-commonparent
                    assert_eq!(device, subpass.render_pass().device().as_ref());
                }
                PipelineSubpassType::BeginRendering(rendering_info) => {
                    if !device.enabled_features().dynamic_rendering {
                        return Err(Box::new(ValidationError {
                            context: "subpass".into(),
                            problem: "is `PipelineRenderPassType::BeginRendering`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("dynamic_rendering"),
                            ])]),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576"],
                        }));
                    }

                    rendering_info
                        .validate(device)
                        .map_err(|err| err.add_context("subpass"))?;
                }
            }
        }

        if let Some(discard_rectangle_state) = discard_rectangle_state {
            if !device.enabled_extensions().ext_discard_rectangles {
                return Err(Box::new(ValidationError {
                    context: "discard_rectangle_state".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_discard_rectangles",
                    )])]),
                    ..Default::default()
                }));
            }

            discard_rectangle_state
                .validate(device)
                .map_err(|err| err.add_context("discard_rectangle_state"))?;
        }

        if let Some(conservative_rasterization_state) = conservative_rasterization_state {
            if !device.enabled_extensions().ext_conservative_rasterization {
                return Err(Box::new(ValidationError {
                    context: "conservative_rasterization_state".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_conservative_rasterization",
                    )])]),
                    ..Default::default()
                }));
            }

            conservative_rasterization_state
                .validate(device)
                .map_err(|err| err.add_context("conservative_rasterization_state"))?;
        }

        for dynamic_state in dynamic_state.iter().copied() {
            dynamic_state.validate_device(device).map_err(|err| {
                err.add_context("dynamic_state")
                    .set_vuids(&["VUID-VkPipelineDynamicStateCreateInfo-pDynamicStates-parameter"])
            })?;
        }

        /*
            Check dynamic states against other things
        */

        if stages_present.intersects(ShaderStages::MESH) {
            if dynamic_state.contains(&DynamicState::PrimitiveTopology) {
                return Err(Box::new(ValidationError {
                    problem: "`stages` includes a mesh shader, but `dynamic_state` contains \
                        `DynamicState::PrimitiveTopology`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07065"],
                    ..Default::default()
                }));
            }

            if dynamic_state.contains(&DynamicState::PrimitiveRestartEnable) {
                return Err(Box::new(ValidationError {
                    problem: "`stages` includes a mesh shader, but `dynamic_state` contains \
                        `DynamicState::PrimitiveRestartEnable`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07066"],
                    ..Default::default()
                }));
            }

            if dynamic_state.contains(&DynamicState::PatchControlPoints) {
                return Err(Box::new(ValidationError {
                    problem: "`stages` includes a mesh shader, but `dynamic_state` contains \
                        `DynamicState::PatchControlPoints`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07066"],
                    ..Default::default()
                }));
            }

            if dynamic_state.contains(&DynamicState::VertexInput) {
                return Err(Box::new(ValidationError {
                    problem: "`stages` includes a mesh shader, but `dynamic_state` contains \
                        `DynamicState::VertexInput`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07067"],
                    ..Default::default()
                }));
            }
        }

        if let Some(_input_assembly_state) = input_assembly_state {
            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-topology-00737
        }

        if let Some(viewport_state) = viewport_state {
            let ViewportState {
                ref viewports,
                ref scissors,
                _ne: _,
            } = viewport_state;

            if dynamic_state.contains(&DynamicState::ViewportWithCount) {
                if !viewports.is_empty() {
                    return Err(Box::new(ValidationError {
                        problem: "`dynamic_state` contains \
                            `DynamicState::ViewportWithCount`, but \
                            `viewport_state.viewports` is not empty"
                            .into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135"],
                        ..Default::default()
                    }));
                }
            } else {
                if viewports.is_empty() {
                    return Err(Box::new(ValidationError {
                        problem: "`dynamic_state` does not contain \
                            `DynamicState::ViewportWithCount`, but \
                            `viewport_state.viewports` is empty"
                            .into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135"],
                        ..Default::default()
                    }));
                }
            }

            if dynamic_state.contains(&DynamicState::ScissorWithCount) {
                if !scissors.is_empty() {
                    return Err(Box::new(ValidationError {
                        problem: "`dynamic_state` contains \
                            `DynamicState::ScissorWithCount`, but \
                            `viewport_state.scissors` is not empty"
                            .into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136"],
                        ..Default::default()
                    }));
                }
            } else {
                if scissors.is_empty() {
                    return Err(Box::new(ValidationError {
                        problem: "`dynamic_state` does not contain \
                            `DynamicState::ScissorWithCount`, but \
                            `viewport_state.scissors` is empty"
                            .into(),
                        vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136"],
                        ..Default::default()
                    }));
                }
            }

            if !dynamic_state.contains(&DynamicState::ViewportWithCount)
                && !dynamic_state.contains(&DynamicState::ScissorWithCount)
                && viewports.len() != scissors.len()
            {
                return Err(Box::new(ValidationError {
                    problem: "`dynamic_state` does not contain both \
                        `DynamicState::ViewportWithCount` and `DynamicState::ScissorWithCount`, \
                        and the lengths of `viewport_state.viewports` and \
                        `viewport_state.scissors` are not equal"
                        .into(),
                    vuids: &["VUID-VkPipelineViewportStateCreateInfo-scissorCount-04134"],
                    ..Default::default()
                }));
            }

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-primitiveFragmentShadingRateWithMultipleViewports-04503
            // VUID-VkGraphicsPipelineCreateInfo-primitiveFragmentShadingRateWithMultipleViewports-04504
        }

        if let Some(rasterization_state) = rasterization_state {
            let &RasterizationState {
                depth_clamp_enable: _,
                rasterizer_discard_enable: _,
                polygon_mode: _,
                cull_mode: _,
                front_face: _,
                ref depth_bias,
                line_width,
                line_rasterization_mode: _,
                line_stipple,
                _ne: _,
            } = rasterization_state;

            if !dynamic_state.contains(&DynamicState::LineWidth)
                && line_width != 1.0
                && !device.enabled_features().wide_lines
            {
                return Err(Box::new(ValidationError {
                    context: "rasterization_state.line_width".into(),
                    problem: "is not 1.0".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "wide_lines",
                    )])]),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00749"],
                }));
            }

            if let Some(depth_bias_state) = depth_bias {
                let &DepthBiasState {
                    constant_factor: _,
                    clamp,
                    slope_factor: _,
                } = depth_bias_state;

                if !dynamic_state.contains(&DynamicState::DepthBias)
                    && clamp != 0.0
                    && !device.enabled_features().depth_bias_clamp
                {
                    return Err(Box::new(ValidationError {
                        context: "rasterization_state.depth_bias.clamp".into(),
                        problem: "is not 0.0".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("depth_bias_clamp"),
                        ])]),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00754"],
                    }));
                }
            } else if dynamic_state.contains(&DynamicState::DepthBiasEnable) {
                return Err(Box::new(ValidationError {
                    problem: "`dynamic_state` contains `DynamicState::DepthBiasEnable`, but \
                        `rasterization_state.depth_bias` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }

            if let Some(line_stipple) = line_stipple {
                if !dynamic_state.contains(&DynamicState::LineStipple)
                    && !(1..=256).contains(&line_stipple.factor)
                {
                    return Err(Box::new(ValidationError {
                        context: "rasterization_state.line_stipple.factor".into(),
                        problem: "is not between 1 and 256 inclusive".into(),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-stippledLineEnable-02767"],
                        ..Default::default()
                    }));
                }
            }

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00740
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06049
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06050
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06059
        }

        if let Some(_multisample_state) = multisample_state {
            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766
        }

        if let Some(depth_stencil_state) = depth_stencil_state {
            let &DepthStencilState {
                flags: _,
                ref depth,
                ref depth_bounds,
                ref stencil,
                _ne: _,
            } = depth_stencil_state;

            if dynamic_state.contains(&DynamicState::DepthTestEnable) && depth.is_none() {
                return Err(Box::new(ValidationError {
                    problem: "`dynamic_state` contains `DynamicState::DepthTestEnable`, but \
                        `depth_stencil_state.depth` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }

            if dynamic_state.contains(&DynamicState::DepthBoundsTestEnable)
                && depth_bounds.is_none()
            {
                return Err(Box::new(ValidationError {
                    problem: "`dynamic_state` contains `DynamicState::DepthBoundsTestEnable`, but \
                        `depth_stencil_state.depth_bounds` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }

            if dynamic_state.contains(&DynamicState::StencilTestEnable) && stencil.is_none() {
                return Err(Box::new(ValidationError {
                    problem: "`dynamic_state` contains `DynamicState::StencilTestEnable`, but \
                        `depth_stencil_state.stencil` is `None`"
                        .into(),
                    ..Default::default()
                }));
            }

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06040
        }

        /*
            Checks that rely on multiple pieces of state
        */

        if let (Some(vertex_stage), Some(vertex_input_state)) = (vertex_stage, vertex_input_state) {
            let required_vertex_inputs = shader_interface_location_info(
                vertex_stage.entry_point.module().spirv(),
                vertex_stage.entry_point.id(),
                StorageClass::Input,
            );

            vertex_input_state
                .validate_required_vertex_inputs(
                    &required_vertex_inputs,
                    RequiredVertexInputsVUIDs {
                        not_present: &["VUID-VkGraphicsPipelineCreateInfo-Input-07904"],
                        numeric_type: &["VUID-VkGraphicsPipelineCreateInfo-Input-08733"],
                        requires32: &["VUID-VkGraphicsPipelineCreateInfo-pVertexInputState-08929"],
                        requires64: &["VUID-VkGraphicsPipelineCreateInfo-pVertexInputState-08930"],
                        requires_second_half: &[
                            "VUID-VkGraphicsPipelineCreateInfo-pVertexInputState-09198",
                        ],
                    },
                )
                .map_err(|mut err| {
                    err.problem = format!(
                        "{}: {}",
                        "`vertex_input_state` does not meet the requirements \
                        of the vertex shader in `stages`",
                        err.problem,
                    )
                    .into();
                    err
                })?;
        }

        if let (Some(_), Some(_)) = (tessellation_control_stage, tessellation_evaluation_stage) {
            // FIXME: must check that the control shader and evaluation shader are compatible

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00732
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00733
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00734
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00735
        }

        if let (Some(_), Some(_)) = (tessellation_evaluation_stage, geometry_stage) {
            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00739
        }

        if let (None, Some(geometry_stage), Some(input_assembly_state)) = (
            tessellation_evaluation_stage,
            geometry_stage,
            input_assembly_state,
        ) {
            let spirv = geometry_stage.entry_point.module().spirv();
            let entry_point_function = spirv.function(geometry_stage.entry_point.id());

            let input = entry_point_function
                .execution_modes()
                .iter()
                .find_map(|instruction| {
                    if let Instruction::ExecutionMode { mode, .. } = *instruction {
                        match mode {
                            ExecutionMode::InputPoints => Some(GeometryShaderInput::Points),
                            ExecutionMode::InputLines => Some(GeometryShaderInput::Lines),
                            ExecutionMode::InputLinesAdjacency => {
                                Some(GeometryShaderInput::LinesWithAdjacency)
                            }
                            ExecutionMode::Triangles => Some(GeometryShaderInput::Triangles),
                            ExecutionMode::InputTrianglesAdjacency => {
                                Some(GeometryShaderInput::TrianglesWithAdjacency)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .unwrap();

            if !input.is_compatible_with(input_assembly_state.topology) {
                return Err(Box::new(ValidationError {
                    problem: "`input_assembly_state.topology` is not compatible with the \
                        input topology of the geometry shader"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-00738"],
                    ..Default::default()
                }));
            }
        }

        if let Some(conservative_rasterization_state) = conservative_rasterization_state {
            let properties = device.physical_device().properties();

            if matches!(
                conservative_rasterization_state.mode,
                ConservativeRasterizationMode::Disabled
            ) && !properties
                .conservative_point_and_line_rasterization
                .unwrap_or(false)
            {
                if let (None, Some(input_assembly_state)) = (geometry_stage, input_assembly_state) {
                    if !matches!(
                        conservative_rasterization_state.mode,
                        ConservativeRasterizationMode::Disabled
                    ) && matches!(
                        input_assembly_state.topology,
                        PrimitiveTopology::PointList
                            | PrimitiveTopology::LineList
                            | PrimitiveTopology::LineStrip
                    ) && (!dynamic_state.contains(&DynamicState::PrimitiveTopology)
                        || match device
                            .physical_device()
                            .properties()
                            .dynamic_primitive_topology_unrestricted
                        {
                            Some(b) => !b,
                            None => false,
                        })
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`input_assembly_state.topology` is not compatible with the \
                                conservative rasterization mode"
                                .into(),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-conservativePointAndLineRasterization-08892"],
                            ..Default::default()
                        }));
                    }
                }

                if let (Some(geometry_stage), Some(_)) = (geometry_stage, input_assembly_state) {
                    let spirv = geometry_stage.entry_point.module().spirv();
                    let entry_point_function = spirv.function(geometry_stage.entry_point.id());

                    let invalid_output =
                        entry_point_function
                            .execution_modes()
                            .iter()
                            .any(|instruction| {
                                matches!(
                                    instruction,
                                    Instruction::ExecutionMode {
                                        mode: ExecutionMode::OutputPoints
                                            | ExecutionMode::OutputLineStrip,
                                        ..
                                    },
                                )
                            });

                    if !matches!(
                        conservative_rasterization_state.mode,
                        ConservativeRasterizationMode::Disabled
                    ) && invalid_output
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the output topology of the geometry shader is not compatible with the \
                                conservative rasterization mode"
                                .into(),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-conservativePointAndLineRasterization-06760"],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(mesh_stage) = mesh_stage {
                    let spirv = mesh_stage.entry_point.module().spirv();
                    let entry_point_function = spirv.function(mesh_stage.entry_point.id());

                    let mut invalid_output = false;

                    for instruction in entry_point_function.execution_modes() {
                        if let Instruction::ExecutionMode { mode, .. } = *instruction {
                            match mode {
                                ExecutionMode::OutputPoints => {
                                    invalid_output = true;
                                    break;
                                }
                                ExecutionMode::OutputLineStrip => {
                                    invalid_output = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }

                    if !matches!(
                        conservative_rasterization_state.mode,
                        ConservativeRasterizationMode::Disabled
                    ) && invalid_output
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the output topology of the mesh shader is not compatible with the \
                                conservative rasterization mode"
                                .into(),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-conservativePointAndLineRasterization-06761"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if let (Some(fragment_stage), Some(color_blend_state), Some(subpass)) =
            (fragment_stage, color_blend_state, subpass)
        {
            let fragment_shader_outputs = shader_interface_location_info(
                fragment_stage.entry_point.module().spirv(),
                fragment_stage.entry_point.id(),
                StorageClass::Output,
            );

            color_blend_state
                .validate_required_fragment_outputs(subpass, &fragment_shader_outputs)
                .map_err(|mut err| {
                    err.problem = format!(
                        "{}: {}",
                        "the fragment shader in `stages` does not meet the requirements of \
                        `color_blend_state` and `subpass`",
                        err.problem,
                    )
                    .into();
                    err
                })?;

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-01565
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06038
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06056
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06061
        }

        if let (Some(input_assembly_state), Some(_)) = (input_assembly_state, tessellation_state) {
            if input_assembly_state.topology != PrimitiveTopology::PatchList {
                return Err(Box::new(ValidationError {
                    problem: "`stages` contains tessellation shaders, but \
                        `input_assembly_state.topology` is not `PrimitiveTopology::PatchList`"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-00736"],
                    ..Default::default()
                }));
            }
        }

        if let (Some(rasterization_state), Some(depth_stencil_state)) =
            (rasterization_state, depth_stencil_state)
        {
            if let Some(stencil_state) = &depth_stencil_state.stencil {
                if device.enabled_extensions().khr_portability_subset
                    && !device.enabled_features().separate_stencil_mask_ref
                    && rasterization_state.cull_mode == CullMode::None
                    && !dynamic_state.contains(&DynamicState::StencilReference)
                    && stencil_state.front.reference != stencil_state.back.reference
                {
                    return Err(Box::new(ValidationError {
                            problem: "this device is a portability subset device, \
                                `rasterization_state.cull_mode` is `CullMode::None`, and \
                                `depth_stencil_state.stencil.front.reference` does not equal \
                                `depth_stencil_state.stencil.back.reference`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "separate_stencil_mask_ref",
                            )])]),
                            vuids: &["VUID-VkPipelineDepthStencilStateCreateInfo-separateStencilMaskRef-04453"],
                            ..Default::default()
                        }));
                }
            }
        }

        if let (Some(multisample_state), Some(subpass)) = (multisample_state, subpass) {
            match subpass {
                PipelineSubpassType::BeginRenderPass(subpass) => {
                    if let Some(samples) = subpass.num_samples() {
                        if multisample_state.rasterization_samples != samples {
                            return Err(Box::new(ValidationError {
                                problem: "`multisample_state.rasterization_samples` does not \
                                    equal the number of samples in the color and depth/stencil \
                                    attachments of `subpass`"
                                    .into(),
                                vuids: &["VUID-VkGraphicsPipelineCreateInfo-subpass-00757"],
                                ..Default::default()
                            }));
                        }
                    }

                    // TODO:
                    // VUID-VkGraphicsPipelineCreateInfo-subpass-00758
                    // VUID-VkGraphicsPipelineCreateInfo-subpass-01505
                    // VUID-VkGraphicsPipelineCreateInfo-subpass-01411
                    // VUID-VkGraphicsPipelineCreateInfo-subpass-01412
                }
                PipelineSubpassType::BeginRendering(_) => {
                    // No equivalent VUIDs for dynamic rendering, as no sample count information
                    // is provided until `begin_rendering`.
                }
            }
        }

        if let (Some(depth_stencil_state), Some(subpass)) = (depth_stencil_state, subpass) {
            if let Some(depth_state) = &depth_stencil_state.depth {
                let has_depth_attachment = match subpass {
                    PipelineSubpassType::BeginRenderPass(subpass) => subpass
                        .subpass_desc()
                        .depth_stencil_attachment
                        .as_ref()
                        .map_or(false, |depth_stencil_attachment| {
                            subpass.render_pass().attachments()
                                [depth_stencil_attachment.attachment as usize]
                                .format
                                .aspects()
                                .intersects(ImageAspects::DEPTH)
                        }),
                    PipelineSubpassType::BeginRendering(rendering_info) => {
                        rendering_info.depth_attachment_format.is_some()
                    }
                };

                if !has_depth_attachment {
                    return Err(Box::new(ValidationError {
                        problem: "`depth_stencil_state.depth` is `Some`, but `subpass` does not \
                            have a depth attachment"
                            .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }

                if !dynamic_state.contains(&DynamicState::DepthWriteEnable)
                    && depth_state.write_enable
                {
                    match subpass {
                        PipelineSubpassType::BeginRenderPass(subpass) => {
                            if !subpass
                                .subpass_desc()
                                .depth_stencil_attachment
                                .as_ref()
                                .filter(|depth_stencil_attachment| {
                                    depth_stencil_attachment
                                        .layout
                                        .is_writable(ImageAspect::Depth)
                                })
                                .map_or(false, |depth_stencil_attachment| {
                                    subpass.render_pass().attachments()
                                        [depth_stencil_attachment.attachment as usize]
                                        .format
                                        .aspects()
                                        .intersects(ImageAspects::DEPTH)
                                })
                            {
                                return Err(Box::new(ValidationError {
                                    problem: "`depth_stencil_state.depth.write_enable` is `true`, \
                                        but `subpass` does not have a depth attachment whose \
                                        layout for the depth aspect allows writing"
                                        .into(),
                                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06039"],
                                    ..Default::default()
                                }));
                            }
                        }
                        PipelineSubpassType::BeginRendering(_) => {
                            // No VUID?
                        }
                    }
                }
            }

            if depth_stencil_state.stencil.is_some() {
                let has_stencil_attachment = match subpass {
                    PipelineSubpassType::BeginRenderPass(subpass) => subpass
                        .subpass_desc()
                        .depth_stencil_attachment
                        .as_ref()
                        .map_or(false, |depth_stencil_attachment| {
                            subpass.render_pass().attachments()
                                [depth_stencil_attachment.attachment as usize]
                                .format
                                .aspects()
                                .intersects(ImageAspects::STENCIL)
                        }),
                    PipelineSubpassType::BeginRendering(rendering_info) => {
                        rendering_info.stencil_attachment_format.is_some()
                    }
                };

                if !has_stencil_attachment {
                    return Err(Box::new(ValidationError {
                        problem: "`depth_stencil_state.stencil` is `Some`, but `subpass` does not \
                            have a stencil attachment"
                            .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
        }

        if let Some(subpass) = subpass {
            let view_mask = match subpass {
                PipelineSubpassType::BeginRenderPass(subpass) => subpass.subpass_desc().view_mask,
                PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.view_mask,
            };

            if view_mask != 0 {
                if stages_present.intersects(
                    ShaderStages::TESSELLATION_CONTROL | ShaderStages::TESSELLATION_EVALUATION,
                ) && !device.enabled_features().multiview_tessellation_shader
                {
                    return Err(Box::new(ValidationError {
                        problem: "`stages` contains tessellation shaders, and \
                            `subpass` has a non-zero `view_mask`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("multiview_tessellation_shader"),
                        ])]),
                        vuids: &[
                            "VUID-VkGraphicsPipelineCreateInfo-renderPass-06047",
                            "VUID-VkGraphicsPipelineCreateInfo-renderPass-06057",
                        ],
                        ..Default::default()
                    }));
                }

                if stages_present.intersects(ShaderStages::GEOMETRY)
                    && !device.enabled_features().multiview_geometry_shader
                {
                    return Err(Box::new(ValidationError {
                        problem: "`stages` contains a geometry shader, and \
                            `subpass` has a non-zero `view_mask`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("multiview_geometry_shader"),
                        ])]),
                        vuids: &[
                            "VUID-VkGraphicsPipelineCreateInfo-renderPass-06048",
                            "VUID-VkGraphicsPipelineCreateInfo-renderPass-06058",
                        ],
                        ..Default::default()
                    }));
                }

                if stages_present.intersects(ShaderStages::MESH)
                    && !device.enabled_features().multiview_mesh_shader
                {
                    return Err(Box::new(ValidationError {
                        problem: "`stages` contains a mesh shader, and \
                            `subpass` has a non-zero `view_mask`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                            Requires::DeviceFeature("multiview_mesh_shader"),
                        ])]),
                        vuids: &[
                            "VUID-VkGraphicsPipelineCreateInfo-renderPass-07064",
                            "VUID-VkGraphicsPipelineCreateInfo-renderPass-07720",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if let (Some(color_blend_state), Some(subpass)) = (color_blend_state, subpass) {
            let color_attachment_count = match subpass {
                PipelineSubpassType::BeginRenderPass(subpass) => {
                    subpass.subpass_desc().color_attachments.len()
                }
                PipelineSubpassType::BeginRendering(rendering_info) => {
                    rendering_info.color_attachment_formats.len()
                }
            };

            if color_attachment_count != color_blend_state.attachments.len() {
                return Err(Box::new(ValidationError {
                    problem: "the length of `color_blend_state.attachments` does not equal the \
                        number of color attachments in `subpass`"
                        .into(),
                    vuids: &[
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06042",
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06055",
                        "VUID-VkGraphicsPipelineCreateInfo-renderPass-06060",
                    ],
                    ..Default::default()
                }));
            }

            for (attachment_index, state) in color_blend_state.attachments.iter().enumerate() {
                if state.blend.is_some() {
                    let attachment_format = match subpass {
                        PipelineSubpassType::BeginRenderPass(subpass) => {
                            subpass.subpass_desc().color_attachments[attachment_index]
                                .as_ref()
                                .map(|color_attachment| {
                                    subpass.render_pass().attachments()
                                        [color_attachment.attachment as usize]
                                        .format
                                })
                        }
                        PipelineSubpassType::BeginRendering(rendering_info) => {
                            rendering_info.color_attachment_formats[attachment_index]
                        }
                    };

                    if !attachment_format.map_or(false, |format| unsafe {
                        device
                            .physical_device()
                            .format_properties_unchecked(format)
                            .potential_format_features()
                            .intersects(FormatFeatures::COLOR_ATTACHMENT_BLEND)
                    }) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`color_blend_state.attachments[{}].blend` is `Some`, but \
                                the format features of that color attachment in `subpass` \
                                do not contain `FormatFeatures::COLOR_ATTACHMENT_BLEND`",
                                attachment_index
                            )
                            .into(),
                            vuids: &[
                                "VUID-VkGraphicsPipelineCreateInfo-renderPass-06041",
                                "VUID-VkGraphicsPipelineCreateInfo-renderPass-06062",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }
}

/// The input primitive type that is expected by a geometry shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GeometryShaderInput {
    Points,
    Lines,
    LinesWithAdjacency,
    Triangles,
    TrianglesWithAdjacency,
}

impl GeometryShaderInput {
    /// Returns true if the given primitive topology can be used as input for this geometry shader.
    #[inline]
    fn is_compatible_with(self, topology: PrimitiveTopology) -> bool {
        match self {
            Self::Points => matches!(topology, PrimitiveTopology::PointList),
            Self::Lines => matches!(
                topology,
                PrimitiveTopology::LineList | PrimitiveTopology::LineStrip
            ),
            Self::LinesWithAdjacency => matches!(
                topology,
                PrimitiveTopology::LineListWithAdjacency
                    | PrimitiveTopology::LineStripWithAdjacency
            ),
            Self::Triangles => matches!(
                topology,
                PrimitiveTopology::TriangleList
                    | PrimitiveTopology::TriangleStrip
                    | PrimitiveTopology::TriangleFan,
            ),
            Self::TrianglesWithAdjacency => matches!(
                topology,
                PrimitiveTopology::TriangleListWithAdjacency
                    | PrimitiveTopology::TriangleStripWithAdjacency,
            ),
        }
    }
}

/// The fragment tests stages that will be executed in a fragment shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FragmentTestsStages {
    Early,
    Late,
    EarlyAndLate,
}
