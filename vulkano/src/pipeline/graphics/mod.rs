// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pipeline that performs graphics processing operations.
//!
//! Unlike a compute pipeline, which performs general-purpose work, a graphics pipeline is geared
//! specifically towards doing graphical processing. To that end, it consists of several shaders,
//! with additional state and glue logic in between.
//!
//! A graphics pipeline performs many separate steps, that execute more or less in sequence.
//! Due to the parallel nature of a GPU, no strict ordering guarantees may exist.
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
//! 5. Vertex post-processing, including:
//!    - Clipping primitives to the view frustum and user-defined clipping planes.
//!    - Perspective division.
//!    - Viewport mapping.
//! 6. Rasterization: converting primitives into a two-dimensional representation. Primitives may be
//!    discarded depending on their orientation, and are then converted into a collection of
//!    fragments that are processed further.
//! 7. Fragment operations. These include invocations of the fragment shader, which generates the
//!    values to be written to the color attachment. Various testing and discarding operations can
//!    be performed both before and after the fragment shader ("early" and "late" fragment tests),
//!    including:
//!    - Discard rectangle test
//!    - Scissor test
//!    - Sample mask test
//!    - Depth bounds test
//!    - Stencil test
//!    - Depth test
//! 8. Color attachment output: the final pixel data is written to a framebuffer. Blending and
//!    logical operations can be applied to combine incoming pixel data with data already present
//!    in the framebuffer.
//!
//! A graphics pipeline contains many configuration options, which are grouped into collections of
//! "state". Often, these directly correspond to one or more steps in the graphics pipeline. Each
//! state collection has a dedicated submodule.
//!
//! Once a graphics pipeline has been created, you can execute it by first *binding* it in a command
//! buffer, binding the necessary vertex buffers, binding any descriptor sets, setting push
//! constants, and setting any dynamic state that the pipeline may need. Then you issue a `draw`
//! command.

use self::{
    color_blend::ColorBlendState,
    depth_stencil::{DepthBoundsState, DepthState, DepthStencilState},
    discard_rectangle::DiscardRectangleState,
    input_assembly::{InputAssemblyState, PrimitiveTopology, PrimitiveTopologyClass},
    multisample::MultisampleState,
    rasterization::RasterizationState,
    subpass::PipelineSubpassType,
    tessellation::TessellationState,
    vertex_input::VertexInputState,
    viewport::ViewportState,
};
use super::{
    cache::PipelineCache, DynamicState, Pipeline, PipelineBindPoint, PipelineCreateFlags,
    PipelineLayout, PipelineShaderStageCreateInfo, StateMode,
};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    format::{FormatFeatures, NumericType},
    image::{ImageAspect, ImageAspects},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    pipeline::{
        graphics::{
            color_blend::ColorBlendAttachmentState,
            depth_stencil::{StencilOpState, StencilState},
            rasterization::{CullMode, FrontFace},
            subpass::PipelineRenderingCreateInfo,
            tessellation::TessellationDomainOrigin,
            vertex_input::VertexInputRate,
        },
        PartialStateMode,
    },
    shader::{
        DescriptorBindingRequirements, FragmentShaderExecution, FragmentTestsStages,
        ShaderExecution, ShaderScalarType, ShaderStage, ShaderStages,
    },
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError, VulkanObject,
};
use ahash::HashMap;
use smallvec::SmallVec;
use std::{
    collections::hash_map::Entry, ffi::CString, fmt::Debug, mem::MaybeUninit, num::NonZeroU64, ptr,
    sync::Arc,
};

pub mod color_blend;
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

    // TODO: replace () with an object that describes the shaders in some way.
    shaders: HashMap<ShaderStage, ()>,
    descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    num_used_descriptor_sets: u32,
    fragment_tests_stages: Option<FragmentTestsStages>,

    vertex_input_state: VertexInputState,
    input_assembly_state: InputAssemblyState,
    tessellation_state: Option<TessellationState>,
    viewport_state: Option<ViewportState>,
    rasterization_state: RasterizationState,
    multisample_state: Option<MultisampleState>,
    depth_stencil_state: Option<DepthStencilState>,
    color_blend_state: Option<ColorBlendState>,
    layout: DeviceOwnedDebugWrapper<Arc<PipelineLayout>>,
    subpass: PipelineSubpassType,
    dynamic_state: HashMap<DynamicState, bool>,

    discard_rectangle_state: Option<DiscardRectangleState>,
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

            ref layout,
            subpass: ref render_pass,

            ref discard_rectangle_state,
            _ne: _,
        } = &create_info;

        let mut dynamic_state: HashMap<DynamicState, bool> = HashMap::default();

        struct PerPipelineShaderStageCreateInfo {
            name_vk: CString,
            specialization_info_vk: ash::vk::SpecializationInfo,
            specialization_map_entries_vk: Vec<ash::vk::SpecializationMapEntry>,
            specialization_data_vk: Vec<u8>,
        }

        let (mut stages_vk, mut per_stage_vk): (SmallVec<[_; 5]>, SmallVec<[_; 5]>) = stages
            .iter()
            .map(|stage| {
                let &PipelineShaderStageCreateInfo {
                    flags,
                    ref entry_point,
                    ref specialization_info,
                    _ne: _,
                } = stage;

                let entry_point_info = entry_point.info();
                let stage = ShaderStage::from(&entry_point_info.execution);

                let mut specialization_data_vk: Vec<u8> = Vec::new();
                let specialization_map_entries_vk: Vec<_> = specialization_info
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
            },
        ) in (stages_vk.iter_mut()).zip(per_stage_vk.iter_mut())
        {
            *stage_vk = ash::vk::PipelineShaderStageCreateInfo {
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
            dynamic_state.insert(DynamicState::VertexInput, false);

            let VertexInputState {
                bindings,
                attributes,
                _ne: _,
            } = vertex_input_state;

            vertex_binding_descriptions_vk.extend(bindings.iter().map(
                |(&binding, binding_desc)| ash::vk::VertexInputBindingDescription {
                    binding,
                    stride: binding_desc.stride,
                    input_rate: binding_desc.input_rate.into(),
                },
            ));

            vertex_attribute_descriptions_vk.extend(attributes.iter().map(
                |(&location, attribute_desc)| ash::vk::VertexInputAttributeDescription {
                    location,
                    binding: attribute_desc.binding,
                    format: attribute_desc.format.into(),
                    offset: attribute_desc.offset,
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

            let topology = match topology {
                PartialStateMode::Fixed(topology) => {
                    dynamic_state.insert(DynamicState::PrimitiveTopology, false);
                    topology.into()
                }
                PartialStateMode::Dynamic(topology_class) => {
                    dynamic_state.insert(DynamicState::PrimitiveTopology, true);
                    topology_class.example().into()
                }
            };

            let primitive_restart_enable = match primitive_restart_enable {
                StateMode::Fixed(primitive_restart_enable) => {
                    dynamic_state.insert(DynamicState::PrimitiveRestartEnable, false);
                    primitive_restart_enable as ash::vk::Bool32
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::PrimitiveRestartEnable, true);
                    Default::default()
                }
            };

            let _ = input_assembly_state_vk.insert(ash::vk::PipelineInputAssemblyStateCreateInfo {
                flags: ash::vk::PipelineInputAssemblyStateCreateFlags::empty(),
                topology,
                primitive_restart_enable,
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

            let patch_control_points = match patch_control_points {
                StateMode::Fixed(patch_control_points) => {
                    dynamic_state.insert(DynamicState::PatchControlPoints, false);
                    patch_control_points
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::PatchControlPoints, true);
                    Default::default()
                }
            };

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

            let viewport_count = match viewports {
                PartialStateMode::Fixed(viewports) => {
                    dynamic_state.insert(DynamicState::Viewport, false);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);
                    viewports_vk.extend(viewports.iter().map(Into::into));
                    viewports.len() as u32
                }
                PartialStateMode::Dynamic(StateMode::Fixed(count)) => {
                    dynamic_state.insert(DynamicState::Viewport, true);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);
                    *count
                }
                PartialStateMode::Dynamic(StateMode::Dynamic) => {
                    dynamic_state.insert(DynamicState::Viewport, true);
                    dynamic_state.insert(DynamicState::ViewportWithCount, true);
                    0
                }
            };

            let scissor_count = match scissors {
                PartialStateMode::Fixed(scissors) => {
                    dynamic_state.insert(DynamicState::Scissor, false);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);
                    scissors_vk.extend(scissors.iter().map(Into::into));
                    scissors.len() as u32
                }
                PartialStateMode::Dynamic(StateMode::Fixed(count)) => {
                    dynamic_state.insert(DynamicState::Scissor, true);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);
                    *count
                }
                PartialStateMode::Dynamic(StateMode::Dynamic) => {
                    dynamic_state.insert(DynamicState::Scissor, true);
                    dynamic_state.insert(DynamicState::ScissorWithCount, true);
                    0
                }
            };

            let _ = viewport_state_vk.insert(ash::vk::PipelineViewportStateCreateInfo {
                flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
                viewport_count,
                p_viewports: if viewports_vk.is_empty() {
                    ptr::null()
                } else {
                    viewports_vk.as_ptr()
                },
                scissor_count,
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
                depth_bias,
                line_width,
                line_rasterization_mode,
                line_stipple,
                _ne: _,
            } = rasterization_state;

            let rasterizer_discard_enable = match rasterizer_discard_enable {
                StateMode::Fixed(rasterizer_discard_enable) => {
                    dynamic_state.insert(DynamicState::RasterizerDiscardEnable, false);
                    rasterizer_discard_enable as ash::vk::Bool32
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::RasterizerDiscardEnable, true);
                    ash::vk::FALSE
                }
            };

            let cull_mode = match cull_mode {
                StateMode::Fixed(cull_mode) => {
                    dynamic_state.insert(DynamicState::CullMode, false);
                    cull_mode.into()
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::CullMode, true);
                    CullMode::default().into()
                }
            };

            let front_face = match front_face {
                StateMode::Fixed(front_face) => {
                    dynamic_state.insert(DynamicState::FrontFace, false);
                    front_face.into()
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::FrontFace, true);
                    FrontFace::default().into()
                }
            };

            let (
                depth_bias_enable,
                depth_bias_constant_factor,
                depth_bias_clamp,
                depth_bias_slope_factor,
            ) = if let Some(depth_bias_state) = depth_bias {
                if depth_bias_state.enable_dynamic {
                    dynamic_state.insert(DynamicState::DepthBiasEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::DepthBiasEnable, false);
                }

                let (constant_factor, clamp, slope_factor) = match depth_bias_state.bias {
                    StateMode::Fixed(bias) => {
                        dynamic_state.insert(DynamicState::DepthBias, false);
                        (bias.constant_factor, bias.clamp, bias.slope_factor)
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::DepthBias, true);
                        (0.0, 0.0, 0.0)
                    }
                };

                (ash::vk::TRUE, constant_factor, clamp, slope_factor)
            } else {
                (ash::vk::FALSE, 0.0, 0.0, 0.0)
            };

            let line_width = match line_width {
                StateMode::Fixed(line_width) => {
                    dynamic_state.insert(DynamicState::LineWidth, false);
                    line_width
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::LineWidth, true);
                    1.0
                }
            };

            let rasterization_state =
                rasterization_state_vk.insert(ash::vk::PipelineRasterizationStateCreateInfo {
                    flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
                    depth_clamp_enable: depth_clamp_enable as ash::vk::Bool32,
                    rasterizer_discard_enable,
                    polygon_mode: polygon_mode.into(),
                    cull_mode,
                    front_face,
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
                        let (factor, pattern) = match line_stipple {
                            StateMode::Fixed(line_stipple) => {
                                dynamic_state.insert(DynamicState::LineStipple, false);
                                (line_stipple.factor, line_stipple.pattern)
                            }
                            StateMode::Dynamic => {
                                dynamic_state.insert(DynamicState::LineStipple, true);
                                (1, 0)
                            }
                        };

                        (ash::vk::TRUE, factor, pattern)
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
                        enable_dynamic,
                        write_enable,
                        compare_op,
                    } = depth_state;

                    if enable_dynamic {
                        dynamic_state.insert(DynamicState::DepthTestEnable, true);
                    } else {
                        dynamic_state.insert(DynamicState::DepthTestEnable, false);
                    }

                    let write_enable = match write_enable {
                        StateMode::Fixed(write_enable) => {
                            dynamic_state.insert(DynamicState::DepthWriteEnable, false);
                            write_enable as ash::vk::Bool32
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthWriteEnable, true);
                            ash::vk::TRUE
                        }
                    };

                    let compare_op = match compare_op {
                        StateMode::Fixed(compare_op) => {
                            dynamic_state.insert(DynamicState::DepthCompareOp, false);
                            compare_op.into()
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthCompareOp, true);
                            ash::vk::CompareOp::ALWAYS
                        }
                    };

                    (ash::vk::TRUE, write_enable, compare_op)
                } else {
                    (ash::vk::FALSE, ash::vk::FALSE, ash::vk::CompareOp::ALWAYS)
                };

            let (depth_bounds_test_enable, min_depth_bounds, max_depth_bounds) =
                if let Some(depth_bounds_state) = depth_bounds {
                    let depth_stencil::DepthBoundsState {
                        enable_dynamic,
                        bounds,
                    } = depth_bounds_state;

                    if *enable_dynamic {
                        dynamic_state.insert(DynamicState::DepthBoundsTestEnable, true);
                    } else {
                        dynamic_state.insert(DynamicState::DepthBoundsTestEnable, false);
                    }

                    let (min_bounds, max_bounds) = match bounds.clone() {
                        StateMode::Fixed(bounds) => {
                            dynamic_state.insert(DynamicState::DepthBounds, false);
                            bounds.into_inner()
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthBounds, true);
                            (0.0, 1.0)
                        }
                    };

                    (ash::vk::TRUE, min_bounds, max_bounds)
                } else {
                    (ash::vk::FALSE, 0.0, 1.0)
                };

            let (stencil_test_enable, front, back) = if let Some(stencil_state) = stencil {
                let StencilState {
                    enable_dynamic,
                    front,
                    back,
                } = stencil_state;

                if *enable_dynamic {
                    dynamic_state.insert(DynamicState::StencilTestEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::StencilTestEnable, false);
                }

                match (front.ops, back.ops) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilOp, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilOp, true);
                    }
                    _ => unreachable!(),
                };

                match (front.compare_mask, back.compare_mask) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilCompareMask, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilCompareMask, true);
                    }
                    _ => unreachable!(),
                };

                match (front.write_mask, back.write_mask) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilWriteMask, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilWriteMask, true);
                    }
                    _ => unreachable!(),
                };

                match (front.reference, back.reference) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilReference, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilReference, true);
                    }
                    _ => unreachable!(),
                };

                let [front, back] = [front, back].map(|stencil_op_state| {
                    let &StencilOpState {
                        ops,
                        compare_mask,
                        write_mask,
                        reference,
                    } = stencil_op_state;

                    let ops = match ops {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let compare_mask = match compare_mask {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let write_mask = match write_mask {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let reference = match reference {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };

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
                let logic_op = match logic_op {
                    StateMode::Fixed(logic_op) => {
                        dynamic_state.insert(DynamicState::LogicOp, false);
                        logic_op.into()
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::LogicOp, true);
                        Default::default()
                    }
                };

                (ash::vk::TRUE, logic_op)
            } else {
                (ash::vk::FALSE, Default::default())
            };

            let blend_constants = match blend_constants {
                StateMode::Fixed(blend_constants) => {
                    dynamic_state.insert(DynamicState::BlendConstants, false);
                    blend_constants
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::BlendConstants, true);
                    Default::default()
                }
            };

            let mut color_blend_state_vk =
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

                        match color_write_enable {
                            StateMode::Fixed(enable) => {
                                dynamic_state.insert(DynamicState::ColorWriteEnable, false);
                                enable as ash::vk::Bool32
                            }
                            StateMode::Dynamic => {
                                dynamic_state.insert(DynamicState::ColorWriteEnable, true);
                                ash::vk::TRUE
                            }
                        }
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

        let mut dynamic_state_list: SmallVec<[_; 4]> = SmallVec::new();
        let mut dynamic_state_vk = None;

        {
            dynamic_state_list.extend(
                dynamic_state
                    .iter()
                    .filter(|(_, d)| **d)
                    .map(|(&state, _)| state.into()),
            );

            if !dynamic_state_list.is_empty() {
                let _ = dynamic_state_vk.insert(ash::vk::PipelineDynamicStateCreateInfo {
                    flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                    dynamic_state_count: dynamic_state_list.len() as u32,
                    p_dynamic_states: dynamic_state_list.as_ptr(),
                    ..Default::default()
                });
            }
        }

        let render_pass = render_pass.as_ref().unwrap();
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
        let mut discard_rectangles: SmallVec<[_; 2]> = SmallVec::new();

        if let Some(discard_rectangle_state) = discard_rectangle_state {
            let DiscardRectangleState {
                mode,
                rectangles,
                _ne: _,
            } = discard_rectangle_state;

            let discard_rectangle_count = match rectangles {
                PartialStateMode::Fixed(rectangles) => {
                    dynamic_state.insert(DynamicState::DiscardRectangle, false);
                    discard_rectangles.extend(rectangles.iter().map(|rect| rect.into()));

                    discard_rectangles.len() as u32
                }
                PartialStateMode::Dynamic(count) => {
                    dynamic_state.insert(DynamicState::DiscardRectangle, true);

                    *count
                }
            };

            let _ = discard_rectangle_state_vk.insert(
                ash::vk::PipelineDiscardRectangleStateCreateInfoEXT {
                    flags: ash::vk::PipelineDiscardRectangleStateCreateFlagsEXT::empty(),
                    discard_rectangle_mode: (*mode).into(),
                    discard_rectangle_count,
                    p_discard_rectangles: discard_rectangles.as_ptr(),
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
            base_pipeline_handle: ash::vk::Pipeline::null(), // TODO:
            base_pipeline_index: -1,                         // TODO:
            ..Default::default()
        };

        if let Some(info) = discard_rectangle_state_vk.as_mut() {
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
            flags: _,
            stages,

            vertex_input_state,
            input_assembly_state,
            tessellation_state,
            viewport_state,
            rasterization_state,
            multisample_state,
            depth_stencil_state,
            color_blend_state,

            layout,
            subpass: render_pass,

            discard_rectangle_state,

            _ne: _,
        } = create_info;

        let mut shaders = HashMap::default();
        let mut descriptor_binding_requirements: HashMap<
            (u32, u32),
            DescriptorBindingRequirements,
        > = HashMap::default();
        let mut fragment_tests_stages = None;

        for stage in &stages {
            let &PipelineShaderStageCreateInfo {
                ref entry_point, ..
            } = stage;

            let entry_point_info = entry_point.info();
            let stage = ShaderStage::from(&entry_point_info.execution);
            shaders.insert(stage, ());

            if let ShaderExecution::Fragment(FragmentShaderExecution {
                fragment_tests_stages: s,
                ..
            }) = entry_point_info.execution
            {
                fragment_tests_stages = Some(s)
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

        let mut dynamic_state: HashMap<DynamicState, bool> = HashMap::default();

        if vertex_input_state.is_some() {
            dynamic_state.insert(DynamicState::VertexInput, false);
        }

        if let Some(input_assembly_state) = &input_assembly_state {
            let &InputAssemblyState {
                topology,
                primitive_restart_enable,
                _ne: _,
            } = input_assembly_state;

            match topology {
                PartialStateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::PrimitiveTopology, false);
                }
                PartialStateMode::Dynamic(_) => {
                    dynamic_state.insert(DynamicState::PrimitiveTopology, true);
                }
            }

            match primitive_restart_enable {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::PrimitiveRestartEnable, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::PrimitiveRestartEnable, true);
                }
            }
        }

        if let Some(tessellation_state) = &tessellation_state {
            let &TessellationState {
                patch_control_points,
                domain_origin: _,
                _ne: _,
            } = tessellation_state;

            match patch_control_points {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::PatchControlPoints, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::PatchControlPoints, true);
                }
            }
        }

        if let Some(viewport_state) = &viewport_state {
            let ViewportState {
                viewports,
                scissors,
                _ne: _,
            } = viewport_state;

            match viewports {
                PartialStateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::Viewport, false);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);
                }
                PartialStateMode::Dynamic(StateMode::Fixed(_)) => {
                    dynamic_state.insert(DynamicState::Viewport, true);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);
                }
                PartialStateMode::Dynamic(StateMode::Dynamic) => {
                    dynamic_state.insert(DynamicState::Viewport, true);
                    dynamic_state.insert(DynamicState::ViewportWithCount, true);
                }
            }

            match scissors {
                PartialStateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::Scissor, false);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);
                }
                PartialStateMode::Dynamic(StateMode::Fixed(_)) => {
                    dynamic_state.insert(DynamicState::Scissor, true);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);
                }
                PartialStateMode::Dynamic(StateMode::Dynamic) => {
                    dynamic_state.insert(DynamicState::Scissor, true);
                    dynamic_state.insert(DynamicState::ScissorWithCount, true);
                }
            }
        }

        if let Some(rasterization_state) = &rasterization_state {
            let &RasterizationState {
                rasterizer_discard_enable,
                cull_mode,
                front_face,
                depth_bias,
                line_width,
                line_stipple,
                ..
            } = rasterization_state;

            match rasterizer_discard_enable {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::RasterizerDiscardEnable, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::RasterizerDiscardEnable, true);
                }
            }

            match cull_mode {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::CullMode, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::CullMode, true);
                }
            }

            match front_face {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::FrontFace, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::FrontFace, true);
                }
            }

            if let Some(depth_bias_state) = depth_bias {
                if depth_bias_state.enable_dynamic {
                    dynamic_state.insert(DynamicState::DepthBiasEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::DepthBiasEnable, false);
                }

                match depth_bias_state.bias {
                    StateMode::Fixed(_) => {
                        dynamic_state.insert(DynamicState::DepthBias, false);
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::DepthBias, true);
                    }
                }
            }

            match line_width {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::LineWidth, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::LineWidth, true);
                }
            }

            if device.enabled_extensions().ext_line_rasterization {
                if let Some(line_stipple) = line_stipple {
                    match line_stipple {
                        StateMode::Fixed(_) => {
                            dynamic_state.insert(DynamicState::LineStipple, false);
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::LineStipple, true);
                        }
                    }
                }
            }
        }

        if let Some(depth_stencil_state) = &depth_stencil_state {
            let DepthStencilState {
                flags: _,
                depth,
                depth_bounds,
                stencil,
                _ne: _,
            } = depth_stencil_state;

            if let Some(depth_state) = depth {
                let &DepthState {
                    enable_dynamic,
                    write_enable,
                    compare_op,
                } = depth_state;

                if enable_dynamic {
                    dynamic_state.insert(DynamicState::DepthTestEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::DepthTestEnable, false);
                }

                match write_enable {
                    StateMode::Fixed(_) => {
                        dynamic_state.insert(DynamicState::DepthWriteEnable, false);
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::DepthWriteEnable, true);
                    }
                }

                match compare_op {
                    StateMode::Fixed(_) => {
                        dynamic_state.insert(DynamicState::DepthCompareOp, false);
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::DepthCompareOp, true);
                    }
                }
            }

            if let Some(depth_bounds_state) = depth_bounds {
                let DepthBoundsState {
                    enable_dynamic,
                    bounds,
                } = depth_bounds_state;

                if *enable_dynamic {
                    dynamic_state.insert(DynamicState::DepthBoundsTestEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::DepthBoundsTestEnable, false);
                }

                match bounds.clone() {
                    StateMode::Fixed(_) => {
                        dynamic_state.insert(DynamicState::DepthBounds, false);
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::DepthBounds, true);
                    }
                }
            }

            if let Some(stencil_state) = stencil {
                let StencilState {
                    enable_dynamic,
                    front,
                    back,
                } = stencil_state;

                if *enable_dynamic {
                    dynamic_state.insert(DynamicState::StencilTestEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::StencilTestEnable, false);
                }

                match (front.ops, back.ops) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilOp, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilOp, true);
                    }
                    _ => unreachable!(),
                }

                match (front.compare_mask, back.compare_mask) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilCompareMask, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilCompareMask, true);
                    }
                    _ => unreachable!(),
                }

                match (front.write_mask, back.write_mask) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilWriteMask, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilWriteMask, true);
                    }
                    _ => unreachable!(),
                }

                match (front.reference, back.reference) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilReference, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilReference, true);
                    }
                    _ => unreachable!(),
                }
            }
        }

        if let Some(color_blend_state) = &color_blend_state {
            let &ColorBlendState {
                flags: _,
                logic_op,
                ref attachments,
                blend_constants,
                _ne: _,
            } = color_blend_state;

            if let Some(logic_op) = logic_op {
                match logic_op {
                    StateMode::Fixed(_) => {
                        dynamic_state.insert(DynamicState::LogicOp, false);
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::LogicOp, true);
                    }
                }
            }

            match blend_constants {
                StateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::BlendConstants, false);
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::BlendConstants, true);
                }
            }

            if device.enabled_extensions().ext_color_write_enable {
                for color_blend_attachment_state in attachments {
                    let &ColorBlendAttachmentState {
                        blend: _,
                        color_write_mask: _,
                        color_write_enable,
                    } = color_blend_attachment_state;

                    match color_write_enable {
                        StateMode::Fixed(_) => {
                            dynamic_state.insert(DynamicState::ColorWriteEnable, false);
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::ColorWriteEnable, true);
                        }
                    }
                }
            }
        }

        if let Some(discard_rectangle_state) = &discard_rectangle_state {
            let DiscardRectangleState { rectangles, .. } = discard_rectangle_state;

            match rectangles {
                PartialStateMode::Fixed(_) => {
                    dynamic_state.insert(DynamicState::DiscardRectangle, false);
                }
                PartialStateMode::Dynamic(_) => {
                    dynamic_state.insert(DynamicState::DiscardRectangle, true);
                }
            }
        }

        Arc::new(Self {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            shaders,
            descriptor_binding_requirements,
            num_used_descriptor_sets,
            fragment_tests_stages,

            vertex_input_state: vertex_input_state.unwrap(), // Can be None if there's a mesh shader, but we don't support that yet
            input_assembly_state: input_assembly_state.unwrap(), // Can be None if there's a mesh shader, but we don't support that yet
            tessellation_state,
            viewport_state,
            rasterization_state: rasterization_state.unwrap(), // Can be None for pipeline libraries, but we don't support that yet
            multisample_state,
            depth_stencil_state,
            color_blend_state,
            layout: DeviceOwnedDebugWrapper(layout),
            subpass: render_pass.unwrap(),
            dynamic_state,

            discard_rectangle_state,
        })
    }

    /// Returns the device used to create this pipeline.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns information about a particular shader.
    ///
    /// `None` is returned if the pipeline does not contain this shader.
    ///
    /// Compatibility note: `()` is temporary, it will be replaced with something else in the
    /// future.
    // TODO: ^ implement and make this public
    #[inline]
    pub(crate) fn shader(&self, stage: ShaderStage) -> Option<()> {
        self.shaders.get(&stage).copied()
    }

    /// Returns the vertex input state used to create this pipeline.
    #[inline]
    pub fn vertex_input_state(&self) -> &VertexInputState {
        &self.vertex_input_state
    }

    /// Returns the input assembly state used to create this pipeline.
    #[inline]
    pub fn input_assembly_state(&self) -> &InputAssemblyState {
        &self.input_assembly_state
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

    /// Returns whether a particular state is must be dynamically set.
    ///
    /// `None` is returned if the pipeline does not contain this state. Previously set dynamic
    /// state is not disturbed when binding it.
    #[inline]
    pub fn dynamic_state(&self, state: DynamicState) -> Option<bool> {
        self.dynamic_state.get(&state).copied()
    }

    /// Returns all potentially dynamic states in the pipeline, and whether they are dynamic or not.
    #[inline]
    pub fn dynamic_states(&self) -> impl ExactSizeIterator<Item = (DynamicState, bool)> + '_ {
        self.dynamic_state.iter().map(|(k, v)| (*k, *v))
    }

    /// Returns the discard rectangle state used to create this pipeline.
    #[inline]
    pub fn discard_rectangle_state(&self) -> Option<&DiscardRectangleState> {
        self.discard_rectangle_state.as_ref()
    }

    /// If the pipeline has a fragment shader, returns the fragment tests stages used.
    #[inline]
    pub fn fragment_tests_stages(&self) -> Option<FragmentTestsStages> {
        self.fragment_tests_stages
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
    /// A vertex shader must always be included. Other stages are optional.
    ///
    /// The default value is empty.
    pub stages: SmallVec<[PipelineShaderStageCreateInfo; 5]>,

    /// The vertex input state.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    pub vertex_input_state: Option<VertexInputState>,

    /// The input assembly state.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    pub input_assembly_state: Option<InputAssemblyState>,

    /// The tessellation state.
    ///
    /// This state is used if `stages` contains tessellation shaders.
    ///
    /// The default value is `None`.
    pub tessellation_state: Option<TessellationState>,

    /// The viewport state.
    ///
    /// This state is used if [rasterizer discarding] is not enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub viewport_state: Option<ViewportState>,

    /// The rasterization state.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    pub rasterization_state: Option<RasterizationState>,

    /// The multisample state.
    ///
    /// This state is used if [rasterizer discarding] is not enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub multisample_state: Option<MultisampleState>,

    /// The depth/stencil state.
    ///
    /// This state is used if `render_pass` has depth/stencil attachments, or if
    /// [rasterizer discarding] is enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub depth_stencil_state: Option<DepthStencilState>,

    /// The color blend state.
    ///
    /// This state is used if `render_pass` has color attachments, and [rasterizer discarding] is
    /// not enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    pub color_blend_state: Option<ColorBlendState>,

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

    /// The discard rectangle state.
    ///
    /// This state is always used if it is provided.
    ///
    /// The default value is `None`.
    pub discard_rectangle_state: Option<DiscardRectangleState>,

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
            layout,
            subpass: None,
            discard_rectangle_state: None,
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

            ref layout,
            ref subpass,

            ref discard_rectangle_state,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkGraphicsPipelineCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        /*
            Gather shader stages
        */

        let mut stages_present = ShaderStages::empty();
        let mut vertex_stage = None;
        let mut tessellation_control_stage = None;
        let mut tessellation_evaluation_stage = None;
        let mut geometry_stage = None;
        let mut fragment_stage = None;

        for (stage_index, stage) in stages.iter().enumerate() {
            let entry_point_info = stage.entry_point.info();
            let stage_enum = ShaderStage::from(&entry_point_info.execution);
            let stage_flag = ShaderStages::from(stage_enum);

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

            const PRIMITIVE_SHADING_STAGES: ShaderStages = ShaderStages::VERTEX
                .union(ShaderStages::TESSELLATION_CONTROL)
                .union(ShaderStages::TESSELLATION_CONTROL)
                .union(ShaderStages::GEOMETRY);
            const MESH_SHADING_STAGES: ShaderStages = ShaderStages::MESH.union(ShaderStages::TASK);

            if stage_flag.intersects(PRIMITIVE_SHADING_STAGES)
                && stages_present.intersects(MESH_SHADING_STAGES)
                || stage_flag.intersects(MESH_SHADING_STAGES)
                    && stages_present.intersects(PRIMITIVE_SHADING_STAGES)
            {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains both primitive shading stages and mesh shading stages"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-02095"],
                    ..Default::default()
                }));
            }

            let stage_slot = match stage_enum {
                ShaderStage::Vertex => &mut vertex_stage,
                ShaderStage::TessellationControl => &mut tessellation_control_stage,
                ShaderStage::TessellationEvaluation => &mut tessellation_evaluation_stage,
                ShaderStage::Geometry => &mut geometry_stage,
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

        let need_vertex_input_state = need_pre_rasterization_shader_state
            && stages
                .iter()
                .any(|stage| matches!(stage.entry_point.info().execution, ShaderExecution::Vertex));
        let need_fragment_shader_state = need_pre_rasterization_shader_state
            && rasterization_state
                .as_ref()
                .unwrap()
                .rasterizer_discard_enable
                != StateMode::Fixed(true);
        let need_fragment_output_state = need_pre_rasterization_shader_state
            && rasterization_state
                .as_ref()
                .unwrap()
                .rasterizer_discard_enable
                != StateMode::Fixed(true);

        match (vertex_stage.is_some(), need_pre_rasterization_shader_state) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization shader state, but `stages` contains a \
                        `ShaderStage::Vertex` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06895"],
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        pre-rasterization shader state, but `stages` does not contain a \
                        `ShaderStage::Vertex` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-stage-02096"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            tessellation_control_stage.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization shader state, but `stages` contains a \
                        `ShaderStage::TessellationControl` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06895"],
                    ..Default::default()
                }));
            }
            (false, true) => (),
            _ => (),
        }

        match (
            tessellation_evaluation_stage.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization shader state, but `stages` contains a \
                        `ShaderStage::TessellationEvaluation` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06895"],
                    ..Default::default()
                }));
            }
            (false, true) => (),
            _ => (),
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

        match (
            geometry_stage.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        pre-rasterization shader state, but `stages` contains a \
                        `ShaderStage::Geometry` stage"
                        .into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-06895"],
                    ..Default::default()
                }));
            }
            (false, true) => (),
            _ => (),
        }

        match (fragment_stage.is_some(), need_fragment_shader_state) {
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

        match (vertex_input_state.is_some(), need_vertex_input_state) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is not being created with \
                        vertex input state, but \
                        `vertex_input_state` is `Some`"
                        .into(),
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "the pipeline is being created with \
                        vertex input state, but \
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
                && rasterization_state
                    .as_ref()
                    .unwrap()
                    .rasterizer_discard_enable
                    != StateMode::Fixed(true),
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

        for (stage_index, stage) in stages.iter().enumerate() {
            stage
                .validate(device)
                .map_err(|err| err.add_context(format!("stages[{}]", stage_index)))?;

            let &PipelineShaderStageCreateInfo {
                flags: _,
                ref entry_point,
                specialization_info: _,
                _ne: _,
            } = stage;

            let entry_point_info = entry_point.info();

            layout
                .ensure_compatible_with_shader(
                    entry_point_info
                        .descriptor_binding_requirements
                        .iter()
                        .map(|(k, v)| (*k, v)),
                    entry_point_info.push_constant_requirements.as_ref(),
                )
                .map_err(|err| ValidationError {
                    context: format!("stages[{}].entry_point", stage_index).into(),
                    vuids: &["VUID-VkGraphicsPipelineCreateInfo-layout-00756"],
                    ..ValidationError::from_error(err)
                })?;
        }

        let ordered_stages: SmallVec<[_; 5]> = [
            vertex_stage,
            tessellation_control_stage,
            tessellation_evaluation_stage,
            geometry_stage,
            fragment_stage,
        ]
        .into_iter()
        .flatten()
        .collect();

        // TODO: this check is too strict; the output only has to be a superset, any variables
        // not used in the input of the next shader are just ignored.
        for (output, input) in ordered_stages.iter().zip(ordered_stages.iter().skip(1)) {
            if let Err(err) = (input.entry_point.info().input_interface)
                .matches(&output.entry_point.info().output_interface)
            {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: format!(
                        "the output interface of the `ShaderStage::{:?}` stage does not \
                        match the input interface of the `ShaderStage::{:?}` stage: {}",
                        ShaderStage::from(&output.entry_point.info().execution),
                        ShaderStage::from(&input.entry_point.info().execution),
                        err
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkGraphicsPipelineCreateInfo-pStages-00742",
                        "VUID-VkGraphicsPipelineCreateInfo-None-04889",
                    ],
                    ..Default::default()
                }));
            }
        }

        // VUID-VkGraphicsPipelineCreateInfo-layout-01688
        // Checked at pipeline layout creation time.

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

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-topology-00737
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

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-primitiveFragmentShadingRateWithMultipleViewports-04503
            // VUID-VkGraphicsPipelineCreateInfo-primitiveFragmentShadingRateWithMultipleViewports-04504
        }

        if let Some(rasterization_state) = rasterization_state {
            rasterization_state
                .validate(device)
                .map_err(|err| err.add_context("rasterization_state"))?;

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00740
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06049
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06050
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06059
        }

        if let Some(multisample_state) = multisample_state {
            multisample_state
                .validate(device)
                .map_err(|err| err.add_context("multisample_state"))?;

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766
        }

        if let Some(depth_stencil_state) = depth_stencil_state {
            depth_stencil_state
                .validate(device)
                .map_err(|err| err.add_context("depth_stencil_state"))?;

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06040
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

                    if subpass.subpass_desc().view_mask != 0 {
                        if stages_present.intersects(
                            ShaderStages::TESSELLATION_CONTROL
                                | ShaderStages::TESSELLATION_EVALUATION,
                        ) && !device.enabled_features().multiview_tessellation_shader
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`stages` contains tessellation shaders, and \
                                    `subpass` has a non-zero `view_mask`"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::Feature("multiview_tessellation_shader"),
                                ])]),
                                vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06047"],
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
                                    Requires::Feature("multiview_geometry_shader"),
                                ])]),
                                vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06048"],
                                ..Default::default()
                            }));
                        }
                    }
                }
                PipelineSubpassType::BeginRendering(rendering_info) => {
                    if !device.enabled_features().dynamic_rendering {
                        return Err(Box::new(ValidationError {
                            context: "subpass".into(),
                            problem: "is `PipelineRenderPassType::BeginRendering`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "dynamic_rendering",
                            )])]),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576"],
                        }));
                    }

                    rendering_info
                        .validate(device)
                        .map_err(|err| err.add_context("subpass"))?;

                    let &PipelineRenderingCreateInfo {
                        view_mask,
                        color_attachment_formats: _,
                        depth_attachment_format: _,
                        stencil_attachment_format: _,
                        _ne: _,
                    } = rendering_info;

                    if view_mask != 0 {
                        if stages_present.intersects(
                            ShaderStages::TESSELLATION_CONTROL
                                | ShaderStages::TESSELLATION_EVALUATION,
                        ) && !device.enabled_features().multiview_tessellation_shader
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`stages` contains tessellation shaders, and \
                                    `subpass.view_mask` is not 0"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::Feature("multiview_tessellation_shader"),
                                ])]),
                                vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06057"],
                                ..Default::default()
                            }));
                        }

                        if stages_present.intersects(ShaderStages::GEOMETRY)
                            && !device.enabled_features().multiview_geometry_shader
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`stages` contains a geometry shader, and \
                                    `subpass.view_mask` is not 0"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::Feature("multiview_geometry_shader"),
                                ])]),
                                vuids: &["VUID-VkGraphicsPipelineCreateInfo-renderPass-06058"],
                                ..Default::default()
                            }));
                        }
                    }
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

        /*
            Checks that rely on multiple pieces of state
        */

        if let (Some(vertex_stage), Some(vertex_input_state)) = (vertex_stage, vertex_input_state) {
            for element in vertex_stage.entry_point.info().input_interface.elements() {
                assert!(!element.ty.is_64bit); // TODO: implement
                let location_range =
                    element.location..element.location + element.ty.num_locations();

                for location in location_range {
                    let attribute_desc = match vertex_input_state.attributes.get(&location) {
                        Some(attribute_desc) => attribute_desc,
                        None => {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the vertex shader has an input variable with location {0}, but \
                                    `vertex_input_state.attributes` does not contain {0}",
                                    location,
                                )
                                .into(),
                                vuids: &["VUID-VkGraphicsPipelineCreateInfo-Input-07905"],
                                ..Default::default()
                            }));
                        }
                    };

                    // TODO: Check component assignments too. Multiple variables can occupy the
                    // same location but in different components.

                    let shader_type = element.ty.base_type;
                    let attribute_type = attribute_desc.format.type_color().unwrap();

                    // VUID?
                    if !matches!(
                        (shader_type, attribute_type),
                        (
                            ShaderScalarType::Float,
                            NumericType::SFLOAT
                                | NumericType::UFLOAT
                                | NumericType::SNORM
                                | NumericType::UNORM
                                | NumericType::SSCALED
                                | NumericType::USCALED
                                | NumericType::SRGB,
                        ) | (ShaderScalarType::Sint, NumericType::SINT)
                            | (ShaderScalarType::Uint, NumericType::UINT)
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`vertex_input_state.attributes[{}].format` has a different \
                                scalar type than the vertex shader input variable with \
                                location {0}",
                                location,
                            )
                            .into(),
                            vuids: &["VUID-VkGraphicsPipelineCreateInfo-Input-07905"],
                            ..Default::default()
                        }));
                    }
                }
            }
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
            let entry_point_info = geometry_stage.entry_point.info();
            let input = match entry_point_info.execution {
                ShaderExecution::Geometry(execution) => execution.input,
                _ => unreachable!(),
            };

            if let PartialStateMode::Fixed(topology) = input_assembly_state.topology {
                if !input.is_compatible_with(topology) {
                    return Err(Box::new(ValidationError {
                        problem: "`input_assembly_state.topology` is not compatible with the \
                            input topology of the geometry shader"
                            .into(),
                        vuids: &["VUID-VkGraphicsPipelineCreateInfo-pStages-00738"],
                        ..Default::default()
                    }));
                }
            }
        }

        if let (Some(fragment_stage), Some(subpass)) = (fragment_stage, subpass) {
            let entry_point_info = fragment_stage.entry_point.info();

            // Check that the subpass can accept the output of the fragment shader.
            match subpass {
                PipelineSubpassType::BeginRenderPass(subpass) => {
                    if !subpass.is_compatible_with(&entry_point_info.output_interface) {
                        return Err(Box::new(ValidationError {
                            problem: "`subpass` is not compatible with the \
                                output interface of the fragment shader"
                                .into(),
                            ..Default::default()
                        }));
                    }
                }
                PipelineSubpassType::BeginRendering(_) => {
                    // TODO:
                }
            }

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-01565
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06038
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06056
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06061
        }

        if let (Some(input_assembly_state), Some(_)) = (input_assembly_state, tessellation_state) {
            if !matches!(
                input_assembly_state.topology,
                PartialStateMode::Dynamic(PrimitiveTopologyClass::Patch)
                    | PartialStateMode::Fixed(PrimitiveTopology::PatchList)
            ) {
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
                if let (StateMode::Fixed(front_reference), StateMode::Fixed(back_reference)) =
                    (stencil_state.front.reference, stencil_state.back.reference)
                {
                    if device.enabled_extensions().khr_portability_subset
                        && !device.enabled_features().separate_stencil_mask_ref
                        && matches!(
                            rasterization_state.cull_mode,
                            StateMode::Fixed(CullMode::None)
                        )
                        && front_reference != back_reference
                    {
                        return Err(Box::new(ValidationError {
                            problem: "this device is a portability subset device, \
                                `rasterization_state.cull_mode` is `CullMode::None`, and \
                                `depth_stencil_state.stencil.front.reference` does not equal \
                                `depth_stencil_state.stencil.back.reference`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "separate_stencil_mask_ref",
                            )])]),
                            vuids: &["VUID-VkPipelineDepthStencilStateCreateInfo-separateStencilMaskRef-04453"],
                            ..Default::default()
                        }));
                    }
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
                                .unwrap()
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

                if let StateMode::Fixed(true) = depth_state.write_enable {
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
                                        .unwrap()
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
                                .unwrap()
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
                                .and_then(|color_attachment| {
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
