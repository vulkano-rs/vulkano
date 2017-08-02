// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// TODO: graphics pipeline params are deprecated, but are still the primary implementation in order
// to avoid duplicating code, so we hide the warnings for now
#![allow(deprecated)]

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::u32;

use Error;
use OomError;
use SafeDeref;
use VulkanObject;
use buffer::BufferAccess;
use check_errors;
use descriptor::PipelineLayoutAbstract;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use descriptor::pipeline_layout::PipelineLayoutDescUnion;
use descriptor::pipeline_layout::PipelineLayoutNotSupersetError;
use descriptor::pipeline_layout::PipelineLayoutSuperset;
use descriptor::pipeline_layout::PipelineLayoutSys;
use device::Device;
use device::DeviceOwned;
use format::ClearValue;
use framebuffer::LayoutAttachmentDescription;
use framebuffer::LayoutPassDependencyDescription;
use framebuffer::LayoutPassDescription;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassDescClearValues;
use framebuffer::RenderPassSubpassInterface;
use framebuffer::RenderPassSys;
use framebuffer::Subpass;
use vk;

use pipeline::blend::AttachmentsBlend;
use pipeline::blend::Blend;
use pipeline::depth_stencil::Compare;
use pipeline::depth_stencil::DepthBounds;
use pipeline::depth_stencil::DepthStencil;
use pipeline::input_assembly::InputAssembly;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::multisample::Multisample;
use pipeline::raster::DepthBiasControl;
use pipeline::raster::PolygonMode;
use pipeline::raster::Rasterization;
use pipeline::shader::EmptyEntryPointDummy;
use pipeline::shader::GraphicsEntryPointAbstract;
use pipeline::shader::GraphicsShaderType;
use pipeline::shader::ShaderInterfaceDefMatch;
use pipeline::shader::ShaderInterfaceMismatchError;
use pipeline::vertex::IncompatibleVertexDefinitionError;
use pipeline::vertex::SingleBufferDefinition;
use pipeline::vertex::VertexDefinition;
use pipeline::vertex::VertexSource;
use pipeline::viewport::ViewportsState;

pub use self::builder::GraphicsPipelineBuilder;

mod builder;
// FIXME: restore
//mod tests;

/// Description of a `GraphicsPipeline`.
#[deprecated = "Use the GraphicsPipelineBuilder instead"]
pub struct GraphicsPipelineParams<
 Vdef,
 Vs,
 Tcs,
 Tes,
 Gs,
 Fs,
 Rp>
{
    /// Describes the layout of the vertex input.
    ///
    /// For example if you want to pass a vertex buffer and an instance buffer, this parameter
    /// should describe it as well as the offsets and data type of various vertex attributes.
    ///
    /// Must implement the `VertexDefinition` trait.
    pub vertex_input: Vdef,

    /// The entry point of the vertex shader that will be run on the vertex input.
    pub vertex_shader: Vs,

    /// Describes how vertices should be assembled into primitives. Essentially contains the type
    /// of primitives.
    pub input_assembly: InputAssembly,

    /// Parameters of the tessellation stage. `None` if you don't want to use tessellation.
    /// If you use tessellation, you must enable the `tessellation_shader` feature on the device.
    pub tessellation: Option<GraphicsPipelineParamsTess<Tcs, Tes>>,

    /// The entry point of the geometry shader. `None` if you don't want a geometry shader.
    /// If you use a geometry shader, you must enable the `geometry_shader` feature on the device.
    pub geometry_shader: Option<Gs>,

    /// Describes the subsection of the framebuffer attachments where the scene will be drawn.
    /// You can use one or multiple viewports, but using multiple viewports is only relevant with
    /// a geometry shader.
    pub viewport: ViewportsState,

    /// Describes how the implementation determines which pixels are covered by the shape.
    pub raster: Rasterization,

    // TODO: document
    pub multisample: Multisample,

    /// The entry point of the fragment shader that will be run on the pixels.
    pub fragment_shader: Fs,

    /// Describes how the implementation should perform the depth and stencil tests.
    pub depth_stencil: DepthStencil,

    /// Describes how the implementation should merge the color output of the fragment shader with
    /// the existing value in the attachments.
    pub blend: Blend,

    /// Which subpass of which render pass this pipeline will run on. It is an error to run a
    /// graphics pipeline on a different subpass.
    pub render_pass: Subpass<Rp>,
}

/// Additional parameters if you use tessellation.
#[deprecated = "Use the GraphicsPipelineBuilder instead"]
pub struct GraphicsPipelineParamsTess<Tcs, Tes> {
    /// The entry point of the tessellation control shader.
    pub tessellation_control_shader: Tcs,
    /// The entry point of the tessellation evaluation shader.
    pub tessellation_evaluation_shader: Tes,
}

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
pub struct GraphicsPipeline<VertexDefinition, Layout, RenderP> {
    inner: Inner,
    layout: Layout,

    render_pass: RenderP,
    render_pass_subpass: u32,

    vertex_definition: VertexDefinition,

    dynamic_line_width: bool,
    dynamic_viewport: bool,
    dynamic_scissor: bool,
    dynamic_depth_bias: bool,
    dynamic_depth_bounds: bool,
    dynamic_stencil_compare_mask: bool,
    dynamic_stencil_write_mask: bool,
    dynamic_stencil_reference: bool,
    dynamic_blend_constants: bool,

    num_viewports: u32,
}

struct Inner {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl GraphicsPipeline<(), (), ()> {
    /// Starts the building process of a graphics pipeline. Returns a builder object that you can
    /// fill with the various parameters.
    pub fn start<'a>()
        -> GraphicsPipelineBuilder<SingleBufferDefinition<()>,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   EmptyEntryPointDummy,
                                   ()>
    {
        GraphicsPipelineBuilder::new()
    }
}

impl<Vdef, Rp> GraphicsPipeline<Vdef, (), Rp>
    where Rp: RenderPassAbstract
{
    /// Builds a new graphics pipeline object.
    ///
    /// See the documentation of `GraphicsPipelineCreateInfo` for more info about the parameter.
    ///
    /// In order to avoid compiler errors caused by not being able to infer template parameters,
    /// this function assumes that you will only use a vertex shader and a fragment shader. See
    /// the other constructors for other possibilities.
    #[inline]
    #[deprecated = "Use the GraphicsPipelineBuilder instead"]
    pub fn new<Vs, Fs>
              (device: Arc<Device>,
               params: GraphicsPipelineParams<Vdef, Vs, EmptyEntryPointDummy, EmptyEntryPointDummy, EmptyEntryPointDummy, Fs, Rp>)
              -> Result<GraphicsPipeline<Vdef, PipelineLayout<PipelineLayoutDescUnion<Vs::PipelineLayout, Fs::PipelineLayout>>, Rp>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vs::InputDefinition>,
              Vs: GraphicsEntryPointAbstract,
              Fs: GraphicsEntryPointAbstract,
              Vs::PipelineLayout: Clone,
              Fs::PipelineLayout: Clone,
              Fs::InputDefinition: ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Rp: RenderPassSubpassInterface<Fs::OutputDefinition>,
    {
        if let Err(err) = params
            .fragment_shader
            .input()
            .matches(params.vertex_shader.output())
        {
            return Err(GraphicsPipelineCreationError::VertexFragmentStagesMismatch(err));
        }

        let pl = params
            .vertex_shader
            .layout()
            .clone()
            .union(params.fragment_shader.layout().clone())
            .build(device.clone())
            .unwrap(); // TODO: error

        GraphicsPipeline::new_inner::<_,
                                      EmptyEntryPointDummy,
                                      EmptyEntryPointDummy,
                                      EmptyEntryPointDummy,
                                      _>(device, params, pl)
    }

    /// Builds a new graphics pipeline object with a geometry shader.
    ///
    /// See the documentation of `GraphicsPipelineCreateInfo` for more info about the parameter.
    ///
    /// In order to avoid compiler errors caused by not being able to infer template parameters,
    /// this function assumes that you will use a vertex shader, a geometry shader and a fragment
    /// shader. See the other constructors for other possibilities.
    #[inline]
    #[deprecated = "Use the GraphicsPipelineBuilder instead"]
    pub fn with_geometry_shader<Vs, Gs, Fs>
              (device: Arc<Device>,
               params: GraphicsPipelineParams<Vdef, Vs, EmptyEntryPointDummy, EmptyEntryPointDummy, Gs, Fs, Rp>)
              -> Result<GraphicsPipeline<Vdef, PipelineLayout<PipelineLayoutDescUnion<PipelineLayoutDescUnion<Vs::PipelineLayout, Fs::PipelineLayout>, Gs::PipelineLayout>>, Rp>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vs::InputDefinition>,
              Vs: GraphicsEntryPointAbstract,
              Fs: GraphicsEntryPointAbstract,
              Gs: GraphicsEntryPointAbstract,
              Vs::PipelineLayout: Clone,
              Fs::PipelineLayout: Clone,
              Gs::PipelineLayout: Clone,
              Gs::InputDefinition: ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Fs::InputDefinition: ShaderInterfaceDefMatch<Gs::OutputDefinition> + ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Rp: RenderPassSubpassInterface<Fs::OutputDefinition>,
    {
        if let Some(ref geometry_shader) = params.geometry_shader {
            if let Err(err) = geometry_shader
                .input()
                .matches(params.vertex_shader.output())
            {
                return Err(GraphicsPipelineCreationError::VertexGeometryStagesMismatch(err));
            };

            if let Err(err) = params
                .fragment_shader
                .input()
                .matches(geometry_shader.output())
            {
                return Err(GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(err));
            }
        } else {
            if let Err(err) = params
                .fragment_shader
                .input()
                .matches(params.vertex_shader.output())
            {
                return Err(GraphicsPipelineCreationError::VertexFragmentStagesMismatch(err));
            }
        }

        let pl = params.vertex_shader.layout().clone()
                    .union(params.fragment_shader.layout().clone())
                    .union(params.geometry_shader.as_ref().unwrap().layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap(); // TODO: error

        GraphicsPipeline::new_inner(device.clone(), params, pl)
    }

    /// Builds a new graphics pipeline object with tessellation shaders.
    ///
    /// See the documentation of `GraphicsPipelineCreateInfo` for more info about the parameter.
    ///
    /// In order to avoid compiler errors caused by not being able to infer template parameters,
    /// this function assumes that you will use a vertex shader, a tessellation control shader, a
    /// tessellation evaluation shader and a fragment shader. See the other constructors for other
    /// possibilities.
    #[inline]
    #[deprecated = "Use the GraphicsPipelineBuilder instead"]
    pub fn with_tessellation<Vs, Tcs, Tes, Fs>
              (device: Arc<Device>,
               params: GraphicsPipelineParams<Vdef, Vs, Tcs, Tes, EmptyEntryPointDummy, Fs, Rp>)
               -> Result<GraphicsPipeline<Vdef, PipelineLayout<PipelineLayoutDescUnion<PipelineLayoutDescUnion<PipelineLayoutDescUnion<Vs::PipelineLayout, Fs::PipelineLayout>, Tcs::PipelineLayout>, Tes::PipelineLayout>>, Rp>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vs::InputDefinition>,
              Vs: GraphicsEntryPointAbstract,
              Fs: GraphicsEntryPointAbstract,
              Tcs: GraphicsEntryPointAbstract,
              Tes: GraphicsEntryPointAbstract,
              Vs::PipelineLayout: Clone,
              Fs::PipelineLayout: Clone,
              Tcs::PipelineLayout: Clone,
              Tes::PipelineLayout: Clone,
              Tcs::InputDefinition: ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Tes::InputDefinition: ShaderInterfaceDefMatch<Tcs::OutputDefinition>,
              Fs::InputDefinition: ShaderInterfaceDefMatch<Tes::OutputDefinition> + ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Rp: RenderPassAbstract + RenderPassSubpassInterface<Fs::OutputDefinition>,
    {
        if let Some(ref tess) = params.tessellation {
            if let Err(err) = tess.tessellation_control_shader
                .input()
                .matches(params.vertex_shader.output())
            {
                return Err(GraphicsPipelineCreationError::VertexTessControlStagesMismatch(err));
            }
            if let Err(err) = tess.tessellation_evaluation_shader
                .input()
                .matches(tess.tessellation_control_shader.output())
            {
                return Err(GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(err));
            }
            if let Err(err) = params
                .fragment_shader
                .input()
                .matches(tess.tessellation_evaluation_shader.output())
            {
                return Err(GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(err));
            }

        } else {
            if let Err(err) = params
                .fragment_shader
                .input()
                .matches(params.vertex_shader.output())
            {
                return Err(GraphicsPipelineCreationError::VertexFragmentStagesMismatch(err));
            }
        }

        let pl = params.vertex_shader.layout().clone()
                    .union(params.fragment_shader.layout().clone())
                    .union(params.tessellation.as_ref().unwrap().tessellation_control_shader.layout().clone())    // FIXME: unwrap()
                    .union(params.tessellation.as_ref().unwrap().tessellation_evaluation_shader.layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap(); // TODO: error

        GraphicsPipeline::new_inner(device, params, pl)
    }

    /// Builds a new graphics pipeline object with a geometry and tessellation shaders.
    ///
    /// See the documentation of `GraphicsPipelineCreateInfo` for more info about the parameter.
    ///
    /// In order to avoid compiler errors caused by not being able to infer template parameters,
    /// this function assumes that you will use a vertex shader, a tessellation control shader, a
    /// tessellation evaluation shader, a geometry shader and a fragment shader. See the other
    /// constructors for other possibilities.
    // TODO: replace Box<PipelineLayoutAbstract> with a PipelineUnion struct without template params
    #[inline]
    #[deprecated = "Use the GraphicsPipelineBuilder instead"]
    pub fn with_tessellation_and_geometry<Vs,
                                          Tcs,
                                          Tes,
                                          Gs,
                                          Fs>(
        device: Arc<Device>,
        params: GraphicsPipelineParams<Vdef,
                                       Vs,
                                       Tcs,
                                       Tes,
                                       Gs,
                                       Fs,
                                       Rp>)
        -> Result<GraphicsPipeline<Vdef, Box<PipelineLayoutAbstract + Send + Sync>, Rp>,
                  GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vs::InputDefinition>,
              Vs: GraphicsEntryPointAbstract,
              Fs: GraphicsEntryPointAbstract,
              Gs: GraphicsEntryPointAbstract,
              Tcs: GraphicsEntryPointAbstract,
              Tes: GraphicsEntryPointAbstract,
              Vs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
              Fs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
              Tcs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
              Tes::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
              Gs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
              Tcs::InputDefinition: ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Tes::InputDefinition: ShaderInterfaceDefMatch<Tcs::OutputDefinition>,
              Gs::InputDefinition: ShaderInterfaceDefMatch<Tes::OutputDefinition> + ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Fs::InputDefinition: ShaderInterfaceDefMatch<Gs::OutputDefinition>
                      + ShaderInterfaceDefMatch<Tes::OutputDefinition>
                      + ShaderInterfaceDefMatch<Vs::OutputDefinition>,
              Rp: RenderPassAbstract + RenderPassSubpassInterface<Fs::OutputDefinition>
    {
        let pl;

        if let Some(ref tess) = params.tessellation {
            if let Some(ref gs) = params.geometry_shader {
                if let Err(err) = tess.tessellation_control_shader
                    .input()
                    .matches(params.vertex_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexTessControlStagesMismatch(err));
                }
                if let Err(err) = tess.tessellation_evaluation_shader
                    .input()
                    .matches(tess.tessellation_control_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(err));
                }
                if let Err(err) = gs.input()
                    .matches(tess.tessellation_evaluation_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(err));
                }
                if let Err(err) = params.fragment_shader.input().matches(gs.output()) {
                    return Err(GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(err));
                }

                pl = Box::new(params.vertex_shader.layout().clone()
                    .union(params.fragment_shader.layout().clone())
                    .union(params.tessellation.as_ref().unwrap().tessellation_control_shader.layout().clone())    // FIXME: unwrap()
                    .union(params.tessellation.as_ref().unwrap().tessellation_evaluation_shader.layout().clone())    // FIXME: unwrap()
                    .union(params.geometry_shader.as_ref().unwrap().layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap()) as Box<_>; // TODO: error

            } else {
                if let Err(err) = tess.tessellation_control_shader
                    .input()
                    .matches(params.vertex_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexTessControlStagesMismatch(err));
                }
                if let Err(err) = tess.tessellation_evaluation_shader
                    .input()
                    .matches(tess.tessellation_control_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(err));
                }
                if let Err(err) = params
                    .fragment_shader
                    .input()
                    .matches(tess.tessellation_evaluation_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(err));
                }

                pl = Box::new(params.vertex_shader.layout().clone()
                    .union(params.fragment_shader.layout().clone())
                    .union(params.tessellation.as_ref().unwrap().tessellation_control_shader.layout().clone())    // FIXME: unwrap()
                    .union(params.tessellation.as_ref().unwrap().tessellation_evaluation_shader.layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap()) as Box<_>; // TODO: error
            }

        } else {
            if let Some(ref geometry_shader) = params.geometry_shader {
                if let Err(err) = geometry_shader
                    .input()
                    .matches(params.vertex_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexGeometryStagesMismatch(err));
                }
                if let Err(err) = params
                    .fragment_shader
                    .input()
                    .matches(geometry_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(err));
                }

                pl = Box::new(params.vertex_shader.layout().clone()
                    .union(params.fragment_shader.layout().clone())
                    .union(params.geometry_shader.as_ref().unwrap().layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap()) as Box<_>; // TODO: error

            } else {
                if let Err(err) = params
                    .fragment_shader
                    .input()
                    .matches(params.vertex_shader.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexFragmentStagesMismatch(err));
                }

                pl = Box::new(params
                                  .vertex_shader
                                  .layout()
                                  .clone()
                                  .union(params.fragment_shader.layout().clone())
                                  .build(device.clone())
                                  .unwrap()) as Box<_>; // TODO: error
            }
        }

        GraphicsPipeline::new_inner(device, params, pl)
    }
}

impl<Vdef, L, Rp> GraphicsPipeline<Vdef, L, Rp>
    where L: PipelineLayoutAbstract
{
    fn new_inner<Vs,
                 Tcs,
                 Tes,
                 Gs,
                 Fs>(
        device: Arc<Device>,
        params: GraphicsPipelineParams<Vdef,
                                       Vs,
                                       Tcs,
                                       Tes,
                                       Gs,
                                       Fs,
                                       Rp>,
        pipeline_layout: L)
        -> Result<GraphicsPipeline<Vdef, L, Rp>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vs::InputDefinition>,
              Vs: GraphicsEntryPointAbstract,
              Fs: GraphicsEntryPointAbstract,
              Gs: GraphicsEntryPointAbstract,
              Tcs: GraphicsEntryPointAbstract,
              Tes: GraphicsEntryPointAbstract,
              Rp: RenderPassAbstract + RenderPassDesc + RenderPassSubpassInterface<Fs::OutputDefinition>
    {
        let vk = device.pointers();

        // Checking that the pipeline layout matches the shader stages.
        // TODO: more details in the errors
        PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                   params.vertex_shader.layout())?;
        PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                   params.fragment_shader.layout())?;
        if let Some(ref geometry_shader) = params.geometry_shader {
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout, geometry_shader.layout())?;
        }
        if let Some(ref tess) = params.tessellation {
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                       tess.tessellation_control_shader.layout())?;
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                       tess.tessellation_evaluation_shader
                                                           .layout())?;
        }

        // Check that the subpass can accept the output of the fragment shader.
        if !RenderPassSubpassInterface::is_compatible_with(&params.render_pass.render_pass(),
                                                           params.render_pass.index(),
                                                           params.fragment_shader.output())
        {
            return Err(GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible);
        }

        // Will contain the list of dynamic states. Filled throughout this function.
        let mut dynamic_states: SmallVec<[vk::DynamicState; 8]> = SmallVec::new();

        // List of shader stages.
        let stages = {
            let mut stages = SmallVec::<[_; 5]>::new();

            match params.vertex_shader.ty() {
                GraphicsShaderType::Vertex => {},
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            stages.push(vk::PipelineShaderStageCreateInfo {
                            sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            pNext: ptr::null(),
                            flags: 0, // reserved
                            stage: vk::SHADER_STAGE_VERTEX_BIT,
                            module: params.vertex_shader.module().internal_object(),
                            pName: params.vertex_shader.name().as_ptr(),
                            pSpecializationInfo: ptr::null(), // TODO:
                        });

            match params.fragment_shader.ty() {
                GraphicsShaderType::Fragment => {},
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            stages.push(vk::PipelineShaderStageCreateInfo {
                            sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            pNext: ptr::null(),
                            flags: 0, // reserved
                            stage: vk::SHADER_STAGE_FRAGMENT_BIT,
                            module: params.fragment_shader.module().internal_object(),
                            pName: params.fragment_shader.name().as_ptr(),
                            pSpecializationInfo: ptr::null(), // TODO:
                        });

            if let Some(ref gs) = params.geometry_shader {
                if !device.enabled_features().geometry_shader {
                    return Err(GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled);
                }

                stages.push(vk::PipelineShaderStageCreateInfo {
                                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: ptr::null(),
                                flags: 0, // reserved
                                stage: vk::SHADER_STAGE_GEOMETRY_BIT,
                                module: gs.module().internal_object(),
                                pName: gs.name().as_ptr(),
                                pSpecializationInfo: ptr::null(), // TODO:
                            });
            }

            if let Some(ref tess) = params.tessellation {
                // FIXME: must check that the control shader and evaluation shader are compatible

                if !device.enabled_features().tessellation_shader {
                    return Err(GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled);
                }

                match tess.tessellation_control_shader.ty() {
                    GraphicsShaderType::TessellationControl => {},
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                match tess.tessellation_evaluation_shader.ty() {
                    GraphicsShaderType::TessellationControl => {},
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                stages.push(vk::PipelineShaderStageCreateInfo {
                                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: ptr::null(),
                                flags: 0, // reserved
                                stage: vk::SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                module: tess.tessellation_control_shader.module().internal_object(),
                                pName: tess.tessellation_control_shader.name().as_ptr(),
                                pSpecializationInfo: ptr::null(), // TODO:
                            });

                stages.push(vk::PipelineShaderStageCreateInfo {
                                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: ptr::null(),
                                flags: 0, // reserved
                                stage: vk::SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                                module: tess.tessellation_evaluation_shader
                                    .module()
                                    .internal_object(),
                                pName: tess.tessellation_evaluation_shader.name().as_ptr(),
                                pSpecializationInfo: ptr::null(), // TODO:
                            });
            }

            stages
        };

        // Vertex bindings.
        let (binding_descriptions, attribute_descriptions) = {
            let (buffers_iter, attribs_iter) =
                params
                    .vertex_input
                    .definition(params.vertex_shader.input())?;

            let mut binding_descriptions = SmallVec::<[_; 8]>::new();
            for (num, stride, rate) in buffers_iter {
                if stride >
                    device
                        .physical_device()
                        .limits()
                        .max_vertex_input_binding_stride() as usize
                {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                        binding: num as usize,
                        max: device.physical_device().limits().max_vertex_input_binding_stride() as usize,
                        obtained: stride,
                    });
                }

                binding_descriptions.push(vk::VertexInputBindingDescription {
                                              binding: num as u32,
                                              stride: stride as u32,
                                              inputRate: rate as u32,
                                          });
            }

            let mut attribute_descriptions = SmallVec::<[_; 8]>::new();
            for (loc, binding, info) in attribs_iter {
                // TODO: check attribute format support

                if info.offset >
                    device
                        .physical_device()
                        .limits()
                        .max_vertex_input_attribute_offset() as usize
                {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                        max: device.physical_device().limits().max_vertex_input_attribute_offset() as usize,
                        obtained: info.offset,
                    });
                }

                debug_assert!(binding_descriptions
                                  .iter()
                                  .find(|b| b.binding == binding)
                                  .is_some());

                attribute_descriptions.push(vk::VertexInputAttributeDescription {
                                                location: loc as u32,
                                                binding: binding as u32,
                                                format: info.format as u32,
                                                offset: info.offset as u32,
                                            });
            }

            (binding_descriptions, attribute_descriptions)
        };

        if binding_descriptions.len() >
            device
                .physical_device()
                .limits()
                .max_vertex_input_bindings() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                           max: device
                               .physical_device()
                               .limits()
                               .max_vertex_input_bindings() as
                               usize,
                           obtained: binding_descriptions.len(),
                       });
        }

        if attribute_descriptions.len() >
            device
                .physical_device()
                .limits()
                .max_vertex_input_attributes() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                           max: device
                               .physical_device()
                               .limits()
                               .max_vertex_input_attributes() as
                               usize,
                           obtained: attribute_descriptions.len(),
                       });
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            vertexBindingDescriptionCount: binding_descriptions.len() as u32,
            pVertexBindingDescriptions: binding_descriptions.as_ptr(),
            vertexAttributeDescriptionCount: attribute_descriptions.len() as u32,
            pVertexAttributeDescriptions: attribute_descriptions.as_ptr(),
        };

        if params.input_assembly.primitive_restart_enable &&
            !params.input_assembly.topology.supports_primitive_restart()
        {
            return Err(GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart {
                           primitive: params.input_assembly.topology,
                       });
        }

        // TODO: should check from the tess eval shader instead of the input assembly
        if let Some(ref gs) = params.geometry_shader {
            match gs.ty() {
                GraphicsShaderType::Geometry(primitives) => {
                    if !primitives.matches(params.input_assembly.topology) {
                        return Err(GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader);
                    }
                },
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }
        }

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            topology: params.input_assembly.topology.into(),
            primitiveRestartEnable: if params.input_assembly.primitive_restart_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
        };

        let tessellation = match params.input_assembly.topology {
            PrimitiveTopology::PatchList { vertices_per_patch } => {
                if params.tessellation.is_none() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }
                if vertices_per_patch >
                    device
                        .physical_device()
                        .limits()
                        .max_tessellation_patch_size()
                {
                    return Err(GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded);
                }

                Some(vk::PipelineTessellationStateCreateInfo {
                         sType: vk::STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                         pNext: ptr::null(),
                         flags: 0, // reserved,
                         patchControlPoints: vertices_per_patch,
                     })
            },
            _ => {
                if params.tessellation.is_some() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }

                None
            },
        };

        let (vp_vp, vp_sc, vp_num) = match params.viewport {
            ViewportsState::Fixed { ref data } => (data.iter()
                                                       .map(|e| e.0.clone().into_vulkan_viewport())
                                                       .collect::<SmallVec<[vk::Viewport; 4]>>(),
                                                   data.iter()
                                                       .map(|e| e.1.clone().into_vulkan_rect())
                                                       .collect::<SmallVec<[vk::Rect2D; 4]>>(),
                                                   data.len() as u32),
            ViewportsState::DynamicViewports { ref scissors } => {
                let num = scissors.len() as u32;
                let scissors = scissors
                    .iter()
                    .map(|e| e.clone().into_vulkan_rect())
                    .collect::<SmallVec<[vk::Rect2D; 4]>>();
                dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                (SmallVec::new(), scissors, num)
            },
            ViewportsState::DynamicScissors { ref viewports } => {
                let num = viewports.len() as u32;
                let viewports = viewports
                    .iter()
                    .map(|e| e.clone().into_vulkan_viewport())
                    .collect::<SmallVec<[vk::Viewport; 4]>>();
                dynamic_states.push(vk::DYNAMIC_STATE_SCISSOR);
                (viewports, SmallVec::new(), num)
            },
            ViewportsState::Dynamic { num } => {
                dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                dynamic_states.push(vk::DYNAMIC_STATE_SCISSOR);
                (SmallVec::new(), SmallVec::new(), num)
            },
        };

        if vp_num > 1 && !device.enabled_features().multi_viewport {
            return Err(GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled);
        }

        if vp_num > device.physical_device().limits().max_viewports() {
            return Err(GraphicsPipelineCreationError::MaxViewportsExceeded {
                           obtained: vp_num,
                           max: device.physical_device().limits().max_viewports(),
                       });
        }

        for vp in vp_vp.iter() {
            if vp.width > device.physical_device().limits().max_viewport_dimensions()[0] as f32 ||
                vp.height > device.physical_device().limits().max_viewport_dimensions()[1] as f32
            {
                return Err(GraphicsPipelineCreationError::MaxViewportDimensionsExceeded);
            }

            if vp.x < device.physical_device().limits().viewport_bounds_range()[0] ||
                vp.x + vp.width > device.physical_device().limits().viewport_bounds_range()[1] ||
                vp.y < device.physical_device().limits().viewport_bounds_range()[0] ||
                vp.y + vp.height > device.physical_device().limits().viewport_bounds_range()[1]
            {
                return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
            }
        }

        let viewport_info = vk::PipelineViewportStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            viewportCount: vp_num,
            pViewports: if vp_vp.is_empty() {
                ptr::null()
            } else {
                vp_vp.as_ptr()
            }, // validation layer crashes if you just pass the pointer
            scissorCount: vp_num,
            pScissors: if vp_sc.is_empty() {
                ptr::null()
            } else {
                vp_sc.as_ptr()
            }, // validation layer crashes if you just pass the pointer
        };

        if let Some(line_width) = params.raster.line_width {
            if line_width != 1.0 && !device.enabled_features().wide_lines {
                return Err(GraphicsPipelineCreationError::WideLinesFeatureNotEnabled);
            }
        } else {
            dynamic_states.push(vk::DYNAMIC_STATE_LINE_WIDTH);
        }

        let (db_enable, db_const, db_clamp, db_slope) = match params.raster.depth_bias {
            DepthBiasControl::Dynamic => {
                dynamic_states.push(vk::DYNAMIC_STATE_DEPTH_BIAS);
                (vk::TRUE, 0.0, 0.0, 0.0)
            },
            DepthBiasControl::Disabled => {
                (vk::FALSE, 0.0, 0.0, 0.0)
            },
            DepthBiasControl::Static(bias) => {
                if bias.clamp != 0.0 && !device.enabled_features().depth_bias_clamp {
                    return Err(GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled);
                }

                (vk::TRUE, bias.constant_factor, bias.clamp, bias.slope_factor)
            },
        };

        if params.raster.depth_clamp && !device.enabled_features().depth_clamp {
            return Err(GraphicsPipelineCreationError::DepthClampFeatureNotEnabled);
        }

        if params.raster.polygon_mode != PolygonMode::Fill &&
            !device.enabled_features().fill_mode_non_solid
        {
            return Err(GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled);
        }

        let rasterization = vk::PipelineRasterizationStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            depthClampEnable: if params.raster.depth_clamp {
                vk::TRUE
            } else {
                vk::FALSE
            },
            rasterizerDiscardEnable: if params.raster.rasterizer_discard {
                vk::TRUE
            } else {
                vk::FALSE
            },
            polygonMode: params.raster.polygon_mode as u32,
            cullMode: params.raster.cull_mode as u32,
            frontFace: params.raster.front_face as u32,
            depthBiasEnable: db_enable,
            depthBiasConstantFactor: db_const,
            depthBiasClamp: db_clamp,
            depthBiasSlopeFactor: db_slope,
            lineWidth: params.raster.line_width.unwrap_or(1.0),
        };

        assert!(params.multisample.rasterization_samples >= 1);
        // FIXME: check that rasterization_samples is equal to what's in the renderpass
        if let Some(s) = params.multisample.sample_shading {
            assert!(s >= 0.0 && s <= 1.0);
        }
        let multisample = vk::PipelineMultisampleStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            rasterizationSamples: params.multisample.rasterization_samples,
            sampleShadingEnable: if params.multisample.sample_shading.is_some() {
                vk::TRUE
            } else {
                vk::FALSE
            },
            minSampleShading: params.multisample.sample_shading.unwrap_or(1.0),
            pSampleMask: ptr::null(), //params.multisample.sample_mask.as_ptr(),     // FIXME:
            alphaToCoverageEnable: if params.multisample.alpha_to_coverage {
                vk::TRUE
            } else {
                vk::FALSE
            },
            alphaToOneEnable: if params.multisample.alpha_to_one {
                vk::TRUE
            } else {
                vk::FALSE
            },
        };

        let depth_stencil = {
            let db = match params.depth_stencil.depth_bounds_test {
                DepthBounds::Disabled => (vk::FALSE, 0.0, 0.0),
                DepthBounds::Fixed(ref range) => {
                    if !device.enabled_features().depth_bounds {
                        return Err(GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled);
                    }

                    (vk::TRUE, range.start, range.end)
                },
                DepthBounds::Dynamic => {
                    if !device.enabled_features().depth_bounds {
                        return Err(GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled);
                    }

                    dynamic_states.push(vk::DYNAMIC_STATE_DEPTH_BOUNDS);

                    (vk::TRUE, 0.0, 1.0)
                },
            };

            match (params.depth_stencil.stencil_front.compare_mask,
                     params.depth_stencil.stencil_back.compare_mask) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_COMPARE_MASK);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            match (params.depth_stencil.stencil_front.write_mask,
                     params.depth_stencil.stencil_back.write_mask) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_WRITE_MASK);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            match (params.depth_stencil.stencil_front.reference,
                     params.depth_stencil.stencil_back.reference) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_REFERENCE);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            if params.depth_stencil.depth_write && !params.render_pass.has_writable_depth() {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if params.depth_stencil.depth_compare != Compare::Always &&
                !params.render_pass.has_depth()
            {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if (!params.depth_stencil.stencil_front.always_keep() ||
                    !params.depth_stencil.stencil_back.always_keep()) &&
                !params.render_pass.has_stencil()
            {
                return Err(GraphicsPipelineCreationError::NoStencilAttachment);
            }

            // FIXME: stencil writability

            vk::PipelineDepthStencilStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                depthTestEnable: if !params.depth_stencil.depth_write &&
                    params.depth_stencil.depth_compare == Compare::Always
                {
                    vk::FALSE
                } else {
                    vk::TRUE
                },
                depthWriteEnable: if params.depth_stencil.depth_write {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                depthCompareOp: params.depth_stencil.depth_compare as u32,
                depthBoundsTestEnable: db.0,
                stencilTestEnable: if params.depth_stencil.stencil_front.always_keep() &&
                    params.depth_stencil.stencil_back.always_keep()
                {
                    vk::FALSE
                } else {
                    vk::TRUE
                },
                front: vk::StencilOpState {
                    failOp: params.depth_stencil.stencil_front.fail_op as u32,
                    passOp: params.depth_stencil.stencil_front.pass_op as u32,
                    depthFailOp: params.depth_stencil.stencil_front.depth_fail_op as u32,
                    compareOp: params.depth_stencil.stencil_front.compare as u32,
                    compareMask: params
                        .depth_stencil
                        .stencil_front
                        .compare_mask
                        .unwrap_or(u32::MAX),
                    writeMask: params
                        .depth_stencil
                        .stencil_front
                        .write_mask
                        .unwrap_or(u32::MAX),
                    reference: params.depth_stencil.stencil_front.reference.unwrap_or(0),
                },
                back: vk::StencilOpState {
                    failOp: params.depth_stencil.stencil_back.fail_op as u32,
                    passOp: params.depth_stencil.stencil_back.pass_op as u32,
                    depthFailOp: params.depth_stencil.stencil_back.depth_fail_op as u32,
                    compareOp: params.depth_stencil.stencil_back.compare as u32,
                    compareMask: params
                        .depth_stencil
                        .stencil_back
                        .compare_mask
                        .unwrap_or(u32::MAX),
                    writeMask: params
                        .depth_stencil
                        .stencil_back
                        .write_mask
                        .unwrap_or(u32::MAX),
                    reference: params.depth_stencil.stencil_back.reference.unwrap_or(0),
                },
                minDepthBounds: db.1,
                maxDepthBounds: db.2,
            }
        };

        let blend_atch: SmallVec<[vk::PipelineColorBlendAttachmentState; 8]> = {
            let num_atch = params.render_pass.num_color_attachments();

            match params.blend.attachments {
                AttachmentsBlend::Collective(blend) => {
                    (0 .. num_atch).map(|_| blend.clone().into_vulkan_state()).collect()
                },
                AttachmentsBlend::Individual(blend) => {
                    if blend.len() != num_atch as usize {
                        return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
                    }

                    if !device.enabled_features().independent_blend {
                        return Err(GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled);
                    }

                    blend.iter().map(|b| b.clone().into_vulkan_state()).collect()
                },
            }
        };

        let blend = vk::PipelineColorBlendStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            logicOpEnable: if params.blend.logic_op.is_some() {
                if !device.enabled_features().logic_op {
                    return Err(GraphicsPipelineCreationError::LogicOpFeatureNotEnabled);
                }
                vk::TRUE
            } else {
                vk::FALSE
            },
            logicOp: params.blend.logic_op.unwrap_or(Default::default()) as u32,
            attachmentCount: blend_atch.len() as u32,
            pAttachments: blend_atch.as_ptr(),
            blendConstants: if let Some(c) = params.blend.blend_constants {
                c
            } else {
                dynamic_states.push(vk::DYNAMIC_STATE_BLEND_CONSTANTS);
                [0.0, 0.0, 0.0, 0.0]
            },
        };

        let dynamic_states = if !dynamic_states.is_empty() {
            Some(vk::PipelineDynamicStateCreateInfo {
                     sType: vk::STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                     pNext: ptr::null(),
                     flags: 0, // reserved
                     dynamicStateCount: dynamic_states.len() as u32,
                     pDynamicStates: dynamic_states.as_ptr(),
                 })
        } else {
            None
        };

        let pipeline = unsafe {
            let infos = vk::GraphicsPipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // TODO: some flags are available but none are critical
                stageCount: stages.len() as u32,
                pStages: stages.as_ptr(),
                pVertexInputState: &vertex_input_state,
                pInputAssemblyState: &input_assembly,
                pTessellationState: tessellation
                    .as_ref()
                    .map(|t| t as *const _)
                    .unwrap_or(ptr::null()),
                pViewportState: &viewport_info,
                pRasterizationState: &rasterization,
                pMultisampleState: &multisample,
                pDepthStencilState: &depth_stencil,
                pColorBlendState: &blend,
                pDynamicState: dynamic_states
                    .as_ref()
                    .map(|s| s as *const _)
                    .unwrap_or(ptr::null()),
                layout: PipelineLayoutAbstract::sys(&pipeline_layout).internal_object(),
                renderPass: params.render_pass.render_pass().inner().internal_object(),
                subpass: params.render_pass.index(),
                basePipelineHandle: 0, // TODO:
                basePipelineIndex: -1, // TODO:
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateGraphicsPipelines(device.internal_object(),
                                                    0,
                                                    1,
                                                    &infos,
                                                    ptr::null(),
                                                    &mut output))?;
            output
        };

        let (render_pass, render_pass_subpass) = params.render_pass.into();

        Ok(GraphicsPipeline {
               inner: Inner {
                   device: device.clone(),
                   pipeline: pipeline,
               },
               layout: pipeline_layout,

               vertex_definition: params.vertex_input,

               render_pass: render_pass,
               render_pass_subpass: render_pass_subpass,

               dynamic_line_width: params.raster.line_width.is_none(),
               dynamic_viewport: params.viewport.dynamic_viewports(),
               dynamic_scissor: params.viewport.dynamic_scissors(),
               dynamic_depth_bias: params.raster.depth_bias.is_dynamic(),
               dynamic_depth_bounds: params.depth_stencil.depth_bounds_test.is_dynamic(),
               dynamic_stencil_compare_mask: params
                   .depth_stencil
                   .stencil_back
                   .compare_mask
                   .is_none(),
               dynamic_stencil_write_mask: params.depth_stencil.stencil_back.write_mask.is_none(),
               dynamic_stencil_reference: params.depth_stencil.stencil_back.reference.is_none(),
               dynamic_blend_constants: params.blend.blend_constants.is_none(),

               num_viewports: params.viewport.num_viewports(),
           })
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp> {
    /// Returns the vertex definition used in the constructor.
    #[inline]
    pub fn vertex_definition(&self) -> &Mv {
        &self.vertex_definition
    }

    /// Returns the device used to create this pipeline.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutAbstract
{
    /// Returns the pipeline layout used in the constructor.
    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassDesc
{
    /// Returns the pass used in the constructor.
    #[inline]
    pub fn subpass(&self) -> Subpass<&Rp> {
        Subpass::from(&self.render_pass, self.render_pass_subpass).unwrap()
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp> {
    /// Returns the render pass used in the constructor.
    #[inline]
    pub fn render_pass(&self) -> &Rp {
        &self.render_pass
    }

    /// Returns true if the line width used by this pipeline is dynamic.
    #[inline]
    pub fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }

    /// Returns the number of viewports and scissors of this pipeline.
    #[inline]
    pub fn num_viewports(&self) -> u32 {
        self.num_viewports
    }

    /// Returns true if the viewports used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_viewports(&self) -> bool {
        self.dynamic_viewport
    }

    /// Returns true if the scissors used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_scissors(&self) -> bool {
        self.dynamic_scissor
    }

    /// Returns true if the depth bounds used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_depth_bounds(&self) -> bool {
        self.dynamic_depth_bounds
    }

    /// Returns true if the stencil compare masks used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_compare_mask(&self) -> bool {
        self.dynamic_stencil_compare_mask
    }

    /// Returns true if the stencil write masks used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_write_mask(&self) -> bool {
        self.dynamic_stencil_write_mask
    }

    /// Returns true if the stencil references used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_reference(&self) -> bool {
        self.dynamic_stencil_reference
    }
}

unsafe impl<Mv, L, Rp> PipelineLayoutAbstract for GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutAbstract
{
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        self.layout.sys()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layout.descriptor_set_layout(index)
    }
}

unsafe impl<Mv, L, Rp> PipelineLayoutDesc for GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutDesc
{
    #[inline]
    fn num_sets(&self) -> usize {
        self.layout.num_sets()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.layout.num_bindings_in_set(set)
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.layout.descriptor(set, binding)
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        self.layout.num_push_constants_ranges()
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        self.layout.push_constants_range(num)
    }
}

unsafe impl<Mv, L, Rp> DeviceOwned for GraphicsPipeline<Mv, L, Rp> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

impl<Mv, L, Rp> fmt::Debug for GraphicsPipeline<Mv, L, Rp> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan graphics pipeline {:?}>", self.inner.pipeline)
    }
}

unsafe impl<Mv, L, Rp> RenderPassAbstract for GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassAbstract
{
    #[inline]
    fn inner(&self) -> RenderPassSys {
        self.render_pass.inner()
    }
}

unsafe impl<Mv, L, Rp> RenderPassDesc for GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassDesc
{
    #[inline]
    fn num_attachments(&self) -> usize {
        self.render_pass.num_attachments()
    }

    #[inline]
    fn attachment_desc(&self, num: usize) -> Option<LayoutAttachmentDescription> {
        self.render_pass.attachment_desc(num)
    }

    #[inline]
    fn num_subpasses(&self) -> usize {
        self.render_pass.num_subpasses()
    }

    #[inline]
    fn subpass_desc(&self, num: usize) -> Option<LayoutPassDescription> {
        self.render_pass.subpass_desc(num)
    }

    #[inline]
    fn num_dependencies(&self) -> usize {
        self.render_pass.num_dependencies()
    }

    #[inline]
    fn dependency_desc(&self, num: usize) -> Option<LayoutPassDependencyDescription> {
        self.render_pass.dependency_desc(num)
    }
}

unsafe impl<C, Mv, L, Rp> RenderPassDescClearValues<C> for GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPassDescClearValues<C>
{
    #[inline]
    fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
        self.render_pass.convert_clear_values(vals)
    }
}

unsafe impl<Mv, L, Rp> VulkanObject for GraphicsPipeline<Mv, L, Rp> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.inner.pipeline
    }
}

impl Drop for Inner {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}

/// Trait implemented on objects that reference a graphics pipeline. Can be made into a trait
/// object.
pub unsafe trait GraphicsPipelineAbstract: PipelineLayoutAbstract + RenderPassAbstract + VertexSource<Vec<Arc<BufferAccess + Send + Sync>>> {
    /// Returns an opaque object that represents the inside of the graphics pipeline.
    fn inner(&self) -> GraphicsPipelineSys;

    /// Returns the index of the subpass this graphics pipeline is rendering to.
    fn subpass_index(&self) -> u32;

    /// Returns the subpass this graphics pipeline is rendering to.
    #[inline]
    fn subpass(self) -> Subpass<Self> where Self: Sized {
        let index = self.subpass_index();
        Subpass::from(self, index).expect("Wrong subpass index in GraphicsPipelineAbstract::subpass")
    }

    /// Returns true if the line width used by this pipeline is dynamic.
    fn has_dynamic_line_width(&self) -> bool;

    /// Returns the number of viewports and scissors of this pipeline.
    fn num_viewports(&self) -> u32;

    /// Returns true if the viewports used by this pipeline are dynamic.
    fn has_dynamic_viewports(&self) -> bool;

    /// Returns true if the scissors used by this pipeline are dynamic.
    fn has_dynamic_scissors(&self) -> bool;

    /// Returns true if the depth bounds used by this pipeline are dynamic.
    fn has_dynamic_depth_bounds(&self) -> bool;

    /// Returns true if the stencil compare masks used by this pipeline are dynamic.
    fn has_dynamic_stencil_compare_mask(&self) -> bool;

    /// Returns true if the stencil write masks used by this pipeline are dynamic.
    fn has_dynamic_stencil_write_mask(&self) -> bool;

    /// Returns true if the stencil references used by this pipeline are dynamic.
    fn has_dynamic_stencil_reference(&self) -> bool;
}

unsafe impl<Mv, L, Rp> GraphicsPipelineAbstract for GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayoutAbstract,
          Rp: RenderPassAbstract,
          Mv: VertexSource<Vec<Arc<BufferAccess + Send + Sync>>>
{
    #[inline]
    fn inner(&self) -> GraphicsPipelineSys {
        GraphicsPipelineSys(self.inner.pipeline, PhantomData)
    }

    #[inline]
    fn subpass_index(&self) -> u32 {
        self.render_pass_subpass
    }

    #[inline]
    fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }

    #[inline]
    fn num_viewports(&self) -> u32 {
        self.num_viewports
    }

    #[inline]
    fn has_dynamic_viewports(&self) -> bool {
        self.dynamic_viewport
    }

    #[inline]
    fn has_dynamic_scissors(&self) -> bool {
        self.dynamic_scissor
    }

    #[inline]
    fn has_dynamic_depth_bounds(&self) -> bool {
        self.dynamic_depth_bounds
    }

    #[inline]
    fn has_dynamic_stencil_compare_mask(&self) -> bool {
        self.dynamic_stencil_compare_mask
    }

    #[inline]
    fn has_dynamic_stencil_write_mask(&self) -> bool {
        self.dynamic_stencil_write_mask
    }

    #[inline]
    fn has_dynamic_stencil_reference(&self) -> bool {
        self.dynamic_stencil_reference
    }
}

unsafe impl<T> GraphicsPipelineAbstract for T
    where T: SafeDeref,
          T::Target: GraphicsPipelineAbstract
{
    #[inline]
    fn inner(&self) -> GraphicsPipelineSys {
        GraphicsPipelineAbstract::inner(&**self)
    }

    #[inline]
    fn subpass_index(&self) -> u32 {
        (**self).subpass_index()
    }

    #[inline]
    fn has_dynamic_line_width(&self) -> bool {
        (**self).has_dynamic_line_width()
    }

    #[inline]
    fn num_viewports(&self) -> u32 {
        (**self).num_viewports()
    }

    #[inline]
    fn has_dynamic_viewports(&self) -> bool {
        (**self).has_dynamic_viewports()
    }

    #[inline]
    fn has_dynamic_scissors(&self) -> bool {
        (**self).has_dynamic_scissors()
    }

    #[inline]
    fn has_dynamic_depth_bounds(&self) -> bool {
        (**self).has_dynamic_depth_bounds()
    }

    #[inline]
    fn has_dynamic_stencil_compare_mask(&self) -> bool {
        (**self).has_dynamic_stencil_compare_mask()
    }

    #[inline]
    fn has_dynamic_stencil_write_mask(&self) -> bool {
        (**self).has_dynamic_stencil_write_mask()
    }

    #[inline]
    fn has_dynamic_stencil_reference(&self) -> bool {
        (**self).has_dynamic_stencil_reference()
    }
}

/// Opaque object that represents the inside of the graphics pipeline.
#[derive(Debug, Copy, Clone)]
pub struct GraphicsPipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for GraphicsPipelineSys<'a> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.0
    }
}

unsafe impl<Mv, L, Rp, I> VertexDefinition<I> for GraphicsPipeline<Mv, L, Rp>
    where Mv: VertexDefinition<I>
{
    type BuffersIter = <Mv as VertexDefinition<I>>::BuffersIter;
    type AttribsIter = <Mv as VertexDefinition<I>>::AttribsIter;

    #[inline]
    fn definition(
        &self, interface: &I)
        -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        self.vertex_definition.definition(interface)
    }
}

unsafe impl<Mv, L, Rp, S> VertexSource<S> for GraphicsPipeline<Mv, L, Rp>
    where Mv: VertexSource<S>
{
    #[inline]
    fn decode(&self, s: S) -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
        self.vertex_definition.decode(s)
    }
}

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphicsPipelineCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The pipeline layout is not compatible with what the shaders expect.
    IncompatiblePipelineLayout(PipelineLayoutNotSupersetError),

    /// The interface between the vertex shader and the geometry shader mismatches.
    VertexGeometryStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the vertex shader and the tessellation control shader mismatches.
    VertexTessControlStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the vertex shader and the fragment shader mismatches.
    VertexFragmentStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the tessellation control shader and the tessellation evaluation
    /// shader mismatches.
    TessControlTessEvalStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the tessellation evaluation shader and the geometry shader
    /// mismatches.
    TessEvalGeometryStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the tessellation evaluation shader and the fragment shader
    /// mismatches.
    TessEvalFragmentStagesMismatch(ShaderInterfaceMismatchError),

    /// The interface between the geometry shader and the fragment shader mismatches.
    GeometryFragmentStagesMismatch(ShaderInterfaceMismatchError),

    /// The output of the fragment shader is not compatible with what the render pass subpass
    /// expects.
    FragmentShaderRenderPassIncompatible,

    /// The vertex definition is not compatible with the input of the vertex shader.
    IncompatibleVertexDefinition(IncompatibleVertexDefinitionError),

    /// The maximum stride value for vertex input (ie. the distance between two vertex elements)
    /// has been exceeded.
    MaxVertexInputBindingStrideExceeded {
        /// Index of the faulty binding.
        binding: usize,
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum number of vertex sources has been exceeded.
    MaxVertexInputBindingsExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum offset for a vertex attribute has been exceeded. This means that your vertex
    /// struct is too large.
    MaxVertexInputAttributeOffsetExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum number of vertex attributes has been exceeded.
    MaxVertexInputAttributesExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The user requested to use primitive restart, but the primitive topology doesn't support it.
    PrimitiveDoesntSupportPrimitiveRestart {
        /// The topology that doesn't support primitive restart.
        primitive: PrimitiveTopology,
    },

    /// The `multi_viewport` feature must be enabled in order to use multiple viewports at once.
    MultiViewportFeatureNotEnabled,

    /// The maximum number of viewports has been exceeded.
    MaxViewportsExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// The maximum dimensions of viewports has been exceeded.
    MaxViewportDimensionsExceeded,

    /// The minimum or maximum bounds of viewports have been exceeded.
    ViewportBoundsExceeded,

    /// The `wide_lines` feature must be enabled in order to use a line width superior to 1.0.
    WideLinesFeatureNotEnabled,

    /// The `depth_clamp` feature must be enabled in order to use depth clamping.
    DepthClampFeatureNotEnabled,

    /// The `depth_bias_clamp` feature must be enabled in order to use a depth bias clamp different
    /// from 0.0.
    DepthBiasClampFeatureNotEnabled,

    /// The `fill_mode_non_solid` feature must be enabled in order to use a polygon mode different
    /// from `Fill`.
    FillModeNonSolidFeatureNotEnabled,

    /// The `depth_bounds` feature must be enabled in order to use depth bounds testing.
    DepthBoundsFeatureNotEnabled,

    /// The requested stencil test is invalid.
    WrongStencilState,

    /// The primitives topology does not match what the geometry shader expects.
    TopologyNotMatchingGeometryShader,

    /// The `geometry_shader` feature must be enabled in order to use geometry shaders.
    GeometryShaderFeatureNotEnabled,

    /// The `tessellation_shader` feature must be enabled in order to use tessellation shaders.
    TessellationShaderFeatureNotEnabled,

    /// The number of attachments specified in the blending does not match the number of
    /// attachments in the subpass.
    MismatchBlendingAttachmentsCount,

    /// The `independent_blend` feature must be enabled in order to use different blending
    /// operations per attachment.
    IndependentBlendFeatureNotEnabled,

    /// The `logic_op` feature must be enabled in order to use logic operations.
    LogicOpFeatureNotEnabled,

    /// The depth test requires a depth attachment but render pass has no depth attachment, or
    /// depth writing is enabled and the depth attachment is read-only.
    NoDepthAttachment,

    /// The stencil test requires a stencil attachment but render pass has no stencil attachment, or
    /// stencil writing is enabled and the stencil attachment is read-only.
    NoStencilAttachment,

    /// Tried to use a patch list without a tessellation shader, or a non-patch-list with a
    /// tessellation shader.
    InvalidPrimitiveTopology,

    /// The `maxTessellationPatchSize` limit was exceeded.
    MaxTessellationPatchSizeExceeded,

    /// The wrong type of shader has been passed.
    ///
    /// For example you passed a vertex shader as the fragment shader.
    WrongShaderType,
}

impl error::Error for GraphicsPipelineCreationError {
    #[inline]
    // TODO: finish
    fn description(&self) -> &str {
        match *self {
            GraphicsPipelineCreationError::OomError(_) => "not enough memory available",
            GraphicsPipelineCreationError::VertexGeometryStagesMismatch(_) => {
                "the interface between the vertex shader and the geometry shader mismatches"
            },
            GraphicsPipelineCreationError::VertexTessControlStagesMismatch(_) => {
                "the interface between the vertex shader and the tessellation control shader \
                 mismatches"
            },
            GraphicsPipelineCreationError::VertexFragmentStagesMismatch(_) => {
                "the interface between the vertex shader and the fragment shader mismatches"
            },
            GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(_) => {
                "the interface between the tessellation control shader and the tessellation \
                 evaluation shader mismatches"
            },
            GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(_) => {
                "the interface between the tessellation evaluation shader and the geometry \
                 shader mismatches"
            },
            GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(_) => {
                "the interface between the tessellation evaluation shader and the fragment \
                 shader mismatches"
            },
            GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(_) => {
                "the interface between the geometry shader and the fragment shader mismatches"
            },
            GraphicsPipelineCreationError::IncompatiblePipelineLayout(_) => {
                "the pipeline layout is not compatible with what the shaders expect"
            },
            GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible => {
                "the output of the fragment shader is not compatible with what the render pass \
                 subpass expects"
            },
            GraphicsPipelineCreationError::IncompatibleVertexDefinition(_) => {
                "the vertex definition is not compatible with the input of the vertex shader"
            },
            GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded { .. } => {
                "the maximum stride value for vertex input (ie. the distance between two vertex \
                 elements) has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded { .. } => {
                "the maximum number of vertex sources has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded { .. } => {
                "the maximum offset for a vertex attribute has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded { .. } => {
                "the maximum number of vertex attributes has been exceeded"
            },
            GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart { .. } => {
                "the user requested to use primitive restart, but the primitive topology \
                 doesn't support it"
            },
            GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled => {
                "the `multi_viewport` feature must be enabled in order to use multiple viewports \
                 at once"
            },
            GraphicsPipelineCreationError::MaxViewportsExceeded { .. } => {
                "the maximum number of viewports has been exceeded"
            },
            GraphicsPipelineCreationError::MaxViewportDimensionsExceeded => {
                "the maximum dimensions of viewports has been exceeded"
            },
            GraphicsPipelineCreationError::ViewportBoundsExceeded => {
                "the minimum or maximum bounds of viewports have been exceeded"
            },
            GraphicsPipelineCreationError::WideLinesFeatureNotEnabled => {
                "the `wide_lines` feature must be enabled in order to use a line width \
                 superior to 1.0"
            },
            GraphicsPipelineCreationError::DepthClampFeatureNotEnabled => {
                "the `depth_clamp` feature must be enabled in order to use depth clamping"
            },
            GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled => {
                "the `depth_bias_clamp` feature must be enabled in order to use a depth bias \
                 clamp different from 0.0."
            },
            GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled => {
                "the `fill_mode_non_solid` feature must be enabled in order to use a polygon mode \
                 different from `Fill`"
            },
            GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled => {
                "the `depth_bounds` feature must be enabled in order to use depth bounds testing"
            },
            GraphicsPipelineCreationError::WrongStencilState => {
                "the requested stencil test is invalid"
            },
            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader => {
                "the primitives topology does not match what the geometry shader expects"
            },
            GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled => {
                "the `geometry_shader` feature must be enabled in order to use geometry shaders"
            },
            GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled => {
                "the `tessellation_shader` feature must be enabled in order to use tessellation \
                 shaders"
            },
            GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount => {
                "the number of attachments specified in the blending does not match the number of \
                 attachments in the subpass"
            },
            GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled => {
                "the `independent_blend` feature must be enabled in order to use different \
                 blending operations per attachment"
            },
            GraphicsPipelineCreationError::LogicOpFeatureNotEnabled => {
                "the `logic_op` feature must be enabled in order to use logic operations"
            },
            GraphicsPipelineCreationError::NoDepthAttachment => {
                "the depth attachment of the render pass does not match the depth test"
            },
            GraphicsPipelineCreationError::NoStencilAttachment => {
                "the stencil attachment of the render pass does not match the stencil test"
            },
            GraphicsPipelineCreationError::InvalidPrimitiveTopology => {
                "trying to use a patch list without a tessellation shader, or a non-patch-list \
                 with a tessellation shader"
            },
            GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded => {
                "the maximum tessellation patch size was exceeded"
            },
            GraphicsPipelineCreationError::WrongShaderType => {
                "the wrong type of shader has been passed"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            GraphicsPipelineCreationError::OomError(ref err) => Some(err),
            GraphicsPipelineCreationError::IncompatiblePipelineLayout(ref err) => Some(err),
            GraphicsPipelineCreationError::VertexGeometryStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::VertexTessControlStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::VertexFragmentStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(ref err) => Some(err),
            GraphicsPipelineCreationError::IncompatibleVertexDefinition(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for GraphicsPipelineCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: OomError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::OomError(err)
    }
}

impl From<PipelineLayoutNotSupersetError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutNotSupersetError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::IncompatiblePipelineLayout(err)
    }
}

impl From<IncompatibleVertexDefinitionError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: IncompatibleVertexDefinitionError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::IncompatibleVertexDefinition(err)
    }
}

impl From<Error> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: Error) -> GraphicsPipelineCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
