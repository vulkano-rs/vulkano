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

use std::sync::Arc;
use descriptor::pipeline_layout::EmptyPipelineDesc;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use descriptor::pipeline_layout::PipelineLayoutDescNames;
use device::Device;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassSubpassInterface;
use framebuffer::Subpass;
use pipeline::blend::Blend;
use pipeline::blend::AttachmentsBlend;
use pipeline::blend::AttachmentBlend;
use pipeline::blend::LogicOp;
use pipeline::depth_stencil::DepthStencil;
use pipeline::graphics_pipeline::GraphicsPipeline;
use pipeline::graphics_pipeline::GraphicsPipelineCreationError;
use pipeline::graphics_pipeline::GraphicsPipelineParams;
use pipeline::graphics_pipeline::GraphicsPipelineParamsTess;
use pipeline::input_assembly::InputAssembly;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::multisample::Multisample;
use pipeline::raster::CullMode;
use pipeline::raster::FrontFace;
use pipeline::raster::PolygonMode;
use pipeline::raster::Rasterization;
use pipeline::shader::EmptyShaderInterfaceDef;
use pipeline::shader::ShaderInterfaceDef;
use pipeline::shader::ShaderInterfaceDefMatch;
use pipeline::shader::VertexShaderEntryPoint;
use pipeline::shader::TessControlShaderEntryPoint;
use pipeline::shader::TessEvaluationShaderEntryPoint;
use pipeline::shader::GeometryShaderEntryPoint;
use pipeline::shader::FragmentShaderEntryPoint;
use pipeline::vertex::SingleBufferDefinition;
use pipeline::vertex::VertexDefinition;
use pipeline::viewport::Scissor;
use pipeline::viewport::Viewport;
use pipeline::viewport::ViewportsState;

/// Prototype for a `GraphicsPipeline`.
// TODO: we can optimize this by filling directly the raw vk structs
pub struct GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                                   Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp> {
    vertex_input: Vdef,
    vertex_shader: Option<VertexShaderEntryPoint<'a, Vsp, Vi, Vo, Vl>>,
    input_assembly: InputAssembly,
    tessellation: Option<GraphicsPipelineParamsTess<'a, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel>>,
    geometry_shader: Option<GeometryShaderEntryPoint<'a, Gs, Gi, Go, Gl>>,
    viewport: Option<ViewportsState>,
    raster: Rasterization,
    multisample: Multisample,
    fragment_shader: Option<FragmentShaderEntryPoint<'a, Fs, Fi, Fo, Fl>>,
    depth_stencil: DepthStencil,
    blend: Blend,
    render_pass: Option<Subpass<Rp>>,
}

impl<'a> GraphicsPipelineBuilder<'a, SingleBufferDefinition<()>, (), (), (), (), (),
                                 EmptyShaderInterfaceDef, EmptyShaderInterfaceDef,
                                 EmptyPipelineDesc, (), EmptyShaderInterfaceDef,
                                 EmptyShaderInterfaceDef, EmptyPipelineDesc, (),
                                 EmptyShaderInterfaceDef, EmptyShaderInterfaceDef,
                                 EmptyPipelineDesc, (), EmptyShaderInterfaceDef,
                                 EmptyShaderInterfaceDef, EmptyPipelineDesc, ()>
{
    /// Builds a new empty builder.
    pub(super) fn new() -> Self {
        GraphicsPipelineBuilder {
            vertex_input: SingleBufferDefinition::new(),        // TODO: should be empty attrs instead
            vertex_shader: None,
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: None,
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: None,
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            render_pass: None,
        }
    }
}

impl<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi,
     Fo, Fl, Rp>
    GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                            Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    where Vdef: VertexDefinition<Vi>,
          Vl: PipelineLayoutDescNames + Clone + 'static + Send + Sync,        // TODO: Clone + 'static + Send + Sync shouldn't be required
          Fl: PipelineLayoutDescNames + Clone + 'static + Send + Sync,        // TODO: Clone + 'static + Send + Sync shouldn't be required
          Tcl: PipelineLayoutDescNames + Clone + 'static + Send + Sync,       // TODO: Clone + 'static + Send + Sync shouldn't be required
          Tel: PipelineLayoutDescNames + Clone + 'static + Send + Sync,       // TODO: Clone + 'static + Send + Sync shouldn't be required
          Gl: PipelineLayoutDescNames + Clone + 'static + Send + Sync,        // TODO: Clone + 'static + Send + Sync shouldn't be required
          Tci: ShaderInterfaceDefMatch<Vo>,
          Tei: ShaderInterfaceDefMatch<Tco>,
          Gi: ShaderInterfaceDefMatch<Teo> + ShaderInterfaceDefMatch<Vo>,
          Vo: ShaderInterfaceDef,
          Tco: ShaderInterfaceDef,
          Teo: ShaderInterfaceDef,
          Go: ShaderInterfaceDef,
          Fi: ShaderInterfaceDefMatch<Go> + ShaderInterfaceDefMatch<Teo> + ShaderInterfaceDefMatch<Vo>,
          Fo: ShaderInterfaceDef,
          Rp: RenderPassAbstract + RenderPassSubpassInterface<Fo>,
{
    /// Builds the graphics pipeline.
    // TODO: replace Box<PipelineLayoutAbstract> with a PipelineUnion struct without template params
    pub fn build(self, device: Arc<Device>) -> Result<GraphicsPipeline<Vdef, Box<PipelineLayoutAbstract + Send + Sync>, Rp>, GraphicsPipelineCreationError> {
        // TODO: return errors instead of panicking if missing param
        GraphicsPipeline::with_tessellation_and_geometry(device, GraphicsPipelineParams {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader.expect("Vertex shader not specified in the builder"),
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport.expect("Viewport state not specified in the builder"),
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader.expect("Fragment shader not specified in the builder"),
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass.expect("Render pass not specified in the builder"),
        })
    }
}

impl<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi,
     Fo, Fl, Rp>
    GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                            Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
{
    /// Sets the vertex input.
    #[inline]
    pub fn vertex_input<T>(self, vertex_input: T)
        -> GraphicsPipelineBuilder<'a, T, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                                   Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    {
        GraphicsPipelineBuilder {
            vertex_input: vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the vertex input to a single vertex buffer.
    ///
    /// You will most likely need to explicitely specify the template parameter to the type of a
    /// vertex.
    #[inline]
    pub fn vertex_input_single_buffer<V>(self)
        -> GraphicsPipelineBuilder<'a, SingleBufferDefinition<V>, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco,
                                   Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    {
        self.vertex_input(SingleBufferDefinition::<V>::new())
    }

    /// Sets the vertex shader to use.
    #[inline]
    pub fn vertex_shader<Vsp2, Vi2, Vo2, Vl2>(self, shader: VertexShaderEntryPoint<'a, Vsp2, Vi2, Vo2, Vl2>)
        -> GraphicsPipelineBuilder<'a, Vdef, Vsp2, Vi2, Vo2, Vl2, Tcs, Tci, Tco,
                                   Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: Some(shader),
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets whether primitive restart if enabled.
    #[inline]
    pub fn primitive_restart(mut self, enabled: bool) -> Self {
        self.input_assembly.primitive_restart_enable = enabled;
        self
    }

    /// Sets the topology of the primitives that are expected by the pipeline.
    #[inline]
    pub fn primitive_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.input_assembly.topology = topology;
        self
    }

    /// Sets the topology of the primitives to a list of points.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PointList)`.
    #[inline]
    pub fn point_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::PointList)
    }
    
    /// Sets the topology of the primitives to a list of lines.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineList)`.
    #[inline]
    pub fn line_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineList)
    }
    
    /// Sets the topology of the primitives to a line strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStrip)`.
    #[inline]
    pub fn line_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStrip)
    }
    
    /// Sets the topology of the primitives to a list of triangles. Note that this is the default.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleList)`.
    #[inline]
    pub fn triangle_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleList)
    }
    
    /// Sets the topology of the primitives to a triangle strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStrip)`.
    #[inline]
    pub fn triangle_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStrip)
    }
    
    /// Sets the topology of the primitives to a fan of triangles.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleFan)`.
    #[inline]
    pub fn triangle_fan(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleFan)
    }
    
    /// Sets the topology of the primitives to a list of lines with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)`.
    #[inline]
    pub fn line_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)
    }
    
    /// Sets the topology of the primitives to a line strip with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)`.
    #[inline]
    pub fn line_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)
    }
    
    /// Sets the topology of the primitives to a list of triangles with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)`.
    #[inline]
    pub fn triangle_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)
    }
    
    /// Sets the topology of the primitives to a triangle strip with adjacency information`
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)`.
    #[inline]
    pub fn triangle_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)
    }
    
    /// Sets the topology of the primitives to a list of patches. Can only be used and must be used
    /// with a tessellation shader.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PatchList { vertices_per_patch })`.
    #[inline]
    pub fn patch_list(self, vertices_per_patch: u32) -> Self {
        self.primitive_topology(PrimitiveTopology::PatchList { vertices_per_patch })
    }

    /// Sets the tessellation shaders to use.
    #[inline]
    pub fn tessellation_shaders<Tcs2, Tci2, Tco2, Tcl2, Tes2, Tei2, Teo2, Tel2>(self,
        tessellation_control_shader: TessControlShaderEntryPoint<'a, Tcs2, Tci2, Tco2, Tcl2>,
        tessellation_evaluation_shader: TessEvaluationShaderEntryPoint<'a, Tes2, Tei2, Teo2, Tel2>)
        -> GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs2, Tci2, Tco2,
                                   Tcl2, Tes2, Tei2, Teo2, Tel2, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            tessellation: Some(GraphicsPipelineParamsTess {
                tessellation_control_shader: tessellation_control_shader,
                tessellation_evaluation_shader: tessellation_evaluation_shader,
            }),
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the tessellation shaders stage as disabled. This is the default.
    #[inline]
    pub fn tessellation_shaders_disabled(mut self) -> Self {
        self.tessellation = None;
        self
    }

    /// Sets the geometry shader to use.
    #[inline]
    pub fn geometry_shader<Gs2, Gi2, Go2, Gl2>(self, shader: GeometryShaderEntryPoint<'a, Gs2, Gi2, Go2, Gl2>)
        -> GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco,
                                   Tcl, Tes, Tei, Teo, Tel, Gs2, Gi2, Go2, Gl2, Fs, Fi, Fo, Fl, Rp>
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: Some(shader),
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the geometry shader stage as disabled. This is the default.
    #[inline]
    pub fn geometry_shader_disabled(mut self) -> Self {
        self.geometry_shader = None;
        self
    }

    /// Sets the viewports to some value, and the scissor boxes to boxes that always cover the
    /// whole viewport.
    #[inline]
    pub fn viewports<I>(self, viewports: I) -> Self
        where I: IntoIterator<Item = Viewport>
    {
        self.viewports_scissors(viewports.into_iter().map(|v| (v, Scissor::irrelevant())))
    }

    /// Sets the characteristics of viewports and scissor boxes in advance.
    #[inline]
    pub fn viewports_scissors<I>(mut self, viewports: I) -> Self
        where I: IntoIterator<Item = (Viewport, Scissor)>
    {
        self.viewport = Some(ViewportsState::Fixed { data: viewports.into_iter().collect() });
        self
    }

    /// Sets the scissor boxes to some values, and viewports to dynamic. The viewports will
    /// need to be set before drawing.
    #[inline]
    pub fn viewports_dynamic_scissors_fixed<I>(mut self, scissors: I) -> Self
        where I: IntoIterator<Item = Scissor>
    {
        self.viewport = Some(ViewportsState::DynamicViewports {
            scissors: scissors.into_iter().collect()
        });
        self
    }

    /// Sets the viewports to dynamic, and the scissor boxes to boxes that always cover the whole
    /// viewport. The viewports will need to be set before drawing.
    #[inline]
    pub fn viewports_dynamic_scissors_irrelevant(mut self, num: u32) -> Self {
        self.viewport = Some(ViewportsState::DynamicViewports {
            scissors: (0 .. num).map(|_| Scissor::irrelevant()).collect()
        });
        self
    }

    /// Sets the viewports to some values, and scissor boxes to dynamic. The scissor boxes will
    /// need to be set before drawing.
    #[inline]
    pub fn viewports_fixed_scissors_dynamic<I>(mut self, viewports: I) -> Self
        where I: IntoIterator<Item = Viewport>
    {
        self.viewport = Some(ViewportsState::DynamicScissors {
            viewports: viewports.into_iter().collect()
        });
        self
    }

    /// Sets the viewports and scissor boxes to dynamic. They will both need to be set before
    /// drawing.
    #[inline]
    pub fn viewports_scissors_dynamic(mut self, num: u32) -> Self {
        self.viewport = Some(ViewportsState::Dynamic {
            num: num
        });
        self
    }

    /// If true, then the depth value of the vertices will be clamped to the range `[0.0 ; 1.0]`.
    /// If false, fragments whose depth is outside of this range will be discarded before the
    /// fragment shader even runs.
    #[inline]
    pub fn depth_clamp(mut self, clamp: bool) -> Self {
        self.raster.depth_clamp = clamp;
        self
    }

    // TODO: this won't work correctly
    /*/// Disables the fragment shader stage.
    #[inline]
    pub fn rasterizer_discard(mut self) -> Self {
        self.rasterization.rasterizer_discard. = true;
        self
    }*/

    /// Sets the front-facing faces to couner-clockwise faces. This is the default.
    ///
    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[inline]
    pub fn front_face_counter_clockwise(mut self) -> Self {
        self.raster.front_face = FrontFace::CounterClockwise;
        self
    }

    /// Sets the front-facing faces to clockwise faces.
    ///
    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[inline]
    pub fn front_face_clockwise(mut self) -> Self {
        self.raster.front_face = FrontFace::Clockwise;
        self
    }

    /// Sets backface culling as disabled. This is the default.
    #[inline]
    pub fn cull_mode_disabled(mut self) -> Self {
        self.raster.cull_mode = CullMode::None;
        self
    }

    /// Sets backface culling to front faces. The front faces (as chosen with the `front_face_*`
    /// methods) will be discarded by the GPU when drawing.
    #[inline]
    pub fn cull_mode_front(mut self) -> Self {
        self.raster.cull_mode = CullMode::Front;
        self
    }

    /// Sets backface culling to back faces. Faces that are not facing the front (as chosen with
    /// the `front_face_*` methods) will be discarded by the GPU when drawing.
    #[inline]
    pub fn cull_mode_back(mut self) -> Self {
        self.raster.cull_mode = CullMode::Back;
        self
    }

    /// Sets backface culling to both front and back faces. All the faces will be discarded.
    ///
    /// > **Note**: This option exists for the sake of completeness. It has no known practical
    /// > usage.
    #[inline]
    pub fn cull_mode_front_and_back(mut self) -> Self {
        self.raster.cull_mode = CullMode::FrontAndBack;
        self
    }

    /// Sets the polygon mode to "fill". This is the default.
    #[inline]
    pub fn polygon_mode_fill(mut self) -> Self {
        self.raster.polygon_mode = PolygonMode::Fill;
        self
    }

    /// Sets the polygon mode to "line". Triangles will each be turned into three lines.
    #[inline]
    pub fn polygon_mode_line(mut self) -> Self {
        self.raster.polygon_mode = PolygonMode::Line;
        self
    }

    /// Sets the polygon mode to "point". Triangles and lines will each be turned into three points.
    #[inline]
    pub fn polygon_mode_point(mut self) -> Self {
        self.raster.polygon_mode = PolygonMode::Point;
        self
    }

    /// Sets the width of the lines, if the GPU needs to draw lines. The default is `1.0`.
    #[inline]
    pub fn line_width(mut self, value: f32) -> Self {
        self.raster.line_width = Some(value);
        self
    }

    /// Sets the width of the lines as dynamic, which means that you will need to set this value
    /// when drawing.
    #[inline]
    pub fn line_width_dynamic(mut self) -> Self {
        self.raster.line_width = None;
        self
    }

    // TODO: missing DepthBiasControl

    // TODO: missing Multisample

    /// Sets the fragment shader to use.
    ///
    /// The fragment shader is run once for each pixel that is covered by each primitive.
    #[inline]
    pub fn fragment_shader<Fs2, Fi2, Fo2, Fl2>(self, shader: FragmentShaderEntryPoint<'a, Fs2, Fi2, Fo2, Fl2>)
        -> GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco,
                                   Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs2, Fi2, Fo2, Fl2, Rp>
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: Some(shader),
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the depth/stencil tests as disabled.
    ///
    /// > **Note**: This is a shortcut for all the other `depth_*` and `depth_stencil_*` methods
    /// > of the builder.
    #[inline]
    pub fn depth_stencil_disabled(mut self) -> Self {
        self.depth_stencil = DepthStencil::disabled();
        self
    }

    /// Sets the depth/stencil tests as a simple depth test and no stencil test.
    ///
    /// > **Note**: This is a shortcut for setting the depth test to `Less`, the depth write Into
    /// > ` true` and disable the stencil test.
    #[inline]
    pub fn depth_stencil_simple_depth(mut self) -> Self {
        self.depth_stencil = DepthStencil::simple_depth_test();
        self
    }

    /// Sets whether the depth buffer will be written.
    #[inline]
    pub fn depth_write(mut self, write: bool) -> Self {
        self.depth_stencil.depth_write = write;
        self
    }

    // TODO: missing tons of depth-stencil stuff


    #[inline]
    pub fn blend_collective(mut self, blend: AttachmentBlend) -> Self {
        self.blend.attachments = AttachmentsBlend::Collective(blend);
        self
    }

    #[inline]
    pub fn blend_individual<I>(mut self, blend: I) -> Self
        where I: IntoIterator<Item = AttachmentBlend>
    {
        self.blend.attachments = AttachmentsBlend::Individual(blend.into_iter().collect());
        self
    }

    /// Each fragment shader output will have its value directly written to the framebuffer
    /// attachment. This is the default.
    #[inline]
    pub fn blend_pass_through(self) -> Self {
        self.blend_collective(AttachmentBlend::pass_through())
    }

    #[inline]
    pub fn blend_alpha_blending(self) -> Self {
        self.blend_collective(AttachmentBlend::alpha_blending())
    }

    #[inline]
    pub fn blend_logic_op(mut self, logic_op: LogicOp) -> Self {
        self.blend.logic_op = Some(logic_op);
        self
    }

    /// Sets the logic operation as disabled. This is the default.
    #[inline]
    pub fn blend_logic_op_disabled(mut self) -> Self {
        self.blend.logic_op = None;
        self
    }

    /// Sets the blend constant. The default is `[0.0, 0.0, 0.0, 0.0]`.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[inline]
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend.blend_constants = Some(constants);
        self
    }

    /// Sets the blend constant value as dynamic. Its value will need to be set before drawing.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[inline]
    pub fn blend_constants_dynamic(mut self) -> Self {
        self.blend.blend_constants = None;
        self
    }

    /// Sets the render pass subpass to use.
    #[inline]
    pub fn render_pass<Rp2>(self, subpass: Subpass<Rp2>)
        -> GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco,
                                   Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp2>
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: Some(subpass),
        }
    }
}

// TODO:
/*impl<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi,
     Fo, Fl, Rp> Copy for
    GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                            Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    where Vdef: Copy, Rp: Copy
{
}

impl<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel, Gs, Gi, Go, Gl, Fs, Fi,
     Fo, Fl, Rp> Clone for
    GraphicsPipelineBuilder<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                            Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, Rp>
    where Vdef: Clone, Rp: Clone
{
    fn clone(&self) -> Self {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input.clone(),
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }
}*/
