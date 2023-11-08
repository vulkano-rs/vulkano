use crate::{
    command_buffer::{
        allocator::CommandBufferAllocator, sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    device::{DeviceOwned, QueueFlags},
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilFaces, StencilOp, StencilOps},
            input_assembly::PrimitiveTopology,
            rasterization::{CullMode, DepthBiasState, FrontFace, LineStipple},
            viewport::{Scissor, Viewport},
        },
        DynamicState,
    },
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::ops::RangeInclusive;

/// # Commands to set dynamic state for pipelines.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    // Helper function for dynamic state setting.
    fn validate_graphics_pipeline_fixed_state(
        &self,
        state: DynamicState,
    ) -> Result<(), Box<ValidationError>> {
        if self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .map_or(false, |pipeline| pipeline.fixed_state().contains(&state))
        {
            return Err(Box::new(ValidationError {
                problem: "the state for this value in the currently bound graphics pipeline \
                    is fixed, and cannot be set"
                    .into(),
                vuids: &["VUID-vkCmdDispatch-None-08608", "VUID-vkCmdDraw-None-08608"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    /// Sets the dynamic blend constants for future draw calls.
    pub fn set_blend_constants(
        &mut self,
        constants: [f32; 4],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_blend_constants(constants)?;

        unsafe { Ok(self.set_blend_constants_unchecked(constants)) }
    }

    fn validate_set_blend_constants(
        &self,
        constants: [f32; 4],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_blend_constants(constants)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::BlendConstants)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_blend_constants_unchecked(&mut self, constants: [f32; 4]) -> &mut Self {
        self.builder_state.blend_constants = Some(constants);
        self.add_command(
            "set_blend_constants",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_blend_constants_unchecked(constants);
            },
        );

        self
    }

    /// Sets whether dynamic color writes should be enabled for each attachment in the
    /// framebuffer.
    pub fn set_color_write_enable(
        &mut self,
        enables: SmallVec<[bool; 4]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_color_write_enable(&enables)?;

        unsafe { Ok(self.set_color_write_enable_unchecked(enables)) }
    }

    fn validate_set_color_write_enable(
        &self,
        enables: &[bool],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_color_write_enable(enables)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ColorWriteEnable)?;

        if let Some(color_blend_state) = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .and_then(|pipeline| pipeline.color_blend_state())
        {
            // Indirectly checked
            if enables.len() != color_blend_state.attachments.len() {
                return Err(Box::new(ValidationError {
                    problem: "the length of `enables` does not match the number of \
                        color attachments in the subpass of the currently bound graphics pipeline"
                        .into(),
                    vuids: &["VUID-vkCmdSetColorWriteEnableEXT-attachmentCount-06656"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_color_write_enable_unchecked(
        &mut self,
        enables: SmallVec<[bool; 4]>,
    ) -> &mut Self {
        self.builder_state.color_write_enable = Some(enables.clone());
        self.add_command(
            "set_color_write_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_color_write_enable_unchecked(&enables);
            },
        );

        self
    }

    /// Sets the dynamic cull mode for future draw calls.
    pub fn set_cull_mode(
        &mut self,
        cull_mode: CullMode,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_cull_mode(cull_mode)?;

        unsafe { Ok(self.set_cull_mode_unchecked(cull_mode)) }
    }

    fn validate_set_cull_mode(&self, cull_mode: CullMode) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_cull_mode(cull_mode)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::CullMode)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_cull_mode_unchecked(&mut self, cull_mode: CullMode) -> &mut Self {
        self.builder_state.cull_mode = Some(cull_mode);
        self.add_command(
            "set_cull_mode",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_cull_mode_unchecked(cull_mode);
            },
        );

        self
    }

    /// Sets the dynamic depth bias values for future draw calls.
    pub fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bias(constant_factor, clamp, slope_factor)?;

        unsafe { Ok(self.set_depth_bias_unchecked(constant_factor, clamp, slope_factor)) }
    }

    fn validate_set_depth_bias(
        &self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_depth_bias(constant_factor, clamp, slope_factor)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBias)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_unchecked(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        self.builder_state.depth_bias = Some(DepthBiasState {
            constant_factor,
            clamp,
            slope_factor,
        });
        self.add_command(
            "set_depth_bias",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bias_unchecked(constant_factor, clamp, slope_factor);
            },
        );

        self
    }

    /// Sets whether dynamic depth bias is enabled for future draw calls.
    pub fn set_depth_bias_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bias_enable(enable)?;

        unsafe { Ok(self.set_depth_bias_enable_unchecked(enable)) }
    }

    fn validate_set_depth_bias_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_bias_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBiasEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_bias_enable = Some(enable);
        self.add_command(
            "set_depth_bias_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bias_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets the dynamic depth bounds for future draw calls.
    pub fn set_depth_bounds(
        &mut self,
        bounds: RangeInclusive<f32>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bounds(bounds.clone())?;

        unsafe { Ok(self.set_depth_bounds_unchecked(bounds)) }
    }

    fn validate_set_depth_bounds(
        &self,
        bounds: RangeInclusive<f32>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_bounds(bounds)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBounds)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_unchecked(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        self.builder_state.depth_bounds = Some(bounds.clone());
        self.add_command(
            "set_depth_bounds",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bounds_unchecked(bounds.clone());
            },
        );

        self
    }

    /// Sets whether dynamic depth bounds testing is enabled for future draw calls.
    pub fn set_depth_bounds_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bounds_test_enable(enable)?;

        unsafe { Ok(self.set_depth_bounds_test_enable_unchecked(enable)) }
    }

    fn validate_set_depth_bounds_test_enable(
        &self,
        enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_bounds_test_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBoundsTestEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_bounds_test_enable = Some(enable);
        self.add_command(
            "set_depth_bounds_test_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bounds_test_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets the dynamic depth compare op for future draw calls.
    pub fn set_depth_compare_op(
        &mut self,
        compare_op: CompareOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_compare_op(compare_op)?;

        unsafe { Ok(self.set_depth_compare_op_unchecked(compare_op)) }
    }

    fn validate_set_depth_compare_op(
        &self,
        compare_op: CompareOp,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_compare_op(compare_op)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthCompareOp)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_compare_op_unchecked(&mut self, compare_op: CompareOp) -> &mut Self {
        self.builder_state.depth_compare_op = Some(compare_op);
        self.add_command(
            "set_depth_compare_op",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_compare_op_unchecked(compare_op);
            },
        );

        self
    }

    /// Sets whether dynamic depth testing is enabled for future draw calls.
    pub fn set_depth_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_test_enable(enable)?;

        unsafe { Ok(self.set_depth_test_enable_unchecked(enable)) }
    }

    fn validate_set_depth_test_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_test_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthTestEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_test_enable = Some(enable);
        self.add_command(
            "set_depth_test_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_test_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets whether dynamic depth write is enabled for future draw calls.
    pub fn set_depth_write_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_write_enable(enable)?;

        unsafe { Ok(self.set_depth_write_enable_unchecked(enable)) }
    }

    fn validate_set_depth_write_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_write_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthWriteEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_write_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_write_enable = Some(enable);
        self.add_command(
            "set_depth_write_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_write_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets the dynamic discard rectangles for future draw calls.
    pub fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: SmallVec<[Scissor; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_discard_rectangle(first_rectangle, &rectangles)?;

        unsafe { Ok(self.set_discard_rectangle_unchecked(first_rectangle, rectangles)) }
    }

    fn validate_set_discard_rectangle(
        &self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_discard_rectangle(first_rectangle, rectangles)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DiscardRectangle)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_discard_rectangle_unchecked(
        &mut self,
        first_rectangle: u32,
        rectangles: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        for (num, rectangle) in rectangles.iter().enumerate() {
            let num = num as u32 + first_rectangle;
            self.builder_state.discard_rectangle.insert(num, *rectangle);
        }

        self.add_command(
            "set_discard_rectangle",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_discard_rectangle_unchecked(first_rectangle, &rectangles);
            },
        );

        self
    }

    /// Sets the dynamic front face for future draw calls.
    pub fn set_front_face(&mut self, face: FrontFace) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_front_face(face)?;

        unsafe { Ok(self.set_front_face_unchecked(face)) }
    }

    fn validate_set_front_face(&self, face: FrontFace) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_front_face(face)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::FrontFace)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_front_face_unchecked(&mut self, face: FrontFace) -> &mut Self {
        self.builder_state.front_face = Some(face);
        self.add_command(
            "set_front_face",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_front_face_unchecked(face);
            },
        );

        self
    }

    /// Sets the dynamic line stipple values for future draw calls.
    pub fn set_line_stipple(
        &mut self,
        factor: u32,
        pattern: u16,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_line_stipple(factor, pattern)?;

        unsafe { Ok(self.set_line_stipple_unchecked(factor, pattern)) }
    }

    fn validate_set_line_stipple(
        &self,
        factor: u32,
        pattern: u16,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_line_stipple(factor, pattern)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::LineStipple)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_stipple_unchecked(&mut self, factor: u32, pattern: u16) -> &mut Self {
        self.builder_state.line_stipple = Some(LineStipple { factor, pattern });
        self.add_command(
            "set_line_stipple",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_line_stipple_unchecked(factor, pattern);
            },
        );

        self
    }

    /// Sets the dynamic line width for future draw calls.
    pub fn set_line_width(&mut self, line_width: f32) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_line_width(line_width)?;

        unsafe { Ok(self.set_line_width_unchecked(line_width)) }
    }

    fn validate_set_line_width(&self, line_width: f32) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_line_width(line_width)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::LineWidth)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_width_unchecked(&mut self, line_width: f32) -> &mut Self {
        self.builder_state.line_width = Some(line_width);
        self.add_command(
            "set_line_width",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_line_width_unchecked(line_width);
            },
        );

        self
    }

    /// Sets the dynamic logic op for future draw calls.
    pub fn set_logic_op(&mut self, logic_op: LogicOp) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_logic_op(logic_op)?;

        unsafe { Ok(self.set_logic_op_unchecked(logic_op)) }
    }

    fn validate_set_logic_op(&self, logic_op: LogicOp) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_logic_op(logic_op)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::LogicOp)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_logic_op_unchecked(&mut self, logic_op: LogicOp) -> &mut Self {
        self.builder_state.logic_op = Some(logic_op);
        self.add_command(
            "set_logic_op",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_logic_op_unchecked(logic_op);
            },
        );

        self
    }

    /// Sets the dynamic number of patch control points for future draw calls.
    pub fn set_patch_control_points(
        &mut self,
        num: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_patch_control_points(num)?;

        unsafe { Ok(self.set_patch_control_points_unchecked(num)) }
    }

    fn validate_set_patch_control_points(&self, num: u32) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_patch_control_points(num)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::PatchControlPoints)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_patch_control_points_unchecked(&mut self, num: u32) -> &mut Self {
        self.builder_state.patch_control_points = Some(num);
        self.add_command(
            "set_patch_control_points",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_patch_control_points_unchecked(num);
            },
        );

        self
    }

    /// Sets whether dynamic primitive restart is enabled for future draw calls.
    pub fn set_primitive_restart_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_primitive_restart_enable(enable)?;

        unsafe { Ok(self.set_primitive_restart_enable_unchecked(enable)) }
    }

    fn validate_set_primitive_restart_enable(
        &self,
        enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_primitive_restart_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::PrimitiveRestartEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_restart_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.primitive_restart_enable = Some(enable);
        self.add_command(
            "set_primitive_restart_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_primitive_restart_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets the dynamic primitive topology for future draw calls.
    pub fn set_primitive_topology(
        &mut self,
        topology: PrimitiveTopology,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_primitive_topology(topology)?;

        unsafe { Ok(self.set_primitive_topology_unchecked(topology)) }
    }

    fn validate_set_primitive_topology(
        &self,
        topology: PrimitiveTopology,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_primitive_topology(topology)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::PrimitiveTopology)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_topology_unchecked(
        &mut self,
        topology: PrimitiveTopology,
    ) -> &mut Self {
        self.builder_state.primitive_topology = Some(topology);
        self.add_command(
            "set_primitive_topology",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_primitive_topology_unchecked(topology);
            },
        );

        self
    }

    /// Sets whether dynamic rasterizer discard is enabled for future draw calls.
    pub fn set_rasterizer_discard_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_rasterizer_discard_enable(enable)?;

        unsafe { Ok(self.set_rasterizer_discard_enable_unchecked(enable)) }
    }

    fn validate_set_rasterizer_discard_enable(
        &self,
        enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_rasterizer_discard_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::RasterizerDiscardEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_rasterizer_discard_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.rasterizer_discard_enable = Some(enable);
        self.add_command(
            "set_rasterizer_discard_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_rasterizer_discard_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets the dynamic scissors for future draw calls.
    pub fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_scissor(first_scissor, &scissors)?;

        unsafe { Ok(self.set_scissor_unchecked(first_scissor, scissors)) }
    }

    fn validate_set_scissor(
        &self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_scissor(first_scissor, scissors)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::Scissor)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_unchecked(
        &mut self,
        first_scissor: u32,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        for (num, scissor) in scissors.iter().enumerate() {
            let num = num as u32 + first_scissor;
            self.builder_state.scissor.insert(num, *scissor);
        }

        self.add_command(
            "set_scissor",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_scissor_unchecked(first_scissor, &scissors);
            },
        );

        self
    }

    /// Sets the dynamic scissors with count for future draw calls.
    pub fn set_scissor_with_count(
        &mut self,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_scissor_with_count(&scissors)?;

        unsafe { Ok(self.set_scissor_with_count_unchecked(scissors)) }
    }

    fn validate_set_scissor_with_count(
        &self,
        scissors: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_scissor_with_count(scissors)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ScissorWithCount)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_with_count_unchecked(
        &mut self,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        self.builder_state.scissor_with_count = Some(scissors.clone());
        self.add_command(
            "set_scissor_with_count",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_scissor_with_count_unchecked(&scissors);
            },
        );

        self
    }

    /// Sets the dynamic stencil compare mask on one or both faces for future draw calls.
    pub fn set_stencil_compare_mask(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_compare_mask(faces, compare_mask)?;

        unsafe { Ok(self.set_stencil_compare_mask_unchecked(faces, compare_mask)) }
    }

    fn validate_set_stencil_compare_mask(
        &self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_compare_mask(faces, compare_mask)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilCompareMask)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_compare_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_compare_mask.front = Some(compare_mask);
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_compare_mask.back = Some(compare_mask);
        }

        self.add_command(
            "set_stencil_compare_mask",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_compare_mask_unchecked(faces, compare_mask);
            },
        );

        self
    }

    /// Sets the dynamic stencil ops on one or both faces for future draw calls.
    pub fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op)?;

        unsafe {
            Ok(self.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op))
        }
    }

    fn validate_set_stencil_op(
        &self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilOp)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_op_unchecked(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_op.front = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_op.back = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        self.add_command(
            "set_stencil_op",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op);
            },
        );

        self
    }

    /// Sets the dynamic stencil reference on one or both faces for future draw calls.
    pub fn set_stencil_reference(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_reference(faces, reference)?;

        unsafe { Ok(self.set_stencil_reference_unchecked(faces, reference)) }
    }

    fn validate_set_stencil_reference(
        &self,
        faces: StencilFaces,
        reference: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_reference(faces, reference)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilReference)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_reference_unchecked(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_reference.front = Some(reference);
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_reference.back = Some(reference);
        }

        self.add_command(
            "set_stencil_reference",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_reference_unchecked(faces, reference);
            },
        );

        self
    }

    /// Sets whether dynamic stencil testing is enabled for future draw calls.
    pub fn set_stencil_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_test_enable(enable)?;

        unsafe { Ok(self.set_stencil_test_enable_unchecked(enable)) }
    }

    fn validate_set_stencil_test_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_stencil_test_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilTestEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.stencil_test_enable = Some(enable);
        self.add_command(
            "set_stencil_test_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_test_enable_unchecked(enable);
            },
        );

        self
    }

    /// Sets the dynamic stencil write mask on one or both faces for future draw calls.
    pub fn set_stencil_write_mask(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_write_mask(faces, write_mask)?;

        unsafe { Ok(self.set_stencil_write_mask_unchecked(faces, write_mask)) }
    }

    fn validate_set_stencil_write_mask(
        &self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_write_mask(faces, write_mask)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilWriteMask)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_write_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_write_mask.front = Some(write_mask);
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_write_mask.back = Some(write_mask);
        }

        self.add_command(
            "set_stencil_write_mask",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_write_mask_unchecked(faces, write_mask);
            },
        );

        self
    }

    /// Sets the dynamic viewports for future draw calls.
    pub fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_viewport(first_viewport, &viewports)?;

        unsafe { Ok(self.set_viewport_unchecked(first_viewport, viewports)) }
    }

    fn validate_set_viewport(
        &self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_viewport(first_viewport, viewports)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::Viewport)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_unchecked(
        &mut self,
        first_viewport: u32,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        for (num, viewport) in viewports.iter().enumerate() {
            let num = num as u32 + first_viewport;
            self.builder_state.viewport.insert(num, viewport.clone());
        }

        self.add_command(
            "set_viewport",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_viewport_unchecked(first_viewport, &viewports);
            },
        );

        self
    }

    /// Sets the dynamic viewports with count for future draw calls.
    pub fn set_viewport_with_count(
        &mut self,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_viewport_with_count(&viewports)?;

        unsafe { Ok(self.set_viewport_with_count_unchecked(viewports)) }
    }

    fn validate_set_viewport_with_count(
        &self,
        viewports: &[Viewport],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_viewport_with_count(viewports)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ViewportWithCount)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_with_count_unchecked(
        &mut self,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        self.builder_state.viewport_with_count = Some(viewports.clone());
        self.add_command(
            "set_viewport",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_viewport_with_count_unchecked(&viewports);
            },
        );

        self
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    pub unsafe fn set_blend_constants(
        &mut self,
        constants: [f32; 4],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_blend_constants(constants)?;

        Ok(self.set_blend_constants_unchecked(constants))
    }

    fn validate_set_blend_constants(
        &self,
        _constants: [f32; 4],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetBlendConstants-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_blend_constants_unchecked(&mut self, constants: [f32; 4]) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_blend_constants)(self.handle(), &constants);

        self
    }

    pub unsafe fn set_color_write_enable(
        &mut self,
        enables: &[bool],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_color_write_enable(enables)?;

        Ok(self.set_color_write_enable_unchecked(enables))
    }

    fn validate_set_color_write_enable(
        &self,
        _enables: &[bool],
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().color_write_enable {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_color_write_enable",
                )])]),
                vuids: &["VUID-vkCmdSetColorWriteEnableEXT-None-04803"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetColorWriteEnableEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_color_write_enable_unchecked(&mut self, enables: &[bool]) -> &mut Self {
        let enables = enables
            .iter()
            .copied()
            .map(|v| v as ash::vk::Bool32)
            .collect::<SmallVec<[_; 4]>>();

        if enables.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.ext_color_write_enable.cmd_set_color_write_enable_ext)(
            self.handle(),
            enables.len() as u32,
            enables.as_ptr(),
        );

        self
    }

    pub unsafe fn set_cull_mode(
        &mut self,
        cull_mode: CullMode,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_cull_mode(cull_mode)?;

        Ok(self.set_cull_mode_unchecked(cull_mode))
    }

    fn validate_set_cull_mode(&self, cull_mode: CullMode) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetCullMode-None-03384"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetCullMode-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        cull_mode.validate_device(self.device()).map_err(|err| {
            err.add_context("cull_mode")
                .set_vuids(&["VUID-vkCmdSetCullMode-cullMode-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_cull_mode_unchecked(&mut self, cull_mode: CullMode) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_cull_mode)(self.handle(), cull_mode.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_cull_mode_ext)(self.handle(), cull_mode.into());
        }

        self
    }

    pub unsafe fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bias(constant_factor, clamp, slope_factor)?;

        Ok(self.set_depth_bias_unchecked(constant_factor, clamp, slope_factor))
    }

    fn validate_set_depth_bias(
        &self,
        _constant_factor: f32,
        clamp: f32,
        _slope_factor: f32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthBias-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if clamp != 0.0 && !self.device().enabled_features().depth_bias_clamp {
            return Err(Box::new(ValidationError {
                context: "clamp".into(),
                problem: "is not `0.0`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "depth_bias_clamp",
                )])]),
                vuids: &["VUID-vkCmdSetDepthBias-depthBiasClamp-00790"],
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_unchecked(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_depth_bias)(self.handle(), constant_factor, clamp, slope_factor);

        self
    }

    pub unsafe fn set_depth_bias_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bias_enable(enable)?;

        Ok(self.set_depth_bias_enable_unchecked(enable))
    }

    fn validate_set_depth_bias_enable(&self, _enable: bool) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                ]),
                vuids: &["VUID-vkCmdSetDepthBiasEnable-None-04872"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthBiasEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_bias_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state2
                .cmd_set_depth_bias_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    pub unsafe fn set_depth_bounds(
        &mut self,
        bounds: RangeInclusive<f32>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bounds(bounds.clone())?;

        Ok(self.set_depth_bounds_unchecked(bounds))
    }

    fn validate_set_depth_bounds(
        &self,
        bounds: RangeInclusive<f32>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthBounds-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !self
            .device()
            .enabled_extensions()
            .ext_depth_range_unrestricted
        {
            if !(0.0..=1.0).contains(bounds.start()) {
                return Err(Box::new(ValidationError {
                    context: "bounds.start()".into(),
                    problem: "is not between `0.0` and `1.0` inclusive".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-vkCmdSetDepthBounds-minDepthBounds-00600"],
                }));
            }

            if !(0.0..=1.0).contains(bounds.end()) {
                return Err(Box::new(ValidationError {
                    context: "bounds.end()".into(),
                    problem: "is not between `0.0` and `1.0` inclusive".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-vkCmdSetDepthBounds-maxDepthBounds-00601"],
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_unchecked(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_depth_bounds)(self.handle(), *bounds.start(), *bounds.end());

        self
    }

    pub unsafe fn set_depth_bounds_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bounds_test_enable(enable)?;

        Ok(self.set_depth_bounds_test_enable_unchecked(enable))
    }

    fn validate_set_depth_bounds_test_enable(
        &self,
        _enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetDepthBoundsTestEnable-None-03349"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthBoundsTestEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_bounds_test_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_depth_bounds_test_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    pub unsafe fn set_depth_compare_op(
        &mut self,
        compare_op: CompareOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_compare_op(compare_op)?;

        Ok(self.set_depth_compare_op_unchecked(compare_op))
    }

    fn validate_set_depth_compare_op(
        &self,
        compare_op: CompareOp,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetDepthCompareOp-None-03353"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthCompareOp-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        compare_op.validate_device(self.device()).map_err(|err| {
            err.add_context("compare_op")
                .set_vuids(&["VUID-vkCmdSetDepthCompareOp-depthCompareOp-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_compare_op_unchecked(&mut self, compare_op: CompareOp) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_compare_op)(self.handle(), compare_op.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_depth_compare_op_ext)(
                self.handle(),
                compare_op.into(),
            );
        }

        self
    }

    pub unsafe fn set_depth_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_test_enable(enable)?;

        Ok(self.set_depth_test_enable_unchecked(enable))
    }

    fn validate_set_depth_test_enable(&self, _enable: bool) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetDepthTestEnable-None-03352"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthTestEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_test_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_depth_test_enable_ext)(
                self.handle(),
                enable.into(),
            );
        }

        self
    }

    pub unsafe fn set_depth_write_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_write_enable(enable)?;

        Ok(self.set_depth_write_enable_unchecked(enable))
    }

    fn validate_set_depth_write_enable(&self, _enable: bool) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetDepthWriteEnable-None-03354"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDepthWriteEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_write_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_write_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_depth_write_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    pub unsafe fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_discard_rectangle(first_rectangle, rectangles)?;

        Ok(self.set_discard_rectangle_unchecked(first_rectangle, rectangles))
    }

    fn validate_set_discard_rectangle(
        &self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        if self.device().enabled_extensions().ext_discard_rectangles {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_discard_rectangles",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetDiscardRectangle-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if first_rectangle + rectangles.len() as u32 > properties.max_discard_rectangles.unwrap() {
            return Err(Box::new(ValidationError {
                problem: "`first_rectangle + rectangles.len()` exceeds the \
                    `max_discard_rectangles` limit"
                    .into(),
                vuids: &["VUID-vkCmdSetDiscardRectangleEXT-firstDiscardRectangle-00585"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_discard_rectangle_unchecked(
        &mut self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> &mut Self {
        let rectangles = rectangles
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if rectangles.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.ext_discard_rectangles.cmd_set_discard_rectangle_ext)(
            self.handle(),
            first_rectangle,
            rectangles.len() as u32,
            rectangles.as_ptr(),
        );

        self
    }

    pub unsafe fn set_front_face(
        &mut self,
        face: FrontFace,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_front_face(face)?;

        Ok(self.set_front_face_unchecked(face))
    }

    fn validate_set_front_face(&self, face: FrontFace) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetFrontFace-None-03383"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetFrontFace-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        face.validate_device(self.device()).map_err(|err| {
            err.add_context("face")
                .set_vuids(&["VUID-vkCmdSetFrontFace-frontFace-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_front_face_unchecked(&mut self, face: FrontFace) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_front_face)(self.handle(), face.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_front_face_ext)(self.handle(), face.into());
        }

        self
    }

    pub unsafe fn set_line_stipple(
        &mut self,
        factor: u32,
        pattern: u16,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_line_stipple(factor, pattern)?;

        Ok(self.set_line_stipple_unchecked(factor, pattern))
    }

    fn validate_set_line_stipple(
        &self,
        factor: u32,
        _pattern: u16,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_extensions().ext_line_rasterization {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_line_rasterization",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetLineStippleEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !(1..=256).contains(&factor) {
            return Err(Box::new(ValidationError {
                context: "factor".into(),
                problem: "is not between 1 and 256 inclusive".into(),
                vuids: &["VUID-vkCmdSetLineStippleEXT-lineStippleFactor-02776"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_stipple_unchecked(&mut self, factor: u32, pattern: u16) -> &mut Self {
        let fns = self.device().fns();
        (fns.ext_line_rasterization.cmd_set_line_stipple_ext)(self.handle(), factor, pattern);

        self
    }

    pub unsafe fn set_line_width(
        &mut self,
        line_width: f32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_line_width(line_width)?;

        Ok(self.set_line_width_unchecked(line_width))
    }

    fn validate_set_line_width(&self, line_width: f32) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetLineWidth-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if line_width != 1.0 && !self.device().enabled_features().wide_lines {
            return Err(Box::new(ValidationError {
                context: "line_width".into(),
                problem: "is not 1.0".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "wide_lines",
                )])]),
                vuids: &["VUID-vkCmdSetLineWidth-lineWidth-00788"],
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_width_unchecked(&mut self, line_width: f32) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_line_width)(self.handle(), line_width);

        self
    }

    pub unsafe fn set_logic_op(
        &mut self,
        logic_op: LogicOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_logic_op(logic_op)?;

        Ok(self.set_logic_op_unchecked(logic_op))
    }

    fn validate_set_logic_op(&self, logic_op: LogicOp) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .extended_dynamic_state2_logic_op
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "extended_dynamic_state2_logic_op",
                )])]),
                vuids: &["VUID-vkCmdSetLogicOpEXT-None-04867"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetLogicOpEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        logic_op.validate_device(self.device()).map_err(|err| {
            err.add_context("logic_op")
                .set_vuids(&["VUID-vkCmdSetLogicOpEXT-logicOp-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_logic_op_unchecked(&mut self, logic_op: LogicOp) -> &mut Self {
        let fns = self.device().fns();
        (fns.ext_extended_dynamic_state2.cmd_set_logic_op_ext)(self.handle(), logic_op.into());

        self
    }

    pub unsafe fn set_patch_control_points(
        &mut self,
        num: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_patch_control_points(num)?;

        Ok(self.set_patch_control_points_unchecked(num))
    }

    fn validate_set_patch_control_points(&self, num: u32) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .extended_dynamic_state2_patch_control_points
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "extended_dynamic_state2_patch_control_points",
                )])]),
                vuids: &["VUID-vkCmdSetPatchControlPointsEXT-None-04873"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetPatchControlPointsEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if num == 0 {
            return Err(Box::new(ValidationError {
                context: "num".into(),
                problem: "is zero".into(),
                vuids: &["VUID-vkCmdSetPatchControlPointsEXT-patchControlPoints-04874"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if num > properties.max_tessellation_patch_size {
            return Err(Box::new(ValidationError {
                context: "num".into(),
                problem: "exceeds the `max_tessellation_patch_size` limit".into(),
                vuids: &["VUID-vkCmdSetPatchControlPointsEXT-patchControlPoints-04874"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_patch_control_points_unchecked(&mut self, num: u32) -> &mut Self {
        let fns = self.device().fns();
        (fns.ext_extended_dynamic_state2
            .cmd_set_patch_control_points_ext)(self.handle(), num);

        self
    }

    pub unsafe fn set_primitive_restart_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_primitive_restart_enable(enable)?;

        Ok(self.set_primitive_restart_enable_unchecked(enable))
    }

    fn validate_set_primitive_restart_enable(
        &self,
        _enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                ]),
                vuids: &["VUID-vkCmdSetPrimitiveRestartEnable-None-04866"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetPrimitiveRestartEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_restart_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_primitive_restart_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state2
                .cmd_set_primitive_restart_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    pub unsafe fn set_primitive_topology(
        &mut self,
        topology: PrimitiveTopology,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_primitive_topology(topology)?;

        Ok(self.set_primitive_topology_unchecked(topology))
    }

    fn validate_set_primitive_topology(
        &self,
        topology: PrimitiveTopology,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetPrimitiveTopology-None-03347"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetPrimitiveTopology-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        topology.validate_device(self.device()).map_err(|err| {
            err.add_context("topology")
                .set_vuids(&["VUID-vkCmdSetPrimitiveTopology-primitiveTopology-parameter"])
        })?;

        // VUID?
        // Since these requirements exist for fixed state when creating the pipeline,
        // I assume they exist for dynamic state as well.
        match topology {
            PrimitiveTopology::TriangleFan => {
                if self.device().enabled_extensions().khr_portability_subset
                    && !self.device().enabled_features().triangle_fans
                {
                    return Err(Box::new(ValidationError {
                        problem: "this device is a portability subset device, and `topology` \
                            is `PrimitiveTopology::TriangleFan`"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "triangle_fans",
                        )])]),
                        ..Default::default()
                    }));
                }
            }
            PrimitiveTopology::LineListWithAdjacency
            | PrimitiveTopology::LineStripWithAdjacency
            | PrimitiveTopology::TriangleListWithAdjacency
            | PrimitiveTopology::TriangleStripWithAdjacency => {
                if !self.device().enabled_features().geometry_shader {
                    return Err(Box::new(ValidationError {
                        problem: "`topology` is `PrimitiveTopology::*WithAdjacency`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "geometry_shader",
                        )])]),
                        ..Default::default()
                    }));
                }
            }
            PrimitiveTopology::PatchList => {
                if !self.device().enabled_features().tessellation_shader {
                    return Err(Box::new(ValidationError {
                        problem: "`topology` is `PrimitiveTopology::PatchList`".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "tessellation_shader",
                        )])]),
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_topology_unchecked(
        &mut self,
        topology: PrimitiveTopology,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_primitive_topology)(self.handle(), topology.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_primitive_topology_ext)(self.handle(), topology.into());
        }

        self
    }

    pub unsafe fn set_rasterizer_discard_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_rasterizer_discard_enable(enable)?;

        Ok(self.set_rasterizer_discard_enable_unchecked(enable))
    }

    fn validate_set_rasterizer_discard_enable(
        &self,
        _enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                ]),
                vuids: &["VUID-vkCmdSetRasterizerDiscardEnable-None-04871"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetRasterizerDiscardEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_rasterizer_discard_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_rasterizer_discard_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state2
                .cmd_set_rasterizer_discard_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    pub unsafe fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_scissor(first_scissor, scissors)?;

        Ok(self.set_scissor_unchecked(first_scissor, scissors))
    }

    fn validate_set_scissor(
        &self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetScissor-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if first_scissor + scissors.len() as u32 > properties.max_viewports {
            return Err(Box::new(ValidationError {
                problem: "`first_scissor + scissors.len()` exceeds the `max_viewports` limit"
                    .into(),
                vuids: &["VUID-vkCmdSetScissor-firstScissor-00592"],
                ..Default::default()
            }));
        }

        if !self.device().enabled_features().multi_viewport {
            if first_scissor != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`first_scissor` is not 0".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                    vuids: &["VUID-vkCmdSetScissor-firstScissor-00593"],
                    ..Default::default()
                }));
            }

            if scissors.len() > 1 {
                return Err(Box::new(ValidationError {
                    problem: "`scissors.len()` is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                    vuids: &["VUID-vkCmdSetScissor-scissorCount-00594"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_unchecked(
        &mut self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> &mut Self {
        let scissors = scissors
            .iter()
            .map(ash::vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.v1_0.cmd_set_scissor)(
            self.handle(),
            first_scissor,
            scissors.len() as u32,
            scissors.as_ptr(),
        );

        self
    }

    pub unsafe fn set_scissor_with_count(
        &mut self,
        scissors: &[Scissor],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_scissor_with_count(scissors)?;

        Ok(self.set_scissor_with_count_unchecked(scissors))
    }

    fn validate_set_scissor_with_count(
        &self,
        scissors: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetScissorWithCount-None-03396"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetScissorWithCount-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if scissors.len() as u32 > properties.max_viewports {
            return Err(Box::new(ValidationError {
                problem: "`scissors.len()` exceeds the `max_viewports` limit".into(),
                vuids: &["VUID-vkCmdSetScissorWithCount-scissorCount-03397"],
                ..Default::default()
            }));
        }

        if !self.device().enabled_features().multi_viewport && scissors.len() > 1 {
            return Err(Box::new(ValidationError {
                problem: "`scissors.len()` is greater than 1".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "multi_viewport",
                )])]),
                vuids: &["VUID-vkCmdSetScissorWithCount-scissorCount-03398"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_with_count_unchecked(&mut self, scissors: &[Scissor]) -> &mut Self {
        let scissors = scissors
            .iter()
            .map(ash::vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return self;
        }

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_scissor_with_count)(
                self.handle(),
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_scissor_with_count_ext)(
                self.handle(),
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        }

        self
    }

    pub unsafe fn set_stencil_compare_mask(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_compare_mask(faces, compare_mask)?;

        Ok(self.set_stencil_compare_mask_unchecked(faces, compare_mask))
    }

    fn validate_set_stencil_compare_mask(
        &self,
        faces: StencilFaces,
        _compare_mask: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetStencilCompareMask-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        faces.validate_device(self.device()).map_err(|err| {
            err.add_context("faces")
                .set_vuids(&["VUID-vkCmdSetStencilCompareMask-faceMask-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_compare_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_stencil_compare_mask)(self.handle(), faces.into(), compare_mask);

        self
    }

    pub unsafe fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op)?;

        Ok(self.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op))
    }

    fn validate_set_stencil_op(
        &self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetStencilOp-None-03351"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetStencilOp-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        faces.validate_device(self.device()).map_err(|err| {
            err.add_context("faces")
                .set_vuids(&["VUID-vkCmdSetStencilOp-faceMask-parameter"])
        })?;

        fail_op.validate_device(self.device()).map_err(|err| {
            err.add_context("fail_op")
                .set_vuids(&["VUID-vkCmdSetStencilOp-failOp-parameter"])
        })?;

        pass_op.validate_device(self.device()).map_err(|err| {
            err.add_context("pass_op")
                .set_vuids(&["VUID-vkCmdSetStencilOp-passOp-parameter"])
        })?;

        depth_fail_op
            .validate_device(self.device())
            .map_err(|err| {
                err.add_context("depth_fail_op")
                    .set_vuids(&["VUID-vkCmdSetStencilOp-depthFailOp-parameter"])
            })?;

        compare_op.validate_device(self.device()).map_err(|err| {
            err.add_context("compare_op")
                .set_vuids(&["VUID-vkCmdSetStencilOp-compareOp-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_op_unchecked(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_stencil_op)(
                self.handle(),
                faces.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_stencil_op_ext)(
                self.handle(),
                faces.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        }

        self
    }

    pub unsafe fn set_stencil_reference(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_reference(faces, reference)?;

        Ok(self.set_stencil_reference_unchecked(faces, reference))
    }

    fn validate_set_stencil_reference(
        &self,
        faces: StencilFaces,
        _reference: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetStencilReference-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        faces.validate_device(self.device()).map_err(|err| {
            err.add_context("faces")
                .set_vuids(&["VUID-vkCmdSetStencilReference-faceMask-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_reference_unchecked(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_stencil_reference)(self.handle(), faces.into(), reference);

        self
    }

    pub unsafe fn set_stencil_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_test_enable(enable)?;

        Ok(self.set_stencil_test_enable_unchecked(enable))
    }

    fn validate_set_stencil_test_enable(&self, _enable: bool) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetStencilTestEnable-None-03350"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetStencilTestEnable-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_stencil_test_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_stencil_test_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    pub unsafe fn set_stencil_write_mask(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_write_mask(faces, write_mask)?;

        Ok(self.set_stencil_write_mask_unchecked(faces, write_mask))
    }

    fn validate_set_stencil_write_mask(
        &self,
        faces: StencilFaces,
        _write_mask: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetStencilWriteMask-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        faces.validate_device(self.device()).map_err(|err| {
            err.add_context("faces")
                .set_vuids(&["VUID-vkCmdSetStencilWriteMask-faceMask-parameter"])
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_write_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_stencil_write_mask)(self.handle(), faces.into(), write_mask);

        self
    }

    pub unsafe fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_viewport(first_viewport, viewports)?;

        Ok(self.set_viewport_unchecked(first_viewport, viewports))
    }

    fn validate_set_viewport(
        &self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetViewport-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if first_viewport + viewports.len() as u32 > properties.max_viewports {
            return Err(Box::new(ValidationError {
                problem: "`first_viewport + viewports.len()` exceeds the `max_viewports` limit"
                    .into(),
                vuids: &["VUID-vkCmdSetViewport-firstViewport-01223"],
                ..Default::default()
            }));
        }

        if !self.device().enabled_features().multi_viewport {
            if first_viewport != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`first_viewport` is not 0".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                    vuids: &["VUID-vkCmdSetViewport-firstViewport-01224"],
                    ..Default::default()
                }));
            }

            if viewports.len() > 1 {
                return Err(Box::new(ValidationError {
                    problem: "`viewports.len()` is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                    vuids: &["VUID-vkCmdSetViewport-viewportCount-01225"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_unchecked(
        &mut self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> &mut Self {
        let viewports = viewports
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.v1_0.cmd_set_viewport)(
            self.handle(),
            first_viewport,
            viewports.len() as u32,
            viewports.as_ptr(),
        );

        self
    }

    pub unsafe fn set_viewport_with_count(
        &mut self,
        viewports: &[Viewport],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_viewport_with_count(viewports)?;

        Ok(self.set_viewport_with_count_unchecked(viewports))
    }

    fn validate_set_viewport_with_count(
        &self,
        viewports: &[Viewport],
    ) -> Result<(), Box<ValidationError>> {
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
                vuids: &["VUID-vkCmdSetViewportWithCount-None-03393"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdSetViewportWithCount-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if viewports.len() as u32 > properties.max_viewports {
            return Err(Box::new(ValidationError {
                problem: "`viewports.len()` exceeds the `max_viewports` limit".into(),
                vuids: &["VUID-vkCmdSetViewportWithCount-viewportCount-03394"],
                ..Default::default()
            }));
        }

        if viewports.len() > 1 && !self.device().enabled_features().multi_viewport {
            return Err(Box::new(ValidationError {
                problem: "`viewports.len()` is greater than 1".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "multi_viewport",
                )])]),
                vuids: &["VUID-vkCmdSetViewportWithCount-viewportCount-03395"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_with_count_unchecked(
        &mut self,
        viewports: &[Viewport],
    ) -> &mut Self {
        let viewports = viewports
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return self;
        }

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_viewport_with_count)(
                self.handle(),
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_viewport_with_count_ext)(
                self.handle(),
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        }

        self
    }
}
