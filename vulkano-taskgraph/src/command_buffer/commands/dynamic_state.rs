use crate::command_buffer::{RecordingCommandBuffer, Result};
use ash::vk;
use smallvec::SmallVec;
use std::ops::RangeInclusive;
use vulkano::{
    device::DeviceOwned,
    pipeline::graphics::{
        color_blend::LogicOp,
        conservative_rasterization::ConservativeRasterizationMode,
        depth_stencil::{CompareOp, StencilFaces, StencilOp},
        input_assembly::PrimitiveTopology,
        rasterization::{CullMode, FrontFace},
        vertex_input::{
            VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
            VertexInputState,
        },
        viewport::{Scissor, Viewport},
    },
    Version, VulkanObject,
};

/// # Commands to set dynamic state for pipelines
///
/// These commands require a queue with a pipeline type that uses the given state.
impl RecordingCommandBuffer<'_> {
    /// Sets the dynamic blend constants for future draw calls.
    pub unsafe fn set_blend_constants(&mut self, constants: &[f32; 4]) -> Result<&mut Self> {
        Ok(unsafe { self.set_blend_constants_unchecked(constants) })
    }

    pub unsafe fn set_blend_constants_unchecked(&mut self, constants: &[f32; 4]) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_set_blend_constants)(self.handle(), constants) };

        self
    }

    /// Sets whether dynamic color writes should be enabled for each attachment in the framebuffer.
    pub unsafe fn set_color_write_enable(&mut self, enables: &[bool]) -> Result<&mut Self> {
        Ok(unsafe { self.set_color_write_enable_unchecked(enables) })
    }

    pub unsafe fn set_color_write_enable_unchecked(&mut self, enables: &[bool]) -> &mut Self {
        if enables.is_empty() {
            return self;
        }

        let enables_vk = enables
            .iter()
            .copied()
            .map(|v| v as vk::Bool32)
            .collect::<SmallVec<[_; 4]>>();

        let fns = self.device().fns();
        unsafe {
            (fns.ext_color_write_enable.cmd_set_color_write_enable_ext)(
                self.handle(),
                enables_vk.len() as u32,
                enables_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic cull mode for future draw calls.
    pub unsafe fn set_cull_mode(&mut self, cull_mode: CullMode) -> Result<&mut Self> {
        Ok(unsafe { self.set_cull_mode_unchecked(cull_mode) })
    }

    pub unsafe fn set_cull_mode_unchecked(&mut self, cull_mode: CullMode) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_cull_mode = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_cull_mode
        } else {
            fns.ext_extended_dynamic_state.cmd_set_cull_mode_ext
        };

        unsafe { cmd_set_cull_mode(self.handle(), cull_mode.into()) };

        self
    }

    /// Sets the dynamic depth bias values for future draw calls.
    pub unsafe fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_bias_unchecked(constant_factor, clamp, slope_factor) })
    }

    pub unsafe fn set_depth_bias_unchecked(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_set_depth_bias)(self.handle(), constant_factor, clamp, slope_factor)
        };

        self
    }

    /// Sets whether dynamic depth bias is enabled for future draw calls.
    pub unsafe fn set_depth_bias_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_bias_enable_unchecked(enable) })
    }

    pub unsafe fn set_depth_bias_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_depth_bias_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_depth_bias_enable
        } else {
            fns.ext_extended_dynamic_state2
                .cmd_set_depth_bias_enable_ext
        };

        unsafe { cmd_set_depth_bias_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets the dynamic depth bounds for future draw calls.
    pub unsafe fn set_depth_bounds(&mut self, bounds: RangeInclusive<f32>) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_bounds_unchecked(bounds.clone()) })
    }

    pub unsafe fn set_depth_bounds_unchecked(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_set_depth_bounds)(self.handle(), *bounds.start(), *bounds.end()) };

        self
    }

    /// Sets whether dynamic depth bounds testing is enabled for future draw calls.
    pub unsafe fn set_depth_bounds_test_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_bounds_test_enable_unchecked(enable) })
    }

    pub unsafe fn set_depth_bounds_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_depth_bounds_test_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_depth_bounds_test_enable
        } else {
            fns.ext_extended_dynamic_state
                .cmd_set_depth_bounds_test_enable_ext
        };

        unsafe { cmd_set_depth_bounds_test_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets the dynamic depth compare op for future draw calls.
    pub unsafe fn set_depth_compare_op(&mut self, compare_op: CompareOp) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_compare_op_unchecked(compare_op) })
    }

    pub unsafe fn set_depth_compare_op_unchecked(&mut self, compare_op: CompareOp) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_depth_compare_op = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_depth_compare_op
        } else {
            fns.ext_extended_dynamic_state.cmd_set_depth_compare_op_ext
        };

        unsafe { cmd_set_depth_compare_op(self.handle(), compare_op.into()) };

        self
    }

    /// Sets whether dynamic depth testing is enabled for future draw calls.
    pub unsafe fn set_depth_test_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_test_enable_unchecked(enable) })
    }

    pub unsafe fn set_depth_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_depth_test_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_depth_test_enable
        } else {
            fns.ext_extended_dynamic_state.cmd_set_depth_test_enable_ext
        };

        unsafe { cmd_set_depth_test_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets whether dynamic depth write is enabled for future draw calls.
    pub unsafe fn set_depth_write_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_depth_write_enable_unchecked(enable) })
    }

    pub unsafe fn set_depth_write_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_depth_write_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_depth_write_enable
        } else {
            fns.ext_extended_dynamic_state
                .cmd_set_depth_write_enable_ext
        };

        unsafe { cmd_set_depth_write_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets the dynamic discard rectangles for future draw calls.
    pub unsafe fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_discard_rectangle_unchecked(first_rectangle, rectangles) })
    }

    pub unsafe fn set_discard_rectangle_unchecked(
        &mut self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> &mut Self {
        if rectangles.is_empty() {
            return self;
        }

        let rectangles_vk = rectangles
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();

        let fns = self.device().fns();
        unsafe {
            (fns.ext_discard_rectangles.cmd_set_discard_rectangle_ext)(
                self.handle(),
                first_rectangle,
                rectangles_vk.len() as u32,
                rectangles_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic front face for future draw calls.
    pub unsafe fn set_front_face(&mut self, face: FrontFace) -> Result<&mut Self> {
        Ok(unsafe { self.set_front_face_unchecked(face) })
    }

    pub unsafe fn set_front_face_unchecked(&mut self, face: FrontFace) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_front_face = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_front_face
        } else {
            fns.ext_extended_dynamic_state.cmd_set_front_face_ext
        };

        unsafe { cmd_set_front_face(self.handle(), face.into()) };

        self
    }

    /// Sets the dynamic line stipple values for future draw calls.
    pub unsafe fn set_line_stipple(&mut self, factor: u32, pattern: u16) -> Result<&mut Self> {
        Ok(unsafe { self.set_line_stipple_unchecked(factor, pattern) })
    }

    pub unsafe fn set_line_stipple_unchecked(&mut self, factor: u32, pattern: u16) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_line_rasterization.cmd_set_line_stipple_ext)(self.handle(), factor, pattern)
        };

        self
    }

    /// Sets the dynamic line width for future draw calls.
    pub unsafe fn set_line_width(&mut self, line_width: f32) -> Result<&mut Self> {
        Ok(unsafe { self.set_line_width_unchecked(line_width) })
    }

    pub unsafe fn set_line_width_unchecked(&mut self, line_width: f32) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_set_line_width)(self.handle(), line_width) };

        self
    }

    /// Sets the dynamic logic op for future draw calls.
    pub unsafe fn set_logic_op(&mut self, logic_op: LogicOp) -> Result<&mut Self> {
        Ok(unsafe { self.set_logic_op_unchecked(logic_op) })
    }

    pub unsafe fn set_logic_op_unchecked(&mut self, logic_op: LogicOp) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_extended_dynamic_state2.cmd_set_logic_op_ext)(self.handle(), logic_op.into())
        };

        self
    }

    /// Sets the dynamic number of patch control points for future draw calls.
    pub unsafe fn set_patch_control_points(&mut self, num: u32) -> Result<&mut Self> {
        Ok(unsafe { self.set_patch_control_points_unchecked(num) })
    }

    pub unsafe fn set_patch_control_points_unchecked(&mut self, num: u32) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_extended_dynamic_state2
                .cmd_set_patch_control_points_ext)(self.handle(), num)
        };

        self
    }

    /// Sets whether dynamic primitive restart is enabled for future draw calls.
    pub unsafe fn set_primitive_restart_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_primitive_restart_enable_unchecked(enable) })
    }

    pub unsafe fn set_primitive_restart_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_primitive_restart_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_primitive_restart_enable
        } else {
            fns.ext_extended_dynamic_state2
                .cmd_set_primitive_restart_enable_ext
        };

        unsafe { cmd_set_primitive_restart_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets the dynamic primitive topology for future draw calls.
    pub unsafe fn set_primitive_topology(
        &mut self,
        topology: PrimitiveTopology,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_primitive_topology_unchecked(topology) })
    }

    pub unsafe fn set_primitive_topology_unchecked(
        &mut self,
        topology: PrimitiveTopology,
    ) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_primitive_topology = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_primitive_topology
        } else {
            fns.ext_extended_dynamic_state
                .cmd_set_primitive_topology_ext
        };

        unsafe { cmd_set_primitive_topology(self.handle(), topology.into()) };

        self
    }

    /// Sets whether dynamic rasterizer discard is enabled for future draw calls.
    pub unsafe fn set_rasterizer_discard_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_rasterizer_discard_enable_unchecked(enable) })
    }

    pub unsafe fn set_rasterizer_discard_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_rasterizer_discard_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_rasterizer_discard_enable
        } else {
            fns.ext_extended_dynamic_state2
                .cmd_set_rasterizer_discard_enable_ext
        };

        unsafe { cmd_set_rasterizer_discard_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets the dynamic scissors for future draw calls.
    pub unsafe fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_scissor_unchecked(first_scissor, scissors) })
    }

    pub unsafe fn set_scissor_unchecked(
        &mut self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> &mut Self {
        if scissors.is_empty() {
            return self;
        }

        let scissors_vk = scissors
            .iter()
            .map(vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_set_scissor)(
                self.handle(),
                first_scissor,
                scissors_vk.len() as u32,
                scissors_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic scissors with count for future draw calls.
    pub unsafe fn set_scissor_with_count(&mut self, scissors: &[Scissor]) -> Result<&mut Self> {
        Ok(unsafe { self.set_scissor_with_count_unchecked(scissors) })
    }

    pub unsafe fn set_scissor_with_count_unchecked(&mut self, scissors: &[Scissor]) -> &mut Self {
        if scissors.is_empty() {
            return self;
        }

        let scissors_vk = scissors
            .iter()
            .map(vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();

        let fns = self.device().fns();
        let cmd_set_scissor_with_count = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_scissor_with_count
        } else {
            fns.ext_extended_dynamic_state
                .cmd_set_scissor_with_count_ext
        };

        unsafe {
            cmd_set_scissor_with_count(
                self.handle(),
                scissors_vk.len() as u32,
                scissors_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic stencil compare mask on one or both faces for future draw calls.
    pub unsafe fn set_stencil_compare_mask(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_stencil_compare_mask_unchecked(faces, compare_mask) })
    }

    pub unsafe fn set_stencil_compare_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_set_stencil_compare_mask)(self.handle(), faces.into(), compare_mask)
        };

        self
    }

    /// Sets the dynamic stencil ops on one or both faces for future draw calls.
    pub unsafe fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op)
        })
    }

    pub unsafe fn set_stencil_op_unchecked(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_stencil_op = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_stencil_op
        } else {
            fns.ext_extended_dynamic_state.cmd_set_stencil_op_ext
        };

        unsafe {
            cmd_set_stencil_op(
                self.handle(),
                faces.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            )
        };

        self
    }

    /// Sets the dynamic stencil reference on one or both faces for future draw calls.
    pub unsafe fn set_stencil_reference(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_stencil_reference_unchecked(faces, reference) })
    }

    pub unsafe fn set_stencil_reference_unchecked(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_set_stencil_reference)(self.handle(), faces.into(), reference) };

        self
    }

    /// Sets whether dynamic stencil testing is enabled for future draw calls.
    pub unsafe fn set_stencil_test_enable(&mut self, enable: bool) -> Result<&mut Self> {
        Ok(unsafe { self.set_stencil_test_enable_unchecked(enable) })
    }

    pub unsafe fn set_stencil_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();
        let cmd_set_stencil_test_enable = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_stencil_test_enable
        } else {
            fns.ext_extended_dynamic_state
                .cmd_set_stencil_test_enable_ext
        };

        unsafe { cmd_set_stencil_test_enable(self.handle(), enable.into()) };

        self
    }

    /// Sets the dynamic stencil write mask on one or both faces for future draw calls.
    pub unsafe fn set_stencil_write_mask(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_stencil_write_mask_unchecked(faces, write_mask) })
    }

    pub unsafe fn set_stencil_write_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_set_stencil_write_mask)(self.handle(), faces.into(), write_mask) };

        self
    }

    /// Sets the dynamic vertex input for future draw calls.
    pub unsafe fn set_vertex_input(
        &mut self,
        vertex_input_state: &VertexInputState,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_vertex_input_unchecked(vertex_input_state) })
    }

    pub unsafe fn set_vertex_input_unchecked(
        &mut self,
        vertex_input_state: &VertexInputState,
    ) -> &mut Self {
        let mut vertex_binding_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_attribute_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();

        let VertexInputState {
            bindings,
            attributes,
            _ne: _,
        } = vertex_input_state;

        vertex_binding_descriptions_vk.extend(bindings.iter().map(|(&binding, binding_desc)| {
            let &VertexInputBindingDescription {
                stride,
                input_rate,
                _ne: _,
            } = binding_desc;

            let divisor = match input_rate {
                // VUID-VkVertexInputBindingDescription2EXT-divisor-06227
                VertexInputRate::Vertex => 1,
                VertexInputRate::Instance { divisor } => divisor,
            };

            vk::VertexInputBindingDescription2EXT {
                binding,
                stride,
                input_rate: input_rate.into(),
                divisor,
                ..Default::default()
            }
        }));

        vertex_attribute_descriptions_vk.extend(attributes.iter().map(
            |(&location, attribute_desc)| {
                let &VertexInputAttributeDescription {
                    binding,
                    format,
                    offset,
                    _ne: _,
                } = attribute_desc;

                vk::VertexInputAttributeDescription2EXT {
                    location,
                    binding,
                    format: format.into(),
                    offset,
                    ..Default::default()
                }
            },
        ));

        let fns = self.device().fns();
        unsafe {
            (fns.ext_vertex_input_dynamic_state.cmd_set_vertex_input_ext)(
                self.handle(),
                vertex_binding_descriptions_vk.len() as u32,
                vertex_binding_descriptions_vk.as_ptr(),
                vertex_attribute_descriptions_vk.len() as u32,
                vertex_attribute_descriptions_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic viewports for future draw calls.
    pub unsafe fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> Result<&mut Self> {
        Ok(unsafe { self.set_viewport_unchecked(first_viewport, viewports) })
    }

    pub unsafe fn set_viewport_unchecked(
        &mut self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> &mut Self {
        if viewports.is_empty() {
            return self;
        }

        let viewports_vk = viewports
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_set_viewport)(
                self.handle(),
                first_viewport,
                viewports_vk.len() as u32,
                viewports_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic viewports with count for future draw calls.
    pub unsafe fn set_viewport_with_count(&mut self, viewports: &[Viewport]) -> Result<&mut Self> {
        Ok(unsafe { self.set_viewport_with_count_unchecked(viewports) })
    }

    pub unsafe fn set_viewport_with_count_unchecked(
        &mut self,
        viewports: &[Viewport],
    ) -> &mut Self {
        if viewports.is_empty() {
            return self;
        }

        let viewports_vk = viewports
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();

        let fns = self.device().fns();
        let cmd_set_viewport_with_count = if self.device().api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_viewport_with_count
        } else {
            fns.ext_extended_dynamic_state
                .cmd_set_viewport_with_count_ext
        };

        unsafe {
            cmd_set_viewport_with_count(
                self.handle(),
                viewports_vk.len() as u32,
                viewports_vk.as_ptr(),
            )
        };

        self
    }

    /// Sets the dynamic conservative rasterization mode for future draw calls.
    pub unsafe fn set_conservative_rasterization_mode(
        &mut self,
        conservative_rasterization_mode: ConservativeRasterizationMode,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.set_conservative_rasterization_mode_unchecked(conservative_rasterization_mode)
        })
    }

    pub unsafe fn set_conservative_rasterization_mode_unchecked(
        &mut self,
        conservative_rasterization_mode: ConservativeRasterizationMode,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_extended_dynamic_state3
                .cmd_set_conservative_rasterization_mode_ext)(
                self.handle(),
                conservative_rasterization_mode.into(),
            )
        };

        self
    }

    /// Sets the dynamic extra primitive overestimation size for future draw calls.
    pub unsafe fn set_extra_primitive_overestimation_size(
        &mut self,
        extra_primitive_overestimation_size: f32,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.set_extra_primitive_overestimation_size_unchecked(
                extra_primitive_overestimation_size,
            )
        })
    }

    pub unsafe fn set_extra_primitive_overestimation_size_unchecked(
        &mut self,
        extra_primitive_overestimation_size: f32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_extended_dynamic_state3
                .cmd_set_extra_primitive_overestimation_size_ext)(
                self.handle(),
                extra_primitive_overestimation_size,
            )
        };

        self
    }
}
