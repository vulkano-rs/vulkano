// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    acceleration_structure::AccelerationStructure,
    buffer::{view::BufferView, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator,
        auto::{RenderPassState, RenderPassStateType, Resource, ResourceUseRef2},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, DispatchIndirectCommand, DrawIndexedIndirectCommand,
        DrawIndirectCommand, ResourceInCommand, SubpassContents,
    },
    descriptor_set::{
        layout::DescriptorType, DescriptorBindingResources, DescriptorBufferInfo,
        DescriptorImageViewInfo,
    },
    device::{DeviceOwned, QueueFlags},
    format::{FormatFeatures, NumericType},
    image::{sampler::Sampler, view::ImageView, ImageAspects, ImageLayout, SampleCount},
    pipeline::{
        graphics::{
            input_assembly::PrimitiveTopology, subpass::PipelineSubpassType,
            vertex_input::VertexInputRate,
        },
        DynamicState, GraphicsPipeline, PartialStateMode, Pipeline, PipelineLayout,
    },
    shader::{DescriptorBindingRequirements, DescriptorIdentifier, ShaderStage, ShaderStages},
    sync::{PipelineStageAccess, PipelineStageAccessFlags},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use std::{mem::size_of, sync::Arc};

macro_rules! vuids {
    ($vuid_type:ident, $($id:literal),+ $(,)?) => {
        match $vuid_type {
            VUIDType::Dispatch => &[$(concat!("VUID-vkCmdDispatch-", $id)),+],
            VUIDType::DispatchIndirect => &[$(concat!("VUID-vkCmdDispatchIndirect-", $id)),+],
            VUIDType::Draw => &[$(concat!("VUID-vkCmdDraw-", $id)),+],
            VUIDType::DrawIndirect => &[$(concat!("VUID-vkCmdDrawIndirect-", $id)),+],
            VUIDType::DrawIndexed => &[$(concat!("VUID-vkCmdDrawIndexed-", $id)),+],
            VUIDType::DrawIndexedIndirect => &[$(concat!("VUID-vkCmdDrawIndexedIndirect-", $id)),+],
        }
    };
}

/// # Commands to execute a bound pipeline.
///
/// Dispatch commands require a compute queue, draw commands require a graphics queue.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Perform a single compute operation using a compute pipeline.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    pub fn dispatch(&mut self, group_counts: [u32; 3]) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch(group_counts)?;

        unsafe { Ok(self.dispatch_unchecked(group_counts)) }
    }

    fn validate_dispatch(&self, group_counts: [u32; 3]) -> Result<(), Box<ValidationError>> {
        self.inner.validate_dispatch(group_counts)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdDispatch-renderpass"],
                ..Default::default()
            }));
        }

        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "no compute pipeline is currently bound".into(),
                vuids: &["VUID-vkCmdDispatch-None-08606"],
                ..Default::default()
            }))?
            .as_ref();

        const VUID_TYPE: VUIDType = VUIDType::Dispatch;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline)?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .unwrap()
            .as_ref();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline);

        self.add_command(
            "dispatch",
            used_resources,
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.dispatch_unchecked(group_counts);
            },
        );

        self
    }

    /// Perform multiple compute operations using a compute pipeline. One dispatch is performed for
    /// each [`DispatchIndirectCommand`] struct in `indirect_buffer`.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    pub fn dispatch_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DispatchIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch_indirect(indirect_buffer.as_bytes())?;

        unsafe { Ok(self.dispatch_indirect_unchecked(indirect_buffer)) }
    }

    fn validate_dispatch_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_dispatch_indirect(indirect_buffer)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdDispatchIndirect-renderpass"],
                ..Default::default()
            }));
        }

        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "no compute pipeline is currently bound".into(),
                vuids: &["VUID-vkCmdDispatchIndirect-None-08606"],
                ..Default::default()
            }))?
            .as_ref();

        const VUID_TYPE: VUIDType = VUIDType::DispatchIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline)?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DispatchIndirectCommand]>,
    ) -> &mut Self {
        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .unwrap()
            .as_ref();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "dispatch",
            used_resources,
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.dispatch_indirect_unchecked(&indirect_buffer);
            },
        );

        self
    }

    /// Perform a single draw operation using a graphics pipeline.
    ///
    /// The parameters specify the first vertex and the number of vertices to draw, and the first
    /// instance and number of instances. For non-instanced drawing, specify `instance_count` as 1
    /// and `first_instance` as 0.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the provided vertex and
    /// instance ranges must be in range of the bound vertex buffers.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        unsafe {
            Ok(self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance))
        }
    }

    fn validate_draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        let render_pass_state =
            self.builder_state
                .render_pass
                .as_ref()
                .ok_or(Box::new(ValidationError {
                    problem: "a render pass instance is not active".into(),
                    vuids: &["VUID-vkCmdDraw-renderpass"],
                    ..Default::default()
                }))?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "no graphics pipeline is currently bound".into(),
                vuids: &["VUID-vkCmdDraw-None-08606"],
                ..Default::default()
            }))?
            .as_ref();

        const VUID_TYPE: VUIDType = VUIDType::Draw;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline)?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(VUID_TYPE, pipeline)?;

        let view_mask = match pipeline.subpass() {
            PipelineSubpassType::BeginRenderPass(subpass) => subpass.render_pass().views_used(),
            PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.view_mask,
        };

        if view_mask != 0 {
            let properties = self.device().physical_device().properties();

            if (first_instance + instance_count).saturating_sub(1)
                > properties.max_multiview_instance_index.unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance has a nonzero view mask, but \
                        `first_instance + instance_count - 1` is greater than the \
                        `max_multiview_instance_index` limit"
                        .into(),
                    vuids: &["VUID-vkCmdDraw-maxMultiviewInstanceIndex-02688"],
                    ..Default::default()
                }));
            }
        }

        for (&binding_num, binding_desc) in &pipeline.vertex_input_state().bindings {
            let vertex_buffer = &self.builder_state.vertex_buffers[&binding_num];

            // Per spec:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-input-address-calculation
            match binding_desc.input_rate {
                VertexInputRate::Vertex => {
                    let max_vertex_offset = (first_vertex as DeviceSize
                        + vertex_count as DeviceSize)
                        * binding_desc.stride as DeviceSize;

                    if max_vertex_offset > vertex_buffer.size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the size of the vertex buffer bound to binding {} is less \
                                than the minimum size required, for the provided \
                                `first_vertex` and `vertex_count` values, and the vertex \
                                input state bindings of the currently bound graphics pipeline",
                                binding_num
                            )
                            .into(),
                            vuids: &["VUID-vkCmdDraw-None-02721"],
                            ..Default::default()
                        }));
                    }
                }
                VertexInputRate::Instance { divisor } => {
                    let max_vertex_offset = if divisor == 0 {
                        (first_instance as DeviceSize + 1) * binding_desc.stride as DeviceSize
                    } else {
                        (first_instance as DeviceSize
                            + instance_count as DeviceSize / divisor as DeviceSize)
                            * binding_desc.stride as DeviceSize
                    };

                    if max_vertex_offset > vertex_buffer.size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the size of the vertex buffer bound to binding {} is less \
                                than the minimum size required, for the provided \
                                `first_instance` and `instance_count` values, and the vertex \
                                input state bindings of the currently bound graphics pipeline",
                                binding_num
                            )
                            .into(),
                            vuids: &["VUID-vkCmdDraw-None-02721"],
                            ..Default::default()
                        }));
                    }
                }
            };
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_unchecked(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline);
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);

        self.add_command(
            "draw",
            used_resources,
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance);
            },
        );

        self
    }

    /// Perform multiple draw operations using a graphics pipeline.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](crate::device::Properties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](crate::device::Features::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the vertex and instance
    /// ranges of each `DrawIndirectCommand` in the indirect buffer must be in range of the bound
    /// vertex buffers.
    pub fn draw_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndirectCommand>() as u32;
        self.validate_draw_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        unsafe { Ok(self.draw_indirect_unchecked(indirect_buffer, draw_count, stride)) }
    }

    fn validate_draw_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_draw_indirect(indirect_buffer, draw_count, stride)?;

        let render_pass_state =
            self.builder_state
                .render_pass
                .as_ref()
                .ok_or(Box::new(ValidationError {
                    problem: "a render pass instance is not active".into(),
                    vuids: &["VUID-vkCmdDrawIndirect-renderpass"],
                    ..Default::default()
                }))?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "no graphics pipeline is currently bound".into(),
                vuids: &["VUID-vkCmdDrawIndirect-None-08606"],
                ..Default::default()
            }))?
            .as_ref();

        const VUID_TYPE: VUIDType = VUIDType::DrawIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline)?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(VUID_TYPE, pipeline)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline);
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "draw_indirect",
            used_resources,
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.draw_indirect_unchecked(&indirect_buffer, draw_count, stride);
            },
        );

        self
    }

    /// Perform a single draw operation using a graphics pipeline, using an index buffer.
    ///
    /// The parameters specify the first index and the number of indices in the index buffer that
    /// should be used, and the first instance and number of instances. For non-instanced drawing,
    /// specify `instance_count` as 1 and `first_instance` as 0. The `vertex_offset` is a constant
    /// value that should be added to each index in the index buffer to produce the final vertex
    /// number to be used.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the provided index range must be in
    /// range of the bound index buffer.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the provided instance
    /// range must be in range of the bound vertex buffers. The vertex indices in the index buffer
    /// must be in range of the bound vertex buffers.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        unsafe {
            Ok(self.draw_indexed_unchecked(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            ))
        }
    }

    fn validate_draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        let render_pass_state =
            self.builder_state
                .render_pass
                .as_ref()
                .ok_or(Box::new(ValidationError {
                    problem: "a render pass instance is not active".into(),
                    vuids: &["VUID-vkCmdDrawIndexed-renderpass"],
                    ..Default::default()
                }))?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "no graphics pipeline is currently bound".into(),
                vuids: &["VUID-vkCmdDrawIndexed-None-08606"],
                ..Default::default()
            }))?
            .as_ref();

        const VUID_TYPE: VUIDType = VUIDType::DrawIndexed;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline)?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(VUID_TYPE, pipeline)?;

        let index_buffer =
            self.builder_state
                .index_buffer
                .as_ref()
                .ok_or(Box::new(ValidationError {
                    problem: "no index buffer is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndexed-None-07312"],
                    ..Default::default()
                }))?;

        let index_buffer_bytes = index_buffer.as_bytes();

        if !self.device().enabled_features().robust_buffer_access2 {
            if index_buffer.index_type().size()
                * (first_index as DeviceSize + index_count as DeviceSize)
                > index_buffer_bytes.size()
            {
                return Err(Box::new(ValidationError {
                    problem: "`first_index + index_count`, \
                        multiplied by the size of the indices in the bound index buffer, \
                        is greater than the size of the bound index buffer"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "robust_buffer_access2",
                    )])]),
                    vuids: &["VUID-vkCmdDrawIndexed-robustBufferAccess2-07825"],
                    ..Default::default()
                }));
            }
        }

        let view_mask = match pipeline.subpass() {
            PipelineSubpassType::BeginRenderPass(subpass) => subpass.render_pass().views_used(),
            PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.view_mask,
        };

        if view_mask != 0 {
            let properties = self.device().physical_device().properties();

            if (first_instance + instance_count).saturating_sub(1)
                > properties.max_multiview_instance_index.unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance has a nonzero view mask, but \
                        `first_instance + instance_count - 1` is greater than the \
                        `max_multiview_instance_index` limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexed-maxMultiviewInstanceIndex-02688"],
                    ..Default::default()
                }));
            }
        }

        for (&binding_num, binding_desc) in &pipeline.vertex_input_state().bindings {
            let vertex_buffer = &self.builder_state.vertex_buffers[&binding_num];

            // Per spec:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-input-address-calculation
            match binding_desc.input_rate {
                VertexInputRate::Vertex => (),
                VertexInputRate::Instance { divisor } => {
                    let max_vertex_offset = if divisor == 0 {
                        (first_instance as DeviceSize + 1) * binding_desc.stride as DeviceSize
                    } else {
                        (first_instance as DeviceSize
                            + instance_count as DeviceSize / divisor as DeviceSize)
                            * binding_desc.stride as DeviceSize
                    };

                    if max_vertex_offset > vertex_buffer.size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the size of the vertex buffer bound to binding {} is less \
                                than the minimum size required, for the provided \
                                `first_instance` and `instance_count` values, and the vertex \
                                input state bindings of the currently bound graphics pipeline",
                                binding_num
                            )
                            .into(),
                            vuids: &["VUID-vkCmdDrawIndexed-None-02721"],
                            ..Default::default()
                        }));
                    }
                }
            };
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_unchecked(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline);
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_index_buffer_resources(&mut used_resources);

        self.add_command(
            "draw_indexed",
            used_resources,
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.draw_indexed_unchecked(
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                );
            },
        );

        self
    }

    /// Perform multiple draw operations using a graphics pipeline, using an index buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](crate::device::Properties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](crate::device::Features::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the index ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound index
    /// buffer.
    ///
    /// A graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the graphics
    /// pipeline, such as descriptor sets, vertex buffers and dynamic state, must have been set
    /// beforehand. If the bound graphics pipeline uses vertex buffers, then the instance ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// vertex buffers.
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndexedIndirectCommand>() as u32;
        self.validate_draw_indexed_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        unsafe { Ok(self.draw_indexed_indirect_unchecked(indirect_buffer, draw_count, stride)) }
    }

    fn validate_draw_indexed_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_draw_indexed_indirect(indirect_buffer, draw_count, stride)?;

        let render_pass_state =
            self.builder_state
                .render_pass
                .as_ref()
                .ok_or(Box::new(ValidationError {
                    problem: "a render pass instance is not active".into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-renderpass"],
                    ..Default::default()
                }))?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "no graphics pipeline is currently bound".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-None-08606"],
                ..Default::default()
            }))?
            .as_ref();

        const VUID_TYPE: VUIDType = VUIDType::DrawIndexedIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline)?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;
        self.validate_pipeline_graphics_vertex_buffers(VUID_TYPE, pipeline)?;

        let _index_buffer =
            self.builder_state
                .index_buffer
                .as_ref()
                .ok_or(Box::new(ValidationError {
                    problem: "no index buffer is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-None-07312"],
                    ..Default::default()
                }))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .unwrap()
            .as_ref();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline);
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_index_buffer_resources(&mut used_resources);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "draw_indexed_indirect",
            used_resources,
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.draw_indexed_indirect_unchecked(&indirect_buffer, draw_count, stride);
            },
        );

        self
    }

    fn validate_pipeline_descriptor_sets<Pl: Pipeline>(
        &self,
        vuid_type: VUIDType,
        pipeline: &Pl,
    ) -> Result<(), Box<ValidationError>> {
        fn validate_resources<T>(
            vuid_type: VUIDType,
            set_num: u32,
            binding_num: u32,
            binding_reqs: &DescriptorBindingRequirements,
            elements: &[Option<T>],
            mut extra_check: impl FnMut(u32, u32, u32, &T) -> Result<(), Box<ValidationError>>,
        ) -> Result<(), Box<ValidationError>> {
            let elements_to_check = if let Some(descriptor_count) = binding_reqs.descriptor_count {
                // The shader has a fixed-sized array, so it will never access more than
                // the first `descriptor_count` elements.
                elements.get(..descriptor_count as usize).ok_or({
                    // There are less than `descriptor_count` elements in `elements`
                    Box::new(ValidationError {
                        problem: format!(
                            "the currently bound pipeline accesses the resource bound to \
                            descriptor set {set_num}, binding {binding_num}, \
                            descriptor index {}, but no descriptor was written to the \
                            descriptor set currently bound to set {set_num}",
                            elements.len()
                        )
                        .into(),
                        vuids: vuids!(vuid_type, "None-02699"),
                        ..Default::default()
                    })
                })?
            } else {
                // The shader has a runtime-sized array, so any element could potentially
                // be accessed. We must check them all.
                elements
            };

            for (index, element) in elements_to_check.iter().enumerate() {
                let index = index as u32;

                let element = match element {
                    Some(x) => x,
                    None => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses the resource bound to \
                                descriptor set {set_num}, binding {binding_num}, \
                                descriptor index {index}, but no descriptor was written to the \
                                descriptor set currently bound to set {set_num}"
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-02699"),
                            ..Default::default()
                        }));
                    }
                };

                extra_check(set_num, binding_num, index, element)?;
            }

            Ok(())
        }

        if pipeline.num_used_descriptor_sets() == 0 {
            return Ok(());
        }

        let descriptor_set_state = self
            .builder_state
            .descriptor_sets
            .get(&pipeline.bind_point())
            .ok_or(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses descriptor sets, but no \
                    descriptor sets were previously bound"
                    .into(),
                vuids: vuids!(vuid_type, "None-02697"),
                ..Default::default()
            }))?;

        if !pipeline.layout().is_compatible_with(
            &descriptor_set_state.pipeline_layout,
            pipeline.num_used_descriptor_sets(),
        ) {
            return Err(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses descriptor sets, but the \
                    pipeline layouts that were used to bind the descriptor sets are \
                    not compatible with the pipeline layout of the currently bound pipeline"
                    .into(),
                vuids: vuids!(vuid_type, "None-02697"),
                ..Default::default()
            }));
        }

        for (&(set_num, binding_num), binding_reqs) in pipeline.descriptor_binding_requirements() {
            let layout_binding =
                &pipeline.layout().set_layouts()[set_num as usize].bindings()[&binding_num];

            let check_buffer =
                |_set_num: u32,
                 _binding_num: u32,
                 _index: u32,
                 _buffer_info: &DescriptorBufferInfo| Ok(());

            let check_buffer_view =
                |set_num: u32, binding_num: u32, index: u32, buffer_view: &Arc<BufferView>| {
                    for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        if layout_binding.descriptor_type == DescriptorType::StorageTexelBuffer {
                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_write.is_empty()
                                && !buffer_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline writes to the buffer view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the buffer view's format do \
                                        not contain `FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06423"),
                                    ..Default::default()
                                }));
                            }

                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_read.is_empty()
                                && !buffer_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_READ_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline reads from the buffer view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the buffer view's format do \
                                        not contain `FormatFeatures::STORAGE_READ_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06424"),
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    Ok(())
                };

            let check_image_view_common =
                |set_num: u32, binding_num: u32, index: u32, image_view: &Arc<ImageView>| {
                    for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        if desc_reqs.storage_image_atomic
                            && !image_view
                                .format_features()
                                .intersects(FormatFeatures::STORAGE_IMAGE_ATOMIC)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline performs atomic operations on \
                                    the image view bound to descriptor set {set_num}, \
                                    binding {binding_num}, descriptor index {index}, but \
                                    the format features of the image view's format do \
                                    not contain `FormatFeatures::STORAGE_IMAGE_ATOMIC`"
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-02691"),
                                ..Default::default()
                            }));
                        }

                        if layout_binding.descriptor_type == DescriptorType::StorageImage {
                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_write.is_empty()
                                && !image_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline writes to the image view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the image view's format do \
                                        not contain `FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06423"),
                                    ..Default::default()
                                }));
                            }

                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_read.is_empty()
                                && !image_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_READ_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline reads from the image view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the image view's format do \
                                        not contain `FormatFeatures::STORAGE_READ_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06424"),
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    /*
                       Instruction/Sampler/Image View Validation
                       https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                    */

                    // The SPIR-V Image Format is not compatible with the image view’s format.
                    if let Some(format) = binding_reqs.image_format {
                        if image_view.format() != format {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the image view \
                                    bound to descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, but the format of the image view \
                                    is not equal to the format required by the pipeline"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    // Rules for viewType
                    if let Some(image_view_type) = binding_reqs.image_view_type {
                        if image_view.view_type() != image_view_type {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the image view \
                                    bound to descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, but the view type of the image view \
                                    is not equal to the view type required by the pipeline"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    // - If the image was created with VkImageCreateInfo::samples equal to
                    //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 0.
                    // - If the image was created with VkImageCreateInfo::samples not equal to
                    //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 1.
                    if binding_reqs.image_multisampled
                        && image_view.image().samples() == SampleCount::Sample1
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses the image view \
                                bound to descriptor set {set_num}, binding {binding_num}, \
                                descriptor index {index}, and the pipeline requires a \
                                multisampled image, but the image view has only one sample"
                            )
                            .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    } else if !binding_reqs.image_multisampled
                        && image_view.image().samples() != SampleCount::Sample1
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses the image view \
                                bound to descriptor set {set_num}, binding {binding_num}, \
                                descriptor index {index}, and the pipeline requires a non-\
                                multisampled image, but the image view has more than one sample"
                            )
                            .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    }

                    // - If the Sampled Type of the OpTypeImage does not match the numeric format of the
                    //   image, as shown in the SPIR-V Sampled Type column of the
                    //   Interpretation of Numeric Format table.
                    // - If the signedness of any read or sample operation does not match the signedness of
                    //   the image’s format.
                    if let Some(shader_numeric_type) = binding_reqs.image_scalar_type {
                        let aspects = image_view.subresource_range().aspects;
                        let view_numeric_type = NumericType::from(
                            if aspects.intersects(
                                ImageAspects::COLOR
                                    | ImageAspects::PLANE_0
                                    | ImageAspects::PLANE_1
                                    | ImageAspects::PLANE_2,
                            ) {
                                image_view.format().numeric_format_color().unwrap()
                            } else if aspects.intersects(ImageAspects::DEPTH) {
                                image_view.format().numeric_format_depth().unwrap()
                            } else if aspects.intersects(ImageAspects::STENCIL) {
                                image_view.format().numeric_format_stencil().unwrap()
                            } else {
                                // Per `ImageViewBuilder::aspects` and
                                // VUID-VkDescriptorImageInfo-imageView-01976
                                unreachable!()
                            },
                        );

                        if shader_numeric_type != view_numeric_type {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the image view \
                                    bound to descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, and the pipeline requires an \
                                    image view whose format has a `{shader_numeric_type:?}` \
                                    numeric type, but the format of the image view has a \
                                    `{view_numeric_type:?}` numeric type"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    Ok(())
                };

            let check_sampler_common =
                |set_num: u32, binding_num: u32, index: u32, sampler: &Arc<Sampler>| {
                    for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        if desc_reqs.sampler_no_unnormalized_coordinates
                            && sampler.unnormalized_coordinates()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that does not support \
                                    samplers with unnormalized coordinates, but \
                                    the sampler currently bound to that descriptor \
                                    uses unnormalized coordinates"
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-02703", "None-02704"),
                                ..Default::default()
                            }));
                        }

                        // - OpImageFetch, OpImageSparseFetch, OpImage*Gather, and OpImageSparse*Gather must not
                        //   be used with a sampler that enables sampler Y′CBCR conversion.
                        // - The ConstOffset and Offset operands must not be used with a sampler that enables
                        //   sampler Y′CBCR conversion.
                        if desc_reqs.sampler_no_ycbcr_conversion
                            && sampler.sampler_ycbcr_conversion().is_some()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that does not support \
                                    samplers with a sampler YCbCr conversion, but the sampler \
                                    currently bound to that descriptor has a \
                                    sampler YCbCr conversion"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }

                        /*
                            Instruction/Sampler/Image View Validation
                            https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                        */

                        if desc_reqs.sampler_compare && sampler.compare().is_none() {
                            // - The SPIR-V instruction is one of the OpImage*Dref* instructions and the sampler
                            //   compareEnable is VK_FALSE
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that requires a sampler \
                                    with a compare operation, but the sampler currently bound to \
                                    that descriptor does not have a compare operation"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        } else if !desc_reqs.sampler_compare && sampler.compare().is_some() {
                            // - The SPIR-V instruction is not one of the OpImage*Dref* instructions and the sampler
                            //   compareEnable is VK_TRUE
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that requires a sampler \
                                    without a compare operation, but the sampler currently bound \
                                    to that descriptor has a compare operation"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    Ok(())
                };

            let check_image_view =
                |set_num: u32,
                 binding_num: u32,
                 index: u32,
                 image_view_info: &DescriptorImageViewInfo| {
                    let DescriptorImageViewInfo {
                        image_view,
                        image_layout: _,
                    } = image_view_info;

                    check_image_view_common(set_num, binding_num, index, image_view)?;

                    if let Some(sampler) = layout_binding.immutable_samplers.get(index as usize) {
                        check_sampler_common(set_num, binding_num, index, sampler)?;
                    }

                    Ok(())
                };

            let check_image_view_sampler = |set_num: u32,
                                            binding_num: u32,
                                            index: u32,
                                            (image_view_info, sampler): &(
                DescriptorImageViewInfo,
                Arc<Sampler>,
            )| {
                let DescriptorImageViewInfo {
                    image_view,
                    image_layout: _,
                } = image_view_info;

                check_image_view_common(set_num, binding_num, index, image_view)?;
                check_sampler_common(set_num, binding_num, index, sampler)?;

                Ok(())
            };

            let check_sampler =
                |set_num: u32, binding_num: u32, index: u32, sampler: &Arc<Sampler>| {
                    check_sampler_common(set_num, binding_num, index, sampler)?;

                    for desc_reqs in (binding_reqs.descriptors.get(&Some(index)).into_iter())
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        // Check sampler-image compatibility. Only done for separate samplers;
                        // combined image samplers are checked when updating the descriptor set.

                        // If the image view isn't actually present in the resources, then just skip it.
                        // It will be caught later by check_resources.
                        let iter = desc_reqs.sampler_with_images.iter().filter_map(|id| {
                            descriptor_set_state
                                .descriptor_sets
                                .get(&id.set)
                                .and_then(|set| set.resources().binding(id.binding))
                                .and_then(|res| match res {
                                    DescriptorBindingResources::ImageView(elements) => elements
                                        .get(id.index as usize)
                                        .and_then(|opt| opt.as_ref().map(|opt| (id, opt))),
                                    _ => None,
                                })
                        });

                        for (id, image_view_info) in iter {
                            let DescriptorIdentifier {
                                set: iset_num,
                                binding: ibinding_num,
                                index: iindex,
                            } = id;
                            let DescriptorImageViewInfo {
                                image_view,
                                image_layout: _,
                            } = image_view_info;

                            if let Err(error) = sampler.check_can_sample(image_view.as_ref()) {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline uses the sampler bound to \
                                        descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, to sample the image bound to \
                                        descriptor set {iset_num}, binding {ibinding_num}, \
                                        descriptor index {iindex}, but the sampler is not \
                                        compatible with the image: {}",
                                        error,
                                    )
                                    .into(),
                                    // vuids?
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    Ok(())
                };

            let check_acceleration_structure =
                |_set_num: u32,
                 _binding_num: u32,
                 _index: u32,
                 _acceleration_structure: &Arc<AccelerationStructure>| Ok(());

            let check_none = |set_num: u32, binding_num: u32, index: u32, _: &()| {
                if let Some(sampler) = layout_binding.immutable_samplers.get(index as usize) {
                    check_sampler(set_num, binding_num, index, sampler)?;
                }

                Ok(())
            };

            let set_resources = descriptor_set_state
                .descriptor_sets
                .get(&set_num)
                .ok_or(Box::new(ValidationError {
                    problem: format!(
                        "the currently bound pipeline accesses descriptor set {set_num}, but \
                        no descriptor set was previously bound"
                    )
                    .into(),
                    // vuids?
                    ..Default::default()
                }))?
                .resources();

            let binding_resources = set_resources.binding(binding_num).unwrap();

            match binding_resources {
                DescriptorBindingResources::None(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_none,
                    )?;
                }
                DescriptorBindingResources::Buffer(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_buffer,
                    )?;
                }
                DescriptorBindingResources::BufferView(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_buffer_view,
                    )?;
                }
                DescriptorBindingResources::ImageView(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_image_view,
                    )?;
                }
                DescriptorBindingResources::ImageViewSampler(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_image_view_sampler,
                    )?;
                }
                DescriptorBindingResources::Sampler(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_sampler,
                    )?;
                }
                // Spec:
                // Descriptor bindings with descriptor type of
                // VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK can be undefined when
                // the descriptor set is consumed; though values in that block will be undefined.
                //
                // TODO: We *may* still want to validate this?
                DescriptorBindingResources::InlineUniformBlock => (),
                DescriptorBindingResources::AccelerationStructure(elements) => {
                    validate_resources(
                        vuid_type,
                        set_num,
                        binding_num,
                        binding_reqs,
                        elements,
                        check_acceleration_structure,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn validate_pipeline_push_constants(
        &self,
        vuid_type: VUIDType,
        pipeline_layout: &PipelineLayout,
    ) -> Result<(), Box<ValidationError>> {
        if pipeline_layout.push_constant_ranges().is_empty()
            || self.device().enabled_features().maintenance4
        {
            return Ok(());
        }

        let constants_pipeline_layout = self
            .builder_state
            .push_constants_pipeline_layout
            .as_ref()
            .ok_or(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses push constants, but no \
                    push constants were previously set"
                    .into(),
                vuids: vuids!(vuid_type, "maintenance4-06425"),
                ..Default::default()
            }))?;

        if pipeline_layout.handle() != constants_pipeline_layout.handle()
            && pipeline_layout.push_constant_ranges()
                != constants_pipeline_layout.push_constant_ranges()
        {
            return Err(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses push constants, but the \
                    pipeline layouts that were used to set the push constants are \
                    not compatible with the pipeline layout of the currently bound pipeline"
                    .into(),
                vuids: vuids!(vuid_type, "maintenance4-06425"),
                ..Default::default()
            }));
        }

        let set_bytes = &self.builder_state.push_constants;

        if !pipeline_layout
            .push_constant_ranges()
            .iter()
            .all(|pc_range| set_bytes.contains(pc_range.offset..pc_range.offset + pc_range.size))
        {
            return Err(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses push constants, but \
                    not all bytes in the push constant ranges of the pipeline layout of the \
                    currently bound pipeline have been set"
                    .into(),
                vuids: vuids!(vuid_type, "maintenance4-06425"),
                ..Default::default()
            }));
        }

        Ok(())
    }

    fn validate_pipeline_graphics_dynamic_state(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        let device = pipeline.device();

        for dynamic_state in pipeline
            .dynamic_states()
            .filter(|(_, d)| *d)
            .map(|(s, _)| s)
        {
            match dynamic_state {
                DynamicState::BlendConstants => {
                    if self.builder_state.blend_constants.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07835"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::ColorWriteEnable => {
                    let enables = if let Some(enables) = &self.builder_state.color_write_enable {
                        enables
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07749"),
                            ..Default::default()
                        }));
                    };

                    if enables.len() < pipeline.color_blend_state().unwrap().attachments.len() {
                        return Err(Box::new(ValidationError {
                            problem: "the currently bound graphics pipeline requires the \
                                `DynamicState::ColorWriteEnable` dynamic state, but \
                                the number of enable values that were set is less than the number \
                                of color attachments in the color blend state of the \
                                graphics pipeline"
                                .into(),
                            vuids: vuids!(vuid_type, "attachmentCount-07750"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::CullMode => {
                    if self.builder_state.cull_mode.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07840"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBias => {
                    if self.builder_state.depth_bias.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07834"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBiasEnable => {
                    if self.builder_state.depth_bias_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-04877"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBounds => {
                    if self.builder_state.depth_bounds.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07836"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBoundsTestEnable => {
                    if self.builder_state.depth_bounds_test_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07846"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthCompareOp => {
                    if self.builder_state.depth_compare_op.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07845"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthTestEnable => {
                    if self.builder_state.depth_test_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07843"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthWriteEnable => {
                    if self.builder_state.depth_write_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07844"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DiscardRectangle => {
                    let discard_rectangle_count =
                        match pipeline.discard_rectangle_state().unwrap().rectangles {
                            PartialStateMode::Dynamic(count) => count,
                            _ => unreachable!(),
                        };

                    for num in 0..discard_rectangle_count {
                        if !self.builder_state.discard_rectangle.contains_key(&num) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set for discard rectangle {1}, or \
                                    it was overwritten by a more recent \
                                    `bind_pipeline_graphics` command",
                                    dynamic_state, num,
                                ).into(),
                                vuids: vuids!(vuid_type, "None-07751"),
                                ..Default::default()
                            }));
                        }
                    }
                }
                DynamicState::ExclusiveScissor => todo!(),
                DynamicState::FragmentShadingRate => todo!(),
                DynamicState::FrontFace => {
                    if self.builder_state.front_face.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-0784"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::LineStipple => {
                    if self.builder_state.line_stipple.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07849"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::LineWidth => {
                    if self.builder_state.line_width.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07833"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::LogicOp => {
                    if self.builder_state.logic_op.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "logicOp-04878"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::PatchControlPoints => {
                    if self.builder_state.patch_control_points.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-04875"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::PrimitiveRestartEnable => {
                    let primitive_restart_enable =
                        if let Some(enable) = self.builder_state.primitive_restart_enable {
                            enable
                        } else {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set, or it was overwritten by a \
                                    more recent `bind_pipeline_graphics` command",
                                    dynamic_state
                                ).into(),
                                vuids: vuids!(vuid_type, "None-04879"),
                                ..Default::default()
                            }));
                        };

                    if primitive_restart_enable {
                        let topology = match pipeline.input_assembly_state().topology {
                            PartialStateMode::Fixed(topology) => topology,
                            PartialStateMode::Dynamic(_) => {
                                if let Some(topology) = self.builder_state.primitive_topology {
                                    topology
                                } else {
                                    return Err(Box::new(ValidationError {
                                        problem: "the currently bound graphics pipeline requires \
                                            the `DynamicState::PrimitiveTopology` dynamic state, \
                                            but this state was either not set, or it was \
                                            overwritten by a more recent `bind_pipeline_graphics` \
                                            command"
                                            .into(),
                                        vuids: vuids!(vuid_type, "None-07842"),
                                        ..Default::default()
                                    }));
                                }
                            }
                        };

                        match topology {
                            PrimitiveTopology::PointList
                            | PrimitiveTopology::LineList
                            | PrimitiveTopology::TriangleList
                            | PrimitiveTopology::LineListWithAdjacency
                            | PrimitiveTopology::TriangleListWithAdjacency => {
                                if !device.enabled_features().primitive_topology_list_restart {
                                    return Err(Box::new(ValidationError {
                                        problem: "primitive restart is enabled for the currently \
                                            bound graphics pipeline, but the currently set \
                                            dynamic primitive topology is \
                                            `PrimitiveTopology::*List`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("primitive_topology_list_restart")])]),
                                        // vuids?
                                        ..Default::default()
                                    }));
                                }
                            }
                            PrimitiveTopology::PatchList => {
                                if !device
                                    .enabled_features()
                                    .primitive_topology_patch_list_restart
                                {
                                    return Err(Box::new(ValidationError {
                                        problem: "primitive restart is enabled for the currently \
                                            bound graphics pipeline, but the currently set \
                                            dynamic primitive topology is \
                                            `PrimitiveTopology::PatchList`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("primitive_topology_patch_list_restart")])]),
                                        // vuids?
                                        ..Default::default()
                                    }));
                                }
                            }
                            _ => (),
                        }
                    }
                }
                DynamicState::PrimitiveTopology => {
                    let topology = if let Some(topology) = self.builder_state.primitive_topology {
                        topology
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07842"),
                            ..Default::default()
                        }));
                    };

                    if pipeline.shader(ShaderStage::TessellationControl).is_some() {
                        if !matches!(topology, PrimitiveTopology::PatchList) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, and the graphics pipeline \
                                    includes tessellation shader stages, but the dynamic \
                                    primitive topology is not `PrimitiveTopology::PatchList`",
                                    dynamic_state
                                ).into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    } else {
                        if matches!(topology, PrimitiveTopology::PatchList) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, and the graphics pipeline \
                                    does not include tessellation shader stages, but the dynamic \
                                    primitive topology is `PrimitiveTopology::PatchList`",
                                    dynamic_state
                                ).into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    let properties = device.physical_device().properties();

                    if !properties.dynamic_primitive_topology_unrestricted.unwrap_or(false) {
                        let required_topology_class = match pipeline.input_assembly_state().topology {
                            PartialStateMode::Dynamic(topology_class) => topology_class,
                            _ => unreachable!(),
                        };

                        if topology.class() != required_topology_class {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, and the \
                                    `dynamic_primitive_topology_unrestricted` device property is \
                                    `false`, but the dynamic primitive topology does not belong \
                                    to the same topology class as the topology that the \
                                    graphics pipeline was created with",
                                    dynamic_state
                                ).into(),
                                vuids: vuids!(vuid_type, "dynamicPrimitiveTopologyUnrestricted-07500"),
                                ..Default::default()
                            }));
                        }
                    }

                    // TODO: check that the topology matches the geometry shader
                }
                DynamicState::RasterizerDiscardEnable => {
                    if self.builder_state.rasterizer_discard_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-04876"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::RayTracingPipelineStackSize => unreachable!(
                    "RayTracingPipelineStackSize dynamic state should not occur on a graphics pipeline"
                ),
                DynamicState::SampleLocations => todo!(),
                DynamicState::Scissor => {
                    for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                        if !self.builder_state.scissor.contains_key(&num) {
                            return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07832"),
                            ..Default::default()
                        }));
                        }
                    }
                }
                DynamicState::ScissorWithCount => {
                    if let Some(scissors) = &self.builder_state.scissor_with_count {
                        if let Some(viewport_count) = pipeline.viewport_state().unwrap().count() {
                            // Check if the counts match, but only if the viewport count is fixed.
                            // If the viewport count is also dynamic, then the
                            // DynamicState::ViewportWithCount match arm will handle it.

                            if viewport_count != scissors.len() as u32 {
                                return Err(Box::new(ValidationError {
                                    problem: "the currently bound graphics pipeline requires the \
                                        `DynamicState::ScissorWithCount` dynamic state, and \
                                        not the `DynamicState::ViewportWithCount` dynamic state, \
                                        but the dynamic scissor count is not equal to the \
                                        viewport count specified when creating the pipeline"
                                        .into(),
                                    vuids: vuids!(vuid_type, "scissorCount-03418"),
                                    ..Default::default()
                                }));
                            }
                        }
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "scissorCount-03418", "viewportCount-03419"),
                            ..Default::default()
                        }));
                    };
                }
                DynamicState::StencilCompareMask => {
                    let state = self.builder_state.stencil_compare_mask;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07837"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilOp => {
                    let state = self.builder_state.stencil_op;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07848"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilReference => {
                    let state = self.builder_state.stencil_reference;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07839"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilTestEnable => {
                    if self.builder_state.stencil_test_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07847"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilWriteMask => {
                    let state = self.builder_state.stencil_write_mask;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07838"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::VertexInput => todo!(),
                DynamicState::VertexInputBindingStride => todo!(),
                DynamicState::Viewport => {
                    for num in 0..pipeline.viewport_state().unwrap().count().unwrap() {
                        if !self.builder_state.viewport.contains_key(&num) {
                            return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "None-07831"),
                            ..Default::default()
                        }));
                        }
                    }
                }
                DynamicState::ViewportCoarseSampleOrder => todo!(),
                DynamicState::ViewportShadingRatePalette => todo!(),
                DynamicState::ViewportWithCount => {
                    let viewport_count = if let Some(viewports) = &self.builder_state.viewport_with_count {
                        viewports.len() as u32
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            ).into(),
                            vuids: vuids!(vuid_type, "viewportCount-03417", "viewportCount-03419"),
                            ..Default::default()
                        }));
                    };

                    if let Some(scissor_count) =
                        pipeline.viewport_state().unwrap().count()
                    {
                        if viewport_count != scissor_count {
                            return Err(Box::new(ValidationError {
                                problem: "the currently bound graphics pipeline requires the \
                                    `DynamicState::ViewportWithCount` dynamic state, and \
                                    not the `DynamicState::ScissorWithCount` dynamic state, but \
                                    the dynamic scissor count is not equal to the scissor count \
                                    specified when creating the pipeline"
                                    .into(),
                                vuids: vuids!(vuid_type, "viewportCount-03417"),
                                ..Default::default()
                            }));
                        }
                    } else {
                        if let Some(scissors) = &self.builder_state.scissor_with_count {
                            if viewport_count != scissors.len() as u32 {
                                return Err(Box::new(ValidationError {
                                    problem: "the currently bound graphics pipeline requires both \
                                        the `DynamicState::ViewportWithCount` and the \
                                        `DynamicState::ScissorWithCount` dynamic states, but \
                                        the dynamic scissor count is not equal to the \
                                        dynamic scissor count "
                                        .into(),
                                    vuids: vuids!(vuid_type, "viewportCount-03419"),
                                    ..Default::default()
                                }));
                            }
                        } else {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set, or it was overwritten by a \
                                    more recent `bind_pipeline_graphics` command",
                                    dynamic_state
                                ).into(),
                                vuids: vuids!(vuid_type, "scissorCount-03418", "viewportCount-03419"),
                                ..Default::default()
                            }));
                        }
                    }

                    // TODO: VUID-vkCmdDrawIndexed-primitiveFragmentShadingRateWithMultipleViewports-04552
                    // If the primitiveFragmentShadingRateWithMultipleViewports limit is not supported,
                    // the bound graphics pipeline was created with the
                    // VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT dynamic state enabled, and any of the
                    // shader stages of the bound graphics pipeline write to the PrimitiveShadingRateKHR
                    // built-in, then vkCmdSetViewportWithCountEXT must have been called in the current
                    // command buffer prior to this drawing command, and the viewportCount parameter of
                    // vkCmdSetViewportWithCountEXT must be 1
                }
                DynamicState::ViewportWScaling => todo!(),
                DynamicState::TessellationDomainOrigin => todo!(),
                DynamicState::DepthClampEnable => todo!(),
                DynamicState::PolygonMode => todo!(),
                DynamicState::RasterizationSamples => todo!(),
                DynamicState::SampleMask => todo!(),
                DynamicState::AlphaToCoverageEnable => todo!(),
                DynamicState::AlphaToOneEnable => todo!(),
                DynamicState::LogicOpEnable => todo!(),
                DynamicState::ColorBlendEnable => todo!(),
                DynamicState::ColorBlendEquation => todo!(),
                DynamicState::ColorWriteMask => todo!(),
                DynamicState::RasterizationStream => todo!(),
                DynamicState::ConservativeRasterizationMode => todo!(),
                DynamicState::ExtraPrimitiveOverestimationSize => todo!(),
                DynamicState::DepthClipEnable => todo!(),
                DynamicState::SampleLocationsEnable => todo!(),
                DynamicState::ColorBlendAdvanced => todo!(),
                DynamicState::ProvokingVertexMode => todo!(),
                DynamicState::LineRasterizationMode => todo!(),
                DynamicState::LineStippleEnable => todo!(),
                DynamicState::DepthClipNegativeOneToOne => todo!(),
                DynamicState::ViewportWScalingEnable => todo!(),
                DynamicState::ViewportSwizzle => todo!(),
                DynamicState::CoverageToColorEnable => todo!(),
                DynamicState::CoverageToColorLocation => todo!(),
                DynamicState::CoverageModulationMode => todo!(),
                DynamicState::CoverageModulationTableEnable => todo!(),
                DynamicState::CoverageModulationTable => todo!(),
                DynamicState::ShadingRateImageEnable => todo!(),
                DynamicState::RepresentativeFragmentTestEnable => todo!(),
                DynamicState::CoverageReductionMode => todo!(),
            }
        }

        Ok(())
    }

    fn validate_pipeline_graphics_render_pass(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
        render_pass_state: &RenderPassState,
    ) -> Result<(), Box<ValidationError>> {
        if render_pass_state.contents != SubpassContents::Inline {
            return Err(Box::new(ValidationError {
                problem: "the contents of the current subpass instance is not \
                    `SubpassContents::Inline`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        match (&render_pass_state.render_pass, pipeline.subpass()) {
            (
                RenderPassStateType::BeginRenderPass(state),
                PipelineSubpassType::BeginRenderPass(pipeline_subpass),
            ) => {
                if !pipeline_subpass
                    .render_pass()
                    .is_compatible_with(state.subpass.render_pass())
                {
                    return Err(Box::new(ValidationError {
                        problem: "the current render pass instance is not compatible with the \
                            render pass that the currently bound graphics pipeline was \
                            created with"
                            .into(),
                        vuids: vuids!(vuid_type, "renderPass-02684"),
                        ..Default::default()
                    }));
                }

                if pipeline_subpass.index() != state.subpass.index() {
                    return Err(Box::new(ValidationError {
                        problem: "the subpass index of the current render pass instance is not \
                            equal to the index of the subpass that the bound graphics pipeline was \
                            created with"
                            .into(),
                        vuids: vuids!(vuid_type, "subpass-02685"),
                        ..Default::default()
                    }));
                }
            }
            (
                RenderPassStateType::BeginRendering(_),
                PipelineSubpassType::BeginRendering(pipeline_rendering_info),
            ) => {
                if pipeline_rendering_info.view_mask != render_pass_state.rendering_info.view_mask {
                    return Err(Box::new(ValidationError {
                        problem: "the `view_mask` of the current render pass instance is not \
                            equal to the `view_mask` the bound graphics pipeline was created with"
                            .into(),
                        vuids: vuids!(vuid_type, "viewMask-06178"),
                        ..Default::default()
                    }));
                }

                if pipeline_rendering_info.color_attachment_formats.len()
                    != render_pass_state
                        .rendering_info
                        .color_attachment_formats
                        .len()
                {
                    return Err(Box::new(ValidationError {
                        problem: "`color_attachments.len()` of the current render pass \
                            instance is not equal to `color_attachment_formats.len()` that the
                            currently bound graphics pipeline was created with"
                            .into(),
                        vuids: vuids!(vuid_type, "colorAttachmentCount-06179"),
                        ..Default::default()
                    }));
                }

                for (color_attachment_index, required_format, pipeline_format) in render_pass_state
                    .rendering_info
                    .color_attachment_formats
                    .iter()
                    .zip(
                        pipeline_rendering_info
                            .color_attachment_formats
                            .iter()
                            .copied(),
                    )
                    .enumerate()
                    .filter_map(|(i, (r, p))| r.map(|r| (i as u32, r, p)))
                {
                    if Some(required_format) != pipeline_format {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`color_attachments[{0}].image_view.format()` of the current \
                                render pass instance is not equal to \
                                `color_attachment_formats[{0}]` that the currently bound \
                                graphics pipeline was created with",
                                color_attachment_index,
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "colorAttachmentCount-06180"),
                            ..Default::default()
                        }));
                    }
                }

                if let Some((required_format, pipeline_format)) = render_pass_state
                    .rendering_info
                    .depth_attachment_format
                    .map(|r| (r, pipeline_rendering_info.depth_attachment_format))
                {
                    if Some(required_format) != pipeline_format {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment.image_view.format()` of the current \
                                render pass instance is not equal to \
                                `depth_attachment_format` that the currently bound \
                                graphics pipeline was created with"
                                .into(),
                            vuids: vuids!(vuid_type, "pDepthAttachment-06181"),
                            ..Default::default()
                        }));
                    }
                }

                if let Some((required_format, pipeline_format)) = render_pass_state
                    .rendering_info
                    .stencil_attachment_format
                    .map(|r| (r, pipeline_rendering_info.stencil_attachment_format))
                {
                    if Some(required_format) != pipeline_format {
                        return Err(Box::new(ValidationError {
                            problem: "`stencil_attachment.image_view.format()` of the current \
                                render pass instance is not equal to \
                                `stencil_attachment_format` that the currently bound \
                                graphics pipeline was created with"
                                .into(),
                            vuids: vuids!(vuid_type, "pStencilAttachment-06182"),
                            ..Default::default()
                        }));
                    }
                }

                // VUID-vkCmdDraw-imageView-06172
                // VUID-vkCmdDraw-imageView-06173
                // VUID-vkCmdDraw-imageView-06174
                // VUID-vkCmdDraw-imageView-06175
                // VUID-vkCmdDraw-imageView-06176
                // VUID-vkCmdDraw-imageView-06177
                // TODO:
            }
            (RenderPassStateType::BeginRenderPass(_), PipelineSubpassType::BeginRendering(_)) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was begun with \
                        `begin_render_pass`, but the currently bound graphics pipeline requires \
                        a render pass instance begun with `begin_rendering`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
            (RenderPassStateType::BeginRendering(_), PipelineSubpassType::BeginRenderPass(_)) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was begun with \
                        `begin_rendering`, but the currently bound graphics pipeline requires \
                        a render pass instance begun with `begin_render_pass`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdDraw-None-02686
        // TODO:

        Ok(())
    }

    fn validate_pipeline_graphics_vertex_buffers(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        let vertex_input = pipeline.vertex_input_state();

        for &binding_num in vertex_input.bindings.keys() {
            if !self.builder_state.vertex_buffers.contains_key(&binding_num) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the currently bound graphics pipeline uses \
                        vertex buffer binding {0}, but \
                        no vertex buffer is currently bound to binding {0}",
                        binding_num
                    )
                    .into(),
                    vuids: vuids!(vuid_type, "None-04007"),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    fn add_descriptor_sets_resources<Pl: Pipeline>(
        &self,
        used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
        pipeline: &Pl,
    ) {
        let descriptor_sets_state = match self
            .builder_state
            .descriptor_sets
            .get(&pipeline.bind_point())
        {
            Some(x) => x,
            None => return,
        };

        for (&(set, binding), binding_reqs) in pipeline.descriptor_binding_requirements() {
            let descriptor_type = descriptor_sets_state.pipeline_layout.set_layouts()[set as usize]
                .bindings()[&binding]
                .descriptor_type;

            // TODO: Should input attachments be handled here or in attachment access?
            if descriptor_type == DescriptorType::InputAttachment {
                continue;
            }

            let default_image_layout = descriptor_type.default_image_layout();

            let use_iter = move |index: u32| {
                let (stages_read, stages_write) = [Some(index), None]
                    .into_iter()
                    .filter_map(|index| binding_reqs.descriptors.get(&index))
                    .fold(
                        (ShaderStages::empty(), ShaderStages::empty()),
                        |(stages_read, stages_write), desc_reqs| {
                            (
                                stages_read | desc_reqs.memory_read,
                                stages_write | desc_reqs.memory_write,
                            )
                        },
                    );
                let use_ref = ResourceUseRef2::from(ResourceInCommand::DescriptorSet {
                    set,
                    binding,
                    index,
                });
                let memory_access = PipelineStageAccess::iter_descriptor_stages(
                    descriptor_type,
                    stages_read,
                    stages_write,
                )
                .fold(PipelineStageAccessFlags::empty(), |total, val| {
                    total | val.into()
                });
                (use_ref, memory_access)
            };

            let descriptor_set_state = &descriptor_sets_state.descriptor_sets[&set];

            match descriptor_set_state.resources().binding(binding).unwrap() {
                DescriptorBindingResources::None(_) => (),
                DescriptorBindingResources::Buffer(elements) => {
                    if matches!(
                        descriptor_type,
                        DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
                    ) {
                        let dynamic_offsets = descriptor_set_state.dynamic_offsets();

                        for (index, element) in elements.iter().enumerate() {
                            if let Some(buffer_info) = element {
                                let DescriptorBufferInfo { buffer, range } = buffer_info;

                                let dynamic_offset = dynamic_offsets[index] as DeviceSize;
                                let (use_ref, memory_access) = use_iter(index as u32);

                                let mut range = range.clone();
                                range.start += buffer.offset() + dynamic_offset;
                                range.end += buffer.offset() + dynamic_offset;

                                used_resources.push((
                                    use_ref,
                                    Resource::Buffer {
                                        buffer: buffer.clone(),
                                        range: range.clone(),
                                        memory_access,
                                    },
                                ));
                            }
                        }
                    } else {
                        for (index, element) in elements.iter().enumerate() {
                            if let Some(buffer_info) = element {
                                let DescriptorBufferInfo { buffer, range } = buffer_info;

                                let (use_ref, memory_access) = use_iter(index as u32);

                                let mut range = range.clone();
                                range.start += buffer.offset();
                                range.end += buffer.offset();

                                used_resources.push((
                                    use_ref,
                                    Resource::Buffer {
                                        buffer: buffer.clone(),
                                        range: range.clone(),
                                        memory_access,
                                    },
                                ));
                            }
                        }
                    }
                }
                DescriptorBindingResources::BufferView(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some(buffer_view) = element {
                            let buffer = buffer_view.buffer();
                            let (use_ref, memory_access) = use_iter(index as u32);

                            let mut range = buffer_view.range();
                            range.start += buffer.offset();
                            range.end += buffer.offset();

                            used_resources.push((
                                use_ref,
                                Resource::Buffer {
                                    buffer: buffer.clone(),
                                    range: range.clone(),
                                    memory_access,
                                },
                            ));
                        }
                    }
                }
                DescriptorBindingResources::ImageView(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some(image_view_info) = element {
                            let &DescriptorImageViewInfo {
                                ref image_view,
                                mut image_layout,
                            } = image_view_info;

                            if image_layout == ImageLayout::Undefined {
                                image_layout = default_image_layout;
                            }

                            let (use_ref, memory_access) = use_iter(index as u32);

                            used_resources.push((
                                use_ref,
                                Resource::Image {
                                    image: image_view.image().clone(),
                                    subresource_range: image_view.subresource_range().clone(),
                                    memory_access,
                                    start_layout: image_layout,
                                    end_layout: image_layout,
                                },
                            ));
                        }
                    }
                }
                DescriptorBindingResources::ImageViewSampler(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some((image_view_info, _sampler)) = element {
                            let &DescriptorImageViewInfo {
                                ref image_view,
                                mut image_layout,
                            } = image_view_info;

                            if image_layout == ImageLayout::Undefined {
                                image_layout = default_image_layout;
                            }

                            let (use_ref, memory_access) = use_iter(index as u32);

                            used_resources.push((
                                use_ref,
                                Resource::Image {
                                    image: image_view.image().clone(),
                                    subresource_range: image_view.subresource_range().clone(),
                                    memory_access,
                                    start_layout: image_layout,
                                    end_layout: image_layout,
                                },
                            ));
                        }
                    }
                }
                DescriptorBindingResources::Sampler(_) => (),
                DescriptorBindingResources::InlineUniformBlock => (),
                DescriptorBindingResources::AccelerationStructure(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some(acceleration_structure) = element {
                            let buffer = acceleration_structure.buffer();
                            let (use_ref, memory_access) = use_iter(index as u32);

                            used_resources.push((
                                use_ref,
                                Resource::Buffer {
                                    buffer: buffer.clone(),
                                    range: 0..buffer.size(),
                                    memory_access,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }

    fn add_vertex_buffers_resources(
        &self,
        used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
        pipeline: &GraphicsPipeline,
    ) {
        used_resources.extend(pipeline.vertex_input_state().bindings.iter().map(
            |(&binding, _)| {
                let vertex_buffer = &self.builder_state.vertex_buffers[&binding];
                (
                    ResourceInCommand::VertexBuffer { binding }.into(),
                    Resource::Buffer {
                        buffer: vertex_buffer.clone(),
                        range: 0..vertex_buffer.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::VertexAttributeInput_VertexAttributeRead,
                    },
                )
            },
        ));
    }

    fn add_index_buffer_resources(&self, used_resources: &mut Vec<(ResourceUseRef2, Resource)>) {
        let index_buffer_bytes = self.builder_state.index_buffer.as_ref().unwrap().as_bytes();
        used_resources.push((
            ResourceInCommand::IndexBuffer.into(),
            Resource::Buffer {
                buffer: index_buffer_bytes.clone(),
                range: 0..index_buffer_bytes.size(), // TODO:
                memory_access: PipelineStageAccessFlags::IndexInput_IndexRead,
            },
        ));
    }

    fn add_indirect_buffer_resources(
        &self,
        used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
        indirect_buffer: &Subbuffer<[u8]>,
    ) {
        used_resources.push((
            ResourceInCommand::IndirectBuffer.into(),
            Resource::Buffer {
                buffer: indirect_buffer.clone(),
                range: 0..indirect_buffer.size(), // TODO:
                memory_access: PipelineStageAccessFlags::DrawIndirect_IndirectCommandRead,
            },
        ));
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    pub unsafe fn dispatch(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch(group_counts)?;

        Ok(self.dispatch_unchecked(group_counts))
    }

    fn validate_dispatch(&self, group_counts: [u32; 3]) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdDispatch-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if group_counts[0] > properties.max_compute_work_group_count[0] {
            return Err(Box::new(ValidationError {
                context: "group_counts[0]".into(),
                problem: "is greater than the `max_compute_work_group_count[0]` limit".into(),
                vuids: &["VUID-vkCmdDispatch-groupCountX-00386"],
                ..Default::default()
            }));
        }

        if group_counts[1] > properties.max_compute_work_group_count[1] {
            return Err(Box::new(ValidationError {
                context: "group_counts[1]".into(),
                problem: "is greater than the `max_compute_work_group_count[1]` limit".into(),
                vuids: &["VUID-vkCmdDispatch-groupCountY-00387"],
                ..Default::default()
            }));
        }

        if group_counts[2] > properties.max_compute_work_group_count[2] {
            return Err(Box::new(ValidationError {
                context: "group_counts[2]".into(),
                problem: "is greater than the `max_compute_work_group_count[2]` limit".into(),
                vuids: &["VUID-vkCmdDispatch-groupCountZ-00388"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_dispatch)(
            self.handle(),
            group_counts[0],
            group_counts[1],
            group_counts[2],
        );

        self
    }

    pub unsafe fn dispatch_indirect(
        &mut self,
        indirect_buffer: &Subbuffer<[DispatchIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch_indirect(indirect_buffer.as_bytes())?;

        Ok(self.dispatch_indirect_unchecked(indirect_buffer))
    }

    fn validate_dispatch_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    compute operations"
                    .into(),
                vuids: &["VUID-vkCmdDispatchIndirect-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdDispatchIndirect-commonparent
        assert_eq!(self.device(), indirect_buffer.device());

        if !indirect_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "indirect_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDispatchIndirect-buffer-02709"],
                ..Default::default()
            }));
        }

        if size_of::<DispatchIndirectCommand>() as DeviceSize > indirect_buffer.size() {
            return Err(Box::new(ValidationError {
                problem: "`size_of::<DrawIndirectCommand>()` is greater than \
                    `indirect_buffer.size()`"
                    .into(),
                vuids: &["VUID-vkCmdDispatchIndirect-offset-00407"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdDispatchIndirect-offset-02710
        // TODO:

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_indirect_unchecked(
        &mut self,
        indirect_buffer: &Subbuffer<[DispatchIndirectCommand]>,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_dispatch_indirect)(
            self.handle(),
            indirect_buffer.buffer().handle(),
            indirect_buffer.offset(),
        );

        self
    }

    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        Ok(self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance))
    }

    fn validate_draw(
        &self,
        _vertex_count: u32,
        _instance_count: u32,
        _first_vertex: u32,
        _first_instance: u32,
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
                vuids: &["VUID-vkCmdDraw-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_unchecked(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_draw)(
            self.handle(),
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );

        self
    }

    pub unsafe fn draw_indirect(
        &mut self,
        indirect_buffer: &Subbuffer<[DrawIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        Ok(self.draw_indirect_unchecked(indirect_buffer, draw_count, stride))
    }

    fn validate_draw_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
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
                vuids: &["VUID-vkCmdDrawIndirect-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !indirect_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "indirect_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndirect-buffer-02709"],
                ..Default::default()
            }));
        }

        if draw_count > 1 {
            if !self.device().enabled_features().multi_draw_indirect {
                return Err(Box::new(ValidationError {
                    context: "draw_count".into(),
                    problem: "is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_draw_indirect",
                    )])]),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-02718"],
                }));
            }

            if stride % 4 != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00476"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize) < size_of::<DrawIndirectCommand>() as DeviceSize {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride` is not greater than `size_of::<DrawIndirectCommand>()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00476"],
                    ..Default::default()
                }));
            }

            if stride as DeviceSize * (draw_count as DeviceSize - 1)
                + size_of::<DrawIndirectCommand>() as DeviceSize
                > indirect_buffer.size()
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride * (draw_count - 1) + size_of::<DrawIndirectCommand>()` is \
                        greater than `indirect_buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00488"],
                    ..Default::default()
                }));
            }
        } else {
            if size_of::<DrawIndirectCommand>() as DeviceSize > indirect_buffer.size() {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is 1, but `size_of::<DrawIndirectCommand>()` is \
                        greater than `indirect_buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndirect-drawCount-00487"],
                    ..Default::default()
                }));
            }
        }

        let properties = self.device().physical_device().properties();

        if draw_count > properties.max_draw_indirect_count {
            return Err(Box::new(ValidationError {
                context: "draw_count".into(),
                problem: "is greater than the `max_draw_indirect_count` limit".into(),
                vuids: &["VUID-vkCmdDrawIndirect-drawCount-02719"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_unchecked(
        &mut self,
        indirect_buffer: &Subbuffer<[DrawIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_draw_indirect)(
            self.handle(),
            indirect_buffer.buffer().handle(),
            indirect_buffer.offset(),
            draw_count,
            stride,
        );

        self
    }

    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        Ok(self.draw_indexed_unchecked(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        ))
    }

    fn validate_draw_indexed(
        &self,
        _index_count: u32,
        _instance_count: u32,
        _first_index: u32,
        _vertex_offset: i32,
        _first_instance: u32,
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
                vuids: &["VUID-vkCmdDrawIndexed-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_unchecked(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_draw_indexed)(
            self.handle(),
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );

        self
    }

    pub unsafe fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Subbuffer<[DrawIndexedIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        Ok(self.draw_indexed_indirect_unchecked(indirect_buffer, draw_count, stride))
    }

    fn validate_draw_indexed_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
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
                vuids: &["VUID-vkCmdDrawIndexedIndirect-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !indirect_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "indirect_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-buffer-02709"],
                ..Default::default()
            }));
        }

        if draw_count > 1 {
            if !self.device().enabled_features().multi_draw_indirect {
                return Err(Box::new(ValidationError {
                    context: "draw_count".into(),
                    problem: "is greater than 1".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_draw_indirect",
                    )])]),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-02718"],
                }));
            }

            if stride % 4 != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00528"],
                    ..Default::default()
                }));
            }

            if (stride as DeviceSize) < size_of::<DrawIndexedIndirectCommand>() as DeviceSize {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride` is not greater than `size_of::<DrawIndexedIndirectCommand>()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00528"],
                    ..Default::default()
                }));
            }

            if stride as DeviceSize * (draw_count as DeviceSize - 1)
                + size_of::<DrawIndexedIndirectCommand>() as DeviceSize
                > indirect_buffer.size()
            {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is greater than 1, but \
                        `stride * (draw_count - 1) + size_of::<DrawIndexedIndirectCommand>()` is \
                        greater than `indirect_buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00540"],
                    ..Default::default()
                }));
            }
        } else {
            if size_of::<DrawIndexedIndirectCommand>() as DeviceSize > indirect_buffer.size() {
                return Err(Box::new(ValidationError {
                    problem: "`draw_count` is 1, but `size_of::<DrawIndexedIndirectCommand>()` is \
                        greater than `indirect_buffer.size()`"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-00539"],
                    ..Default::default()
                }));
            }
        }

        let properties = self.device().physical_device().properties();

        if draw_count > properties.max_draw_indirect_count {
            return Err(Box::new(ValidationError {
                context: "draw_count".into(),
                problem: "is greater than the `max_draw_indirect_count` limit".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-drawCount-02719"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_unchecked(
        &mut self,
        indirect_buffer: &Subbuffer<[DrawIndexedIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_draw_indexed_indirect)(
            self.handle(),
            indirect_buffer.buffer().handle(),
            indirect_buffer.offset(),
            draw_count,
            stride,
        );

        self
    }
}

#[derive(Clone, Copy)]
enum VUIDType {
    Dispatch,
    DispatchIndirect,
    Draw,
    DrawIndirect,
    DrawIndexed,
    DrawIndexedIndirect,
}
