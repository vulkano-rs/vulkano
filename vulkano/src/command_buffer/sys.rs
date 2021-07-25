// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::buffer::BufferInner;
use crate::buffer::TypedBufferAccess;
use crate::check_errors;
use crate::command_buffer::pool::UnsafeCommandPoolAlloc;
use crate::command_buffer::CommandBufferInheritance;
use crate::command_buffer::CommandBufferLevel;
use crate::command_buffer::CommandBufferUsage;
use crate::command_buffer::SecondaryCommandBuffer;
use crate::command_buffer::SubpassContents;
use crate::descriptor_set::sys::UnsafeDescriptorSet;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::format::ClearValue;
use crate::format::FormatTy;
use crate::image::ImageAccess;
use crate::image::ImageAspect;
use crate::image::ImageAspects;
use crate::image::ImageLayout;
use crate::image::SampleCount;
use crate::pipeline::depth_stencil::StencilFaceFlags;
use crate::pipeline::input_assembly::IndexType;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::shader::ShaderStages;
use crate::pipeline::viewport::Scissor;
use crate::pipeline::viewport::Viewport;
use crate::pipeline::ComputePipelineAbstract;
use crate::pipeline::GraphicsPipelineAbstract;
use crate::pipeline::PipelineBindPoint;
use crate::query::QueriesRange;
use crate::query::Query;
use crate::query::QueryControlFlags;
use crate::query::QueryResultElement;
use crate::query::QueryResultFlags;
use crate::render_pass::FramebufferAbstract;
use crate::sampler::Filter;
use crate::sync::AccessFlags;
use crate::sync::Event;
use crate::sync::PipelineStage;
use crate::sync::PipelineStages;
use crate::OomError;
use crate::VulkanObject;
use ash::vk::Handle;
use smallvec::SmallVec;
use std::ffi::CStr;
use std::fmt;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

/// Command buffer being built.
///
/// You can add commands to an `UnsafeCommandBufferBuilder` by using the `AddCommand` trait.
/// The `AddCommand<&Cmd>` trait is implemented on the `UnsafeCommandBufferBuilder` for any `Cmd`
/// that is a raw Vulkan command.
///
/// When you are finished adding commands, you can use the `CommandBufferBuild` trait to turn this
/// builder into an `UnsafeCommandBuffer`.
pub struct UnsafeCommandBufferBuilder {
    command_buffer: ash::vk::CommandBuffer,
    device: Arc<Device>,
    usage: CommandBufferUsage,
}

impl fmt::Debug for UnsafeCommandBufferBuilder {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "<Vulkan command buffer builder #{}>",
            self.command_buffer.as_raw()
        )
    }
}

impl UnsafeCommandBufferBuilder {
    /// Creates a new builder, for recording commands.
    ///
    /// # Safety
    ///
    /// - `pool_alloc` must outlive the returned builder and its created command buffer.
    /// - `kind` must match how `pool_alloc` was created.
    /// - All submitted commands must be valid and follow the requirements of the Vulkan specification.
    /// - Any resources used by submitted commands must outlive the returned builder and its created command buffer. They must be protected against data races through manual synchronization.
    ///
    /// > **Note**: Some checks are still made with `debug_assert!`. Do not expect to be able to
    /// > submit invalid commands.
    pub unsafe fn new<F>(
        pool_alloc: &UnsafeCommandPoolAlloc,
        level: CommandBufferLevel<F>,
        usage: CommandBufferUsage,
    ) -> Result<UnsafeCommandBufferBuilder, OomError>
    where
        F: FramebufferAbstract,
    {
        let secondary = match level {
            CommandBufferLevel::Primary => false,
            CommandBufferLevel::Secondary(..) => true,
        };

        let device = pool_alloc.device().clone();
        let fns = device.fns();

        let vk_flags = {
            let a = ash::vk::CommandBufferUsageFlags::from(usage);
            let b = match level {
                CommandBufferLevel::Secondary(ref inheritance)
                    if inheritance.render_pass.is_some() =>
                {
                    ash::vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE
                }
                _ => ash::vk::CommandBufferUsageFlags::empty(),
            };

            a | b
        };

        let (rp, sp, fb) = match level {
            CommandBufferLevel::Secondary(CommandBufferInheritance {
                render_pass: Some(ref render_pass),
                ..
            }) => {
                let rp = render_pass.subpass.render_pass().inner().internal_object();
                let sp = render_pass.subpass.index();
                let fb = match render_pass.framebuffer {
                    Some(ref fb) => {
                        // TODO: debug assert that the framebuffer is compatible with
                        //       the render pass?
                        FramebufferAbstract::inner(fb).internal_object()
                    }
                    None => ash::vk::Framebuffer::null(),
                };
                (rp, sp, fb)
            }
            _ => (ash::vk::RenderPass::null(), 0, ash::vk::Framebuffer::null()),
        };

        let (oqe, qf, ps) = match level {
            CommandBufferLevel::Secondary(CommandBufferInheritance {
                occlusion_query,
                query_statistics_flags,
                ..
            }) => {
                let ps: ash::vk::QueryPipelineStatisticFlags = query_statistics_flags.into();
                let (oqe, qf) = match occlusion_query {
                    Some(flags) => {
                        let qf = if flags.precise {
                            ash::vk::QueryControlFlags::PRECISE
                        } else {
                            ash::vk::QueryControlFlags::empty()
                        };
                        (ash::vk::TRUE, qf)
                    }
                    None => (0, ash::vk::QueryControlFlags::empty()),
                };

                (oqe, qf, ps)
            }
            _ => (
                0,
                ash::vk::QueryControlFlags::empty(),
                ash::vk::QueryPipelineStatisticFlags::empty(),
            ),
        };

        let inheritance = ash::vk::CommandBufferInheritanceInfo {
            render_pass: rp,
            subpass: sp,
            framebuffer: fb,
            occlusion_query_enable: oqe,
            query_flags: qf,
            pipeline_statistics: ps,
            ..Default::default()
        };

        let infos = ash::vk::CommandBufferBeginInfo {
            flags: vk_flags,
            p_inheritance_info: &inheritance,
            ..Default::default()
        };

        check_errors(
            fns.v1_0
                .begin_command_buffer(pool_alloc.internal_object(), &infos),
        )?;

        Ok(UnsafeCommandBufferBuilder {
            command_buffer: pool_alloc.internal_object(),
            device: device.clone(),
            usage,
        })
    }

    /// Turns the builder into an actual command buffer.
    #[inline]
    pub fn build(self) -> Result<UnsafeCommandBuffer, OomError> {
        unsafe {
            let fns = self.device.fns();
            check_errors(fns.v1_0.end_command_buffer(self.command_buffer))?;

            Ok(UnsafeCommandBuffer {
                command_buffer: self.command_buffer,
                device: self.device.clone(),
                usage: self.usage,
            })
        }
    }

    /// Calls `vkCmdBeginQuery` on the builder.
    #[inline]
    pub unsafe fn begin_query(&mut self, query: Query, flags: QueryControlFlags) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        let flags = if flags.precise {
            ash::vk::QueryControlFlags::PRECISE
        } else {
            ash::vk::QueryControlFlags::empty()
        };
        fns.v1_0
            .cmd_begin_query(cmd, query.pool().internal_object(), query.index(), flags);
    }

    /// Calls `vkCmdBeginRenderPass` on the builder.
    #[inline]
    pub unsafe fn begin_render_pass<F, I>(
        &mut self,
        framebuffer: &F,
        subpass_contents: SubpassContents,
        clear_values: I,
    ) where
        F: ?Sized + FramebufferAbstract,
        I: IntoIterator<Item = ClearValue>,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        // TODO: allow passing a different render pass
        let raw_render_pass = framebuffer.render_pass().inner().internal_object();
        let raw_framebuffer = framebuffer.inner().internal_object();

        let raw_clear_values: SmallVec<[_; 12]> = clear_values
            .into_iter()
            .map(|clear_value| match clear_value {
                ClearValue::None => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { float32: [0.0; 4] },
                },
                ClearValue::Float(val) => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { float32: val },
                },
                ClearValue::Int(val) => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { int32: val },
                },
                ClearValue::Uint(val) => ash::vk::ClearValue {
                    color: ash::vk::ClearColorValue { uint32: val },
                },
                ClearValue::Depth(val) => ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: val,
                        stencil: 0,
                    },
                },
                ClearValue::Stencil(val) => ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: val,
                    },
                },
                ClearValue::DepthStencil((depth, stencil)) => ash::vk::ClearValue {
                    depth_stencil: ash::vk::ClearDepthStencilValue { depth, stencil },
                },
            })
            .collect();

        // TODO: allow customizing
        let rect = [
            0..framebuffer.dimensions()[0],
            0..framebuffer.dimensions()[1],
        ];

        let begin = ash::vk::RenderPassBeginInfo {
            render_pass: raw_render_pass,
            framebuffer: raw_framebuffer,
            render_area: ash::vk::Rect2D {
                offset: ash::vk::Offset2D {
                    x: rect[0].start as i32,
                    y: rect[1].start as i32,
                },
                extent: ash::vk::Extent2D {
                    width: rect[0].end - rect[0].start,
                    height: rect[1].end - rect[1].start,
                },
            },
            clear_value_count: raw_clear_values.len() as u32,
            p_clear_values: raw_clear_values.as_ptr(),
            ..Default::default()
        };

        fns.v1_0
            .cmd_begin_render_pass(cmd, &begin, subpass_contents.into());
    }

    /// Calls `vkCmdBindDescriptorSets` on the builder.
    ///
    /// Does nothing if the list of descriptor sets is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn bind_descriptor_sets<'s, S, I>(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_binding: u32,
        sets: S,
        dynamic_offsets: I,
    ) where
        S: IntoIterator<Item = &'s UnsafeDescriptorSet>,
        I: IntoIterator<Item = u32>,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        let sets: SmallVec<[_; 12]> = sets.into_iter().map(|s| s.internal_object()).collect();
        if sets.is_empty() {
            return;
        }
        let dynamic_offsets: SmallVec<[u32; 32]> = dynamic_offsets.into_iter().collect();

        let num_bindings = sets.len() as u32;
        debug_assert!(
            first_binding + num_bindings <= pipeline_layout.descriptor_set_layouts().len() as u32
        );

        fns.v1_0.cmd_bind_descriptor_sets(
            cmd,
            pipeline_bind_point.into(),
            pipeline_layout.internal_object(),
            first_binding,
            num_bindings,
            sets.as_ptr(),
            dynamic_offsets.len() as u32,
            dynamic_offsets.as_ptr(),
        );
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer<B>(&mut self, buffer: &B, index_ty: IndexType)
    where
        B: ?Sized + BufferAccess,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().index_buffer);

        fns.v1_0.cmd_bind_index_buffer(
            cmd,
            inner.buffer.internal_object(),
            inner.offset as ash::vk::DeviceSize,
            index_ty.into(),
        );
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute<Cp>(&mut self, pipeline: &Cp)
    where
        Cp: ?Sized + ComputePipelineAbstract,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_bind_pipeline(
            cmd,
            ash::vk::PipelineBindPoint::COMPUTE,
            pipeline.inner().internal_object(),
        );
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics<Gp>(&mut self, pipeline: &Gp)
    where
        Gp: ?Sized + GraphicsPipelineAbstract,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        let inner = GraphicsPipelineAbstract::inner(pipeline).internal_object();
        fns.v1_0
            .cmd_bind_pipeline(cmd, ash::vk::PipelineBindPoint::GRAPHICS, inner);
    }

    /// Calls `vkCmdBindVertexBuffers` on the builder.
    ///
    /// Does nothing if the list of buffers is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        params: UnsafeCommandBufferBuilderBindVertexBuffer,
    ) {
        debug_assert_eq!(params.raw_buffers.len(), params.offsets.len());

        if params.raw_buffers.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();

        let num_bindings = params.raw_buffers.len() as u32;

        debug_assert!({
            let max_bindings = self
                .device()
                .physical_device()
                .properties()
                .max_vertex_input_bindings
                .unwrap();
            first_binding + num_bindings <= max_bindings
        });

        fns.v1_0.cmd_bind_vertex_buffers(
            cmd,
            first_binding,
            num_bindings,
            params.raw_buffers.as_ptr(),
            params.offsets.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image<S, D, R>(
        &mut self,
        source: &S,
        source_layout: ImageLayout,
        destination: &D,
        destination_layout: ImageLayout,
        regions: R,
    ) where
        S: ?Sized + ImageAccess,
        D: ?Sized + ImageAccess,
        R: IntoIterator<Item = UnsafeCommandBufferBuilderImageCopy>,
    {
        // TODO: The correct check here is that the uncompressed element size of the source is
        // equal to the compressed element size of the destination.
        debug_assert!(
            source.format().ty() == FormatTy::Compressed
                || destination.format().ty() == FormatTy::Compressed
                || source.format().size() == destination.format().size()
        );

        // Depth/Stencil formats are required to match exactly.
        debug_assert!(
            !matches!(
                source.format().ty(),
                FormatTy::Depth | FormatTy::Stencil | FormatTy::DepthStencil
            ) || source.format() == destination.format()
        );

        debug_assert_eq!(source.samples(), destination.samples());
        let source = source.inner();
        debug_assert!(source.image.usage().transfer_source);
        debug_assert!(
            source_layout == ImageLayout::General
                || source_layout == ImageLayout::TransferSrcOptimal
        );

        let destination = destination.inner();
        debug_assert!(destination.image.usage().transfer_destination);
        debug_assert!(
            destination_layout == ImageLayout::General
                || destination_layout == ImageLayout::TransferDstOptimal
        );

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|copy| {
                // TODO: not everything is checked here
                debug_assert!(
                    copy.source_base_array_layer + copy.layer_count <= source.num_layers as u32
                );
                debug_assert!(
                    copy.destination_base_array_layer + copy.layer_count
                        <= destination.num_layers as u32
                );
                debug_assert!(copy.source_mip_level < destination.num_mipmap_levels as u32);
                debug_assert!(copy.destination_mip_level < destination.num_mipmap_levels as u32);

                if copy.layer_count == 0 {
                    return None;
                }

                Some(ash::vk::ImageCopy {
                    src_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.aspects.into(),
                        mip_level: copy.source_mip_level,
                        base_array_layer: copy.source_base_array_layer + source.first_layer as u32,
                        layer_count: copy.layer_count,
                    },
                    src_offset: ash::vk::Offset3D {
                        x: copy.source_offset[0],
                        y: copy.source_offset[1],
                        z: copy.source_offset[2],
                    },
                    dst_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.aspects.into(),
                        mip_level: copy.destination_mip_level,
                        base_array_layer: copy.destination_base_array_layer
                            + destination.first_layer as u32,
                        layer_count: copy.layer_count,
                    },
                    dst_offset: ash::vk::Offset3D {
                        x: copy.destination_offset[0],
                        y: copy.destination_offset[1],
                        z: copy.destination_offset[2],
                    },
                    extent: ash::vk::Extent3D {
                        width: copy.extent[0],
                        height: copy.extent[1],
                        depth: copy.extent[2],
                    },
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_copy_image(
            cmd,
            source.image.internal_object(),
            source_layout.into(),
            destination.image.internal_object(),
            destination_layout.into(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image<S, D, R>(
        &mut self,
        source: &S,
        source_layout: ImageLayout,
        destination: &D,
        destination_layout: ImageLayout,
        regions: R,
        filter: Filter,
    ) where
        S: ?Sized + ImageAccess,
        D: ?Sized + ImageAccess,
        R: IntoIterator<Item = UnsafeCommandBufferBuilderImageBlit>,
    {
        debug_assert!(
            filter == Filter::Nearest
                || !matches!(
                    source.format().ty(),
                    FormatTy::Depth | FormatTy::Stencil | FormatTy::DepthStencil
                )
        );
        debug_assert!(
            (source.format().ty() == FormatTy::Uint)
                == (destination.format().ty() == FormatTy::Uint)
        );
        debug_assert!(
            (source.format().ty() == FormatTy::Sint)
                == (destination.format().ty() == FormatTy::Sint)
        );
        debug_assert!(
            source.format() == destination.format()
                || !matches!(
                    source.format().ty(),
                    FormatTy::Depth | FormatTy::Stencil | FormatTy::DepthStencil
                )
        );

        debug_assert_eq!(source.samples(), SampleCount::Sample1);
        let source = source.inner();
        debug_assert!(source.image.format_features().blit_src);
        debug_assert!(source.image.usage().transfer_source);
        debug_assert!(
            source_layout == ImageLayout::General
                || source_layout == ImageLayout::TransferSrcOptimal
        );

        debug_assert_eq!(destination.samples(), SampleCount::Sample1);
        let destination = destination.inner();
        debug_assert!(destination.image.format_features().blit_dst);
        debug_assert!(destination.image.usage().transfer_destination);
        debug_assert!(
            destination_layout == ImageLayout::General
                || destination_layout == ImageLayout::TransferDstOptimal
        );

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|blit| {
                // TODO: not everything is checked here
                debug_assert!(
                    blit.source_base_array_layer + blit.layer_count <= source.num_layers as u32
                );
                debug_assert!(
                    blit.destination_base_array_layer + blit.layer_count
                        <= destination.num_layers as u32
                );
                debug_assert!(blit.source_mip_level < destination.num_mipmap_levels as u32);
                debug_assert!(blit.destination_mip_level < destination.num_mipmap_levels as u32);

                if blit.layer_count == 0 {
                    return None;
                }

                Some(ash::vk::ImageBlit {
                    src_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: blit.aspects.into(),
                        mip_level: blit.source_mip_level,
                        base_array_layer: blit.source_base_array_layer + source.first_layer as u32,
                        layer_count: blit.layer_count,
                    },
                    src_offsets: [
                        ash::vk::Offset3D {
                            x: blit.source_top_left[0],
                            y: blit.source_top_left[1],
                            z: blit.source_top_left[2],
                        },
                        ash::vk::Offset3D {
                            x: blit.source_bottom_right[0],
                            y: blit.source_bottom_right[1],
                            z: blit.source_bottom_right[2],
                        },
                    ],
                    dst_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: blit.aspects.into(),
                        mip_level: blit.destination_mip_level,
                        base_array_layer: blit.destination_base_array_layer
                            + destination.first_layer as u32,
                        layer_count: blit.layer_count,
                    },
                    dst_offsets: [
                        ash::vk::Offset3D {
                            x: blit.destination_top_left[0],
                            y: blit.destination_top_left[1],
                            z: blit.destination_top_left[2],
                        },
                        ash::vk::Offset3D {
                            x: blit.destination_bottom_right[0],
                            y: blit.destination_bottom_right[1],
                            z: blit.destination_bottom_right[2],
                        },
                    ],
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_blit_image(
            cmd,
            source.image.internal_object(),
            source_layout.into(),
            destination.image.internal_object(),
            destination_layout.into(),
            regions.len() as u32,
            regions.as_ptr(),
            filter.into(),
        );
    }

    // TODO: missing structs
    /*/// Calls `vkCmdClearAttachments` on the builder.
    ///
    /// Does nothing if the list of attachments or the list of rects is empty, as it would be a
    /// no-op and isn't a valid usage of the command anyway.
    #[inline]
    pub unsafe fn clear_attachments<A, R>(&mut self, attachments: A, rects: R)
        where A: IntoIterator<Item = >,
              R: IntoIterator<Item = >
    {
        let attachments: SmallVec<[_; 16]> = attachments.map().collect();
        let rects: SmallVec<[_; 4]> = rects.map().collect();

        if attachments.is_empty() || rects.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.CmdClearAttachments(cmd, attachments.len() as u32, attachments.as_ptr(),
                               rects.len() as u32, rects.as_ptr());
    }*/

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    // TODO: ClearValue could be more precise
    pub unsafe fn clear_color_image<I, R>(
        &mut self,
        image: &I,
        layout: ImageLayout,
        color: ClearValue,
        regions: R,
    ) where
        I: ?Sized + ImageAccess,
        R: IntoIterator<Item = UnsafeCommandBufferBuilderColorImageClear>,
    {
        debug_assert!(
            image.format().ty() == FormatTy::Float
                || image.format().ty() == FormatTy::Uint
                || image.format().ty() == FormatTy::Sint
        );

        let image = image.inner();
        debug_assert!(image.image.usage().transfer_destination);
        debug_assert!(layout == ImageLayout::General || layout == ImageLayout::TransferDstOptimal);

        let color = match color {
            ClearValue::Float(val) => ash::vk::ClearColorValue { float32: val },
            ClearValue::Int(val) => ash::vk::ClearColorValue { int32: val },
            ClearValue::Uint(val) => ash::vk::ClearColorValue { uint32: val },
            _ => ash::vk::ClearColorValue { float32: [0.0; 4] },
        };

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|region| {
                debug_assert!(
                    region.layer_count + region.base_array_layer <= image.num_layers as u32
                );
                debug_assert!(
                    region.level_count + region.base_mip_level <= image.num_mipmap_levels as u32
                );

                if region.layer_count == 0 || region.level_count == 0 {
                    return None;
                }

                Some(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_mip_level: region.base_mip_level + image.first_mipmap_level as u32,
                    level_count: region.level_count,
                    base_array_layer: region.base_array_layer + image.first_layer as u32,
                    layer_count: region.layer_count,
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_clear_color_image(
            cmd,
            image.image.internal_object(),
            layout.into(),
            &color,
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer<S, D, R>(&mut self, source: &S, destination: &D, regions: R)
    where
        S: ?Sized + BufferAccess,
        D: ?Sized + BufferAccess,
        R: IntoIterator<Item = (usize, usize, usize)>,
    {
        // TODO: debug assert that there's no overlap in the destinations?

        let source = source.inner();
        debug_assert!(source.offset < source.buffer.size());
        debug_assert!(source.buffer.usage().transfer_source);

        let destination = destination.inner();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage().transfer_destination);

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .map(|(sr, de, sz)| ash::vk::BufferCopy {
                src_offset: (sr + source.offset) as ash::vk::DeviceSize,
                dst_offset: (de + destination.offset) as ash::vk::DeviceSize,
                size: sz as ash::vk::DeviceSize,
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_copy_buffer(
            cmd,
            source.buffer.internal_object(),
            destination.buffer.internal_object(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image<S, D, R>(
        &mut self,
        source: &S,
        destination: &D,
        destination_layout: ImageLayout,
        regions: R,
    ) where
        S: ?Sized + BufferAccess,
        D: ?Sized + ImageAccess,
        R: IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>,
    {
        let source = source.inner();
        debug_assert!(source.offset < source.buffer.size());
        debug_assert!(source.buffer.usage().transfer_source);

        debug_assert_eq!(destination.samples(), SampleCount::Sample1);
        let destination = destination.inner();
        debug_assert!(destination.image.usage().transfer_destination);
        debug_assert!(
            destination_layout == ImageLayout::General
                || destination_layout == ImageLayout::TransferDstOptimal
        );

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .map(|copy| {
                debug_assert!(copy.image_layer_count <= destination.num_layers as u32);
                debug_assert!(copy.image_mip_level < destination.num_mipmap_levels as u32);

                ash::vk::BufferImageCopy {
                    buffer_offset: (source.offset + copy.buffer_offset) as ash::vk::DeviceSize,
                    buffer_row_length: copy.buffer_row_length,
                    buffer_image_height: copy.buffer_image_height,
                    image_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.image_aspect.into(),
                        mip_level: copy.image_mip_level + destination.first_mipmap_level as u32,
                        base_array_layer: copy.image_base_array_layer
                            + destination.first_layer as u32,
                        layer_count: copy.image_layer_count,
                    },
                    image_offset: ash::vk::Offset3D {
                        x: copy.image_offset[0],
                        y: copy.image_offset[1],
                        z: copy.image_offset[2],
                    },
                    image_extent: ash::vk::Extent3D {
                        width: copy.image_extent[0],
                        height: copy.image_extent[1],
                        depth: copy.image_extent[2],
                    },
                }
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_copy_buffer_to_image(
            cmd,
            source.buffer.internal_object(),
            destination.image.internal_object(),
            destination_layout.into(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer<S, D, R>(
        &mut self,
        source: &S,
        source_layout: ImageLayout,
        destination: &D,
        regions: R,
    ) where
        S: ?Sized + ImageAccess,
        D: ?Sized + BufferAccess,
        R: IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>,
    {
        debug_assert_eq!(source.samples(), SampleCount::Sample1);
        let source = source.inner();
        debug_assert!(source.image.usage().transfer_source);
        debug_assert!(
            source_layout == ImageLayout::General
                || source_layout == ImageLayout::TransferSrcOptimal
        );

        let destination = destination.inner();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage().transfer_destination);

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .map(|copy| {
                debug_assert!(copy.image_layer_count <= source.num_layers as u32);
                debug_assert!(copy.image_mip_level < source.num_mipmap_levels as u32);

                ash::vk::BufferImageCopy {
                    buffer_offset: (destination.offset + copy.buffer_offset) as ash::vk::DeviceSize,
                    buffer_row_length: copy.buffer_row_length,
                    buffer_image_height: copy.buffer_image_height,
                    image_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.image_aspect.into(),
                        mip_level: copy.image_mip_level + source.first_mipmap_level as u32,
                        base_array_layer: copy.image_base_array_layer + source.first_layer as u32,
                        layer_count: copy.image_layer_count,
                    },
                    image_offset: ash::vk::Offset3D {
                        x: copy.image_offset[0],
                        y: copy.image_offset[1],
                        z: copy.image_offset[2],
                    },
                    image_extent: ash::vk::Extent3D {
                        width: copy.image_extent[0],
                        height: copy.image_extent[1],
                        depth: copy.image_extent[2],
                    },
                }
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_copy_image_to_buffer(
            cmd,
            source.image.internal_object(),
            source_layout.into(),
            destination.buffer.internal_object(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyQueryPoolResults` on the builder.
    #[inline]
    pub unsafe fn copy_query_pool_results<D, T>(
        &mut self,
        queries: QueriesRange,
        destination: D,
        stride: usize,
        flags: QueryResultFlags,
    ) where
        D: BufferAccess + TypedBufferAccess<Content = [T]>,
        T: QueryResultElement,
    {
        let destination = destination.inner();
        let range = queries.range();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage().transfer_destination);
        debug_assert!(destination.offset % std::mem::size_of::<T>() == 0);
        debug_assert!(stride % std::mem::size_of::<T>() == 0);

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_copy_query_pool_results(
            cmd,
            queries.pool().internal_object(),
            range.start,
            range.end - range.start,
            destination.buffer.internal_object(),
            destination.offset as ash::vk::DeviceSize,
            stride as ash::vk::DeviceSize,
            ash::vk::QueryResultFlags::from(flags) | T::FLAG,
        );
    }

    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(&mut self, group_counts: [u32; 3]) {
        debug_assert!({
            let max_group_counts = self
                .device()
                .physical_device()
                .properties()
                .max_compute_work_group_count
                .unwrap();
            group_counts[0] <= max_group_counts[0]
                && group_counts[1] <= max_group_counts[1]
                && group_counts[2] <= max_group_counts[2]
        });

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_dispatch(cmd, group_counts[0], group_counts[1], group_counts[2]);
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect<B>(&mut self, buffer: &B)
    where
        B: ?Sized + BufferAccess,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().indirect_buffer);
        debug_assert_eq!(inner.offset % 4, 0);

        fns.v1_0.cmd_dispatch_indirect(
            cmd,
            inner.buffer.internal_object(),
            inner.offset as ash::vk::DeviceSize,
        );
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_draw(
            cmd,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_draw_indexed(
            cmd,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect<B>(&mut self, buffer: &B, draw_count: u32, stride: u32)
    where
        B: ?Sized + BufferAccess,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        debug_assert!(
            draw_count == 0
                || ((stride % 4) == 0)
                    && stride as usize >= mem::size_of::<ash::vk::DrawIndirectCommand>()
        );

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().indirect_buffer);

        fns.v1_0.cmd_draw_indirect(
            cmd,
            inner.buffer.internal_object(),
            inner.offset as ash::vk::DeviceSize,
            draw_count,
            stride,
        );
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect<B>(&mut self, buffer: &B, draw_count: u32, stride: u32)
    where
        B: ?Sized + BufferAccess,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage().indirect_buffer);

        fns.v1_0.cmd_draw_indexed_indirect(
            cmd,
            inner.buffer.internal_object(),
            inner.offset as ash::vk::DeviceSize,
            draw_count,
            stride,
        );
    }

    /// Calls `vkCmdEndQuery` on the builder.
    #[inline]
    pub unsafe fn end_query(&mut self, query: Query) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_end_query(cmd, query.pool().internal_object(), query.index());
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_end_render_pass(cmd);
    }

    /// Calls `vkCmdExecuteCommands` on the builder.
    ///
    /// Does nothing if the list of command buffers is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn execute_commands(&mut self, cbs: UnsafeCommandBufferBuilderExecuteCommands) {
        if cbs.raw_cbs.is_empty() {
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_execute_commands(cmd, cbs.raw_cbs.len() as u32, cbs.raw_cbs.as_ptr());
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer<B>(&mut self, buffer: &B, data: u32)
    where
        B: ?Sized + BufferAccess,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        let size = buffer.size();

        let (buffer_handle, offset) = {
            let BufferInner {
                buffer: buffer_inner,
                offset,
            } = buffer.inner();
            debug_assert!(buffer_inner.usage().transfer_destination);
            debug_assert_eq!(offset % 4, 0);
            (buffer_inner.internal_object(), offset)
        };

        fns.v1_0.cmd_fill_buffer(
            cmd,
            buffer_handle,
            offset as ash::vk::DeviceSize,
            size as ash::vk::DeviceSize,
            data,
        );
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_next_subpass(cmd, subpass_contents.into());
    }

    #[inline]
    pub unsafe fn pipeline_barrier(&mut self, command: &UnsafeCommandBufferBuilderPipelineBarrier) {
        // If barrier is empty, don't do anything.
        if command.src_stage_mask.is_empty() || command.dst_stage_mask.is_empty() {
            debug_assert!(command.src_stage_mask.is_empty() && command.dst_stage_mask.is_empty());
            debug_assert!(command.memory_barriers.is_empty());
            debug_assert!(command.buffer_barriers.is_empty());
            debug_assert!(command.image_barriers.is_empty());
            return;
        }

        let fns = self.device().fns();
        let cmd = self.internal_object();

        debug_assert!(!command.src_stage_mask.is_empty());
        debug_assert!(!command.dst_stage_mask.is_empty());

        fns.v1_0.cmd_pipeline_barrier(
            cmd,
            command.src_stage_mask,
            command.dst_stage_mask,
            command.dependency_flags,
            command.memory_barriers.len() as u32,
            command.memory_barriers.as_ptr(),
            command.buffer_barriers.len() as u32,
            command.buffer_barriers.as_ptr(),
            command.image_barriers.len() as u32,
            command.image_barriers.as_ptr(),
        );
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
    pub unsafe fn push_constants<D>(
        &mut self,
        pipeline_layout: &PipelineLayout,
        stages: ShaderStages,
        offset: u32,
        size: u32,
        data: &D,
    ) where
        D: ?Sized,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        debug_assert!(stages != ShaderStages::none());
        debug_assert!(size > 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert_eq!(offset % 4, 0);
        debug_assert!(mem::size_of_val(data) >= size as usize);

        fns.v1_0.cmd_push_constants(
            cmd,
            pipeline_layout.internal_object(),
            stages.into(),
            offset as u32,
            size as u32,
            data as *const D as *const _,
        );
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: &Event, stages: PipelineStages) {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        debug_assert!(!stages.host);
        debug_assert_ne!(stages, PipelineStages::none());

        fns.v1_0
            .cmd_reset_event(cmd, event.internal_object(), stages.into());
    }

    /// Calls `vkCmdResetQueryPool` on the builder.
    #[inline]
    pub unsafe fn reset_query_pool(&mut self, queries: QueriesRange) {
        let range = queries.range();
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_reset_query_pool(
            cmd,
            queries.pool().internal_object(),
            range.start,
            range.end - range.start,
        );
    }

    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_set_blend_constants(cmd, &constants);
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        debug_assert!(clamp == 0.0 || self.device().enabled_features().depth_bias_clamp);
        fns.v1_0
            .cmd_set_depth_bias(cmd, constant_factor, clamp, slope_factor);
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        debug_assert!(min >= 0.0 && min <= 1.0);
        debug_assert!(max >= 0.0 && max <= 1.0);
        fns.v1_0.cmd_set_depth_bounds(cmd, min, max);
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: &Event, stages: PipelineStages) {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        debug_assert!(!stages.host);
        debug_assert_ne!(stages, PipelineStages::none());

        fns.v1_0
            .cmd_set_event(cmd, event.internal_object(), stages.into());
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        debug_assert!(line_width == 1.0 || self.device().enabled_features().wide_lines);
        fns.v1_0.cmd_set_line_width(cmd, line_width);
    }

    /// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(
        &mut self,
        face_mask: StencilFaceFlags,
        compare_mask: u32,
    ) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_set_stencil_compare_mask(cmd, face_mask.into(), compare_mask);
    }

    /// Calls `vkCmdSetStencilReference` on the builder.
    #[inline]
    pub unsafe fn set_stencil_reference(&mut self, face_mask: StencilFaceFlags, reference: u32) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_set_stencil_reference(cmd, face_mask.into(), reference);
    }

    /// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(&mut self, face_mask: StencilFaceFlags, write_mask: u32) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_set_stencil_write_mask(cmd, face_mask.into(), write_mask);
    }

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
    where
        I: IntoIterator<Item = Scissor>,
    {
        let scissors = scissors
            .into_iter()
            .map(|v| ash::vk::Rect2D::from(v.clone()))
            .collect::<SmallVec<[_; 16]>>();
        if scissors.is_empty() {
            return;
        }

        debug_assert!(scissors.iter().all(|s| s.offset.x >= 0 && s.offset.y >= 0));
        debug_assert!(scissors.iter().all(|s| {
            s.extent.width < i32::max_value() as u32
                && s.extent.height < i32::max_value() as u32
                && s.offset.x.checked_add(s.extent.width as i32).is_some()
                && s.offset.y.checked_add(s.extent.height as i32).is_some()
        }));
        debug_assert!(
            (first_scissor == 0 && scissors.len() == 1)
                || self.device().enabled_features().multi_viewport
        );
        debug_assert!({
            let max = self
                .device()
                .physical_device()
                .properties()
                .max_viewports
                .unwrap();
            first_scissor + scissors.len() as u32 <= max
        });

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0
            .cmd_set_scissor(cmd, first_scissor, scissors.len() as u32, scissors.as_ptr());
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
    where
        I: IntoIterator<Item = Viewport>,
    {
        let viewports = viewports
            .into_iter()
            .map(|v| v.clone().into())
            .collect::<SmallVec<[_; 16]>>();
        if viewports.is_empty() {
            return;
        }

        debug_assert!(
            (first_viewport == 0 && viewports.len() == 1)
                || self.device().enabled_features().multi_viewport
        );
        debug_assert!({
            let max = self
                .device()
                .physical_device()
                .properties()
                .max_viewports
                .unwrap();
            first_viewport + viewports.len() as u32 <= max
        });

        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_set_viewport(
            cmd,
            first_viewport,
            viewports.len() as u32,
            viewports.as_ptr(),
        );
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<B, D>(&mut self, buffer: &B, data: &D)
    where
        B: ?Sized + BufferAccess,
        D: ?Sized,
    {
        let fns = self.device().fns();
        let cmd = self.internal_object();

        let size = buffer.size();
        debug_assert_eq!(size % 4, 0);
        debug_assert!(size <= 65536);
        debug_assert!(size <= mem::size_of_val(data));

        let (buffer_handle, offset) = {
            let BufferInner {
                buffer: buffer_inner,
                offset,
            } = buffer.inner();
            debug_assert!(buffer_inner.usage().transfer_destination);
            debug_assert_eq!(offset % 4, 0);
            (buffer_inner.internal_object(), offset)
        };

        fns.v1_0.cmd_update_buffer(
            cmd,
            buffer_handle,
            offset as ash::vk::DeviceSize,
            size as ash::vk::DeviceSize,
            data as *const D as *const _,
        );
    }

    /// Calls `vkCmdWriteTimestamp` on the builder.
    #[inline]
    pub unsafe fn write_timestamp(&mut self, query: Query, stage: PipelineStage) {
        let fns = self.device().fns();
        let cmd = self.internal_object();
        fns.v1_0.cmd_write_timestamp(
            cmd,
            stage.into(),
            query.pool().internal_object(),
            query.index(),
        );
    }

    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn debug_marker_begin(&mut self, name: &CStr, color: [f32; 4]) {
        let fns = self.device().instance().fns();
        let cmd = self.internal_object();
        let info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: name.as_ptr(),
            color,
            ..Default::default()
        };
        fns.ext_debug_utils
            .cmd_begin_debug_utils_label_ext(cmd, &info);
    }

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// There must be an outstanding `vkCmdBeginDebugUtilsLabelEXT` command prior to the
    /// `vkQueueEndDebugUtilsLabelEXT` on the queue tha `CommandBuffer` is submitted to.
    #[inline]
    pub unsafe fn debug_marker_end(&mut self) {
        let fns = self.device().instance().fns();
        let cmd = self.internal_object();
        fns.ext_debug_utils.cmd_end_debug_utils_label_ext(cmd);
    }

    /// Calls `vkCmdInsertDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn debug_marker_insert(&mut self, name: &CStr, color: [f32; 4]) {
        let fns = self.device().instance().fns();
        let cmd = self.internal_object();
        let info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: name.as_ptr(),
            color,
            ..Default::default()
        };
        fns.ext_debug_utils
            .cmd_insert_debug_utils_label_ext(cmd, &info);
    }
}

unsafe impl DeviceOwned for UnsafeCommandBufferBuilder {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for UnsafeCommandBufferBuilder {
    type Object = ash::vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct UnsafeCommandBufferBuilderBindVertexBuffer {
    // Raw handles of the buffers to bind.
    raw_buffers: SmallVec<[ash::vk::Buffer; 4]>,
    // Raw offsets of the buffers to bind.
    offsets: SmallVec<[ash::vk::DeviceSize; 4]>,
}

impl UnsafeCommandBufferBuilderBindVertexBuffer {
    /// Builds a new empty list.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderBindVertexBuffer {
        UnsafeCommandBufferBuilderBindVertexBuffer {
            raw_buffers: SmallVec::new(),
            offsets: SmallVec::new(),
        }
    }

    /// Adds a buffer to the list.
    #[inline]
    pub fn add<B>(&mut self, buffer: &B)
    where
        B: ?Sized + BufferAccess,
    {
        let inner = buffer.inner();
        debug_assert!(inner.buffer.usage().vertex_buffer);
        self.raw_buffers.push(inner.buffer.internal_object());
        self.offsets.push(inner.offset as ash::vk::DeviceSize);
    }
}

/// Prototype for a `vkCmdExecuteCommands`.
pub struct UnsafeCommandBufferBuilderExecuteCommands {
    // Raw handles of the command buffers to execute.
    raw_cbs: SmallVec<[ash::vk::CommandBuffer; 4]>,
}

impl UnsafeCommandBufferBuilderExecuteCommands {
    /// Builds a new empty list.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderExecuteCommands {
        UnsafeCommandBufferBuilderExecuteCommands {
            raw_cbs: SmallVec::new(),
        }
    }

    /// Adds a command buffer to the list.
    #[inline]
    pub fn add<C>(&mut self, cb: &C)
    where
        C: ?Sized + SecondaryCommandBuffer,
    {
        // TODO: debug assert that it is a secondary command buffer?
        self.raw_cbs.push(cb.inner().internal_object());
    }

    /// Adds a command buffer to the list.
    #[inline]
    pub unsafe fn add_raw(&mut self, cb: ash::vk::CommandBuffer) {
        self.raw_cbs.push(cb);
    }
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderColorImageClear {
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderBufferImageCopy {
    pub buffer_offset: usize,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_aspect: ImageAspect,
    pub image_mip_level: u32,
    pub image_base_array_layer: u32,
    pub image_layer_count: u32,
    pub image_offset: [i32; 3],
    pub image_extent: [u32; 3],
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageCopy {
    pub aspects: ImageAspects,
    pub source_mip_level: u32,
    pub destination_mip_level: u32,
    pub source_base_array_layer: u32,
    pub destination_base_array_layer: u32,
    pub layer_count: u32,
    pub source_offset: [i32; 3],
    pub destination_offset: [i32; 3],
    pub extent: [u32; 3],
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageBlit {
    pub aspects: ImageAspects,
    pub source_mip_level: u32,
    pub destination_mip_level: u32,
    pub source_base_array_layer: u32,
    pub destination_base_array_layer: u32,
    pub layer_count: u32,
    pub source_top_left: [i32; 3],
    pub source_bottom_right: [i32; 3],
    pub destination_top_left: [i32; 3],
    pub destination_bottom_right: [i32; 3],
}

/// Command that adds a pipeline barrier to a command buffer builder.
///
/// A pipeline barrier is a low-level system-ish command that is often necessary for safety. By
/// default all commands that you add to a command buffer can potentially run simultaneously.
/// Adding a pipeline barrier separates commands before the barrier from commands after the barrier
/// and prevents them from running simultaneously.
///
/// Please take a look at the Vulkan specifications for more information. Pipeline barriers are a
/// complex topic and explaining them in this documentation would be redundant.
///
/// > **Note**: We use a builder-like API here so that users can pass multiple buffers or images of
/// > multiple different types. Doing so with a single function would be very tedious in terms of
/// > API.
pub struct UnsafeCommandBufferBuilderPipelineBarrier {
    src_stage_mask: ash::vk::PipelineStageFlags,
    dst_stage_mask: ash::vk::PipelineStageFlags,
    dependency_flags: ash::vk::DependencyFlags,
    memory_barriers: SmallVec<[ash::vk::MemoryBarrier; 2]>,
    buffer_barriers: SmallVec<[ash::vk::BufferMemoryBarrier; 8]>,
    image_barriers: SmallVec<[ash::vk::ImageMemoryBarrier; 8]>,
}

impl UnsafeCommandBufferBuilderPipelineBarrier {
    /// Creates a new empty pipeline barrier command.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderPipelineBarrier {
        UnsafeCommandBufferBuilderPipelineBarrier {
            src_stage_mask: ash::vk::PipelineStageFlags::empty(),
            dst_stage_mask: ash::vk::PipelineStageFlags::empty(),
            dependency_flags: ash::vk::DependencyFlags::BY_REGION,
            memory_barriers: SmallVec::new(),
            buffer_barriers: SmallVec::new(),
            image_barriers: SmallVec::new(),
        }
    }

    /// Returns true if no barrier or execution dependency has been added yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.src_stage_mask.is_empty() || self.dst_stage_mask.is_empty()
    }

    /// Merges another pipeline builder into this one.
    #[inline]
    pub fn merge(&mut self, other: UnsafeCommandBufferBuilderPipelineBarrier) {
        self.src_stage_mask |= other.src_stage_mask;
        self.dst_stage_mask |= other.dst_stage_mask;
        self.dependency_flags &= other.dependency_flags;

        self.memory_barriers
            .extend(other.memory_barriers.into_iter());
        self.buffer_barriers
            .extend(other.buffer_barriers.into_iter());
        self.image_barriers.extend(other.image_barriers.into_iter());
    }

    /// Adds an execution dependency. This means that all the stages in `source` of the previous
    /// commands must finish before any of the stages in `destination` of the following commands can start.
    ///
    /// # Safety
    ///
    /// - If the pipeline stages include geometry or tessellation stages, then the corresponding
    ///   features must have been enabled in the device.
    /// - There are certain rules regarding the pipeline barriers inside render passes.
    ///
    #[inline]
    pub unsafe fn add_execution_dependency(
        &mut self,
        source: PipelineStages,
        destination: PipelineStages,
        by_region: bool,
    ) {
        if !by_region {
            self.dependency_flags = ash::vk::DependencyFlags::empty();
        }

        debug_assert_ne!(source, PipelineStages::none());
        debug_assert_ne!(destination, PipelineStages::none());

        self.src_stage_mask |= ash::vk::PipelineStageFlags::from(source);
        self.dst_stage_mask |= ash::vk::PipelineStageFlags::from(destination);
    }

    /// Adds a memory barrier. This means that all the memory writes by the given source stages
    /// for the given source accesses must be visible by the given destination stages for the given
    /// destination accesses.
    ///
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    ///
    pub unsafe fn add_memory_barrier(
        &mut self,
        source_stage: PipelineStages,
        source_access: AccessFlags,
        destination_stage: PipelineStages,
        destination_access: AccessFlags,
        by_region: bool,
    ) {
        debug_assert!(source_access.is_compatible_with(&source_stage));
        debug_assert!(destination_access.is_compatible_with(&destination_stage));

        self.add_execution_dependency(source_stage, destination_stage, by_region);

        self.memory_barriers.push(ash::vk::MemoryBarrier {
            src_access_mask: source_access.into(),
            dst_access_mask: destination_access.into(),
            ..Default::default()
        });
    }

    /// Adds a buffer memory barrier. This means that all the memory writes to the given buffer by
    /// the given source stages for the given source accesses must be visible by the given dest
    /// stages for the given destination accesses.
    ///
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// Also allows transferring buffer ownership between queues.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    ///
    pub unsafe fn add_buffer_memory_barrier<B>(
        &mut self,
        buffer: &B,
        source_stage: PipelineStages,
        source_access: AccessFlags,
        destination_stage: PipelineStages,
        destination_access: AccessFlags,
        by_region: bool,
        queue_transfer: Option<(u32, u32)>,
        offset: usize,
        size: usize,
    ) where
        B: ?Sized + BufferAccess,
    {
        debug_assert!(source_access.is_compatible_with(&source_stage));
        debug_assert!(destination_access.is_compatible_with(&destination_stage));

        self.add_execution_dependency(source_stage, destination_stage, by_region);

        debug_assert!(size <= buffer.size());
        let BufferInner {
            buffer,
            offset: org_offset,
        } = buffer.inner();
        let offset = offset + org_offset;

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            (src_queue, dest_queue)
        } else {
            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED)
        };

        self.buffer_barriers.push(ash::vk::BufferMemoryBarrier {
            src_access_mask: source_access.into(),
            dst_access_mask: destination_access.into(),
            src_queue_family_index: src_queue,
            dst_queue_family_index: dest_queue,
            buffer: buffer.internal_object(),
            offset: offset as ash::vk::DeviceSize,
            size: size as ash::vk::DeviceSize,
            ..Default::default()
        });
    }

    /// Adds an image memory barrier. This is the equivalent of `add_buffer_memory_barrier` but
    /// for images.
    ///
    /// In addition to transferring image ownership between queues, it also allows changing the
    /// layout of images.
    ///
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    /// - Image layouts transfers must be correct.
    /// - Access flags must be compatible with the image usage flags passed at image creation.
    ///
    pub unsafe fn add_image_memory_barrier<I>(
        &mut self,
        image: &I,
        mipmaps: Range<u32>,
        layers: Range<u32>,
        source_stage: PipelineStages,
        source_access: AccessFlags,
        destination_stage: PipelineStages,
        destination_access: AccessFlags,
        by_region: bool,
        queue_transfer: Option<(u32, u32)>,
        current_layout: ImageLayout,
        new_layout: ImageLayout,
    ) where
        I: ?Sized + ImageAccess,
    {
        debug_assert!(source_access.is_compatible_with(&source_stage));
        debug_assert!(destination_access.is_compatible_with(&destination_stage));

        self.add_execution_dependency(source_stage, destination_stage, by_region);

        debug_assert_ne!(new_layout, ImageLayout::Undefined);
        debug_assert_ne!(new_layout, ImageLayout::Preinitialized);

        debug_assert!(mipmaps.start < mipmaps.end);
        debug_assert!(mipmaps.end <= image.mipmap_levels());
        debug_assert!(layers.start < layers.end);
        debug_assert!(layers.end <= image.dimensions().array_layers());

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            (src_queue, dest_queue)
        } else {
            (ash::vk::QUEUE_FAMILY_IGNORED, ash::vk::QUEUE_FAMILY_IGNORED)
        };

        if image.format().ty() == FormatTy::Ycbcr {
            unimplemented!();
        }

        // TODO: Let user choose
        let aspects = image.format().aspects();
        let image = image.inner();

        self.image_barriers.push(ash::vk::ImageMemoryBarrier {
            src_access_mask: source_access.into(),
            dst_access_mask: destination_access.into(),
            old_layout: current_layout.into(),
            new_layout: new_layout.into(),
            src_queue_family_index: src_queue,
            dst_queue_family_index: dest_queue,
            image: image.image.internal_object(),
            subresource_range: ash::vk::ImageSubresourceRange {
                aspect_mask: aspects.into(),
                base_mip_level: mipmaps.start + image.first_mipmap_level as u32,
                level_count: mipmaps.end - mipmaps.start,
                base_array_layer: layers.start + image.first_layer as u32,
                layer_count: layers.end - layers.start,
            },
            ..Default::default()
        });
    }
}

/// Command buffer that has been built.
///
/// # Safety
///
/// The command buffer must not outlive the command pool that it was created from,
/// nor the resources used by the recorded commands.
pub struct UnsafeCommandBuffer {
    command_buffer: ash::vk::CommandBuffer,
    device: Arc<Device>,
    usage: CommandBufferUsage,
}

impl UnsafeCommandBuffer {
    #[inline]
    pub fn usage(&self) -> CommandBufferUsage {
        self.usage
    }
}

unsafe impl DeviceOwned for UnsafeCommandBuffer {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for UnsafeCommandBuffer {
    type Object = ash::vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> ash::vk::CommandBuffer {
        self.command_buffer
    }
}
