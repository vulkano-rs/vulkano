// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::iter;
use std::mem;
use std::slice;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use buffer::BufferAccess;
use buffer::TypedBufferAccess;
use command_buffer::pool::standard::StandardCommandPoolAlloc;
use command_buffer::pool::standard::StandardCommandPoolBuilder;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::synced::SyncCommandBuffer;
use command_buffer::synced::SyncCommandBufferBuilder;
use command_buffer::synced::SyncCommandBufferBuilderError;
use command_buffer::sys::Flags;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use command_buffer::sys::UnsafeCommandBufferBuilderColorImageClear;
use command_buffer::sys::UnsafeCommandBufferBuilderImageAspect;
use command_buffer::sys::UnsafeCommandBufferBuilderImageBlit;
use command_buffer::sys::UnsafeCommandBufferBuilderImageCopy;
use command_buffer::validity::*;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecError;
use command_buffer::DrawIndexedIndirectCommand;
use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use command_buffer::Kind;
use command_buffer::KindOcclusionQuery;
use command_buffer::KindSecondaryRenderPass;
use command_buffer::StateCacher;
use command_buffer::StateCacherOutcome;
use command_buffer::SubpassContents;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use format::AcceptsPixels;
use format::ClearValue;
use format::Format;
use format::FormatTy;
use framebuffer::EmptySinglePassRenderPassDesc;
use framebuffer::Framebuffer;
use framebuffer::FramebufferAbstract;
use framebuffer::LoadOp;
use framebuffer::RenderPass;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassCompatible;
use framebuffer::RenderPassDescClearValues;
use framebuffer::Subpass;
use image::ImageAccess;
use image::ImageLayout;
use instance::QueueFamily;
use pipeline::input_assembly::Index;
use pipeline::vertex::VertexSource;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use query::QueryPipelineStatisticFlags;
use sampler::Filter;
use std::ffi::CStr;
use sync::AccessCheckError;
use sync::AccessFlagBits;
use sync::GpuFuture;
use sync::PipelineStages;
use OomError;
use VulkanObject;

/// Note that command buffers allocated from the default command pool (`Arc<StandardCommandPool>`)
/// don't implement the `Send` and `Sync` traits. If you use this pool, then the
/// `AutoCommandBufferBuilder` will not implement `Send` and `Sync` either. Once a command buffer
/// is built, however, it *does* implement `Send` and `Sync`.
pub struct AutoCommandBufferBuilder<P = StandardCommandPoolBuilder> {
    inner: SyncCommandBufferBuilder<P>,
    state_cacher: StateCacher,

    // True if the queue family supports graphics operations.
    graphics_allowed: bool,

    // True if the queue family supports compute operations.
    compute_allowed: bool,

    // The kind of command buffer that we're constructing.
    kind:
        Kind<Box<dyn RenderPassAbstract + Send + Sync>, Box<dyn FramebufferAbstract + Send + Sync>>,

    // Flags passed when creating the command buffer.
    flags: Flags,

    // If we're inside a render pass, contains the render pass state.
    // Should always be None for secondary command buffers.
    render_pass_state: Option<RenderPassState>,
}

// The state of the current render pass, specifying the pass, subpass index and its intended contents.
struct RenderPassState {
    subpass: (Box<dyn RenderPassAbstract + Send + Sync>, u32),
    contents: SubpassContents,
    framebuffer: Box<dyn FramebufferAbstract + Send + Sync>,
}

impl AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
    #[inline]
    pub fn new(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(device, queue_family, Kind::primary(), Flags::None)
    }

    /// Starts building a primary command buffer.
    ///
    /// The final command buffer can only be executed once at a time. In other words, it is as if
    /// executing the command buffer modifies it.
    #[inline]
    pub fn primary(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(device, queue_family, Kind::primary(), Flags::None)
    }

    /// Starts building a primary command buffer.
    ///
    /// Contrary to `primary`, the final command buffer can only be submitted once before being
    /// destroyed. This makes it possible for the implementation to perform additional
    /// optimizations.
    #[inline]
    pub fn primary_one_time_submit(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(
            device,
            queue_family,
            Kind::primary(),
            Flags::OneTimeSubmit,
        )
    }

    /// Starts building a primary command buffer.
    ///
    /// Contrary to `primary`, the final command buffer can be executed multiple times in parallel
    /// in multiple different queues.
    #[inline]
    pub fn primary_simultaneous_use(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(
            device,
            queue_family,
            Kind::primary(),
            Flags::SimultaneousUse,
        )
    }

    /// Starts building a secondary compute command buffer.
    ///
    /// The final command buffer can only be executed once at a time. In other words, it is as if
    /// executing the command buffer modifies it.
    #[inline]
    pub fn secondary_compute(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(
            KindOcclusionQuery::Forbidden,
            QueryPipelineStatisticFlags::none(),
        );
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Starts building a secondary compute command buffer.
    ///
    /// Contrary to `secondary_compute`, the final command buffer can only be submitted once before
    /// being destroyed. This makes it possible for the implementation to perform additional
    /// optimizations.
    #[inline]
    pub fn secondary_compute_one_time_submit(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(
            KindOcclusionQuery::Forbidden,
            QueryPipelineStatisticFlags::none(),
        );
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Starts building a secondary compute command buffer.
    ///
    /// Contrary to `secondary_compute`, the final command buffer can be executed multiple times in
    /// parallel in multiple different queues.
    #[inline]
    pub fn secondary_compute_simultaneous_use(
        device: Arc<Device>,
        queue_family: QueueFamily,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(
            KindOcclusionQuery::Forbidden,
            QueryPipelineStatisticFlags::none(),
        );
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    /// Same as `secondary_compute`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_inherit_queries(
        device: Arc<Device>,
        queue_family: QueueFamily,
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(occlusion_query, query_statistics_flags);
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Same as `secondary_compute_one_time_submit`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_one_time_submit_inherit_queries(
        device: Arc<Device>,
        queue_family: QueueFamily,
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(occlusion_query, query_statistics_flags);
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Same as `secondary_compute_simultaneous_use`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_simultaneous_use_inherit_queries(
        device: Arc<Device>,
        queue_family: QueueFamily,
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(occlusion_query, query_statistics_flags);
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    /// Starts building a secondary graphics command buffer.
    ///
    /// The final command buffer can only be executed once at a time. In other words, it is as if
    /// executing the command buffer modifies it.
    #[inline]
    pub fn secondary_graphics<R>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        subpass: Subpass<R>,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>>,
            }),
            occlusion_query: KindOcclusionQuery::Forbidden,
            query_statistics_flags: QueryPipelineStatisticFlags::none(),
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Starts building a secondary graphics command buffer.
    ///
    /// Contrary to `secondary_graphics`, the final command buffer can only be submitted once
    /// before being destroyed. This makes it possible for the implementation to perform additional
    /// optimizations.
    #[inline]
    pub fn secondary_graphics_one_time_submit<R>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        subpass: Subpass<R>,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>>,
            }),
            occlusion_query: KindOcclusionQuery::Forbidden,
            query_statistics_flags: QueryPipelineStatisticFlags::none(),
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Starts building a secondary graphics command buffer.
    ///
    /// Contrary to `secondary_graphics`, the final command buffer can be executed multiple times
    /// in parallel in multiple different queues.
    #[inline]
    pub fn secondary_graphics_simultaneous_use<R>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        subpass: Subpass<R>,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>>,
            }),
            occlusion_query: KindOcclusionQuery::Forbidden,
            query_statistics_flags: QueryPipelineStatisticFlags::none(),
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    /// Same as `secondary_graphics`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_inherit_queries<R>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        subpass: Subpass<R>,
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>>,
            }),
            occlusion_query,
            query_statistics_flags,
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Same as `secondary_graphics_one_time_submit`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_one_time_submit_inherit_queries<R>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        subpass: Subpass<R>,
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>>,
            }),
            occlusion_query,
            query_statistics_flags,
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Same as `secondary_graphics_simultaneous_use`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_simultaneous_use_inherit_queries<R>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        subpass: Subpass<R>,
        occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>>,
            }),
            occlusion_query,
            query_statistics_flags,
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    // Actual constructor. Private.
    fn with_flags<R, F>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        kind: Kind<R, F>,
        flags: Flags,
    ) -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
    where
        R: RenderPassAbstract + Clone + Send + Sync + 'static,
        F: FramebufferAbstract + Clone + Send + Sync + 'static,
    {
        let new_kind = match &kind {
            Kind::Primary => Kind::Primary,
            Kind::Secondary {
                render_pass,
                occlusion_query,
                query_statistics_flags,
            } => Kind::Secondary {
                render_pass: render_pass.as_ref().map(
                    |KindSecondaryRenderPass {
                         subpass,
                         framebuffer,
                     }| {
                        KindSecondaryRenderPass {
                            subpass: Subpass::from(
                                Box::new(subpass.render_pass().clone()) as Box<_>,
                                subpass.index(),
                            )
                            .unwrap(),
                            framebuffer: framebuffer
                                .as_ref()
                                .map(|f| Box::new(f.clone()) as Box<_>),
                        }
                    },
                ),
                occlusion_query: *occlusion_query,
                query_statistics_flags: *query_statistics_flags,
            },
        };

        unsafe {
            let pool = Device::standard_command_pool(&device, queue_family);
            let inner = SyncCommandBufferBuilder::new(&pool, kind, flags);
            let state_cacher = StateCacher::new();

            let graphics_allowed = queue_family.supports_graphics();
            let compute_allowed = queue_family.supports_compute();

            Ok(AutoCommandBufferBuilder {
                inner: inner?,
                state_cacher,
                graphics_allowed,
                compute_allowed,
                render_pass_state: None,
                kind: new_kind,
                flags,
            })
        }
    }
}

impl<P> AutoCommandBufferBuilder<P> {
    #[inline]
    fn ensure_outside_render_pass(&self) -> Result<(), AutoCommandBufferBuilderContextError> {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass);
        }

        Ok(())
    }

    #[inline]
    fn ensure_inside_render_pass_secondary(
        &self,
        render_pass: &KindSecondaryRenderPass<&dyn RenderPassAbstract, &dyn FramebufferAbstract>,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
        if let Some(render_pass_state) = self.render_pass_state.as_ref() {
            if render_pass_state.contents == SubpassContents::Inline {
                return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
            }

            // Subpasses must be the same.
            if render_pass.subpass.index() != render_pass_state.subpass.1 {
                return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
            }

            // Render passes must be compatible.
            if !RenderPassCompatible::is_compatible_with(
                render_pass.subpass.render_pass(),
                &render_pass_state.subpass.0,
            ) {
                return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
            }

            // Framebuffer, if present on the secondary command buffer, must be the
            // same as the one in the current render pass.
            if let Some(framebuffer) = render_pass.framebuffer {
                if FramebufferAbstract::inner(framebuffer).internal_object()
                    != FramebufferAbstract::inner(&render_pass_state.framebuffer).internal_object()
                {
                    return Err(AutoCommandBufferBuilderContextError::IncompatibleFramebuffer);
                }
            }
        } else {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
        }

        Ok(())
    }

    #[inline]
    fn ensure_inside_render_pass_inline<Gp>(
        &self,
        pipeline: &Gp,
    ) -> Result<(), AutoCommandBufferBuilderContextError>
    where
        Gp: ?Sized + GraphicsPipelineAbstract,
    {
        match &self.kind {
            Kind::Primary => {
                if let Some(render_pass_state) = self.render_pass_state.as_ref() {
                    if render_pass_state.contents == SubpassContents::SecondaryCommandBuffers {
                        return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
                    }

                    // Subpasses must be the same.
                    if pipeline.subpass_index() != render_pass_state.subpass.1 {
                        return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
                    }

                    // Render passes must be compatible.
                    if !RenderPassCompatible::is_compatible_with(
                        pipeline,
                        &render_pass_state.subpass.0,
                    ) {
                        return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
                    }
                } else {
                    return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
                }
            }
            Kind::Secondary { render_pass, .. } => {
                if render_pass.is_none() {
                    return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
                }
            }
        }

        Ok(())
    }

    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<AutoCommandBuffer<P::Alloc>, BuildError>
    where
        P: CommandPoolBuilderAlloc,
    {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass.into());
        }

        let submit_state = match self.flags {
            Flags::None => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            Flags::SimultaneousUse => SubmitState::Concurrent,
            Flags::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        Ok(AutoCommandBuffer {
            inner: self.inner.build()?,
            kind: self.kind,
            submit_state,
        })
    }

    /// Adds a command that enters a render pass.
    ///
    /// If `secondary` is true, then you will only be able to add secondary command buffers while
    /// you're inside the first subpass of the render pass. If `secondary` is false, you will only
    /// be able to add inline draw commands and not secondary command buffers.
    ///
    /// C must contain exactly one clear value for each attachment in the framebuffer.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    pub fn begin_render_pass<F, C>(
        &mut self,
        framebuffer: F,
        contents: SubpassContents,
        clear_values: C,
    ) -> Result<&mut Self, BeginRenderPassError>
    where
        F: FramebufferAbstract + RenderPassDescClearValues<C> + Clone + Send + Sync + 'static,
    {
        unsafe {
            if let Kind::Secondary { .. } = self.kind {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary.into());
            }

            if !self.graphics_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            let clear_values = framebuffer.convert_clear_values(clear_values);
            let clear_values = clear_values.collect::<Vec<_>>().into_iter(); // TODO: necessary for Send + Sync ; needs an API rework of convert_clear_values
            let mut clear_values_copy = clear_values.clone().enumerate(); // TODO: Proper errors for clear value errors instead of panics

            for (atch_i, atch_desc) in framebuffer.attachment_descs().enumerate() {
                match clear_values_copy.next() {
                    Some((clear_i, clear_value)) => {
                        if atch_desc.load == LoadOp::Clear {
                            match clear_value {
                                ClearValue::None => panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: None",
                                    clear_i, atch_i, atch_desc.format.ty()),
                                ClearValue::Float(_) => if atch_desc.format.ty() != FormatTy::Float {
                                   panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: Float",
                                       clear_i, atch_i, atch_desc.format.ty());
                                }
                                ClearValue::Int(_) => if atch_desc.format.ty() != FormatTy::Sint {
                                    panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: Int",
                                       clear_i, atch_i, atch_desc.format.ty());
                                }
                                ClearValue::Uint(_) => if atch_desc.format.ty() != FormatTy::Uint {
                                    panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: Uint",
                                       clear_i, atch_i, atch_desc.format.ty());
                                }
                                ClearValue::Depth(_) => if atch_desc.format.ty() != FormatTy::Depth {
                                    panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: Depth",
                                       clear_i, atch_i, atch_desc.format.ty());
                                }
                                ClearValue::Stencil(_) => if atch_desc.format.ty() != FormatTy::Stencil {
                                    panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: Stencil",
                                       clear_i, atch_i, atch_desc.format.ty());
                                }
                                ClearValue::DepthStencil(_) => if atch_desc.format.ty() != FormatTy::DepthStencil {
                                    panic!("Bad ClearValue! index: {}, attachment index: {}, expected: {:?}, got: DepthStencil",
                                       clear_i, atch_i, atch_desc.format.ty());
                                }
                            }
                        } else {
                            if clear_value != ClearValue::None {
                                panic!("Bad ClearValue! index: {}, attachment index: {}, expected: None, got: {:?}",
                                   clear_i, atch_i, clear_value);
                            }
                        }
                    }
                    None => panic!("Not enough clear values"),
                }
            }

            if clear_values_copy.count() != 0 {
                panic!("Too many clear values")
            }

            self.inner
                .begin_render_pass(framebuffer.clone(), contents, clear_values)?;
            self.render_pass_state = Some(RenderPassState {
                subpass: (Box::new(framebuffer.clone()) as Box<_>, 0),
                contents,
                framebuffer: Box::new(framebuffer) as Box<_>,
            });
            Ok(self)
        }
    }

    /// Adds a command that copies an image to another.
    ///
    /// Copy operations have several restrictions:
    ///
    /// - Copy operations are only allowed on queue families that support transfer, graphics, or
    ///   compute operations.
    /// - The number of samples in the source and destination images must be equal.
    /// - The size of the uncompressed element format of the source image must be equal to the
    ///   compressed element format of the destination.
    /// - If you copy between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - For two-dimensional images, the Z coordinate must be 0 for the image offsets and 1 for
    ///   the extent. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the copy will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panic
    ///
    /// - Panics if the source or the destination was not created with `device`.
    ///
    pub fn copy_image<S, D>(
        &mut self,
        source: S,
        source_offset: [i32; 3],
        source_base_array_layer: u32,
        source_mip_level: u32,
        destination: D,
        destination_offset: [i32; 3],
        destination_base_array_layer: u32,
        destination_mip_level: u32,
        extent: [u32; 3],
        layer_count: u32,
    ) -> Result<&mut Self, CopyImageError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
    {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_image(
                self.device(),
                &source,
                source_offset,
                source_base_array_layer,
                source_mip_level,
                &destination,
                destination_offset,
                destination_base_array_layer,
                destination_mip_level,
                extent,
                layer_count,
            )?;

            let copy = UnsafeCommandBufferBuilderImageCopy {
                // TODO: Allowing choosing a subset of the image aspects, but note that if color
                // is included, neither depth nor stencil may.
                aspect: UnsafeCommandBufferBuilderImageAspect {
                    color: source.has_color(),
                    depth: !source.has_color() && source.has_depth() && destination.has_depth(),
                    stencil: !source.has_color()
                        && source.has_stencil()
                        && destination.has_stencil(),
                },
                source_mip_level,
                destination_mip_level,
                source_base_array_layer,
                destination_base_array_layer,
                layer_count,
                source_offset,
                destination_offset,
                extent,
            };

            // TODO: Allow choosing layouts, but note that only Transfer*Optimal and General are
            // valid.
            self.inner.copy_image(
                source,
                ImageLayout::TransferSrcOptimal,
                destination,
                ImageLayout::TransferDstOptimal,
                iter::once(copy),
            )?;
            Ok(self)
        }
    }

    /// Adds a command that blits an image to another.
    ///
    /// A *blit* is similar to an image copy operation, except that the portion of the image that
    /// is transferred can be resized. You choose an area of the source and an area of the
    /// destination, and the implementation will resize the area of the source so that it matches
    /// the size of the area of the destination before writing it.
    ///
    /// Blit operations have several restrictions:
    ///
    /// - Blit operations are only allowed on queue families that support graphics operations.
    /// - The format of the source and destination images must support blit operations, which
    ///   depends on the Vulkan implementation. Vulkan guarantees that some specific formats must
    ///   always be supported. See tables 52 to 61 of the specifications.
    /// - Only single-sampled images are allowed.
    /// - You can only blit between two images whose formats belong to the same type. The types
    ///   are: floating-point, signed integers, unsigned integers, depth-stencil.
    /// - If you blit between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - If you blit between depth, stencil or depth-stencil images, only the `Nearest` filter is
    ///   allowed.
    /// - For two-dimensional images, the Z coordinate must be 0 for the top-left offset and 1 for
    ///   the bottom-right offset. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the blit will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panic
    ///
    /// - Panics if the source or the destination was not created with `device`.
    ///
    pub fn blit_image<S, D>(
        &mut self,
        source: S,
        source_top_left: [i32; 3],
        source_bottom_right: [i32; 3],
        source_base_array_layer: u32,
        source_mip_level: u32,
        destination: D,
        destination_top_left: [i32; 3],
        destination_bottom_right: [i32; 3],
        destination_base_array_layer: u32,
        destination_mip_level: u32,
        layer_count: u32,
        filter: Filter,
    ) -> Result<&mut Self, BlitImageError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
    {
        unsafe {
            if !self.graphics_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            check_blit_image(
                self.device(),
                &source,
                source_top_left,
                source_bottom_right,
                source_base_array_layer,
                source_mip_level,
                &destination,
                destination_top_left,
                destination_bottom_right,
                destination_base_array_layer,
                destination_mip_level,
                layer_count,
                filter,
            )?;

            let blit = UnsafeCommandBufferBuilderImageBlit {
                // TODO:
                aspect: if source.has_color() {
                    UnsafeCommandBufferBuilderImageAspect {
                        color: true,
                        depth: false,
                        stencil: false,
                    }
                } else {
                    unimplemented!()
                },
                source_mip_level,
                destination_mip_level,
                source_base_array_layer,
                destination_base_array_layer,
                layer_count,
                source_top_left,
                source_bottom_right,
                destination_top_left,
                destination_bottom_right,
            };

            self.inner.blit_image(
                source,
                ImageLayout::TransferSrcOptimal,
                destination, // TODO: let choose layout
                ImageLayout::TransferDstOptimal,
                iter::once(blit),
                filter,
            )?;
            Ok(self)
        }
    }

    /// Adds a command that clears all the layers and mipmap levels of a color image with a
    /// specific value.
    ///
    /// # Panic
    ///
    /// Panics if `color` is not a color value.
    ///
    pub fn clear_color_image<I>(
        &mut self,
        image: I,
        color: ClearValue,
    ) -> Result<&mut Self, ClearColorImageError>
    where
        I: ImageAccess + Send + Sync + 'static,
    {
        let layers = image.dimensions().array_layers();
        let levels = image.mipmap_levels();

        self.clear_color_image_dimensions(image, 0, layers, 0, levels, color)
    }

    /// Adds a command that clears a color image with a specific value.
    ///
    /// # Panic
    ///
    /// - Panics if `color` is not a color value.
    ///
    pub fn clear_color_image_dimensions<I>(
        &mut self,
        image: I,
        first_layer: u32,
        num_layers: u32,
        first_mipmap: u32,
        num_mipmaps: u32,
        color: ClearValue,
    ) -> Result<&mut Self, ClearColorImageError>
    where
        I: ImageAccess + Send + Sync + 'static,
    {
        unsafe {
            if !self.graphics_allowed && !self.compute_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_clear_color_image(
                self.device(),
                &image,
                first_layer,
                num_layers,
                first_mipmap,
                num_mipmaps,
            )?;

            match color {
                ClearValue::Float(_) | ClearValue::Int(_) | ClearValue::Uint(_) => {}
                _ => panic!("The clear color is not a color value"),
            };

            let region = UnsafeCommandBufferBuilderColorImageClear {
                base_mip_level: first_mipmap,
                level_count: num_mipmaps,
                base_array_layer: first_layer,
                layer_count: num_layers,
            };

            // TODO: let choose layout
            self.inner.clear_color_image(
                image,
                ImageLayout::TransferDstOptimal,
                color,
                iter::once(region),
            )?;
            Ok(self)
        }
    }

    /// Adds a command that copies from a buffer to another.
    ///
    /// This command will copy from the source to the destination. If their size is not equal, then
    /// the amount of data copied is equal to the smallest of the two.
    #[inline]
    pub fn copy_buffer<S, D, T>(
        &mut self,
        source: S,
        destination: D,
    ) -> Result<&mut Self, CopyBufferError>
    where
        S: TypedBufferAccess<Content = T> + Send + Sync + 'static,
        D: TypedBufferAccess<Content = T> + Send + Sync + 'static,
        T: ?Sized,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            let infos = check_copy_buffer(self.device(), &source, &destination)?;
            self.inner
                .copy_buffer(source, destination, iter::once((0, 0, infos.copy_size)))?;
            Ok(self)
        }
    }

    /// Adds a command that copies a range from the source to the destination buffer.
    /// Panics if out of bounds.
    #[inline]
    pub fn copy_buffer_dimensions<S, D, T>(
        &mut self,
        source: S,
        source_offset: usize,
        destination: D,
        destination_offset: usize,
        count: usize,
    ) -> Result<&mut Self, CopyBufferError>
    where
        S: TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
        D: TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
    {
        self.ensure_outside_render_pass()?;

        let _infos = check_copy_buffer(self.device(), &source, &destination)?;
        debug_assert!(source_offset + count <= source.len());
        debug_assert!(destination_offset + count <= destination.len());

        let size = std::mem::size_of::<T>();
        unsafe {
            self.inner.copy_buffer(
                source,
                destination,
                iter::once((
                    source_offset * size,
                    destination_offset * size,
                    count * size,
                )),
            )?;
        }
        Ok(self)
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image<S, D, Px>(
        &mut self,
        source: S,
        destination: D,
    ) -> Result<&mut Self, CopyBufferImageError>
    where
        S: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
        Format: AcceptsPixels<Px>,
    {
        self.ensure_outside_render_pass()?;

        let dims = destination.dimensions().width_height_depth();
        self.copy_buffer_to_image_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image_dimensions<S, D, Px>(
        &mut self,
        source: S,
        destination: D,
        offset: [u32; 3],
        size: [u32; 3],
        first_layer: u32,
        num_layers: u32,
        mipmap: u32,
    ) -> Result<&mut Self, CopyBufferImageError>
    where
        S: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
        Format: AcceptsPixels<Px>,
    {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_buffer_image(
                self.device(),
                &source,
                &destination,
                CheckCopyBufferImageTy::BufferToImage,
                offset,
                size,
                first_layer,
                num_layers,
                mipmap,
            )?;

            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_aspect: if destination.has_color() {
                    UnsafeCommandBufferBuilderImageAspect {
                        color: true,
                        depth: false,
                        stencil: false,
                    }
                } else {
                    unimplemented!()
                },
                image_mip_level: mipmap,
                image_base_array_layer: first_layer,
                image_layer_count: num_layers,
                image_offset: [offset[0] as i32, offset[1] as i32, offset[2] as i32],
                image_extent: size,
            };

            self.inner.copy_buffer_to_image(
                source,
                destination,
                ImageLayout::TransferDstOptimal, // TODO: let choose layout
                iter::once(copy),
            )?;
            Ok(self)
        }
    }

    /*
    Adds a command that copies from an image to a buffer.
    The data layout of the image on the gpu is opaque, as in, it is non of our business how the gpu stores the image.
    This does not matter since the act of copying the image into a buffer converts it to linear form.
    */
    pub fn copy_image_to_buffer<S, D, Px>(
        &mut self,
        source: S,
        destination: D,
    ) -> Result<&mut Self, CopyBufferImageError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
        Format: AcceptsPixels<Px>,
    {
        self.ensure_outside_render_pass()?;

        let dims = source.dimensions().width_height_depth();
        self.copy_image_to_buffer_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from an image to a buffer.
    pub fn copy_image_to_buffer_dimensions<S, D, Px>(
        &mut self,
        source: S,
        destination: D,
        offset: [u32; 3],
        size: [u32; 3],
        first_layer: u32,
        num_layers: u32,
        mipmap: u32,
    ) -> Result<&mut Self, CopyBufferImageError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
        Format: AcceptsPixels<Px>,
    {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_buffer_image(
                self.device(),
                &destination,
                &source,
                CheckCopyBufferImageTy::ImageToBuffer,
                offset,
                size,
                first_layer,
                num_layers,
                mipmap,
            )?;

            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_aspect: UnsafeCommandBufferBuilderImageAspect {
                    color: source.has_color(),
                    depth: source.has_depth(),
                    stencil: source.has_stencil(),
                },
                image_mip_level: mipmap,
                image_base_array_layer: first_layer,
                image_layer_count: num_layers,
                image_offset: [offset[0] as i32, offset[1] as i32, offset[2] as i32],
                image_extent: size,
            };

            self.inner.copy_image_to_buffer(
                source,
                ImageLayout::TransferSrcOptimal,
                destination, // TODO: let choose layout
                iter::once(copy),
            )?;
            Ok(self)
        }
    }

    /// Open a command buffer debug label region.
    ///
    /// Note: you need to enable `VK_EXT_debug_utils` extension when creating an instance.
    #[inline]
    pub fn debug_marker_begin(
        &mut self,
        name: &'static CStr,
        color: [f32; 4],
    ) -> Result<&mut Self, DebugMarkerError> {
        if !self.graphics_allowed && self.compute_allowed {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        check_debug_marker_color(color)?;

        unsafe {
            self.inner.debug_marker_begin(name.into(), color);
        }

        Ok(self)
    }

    /// Close a command buffer label region.
    ///
    /// Note: you need to open a command buffer label region first with `debug_marker_begin`.
    /// Note: you need to enable `VK_EXT_debug_utils` extension when creating an instance.
    #[inline]
    pub fn debug_marker_end(&mut self) -> Result<&mut Self, DebugMarkerError> {
        if !self.graphics_allowed && self.compute_allowed {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        // TODO: validate that debug_marker_begin with same name was sent earlier

        unsafe {
            self.inner.debug_marker_end();
        }

        Ok(self)
    }

    /// Insert a label into a command buffer.
    ///
    /// Note: you need to enable `VK_EXT_debug_utils` extension when creating an instance.
    #[inline]
    pub fn debug_marker_insert(
        &mut self,
        name: &'static CStr,
        color: [f32; 4],
    ) -> Result<&mut Self, DebugMarkerError> {
        if !self.graphics_allowed && self.compute_allowed {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        check_debug_marker_color(color)?;

        unsafe {
            self.inner.debug_marker_insert(name.into(), color);
        }

        Ok(self)
    }

    #[inline]
    pub fn dispatch<Cp, S, Pc>(
        &mut self,
        dimensions: [u32; 3],
        pipeline: Cp,
        sets: S,
        constants: Pc,
    ) -> Result<&mut Self, DispatchError>
    where
        Cp: ComputePipelineAbstract + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
    {
        unsafe {
            if !self.compute_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            check_dispatch(pipeline.device(), dimensions)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_compute_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_compute(pipeline.clone());
            }

            push_constants(&mut self.inner, pipeline.clone(), constants);
            descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                false,
                pipeline.clone(),
                sets,
            )?;

            self.inner.dispatch(dimensions);
            Ok(self)
        }
    }

    /// Draw once, using the `vertex_buffer`.
    ///
    /// To use only some data in the buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw<V, Gp, S, Pc>(
        &mut self,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffer: V,
        sets: S,
        constants: Pc,
    ) -> Result<&mut Self, DrawError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertex_buffer)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, &dynamic);
            descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                true,
                pipeline.clone(),
                sets,
            )?;
            vertex_buffers(
                &mut self.inner,
                &mut self.state_cacher,
                vb_infos.vertex_buffers,
            )?;

            debug_assert!(self.graphics_allowed);

            self.inner.draw(
                vb_infos.vertex_count as u32,
                vb_infos.instance_count as u32,
                0,
                0,
            );
            Ok(self)
        }
    }

    /// Draw once, using the `vertex_buffer` and the `index_buffer`.
    ///
    /// To use only some data in a buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw_indexed<V, Gp, S, Pc, Ib, I>(
        &mut self,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffer: V,
        index_buffer: Ib,
        sets: S,
        constants: Pc,
    ) -> Result<&mut Self, DrawIndexedError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
        Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
        I: Index + 'static,
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            let ib_infos = check_index_buffer(self.device(), &index_buffer)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertex_buffer)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_index_buffer(&index_buffer, I::ty())
            {
                self.inner.bind_index_buffer(index_buffer, I::ty())?;
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, &dynamic);
            descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                true,
                pipeline.clone(),
                sets,
            )?;
            vertex_buffers(
                &mut self.inner,
                &mut self.state_cacher,
                vb_infos.vertex_buffers,
            )?;
            // TODO: how to handle an index out of range of the vertex buffers?

            debug_assert!(self.graphics_allowed);

            self.inner.draw_indexed(
                ib_infos.num_indices as u32,
                vb_infos.instance_count as u32,
                0,
                0,
                0,
            );
            Ok(self)
        }
    }

    /// Performs multiple draws, one draw for each `vulkano::command_buffer::DrawIndirectCommand` struct in `indirect_buffer`.
    /// The `vertex_buffer` is used by all draws.
    ///
    /// To use only some data in a buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw_indirect<V, Gp, S, Pc, Ib>(
        &mut self,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffer: V,
        indirect_buffer: Ib,
        sets: S,
        constants: Pc,
    ) -> Result<&mut Self, DrawIndirectError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
        Ib: BufferAccess
            + TypedBufferAccess<Content = [DrawIndirectCommand]>
            + Send
            + Sync
            + 'static,
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertex_buffer)?;

            let draw_count = indirect_buffer.len() as u32;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, &dynamic);
            descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                true,
                pipeline.clone(),
                sets,
            )?;
            vertex_buffers(
                &mut self.inner,
                &mut self.state_cacher,
                vb_infos.vertex_buffers,
            )?;

            debug_assert!(self.graphics_allowed);

            self.inner.draw_indirect(
                indirect_buffer,
                draw_count,
                mem::size_of::<DrawIndirectCommand>() as u32,
            )?;
            Ok(self)
        }
    }

    /// Performs multiple draws, one draw for each `vulkano::command_buffer::DrawIndexedIndirectCommand` struct in `indirect_buffer`.
    /// The `index_buffer` and `vertex_buffer` are used by all draws.
    ///
    /// To use only some data in a buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw_indexed_indirect<V, Gp, S, Pc, Ib, Inb, I>(
        &mut self,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffer: V,
        index_buffer: Ib,
        indirect_buffer: Inb,
        sets: S,
        constants: Pc,
    ) -> Result<&mut Self, DrawIndexedIndirectError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
        Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
        Inb: BufferAccess
            + TypedBufferAccess<Content = [DrawIndexedIndirectCommand]>
            + Send
            + Sync
            + 'static,
        I: Index + 'static,
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            let ib_infos = check_index_buffer(self.device(), &index_buffer)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertex_buffer)?;

            let draw_count = indirect_buffer.len() as u32;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_index_buffer(&index_buffer, I::ty())
            {
                self.inner.bind_index_buffer(index_buffer, I::ty())?;
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, &dynamic);
            descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                true,
                pipeline.clone(),
                sets,
            )?;
            vertex_buffers(
                &mut self.inner,
                &mut self.state_cacher,
                vb_infos.vertex_buffers,
            )?;

            debug_assert!(self.graphics_allowed);

            self.inner.draw_indexed_indirect(
                indirect_buffer,
                draw_count,
                mem::size_of::<DrawIndexedIndirectCommand>() as u32,
            )?;
            Ok(self)
        }
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    pub fn end_render_pass(&mut self) -> Result<&mut Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if let Kind::Secondary { .. } = self.kind {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary);
            }

            if let Some(render_pass_state) = self.render_pass_state.as_ref() {
                let (ref rp, index) = render_pass_state.subpass;

                if rp.num_subpasses() as u32 != index + 1 {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                        actual: rp.num_subpasses() as u32,
                        current: index,
                    });
                }
            } else {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
            }

            debug_assert!(self.graphics_allowed);

            self.inner.end_render_pass();
            self.render_pass_state = None;
            Ok(self)
        }
    }

    /// Adds a command that executes a secondary command buffer.
    ///
    /// # Safety
    /// The execution state of the secondary command buffer is not checked, and resources are not
    /// synchronised against concurrent access.
    pub unsafe fn execute_commands<C>(
        &mut self,
        command_buffer: C,
    ) -> Result<&mut Self, ExecuteCommandsError>
    where
        C: CommandBuffer + Send + Sync + 'static,
    {
        if let Kind::Secondary { .. } = self.kind {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary.into());
        }

        self.check_command_buffer(&command_buffer)?;

        {
            let mut builder = self.inner.execute_commands();
            builder.add(command_buffer);
            builder.submit()?;
        }

        self.state_cacher.invalidate();
        Ok(self)
    }

    /// Adds a command that multiple secondary command buffers in a vector.
    ///
    /// # Safety
    /// The execution state of the secondary command buffer is not checked, and resources are not
    /// synchronised against concurrent access.
    pub unsafe fn execute_commands_from_vec<C>(
        &mut self,
        command_buffers: Vec<C>,
    ) -> Result<&mut Self, ExecuteCommandsError>
    where
        C: CommandBuffer + Send + Sync + 'static,
    {
        if let Kind::Secondary { .. } = self.kind {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary.into());
        }

        for command_buffer in &command_buffers {
            self.check_command_buffer(command_buffer)?;
        }

        {
            let mut builder = self.inner.execute_commands();
            for command_buffer in command_buffers {
                builder.add(command_buffer);
            }
            builder.submit()?;
        }

        self.state_cacher.invalidate();
        Ok(self)
    }

    // Helper function for execute_commands
    fn check_command_buffer<C>(
        &self,
        command_buffer: &C,
    ) -> Result<(), AutoCommandBufferBuilderContextError>
    where
        C: CommandBuffer + Send + Sync + 'static,
    {
        if let Kind::Secondary { render_pass, .. } = command_buffer.kind() {
            if let Some(render_pass) = render_pass {
                // TODO: If support for queries is added, their compatibility should be checked
                // here too per vkCmdExecuteCommands specs
                self.ensure_inside_render_pass_secondary(&render_pass)?;
            } else {
                self.ensure_outside_render_pass()?;
            }
        } else {
            return Err(AutoCommandBufferBuilderContextError::NotSecondary.into());
        }

        Ok(())
    }

    /// Adds a command that writes the content of a buffer.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    ///
    /// > **Note**: This function is technically safe because buffers can only contain integers or
    /// > floating point numbers, which are always valid whatever their memory representation is.
    /// > But unless your buffer actually contains only 32-bits integers, you are encouraged to use
    /// > this function only for zeroing the content of a buffer by passing `0` for the data.
    // TODO: not safe because of signalling NaNs
    #[inline]
    pub fn fill_buffer<B>(&mut self, buffer: B, data: u32) -> Result<&mut Self, FillBufferError>
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            check_fill_buffer(self.device(), &buffer)?;
            self.inner.fill_buffer(buffer, data);
            Ok(self)
        }
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    pub fn next_subpass(
        &mut self,
        contents: SubpassContents,
    ) -> Result<&mut Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if let Kind::Secondary { .. } = self.kind {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary);
            }

            if let Some(render_pass_state) = self.render_pass_state.as_mut() {
                let (ref rp, ref mut index) = render_pass_state.subpass;

                if *index + 1 >= rp.num_subpasses() as u32 {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                        actual: rp.num_subpasses() as u32,
                        current: *index,
                    });
                } else {
                    *index += 1;
                    render_pass_state.contents = contents;
                }
            } else {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
            }

            debug_assert!(self.graphics_allowed);

            self.inner.next_subpass(contents);
            Ok(self)
        }
    }

    /// Adds a command that writes data to a buffer.
    ///
    /// If `data` is larger than the buffer, only the part of `data` that fits is written. If the
    /// buffer is larger than `data`, only the start of the buffer is written.
    // TODO: allow unsized values
    #[inline]
    pub fn update_buffer<B, D>(
        &mut self,
        buffer: B,
        data: D,
    ) -> Result<&mut Self, UpdateBufferError>
    where
        B: TypedBufferAccess<Content = D> + Send + Sync + 'static,
        D: Send + Sync + 'static,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            check_update_buffer(self.device(), &buffer, &data)?;

            let size_of_data = mem::size_of_val(&data);
            if buffer.size() >= size_of_data {
                self.inner.update_buffer(buffer, data);
            } else {
                unimplemented!() // TODO:
                                 //self.inner.update_buffer(buffer.slice(0 .. size_of_data), data);
            }

            Ok(self)
        }
    }
}

unsafe impl<P> DeviceOwned for AutoCommandBufferBuilder<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

// Shortcut function to set the push constants.
unsafe fn push_constants<P, Pl, Pc>(
    destination: &mut SyncCommandBufferBuilder<P>,
    pipeline: Pl,
    push_constants: Pc,
) where
    Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static,
{
    for num_range in 0..pipeline.num_push_constants_ranges() {
        let range = match pipeline.push_constants_range(num_range) {
            Some(r) => r,
            None => continue,
        };

        debug_assert_eq!(range.offset % 4, 0);
        debug_assert_eq!(range.size % 4, 0);

        let data = slice::from_raw_parts(
            (&push_constants as *const Pc as *const u8).offset(range.offset as isize),
            range.size as usize,
        );

        destination.push_constants::<_, [u8]>(
            pipeline.clone(),
            range.stages,
            range.offset as u32,
            range.size as u32,
            data,
        );
    }
}

// Shortcut function to change the state of the pipeline.
unsafe fn set_state<P>(destination: &mut SyncCommandBufferBuilder<P>, dynamic: &DynamicState) {
    if let Some(line_width) = dynamic.line_width {
        destination.set_line_width(line_width);
    }

    if let Some(ref viewports) = dynamic.viewports {
        destination.set_viewport(0, viewports.iter().cloned().collect::<Vec<_>>().into_iter());
        // TODO: don't collect
    }

    if let Some(ref scissors) = dynamic.scissors {
        destination.set_scissor(0, scissors.iter().cloned().collect::<Vec<_>>().into_iter());
        // TODO: don't collect
    }

    if let Some(compare_mask) = dynamic.compare_mask {
        destination.set_stencil_compare_mask(compare_mask);
    }

    if let Some(write_mask) = dynamic.write_mask {
        destination.set_stencil_write_mask(write_mask);
    }

    if let Some(reference) = dynamic.reference {
        destination.set_stencil_reference(reference);
    }
}

// Shortcut function to bind vertex buffers.
unsafe fn vertex_buffers<P>(
    destination: &mut SyncCommandBufferBuilder<P>,
    state_cacher: &mut StateCacher,
    vertex_buffers: Vec<Box<dyn BufferAccess + Send + Sync>>,
) -> Result<(), SyncCommandBufferBuilderError> {
    let binding_range = {
        let mut compare = state_cacher.bind_vertex_buffers();
        for vb in vertex_buffers.iter() {
            compare.add(vb);
        }
        match compare.compare() {
            Some(r) => r,
            None => return Ok(()),
        }
    };

    let first_binding = binding_range.start;
    let num_bindings = binding_range.end - binding_range.start;

    let mut binder = destination.bind_vertex_buffers();
    for vb in vertex_buffers
        .into_iter()
        .skip(first_binding as usize)
        .take(num_bindings as usize)
    {
        binder.add(vb);
    }
    binder.submit(first_binding)?;
    Ok(())
}

unsafe fn descriptor_sets<P, Pl, S>(
    destination: &mut SyncCommandBufferBuilder<P>,
    state_cacher: &mut StateCacher,
    gfx: bool,
    pipeline: Pl,
    sets: S,
) -> Result<(), SyncCommandBufferBuilderError>
where
    Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static,
    S: DescriptorSetsCollection,
{
    let sets = sets.into_vec();

    let first_binding = {
        let mut compare = state_cacher.bind_descriptor_sets(gfx);
        for set in sets.iter() {
            compare.add(set);
        }
        compare.compare()
    };

    let first_binding = match first_binding {
        None => return Ok(()),
        Some(fb) => fb,
    };

    let mut sets_binder = destination.bind_descriptor_sets();
    for set in sets.into_iter().skip(first_binding as usize) {
        sets_binder.add(set);
    }
    sets_binder.submit(gfx, pipeline.clone(), first_binding, iter::empty())?;
    Ok(())
}

pub struct AutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer<P>,
    kind:
        Kind<Box<dyn RenderPassAbstract + Send + Sync>, Box<dyn FramebufferAbstract + Send + Sync>>,

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

// Whether the command buffer can be submitted.
#[derive(Debug)]
enum SubmitState {
    // The command buffer was created with the "SimultaneousUse" flag. Can always be submitted at
    // any time.
    Concurrent,

    // The command buffer can only be submitted once simultaneously.
    ExclusiveUse {
        // True if the command buffer is current in use by the GPU.
        in_use: AtomicBool,
    },

    // The command buffer can only ever be submitted once.
    OneTime {
        // True if the command buffer has already been submitted once and can be no longer be
        // submitted.
        already_submitted: AtomicBool,
    },
}

unsafe impl<P> CommandBuffer for AutoCommandBuffer<P> {
    type PoolAlloc = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<P> {
        self.inner.as_ref()
    }

    #[inline]
    fn lock_submit(
        &self,
        future: &dyn GpuFuture,
        queue: &Queue,
    ) -> Result<(), CommandBufferExecError> {
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(CommandBufferExecError::OneTimeSubmitAlreadySubmitted);
                }
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(CommandBufferExecError::ExclusiveAlreadyInUse);
                }
            }
            SubmitState::Concurrent => (),
        };

        let err = match self.inner.lock_submit(future, queue) {
            Ok(()) => return Ok(()),
            Err(err) => err,
        };

        // If `self.inner.lock_submit()` failed, we revert action.
        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                already_submitted.store(false, Ordering::SeqCst);
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                in_use.store(false, Ordering::SeqCst);
            }
            SubmitState::Concurrent => (),
        };

        Err(err)
    }

    #[inline]
    unsafe fn unlock(&self) {
        // Because of panic safety, we unlock the inner command buffer first.
        self.inner.unlock();

        match self.submit_state {
            SubmitState::OneTime {
                ref already_submitted,
            } => {
                debug_assert!(already_submitted.load(Ordering::SeqCst));
            }
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::SeqCst);
                debug_assert!(old_val);
            }
            SubmitState::Concurrent => (),
        };
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &dyn BufferAccess,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        self.inner.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        self.inner
            .check_image_access(image, layout, exclusive, queue)
    }

    fn kind(&self) -> Kind<&dyn RenderPassAbstract, &dyn FramebufferAbstract> {
        match &self.kind {
            Kind::Primary => Kind::Primary,
            Kind::Secondary {
                render_pass,
                occlusion_query,
                query_statistics_flags,
            } => Kind::Secondary {
                render_pass: render_pass.as_ref().map(
                    |KindSecondaryRenderPass {
                         subpass,
                         framebuffer,
                     }| {
                        KindSecondaryRenderPass {
                            subpass: Subpass::from(
                                subpass.render_pass().as_ref() as &_,
                                subpass.index(),
                            )
                            .unwrap(),
                            framebuffer: framebuffer.as_ref().map(|f| f.as_ref() as &_),
                        }
                    },
                ),
                occlusion_query: *occlusion_query,
                query_statistics_flags: *query_statistics_flags,
            },
        }
    }
}

unsafe impl<P> DeviceOwned for AutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

macro_rules! err_gen {
    ($name:ident { $($err:ident,)+ }) => (
        #[derive(Debug, Clone)]
        pub enum $name {
            $(
                $err($err),
            )+
        }

        impl error::Error for $name {
            #[inline]
            fn cause(&self) -> Option<&dyn error::Error> {
                match *self {
                    $(
                        $name::$err(ref err) => Some(err),
                    )+
                }
            }
        }

        impl fmt::Display for $name {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
                write!(fmt, "{}", match *self {
                    $(
                        $name::$err(_) => {
                            concat!("a ", stringify!($err))
                        }
                    )+
                })
            }
        }

        $(
            impl From<$err> for $name {
                #[inline]
                fn from(err: $err) -> $name {
                    $name::$err(err)
                }
            }
        )+
    );
}

err_gen!(BuildError {
    AutoCommandBufferBuilderContextError,
    OomError,
});

err_gen!(BeginRenderPassError {
    AutoCommandBufferBuilderContextError,
    SyncCommandBufferBuilderError,
});

err_gen!(CopyImageError {
    AutoCommandBufferBuilderContextError,
    CheckCopyImageError,
    SyncCommandBufferBuilderError,
});

err_gen!(BlitImageError {
    AutoCommandBufferBuilderContextError,
    CheckBlitImageError,
    SyncCommandBufferBuilderError,
});

err_gen!(ClearColorImageError {
    AutoCommandBufferBuilderContextError,
    CheckClearColorImageError,
    SyncCommandBufferBuilderError,
});

err_gen!(CopyBufferError {
    AutoCommandBufferBuilderContextError,
    CheckCopyBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(CopyBufferImageError {
    AutoCommandBufferBuilderContextError,
    CheckCopyBufferImageError,
    SyncCommandBufferBuilderError,
});

err_gen!(FillBufferError {
    AutoCommandBufferBuilderContextError,
    CheckFillBufferError,
});

err_gen!(DebugMarkerError {
    AutoCommandBufferBuilderContextError,
    CheckColorError,
});

err_gen!(DispatchError {
    AutoCommandBufferBuilderContextError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckDispatchError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawError {
    AutoCommandBufferBuilderContextError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedError {
    AutoCommandBufferBuilderContextError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(ExecuteCommandsError {
    AutoCommandBufferBuilderContextError,
    SyncCommandBufferBuilderError,
});

err_gen!(UpdateBufferError {
    AutoCommandBufferBuilderContextError,
    CheckUpdateBufferError,
});

#[derive(Debug, Copy, Clone)]
pub enum AutoCommandBufferBuilderContextError {
    /// Operation forbidden in a secondary command buffer.
    ForbiddenInSecondary,
    /// Operation forbidden inside of a render pass.
    ForbiddenInsideRenderPass,
    /// Operation forbidden outside of a render pass.
    ForbiddenOutsideRenderPass,
    /// The provided command buffer is not a secondary command buffer.
    NotSecondary,
    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,
    /// Tried to end a render pass with subpasses remaining, or tried to go to next subpass with no
    /// subpass remaining.
    NumSubpassesMismatch {
        /// Actual number of subpasses in the current render pass.
        actual: u32,
        /// Current subpass index before the failing command.
        current: u32,
    },
    /// Tried to execute a secondary command buffer inside a subpass that only allows inline
    /// commands, or a draw command in a subpass that only allows secondary command buffers.
    WrongSubpassType,
    /// Tried to use a graphics pipeline or secondary command buffer whose subpass index
    /// didn't match the current subpass index.
    WrongSubpassIndex,
    /// Tried to use a secondary command buffer with a specified framebuffer that is
    /// incompatible with the current framebuffer.
    IncompatibleFramebuffer,
    /// Tried to use a graphics pipeline or secondary command buffer whose render pass
    /// is incompatible with the current render pass.
    IncompatibleRenderPass,
}

impl error::Error for AutoCommandBufferBuilderContextError {}

impl fmt::Display for AutoCommandBufferBuilderContextError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                AutoCommandBufferBuilderContextError::ForbiddenInSecondary => {
                    "operation forbidden in a secondary command buffer"
                }
                AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass => {
                    "operation forbidden inside of a render pass"
                }
                AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass => {
                    "operation forbidden outside of a render pass"
                }
                AutoCommandBufferBuilderContextError::NotSecondary => {
                    "tried to execute a command buffer that was not a secondary command buffer"
                }
                AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily => {
                    "the queue family doesn't allow this operation"
                }
                AutoCommandBufferBuilderContextError::NumSubpassesMismatch { .. } => {
                    "tried to end a render pass with subpasses remaining, or tried to go to next \
                 subpass with no subpass remaining"
                }
                AutoCommandBufferBuilderContextError::WrongSubpassType => {
                    "tried to execute a secondary command buffer inside a subpass that only allows \
                 inline commands, or a draw command in a subpass that only allows secondary \
                 command buffers"
                }
                AutoCommandBufferBuilderContextError::WrongSubpassIndex => {
                    "tried to use a graphics pipeline whose subpass index didn't match the current \
                 subpass index"
                }
                AutoCommandBufferBuilderContextError::IncompatibleFramebuffer => {
                    "tried to use a secondary command buffer with a specified framebuffer that is \
                 incompatible with the current framebuffer"
                }
                AutoCommandBufferBuilderContextError::IncompatibleRenderPass => {
                    "tried to use a graphics pipeline or secondary command buffer whose render pass \
                  is incompatible with the current render pass"
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::BufferUsage;
    use crate::buffer::CpuAccessibleBuffer;
    use crate::command_buffer::AutoCommandBufferBuilder;
    use crate::command_buffer::CommandBuffer;
    use crate::device::Device;
    use crate::device::DeviceExtensions;
    use crate::device::Features;
    use crate::sync::GpuFuture;
    use instance;

    #[test]
    fn copy_buffer_dimensions() {
        let instance = instance!();

        let phys = match instance::PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let queue_family = match phys.queue_families().next() {
            Some(q) => q,
            None => return,
        };

        let (device, mut queues) = Device::new(
            phys,
            &Features::none(),
            &DeviceExtensions::none(),
            std::iter::once((queue_family, 0.5)),
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [1_u32, 2].iter().copied(),
        )
        .unwrap();

        let destination = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [0_u32, 10, 20, 3, 4].iter().copied(),
        )
        .unwrap();

        let mut cbb =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap();

        cbb.copy_buffer_dimensions(source.clone(), 0, destination.clone(), 1, 2)
            .unwrap();

        let cb = cbb.build().unwrap();

        let future = cb
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = destination.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 2, 3, 4]);
    }
}
