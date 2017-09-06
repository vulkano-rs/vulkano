// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::iter;
use std::mem;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use OomError;
use buffer::BufferAccess;
use buffer::TypedBufferAccess;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecError;
use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use command_buffer::StateCacher;
use command_buffer::StateCacherOutcome;
use command_buffer::pool::CommandPoolBuilderAlloc;
use command_buffer::pool::standard::StandardCommandPoolAlloc;
use command_buffer::pool::standard::StandardCommandPoolBuilder;
use command_buffer::synced::SyncCommandBuffer;
use command_buffer::synced::SyncCommandBufferBuilder;
use command_buffer::synced::SyncCommandBufferBuilderError;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use command_buffer::sys::KindOcclusionQuery;
use command_buffer::sys::KindSecondaryRenderPass;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use command_buffer::sys::UnsafeCommandBufferBuilderColorImageClear;
use command_buffer::sys::UnsafeCommandBufferBuilderImageAspect;
use command_buffer::sys::UnsafeCommandBufferBuilderImageBlit;
use command_buffer::validity::*;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use device::Queue;
use format::AcceptsPixels;
use format::ClearValue;
use format::Format;
use framebuffer::EmptySinglePassRenderPassDesc;
use framebuffer::Framebuffer;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPass;
use framebuffer::RenderPassAbstract;
use framebuffer::RenderPassCompatible;
use framebuffer::RenderPassDescClearValues;
use framebuffer::Subpass;
use framebuffer::SubpassContents;
use image::ImageAccess;
use image::ImageLayout;
use instance::QueueFamily;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::input_assembly::Index;
use pipeline::vertex::VertexSource;
use query::QueryPipelineStatisticFlags;
use sampler::Filter;
use sync::AccessCheckError;
use sync::AccessFlagBits;
use sync::GpuFuture;
use sync::PipelineStages;

///
///
/// Note that command buffers allocated from the default command pool (`Arc<StandardCommandPool>`)
/// don't implement the `Send` and `Sync` traits. If you use this pool, then the
/// `AutoCommandBufferBuilder` will not implement `Send` and `Sync` either. Once a command buffer
/// is built, however, it *does* implement `Send` and `Sync`.
///
pub struct AutoCommandBufferBuilder<P = StandardCommandPoolBuilder> {
    inner: SyncCommandBufferBuilder<P>,
    state_cacher: StateCacher,

    // True if the queue family supports graphics operations.
    graphics_allowed: bool,

    // True if the queue family supports compute operations.
    compute_allowed: bool,

    // If we're inside a render pass, contains the render pass and the subpass index.
    render_pass: Option<(Box<RenderPassAbstract>, u32)>,

    // True if we are a secondary command buffer.
    secondary_cb: bool,

    // True if we're in a subpass that only allows executing secondary command buffers. False if
    // we're in a subpass that only allows inline commands. Irrelevant if not in a subpass.
    subpass_secondary: bool,

    // Flags passed when creating the command buffer.
    flags: Flags,
}

impl AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
    #[inline]
    pub fn new(device: Arc<Device>, queue_family: QueueFamily)
               -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(device, queue_family, Kind::primary(), Flags::None)
    }

    /// Starts building a primary command buffer.
    ///
    /// The final command buffer can only be executed once at a time. In other words, it is as if
    /// executing the command buffer modifies it.
    #[inline]
    pub fn primary(device: Arc<Device>, queue_family: QueueFamily)
                   -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(device, queue_family, Kind::primary(), Flags::None)
    }

    /// Starts building a primary command buffer.
    ///
    /// Contrary to `primary`, the final command buffer can only be submitted once before being
    /// destroyed. This makes it possible for the implementation to perform additional
    /// optimizations.
    #[inline]
    pub fn primary_one_time_submit(
        device: Arc<Device>, queue_family: QueueFamily)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(device,
                                             queue_family,
                                             Kind::primary(),
                                             Flags::OneTimeSubmit)
    }

    /// Starts building a primary command buffer.
    ///
    /// Contrary to `primary`, the final command buffer can be executed multiple times in parallel
    /// in multiple different queues.
    #[inline]
    pub fn primary_simultaneous_use(
        device: Arc<Device>, queue_family: QueueFamily)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        AutoCommandBufferBuilder::with_flags(device,
                                             queue_family,
                                             Kind::primary(),
                                             Flags::SimultaneousUse)
    }

    /// Starts building a secondary compute command buffer.
    ///
    /// The final command buffer can only be executed once at a time. In other words, it is as if
    /// executing the command buffer modifies it.
    #[inline]
    pub fn secondary_compute(
        device: Arc<Device>, queue_family: QueueFamily)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(KindOcclusionQuery::Forbidden,
                                   QueryPipelineStatisticFlags::none());
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Starts building a secondary compute command buffer.
    ///
    /// Contrary to `secondary_compute`, the final command buffer can only be submitted once before
    /// being destroyed. This makes it possible for the implementation to perform additional
    /// optimizations.
    #[inline]
    pub fn secondary_compute_one_time_submit(
        device: Arc<Device>, queue_family: QueueFamily)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(KindOcclusionQuery::Forbidden,
                                   QueryPipelineStatisticFlags::none());
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Starts building a secondary compute command buffer.
    ///
    /// Contrary to `secondary_compute`, the final command buffer can be executed multiple times in
    /// parallel in multiple different queues.
    #[inline]
    pub fn secondary_compute_simultaneous_use(
        device: Arc<Device>, queue_family: QueueFamily)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(KindOcclusionQuery::Forbidden,
                                   QueryPipelineStatisticFlags::none());
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    /// Same as `secondary_compute`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_inherit_queries(
        device: Arc<Device>, queue_family: QueueFamily, occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(occlusion_query, query_statistics_flags);
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Same as `secondary_compute_one_time_submit`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_one_time_submit_inherit_queries(
        device: Arc<Device>, queue_family: QueueFamily, occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(occlusion_query, query_statistics_flags);
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Same as `secondary_compute_simultaneous_use`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_simultaneous_use_inherit_queries(
        device: Arc<Device>, queue_family: QueueFamily, occlusion_query: KindOcclusionQuery,
        query_statistics_flags: QueryPipelineStatisticFlags)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError> {
        let kind = Kind::secondary(occlusion_query, query_statistics_flags);
        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    /// Starts building a secondary graphics command buffer.
    ///
    /// The final command buffer can only be executed once at a time. In other words, it is as if
    /// executing the command buffer modifies it.
    #[inline]
    pub fn secondary_graphics<R>(
        device: Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                                  subpass,
                                  framebuffer:
                                      None::<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>,
                                                         ()>>,
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
        device: Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                                  subpass,
                                  framebuffer:
                                      None::<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>,
                                                         ()>>,
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
        device: Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                                  subpass,
                                  framebuffer:
                                      None::<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>,
                                                         ()>>,
                              }),
            occlusion_query: KindOcclusionQuery::Forbidden,
            query_statistics_flags: QueryPipelineStatisticFlags::none(),
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    /// Same as `secondary_graphics`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_inherit_queries<R>(
        device: Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>,
        occlusion_query: KindOcclusionQuery, query_statistics_flags: QueryPipelineStatisticFlags)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                                  subpass,
                                  framebuffer:
                                      None::<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>,
                                                         ()>>,
                              }),
            occlusion_query,
            query_statistics_flags,
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::None)
    }

    /// Same as `secondary_graphics_one_time_submit`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_one_time_submit_inherit_queries<R>(
        device: Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>,
        occlusion_query: KindOcclusionQuery, query_statistics_flags: QueryPipelineStatisticFlags)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                                  subpass,
                                  framebuffer:
                                      None::<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>,
                                                         ()>>,
                              }),
            occlusion_query,
            query_statistics_flags,
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::OneTimeSubmit)
    }

    /// Same as `secondary_graphics_simultaneous_use`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_simultaneous_use_inherit_queries<R>(
        device: Arc<Device>, queue_family: QueueFamily, subpass: Subpass<R>,
        occlusion_query: KindOcclusionQuery, query_statistics_flags: QueryPipelineStatisticFlags)
        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        let kind = Kind::Secondary {
            render_pass: Some(KindSecondaryRenderPass {
                                  subpass,
                                  framebuffer:
                                      None::<Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>,
                                                         ()>>,
                              }),
            occlusion_query,
            query_statistics_flags,
        };

        AutoCommandBufferBuilder::with_flags(device, queue_family, kind, Flags::SimultaneousUse)
    }

    // Actual constructor. Private.
    fn with_flags<R, F>(device: Arc<Device>, queue_family: QueueFamily, kind: Kind<R, F>,
                        flags: Flags)
                        -> Result<AutoCommandBufferBuilder<StandardCommandPoolBuilder>, OomError>
        where R: RenderPassAbstract + Clone + Send + Sync + 'static,
              F: FramebufferAbstract
    {
        unsafe {
            let (secondary_cb, render_pass) = match kind {
                Kind::Primary => (false, None),
                Kind::Secondary { render_pass: Some(ref sec), .. } => {
                    let render_pass = sec.subpass.render_pass().clone();
                    let index = sec.subpass.index();
                    (true, Some((Box::new(render_pass) as Box<_>, index)))
                },
                Kind::Secondary { render_pass: None, .. } => (true, None),
            };

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
                   render_pass,
                   secondary_cb,
                   subpass_secondary: false,
                   flags,
               })
        }
    }
}

impl<P> AutoCommandBufferBuilder<P> {
    #[inline]
    fn ensure_outside_render_pass(&self) -> Result<(), AutoCommandBufferBuilderContextError> {
        if self.render_pass.is_none() {
            Ok(())
        } else {
            Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass)
        }
    }

    #[inline]
    fn ensure_inside_render_pass_secondary(&self)
                                           -> Result<(), AutoCommandBufferBuilderContextError> {
        if self.render_pass.is_some() {
            if self.subpass_secondary {
                Ok(())
            } else {
                Err(AutoCommandBufferBuilderContextError::WrongSubpassType)
            }
        } else {
            Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass)
        }
    }

    #[inline]
    fn ensure_inside_render_pass_inline<Gp>(&self, pipeline: &Gp)
                                            -> Result<(), AutoCommandBufferBuilderContextError>
        where Gp: ?Sized + GraphicsPipelineAbstract
    {
        if self.render_pass.is_none() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
        }

        if self.subpass_secondary {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
        }

        let local_render_pass = self.render_pass.as_ref().unwrap();

        if pipeline.subpass_index() != local_render_pass.1 {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
        }

        if !RenderPassCompatible::is_compatible_with(pipeline, &local_render_pass.0) {
            return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
        }

        Ok(())
    }

    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<AutoCommandBuffer<P::Alloc>, BuildError>
        where P: CommandPoolBuilderAlloc
    {
        if !self.secondary_cb && self.render_pass.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass.into());
        }

        let submit_state = match self.flags {
            Flags::None => {
                SubmitState::ExclusiveUse { in_use: AtomicBool::new(false) }
            },
            Flags::SimultaneousUse => {
                SubmitState::Concurrent
            },
            Flags::OneTimeSubmit => {
                SubmitState::OneTime { already_submitted: AtomicBool::new(false) }
            },
        };

        Ok(AutoCommandBuffer {
               inner: self.inner.build()?,
               submit_state,
           })
    }

    /// Adds a command that enters a render pass.
    ///
    /// If `secondary` is true, then you will only be able to add secondary command buffers while
    /// you're inside the first subpass of the render pass. If `secondary` is false, you will only
    /// be able to add inline draw commands and not secondary command buffers.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    pub fn begin_render_pass<F, C>(mut self, framebuffer: F, secondary: bool, clear_values: C)
                                   -> Result<Self, BeginRenderPassError>
        where F: FramebufferAbstract + RenderPassDescClearValues<C> + Clone + Send + Sync + 'static
    {
        unsafe {
            if self.secondary_cb {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary.into());
            }

            if !self.graphics_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            let clear_values = framebuffer.convert_clear_values(clear_values);
            let clear_values = clear_values.collect::<Vec<_>>().into_iter(); // TODO: necessary for Send + Sync ; needs an API rework of convert_clear_values
            let contents = if secondary {
                SubpassContents::SecondaryCommandBuffers
            } else {
                SubpassContents::Inline
            };
            self.inner
                .begin_render_pass(framebuffer.clone(), contents, clear_values)?;
            self.render_pass = Some((Box::new(framebuffer) as Box<_>, 0));
            self.subpass_secondary = secondary;
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
    /// If `layer_count` is superior to 1, the blit will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panic
    ///
    /// - Panics if the source or the destination was not created with `device`.
    ///
    pub fn blit_image<S, D>(mut self, source: S, source_top_left: [i32; 3],
                            source_bottom_right: [i32; 3], source_base_array_layer: u32,
                            source_mip_level: u32, destination: D, destination_top_left: [i32; 3],
                            destination_bottom_right: [i32; 3], destination_base_array_layer: u32,
                            destination_mip_level: u32, layer_count: u32, filter: Filter)
                            -> Result<Self, BlitImageError>
        where S: ImageAccess + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static
    {
        unsafe {
            if !self.graphics_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            check_blit_image(self.device(),
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
                             filter)?;

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

            self.inner
                .blit_image(source,
                            ImageLayout::TransferSrcOptimal,
                            destination, // TODO: let choose layout
                            ImageLayout::TransferDstOptimal,
                            iter::once(blit),
                            filter)?;
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
    pub fn clear_color_image<I>(self, image: I, color: ClearValue)
                                -> Result<Self, ClearColorImageError>
        where I: ImageAccess + Send + Sync + 'static
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
    pub fn clear_color_image_dimensions<I>(mut self, image: I, first_layer: u32, num_layers: u32,
                                           first_mipmap: u32, num_mipmaps: u32, color: ClearValue)
                                           -> Result<Self, ClearColorImageError>
        where I: ImageAccess + Send + Sync + 'static
    {
        unsafe {
            if !self.graphics_allowed && !self.compute_allowed {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_clear_color_image(self.device(),
                                    &image,
                                    first_layer,
                                    num_layers,
                                    first_mipmap,
                                    num_mipmaps)?;

            match color {
                ClearValue::Float(_) |
                ClearValue::Int(_) |
                ClearValue::Uint(_) => {},
                _ => panic!("The clear color is not a color value"),
            };

            let region = UnsafeCommandBufferBuilderColorImageClear {
                base_mip_level: first_mipmap,
                level_count: num_mipmaps,
                base_array_layer: first_layer,
                layer_count: num_layers,
            };

            // TODO: let choose layout
            self.inner
                .clear_color_image(image,
                                   ImageLayout::TransferDstOptimal,
                                   color,
                                   iter::once(region))?;
            Ok(self)
        }
    }

    /// Adds a command that copies from a buffer to another.
    ///
    /// This command will copy from the source to the destination. If their size is not equal, then
    /// the amount of data copied is equal to the smallest of the two.
    #[inline]
    pub fn copy_buffer<S, D, T>(mut self, source: S, destination: D)
                                -> Result<Self, CopyBufferError>
        where S: TypedBufferAccess<Content = T> + Send + Sync + 'static,
              D: TypedBufferAccess<Content = T> + Send + Sync + 'static,
              T: ?Sized
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            let infos = check_copy_buffer(self.device(), &source, &destination)?;
            self.inner
                .copy_buffer(source, destination, iter::once((0, 0, infos.copy_size)))?;
            Ok(self)
        }
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image<S, D, Px>(self, source: S, destination: D)
                                          -> Result<Self, CopyBufferImageError>
        where S: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
              Format: AcceptsPixels<Px>
    {
        self.ensure_outside_render_pass()?;

        let dims = destination.dimensions().width_height_depth();
        self.copy_buffer_to_image_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image_dimensions<S, D, Px>(mut self, source: S, destination: D,
                                                     offset: [u32; 3], size: [u32; 3],
                                                     first_layer: u32, num_layers: u32, mipmap: u32)
                                                     -> Result<Self, CopyBufferImageError>
        where S: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
              D: ImageAccess + Send + Sync + 'static,
              Format: AcceptsPixels<Px>
    {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_buffer_image(self.device(),
                                    &source,
                                    &destination,
                                    CheckCopyBufferImageTy::BufferToImage,
                                    offset,
                                    size,
                                    first_layer,
                                    num_layers,
                                    mipmap)?;

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

            self.inner
                .copy_buffer_to_image(source,
                                      destination,
                                      ImageLayout::TransferDstOptimal, // TODO: let choose layout
                                      iter::once(copy))?;
            Ok(self)
        }
    }

    /// Adds a command that copies from an image to a buffer.
    pub fn copy_image_to_buffer<S, D, Px>(self, source: S, destination: D)
                                          -> Result<Self, CopyBufferImageError>
        where S: ImageAccess + Send + Sync + 'static,
              D: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
              Format: AcceptsPixels<Px>
    {
        self.ensure_outside_render_pass()?;

        let dims = source.dimensions().width_height_depth();
        self.copy_image_to_buffer_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from an image to a buffer.
    pub fn copy_image_to_buffer_dimensions<S, D, Px>(mut self, source: S, destination: D,
                                                     offset: [u32; 3], size: [u32; 3],
                                                     first_layer: u32, num_layers: u32, mipmap: u32)
                                                     -> Result<Self, CopyBufferImageError>
        where S: ImageAccess + Send + Sync + 'static,
              D: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
              Format: AcceptsPixels<Px>
    {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_buffer_image(self.device(),
                                    &destination,
                                    &source,
                                    CheckCopyBufferImageTy::ImageToBuffer,
                                    offset,
                                    size,
                                    first_layer,
                                    num_layers,
                                    mipmap)?;

            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_aspect: if source.has_color() {
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

            self.inner
                .copy_image_to_buffer(source,
                                      ImageLayout::TransferSrcOptimal,
                                      destination, // TODO: let choose layout
                                      iter::once(copy))?;
            Ok(self)
        }
    }

    #[inline]
    pub fn dispatch<Cp, S, Pc>(mut self, dimensions: [u32; 3], pipeline: Cp, sets: S, constants: Pc)
                               -> Result<Self, DispatchError>
        where Cp: ComputePipelineAbstract + Send + Sync + 'static + Clone, // TODO: meh for Clone
              S: DescriptorSetsCollection
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
            descriptor_sets(&mut self.inner,
                            &mut self.state_cacher,
                            false,
                            pipeline.clone(),
                            sets)?;

            self.inner.dispatch(dimensions);
            Ok(self)
        }
    }

    #[inline]
    pub fn draw<V, Gp, S, Pc>(mut self, pipeline: Gp, dynamic: DynamicState, vertices: V, sets: S,
                              constants: Pc)
                              -> Result<Self, DrawError>
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
              S: DescriptorSetsCollection
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_dynamic_state_validity(&pipeline, &dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertices)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, dynamic);
            descriptor_sets(&mut self.inner,
                            &mut self.state_cacher,
                            true,
                            pipeline.clone(),
                            sets)?;
            vertex_buffers(&mut self.inner,
                           &mut self.state_cacher,
                           vb_infos.vertex_buffers)?;

            debug_assert!(self.graphics_allowed);

            self.inner.draw(vb_infos.vertex_count as u32,
                            vb_infos.instance_count as u32,
                            0,
                            0);
            Ok(self)
        }
    }

    #[inline]
    pub fn draw_indexed<V, Gp, S, Pc, Ib, I>(mut self, pipeline: Gp, dynamic: DynamicState,
                                             vertices: V, index_buffer: Ib, sets: S, constants: Pc)
                                             -> Result<Self, DrawIndexedError>
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
              S: DescriptorSetsCollection,
              Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
              I: Index + 'static
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            let ib_infos = check_index_buffer(self.device(), &index_buffer)?;
            check_dynamic_state_validity(&pipeline, &dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertices)?;

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
            set_state(&mut self.inner, dynamic);
            descriptor_sets(&mut self.inner,
                            &mut self.state_cacher,
                            true,
                            pipeline.clone(),
                            sets)?;
            vertex_buffers(&mut self.inner,
                           &mut self.state_cacher,
                           vb_infos.vertex_buffers)?;
            // TODO: how to handle an index out of range of the vertex buffers?

            debug_assert!(self.graphics_allowed);

            self.inner
                .draw_indexed(ib_infos.num_indices as u32, 1, 0, 0, 0);
            Ok(self)
        }
    }

    #[inline]
    pub fn draw_indirect<V, Gp, S, Pc, Ib>(mut self, pipeline: Gp, dynamic: DynamicState,
                                           vertices: V, indirect_buffer: Ib, sets: S, constants: Pc)
                                           -> Result<Self, DrawIndirectError>
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
              S: DescriptorSetsCollection,
              Ib: BufferAccess
                      + TypedBufferAccess<Content = [DrawIndirectCommand]>
                      + Send
                      + Sync
                      + 'static
    {
        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_dynamic_state_validity(&pipeline, &dynamic)?;
            check_push_constants_validity(&pipeline, &constants)?;
            check_descriptor_sets_validity(&pipeline, &sets)?;
            let vb_infos = check_vertex_buffers(&pipeline, vertices)?;

            let draw_count = indirect_buffer.len() as u32;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            push_constants(&mut self.inner, pipeline.clone(), constants);
            set_state(&mut self.inner, dynamic);
            descriptor_sets(&mut self.inner,
                            &mut self.state_cacher,
                            true,
                            pipeline.clone(),
                            sets)?;
            vertex_buffers(&mut self.inner,
                           &mut self.state_cacher,
                           vb_infos.vertex_buffers)?;

            debug_assert!(self.graphics_allowed);

            self.inner
                .draw_indirect(indirect_buffer,
                               draw_count,
                               mem::size_of::<DrawIndirectCommand>() as u32)?;
            Ok(self)
        }
    }

    /// Adds a command that ends the current render pass.
    ///
    /// This must be called after you went through all the subpasses and before you can build
    /// the command buffer or add further commands.
    #[inline]
    pub fn end_render_pass(mut self) -> Result<Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if self.secondary_cb {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary);
            }

            match self.render_pass {
                Some((ref rp, index)) if rp.num_subpasses() as u32 == index + 1 => (),
                None => {
                    return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
                },
                Some((ref rp, index)) => {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                                   actual: rp.num_subpasses() as u32,
                                   current: index,
                               });
                },
            }

            debug_assert!(self.graphics_allowed);

            self.inner.end_render_pass();
            self.render_pass = None;
            Ok(self)
        }
    }

    /// Adds a command that executes a secondary command buffer.
    ///
    /// **This function is unsafe for now because safety checks and synchronization are not
    /// implemented.**
    // TODO: implement correctly
    pub unsafe fn execute_commands<C>(mut self, command_buffer: C)
                                      -> Result<Self, ExecuteCommandsError>
        where C: CommandBuffer + Send + Sync + 'static
    {
        {
            let mut builder = self.inner.execute_commands();
            builder.add(command_buffer);
            builder.submit()?;
        }

        self.state_cacher.invalidate();

        Ok(self)
    }

    /// Adds a command that writes the content of a buffer.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatidely written through the entire buffer.
    ///
    /// > **Note**: This function is technically safe because buffers can only contain integers or
    /// > floating point numbers, which are always valid whatever their memory representation is.
    /// > But unless your buffer actually contains only 32-bits integers, you are encouraged to use
    /// > this function only for zeroing the content of a buffer by passing `0` for the data.
    // TODO: not safe because of signalling NaNs
    #[inline]
    pub fn fill_buffer<B>(mut self, buffer: B, data: u32) -> Result<Self, FillBufferError>
        where B: BufferAccess + Send + Sync + 'static
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
    pub fn next_subpass(mut self, secondary: bool)
                        -> Result<Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if self.secondary_cb {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenInSecondary);
            }

            match self.render_pass {
                None => {
                    return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
                },
                Some((ref rp, ref mut index)) => {
                    if *index + 1 >= rp.num_subpasses() as u32 {
                        return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                                       actual: rp.num_subpasses() as u32,
                                       current: *index,
                                   });
                    } else {
                        *index += 1;
                    }
                },
            };

            self.subpass_secondary = secondary;

            debug_assert!(self.graphics_allowed);

            let contents = if secondary {
                SubpassContents::SecondaryCommandBuffers
            } else {
                SubpassContents::Inline
            };
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
    pub fn update_buffer<B, D>(mut self, buffer: B, data: D) -> Result<Self, UpdateBufferError>
        where B: TypedBufferAccess<Content = D> + Send + Sync + 'static,
              D: Send + Sync + 'static
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
unsafe fn push_constants<P, Pl, Pc>(destination: &mut SyncCommandBufferBuilder<P>, pipeline: Pl,
                                    push_constants: Pc)
    where Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static
{
    for num_range in 0 .. pipeline.num_push_constants_ranges() {
        let range = match pipeline.push_constants_range(num_range) {
            Some(r) => r,
            None => continue,
        };

        debug_assert_eq!(range.offset % 4, 0);
        debug_assert_eq!(range.size % 4, 0);

        let data = slice::from_raw_parts((&push_constants as *const Pc as *const u8)
                                             .offset(range.offset as isize),
                                         range.size as usize);

        destination.push_constants::<_, [u8]>(pipeline.clone(),
                                              range.stages,
                                              range.offset as u32,
                                              range.size as u32,
                                              data);
    }
}

// Shortcut function to change the state of the pipeline.
unsafe fn set_state<P>(destination: &mut SyncCommandBufferBuilder<P>, dynamic: DynamicState) {
    if let Some(line_width) = dynamic.line_width {
        destination.set_line_width(line_width);
    }

    if let Some(ref viewports) = dynamic.viewports {
        destination.set_viewport(0, viewports.iter().cloned().collect::<Vec<_>>().into_iter()); // TODO: don't collect
    }

    if let Some(ref scissors) = dynamic.scissors {
        destination.set_scissor(0, scissors.iter().cloned().collect::<Vec<_>>().into_iter()); // TODO: don't collect
    }
}

// Shortcut function to bind vertex buffers.
unsafe fn vertex_buffers<P>(destination: &mut SyncCommandBufferBuilder<P>,
                            state_cacher: &mut StateCacher,
                            vertex_buffers: Vec<Box<BufferAccess + Send + Sync>>)
                            -> Result<(), SyncCommandBufferBuilderError> {
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

unsafe fn descriptor_sets<P, Pl, S>(destination: &mut SyncCommandBufferBuilder<P>,
                                    state_cacher: &mut StateCacher, gfx: bool, pipeline: Pl,
                                    sets: S)
                                    -> Result<(), SyncCommandBufferBuilderError>
    where Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static,
          S: DescriptorSetsCollection
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
    sets_binder
        .submit(gfx, pipeline.clone(), first_binding, iter::empty())?;
    Ok(())
}

pub struct AutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer<P>,

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
    fn lock_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError> {
        match self.submit_state {
            SubmitState::OneTime { ref already_submitted } => {
                let was_already_submitted = already_submitted.swap(true, Ordering::SeqCst);
                if was_already_submitted {
                    return Err(CommandBufferExecError::OneTimeSubmitAlreadySubmitted);
                }
            },
            SubmitState::ExclusiveUse { ref in_use } => {
                let already_in_use = in_use.swap(true, Ordering::SeqCst);
                if already_in_use {
                    return Err(CommandBufferExecError::ExclusiveAlreadyInUse);
                }
            },
            SubmitState::Concurrent => (),
        };

        let err = match self.inner.lock_submit(future, queue) {
            Ok(()) => return Ok(()),
            Err(err) => err,
        };

        // If `self.inner.lock_submit()` failed, we revert action.
        match self.submit_state {
            SubmitState::OneTime { ref already_submitted } => {
                already_submitted.store(false, Ordering::SeqCst);
            },
            SubmitState::ExclusiveUse { ref in_use } => {
                in_use.store(false, Ordering::SeqCst);
            },
            SubmitState::Concurrent => (),
        };

        Err(err)
    }

    #[inline]
    unsafe fn unlock(&self) {
        // Because of panic safety, we unlock the inner command buffer first.
        self.inner.unlock();

        match self.submit_state {
            SubmitState::OneTime { ref already_submitted } => {
                debug_assert!(already_submitted.load(Ordering::SeqCst));
            },
            SubmitState::ExclusiveUse { ref in_use } => {
                let old_val = in_use.swap(false, Ordering::SeqCst);
                debug_assert!(old_val);
            },
            SubmitState::Concurrent => (),
        };
    }

    #[inline]
    fn check_buffer_access(
        &self, buffer: &BufferAccess, exclusive: bool, queue: &Queue)
        -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        self.inner.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool,
                          queue: &Queue)
                          -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
        self.inner
            .check_image_access(image, layout, exclusive, queue)
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
            fn description(&self) -> &str {
                match *self {
                    $(
                        $name::$err(_) => {
                            concat!("a ", stringify!($err))
                        }
                    )+
                }
            }

            #[inline]
            fn cause(&self) -> Option<&error::Error> {
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
                write!(fmt, "{}", error::Error::description(self))
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
    /// Tried to use a graphics pipeline whose subpass index didn't match the current subpass
    /// index.
    WrongSubpassIndex,
    /// Tried to use a graphics pipeline whose render pass is incompatible with the current render
    /// pass.
    IncompatibleRenderPass,
}

impl error::Error for AutoCommandBufferBuilderContextError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            AutoCommandBufferBuilderContextError::ForbiddenInSecondary => {
                "operation forbidden in a secondary command buffer"
            },
            AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass => {
                "operation forbidden inside of a render pass"
            },
            AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass => {
                "operation forbidden outside of a render pass"
            },
            AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily => {
                "the queue family doesn't allow this operation"
            },
            AutoCommandBufferBuilderContextError::NumSubpassesMismatch { .. } => {
                "tried to end a render pass with subpasses remaining, or tried to go to next \
                 subpass with no subpass remaining"
            },
            AutoCommandBufferBuilderContextError::WrongSubpassType => {
                "tried to execute a secondary command buffer inside a subpass that only allows \
                 inline commands, or a draw command in a subpass that only allows secondary \
                 command buffers"
            },
            AutoCommandBufferBuilderContextError::WrongSubpassIndex => {
                "tried to use a graphics pipeline whose subpass index didn't match the current \
                 subpass index"
            },
            AutoCommandBufferBuilderContextError::IncompatibleRenderPass => {
                "tried to use a graphics pipeline whose render pass is incompatible with the \
                 current render pass"
            },
        }
    }
}

impl fmt::Display for AutoCommandBufferBuilderContextError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
