// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::buffer::TypedBufferAccess;
use crate::command_buffer::pool::standard::StandardCommandPoolAlloc;
use crate::command_buffer::pool::standard::StandardCommandPoolBuilder;
use crate::command_buffer::pool::CommandPool;
use crate::command_buffer::pool::CommandPoolBuilderAlloc;
use crate::command_buffer::synced::SyncCommandBuffer;
use crate::command_buffer::synced::SyncCommandBufferBuilder;
use crate::command_buffer::synced::SyncCommandBufferBuilderError;
use crate::command_buffer::sys::UnsafeCommandBuffer;
use crate::command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use crate::command_buffer::sys::UnsafeCommandBufferBuilderColorImageClear;
use crate::command_buffer::sys::UnsafeCommandBufferBuilderImageBlit;
use crate::command_buffer::sys::UnsafeCommandBufferBuilderImageCopy;
use crate::command_buffer::validity::*;
use crate::command_buffer::CommandBufferExecError;
use crate::command_buffer::CommandBufferInheritance;
use crate::command_buffer::CommandBufferInheritanceRenderPass;
use crate::command_buffer::CommandBufferLevel;
use crate::command_buffer::CommandBufferUsage;
use crate::command_buffer::DispatchIndirectCommand;
use crate::command_buffer::DrawIndexedIndirectCommand;
use crate::command_buffer::DrawIndirectCommand;
use crate::command_buffer::DynamicState;
use crate::command_buffer::ImageUninitializedSafe;
use crate::command_buffer::PrimaryCommandBuffer;
use crate::command_buffer::SecondaryCommandBuffer;
use crate::command_buffer::StateCacher;
use crate::command_buffer::StateCacherOutcome;
use crate::command_buffer::SubpassContents;
use crate::descriptor_set::DescriptorSetWithOffsets;
use crate::descriptor_set::DescriptorSetsCollection;
use crate::device::physical::QueueFamily;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::format::ClearValue;
use crate::format::FormatTy;
use crate::format::Pixel;
use crate::image::ImageAccess;
use crate::image::ImageAspect;
use crate::image::ImageAspects;
use crate::image::ImageLayout;
use crate::pipeline::input_assembly::Index;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::vertex::VertexSource;
use crate::pipeline::ComputePipelineAbstract;
use crate::pipeline::GraphicsPipelineAbstract;
use crate::pipeline::PipelineBindPoint;
use crate::query::QueryControlFlags;
use crate::query::QueryPipelineStatisticFlags;
use crate::query::QueryPool;
use crate::query::QueryResultElement;
use crate::query::QueryResultFlags;
use crate::query::QueryType;
use crate::render_pass::Framebuffer;
use crate::render_pass::FramebufferAbstract;
use crate::render_pass::LoadOp;
use crate::render_pass::RenderPass;
use crate::render_pass::Subpass;
use crate::sampler::Filter;
use crate::sync::AccessCheckError;
use crate::sync::AccessFlags;
use crate::sync::GpuFuture;
use crate::sync::PipelineMemoryAccess;
use crate::sync::PipelineStage;
use crate::sync::PipelineStages;
use crate::VulkanObject;
use crate::{OomError, SafeDeref};
use fnv::FnvHashMap;
use std::convert::TryInto;
use std::error;
use std::ffi::CStr;
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::slice;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Note that command buffers allocated from the default command pool (`Arc<StandardCommandPool>`)
/// don't implement the `Send` and `Sync` traits. If you use this pool, then the
/// `AutoCommandBufferBuilder` will not implement `Send` and `Sync` either. Once a command buffer
/// is built, however, it *does* implement `Send` and `Sync`.
pub struct AutoCommandBufferBuilder<L, P = StandardCommandPoolBuilder> {
    inner: SyncCommandBufferBuilder,
    pool_builder_alloc: P, // Safety: must be dropped after `inner`
    state_cacher: StateCacher,

    // The queue family that this command buffer is being created for.
    queue_family_id: u32,

    // The inheritance for secondary command buffers.
    inheritance: Option<CommandBufferInheritance<Box<dyn FramebufferAbstract + Send + Sync>>>,

    // Usage flags passed when creating the command buffer.
    usage: CommandBufferUsage,

    // If we're inside a render pass, contains the render pass state.
    render_pass_state: Option<RenderPassState>,

    // If any queries are active, this hashmap contains their state.
    query_state: FnvHashMap<ash::vk::QueryType, QueryState>,

    _data: PhantomData<L>,
}

// The state of the current render pass, specifying the pass, subpass index and its intended contents.
struct RenderPassState {
    subpass: (Arc<RenderPass>, u32),
    contents: SubpassContents,
    framebuffer: ash::vk::Framebuffer, // Always null for secondary command buffers
}

// The state of an active query.
struct QueryState {
    query_pool: ash::vk::QueryPool,
    query: u32,
    ty: QueryType,
    flags: QueryControlFlags,
    in_subpass: bool,
}

impl AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder> {
    /// Starts building a primary command buffer.
    #[inline]
    pub fn primary(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
    ) -> Result<
        AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        OomError,
    > {
        AutoCommandBufferBuilder::with_level(
            device,
            queue_family,
            usage,
            CommandBufferLevel::primary(),
        )
    }
}

impl AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder> {
    /// Starts building a secondary compute command buffer.
    #[inline]
    pub fn secondary_compute(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
    ) -> Result<
        AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        OomError,
    > {
        let level = CommandBufferLevel::secondary(None, QueryPipelineStatisticFlags::none());
        AutoCommandBufferBuilder::with_level(device, queue_family, usage, level)
    }

    /// Same as `secondary_compute`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_compute_inherit_queries(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
        occlusion_query: Option<QueryControlFlags>,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<
        AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        BeginError,
    > {
        if occlusion_query.is_some() && !device.enabled_features().inherited_queries {
            return Err(BeginError::InheritedQueriesFeatureNotEnabled);
        }

        if query_statistics_flags.count() > 0
            && !device.enabled_features().pipeline_statistics_query
        {
            return Err(BeginError::PipelineStatisticsQueryFeatureNotEnabled);
        }

        let level = CommandBufferLevel::secondary(occlusion_query, query_statistics_flags);
        Ok(AutoCommandBufferBuilder::with_level(
            device,
            queue_family,
            usage,
            level,
        )?)
    }

    /// Starts building a secondary graphics command buffer.
    #[inline]
    pub fn secondary_graphics(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
        subpass: Subpass,
    ) -> Result<
        AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        OomError,
    > {
        let level = CommandBufferLevel::Secondary(CommandBufferInheritance {
            render_pass: Some(CommandBufferInheritanceRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<()>>>,
            }),
            occlusion_query: None,
            query_statistics_flags: QueryPipelineStatisticFlags::none(),
        });

        AutoCommandBufferBuilder::with_level(device, queue_family, usage, level)
    }

    /// Same as `secondary_graphics`, but allows specifying how queries are being inherited.
    #[inline]
    pub fn secondary_graphics_inherit_queries(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
        subpass: Subpass,
        occlusion_query: Option<QueryControlFlags>,
        query_statistics_flags: QueryPipelineStatisticFlags,
    ) -> Result<
        AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, StandardCommandPoolBuilder>,
        BeginError,
    > {
        if occlusion_query.is_some() && !device.enabled_features().inherited_queries {
            return Err(BeginError::InheritedQueriesFeatureNotEnabled);
        }

        if query_statistics_flags.count() > 0
            && !device.enabled_features().pipeline_statistics_query
        {
            return Err(BeginError::PipelineStatisticsQueryFeatureNotEnabled);
        }

        let level = CommandBufferLevel::Secondary(CommandBufferInheritance {
            render_pass: Some(CommandBufferInheritanceRenderPass {
                subpass,
                framebuffer: None::<Arc<Framebuffer<()>>>,
            }),
            occlusion_query,
            query_statistics_flags,
        });

        Ok(AutoCommandBufferBuilder::with_level(
            device,
            queue_family,
            usage,
            level,
        )?)
    }
}

impl<L> AutoCommandBufferBuilder<L, StandardCommandPoolBuilder> {
    // Actual constructor. Private.
    fn with_level<F>(
        device: Arc<Device>,
        queue_family: QueueFamily,
        usage: CommandBufferUsage,
        level: CommandBufferLevel<F>,
    ) -> Result<AutoCommandBufferBuilder<L, StandardCommandPoolBuilder>, OomError>
    where
        F: FramebufferAbstract + Clone + Send + Sync + 'static,
    {
        let (inheritance, render_pass_state) = match &level {
            CommandBufferLevel::Primary => (None, None),
            CommandBufferLevel::Secondary(inheritance) => {
                let (render_pass, render_pass_state) = match inheritance.render_pass.as_ref() {
                    Some(CommandBufferInheritanceRenderPass {
                        subpass,
                        framebuffer,
                    }) => {
                        let render_pass = CommandBufferInheritanceRenderPass {
                            subpass: Subpass::from(subpass.render_pass().clone(), subpass.index())
                                .unwrap(),
                            framebuffer: framebuffer
                                .as_ref()
                                .map(|f| Box::new(f.clone()) as Box<_>),
                        };
                        let render_pass_state = RenderPassState {
                            subpass: (subpass.render_pass().clone(), subpass.index()),
                            contents: SubpassContents::Inline,
                            framebuffer: ash::vk::Framebuffer::null(), // Only needed for primary command buffers
                        };
                        (Some(render_pass), Some(render_pass_state))
                    }
                    None => (None, None),
                };

                (
                    Some(CommandBufferInheritance {
                        render_pass,
                        occlusion_query: inheritance.occlusion_query,
                        query_statistics_flags: inheritance.query_statistics_flags,
                    }),
                    render_pass_state,
                )
            }
        };

        unsafe {
            let pool = Device::standard_command_pool(&device, queue_family);
            let pool_builder_alloc = pool
                .alloc(!matches!(level, CommandBufferLevel::Primary), 1)?
                .next()
                .expect("Requested one command buffer from the command pool, but got zero.");
            let inner = SyncCommandBufferBuilder::new(pool_builder_alloc.inner(), level, usage)?;

            Ok(AutoCommandBufferBuilder {
                inner,
                pool_builder_alloc,
                state_cacher: StateCacher::new(),
                queue_family_id: queue_family.id(),
                render_pass_state,
                query_state: FnvHashMap::default(),
                inheritance,
                usage,
                _data: PhantomData,
            })
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BeginError {
    /// Occlusion query inheritance was requested, but the `inherited_queries` feature was not enabled.
    InheritedQueriesFeatureNotEnabled,
    /// Not enough memory.
    OomError(OomError),
    /// Pipeline statistics query inheritance was requested, but the `pipeline_statistics_query` feature was not enabled.
    PipelineStatisticsQueryFeatureNotEnabled,
}

impl error::Error for BeginError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for BeginError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::InheritedQueriesFeatureNotEnabled => {
                    "occlusion query inheritance was requested but the corresponding feature \
                 wasn't enabled"
                }
                Self::OomError(_) => "not enough memory available",
                Self::PipelineStatisticsQueryFeatureNotEnabled => {
                    "pipeline statistics query inheritance was requested but the corresponding \
                 feature wasn't enabled"
                }
            }
        )
    }
}

impl From<OomError> for BeginError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<PrimaryAutoCommandBuffer<P::Alloc>, BuildError> {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass.into());
        }

        if !self.query_state.is_empty() {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        let submit_state = match self.usage {
            CommandBufferUsage::MultipleSubmit => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            CommandBufferUsage::SimultaneousUse => SubmitState::Concurrent,
            CommandBufferUsage::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        Ok(PrimaryAutoCommandBuffer {
            inner: self.inner.build()?,
            pool_alloc: self.pool_builder_alloc.into_alloc(),
            submit_state,
        })
    }
}

impl<P> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Builds the command buffer.
    #[inline]
    pub fn build(self) -> Result<SecondaryAutoCommandBuffer<P::Alloc>, BuildError> {
        if !self.query_state.is_empty() {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        let submit_state = match self.usage {
            CommandBufferUsage::MultipleSubmit => SubmitState::ExclusiveUse {
                in_use: AtomicBool::new(false),
            },
            CommandBufferUsage::SimultaneousUse => SubmitState::Concurrent,
            CommandBufferUsage::OneTimeSubmit => SubmitState::OneTime {
                already_submitted: AtomicBool::new(false),
            },
        };

        Ok(SecondaryAutoCommandBuffer {
            inner: self.inner.build()?,
            pool_alloc: self.pool_builder_alloc.into_alloc(),
            inheritance: self.inheritance.unwrap(),
            submit_state,
        })
    }
}

impl<L, P> AutoCommandBufferBuilder<L, P> {
    #[inline]
    fn ensure_outside_render_pass(&self) -> Result<(), AutoCommandBufferBuilderContextError> {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass);
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
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass)?;

        // Subpass must be for inline commands
        if render_pass_state.contents != SubpassContents::Inline {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
        }

        // Subpasses must be the same.
        if pipeline.subpass().index() != render_pass_state.subpass.1 {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
        }

        // Render passes must be compatible.
        if !pipeline
            .subpass()
            .render_pass()
            .desc()
            .is_compatible_with_desc(&render_pass_state.subpass.0.desc())
        {
            return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
        }

        Ok(())
    }

    #[inline]
    fn queue_family(&self) -> QueueFamily {
        self.device()
            .physical_device()
            .queue_family_by_id(self.queue_family_id)
            .unwrap()
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
                aspects: ImageAspects {
                    color: source.has_color(),
                    depth: !source.has_color() && source.has_depth() && destination.has_depth(),
                    stencil: !source.has_color()
                        && source.has_stencil()
                        && destination.has_stencil(),
                    ..ImageAspects::none()
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
            if !self.queue_family().supports_graphics() {
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
                aspects: if source.has_color() {
                    ImageAspects {
                        color: true,
                        ..ImageAspects::none()
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
            if !self.queue_family().supports_graphics() && !self.queue_family().supports_compute() {
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
        Px: Pixel,
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
        Px: Pixel,
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
                    ImageAspect::Color
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

    /// Adds a command that copies from an image to a buffer.
    // The data layout of the image on the gpu is opaque, as in, it is non of our business how the gpu stores the image.
    // This does not matter since the act of copying the image into a buffer converts it to linear form.
    pub fn copy_image_to_buffer<S, D, Px>(
        &mut self,
        source: S,
        destination: D,
    ) -> Result<&mut Self, CopyBufferImageError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: TypedBufferAccess<Content = [Px]> + Send + Sync + 'static,
        Px: Pixel,
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
        Px: Pixel,
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
                // TODO: Allow the user to choose aspect
                image_aspect: if source.has_color() {
                    ImageAspect::Color
                } else if source.has_depth() {
                    ImageAspect::Depth
                } else if source.has_stencil() {
                    ImageAspect::Stencil
                } else {
                    unimplemented!()
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
        if !self.queue_family().supports_graphics() && self.queue_family().supports_compute() {
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
        if !self.queue_family().supports_graphics() && self.queue_family().supports_compute() {
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
        if !self.queue_family().supports_graphics() && self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        check_debug_marker_color(color)?;

        unsafe {
            self.inner.debug_marker_insert(name.into(), color);
        }

        Ok(self)
    }

    /// Perform a single compute operation using a compute pipeline.
    #[inline]
    pub fn dispatch<Cp, S, Pc>(
        &mut self,
        group_counts: [u32; 3],
        pipeline: Cp,
        descriptor_sets: S,
        push_constants: Pc,
    ) -> Result<&mut Self, DispatchError>
    where
        Cp: ComputePipelineAbstract + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
    {
        let descriptor_sets = descriptor_sets.into_vec();

        unsafe {
            if !self.queue_family().supports_compute() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_push_constants_validity(pipeline.layout(), &push_constants)?;
            check_descriptor_sets_validity(pipeline.layout(), &descriptor_sets)?;
            check_dispatch(pipeline.device(), group_counts)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_compute_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_compute(pipeline.clone());
            }

            set_push_constants(&mut self.inner, pipeline.layout(), push_constants);
            bind_descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                PipelineBindPoint::Compute,
                pipeline.layout(),
                descriptor_sets,
            )?;

            self.inner.dispatch(group_counts);
            Ok(self)
        }
    }

    /// Perform multiple compute operations using a compute pipeline. One dispatch is performed for
    /// each `vulkano::command_buffer::DispatchIndirectCommand` struct in `indirect_buffer`.
    #[inline]
    pub fn dispatch_indirect<Inb, Cp, S, Pc>(
        &mut self,
        indirect_buffer: Inb,
        pipeline: Cp,
        descriptor_sets: S,
        push_constants: Pc,
    ) -> Result<&mut Self, DispatchIndirectError>
    where
        Inb: BufferAccess
            + TypedBufferAccess<Content = [DispatchIndirectCommand]>
            + Send
            + Sync
            + 'static,
        Cp: ComputePipelineAbstract + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
    {
        let descriptor_sets = descriptor_sets.into_vec();

        unsafe {
            if !self.queue_family().supports_compute() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_indirect_buffer(self.device(), &indirect_buffer)?;
            check_push_constants_validity(pipeline.layout(), &push_constants)?;
            check_descriptor_sets_validity(pipeline.layout(), &descriptor_sets)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_compute_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_compute(pipeline.clone());
            }

            set_push_constants(&mut self.inner, pipeline.layout(), push_constants);
            bind_descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                PipelineBindPoint::Compute,
                pipeline.layout(),
                descriptor_sets,
            )?;

            self.inner.dispatch_indirect(indirect_buffer)?;
            Ok(self)
        }
    }

    /// Perform a single draw operation using a graphics pipeline.
    ///
    /// `vertex_buffer` is a set of vertex and/or instance buffers used to provide input.
    ///
    /// All data in `vertex_buffer` is used for the draw operation. To use only some data in the
    /// buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw<V, Gp, S, Pc>(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffers: V,
        descriptor_sets: S,
        push_constants: Pc,
    ) -> Result<&mut Self, DrawError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
    {
        let descriptor_sets = descriptor_sets.into_vec();
        let vertex_buffers = pipeline.decode(vertex_buffers).0;

        let (max_vertex_count, max_instance_count) =
            pipeline.vertex_input().max_vertices_instances(
                vertex_buffers
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as u32, v as _)),
            );

        if first_vertex + vertex_count > max_vertex_count {
            return Err(CheckVertexBufferError::TooManyVertices {
                vertex_count,
                max_vertex_count,
            }
            .into());
        }

        if first_instance + instance_count > max_instance_count {
            return Err(CheckVertexBufferError::TooManyInstances {
                instance_count,
                max_instance_count,
            }
            .into());
        }

        if let Some(multiview) = pipeline.subpass().render_pass().desc().multiview() {
            let max_instance_index = pipeline
                .device()
                .physical_device()
                .properties()
                .max_multiview_instance_index
                .unwrap_or(0);

            if first_instance + instance_count > max_instance_index + 1 {
                return Err(CheckVertexBufferError::TooManyInstances {
                    instance_count,
                    max_instance_count: max_instance_index + 1, // TODO: this can overflow
                }
                .into());
            }
        }

        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(pipeline.layout(), &push_constants)?;
            check_descriptor_sets_validity(pipeline.layout(), &descriptor_sets)?;
            check_vertex_buffers(&pipeline, &vertex_buffers)?;

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            set_push_constants(&mut self.inner, pipeline.layout(), push_constants);
            set_state(&mut self.inner, &dynamic);
            bind_descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                PipelineBindPoint::Graphics,
                pipeline.layout(),
                descriptor_sets,
            )?;
            bind_vertex_buffers(&mut self.inner, &mut self.state_cacher, vertex_buffers)?;

            debug_assert!(self.queue_family().supports_graphics());

            self.inner
                .draw(vertex_count, instance_count, first_vertex, first_instance);
            Ok(self)
        }
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
    /// `vertex_buffer` is a set of vertex and/or instance buffers used to provide input. It is
    /// used for every draw operation.
    ///
    /// All data in `vertex_buffer` is used for every draw operation. To use only some data in the
    /// buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw_indirect<V, Gp, S, Pc, Inb>(
        &mut self,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffers: V,
        indirect_buffer: Inb,
        descriptor_sets: S,
        push_constants: Pc,
    ) -> Result<&mut Self, DrawIndirectError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
        Inb: BufferAccess
            + TypedBufferAccess<Content = [DrawIndirectCommand]>
            + Send
            + Sync
            + 'static,
    {
        let descriptor_sets = descriptor_sets.into_vec();
        let vertex_buffers = pipeline.decode(vertex_buffers).0;

        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_indirect_buffer(self.device(), &indirect_buffer)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(pipeline.layout(), &push_constants)?;
            check_descriptor_sets_validity(pipeline.layout(), &descriptor_sets)?;
            check_vertex_buffers(&pipeline, &vertex_buffers)?;

            let requested = indirect_buffer.len() as u32;
            let limit = self
                .device()
                .physical_device()
                .properties()
                .max_draw_indirect_count
                .unwrap();

            if requested > limit {
                return Err(
                    CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded {
                        limit,
                        requested,
                    }
                    .into(),
                );
            }

            if let StateCacherOutcome::NeedChange =
                self.state_cacher.bind_graphics_pipeline(&pipeline)
            {
                self.inner.bind_pipeline_graphics(pipeline.clone());
            }

            let dynamic = self.state_cacher.dynamic_state(dynamic);

            set_push_constants(&mut self.inner, pipeline.layout(), push_constants);
            set_state(&mut self.inner, &dynamic);
            bind_descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                PipelineBindPoint::Graphics,
                pipeline.layout(),
                descriptor_sets,
            )?;
            bind_vertex_buffers(&mut self.inner, &mut self.state_cacher, vertex_buffers)?;

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.draw_indirect(
                indirect_buffer,
                requested,
                mem::size_of::<DrawIndirectCommand>() as u32,
            )?;
            Ok(self)
        }
    }

    /// Perform a single draw operation using a graphics pipeline, using an index buffer.
    ///
    /// `vertex_buffer` is a set of vertex and/or instance buffers used to provide input.
    /// `index_buffer` is a buffer containing indices into the vertex buffer that should be
    /// processed in order.
    ///
    /// All data in `vertex_buffer` and `index_buffer` is used for the draw operation. To use
    /// only some data in the buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw_indexed<V, Gp, S, Pc, Ib, I>(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffers: V,
        index_buffer: Ib,
        descriptor_sets: S,
        push_constants: Pc,
    ) -> Result<&mut Self, DrawIndexedError>
    where
        Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone, // TODO: meh for Clone
        S: DescriptorSetsCollection,
        Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
        I: Index + 'static,
    {
        let descriptor_sets = descriptor_sets.into_vec();
        let vertex_buffers = pipeline.decode(vertex_buffers).0;

        let (max_vertex_count, max_instance_count) =
            pipeline.vertex_input().max_vertices_instances(
                vertex_buffers
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as u32, v as _)),
            );
        let max_index_count = index_buffer.len().try_into().unwrap_or(u32::MAX);

        if first_index + index_count > max_index_count {
            return Err(CheckVertexBufferError::TooManyIndices {
                index_count,
                max_index_count,
            }
            .into());
        }

        if first_instance + instance_count > max_instance_count {
            return Err(CheckVertexBufferError::TooManyInstances {
                instance_count,
                max_instance_count,
            }
            .into());
        }

        if let Some(multiview) = pipeline.subpass().render_pass().desc().multiview() {
            let max_instance_index = pipeline
                .device()
                .physical_device()
                .properties()
                .max_multiview_instance_index
                .unwrap_or(0);

            if first_instance + instance_count > max_instance_index + 1 {
                return Err(CheckVertexBufferError::TooManyInstances {
                    instance_count,
                    max_instance_count: max_instance_index + 1, // TODO: this can overflow
                }
                .into());
            }
        }

        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_index_buffer(self.device(), &index_buffer)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(pipeline.layout(), &push_constants)?;
            check_descriptor_sets_validity(pipeline.layout(), &descriptor_sets)?;
            check_vertex_buffers(&pipeline, &vertex_buffers)?;

            if let Some(multiview) = pipeline.subpass().render_pass().desc().multiview() {
                let max_instance_index = pipeline
                    .device()
                    .physical_device()
                    .properties()
                    .max_multiview_instance_index
                    .unwrap_or(0);

                // vulkano currently always uses `0` as the first instance which means the highest
                // used index will just be `instance_count - 1`
                if instance_count > max_instance_index + 1 {
                    return Err(CheckVertexBufferError::TooManyInstances {
                        instance_count,
                        max_instance_count: max_instance_index + 1,
                    }
                    .into());
                }
            }

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

            set_push_constants(&mut self.inner, pipeline.layout(), push_constants);
            set_state(&mut self.inner, &dynamic);
            bind_descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                PipelineBindPoint::Graphics,
                pipeline.layout(),
                descriptor_sets,
            )?;
            bind_vertex_buffers(&mut self.inner, &mut self.state_cacher, vertex_buffers)?;
            // TODO: how to handle an index out of range of the vertex buffers?

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
            Ok(self)
        }
    }

    /// Perform multiple draw operations using a graphics pipeline, using an index buffer.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](crate::device::Properties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](crate::device::Features::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// `vertex_buffer` is a set of vertex and/or instance buffers used to provide input.
    /// `index_buffer` is a buffer containing indices into the vertex buffer that should be
    /// processed in order.
    ///
    /// All data in `vertex_buffer` and `index_buffer` is used for every draw operation. To use
    /// only some data in the buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw_indexed_indirect<V, Gp, S, Pc, Ib, Inb, I>(
        &mut self,
        pipeline: Gp,
        dynamic: &DynamicState,
        vertex_buffers: V,
        index_buffer: Ib,
        indirect_buffer: Inb,
        descriptor_sets: S,
        push_constants: Pc,
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
        let descriptor_sets = descriptor_sets.into_vec();
        let vertex_buffers = pipeline.decode(vertex_buffers).0;

        unsafe {
            // TODO: must check that pipeline is compatible with render pass

            self.ensure_inside_render_pass_inline(&pipeline)?;
            check_index_buffer(self.device(), &index_buffer)?;
            check_indirect_buffer(self.device(), &indirect_buffer)?;
            check_dynamic_state_validity(&pipeline, dynamic)?;
            check_push_constants_validity(pipeline.layout(), &push_constants)?;
            check_descriptor_sets_validity(pipeline.layout(), &descriptor_sets)?;
            check_vertex_buffers(&pipeline, &vertex_buffers)?;

            let requested = indirect_buffer.len() as u32;
            let limit = self
                .device()
                .physical_device()
                .properties()
                .max_draw_indirect_count
                .unwrap();

            if requested > limit {
                return Err(
                    CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded {
                        limit,
                        requested,
                    }
                    .into(),
                );
            }

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

            set_push_constants(&mut self.inner, pipeline.layout(), push_constants);
            set_state(&mut self.inner, &dynamic);
            bind_descriptor_sets(
                &mut self.inner,
                &mut self.state_cacher,
                PipelineBindPoint::Graphics,
                pipeline.layout(),
                descriptor_sets,
            )?;
            bind_vertex_buffers(&mut self.inner, &mut self.state_cacher, vertex_buffers)?;

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.draw_indexed_indirect(
                indirect_buffer,
                requested,
                mem::size_of::<DrawIndexedIndirectCommand>() as u32,
            )?;
            Ok(self)
        }
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

    /// Adds a command that writes data to a buffer.
    ///
    /// If `data` is larger than the buffer, only the part of `data` that fits is written. If the
    /// buffer is larger than `data`, only the start of the buffer is written.
    #[inline]
    pub fn update_buffer<B, D, Dd>(
        &mut self,
        buffer: B,
        data: Dd,
    ) -> Result<&mut Self, UpdateBufferError>
    where
        B: TypedBufferAccess<Content = D> + Send + Sync + 'static,
        D: ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            check_update_buffer(self.device(), &buffer, data.deref())?;

            let size_of_data = mem::size_of_val(data.deref());
            if buffer.size() >= size_of_data {
                self.inner.update_buffer(buffer, data);
            } else {
                unimplemented!() // TODO:
                                 //self.inner.update_buffer(buffer.slice(0 .. size_of_data), data);
            }

            Ok(self)
        }
    }

    /// Adds a command that begins a query.
    ///
    /// The query will be active until [`end_query`](Self::end_query) is called for the same query.
    ///
    /// # Safety
    /// The query must be unavailable, ensured by calling [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn begin_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, BeginQueryError> {
        check_begin_query(self.device(), &query_pool, query, flags)?;

        match query_pool.ty() {
            QueryType::Occlusion => {
                if !self.queue_family().supports_graphics() {
                    return Err(
                        AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into(),
                    );
                }
            }
            QueryType::PipelineStatistics(flags) => {
                if flags.is_compute() && !self.queue_family().supports_compute()
                    || flags.is_graphics() && !self.queue_family().supports_graphics()
                {
                    return Err(
                        AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into(),
                    );
                }
            }
            QueryType::Timestamp => unreachable!(),
        }

        let ty = query_pool.ty();
        let raw_ty = ty.into();
        let raw_query_pool = query_pool.internal_object();
        if self.query_state.contains_key(&raw_ty) {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        // TODO: validity checks
        self.inner.begin_query(query_pool, query, flags);
        self.query_state.insert(
            raw_ty,
            QueryState {
                query_pool: raw_query_pool,
                query,
                ty,
                flags,
                in_subpass: self.render_pass_state.is_some(),
            },
        );

        Ok(self)
    }

    /// Adds a command that ends an active query.
    pub fn end_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> Result<&mut Self, EndQueryError> {
        unsafe {
            check_end_query(self.device(), &query_pool, query)?;

            let raw_ty = query_pool.ty().into();
            let raw_query_pool = query_pool.internal_object();
            if !self.query_state.get(&raw_ty).map_or(false, |state| {
                state.query_pool == raw_query_pool && state.query == query
            }) {
                return Err(AutoCommandBufferBuilderContextError::QueryNotActive.into());
            }

            self.inner.end_query(query_pool, query);
            self.query_state.remove(&raw_ty);
        }

        Ok(self)
    }

    /// Adds a command that writes a timestamp to a timestamp query.
    ///
    /// # Safety
    /// The query must be unavailable, ensured by calling [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, WriteTimestampError> {
        check_write_timestamp(
            self.device(),
            self.queue_family(),
            &query_pool,
            query,
            stage,
        )?;

        if !(self.queue_family().supports_graphics()
            || self.queue_family().supports_compute()
            || self.queue_family().explicitly_supports_transfers())
        {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        // TODO: validity checks
        self.inner.write_timestamp(query_pool, query, stage);

        Ok(self)
    }

    /// Adds a command that copies the results of a range of queries to a buffer on the GPU.
    ///
    /// [`query_pool.ty().result_size()`](crate::query::QueryType::result_size) elements
    /// will be written for each query in the range, plus 1 extra element per query if
    /// [`QueryResultFlags::with_availability`] is enabled.
    /// The provided buffer must be large enough to hold the data.
    ///
    /// See also [`get_results`](crate::query::QueriesRange::get_results).
    pub fn copy_query_pool_results<D, T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: D,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, CopyQueryPoolResultsError>
    where
        D: BufferAccess + TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
        T: QueryResultElement,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            let stride = check_copy_query_pool_results(
                self.device(),
                &query_pool,
                queries.clone(),
                &destination,
                flags,
            )?;
            self.inner
                .copy_query_pool_results(query_pool, queries, destination, stride, flags)?;
        }

        Ok(self)
    }

    /// Adds a command to reset a range of queries on a query pool.
    ///
    /// The affected queries will be marked as "unavailable" after this command runs, and will no
    /// longer return any results. They will be ready to have new results recorded for them.
    ///
    /// # Safety
    /// The queries in the specified range must not be active in another command buffer.
    pub unsafe fn reset_query_pool(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> Result<&mut Self, ResetQueryPoolError> {
        self.ensure_outside_render_pass()?;
        check_reset_query_pool(self.device(), &query_pool, queries.clone())?;

        let raw_query_pool = query_pool.internal_object();
        if self
            .query_state
            .values()
            .any(|state| state.query_pool == raw_query_pool && queries.contains(&state.query))
        {
            return Err(AutoCommandBufferBuilderContextError::QueryIsActive.into());
        }

        // TODO: validity checks
        // Do other command buffers actually matter here? Not sure on the Vulkan spec.
        self.inner.reset_query_pool(query_pool, queries);

        Ok(self)
    }
}

/// Commands that can only be executed on primary command buffers
impl<P> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<P::Alloc>, P>
where
    P: CommandPoolBuilderAlloc,
{
    /// Adds a command that enters a render pass.
    ///
    /// If `contents` is `SubpassContents::SecondaryCommandBuffers`, then you will only be able to
    /// add secondary command buffers while you're inside the first subpass of the render pass.
    /// If it is `SubpassContents::Inline`, you will only be able to add inline draw commands and
    /// not secondary command buffers.
    ///
    /// C must contain exactly one clear value for each attachment in the framebuffer.
    ///
    /// You must call this before you can add draw commands.
    #[inline]
    pub fn begin_render_pass<F, I>(
        &mut self,
        framebuffer: F,
        contents: SubpassContents,
        clear_values: I,
    ) -> Result<&mut Self, BeginRenderPassError>
    where
        F: FramebufferAbstract + Clone + Send + Sync + 'static,
        I: IntoIterator<Item = ClearValue>,
    {
        unsafe {
            if !self.queue_family().supports_graphics() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            let clear_values = framebuffer
                .render_pass()
                .desc()
                .convert_clear_values(clear_values);
            let clear_values = clear_values.collect::<Vec<_>>().into_iter(); // TODO: necessary for Send + Sync ; needs an API rework of convert_clear_values
            let mut clear_values_copy = clear_values.clone().enumerate(); // TODO: Proper errors for clear value errors instead of panics

            for (atch_i, atch_desc) in framebuffer
                .render_pass()
                .desc()
                .attachments()
                .into_iter()
                .enumerate()
            {
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

            if let Some(multiview_desc) = framebuffer.render_pass().desc().multiview() {
                // When multiview is enabled, at the beginning of each subpass all non-render pass state is undefined
                self.state_cacher.invalidate();

                // ensure that the framebuffer is compatible with the render pass multiview configuration
                if multiview_desc
                    .view_masks
                    .iter()
                    .chain(multiview_desc.correlation_masks.iter())
                    .map(|&mask| 32 - mask.leading_zeros()) // calculates the highest used layer index of the mask
                    .any(|highest_used_layer| highest_used_layer > framebuffer.layers())
                {
                    panic!("A multiview mask references more layers than exist in the framebuffer");
                }
            }

            let framebuffer_object = FramebufferAbstract::inner(&framebuffer).internal_object();
            self.inner
                .begin_render_pass(framebuffer.clone(), contents, clear_values)?;
            self.render_pass_state = Some(RenderPassState {
                subpass: (framebuffer.render_pass().clone(), 0),
                contents,
                framebuffer: framebuffer_object,
            });
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
            if let Some(render_pass_state) = self.render_pass_state.as_ref() {
                let (ref rp, index) = render_pass_state.subpass;

                if rp.desc().subpasses().len() as u32 != index + 1 {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                        actual: rp.desc().subpasses().len() as u32,
                        current: index,
                    });
                }
            } else {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
            }

            if self.query_state.values().any(|state| state.in_subpass) {
                return Err(AutoCommandBufferBuilderContextError::QueryIsActive);
            }

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.end_render_pass();
            self.render_pass_state = None;
            Ok(self)
        }
    }

    /// Adds a command that executes a secondary command buffer.
    ///
    /// If the `flags` that `command_buffer` was created with are more restrictive than those of
    /// `self`, then `self` will be restricted to match. E.g. executing a secondary command buffer
    /// with `Flags::OneTimeSubmit` will set `self`'s flags to `Flags::OneTimeSubmit` also.
    pub fn execute_commands<C>(
        &mut self,
        command_buffer: C,
    ) -> Result<&mut Self, ExecuteCommandsError>
    where
        C: SecondaryCommandBuffer + Send + Sync + 'static,
    {
        self.check_command_buffer(&command_buffer)?;
        let secondary_usage = command_buffer.inner().usage();

        unsafe {
            let mut builder = self.inner.execute_commands();
            builder.add(command_buffer);
            builder.submit()?;
        }

        // Secondary command buffer could leave the primary in any state.
        self.state_cacher.invalidate();

        // If the secondary is non-concurrent or one-time use, that restricts the primary as well.
        self.usage = std::cmp::min(self.usage, secondary_usage);

        Ok(self)
    }

    /// Adds a command that multiple secondary command buffers in a vector.
    ///
    /// This requires that the secondary command buffers do not have resource conflicts; an error
    /// will be returned if there are any. Use `execute_commands` if you want to ensure that
    /// resource conflicts are automatically resolved.
    // TODO ^ would be nice if this just worked without errors
    pub fn execute_commands_from_vec<C>(
        &mut self,
        command_buffers: Vec<C>,
    ) -> Result<&mut Self, ExecuteCommandsError>
    where
        C: SecondaryCommandBuffer + Send + Sync + 'static,
    {
        for command_buffer in &command_buffers {
            self.check_command_buffer(command_buffer)?;
        }

        let mut secondary_usage = CommandBufferUsage::SimultaneousUse; // Most permissive usage
        unsafe {
            let mut builder = self.inner.execute_commands();
            for command_buffer in command_buffers {
                secondary_usage = std::cmp::min(secondary_usage, command_buffer.inner().usage());
                builder.add(command_buffer);
            }
            builder.submit()?;
        }

        // Secondary command buffer could leave the primary in any state.
        self.state_cacher.invalidate();

        // If the secondary is non-concurrent or one-time use, that restricts the primary as well.
        self.usage = std::cmp::min(self.usage, secondary_usage);

        Ok(self)
    }

    // Helper function for execute_commands
    fn check_command_buffer<C>(
        &self,
        command_buffer: &C,
    ) -> Result<(), AutoCommandBufferBuilderContextError>
    where
        C: SecondaryCommandBuffer + Send + Sync + 'static,
    {
        if let Some(render_pass) = command_buffer.inheritance().render_pass {
            self.ensure_inside_render_pass_secondary(&render_pass)?;
        } else {
            self.ensure_outside_render_pass()?;
        }

        for state in self.query_state.values() {
            match state.ty {
                QueryType::Occlusion => match command_buffer.inheritance().occlusion_query {
                    Some(inherited_flags) => {
                        let inherited_flags = ash::vk::QueryControlFlags::from(inherited_flags);
                        let state_flags = ash::vk::QueryControlFlags::from(state.flags);

                        if inherited_flags & state_flags != state_flags {
                            return Err(AutoCommandBufferBuilderContextError::QueryNotInherited);
                        }
                    }
                    None => return Err(AutoCommandBufferBuilderContextError::QueryNotInherited),
                },
                QueryType::PipelineStatistics(state_flags) => {
                    let inherited_flags = command_buffer.inheritance().query_statistics_flags;
                    let inherited_flags =
                        ash::vk::QueryPipelineStatisticFlags::from(inherited_flags);
                    let state_flags = ash::vk::QueryPipelineStatisticFlags::from(state_flags);

                    if inherited_flags & state_flags != state_flags {
                        return Err(AutoCommandBufferBuilderContextError::QueryNotInherited);
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }

    #[inline]
    fn ensure_inside_render_pass_secondary(
        &self,
        render_pass: &CommandBufferInheritanceRenderPass<&dyn FramebufferAbstract>,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
        let render_pass_state = self
            .render_pass_state
            .as_ref()
            .ok_or(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass)?;

        if render_pass_state.contents != SubpassContents::SecondaryCommandBuffers {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassType);
        }

        // Subpasses must be the same.
        if render_pass.subpass.index() != render_pass_state.subpass.1 {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
        }

        // Render passes must be compatible.
        if !render_pass
            .subpass
            .render_pass()
            .desc()
            .is_compatible_with_desc(render_pass_state.subpass.0.desc())
        {
            return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
        }

        // Framebuffer, if present on the secondary command buffer, must be the
        // same as the one in the current render pass.
        if let Some(framebuffer) = render_pass.framebuffer {
            if FramebufferAbstract::inner(framebuffer).internal_object()
                != render_pass_state.framebuffer
            {
                return Err(AutoCommandBufferBuilderContextError::IncompatibleFramebuffer);
            }
        }

        Ok(())
    }

    /// Adds a command that jumps to the next subpass of the current render pass.
    #[inline]
    pub fn next_subpass(
        &mut self,
        contents: SubpassContents,
    ) -> Result<&mut Self, AutoCommandBufferBuilderContextError> {
        unsafe {
            if let Some(render_pass_state) = self.render_pass_state.as_mut() {
                let (ref rp, ref mut index) = render_pass_state.subpass;

                if *index + 1 >= rp.desc().subpasses().len() as u32 {
                    return Err(AutoCommandBufferBuilderContextError::NumSubpassesMismatch {
                        actual: rp.desc().subpasses().len() as u32,
                        current: *index,
                    });
                } else {
                    *index += 1;
                    render_pass_state.contents = contents;
                }

                if let Some(multiview) = rp.desc().multiview() {
                    // When multiview is enabled, at the beginning of each subpass all non-render pass state is undefined
                    self.state_cacher.invalidate();
                }
            } else {
                return Err(AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass);
            }

            if self.query_state.values().any(|state| state.in_subpass) {
                return Err(AutoCommandBufferBuilderContextError::QueryIsActive);
            }

            debug_assert!(self.queue_family().supports_graphics());

            self.inner.next_subpass(contents);
            Ok(self)
        }
    }
}

impl<P> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer<P::Alloc>, P> where
    P: CommandPoolBuilderAlloc
{
}

unsafe impl<L, P> DeviceOwned for AutoCommandBufferBuilder<L, P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

// Shortcut function to set the push constants.
unsafe fn set_push_constants<Pc>(
    destination: &mut SyncCommandBufferBuilder,
    pipeline_layout: &Arc<PipelineLayout>,
    push_constants: Pc,
) {
    for range in pipeline_layout.push_constant_ranges() {
        debug_assert_eq!(range.offset % 4, 0);
        debug_assert_eq!(range.size % 4, 0);

        let data = slice::from_raw_parts(
            (&push_constants as *const Pc as *const u8).offset(range.offset as isize),
            range.size as usize,
        );

        destination.push_constants::<[u8]>(
            pipeline_layout.clone(),
            range.stages,
            range.offset as u32,
            range.size as u32,
            data,
        );
    }
}

// Shortcut function to change the state of the pipeline.
unsafe fn set_state(destination: &mut SyncCommandBufferBuilder, dynamic: &DynamicState) {
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
unsafe fn bind_vertex_buffers(
    destination: &mut SyncCommandBufferBuilder,
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

unsafe fn bind_descriptor_sets(
    destination: &mut SyncCommandBufferBuilder,
    state_cacher: &mut StateCacher,
    pipeline_bind_point: PipelineBindPoint,
    pipeline_layout: &Arc<PipelineLayout>,
    descriptor_sets: Vec<DescriptorSetWithOffsets>,
) -> Result<(), SyncCommandBufferBuilderError> {
    let first_binding = {
        let mut compare = state_cacher.bind_descriptor_sets(pipeline_bind_point);
        for descriptor_set in descriptor_sets.iter() {
            compare.add(descriptor_set);
        }
        compare.compare()
    };

    let first_binding = match first_binding {
        None => return Ok(()),
        Some(fb) => fb,
    };

    let mut sets_binder = destination.bind_descriptor_sets();
    for set in descriptor_sets.into_iter().skip(first_binding as usize) {
        sets_binder.add(set);
    }
    sets_binder.submit(pipeline_bind_point, pipeline_layout.clone(), first_binding)?;
    Ok(())
}

pub struct PrimaryAutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer,
    pool_alloc: P, // Safety: must be dropped after `inner`

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<P> DeviceOwned for PrimaryAutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<P> PrimaryCommandBuffer for PrimaryAutoCommandBuffer<P> {
    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer {
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
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.inner.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.inner
            .check_image_access(image, layout, exclusive, queue)
    }
}

pub struct SecondaryAutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer,
    pool_alloc: P, // Safety: must be dropped after `inner`
    inheritance: CommandBufferInheritance<Box<dyn FramebufferAbstract + Send + Sync>>,

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<P> DeviceOwned for SecondaryAutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<P> SecondaryCommandBuffer for SecondaryAutoCommandBuffer<P> {
    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer {
        self.inner.as_ref()
    }

    #[inline]
    fn lock_record(&self) -> Result<(), CommandBufferExecError> {
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

        Ok(())
    }

    #[inline]
    unsafe fn unlock(&self) {
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

    fn inheritance(&self) -> CommandBufferInheritance<&dyn FramebufferAbstract> {
        CommandBufferInheritance {
            render_pass: self.inheritance.render_pass.as_ref().map(
                |CommandBufferInheritanceRenderPass {
                     subpass,
                     framebuffer,
                 }| {
                    CommandBufferInheritanceRenderPass {
                        subpass: subpass.clone(),
                        framebuffer: framebuffer.as_ref().map(|f| f.as_ref() as &_),
                    }
                },
            ),
            occlusion_query: self.inheritance.occlusion_query,
            query_statistics_flags: self.inheritance.query_statistics_flags,
        }
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        self.inner.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, PipelineMemoryAccess)> {
        self.inner.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.inner.num_images()
    }

    #[inline]
    fn image(
        &self,
        index: usize,
    ) -> Option<(
        &dyn ImageAccess,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
        ImageUninitializedSafe,
    )> {
        self.inner.image(index)
    }
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
            fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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

err_gen!(CopyQueryPoolResultsError {
    AutoCommandBufferBuilderContextError,
    CheckCopyQueryPoolResultsError,
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

err_gen!(DispatchIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckIndirectBufferError,
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
    CheckIndirectBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndexBufferError,
    CheckIndirectBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(ExecuteCommandsError {
    AutoCommandBufferBuilderContextError,
    SyncCommandBufferBuilderError,
});

err_gen!(BeginQueryError {
    AutoCommandBufferBuilderContextError,
    CheckBeginQueryError,
});

err_gen!(EndQueryError {
    AutoCommandBufferBuilderContextError,
    CheckEndQueryError,
});

err_gen!(WriteTimestampError {
    AutoCommandBufferBuilderContextError,
    CheckWriteTimestampError,
});

err_gen!(ResetQueryPoolError {
    AutoCommandBufferBuilderContextError,
    CheckResetQueryPoolError,
});

err_gen!(UpdateBufferError {
    AutoCommandBufferBuilderContextError,
    CheckUpdateBufferError,
});

#[derive(Debug, Copy, Clone)]
pub enum AutoCommandBufferBuilderContextError {
    /// Operation forbidden inside of a render pass.
    ForbiddenInsideRenderPass,
    /// Operation forbidden outside of a render pass.
    ForbiddenOutsideRenderPass,
    /// Tried to use a secondary command buffer with a specified framebuffer that is
    /// incompatible with the current framebuffer.
    IncompatibleFramebuffer,
    /// Tried to use a graphics pipeline or secondary command buffer whose render pass
    /// is incompatible with the current render pass.
    IncompatibleRenderPass,
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
    /// A query is active that conflicts with the current operation.
    QueryIsActive,
    /// This query was not active.
    QueryNotActive,
    /// A query is active that is not included in the `inheritance` of the secondary command buffer.
    QueryNotInherited,
    /// Tried to use a graphics pipeline or secondary command buffer whose subpass index
    /// didn't match the current subpass index.
    WrongSubpassIndex,
    /// Tried to execute a secondary command buffer inside a subpass that only allows inline
    /// commands, or a draw command in a subpass that only allows secondary command buffers.
    WrongSubpassType,
}

impl error::Error for AutoCommandBufferBuilderContextError {}

impl fmt::Display for AutoCommandBufferBuilderContextError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass => {
                    "operation forbidden inside of a render pass"
                }
                AutoCommandBufferBuilderContextError::ForbiddenOutsideRenderPass => {
                    "operation forbidden outside of a render pass"
                }
                AutoCommandBufferBuilderContextError::IncompatibleFramebuffer => {
                    "tried to use a secondary command buffer with a specified framebuffer that is \
                 incompatible with the current framebuffer"
                }
                AutoCommandBufferBuilderContextError::IncompatibleRenderPass => {
                    "tried to use a graphics pipeline or secondary command buffer whose render pass \
                  is incompatible with the current render pass"
                }
                AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily => {
                    "the queue family doesn't allow this operation"
                }
                AutoCommandBufferBuilderContextError::NumSubpassesMismatch { .. } => {
                    "tried to end a render pass with subpasses remaining, or tried to go to next \
                 subpass with no subpass remaining"
                }
                AutoCommandBufferBuilderContextError::QueryIsActive => {
                    "a query is active that conflicts with the current operation"
                }
                AutoCommandBufferBuilderContextError::QueryNotActive => {
                    "this query was not active"
                }
                AutoCommandBufferBuilderContextError::QueryNotInherited => {
                    "a query is active that is not included in the inheritance of the secondary command buffer"
                }
                AutoCommandBufferBuilderContextError::WrongSubpassIndex => {
                    "tried to use a graphics pipeline whose subpass index didn't match the current \
                 subpass index"
                }
                AutoCommandBufferBuilderContextError::WrongSubpassType => {
                    "tried to execute a secondary command buffer inside a subpass that only allows \
                 inline commands, or a draw command in a subpass that only allows secondary \
                 command buffers"
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::BufferUsage;
    use crate::buffer::CpuAccessibleBuffer;
    use crate::command_buffer::synced::SyncCommandBufferBuilderError;
    use crate::command_buffer::AutoCommandBufferBuilder;
    use crate::command_buffer::CommandBufferExecError;
    use crate::command_buffer::CommandBufferUsage;
    use crate::command_buffer::ExecuteCommandsError;
    use crate::command_buffer::PrimaryCommandBuffer;
    use crate::device::physical::PhysicalDevice;
    use crate::device::Device;
    use crate::device::DeviceExtensions;
    use crate::device::Features;
    use crate::sync::GpuFuture;
    use std::sync::Arc;

    #[test]
    fn copy_buffer_dimensions() {
        let instance = instance!();

        let phys = match PhysicalDevice::enumerate(&instance).next() {
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

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
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

    #[test]
    fn secondary_nonconcurrent_conflict() {
        let (device, queue) = gfx_dev_and_queue!();

        // Make a secondary CB that doesn't support simultaneous use.
        let builder = AutoCommandBufferBuilder::secondary_compute(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();
        let secondary = Arc::new(builder.build().unwrap());

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Add the secondary a first time
            builder.execute_commands(secondary.clone()).unwrap();

            // Recording the same non-concurrent secondary command buffer twice into the same
            // primary is an error.
            assert!(matches!(
                builder.execute_commands(secondary.clone()),
                Err(ExecuteCommandsError::SyncCommandBufferBuilderError(
                    SyncCommandBufferBuilderError::ExecError(
                        CommandBufferExecError::ExclusiveAlreadyInUse
                    )
                ))
            ));
        }

        {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();
            builder.execute_commands(secondary.clone()).unwrap();
            let cb1 = builder.build().unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::SimultaneousUse,
            )
            .unwrap();

            // Recording the same non-concurrent secondary command buffer into multiple
            // primaries is an error.
            assert!(matches!(
                builder.execute_commands(secondary.clone()),
                Err(ExecuteCommandsError::SyncCommandBufferBuilderError(
                    SyncCommandBufferBuilderError::ExecError(
                        CommandBufferExecError::ExclusiveAlreadyInUse
                    )
                ))
            ));

            std::mem::drop(cb1);

            // Now that the first cb is dropped, we should be able to record.
            builder.execute_commands(secondary.clone()).unwrap();
        }
    }
}
