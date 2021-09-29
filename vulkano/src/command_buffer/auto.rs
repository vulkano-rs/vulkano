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
use crate::command_buffer::sys::UnsafeCommandBufferBuilderDepthStencilImageClear;
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
use crate::command_buffer::ImageUninitializedSafe;
use crate::command_buffer::PrimaryCommandBuffer;
use crate::command_buffer::SecondaryCommandBuffer;
use crate::command_buffer::SubpassContents;
use crate::descriptor_set::DescriptorSetsCollection;
use crate::device::physical::QueueFamily;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::format::ClearValue;
use crate::format::NumericType;
use crate::format::Pixel;
use crate::image::ImageAccess;
use crate::image::ImageAspect;
use crate::image::ImageAspects;
use crate::image::ImageLayout;
use crate::pipeline::depth_stencil::StencilFaces;
use crate::pipeline::input_assembly::Index;
use crate::pipeline::input_assembly::IndexType;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::shader::ShaderStages;
use crate::pipeline::vertex::VertexBuffersCollection;
use crate::pipeline::viewport::Scissor;
use crate::pipeline::viewport::Viewport;
use crate::pipeline::ComputePipeline;
use crate::pipeline::GraphicsPipeline;
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
use crate::DeviceSize;
use crate::VulkanObject;
use crate::{OomError, SafeDeref};
use fnv::FnvHashMap;
use smallvec::SmallVec;
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

    // The queue family that this command buffer is being created for.
    queue_family_id: u32,

    // The inheritance for secondary command buffers.
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
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
    fn ensure_inside_render_pass_inline(
        &self,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
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

    /// Returns the inner `SyncCommandBufferBuilder`, which can be queried for the current state.
    #[inline]
    pub fn inner(&self) -> &SyncCommandBufferBuilder {
        &self.inner
    }

    /// Binds descriptor sets for future dispatch or draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support `pipeline_bind_point`.
    /// - Panics if the highest descriptor set slot being bound is not less than the number of sets
    ///   in `pipeline_layout`.
    /// - Panics if `self` and any element of `descriptor_sets` do not belong to the same device.
    pub fn bind_descriptor_sets<S>(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: S,
    ) -> &mut Self
    where
        S: DescriptorSetsCollection,
    {
        match pipeline_bind_point {
            PipelineBindPoint::Compute => assert!(
                self.queue_family().supports_compute(),
                "the queue family of the command buffer must support compute operations"
            ),
            PipelineBindPoint::Graphics => assert!(
                self.queue_family().supports_graphics(),
                "the queue family of the command buffer must support graphics operations"
            ),
        }

        let descriptor_sets = descriptor_sets.into_vec();

        assert!(
            first_set as usize + descriptor_sets.len()
                <= pipeline_layout.descriptor_set_layouts().len(),
            "the highest descriptor set slot being bound must be less than the number of sets in pipeline_layout"
        );

        for (num, set) in descriptor_sets.iter().enumerate() {
            assert_eq!(
                set.as_ref().0.device().internal_object(),
                self.device().internal_object()
            );

            let pipeline_set = &pipeline_layout.descriptor_set_layouts()[first_set as usize + num];
            assert!(
                pipeline_set.is_compatible_with(set.as_ref().0.layout()),
                "the element of descriptor_sets being bound to slot {} is not compatible with the corresponding slot in pipeline_layout",
                first_set as usize + num,
            );

            // TODO: see https://github.com/vulkano-rs/vulkano/issues/1643
            // For each dynamic uniform or storage buffer binding in pDescriptorSets, the sum of the
            // effective offset, as defined above, and the range of the binding must be less than or
            // equal to the size of the buffer

            // TODO:
            // Each element of pDescriptorSets must not have been allocated from a VkDescriptorPool
            // with the VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_VALVE flag set
        }

        unsafe {
            let mut sets_binder = self.inner.bind_descriptor_sets();
            for set in descriptor_sets.into_iter() {
                sets_binder.add(set);
            }
            sets_binder.submit(pipeline_bind_point, pipeline_layout, first_set);
        }

        self
    }

    /// Binds an index buffer for future indexed draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `index_buffer` do not belong to the same device.
    /// - Panics if `index_buffer` does not have the
    ///   [`index_buffer`](crate::buffer::BufferUsage::index_buffer) usage enabled.
    /// - If the index buffer contains `u8` indices, panics if the
    ///   [`ext_index_type_uint8`](crate::device::DeviceExtensions::ext_index_type_uint8) extension is not
    ///   enabled on the device.
    pub fn bind_index_buffer<Ib, I>(&mut self, index_buffer: Ib) -> &mut Self
    where
        Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
        I: Index + 'static,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        assert_eq!(
            index_buffer.device().internal_object(),
            self.device().internal_object()
        );

        // TODO:
        // The sum of offset and the address of the range of VkDeviceMemory object that is backing
        // buffer, must be a multiple of the type indicated by indexType

        assert!(
            index_buffer.inner().buffer.usage().index_buffer,
            "index_buffer must have the index_buffer usage enabled"
        );

        // TODO:
        // If buffer is non-sparse then it must be bound completely and contiguously to a single
        // VkDeviceMemory object

        if !self.device().enabled_features().index_type_uint8 {
            assert!(I::ty() != IndexType::U8, "if the index buffer contains u8 indices, the index_type_uint8 extension must be enabled on the device");
        }

        unsafe {
            self.inner.bind_index_buffer(index_buffer, I::ty());
        }

        self
    }

    /// Binds a compute pipeline for future dispatch calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support compute operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_compute(&mut self, pipeline: Arc<ComputePipeline>) -> &mut Self {
        assert!(
            self.queue_family().supports_compute(),
            "the queue family of the command buffer must support compute operations"
        );

        assert_eq!(
            pipeline.device().internal_object(),
            self.device().internal_object()
        );

        // TODO:
        // This command must not be recorded when transform feedback is active

        // TODO:
        // pipeline must not have been created with VK_PIPELINE_CREATE_LIBRARY_BIT_KHR set

        unsafe {
            self.inner.bind_pipeline_compute(pipeline);
        }

        self
    }

    /// Binds a graphics pipeline for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if `self` and `pipeline` do not belong to the same device.
    pub fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        assert_eq!(
            pipeline.device().internal_object(),
            self.device().internal_object()
        );

        // TODO:
        // If the variable multisample rate feature is not supported, pipeline is a graphics
        // pipeline, the current subpass uses no attachments, and this is not the first call to
        // this function with a graphics pipeline after transitioning to the current subpass, then
        // the sample count specified by this pipeline must match that set in the previous pipeline

        // TODO:
        // If VkPhysicalDeviceSampleLocationsPropertiesEXT::variableSampleLocations is VK_FALSE, and
        // pipeline is a graphics pipeline created with a
        // VkPipelineSampleLocationsStateCreateInfoEXT structure having its sampleLocationsEnable
        // member set to VK_TRUE but without VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT enabled then the
        // current render pass instance must have been begun by specifying a
        // VkRenderPassSampleLocationsBeginInfoEXT structure whose pPostSubpassSampleLocations
        // member contains an element with a subpassIndex matching the current subpass index and the
        // sampleLocationsInfo member of that element must match the sampleLocationsInfo specified
        // in VkPipelineSampleLocationsStateCreateInfoEXT when the pipeline was created

        // TODO:
        // This command must not be recorded when transform feedback is active

        // TODO:
        // pipeline must not have been created with VK_PIPELINE_CREATE_LIBRARY_BIT_KHR set

        // TODO:
        // If commandBuffer is a secondary command buffer with
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled and
        // pipelineBindPoint is VK_PIPELINE_BIND_POINT_GRAPHICS, then the pipeline must have been
        // created with VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT or VK_DYNAMIC_STATE_VIEWPORT, and
        // VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT or VK_DYNAMIC_STATE_SCISSOR enabled

        // TODO:
        // If pipelineBindPoint is VK_PIPELINE_BIND_POINT_GRAPHICS and the
        // provokingVertexModePerPipeline limit is VK_FALSE, then pipeline’s
        // VkPipelineRasterizationProvokingVertexStateCreateInfoEXT::provokingVertexMode must be the
        // same as that of any other pipelines previously bound to this bind point within the
        // current renderpass instance, including any pipeline already bound when beginning the
        // renderpass instance

        unsafe {
            self.inner.bind_pipeline_graphics(pipeline);
        }

        self
    }

    /// Binds vertex buffers for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the highest vertex buffer binding being bound is greater than the
    ///   [`max_vertex_input_bindings`](crate::device::Properties::max_vertex_input_bindings)
    //    device property.
    /// - Panics if `self` and any element of `vertex_buffers` do not belong to the same device.
    /// - Panics if any element of `vertex_buffers` does not have the
    ///   [`vertex_buffer`](crate::buffer::BufferUsage::vertex_buffer) usage enabled.
    pub fn bind_vertex_buffers<V>(&mut self, first_binding: u32, vertex_buffers: V) -> &mut Self
    where
        V: VertexBuffersCollection,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        let vertex_buffers = vertex_buffers.into_vec();

        assert!(
            first_binding + vertex_buffers.len() as u32
                <= self
                    .device()
                    .physical_device()
                    .properties()
                    .max_vertex_input_bindings,
            "the highest vertex buffer binding being bound must not be higher than the max_vertex_input_bindings device property"
        );

        for (num, buf) in vertex_buffers.iter().enumerate() {
            assert_eq!(
                buf.device().internal_object(),
                self.device().internal_object()
            );

            assert!(
                buf.inner().buffer.usage().vertex_buffer,
                "vertex_buffers element {} must have the vertex_buffer usage",
                num
            );

            // TODO:
            // Each element of pBuffers that is non-sparse must be bound completely and contiguously
            // to a single VkDeviceMemory object

            // TODO:
            // If the nullDescriptor feature is not enabled, all elements of pBuffers must not be
            // VK_NULL_HANDLE

            // TODO:
            // If an element of pBuffers is VK_NULL_HANDLE, then the corresponding element of
            // pOffsets must be zero
        }

        unsafe {
            let mut binder = self.inner.bind_vertex_buffers();
            for vb in vertex_buffers.into_iter() {
                binder.add(vb);
            }
            binder.submit(first_binding);
        }

        self
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
                aspects: if source.format().aspects().color {
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

    /// Adds a command that clears all the layers of a depth / stencil image with a
    /// specific value.
    ///
    /// # Panic
    ///
    /// Panics if `clear_value` is not a depth / stencil value.
    ///
    pub fn clear_depth_stencil_image<I>(
        &mut self,
        image: I,
        clear_value: ClearValue,
    ) -> Result<&mut Self, ClearDepthStencilImageError>
    where
        I: ImageAccess + Send + Sync + 'static,
    {
        let layers = image.dimensions().array_layers();

        self.clear_depth_stencil_image_dimensions(image, 0, layers, clear_value)
    }

    /// Adds a command that clears a depth / stencil image with a specific value.
    ///
    /// # Panic
    ///
    /// - Panics if `clear_value` is not a depth / stencil value.
    ///
    pub fn clear_depth_stencil_image_dimensions<I>(
        &mut self,
        image: I,
        first_layer: u32,
        num_layers: u32,
        clear_value: ClearValue,
    ) -> Result<&mut Self, ClearDepthStencilImageError>
    where
        I: ImageAccess + Send + Sync + 'static,
    {
        unsafe {
            if !self.queue_family().supports_graphics() && !self.queue_family().supports_compute() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_clear_depth_stencil_image(self.device(), &image, first_layer, num_layers)?;

            let (clear_depth, clear_stencil) = match clear_value {
                ClearValue::Depth(_) => (true, false),
                ClearValue::Stencil(_) => (false, true),
                ClearValue::DepthStencil(_) => (true, true),
                _ => panic!("The clear value is not a depth / stencil value"),
            };

            let region = UnsafeCommandBufferBuilderDepthStencilImageClear {
                base_array_layer: first_layer,
                layer_count: num_layers,
                clear_depth,
                clear_stencil,
            };

            // TODO: let choose layout
            self.inner.clear_depth_stencil_image(
                image,
                ImageLayout::TransferDstOptimal,
                clear_value,
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
        source_offset: DeviceSize,
        destination: D,
        destination_offset: DeviceSize,
        count: DeviceSize,
    ) -> Result<&mut Self, CopyBufferError>
    where
        S: TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
        D: TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
    {
        self.ensure_outside_render_pass()?;

        let _infos = check_copy_buffer(self.device(), &source, &destination)?;
        debug_assert!(source_offset + count <= source.len());
        debug_assert!(destination_offset + count <= destination.len());

        let size = std::mem::size_of::<T>() as DeviceSize;
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
                image_aspect: if destination.format().aspects().color {
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

            let source_aspects = source.format().aspects();
            let destination_aspects = destination.format().aspects();
            let copy = UnsafeCommandBufferBuilderImageCopy {
                // TODO: Allowing choosing a subset of the image aspects, but note that if color
                // is included, neither depth nor stencil may.
                aspects: ImageAspects {
                    color: source_aspects.color,
                    depth: !source_aspects.color
                        && source_aspects.depth
                        && destination_aspects.depth,
                    stencil: !source_aspects.color
                        && source_aspects.stencil
                        && destination_aspects.stencil,
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

            let source_aspects = source.format().aspects();
            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                // TODO: Allow the user to choose aspect
                image_aspect: if source_aspects.color {
                    ImageAspect::Color
                } else if source_aspects.depth {
                    ImageAspect::Depth
                } else if source_aspects.stencil {
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
    pub fn dispatch(&mut self, group_counts: [u32; 3]) -> Result<&mut Self, DispatchError> {
        if !self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        let pipeline = check_pipeline_compute(&self.inner)?;
        self.ensure_outside_render_pass()?;
        check_descriptor_sets_validity(&self.inner, pipeline.layout(), PipelineBindPoint::Compute)?;
        check_push_constants_validity(&self.inner, pipeline.layout())?;
        check_dispatch(self.device(), group_counts)?;

        unsafe {
            self.inner.dispatch(group_counts);
        }

        Ok(self)
    }

    /// Perform multiple compute operations using a compute pipeline. One dispatch is performed for
    /// each `vulkano::command_buffer::DispatchIndirectCommand` struct in `indirect_buffer`.
    #[inline]
    pub fn dispatch_indirect<Inb>(
        &mut self,
        indirect_buffer: Inb,
    ) -> Result<&mut Self, DispatchIndirectError>
    where
        Inb: BufferAccess
            + TypedBufferAccess<Content = [DispatchIndirectCommand]>
            + Send
            + Sync
            + 'static,
    {
        if !self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        let pipeline = check_pipeline_compute(&self.inner)?;
        self.ensure_outside_render_pass()?;
        check_descriptor_sets_validity(&self.inner, pipeline.layout(), PipelineBindPoint::Compute)?;
        check_push_constants_validity(&self.inner, pipeline.layout())?;
        check_indirect_buffer(self.device(), &indirect_buffer)?;

        unsafe {
            self.inner.dispatch_indirect(indirect_buffer)?;
        }

        Ok(self)
    }

    /// Perform a single draw operation using a graphics pipeline.
    ///
    /// `vertex_buffer` is a set of vertex and/or instance buffers used to provide input.
    ///
    /// All data in `vertex_buffer` is used for the draw operation. To use only some data in the
    /// buffer, wrap it in a `vulkano::buffer::BufferSlice`.
    #[inline]
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, DrawError> {
        let pipeline = check_pipeline_graphics(&self.inner)?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(&self.inner, pipeline)?;
        check_descriptor_sets_validity(
            &self.inner,
            pipeline.layout(),
            PipelineBindPoint::Graphics,
        )?;
        check_push_constants_validity(&self.inner, pipeline.layout())?;
        check_vertex_buffers(
            &self.inner,
            pipeline,
            Some((first_vertex, vertex_count)),
            Some((first_instance, instance_count)),
        )?;

        unsafe {
            self.inner
                .draw(vertex_count, instance_count, first_vertex, first_instance);
        }

        Ok(self)
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
    pub fn draw_indirect<Inb>(
        &mut self,
        indirect_buffer: Inb,
    ) -> Result<&mut Self, DrawIndirectError>
    where
        Inb: BufferAccess
            + TypedBufferAccess<Content = [DrawIndirectCommand]>
            + Send
            + Sync
            + 'static,
    {
        let pipeline = check_pipeline_graphics(&self.inner)?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(&self.inner, pipeline)?;
        check_descriptor_sets_validity(
            &self.inner,
            pipeline.layout(),
            PipelineBindPoint::Graphics,
        )?;
        check_push_constants_validity(&self.inner, pipeline.layout())?;
        check_vertex_buffers(&self.inner, pipeline, None, None)?;
        check_indirect_buffer(self.device(), &indirect_buffer)?;

        let requested = indirect_buffer.len() as u32;
        let limit = self
            .device()
            .physical_device()
            .properties()
            .max_draw_indirect_count;

        if requested > limit {
            return Err(
                CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded { limit, requested }
                    .into(),
            );
        }

        unsafe {
            self.inner.draw_indirect(
                indirect_buffer,
                requested,
                mem::size_of::<DrawIndirectCommand>() as u32,
            )?;
        }

        Ok(self)
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
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, DrawIndexedError> {
        // TODO: how to handle an index out of range of the vertex buffers?
        let pipeline = check_pipeline_graphics(&self.inner)?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(&self.inner, pipeline)?;
        check_descriptor_sets_validity(
            &self.inner,
            pipeline.layout(),
            PipelineBindPoint::Graphics,
        )?;
        check_push_constants_validity(&self.inner, pipeline.layout())?;
        check_vertex_buffers(
            &self.inner,
            pipeline,
            None,
            Some((first_instance, instance_count)),
        )?;
        check_index_buffer(&self.inner, Some((first_index, index_count)))?;

        unsafe {
            self.inner.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }

        Ok(self)
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
    pub fn draw_indexed_indirect<Inb>(
        &mut self,
        indirect_buffer: Inb,
    ) -> Result<&mut Self, DrawIndexedIndirectError>
    where
        Inb: BufferAccess
            + TypedBufferAccess<Content = [DrawIndexedIndirectCommand]>
            + Send
            + Sync
            + 'static,
    {
        let pipeline = check_pipeline_graphics(&self.inner)?;
        self.ensure_inside_render_pass_inline(pipeline)?;
        check_dynamic_state_validity(&self.inner, pipeline)?;
        check_descriptor_sets_validity(
            &self.inner,
            pipeline.layout(),
            PipelineBindPoint::Graphics,
        )?;
        check_push_constants_validity(&self.inner, pipeline.layout())?;
        check_vertex_buffers(&self.inner, pipeline, None, None)?;
        check_index_buffer(&self.inner, None)?;
        check_indirect_buffer(self.device(), &indirect_buffer)?;

        let requested = indirect_buffer.len() as u32;
        let limit = self
            .device()
            .physical_device()
            .properties()
            .max_draw_indirect_count;

        if requested > limit {
            return Err(
                CheckIndirectBufferError::MaxDrawIndirectCountLimitExceeded { limit, requested }
                    .into(),
            );
        }

        unsafe {
            self.inner.draw_indexed_indirect(
                indirect_buffer,
                requested,
                mem::size_of::<DrawIndexedIndirectCommand>() as u32,
            )?;
        }

        Ok(self)
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
        B: BufferAccess + 'static,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            check_fill_buffer(self.device(), &buffer)?;
            self.inner.fill_buffer(buffer, data);
            Ok(self)
        }
    }

    /// Sets push constants for future dispatch or draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if `offset` is not a multiple of 4.
    /// - Panics if the size of `push_constants` is not a multiple of 4.
    /// - Panics if any of the bytes in `push_constants` do not fall within any of the pipeline
    ///   layout's push constant ranges.
    pub fn push_constants<Pc>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: Pc,
    ) -> &mut Self {
        let size = mem::size_of::<Pc>() as u32;

        if size == 0 {
            return self;
        }

        assert!(offset % 4 == 0, "the offset must be a multiple of 4");
        assert!(
            size % 4 == 0,
            "the size of push_constants must be a multiple of 4"
        );

        // Figure out which shader stages in the pipeline layout overlap this byte range.
        // Also check that none of the bytes being set are outside all push constant ranges.
        let shader_stages = pipeline_layout
            .push_constant_ranges()
            .iter()
            .filter(|range| range.offset < offset + size && offset < range.offset + range.size)
            .try_fold(
                (ShaderStages::none(), offset),
                |(shader_stages, last_bound), range| {
                    if range.offset > last_bound {
                        Err(())
                    } else {
                        Ok((
                            shader_stages.union(&range.stages),
                            last_bound.max(range.offset + range.size),
                        ))
                    }
                },
            )
            .and_then(|(shader_stages, last_bound)| {
                if shader_stages == ShaderStages::none() || last_bound < offset + size {
                    Err(())
                } else {
                    Ok(shader_stages)
                }
            })
            .expect(
                "not all bytes in push_constants fall within the pipeline layout's push constant ranges",
            );

        unsafe {
            let data = slice::from_raw_parts(
                (&push_constants as *const Pc as *const u8).offset(offset as isize),
                size as usize,
            );

            self.inner.push_constants::<[u8]>(
                pipeline_layout.clone(),
                shader_stages,
                offset,
                size,
                data,
            );
        }

        self
    }

    /// Sets the dynamic blend constants for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    pub fn set_blend_constants(&mut self, constants: [f32; 4]) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        unsafe {
            self.inner.set_blend_constants(constants);
        }

        self
    }

    /// Sets the dynamic depth bounds for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - If the
    ///   [`ext_depth_range_unrestricted`](crate::device::DeviceExtensions::ext_depth_range_unrestricted)
    ///   device extension is not enabled, panics if `min` or `max` is not between 0.0 and 1.0 inclusive.
    pub fn set_depth_bounds(&mut self, min: f32, max: f32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        if !self
            .device()
            .enabled_extensions()
            .ext_depth_range_unrestricted
        {
            assert!(
                min >= 0.0 && min <= 1.0 && max >= 0.0 && max <= 1.0,
                "if the ext_depth_range_unrestricted device extension is not enabled, depth bounds values must be between 0.0 and 1.0"
            );
        }

        unsafe {
            self.inner.set_depth_bounds(min, max);
        }

        self
    }

    /// Sets the dynamic line width for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - If the [`wide_lines`](crate::device::Features::wide_lines) feature is not enabled, panics
    ///   if `line_width` is not 1.0.
    pub fn set_line_width(&mut self, line_width: f32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        if !self.device().enabled_features().wide_lines {
            assert!(
                line_width == 1.0,
                "if the wide_line features is not enabled, line width must be 1.0"
            );
        }

        unsafe {
            self.inner.set_line_width(line_width);
        }

        self
    }

    /// Sets the dynamic scissors for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the highest scissor slot being set is greater than the
    ///   [`max_viewports`](crate::device::Properties::max_viewports) device property.
    /// - If the [`multi_viewport`](crate::device::Features::multi_viewport) feature is not enabled,
    ///   panics if `first_scissor` is not 0, or if more than 1 scissor is provided.
    pub fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I) -> &mut Self
    where
        I: IntoIterator<Item = Scissor>,
    {
        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        assert!(
            first_scissor + scissors.len() as u32 <= self.device().physical_device().properties().max_viewports,
            "the highest scissor slot being set must not be higher than the max_viewports device property"
        );

        if !self.device().enabled_features().multi_viewport {
            assert!(
                first_scissor == 0,
                "if the multi_viewport feature is not enabled, first_scissor must be 0"
            );

            assert!(
                scissors.len() <= 1,
                "if the multi_viewport feature is not enabled, no more than 1 scissor must be provided"
            );
        }

        // TODO:
        // If this command is recorded in a secondary command buffer with
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled, then this
        // function must not be called

        unsafe {
            self.inner.set_scissor(first_scissor, scissors);
        }

        self
    }

    /// Sets the dynamic stencil compare mask on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    pub fn set_stencil_compare_mask(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        unsafe {
            self.inner.set_stencil_compare_mask(faces, compare_mask);
        }

        self
    }

    /// Sets the dynamic stencil reference on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    pub fn set_stencil_reference(&mut self, faces: StencilFaces, reference: u32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        unsafe {
            self.inner.set_stencil_reference(faces, reference);
        }

        self
    }

    /// Sets the dynamic stencil write mask on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    pub fn set_stencil_write_mask(&mut self, faces: StencilFaces, write_mask: u32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        unsafe {
            self.inner.set_stencil_write_mask(faces, write_mask);
        }

        self
    }

    /// Sets the dynamic viewports for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the highest viewport slot being set is greater than the
    ///   [`max_viewports`](crate::device::Properties::max_viewports) device property.
    /// - If the [`multi_viewport`](crate::device::Features::multi_viewport) feature is not enabled,
    ///   panics if `first_viewport` is not 0, or if more than 1 viewport is provided.
    pub fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I) -> &mut Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();

        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );

        assert!(
            first_viewport + viewports.len() as u32 <= self.device().physical_device().properties().max_viewports,
            "the highest viewport slot being set must not be higher than the max_viewports device property"
        );

        if !self.device().enabled_features().multi_viewport {
            assert!(
                first_viewport == 0,
                "if the multi_viewport feature is not enabled, first_viewport must be 0"
            );

            assert!(
                viewports.len() <= 1,
                "if the multi_viewport feature is not enabled, no more than 1 viewport must be provided"
            );
        }

        // TODO:
        // commandBuffer must not have
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled

        unsafe {
            self.inner.set_viewport(first_viewport, viewports);
        }

        self
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

            let size_of_data = mem::size_of_val(data.deref()) as DeviceSize;
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
                            let aspects = atch_desc.format.aspects();

                            if aspects.depth && aspects.stencil {
                                assert!(
                                    matches!(clear_value, ClearValue::DepthStencil(_)),
                                    "Bad ClearValue! index: {}, attachment index: {}, expected: DepthStencil, got: {:?}",
                                    clear_i,
                                    atch_i,
                                    clear_value,
                                );
                            } else if aspects.depth {
                                assert!(
                                    matches!(clear_value, ClearValue::Depth(_)),
                                    "Bad ClearValue! index: {}, attachment index: {}, expected: Depth, got: {:?}",
                                    clear_i,
                                    atch_i,
                                    clear_value,
                                );
                            } else if aspects.depth {
                                assert!(
                                    matches!(clear_value, ClearValue::Stencil(_)),
                                    "Bad ClearValue! index: {}, attachment index: {}, expected: Stencil, got: {:?}",
                                    clear_i,
                                    atch_i,
                                    clear_value,
                                );
                            } else if let Some(numeric_type) = atch_desc.format.type_color() {
                                match numeric_type {
                                    NumericType::SFLOAT
                                    | NumericType::UFLOAT
                                    | NumericType::SNORM
                                    | NumericType::UNORM
                                    | NumericType::SSCALED
                                    | NumericType::USCALED
                                    | NumericType::SRGB => {
                                        assert!(
                                            matches!(clear_value, ClearValue::Float(_)),
                                            "Bad ClearValue! index: {}, attachment index: {}, expected: Float, got: {:?}",
                                            clear_i,
                                            atch_i,
                                            clear_value,
                                        );
                                    }
                                    NumericType::SINT => {
                                        assert!(
                                            matches!(clear_value, ClearValue::Int(_)),
                                            "Bad ClearValue! index: {}, attachment index: {}, expected: Int, got: {:?}",
                                            clear_i,
                                            atch_i,
                                            clear_value,
                                        );
                                    }
                                    NumericType::UINT => {
                                        assert!(
                                            matches!(clear_value, ClearValue::Uint(_)),
                                            "Bad ClearValue! index: {}, attachment index: {}, expected: Uint, got: {:?}",
                                            clear_i,
                                            atch_i,
                                            clear_value,
                                        );
                                    }
                                }
                            } else {
                                panic!("Shouldn't happen!");
                            }
                        } else {
                            assert!(
                                matches!(clear_value, ClearValue::None),
                                "Bad ClearValue! index: {}, attachment index: {}, expected: None, got: {:?}",
                                clear_i,
                                atch_i,
                                clear_value,
                            );
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
                self.inner.reset_state();

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
        self.inner.reset_state();

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
        self.inner.reset_state();

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
                    self.inner.reset_state();
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

err_gen!(ClearDepthStencilImageError {
    AutoCommandBufferBuilderContextError,
    CheckClearDepthStencilImageError,
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
    CheckPipelineError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckDispatchError,
    SyncCommandBufferBuilderError,
});

err_gen!(DispatchIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckIndirectBufferError,
    CheckDispatchError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndexBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
    CheckDynamicStateValidityError,
    CheckPushConstantsValidityError,
    CheckDescriptorSetsValidityError,
    CheckVertexBufferError,
    CheckIndirectBufferError,
    SyncCommandBufferBuilderError,
});

err_gen!(DrawIndexedIndirectError {
    AutoCommandBufferBuilderContextError,
    CheckPipelineError,
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
