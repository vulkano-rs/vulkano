// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    commands::{
        debug::CheckColorError,
        image::{CheckBlitImageError, CheckClearColorImageError, CheckClearDepthStencilImageError},
        pipeline::{
            CheckDescriptorSetsValidityError, CheckDispatchError, CheckDynamicStateValidityError,
            CheckIndexBufferError, CheckIndirectBufferError, CheckPipelineError,
            CheckPushConstantsValidityError, CheckVertexBufferError,
        },
        query::{
            CheckBeginQueryError, CheckCopyQueryPoolResultsError, CheckEndQueryError,
            CheckResetQueryPoolError, CheckWriteTimestampError,
        },
        transfer::{
            CheckCopyBufferError, CheckCopyBufferImageError, CheckCopyImageError,
            CheckFillBufferError, CheckUpdateBufferError,
        },
    },
    pool::{
        standard::{StandardCommandPoolAlloc, StandardCommandPoolBuilder},
        CommandPool, CommandPoolAlloc, CommandPoolBuilderAlloc,
    },
    synced::{
        CommandBufferState, SyncCommandBuffer, SyncCommandBufferBuilder,
        SyncCommandBufferBuilderError,
    },
    sys::{CommandBufferBeginInfo, UnsafeCommandBuffer},
    CommandBufferExecError, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
    CommandBufferLevel, CommandBufferUsage, PrimaryCommandBuffer, SecondaryCommandBuffer,
    SubpassContents,
};
use crate::{
    buffer::{sys::UnsafeBuffer, BufferAccess},
    device::{physical::QueueFamily, Device, DeviceOwned, Queue},
    image::{sys::UnsafeImage, ImageAccess, ImageLayout},
    pipeline::GraphicsPipeline,
    query::{QueryControlFlags, QueryPipelineStatisticFlags, QueryType},
    render_pass::Subpass,
    sync::{AccessCheckError, AccessFlags, GpuFuture, PipelineMemoryAccess, PipelineStages},
    DeviceSize, OomError,
};
use smallvec::SmallVec;
use std::{
    collections::HashMap,
    error, fmt,
    marker::PhantomData,
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// Note that command buffers allocated from the default command pool (`Arc<StandardCommandPool>`)
/// don't implement the `Send` and `Sync` traits. If you use this pool, then the
/// `AutoCommandBufferBuilder` will not implement `Send` and `Sync` either. Once a command buffer
/// is built, however, it *does* implement `Send` and `Sync`.
pub struct AutoCommandBufferBuilder<L, P = StandardCommandPoolBuilder> {
    pub(super) inner: SyncCommandBufferBuilder,
    pool_builder_alloc: P, // Safety: must be dropped after `inner`

    // The queue family that this command buffer is being created for.
    queue_family_id: u32,

    // The inheritance for secondary command buffers.
    // Must be `None` in a primary command buffer and `Some` in a secondary command buffer.
    inheritance_info: Option<CommandBufferInheritanceInfo>,

    // Usage flags passed when creating the command buffer.
    pub(super) usage: CommandBufferUsage,

    // If we're inside a render pass, contains the render pass state.
    pub(super) render_pass_state: Option<RenderPassState>,

    // If any queries are active, this hashmap contains their state.
    pub(super) query_state: HashMap<ash::vk::QueryType, QueryState>,

    _data: PhantomData<L>,
}

// The state of the current render pass, specifying the pass, subpass index and its intended contents.
pub(super) struct RenderPassState {
    pub(super) subpass: Subpass,
    pub(super) contents: SubpassContents,
    pub(super) attached_layers_ranges: SmallVec<[Range<u32>; 4]>,
    pub(super) extent: [u32; 2],
    pub(super) framebuffer: ash::vk::Framebuffer, // Always null for secondary command buffers
}

// The state of an active query.
pub(super) struct QueryState {
    pub(super) query_pool: ash::vk::QueryPool,
    pub(super) query: u32,
    pub(super) ty: QueryType,
    pub(super) flags: QueryControlFlags,
    pub(super) in_subpass: bool,
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
        unsafe {
            AutoCommandBufferBuilder::with_level(
                device,
                queue_family,
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage,
                    ..Default::default()
                },
            )
        }
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
        unsafe {
            Ok(AutoCommandBufferBuilder::with_level(
                device,
                queue_family,
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: Some(CommandBufferInheritanceInfo::default()),
                    ..Default::default()
                },
            )?)
        }
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

        unsafe {
            Ok(AutoCommandBufferBuilder::with_level(
                device,
                queue_family,
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: Some(CommandBufferInheritanceInfo {
                        occlusion_query,
                        query_statistics_flags,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )?)
        }
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
        unsafe {
            Ok(AutoCommandBufferBuilder::with_level(
                device,
                queue_family,
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: Some(CommandBufferInheritanceInfo {
                        render_pass: Some(CommandBufferInheritanceRenderPassInfo {
                            subpass,
                            framebuffer: None,
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )?)
        }
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

        unsafe {
            Ok(AutoCommandBufferBuilder::with_level(
                device,
                queue_family,
                CommandBufferLevel::Secondary,
                CommandBufferBeginInfo {
                    usage,
                    inheritance_info: Some(CommandBufferInheritanceInfo {
                        render_pass: Some(CommandBufferInheritanceRenderPassInfo {
                            subpass,
                            framebuffer: None,
                        }),
                        occlusion_query,
                        query_statistics_flags,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )?)
        }
    }
}

impl<L> AutoCommandBufferBuilder<L, StandardCommandPoolBuilder> {
    // Actual constructor. Private.
    //
    // `begin_info.inheritance_info` must match `level`.
    unsafe fn with_level(
        device: Arc<Device>,
        queue_family: QueueFamily,
        level: CommandBufferLevel,
        begin_info: CommandBufferBeginInfo,
    ) -> Result<AutoCommandBufferBuilder<L, StandardCommandPoolBuilder>, OomError> {
        let usage = begin_info.usage;
        let inheritance_info = begin_info.inheritance_info.clone();
        let render_pass_state = begin_info
            .inheritance_info
            .as_ref()
            .and_then(|inheritance_info| inheritance_info.render_pass.as_ref())
            .map(
                |CommandBufferInheritanceRenderPassInfo {
                     subpass,
                     framebuffer,
                 }| RenderPassState {
                    subpass: subpass.clone(),
                    contents: SubpassContents::Inline,
                    extent: framebuffer.as_ref().map(|f| f.extent()).unwrap_or_default(),
                    attached_layers_ranges: framebuffer
                        .as_ref()
                        .map(|f| f.attached_layers_ranges())
                        .unwrap_or_default(),
                    framebuffer: ash::vk::Framebuffer::null(), // Only needed for primary command buffers
                },
            );

        let pool = Device::standard_command_pool(&device, queue_family);
        let pool_builder_alloc = pool
            .allocate(level, 1)?
            .next()
            .expect("Requested one command buffer from the command pool, but got zero.");
        let inner = SyncCommandBufferBuilder::new(pool_builder_alloc.inner(), begin_info)?;

        Ok(AutoCommandBufferBuilder {
            inner,
            pool_builder_alloc,
            queue_family_id: queue_family.id(),
            render_pass_state,
            query_state: HashMap::default(),
            inheritance_info,
            usage,
            _data: PhantomData,
        })
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
            inheritance_info: self.inheritance_info.unwrap(),
            submit_state,
        })
    }
}

impl<L, P> AutoCommandBufferBuilder<L, P> {
    #[inline]
    pub(super) fn ensure_outside_render_pass(
        &self,
    ) -> Result<(), AutoCommandBufferBuilderContextError> {
        if self.render_pass_state.is_some() {
            return Err(AutoCommandBufferBuilderContextError::ForbiddenInsideRenderPass);
        }

        Ok(())
    }

    #[inline]
    pub(super) fn ensure_inside_render_pass_inline(
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
        if pipeline.subpass().index() != render_pass_state.subpass.index() {
            return Err(AutoCommandBufferBuilderContextError::WrongSubpassIndex);
        }

        // Render passes must be compatible.
        if !pipeline
            .subpass()
            .render_pass()
            .is_compatible_with(&render_pass_state.subpass.render_pass())
        {
            return Err(AutoCommandBufferBuilderContextError::IncompatibleRenderPass);
        }

        Ok(())
    }

    #[inline]
    pub(super) fn queue_family(&self) -> QueueFamily {
        self.device()
            .physical_device()
            .queue_family_by_id(self.queue_family_id)
            .unwrap()
    }

    /// Returns the binding/setting state.
    #[inline]
    pub fn state(&self) -> CommandBufferState {
        self.inner.state()
    }
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

unsafe impl<P> PrimaryCommandBuffer for PrimaryAutoCommandBuffer<P>
where
    P: CommandPoolAlloc,
{
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
        buffer: &UnsafeBuffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.inner
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &UnsafeImage,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.inner
            .check_image_access(image, range, exclusive, expected_layout, queue)
    }
}

pub struct SecondaryAutoCommandBuffer<P = StandardCommandPoolAlloc> {
    inner: SyncCommandBuffer,
    pool_alloc: P, // Safety: must be dropped after `inner`
    inheritance_info: CommandBufferInheritanceInfo,

    // Tracks usage of the command buffer on the GPU.
    submit_state: SubmitState,
}

unsafe impl<P> DeviceOwned for SecondaryAutoCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl<P> SecondaryCommandBuffer for SecondaryAutoCommandBuffer<P>
where
    P: CommandPoolAlloc,
{
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

    #[inline]
    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        &self.inheritance_info
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        self.inner.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&Arc<dyn BufferAccess>, PipelineMemoryAccess)> {
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
        &Arc<dyn ImageAccess>,
        PipelineMemoryAccess,
        ImageLayout,
        ImageLayout,
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
    SyncCommandBufferBuilderError,
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
    SyncCommandBufferBuilderError,
});

/// Errors that can happen when calling [`clear_attachments`](AutoCommandBufferBuilder::clear_attachments)
#[derive(Debug, Copy, Clone)]
pub enum ClearAttachmentsError {
    /// AutoCommandBufferBuilderContextError
    AutoCommandBufferBuilderContextError(AutoCommandBufferBuilderContextError),
    /// CheckPipelineError
    CheckPipelineError(CheckPipelineError),

    /// The index of the color attachment is not present
    InvalidColorAttachmentIndex(u32),
    /// There is no depth/stencil attachment present
    DepthStencilAttachmentNotPresent,
    /// The clear rect cannot have extent of `0`
    ZeroRectExtent,
    /// The layer count cannot be `0`
    ZeroLayerCount,
    /// The clear rect region must be inside the render area of the render pass
    RectOutOfBounds,
    /// The clear rect's layers must be inside the layers ranges for all the attachments
    LayersOutOfBounds,
    /// If the render pass instance this is recorded in uses multiview,
    /// then `ClearRect.base_array_layer` must be zero and `ClearRect.layer_count` must be one
    InvalidMultiviewLayerRange,
}

impl error::Error for ClearAttachmentsError {}

impl fmt::Display for ClearAttachmentsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ClearAttachmentsError::AutoCommandBufferBuilderContextError(e) => write!(fmt, "{}", e)?,
            ClearAttachmentsError::CheckPipelineError(e) => write!(fmt, "{}", e)?,
            ClearAttachmentsError::InvalidColorAttachmentIndex(index) => {
                write!(fmt, "Color attachment {} is not present", index)?
            }
            ClearAttachmentsError::DepthStencilAttachmentNotPresent => {
                write!(fmt, "There is no depth/stencil attachment present")?
            }
            ClearAttachmentsError::ZeroRectExtent => {
                write!(fmt, "The clear rect cannot have extent of 0")?
            }
            ClearAttachmentsError::ZeroLayerCount => write!(fmt, "The layer count cannot be 0")?,
            ClearAttachmentsError::RectOutOfBounds => write!(
                fmt,
                "The clear rect region must be inside the render area of the render pass"
            )?,
            ClearAttachmentsError::LayersOutOfBounds => write!(
                fmt,
                "The clear rect's layers must be inside the layers ranges for all the attachments"
            )?,
            ClearAttachmentsError::InvalidMultiviewLayerRange => write!(
                fmt,
                "If the render pass instance this is recorded in uses multiview, then `ClearRect.base_array_layer` must be zero and `ClearRect.layer_count` must be one" 
            )?,
        }
        Ok(())
    }
}

impl From<AutoCommandBufferBuilderContextError> for ClearAttachmentsError {
    #[inline]
    fn from(err: AutoCommandBufferBuilderContextError) -> ClearAttachmentsError {
        ClearAttachmentsError::AutoCommandBufferBuilderContextError(err)
    }
}

impl From<CheckPipelineError> for ClearAttachmentsError {
    #[inline]
    fn from(err: CheckPipelineError) -> ClearAttachmentsError {
        ClearAttachmentsError::CheckPipelineError(err)
    }
}

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
    use super::*;
    use crate::{
        buffer::{BufferUsage, CpuAccessibleBuffer},
        device::{physical::PhysicalDevice, DeviceCreateInfo, QueueCreateInfo},
    };

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
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
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

    #[test]
    fn buffer_self_copy_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer_dimensions(source.clone(), 0, source.clone(), 2, 2)
            .unwrap();

        let cb = builder.build().unwrap();

        let future = cb
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let result = source.read().unwrap();

        assert_eq!(*result, [0_u32, 1, 0, 1]);
    }

    #[test]
    fn buffer_self_copy_not_overlapping() {
        let (device, queue) = gfx_dev_and_queue!();

        let source = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            true,
            [0_u32, 1, 2, 3].iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        assert!(matches!(
            builder.copy_buffer_dimensions(source.clone(), 0, source.clone(), 1, 2),
            Err(CopyBufferError::CheckCopyBufferError(
                CheckCopyBufferError::OverlappingRanges
            ))
        ));
    }
}
