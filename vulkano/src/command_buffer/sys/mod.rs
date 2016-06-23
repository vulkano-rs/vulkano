// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level command buffer construction.
//! 
//! # Safety
//! 
//! The `UnsafeCommandBufferBuilder` doesn't handle:
//! 
//! - Pipeline barriers.
//! - Checking render pass compatibility when submitting a draw command.

use std::mem;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::pool::AllocatedCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolFinished;
use command_buffer::CommandBuffer;
use command_buffer::DynamicState;
use device::Device;
use device::Queue;
use framebuffer::RenderPass;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use sync::Fence;
use sync::PipelineStages;
use sync::Semaphore;

use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub use self::copy_buffer::BufferCopyCommand;
pub use self::copy_buffer::BufferCopyError;
pub use self::copy_buffer::BufferCopyRegion;
pub use self::dispatch::DispatchCommand;
pub use self::draw::DrawCommand;
pub use self::draw::DrawTy;
pub use self::execute::ExecuteCommand;
pub use self::fill_buffer::BufferFillCommand;
pub use self::fill_buffer::BufferFillError;
pub use self::pipeline_barrier::PipelineBarrierCommand;
pub use self::render_pass::BeginRenderPassCommand;
pub use self::render_pass::NextSubpassCommand;
pub use self::render_pass::EndRenderPassCommand;
pub use self::update_buffer::BufferUpdateCommand;

macro_rules! error_ty {
    ($err_name:ident => $doc:expr, $($member:ident => $desc:expr,)*) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum $err_name {
            $(
                #[doc = $desc]
                $member
            ),*
        }

        impl error::Error for $err_name {
            #[inline]
            fn description(&self) -> &str {
                match *self {
                    $(
                        $err_name::$member => $desc,
                    )*
                }
            }
        }

        impl fmt::Display for $err_name {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
                write!(fmt, "{}", error::Error::description(self))
            }
        }
    };
}

mod bind_ib;
mod bind_pipeline;
mod bind_sets;
mod bind_vb;
mod copy_buffer;
mod copy_buffer_image;
mod dispatch;
mod draw;
mod execute;
mod fill_buffer;
mod pipeline_barrier;
mod render_pass;
mod update_buffer;

pub struct UnsafeCommandBufferBuilder<P> where P: CommandPool {
    cmd: Option<vk::CommandBuffer>,
    pool: Option<P>,
    device: Arc<Device>,

    // Flags that were used at creation.
    flags: Flags,

    // List of resources that must be kept alive because they are used by this command buffer.
    keep_alive: Vec<Arc<KeepAlive>>,

    // Current pipeline object binded to the graphics bind point. Includes all staging commands.
    current_graphics_pipeline: Option<vk::Pipeline>,

    // Current pipeline object binded to the compute bind point. Includes all staging commands.
    current_compute_pipeline: Option<vk::Pipeline>,

    // Current state of the dynamic state within the command buffer. Includes all staging commands.
    current_dynamic_state: DynamicState,

    // List of vertex buffers currently binded.
    current_vertex_buffers: SmallVec<[(vk::Buffer, vk::DeviceSize); 8]>,

    // Current index buffer.
    current_index_buffer: Option<(vk::Buffer, vk::DeviceSize, vk::IndexType)>,

    // True if we are a secondary command buffer.
    secondary_cb: bool,

    // True if we are within a render pass.
    within_render_pass: bool,
}

impl<P> UnsafeCommandBufferBuilder<P> where P: CommandPool {
    /// Creates a new builder.
    pub fn new<R, Rf>(pool: P, kind: Kind<R, Rf>, flags: Flags)
                      -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPass + 'static + Send + Sync,
              Rf: RenderPass + 'static + Send + Sync
    {
        let secondary = match kind {
            Kind::Primary => false,
            Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
        };

        let cmd = try!(pool.alloc(secondary, 1)).next().unwrap();
        
        match unsafe { UnsafeCommandBufferBuilder::already_allocated(pool, cmd, kind, flags) } {
            Ok(cmd) => Ok(cmd),
            Err(err) => {
                // FIXME: uncomment this and solve the fact that `pool` has been moved
                //unsafe { pool.free(secondary, Some(cmd.into()).into_iter()) };
                Err(err)
            },
        }
    }

    /// Creates a new command buffer builder from an already-allocated command buffer.
    ///
    /// # Safety
    ///
    /// - The allocated command buffer must belong to the pool and must not be used anywhere else
    ///   in the code for the duration of this command buffer.
    ///
    pub unsafe fn already_allocated<R, Rf>(pool: P, cmd: AllocatedCommandBuffer,
                                           kind: Kind<R, Rf>, flags: Flags)
                                           -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPass + 'static + Send + Sync,
              Rf: RenderPass + 'static + Send + Sync
    {
        let device = pool.device().clone();
        let vk = device.pointers();
        let cmd = cmd.internal_object();

        let mut keep_alive = Vec::new();

        let vk_flags = {
            let a = match flags {
                Flags::None => 0,
                Flags::SimultaneousUse => vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                Flags::OneTimeSubmit => vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            let b = match kind {
                Kind::Primary | Kind::Secondary => 0,
                Kind::SecondaryRenderPass { .. } => {
                    vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
                },
            };

            a | b
        };

        let (rp, sp) = if let Kind::SecondaryRenderPass { subpass, .. } = kind {
            keep_alive.push(subpass.render_pass().clone() as Arc<_>);
            (subpass.render_pass().render_pass().internal_object(), subpass.index())
        } else {
            (0, 0)
        };

        let framebuffer = if let Kind::SecondaryRenderPass { subpass, framebuffer: Some(ref framebuffer) } = kind {
            assert!(framebuffer.is_compatible_with(subpass.render_pass()));     // TODO: proper error
            keep_alive.push(framebuffer.clone() as Arc<_>);
            framebuffer.internal_object()
        } else {
            0
        };

        let inheritance = vk::CommandBufferInheritanceInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            pNext: ptr::null(),
            renderPass: rp,
            subpass: sp,
            framebuffer: framebuffer,
            occlusionQueryEnable: 0,            // TODO:
            queryFlags: 0,          // TODO:
            pipelineStatistics: 0,          // TODO:
        };

        let infos = vk::CommandBufferBeginInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: vk_flags,
            pInheritanceInfo: &inheritance,
        };

        try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));

        Ok(UnsafeCommandBufferBuilder {
            device: device.clone(),
            pool: Some(pool),
            cmd: Some(cmd),
            flags: flags,
            keep_alive: keep_alive,
            current_graphics_pipeline: None,
            current_compute_pipeline: None,
            current_dynamic_state: DynamicState::none(),
            current_vertex_buffers: SmallVec::new(),
            current_index_buffer: None,
            secondary_cb: match kind {
                Kind::Primary => false,
                Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
            },
            within_render_pass: match kind {
                Kind::Primary | Kind::Secondary => false,
                Kind::SecondaryRenderPass { .. } => true,
            },
        })
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<UnsafeCommandBuffer<P>, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let cmd = self.cmd.take().unwrap();
            try!(check_errors(vk.EndCommandBuffer(cmd)));

            Ok(UnsafeCommandBuffer {
                cmd: cmd,
                device: self.device.clone(),
                pool: self.pool.take().unwrap().finish(),
                flags: self.flags,
                already_submitted: AtomicBool::new(false),
                keep_alive: mem::replace(&mut self.keep_alive, Vec::new()),
                secondary_cb: self.secondary_cb,
            })
        }
    }

    /// Returns the pool used to create this command buffer builder.
    #[inline]
    pub fn pool(&self) -> &P {
        self.pool.as_ref().unwrap()
    }

    /// Returns the device this command buffer belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns true if this is a secondary command buffer.
    #[inline]
    pub fn is_secondary(&self) -> bool {
        self.secondary_cb
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd.unwrap()
    }
}

impl<P> Drop for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    #[inline]
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd {
            unsafe {
                let vk = self.device.pointers();
                vk.EndCommandBuffer(cmd);       // TODO: really needed?

                self.pool.as_ref().unwrap().free(self.secondary_cb, Some(cmd.into()).into_iter());
            }
        }
    }
}

pub struct UnsafeCommandBuffer<P> where P: CommandPool {
    // The Vulkan command buffer.
    cmd: vk::CommandBuffer,

    // Device that owns the command buffer.
    device: Arc<Device>,

    // Pool that owns the command buffer.
    pool: P::Finished,

    // Flags that were used at creation.
    flags: Flags,

    // True if the command buffer has always been submitted once. Only relevant if `flags` is
    // `OneTimeSubmit`.
    already_submitted: AtomicBool,

    // List of resources that must be kept alive because they are used by this command buffer.
    keep_alive: Vec<Arc<KeepAlive>>,

    // True if we are a secondary command buffer.
    secondary_cb: bool,
}

// Since the only moment where we access the pool is in the `Drop` trait, we can safely implement
// `Sync` on the command buffer.
// TODO: this could be generalized with a general-purpose wrapper that only allows &mut access
unsafe impl<P> Sync for UnsafeCommandBuffer<P> where P: CommandPool {}

impl<P> UnsafeCommandBuffer<P> where P: CommandPool {
    /// Returns the pool used to create this command buffer.
    #[inline]
    pub fn pool(&self) -> &P::Finished {
        &self.pool
    }

    /// Returns the device this command buffer belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns true if the command buffer was created with the `SimultaneousUse` flag.
    ///
    /// If false, then the command buffer must be properly synchronized.
    #[inline]
    pub fn simultaneous_use(&self) -> bool {
        self.flags == Flags::SimultaneousUse
    }

    /// Returns true if this is a secondary command buffer.
    #[inline]
    pub fn is_secondary(&self) -> bool {
        self.secondary_cb
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBuffer<P> where P: CommandPool {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd
    }
}

impl<P> Drop for UnsafeCommandBuffer<P> where P: CommandPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.pool.free(self.secondary_cb, Some(self.cmd.into()).into_iter());
        }
    }
}

pub struct UnsafeSubmission {
    // The fence to keep alive.
    fence: Option<Arc<Fence>>,

    // The queue on which this was submitted.
    queue: Arc<Queue>,

    // List of objects to keep alive because they are used by this submission.
    keep_alive: SmallVec<[Arc<KeepAlive>; 16]>,
}

impl UnsafeSubmission {
    /// Submits a bunch of batches to a queue and returns an `UnsafeSubmission` object.
    ///
    /// If `fence` is not `None`, it contains a fence that will be signalled when all the batches
    /// have finished execution.
    ///
    /// # Panic
    ///
    /// - Panicks if the stages are all `false` in one of the wait semaphores.
    /// - Panicks if all the objects don't belong to the same device.
    ///
    /// # Safety
    ///
    /// - The returned `UnsafeSubmission` object must not be destroyed before the submission
    ///   finishes.
    /// - The fence, if any, must not be signalled by another currently running submission.
    /// - Doesn't check that concurrency is correct. Resources and command buffers that were not
    ///   created with the `SimultaneousUse` flag must be correctly synchronized.
    /// - The signal semaphores must be in the unsignalled state.
    /// - The wait semaphores must be signalled by an earlier submission to this queue or another.
    ///
    pub unsafe fn new<I, P, Ci, Cb, W, S>(queue: &Arc<Queue>, fence: Option<&Arc<Fence>>, batches: I)
                                          -> Result<UnsafeSubmission, OomError>   // TODO: wrong error type
        where I: IntoIterator<Item = SubmissionDesc<Ci, W, S>>,
              Ci: Iterator<Item = Arc<Cb>>,
              Cb: CommandBuffer + Send + Sync + 'static,
              W: Iterator<Item = (Arc<Semaphore>, PipelineStages)>,
              S: Iterator<Item = Arc<Semaphore>>,
    {
        let device = queue.device().clone();

        // Objects to keep alive.
        let mut keep_alive: SmallVec<[_; 16]> = SmallVec::new();

        // Storage for semaphores and command buffers that we are going to pass to Vulkan.
        let mut semaphores_storage: SmallVec<[_; 8]> = SmallVec::new();
        let mut wait_stages_storage: SmallVec<[_; 8]> = SmallVec::new();
        let mut cbs_storage: SmallVec<[_; 8]> = SmallVec::new();

        // Iterating a first time to fill the storages and build incomplete vkSubmitInfo structs.
        let mut submit_infos = batches.into_iter().map(|mut sub| {
            let mut vk_info = vk::SubmitInfo {
                sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 0,
                pWaitSemaphores: ptr::null(),
                pWaitDstStageMask: ptr::null(),
                commandBufferCount: 0,
                pCommandBuffers: ptr::null(),
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };

            for cb in sub.command_buffers.by_ref() {
                if cb.inner_cb().flags == Flags::OneTimeSubmit {
                    let already_submitted = cb.inner_cb().already_submitted.swap(true, Ordering::Relaxed);
                    if already_submitted {
                        panic!("One-time CB was submitted twice")        // TODO: return error instead
                    }
                }

                assert_eq!(cb.device().internal_object(), device.internal_object());
                vk_info.commandBufferCount += 1;
                cbs_storage.push(cb.inner_cb().internal_object());
                keep_alive.push(cb as Arc<_>);
            }

            for (sem, stages) in sub.wait_semaphores.by_ref() {
                assert_eq!(sem.device().internal_object(), device.internal_object());
                assert!(stages != PipelineStages::none());
                vk_info.waitSemaphoreCount += 1;
                semaphores_storage.push(sem.internal_object());
                wait_stages_storage.push(stages.into());
                keep_alive.push(sem as Arc<_>);
            }

            for sem in sub.signal_semaphores.by_ref() {
                assert_eq!(sem.device().internal_object(), device.internal_object());
                vk_info.signalSemaphoreCount += 1;
                semaphores_storage.push(sem.internal_object());
                keep_alive.push(sem as Arc<_>);
            }

            vk_info
        }).collect::<SmallVec<[_; 16]>>();

        // Now that the storages are in the right positions on the heap, iterating over the info
        // again in order to fill the pointers.
        {
            let mut next_cb = cbs_storage.as_ptr();
            let mut next_stages = wait_stages_storage.as_ptr();
            let mut next_sem = semaphores_storage.as_ptr();

            for vk_info in submit_infos.iter_mut() {
                vk_info.pCommandBuffers = next_cb;
                next_cb = next_cb.offset(vk_info.commandBufferCount as isize);
                vk_info.pWaitDstStageMask = next_stages;
                vk_info.pWaitSemaphores = next_sem;
                next_stages = next_stages.offset(vk_info.waitSemaphoreCount as isize);
                next_sem = next_sem.offset(vk_info.waitSemaphoreCount as isize);
                vk_info.pSignalSemaphores = next_sem;
                next_sem = next_sem.offset(vk_info.signalSemaphoreCount as isize);
            }
        }

        // Submitting.
        {
            let fence = fence.as_ref().map(|f| f.internal_object()).unwrap_or(0);
            let queue = queue.internal_object_guard();      // TODO: maybe let the user lock themselves the queue?

            let vk = device.pointers();
            try!(check_errors(vk.QueueSubmit(*queue, submit_infos.len() as u32,
                                             submit_infos.as_ptr(), fence)));
        }

        Ok(UnsafeSubmission {
            fence: fence.as_ref().map(|&f| f.clone()),
            queue: queue.clone(),
            keep_alive: keep_alive,
        })
    }

    /// Returns the fence that is signalled when this submission is finished.
    #[inline]
    pub fn fence(&self) -> Option<&Arc<Fence>> {
        self.fence.as_ref()
    }

    /// Returns the `queue` the command buffers were submitted to.
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

/// Describes an individual batch that can be submitted to a queue.
pub struct SubmissionDesc<C, W, S> {
    /// List of command buffers to execute as part of this batch. The implementation will behave
    /// as if the command buffers were fused together in the order that is given. This means that
    /// pipeline barriers must take that into account.
    pub command_buffers: C,

    /// List of semaphores and stages to wait upon before this batch can start.
    pub wait_semaphores: W,

    /// List of semaphores to signal after this batch has finished.
    pub signal_semaphores: S,
}

/// Determines the kind of command buffer that we want to create.
#[derive(Clone)]        // TODO: Debug
pub enum Kind<'a, R: 'a, Rf> {
    /// A primary command buffer can execute all commands and can call secondary command buffers.
    Primary,

    /// A secondary command buffer can execute all dispatch and transfer operations, but not
    /// drawing operations.
    Secondary,

    /// A secondary command buffer within a render pass can only call draw operations that can
    /// be executed from within a specific subpass.
    SecondaryRenderPass {
        /// Which subpass this secondary command buffer can be called from.
        subpass: Subpass<'a, R>,
        /// The framebuffer object that will be used when calling the command buffer.
        /// This parameter is optional and is an optimization hint for the implementation.
        framebuffer: Option<Arc<Framebuffer<Rf>>>,
    },
}

/// Flags to pass when creating a command buffer.
///
/// The safest option is `SimultaneousUse`, but it may be slower than the other two.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Flags {
    /// The command buffer can be used multiple times, but must not execute more than once
    /// simultaneously.
    None,

    /// The command buffer can be executed multiple times in parallel.
    SimultaneousUse,

    /// The command buffer can only be submitted once. Any further submit is forbidden.
    OneTimeSubmit,
}

/// Dummy trait that is implemented on everything and that allows us to keep Arcs alive.
trait KeepAlive: 'static + Send + Sync {}
impl<T> KeepAlive for T where T: 'static + Send + Sync {}
