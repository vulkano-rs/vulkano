//! Recording commands to execute on the device.

#[allow(unused_imports)] // everything is exported for future-proofing
pub use self::commands::{clear::*, copy::*, dynamic_state::*, pipeline::*, sync::*};
use crate::{graph::ResourceMap, resource::DeathRow, Id};
use ash::vk;
use std::sync::Arc;
use vulkano::{
    buffer::Buffer,
    command_buffer::sys::RawRecordingCommandBuffer,
    device::{Device, DeviceOwned},
    image::Image,
    VulkanObject,
};

mod commands;

/// A command buffer in the recording state.
///
/// Unlike [`RawRecordingCommandBuffer`], this type has knowledge of the current task context and
/// can therefore validate resource accesses. (TODO)
pub struct RecordingCommandBuffer<'a> {
    inner: &'a mut RawRecordingCommandBuffer,
    accesses: ResourceAccesses<'a>,
    death_row: &'a mut DeathRow,
}

struct ResourceAccesses<'a> {
    resource_map: &'a ResourceMap<'a>,
}

impl<'a> RecordingCommandBuffer<'a> {
    pub(crate) unsafe fn new(
        inner: &'a mut RawRecordingCommandBuffer,
        resource_map: &'a ResourceMap<'a>,
        death_row: &'a mut DeathRow,
    ) -> Self {
        RecordingCommandBuffer {
            inner,
            accesses: ResourceAccesses { resource_map },
            death_row,
        }
    }

    /// Returns the underlying raw command buffer.
    ///
    /// While this method is safe, using the command buffer isn't. You must guarantee that any
    /// subresources you use while recording commands are either accounted for in the [task's
    /// access set], or that those subresources don't require any synchronization (including layout
    /// transitions and queue family ownership transfers), or that no other task is accessing the
    /// subresources at the same time without appropriate synchronization.
    #[inline]
    pub fn as_raw(&mut self) -> &mut RawRecordingCommandBuffer {
        self.inner
    }
}

unsafe impl DeviceOwned for RecordingCommandBuffer<'_> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl VulkanObject for RecordingCommandBuffer<'_> {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

impl<'a> ResourceAccesses<'a> {
    unsafe fn buffer_unchecked(&self, id: Id<Buffer>) -> &'a Arc<Buffer> {
        if id.is_virtual() {
            // SAFETY:
            // * The caller of `Task::execute` must ensure that `self.resource_map` maps the virtual
            //   IDs of the graph exhaustively.
            // * The caller must ensure that `id` is valid.
            unsafe { self.resource_map.buffer_unchecked(id) }.buffer()
        } else {
            let resources = self.resource_map.resources();

            // SAFETY:
            // * `ResourceMap` owns an `epoch::Guard`.
            // * The caller must ensure that `id` is valid.
            unsafe { resources.buffer_unchecked_unprotected(id) }.buffer()
        }
    }

    unsafe fn image_unchecked(&self, id: Id<Image>) -> &'a Arc<Image> {
        if id.is_virtual() {
            // SAFETY:
            // * The caller must ensure that `id` is valid.
            // * The caller of `Task::execute` must ensure that `self.resource_map` maps the virtual
            //   IDs of the graph exhaustively.
            unsafe { self.resource_map.image_unchecked(id) }.image()
        } else {
            let resources = self.resource_map.resources();

            // SAFETY:
            // * The caller must ensure that `id` is valid.
            // * `ResourceMap` owns an `epoch::Guard`.
            unsafe { resources.image_unchecked_unprotected(id) }.image()
        }
    }
}

type Result<T = (), E = Box<vulkano::ValidationError>> = ::std::result::Result<T, E>;
