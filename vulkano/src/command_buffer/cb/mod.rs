// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use command_buffer::pool::StandardCommandPool;
use device::Device;
use instance::QueueFamily;
use OomError;

pub use self::autobarriers::AutobarriersCommandBuffer;
pub use self::unsynced::Flags;
pub use self::unsynced::Kind;
pub use self::unsynced::UnsyncedCommandBuffer;

mod autobarriers;
mod unsynced;

pub trait CommandsListBuildPrimaryPool<L, P> {
    fn build_primary_with_pool(pool: P, list: L) -> Result<Self, OomError> where Self: Sized;
}

pub trait CommandsListBuildPrimary<L> {
    fn build_primary(device: &Arc<Device>, queue_family: QueueFamily, list: L) -> Result<Self, OomError> where Self: Sized;
}

impl<T, L> CommandsListBuildPrimary<L> for T
    where T: CommandsListBuildPrimaryPool<L, Arc<StandardCommandPool>>
{
    fn build_primary(device: &Arc<Device>, queue_family: QueueFamily, list: L) -> Result<Self, OomError> {
        let pool = Device::standard_command_pool(device, queue_family);
        CommandsListBuildPrimaryPool::build_primary_with_pool(pool, list)
    }
}
