// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferUsage;
use crate::buffer::CpuAccessibleBuffer;
use crate::command_buffer::pool::CommandPool;
use crate::command_buffer::pool::CommandPoolBuilderAlloc;
use crate::command_buffer::synced::base::SyncCommandBufferBuilder;
use crate::command_buffer::synced::base::SyncCommandBufferBuilderError;
use crate::command_buffer::sys::Flags;
use crate::command_buffer::Kind;
use crate::device::Device;
use std::iter;

#[test]
fn basic_creation() {
    unsafe {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = Device::standard_command_pool(&device, queue.family());
        let pool_builder_alloc = pool.alloc(false, 1).unwrap().next().unwrap();
        SyncCommandBufferBuilder::new(&pool_builder_alloc.inner(), Kind::primary(), Flags::None)
            .unwrap();
    }
}

#[test]
fn basic_conflict() {
    unsafe {
        let (device, queue) = gfx_dev_and_queue!();

        let pool = Device::standard_command_pool(&device, queue.family());
        let pool_builder_alloc = pool.alloc(false, 1).unwrap().next().unwrap();
        let mut sync = SyncCommandBufferBuilder::new(
            &pool_builder_alloc.inner(),
            Kind::primary(),
            Flags::None,
        )
        .unwrap();
        let buf = CpuAccessibleBuffer::from_data(device, BufferUsage::all(), false, 0u32).unwrap();

        match sync.copy_buffer(buf.clone(), buf.clone(), iter::once((0, 0, 4))) {
            Err(SyncCommandBufferBuilderError::Conflict { .. }) => (),
            _ => panic!(),
        };
    }
}
