// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;

use buffer::BufferUsage;
use buffer::CpuAccessibleBuffer;
use command_buffer::synced::base::SyncCommandBufferBuilder;
use command_buffer::synced::base::SyncCommandBufferBuilderError;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use device::Device;

#[test]
fn basic_creation() {
    unsafe {
        let (device, queue) = gfx_dev_and_queue!();
        let pool = Device::standard_command_pool(&device, queue.family());
        SyncCommandBufferBuilder::new(&pool, Kind::primary(), Flags::None).unwrap();
    }
}

#[test]
fn basic_conflict() {
    unsafe {
        let (device, queue) = gfx_dev_and_queue!();

        let pool = Device::standard_command_pool(&device, queue.family());
        let mut sync = SyncCommandBufferBuilder::new(&pool, Kind::primary(), Flags::None).unwrap();

        let buf = CpuAccessibleBuffer::from_data(device, BufferUsage::all(), false, 0u32).unwrap();

        match sync.copy_buffer(buf.clone(), buf.clone(), iter::once((0, 0, 4))) {
            Err(SyncCommandBufferBuilderError::Conflict { .. }) => (),
            _ => panic!(),
        };
    }
}
