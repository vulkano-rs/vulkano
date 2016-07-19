// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Helps choosing the pipeline barriers that are required when building or submitting a command
//! buffer.

// Note that everything here is unstable, as the current "blocks" system may be reworked.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash;
use std::hash::BuildHasherDefault;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::u64;
use fnv::FnvHasher;
use smallvec::SmallVec;

use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolFinished;
use command_buffer::pool::StandardCommandPool;
use descriptor::descriptor_set::DescriptorSetsCollection;

/// Helps building the pipeline barriers while constructing a command buffer.
pub struct BuilderSyncHelper {
    // List of accesses made by this command buffer to buffers and images, exclusing the staging
    // commands and the staging render pass.
    //
    // If a buffer/image is missing in this list, that means it hasn't been used by this command
    // buffer yet and is still in its default state.
    //
    // This list is only updated by the `flush()` function.
    buffers_state: HashMap<(vk::Buffer, usize), InternalBufferBlockAccess,
                           BuildHasherDefault<FnvHasher>>,
    images_state: HashMap<(vk::Image, (u32, u32)), InternalImageBlockAccess,
                          BuildHasherDefault<FnvHasher>>,

    // List of commands that are waiting to be submitted to the Vulkan command buffer. Doesn't
    // include commands that were submitted within a render pass.
    staging_commands: Vec<Box<FnMut(&vk::DevicePointers, vk::CommandBuffer) + Send + Sync>>,

    // List of resources accesses made by the comands in `staging_commands`. Doesn't include
    // commands added to the current render pass.
    staging_required_buffer_accesses: HashMap<(vk::Buffer, usize), InternalBufferBlockAccess,
                                              BuildHasherDefault<FnvHasher>>,
    staging_required_image_accesses: HashMap<(vk::Image, (u32, u32)), InternalImageBlockAccess,
                                             BuildHasherDefault<FnvHasher>>,

    // List of commands that are waiting to be submitted to the Vulkan command buffer when we're
    // inside a render pass. Flushed when `end_renderpass` is called.
    render_pass_staging_commands: Vec<Box<FnMut(&vk::DevicePointers, vk::CommandBuffer) +
                                          Send + Sync>>,

    // List of resources accesses made by the current render pass. Merged with
    // `staging_required_buffer_accesses` and `staging_required_image_accesses` when
    // `end_renderpass` is called.
    render_pass_staging_required_buffer_accesses: HashMap<(vk::Buffer, usize),
                                                          InternalBufferBlockAccess,
                                                          BuildHasherDefault<FnvHasher>>,
    render_pass_staging_required_image_accesses: HashMap<(vk::Image, (u32, u32)),
                                                         InternalImageBlockAccess,
                                                         BuildHasherDefault<FnvHasher>>,
}

impl BuilderSyncHelper {
    #[inline]
    pub fn new() -> BuilderSyncHelper {
        buffers_state: HashMap::new(),
        images_state: HashMap::new(),
        staging_commands: Vec::new(),
        staging_required_buffer_accesses: HashMap::new(),
        staging_required_image_accesses: HashMap::new(),
        render_pass_staging_commands: Vec::new(),
        render_pass_staging_required_buffer_accesses: HashMap::new(),
        render_pass_staging_required_image_accesses: HashMap::new(),
    }
}
