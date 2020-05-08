// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use VulkanObject;
use buffer::BufferAccess;
use command_buffer::DynamicState;
use descriptor::DescriptorSet;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::input_assembly::IndexType;
use smallvec::SmallVec;
use std::ops::Range;
use vk;

/// Keep track of the state of a command buffer builder, so that you don't need to bind objects
/// that were already bound.
///
/// > **Important**: Executing a secondary command buffer invalidates the state of a command buffer
/// > builder. When you do so, you need to call `invalidate()`.
pub struct StateCacher {
    // The dynamic state to synchronize with `CmdSetState`.
    dynamic_state: DynamicState,
    // The compute pipeline currently bound. 0 if nothing bound.
    compute_pipeline: vk::Pipeline,
    // The graphics pipeline currently bound. 0 if nothing bound.
    graphics_pipeline: vk::Pipeline,
    // The descriptor sets for the compute pipeline.
    compute_descriptor_sets: SmallVec<[vk::DescriptorSet; 12]>,
    // The descriptor sets for the graphics pipeline.
    graphics_descriptor_sets: SmallVec<[vk::DescriptorSet; 12]>,
    // If the user starts comparing descriptor sets, but drops the helper struct in the middle of
    // the processing then we will end up in a weird state. This bool is true when we start
    // comparing sets, and is set to false when we end up comparing. If it was true when we start
    // comparing, we know that something bad happened and we flush the cache.
    poisoned_descriptor_sets: bool,
    // The vertex buffers currently bound.
    vertex_buffers: SmallVec<[(vk::Buffer, vk::DeviceSize); 12]>,
    // Same as `poisoned_descriptor_sets` but for vertex buffers.
    poisoned_vertex_buffers: bool,
    // The index buffer, offset, and index type currently bound. `None` if nothing bound.
    index_buffer: Option<(vk::Buffer, usize, IndexType)>,
}

/// Outcome of an operation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StateCacherOutcome {
    /// The caller needs to perform the state change in the actual command buffer builder.
    NeedChange,
    /// The state change is not necessary.
    AlreadyOk,
}

impl StateCacher {
    /// Builds a new `StateCacher`.
    #[inline]
    pub fn new() -> StateCacher {
        StateCacher {
            dynamic_state: DynamicState::none(),
            compute_pipeline: 0,
            graphics_pipeline: 0,
            compute_descriptor_sets: SmallVec::new(),
            graphics_descriptor_sets: SmallVec::new(),
            poisoned_descriptor_sets: false,
            vertex_buffers: SmallVec::new(),
            poisoned_vertex_buffers: false,
            index_buffer: None,
        }
    }

    /// Resets the cache to its default state. You **must** call this after executing a secondary
    /// command buffer.
    #[inline]
    pub fn invalidate(&mut self) {
        self.dynamic_state = DynamicState::none();
        self.compute_pipeline = 0;
        self.graphics_pipeline = 0;
        self.compute_descriptor_sets = SmallVec::new();
        self.graphics_descriptor_sets = SmallVec::new();
        self.vertex_buffers = SmallVec::new();
        self.index_buffer = None;
    }

    /// Compares the current state with `incoming`, and returns a new state that contains the
    /// states that differ and that need to be actually set in the command buffer builder.
    ///
    /// This function also updates the state cacher. The state cacher assumes that the state
    /// changes are going to be performed after this function returns.
    pub fn dynamic_state(&mut self, incoming: &DynamicState) -> DynamicState {
        let mut changed = DynamicState::none();

        macro_rules! cmp {
            ($field:ident) => (
                if self.dynamic_state.$field != incoming.$field {
                    changed.$field = incoming.$field.clone();
                    if incoming.$field.is_some() {
                        self.dynamic_state.$field = incoming.$field.clone();
                    }
                }
            );
        }

        cmp!(line_width);
        cmp!(viewports);
        cmp!(scissors);
        cmp!(compare_mask);
        cmp!(reference);
        cmp!(write_mask);

        changed
    }

    /// Starts the process of comparing a list of descriptor sets to the descriptor sets currently
    /// in cache.
    ///
    /// After calling this function, call `add` for each set one by one. Then call `compare` in
    /// order to get the index of the first set to bind, or `None` if the sets were identical to
    /// what is in cache.
    ///
    /// This process also updates the state cacher. The state cacher assumes that the state
    /// changes are going to be performed after the `compare` function returns.
    #[inline]
    pub fn bind_descriptor_sets(&mut self, graphics: bool) -> StateCacherDescriptorSets {
        if self.poisoned_descriptor_sets {
            self.compute_descriptor_sets = SmallVec::new();
            self.graphics_descriptor_sets = SmallVec::new();
        }

        self.poisoned_descriptor_sets = true;

        StateCacherDescriptorSets {
            poisoned: &mut self.poisoned_descriptor_sets,
            state: if graphics {
                &mut self.graphics_descriptor_sets
            } else {
                &mut self.compute_descriptor_sets
            },
            offset: 0,
            found_diff: None,
        }
    }

    /// Checks whether we need to bind a graphics pipeline. Returns `StateCacherOutcome::AlreadyOk`
    /// if the pipeline was already bound earlier, and `StateCacherOutcome::NeedChange` if you need
    /// to actually bind the pipeline.
    ///
    /// This function also updates the state cacher. The state cacher assumes that the state
    /// changes are going to be performed after this function returns.
    pub fn bind_graphics_pipeline<P>(&mut self, pipeline: &P) -> StateCacherOutcome
        where P: GraphicsPipelineAbstract
    {
        let inner = GraphicsPipelineAbstract::inner(pipeline).internal_object();
        if inner == self.graphics_pipeline {
            StateCacherOutcome::AlreadyOk
        } else {
            self.graphics_pipeline = inner;
            StateCacherOutcome::NeedChange
        }
    }

    /// Checks whether we need to bind a compute pipeline. Returns `StateCacherOutcome::AlreadyOk`
    /// if the pipeline was already bound earlier, and `StateCacherOutcome::NeedChange` if you need
    /// to actually bind the pipeline.
    ///
    /// This function also updates the state cacher. The state cacher assumes that the state
    /// changes are going to be performed after this function returns.
    pub fn bind_compute_pipeline<P>(&mut self, pipeline: &P) -> StateCacherOutcome
        where P: ComputePipelineAbstract
    {
        let inner = pipeline.inner().internal_object();
        if inner == self.compute_pipeline {
            StateCacherOutcome::AlreadyOk
        } else {
            self.compute_pipeline = inner;
            StateCacherOutcome::NeedChange
        }
    }

    /// Starts the process of comparing a list of vertex buffers to the vertex buffers currently
    /// in cache.
    ///
    /// After calling this function, call `add` for each set one by one. Then call `compare` in
    /// order to get the range of the vertex buffers to bind, or `None` if the sets were identical
    /// to what is in cache.
    ///
    /// This process also updates the state cacher. The state cacher assumes that the state
    /// changes are going to be performed after the `compare` function returns.
    #[inline]
    pub fn bind_vertex_buffers(&mut self) -> StateCacherVertexBuffers {
        if self.poisoned_vertex_buffers {
            self.vertex_buffers = SmallVec::new();
        }

        self.poisoned_vertex_buffers = true;

        StateCacherVertexBuffers {
            poisoned: &mut self.poisoned_vertex_buffers,
            state: &mut self.vertex_buffers,
            offset: 0,
            first_diff: None,
            last_diff: 0,
        }
    }

    /// Checks whether we need to bind an index buffer. Returns `StateCacherOutcome::AlreadyOk`
    /// if the index buffer was already bound earlier, and `StateCacherOutcome::NeedChange` if you
    /// need to actually bind the buffer.
    ///
    /// This function also updates the state cacher. The state cacher assumes that the state
    /// changes are going to be performed after this function returns.
    pub fn bind_index_buffer<B>(&mut self, index_buffer: &B, ty: IndexType) -> StateCacherOutcome
        where B: ?Sized + BufferAccess
    {
        let value = {
            let inner = index_buffer.inner();
            (inner.buffer.internal_object(), inner.offset, ty)
        };

        if self.index_buffer == Some(value) {
            StateCacherOutcome::AlreadyOk
        } else {
            self.index_buffer = Some(value);
            StateCacherOutcome::NeedChange
        }
    }
}

/// Helper struct for comparing descriptor sets.
///
/// > **Note**: For reliability reasons, if you drop/leak this struct before calling `compare` then
/// > the cache of the currently bound descriptor sets will be reset.
pub struct StateCacherDescriptorSets<'s> {
    // Reference to the parent's `poisoned_descriptor_sets`.
    poisoned: &'s mut bool,
    // Reference to the descriptor sets list to compare to.
    state: &'s mut SmallVec<[vk::DescriptorSet; 12]>,
    // Next offset within the list to compare to.
    offset: usize,
    // Contains the return value of `compare`.
    found_diff: Option<u32>,
}

impl<'s> StateCacherDescriptorSets<'s> {
    /// Adds a descriptor set to the list to compare.
    #[inline]
    pub fn add<S>(&mut self, set: &S)
        where S: ?Sized + DescriptorSet
    {
        let raw = set.inner().internal_object();

        if self.offset < self.state.len() {
            if self.state[self.offset] == raw {
                self.offset += 1;
                return;
            }

            self.state[self.offset] = raw;

        } else {
            self.state.push(raw);
        }

        if self.found_diff.is_none() {
            self.found_diff = Some(self.offset as u32);
        }
        self.offset += 1;
    }

    /// Compares your list to the list in cache, and returns the offset of the first set to bind.
    /// Returns `None` if the two lists were identical.
    ///
    /// After this function returns, the cache will be updated to match your list.
    #[inline]
    pub fn compare(self) -> Option<u32> {
        *self.poisoned = false;
        // Removing from the cache any set that wasn't added with `add`.
        self.state.truncate(self.offset);
        self.found_diff
    }
}

/// Helper struct for comparing vertex buffers.
///
/// > **Note**: For reliability reasons, if you drop/leak this struct before calling `compare` then
/// > the cache of the currently bound vertex buffers will be reset.
pub struct StateCacherVertexBuffers<'s> {
    // Reference to the parent's `poisoned_vertex_buffers`.
    poisoned: &'s mut bool,
    // Reference to the vertex buffers list to compare to.
    state: &'s mut SmallVec<[(vk::Buffer, vk::DeviceSize); 12]>,
    // Next offset within the list to compare to.
    offset: usize,
    // Contains the offset of the first vertex buffer that differs.
    first_diff: Option<u32>,
    // Offset of the last vertex buffer that differs.
    last_diff: u32,
}

impl<'s> StateCacherVertexBuffers<'s> {
    /// Adds a vertex buffer to the list to compare.
    #[inline]
    pub fn add<B>(&mut self, buffer: &B)
        where B: ?Sized + BufferAccess
    {
        let raw = {
            let inner = buffer.inner();
            let raw = inner.buffer.internal_object();
            let offset = inner.offset as vk::DeviceSize;
            (raw, offset)
        };

        if self.offset < self.state.len() {
            if self.state[self.offset] == raw {
                self.offset += 1;
                return;
            }

            self.state[self.offset] = raw;

        } else {
            self.state.push(raw);
        }

        self.last_diff = self.offset as u32;
        if self.first_diff.is_none() {
            self.first_diff = Some(self.offset as u32);
        }
        self.offset += 1;
    }

    /// Compares your list to the list in cache, and returns the range of the vertex buffers to
    /// bind. Returns `None` if the two lists were identical.
    ///
    /// After this function returns, the cache will be updated to match your list.
    ///
    /// > **Note**: Keep in mind that `range.end` is *after* the last element. For example the
    /// > range `1 .. 2` only contains one element.
    #[inline]
    pub fn compare(self) -> Option<Range<u32>> {
        *self.poisoned = false;

        // Removing from the cache any set that wasn't added with `add`.
        self.state.truncate(self.offset);

        self.first_diff.map(|first| {
                                debug_assert!(first <= self.last_diff);
                                first .. (self.last_diff + 1)
                            })
    }
}

#[cfg(test)]
mod tests {
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;
    use command_buffer::state_cacher::StateCacher;

    #[test]
    fn vb_caching_single() {
        let (device, queue) = gfx_dev_and_queue!();

        const EMPTY: [i32; 0] = [];
        let buf =
            CpuAccessibleBuffer::from_data(device, BufferUsage::vertex_buffer(), false, EMPTY.iter())
                .unwrap();

        let mut cacher = StateCacher::new();

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf);
            assert_eq!(bind_vb.compare(), Some(0 .. 1));
        }

        for _ in 0 .. 3 {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf);
            assert_eq!(bind_vb.compare(), None);
        }
    }

    #[test]
    fn vb_caching_invalidated() {
        let (device, queue) = gfx_dev_and_queue!();

        const EMPTY: [i32; 0] = [];
        let buf =
            CpuAccessibleBuffer::from_data(device, BufferUsage::vertex_buffer(), false, EMPTY.iter())
                .unwrap();

        let mut cacher = StateCacher::new();

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf);
            assert_eq!(bind_vb.compare(), Some(0 .. 1));
        }

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf);
            assert_eq!(bind_vb.compare(), None);
        }

        cacher.invalidate();

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf);
            assert_eq!(bind_vb.compare(), Some(0 .. 1));
        }
    }

    #[test]
    fn vb_caching_multi() {
        let (device, queue) = gfx_dev_and_queue!();

        const EMPTY: [i32; 0] = [];
        let buf1 = CpuAccessibleBuffer::from_data(device.clone(),
                                                  BufferUsage::vertex_buffer(),
                                                  false,
                                                  EMPTY.iter())
            .unwrap();
        let buf2 = CpuAccessibleBuffer::from_data(device.clone(),
                                                  BufferUsage::vertex_buffer(),
                                                  false,
                                                  EMPTY.iter())
            .unwrap();
        let buf3 = CpuAccessibleBuffer::from_data(device,
                                                  BufferUsage::vertex_buffer(),
                                                  false,
                                                  EMPTY.iter())
            .unwrap();

        let mut cacher = StateCacher::new();

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf1);
            bind_vb.add(&buf2);
            assert_eq!(bind_vb.compare(), Some(0 .. 2));
        }

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf1);
            bind_vb.add(&buf2);
            bind_vb.add(&buf3);
            assert_eq!(bind_vb.compare(), Some(2 .. 3));
        }

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf1);
            assert_eq!(bind_vb.compare(), None);
        }

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf1);
            bind_vb.add(&buf3);
            assert_eq!(bind_vb.compare(), Some(1 .. 2));
        }

        {
            let mut bind_vb = cacher.bind_vertex_buffers();
            bind_vb.add(&buf2);
            bind_vb.add(&buf3);
            assert_eq!(bind_vb.compare(), Some(0 .. 1));
        }
    }
}
