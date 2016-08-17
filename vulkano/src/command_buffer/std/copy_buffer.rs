// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;

pub struct CopyCommand<L, B> where B: Buffer, L: CommandsList {
    previous: L,
    buffer: B,
    buffer_state: B::CommandBufferState,
    transition: Option<>,
}

impl<L, B> CommandsList for CopyCommand<L, B> where B: Buffer, L: CommandsList {
    pub fn new(previous: L, buffer: B) -> CopyCommand<L, B> {
        let (state, transition) = previous.current_buffer_state(buffer)
                                          .transition(false, self.num_commands(), ShaderStages, AccessFlagBits);
        assert!(transition.after_command_num < self.num_commands());
    }
}

unsafe impl<L, B> CommandsList for CopyCommand<L, B> where B: Buffer, L: CommandsList {
    fn num_commands(&self) -> usize {
        self.previous.num_commands() + 1
    }

    #[inline]
    fn requires_graphics_queue(&self) -> bool {
        self.previous.requires_graphics_queue()
    }

    #[inline]
    fn requires_compute_queue(&self) -> bool {
        self.previous.requires_compute_queue()
    }

    fn current_buffer_state<Ob>(&self, other: &Ob) -> Ob::CommandBufferState
        where Ob: Buffer
    {
        if self.buffer.is_same(b) {
            let s: &Ob::CommandBufferState = (&self.buffer_state as &Any).downcast_ref().unwrap();
            s.clone()
        } else {
            self.previous.current_buffer_state(b)
        }
    }

    unsafe fn build_unsafe_command_buffer<P, I>(&self, pool: P, transitions: I) -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        let my_command_num = self.num_commands();

        let mut transitions_to_apply = PipelineBarrierBuilder::new();

        let transitions = transitions.filter_map(|transition| {
            if transition.after_command_num >= my_command_num {
                transitions_to_apply.push(transition);
                None
            } else {
                Some(transition)
            }
        }).chain(self.transition.clone().into_iter());

        let mut parent_cb = self.previous.build_unsafe_command_buffer(pool, transitions.clone().chain());
        parent_cb.copy_buffer();
        parent_cb.pipeline_barrier(transitions_to_apply);
        parent_cb
    }
}

unsafe impl StdPrimaryCommandsList
