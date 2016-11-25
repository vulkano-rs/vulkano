// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::error::Error;
use std::mem;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use command_buffer::cb::CommandsListBuildPrimaryPool;
use command_buffer::cb::Flags;
use command_buffer::cb::Kind;
use command_buffer::cb::UnsyncedCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::submit::Submit;
use command_buffer::submit::SubmitBuilder;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use command_buffer::CommandsListSinkCaller;
use command_buffer::PipelineBarrierBuilder;
use command_buffer::SecondaryCommandBuffer;
use device::Device;
use device::Queue;
use image::Layout;
use image::Image;
use sync::AccessFlagBits;
use sync::PipelineStages;
use VulkanObject;
use vk;

use OomError;

pub struct AutobarriersCommandBuffer<L, P> where P: CommandPool {
    // The actual command buffer. 
    inner: UnsyncedCommandBuffer<WrappedCommandsList<L>, P>
}

impl<L, P> CommandsListBuildPrimaryPool<L, P> for AutobarriersCommandBuffer<L, P>
    where L: CommandsList, P: CommandPool
{
    fn build_primary_with_pool(pool: P, list: L)
                               -> Result<AutobarriersCommandBuffer<L, P>, OomError>
        where Self: Sized
    {
        let kind = Kind::primary();
        let flags = Flags::SimultaneousUse;

        let cmd = unsafe {
            let device = pool.device().clone();
            try!(UnsyncedCommandBuffer::new(WrappedCommandsList(list, device), pool, kind, flags))
        };

        Ok(AutobarriersCommandBuffer {
            inner: cmd,
        })
    }
}

unsafe impl<L, P> Submit for AutobarriersCommandBuffer<L, P>
    where L: CommandsList, P: CommandPool
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }

    unsafe fn append_submission<'a>(&'a self, base: SubmitBuilder<'a>, queue: &Arc<Queue>)
                                    -> Result<SubmitBuilder<'a>, Box<Error>>
    {
        // FIXME: totally unsynchronized here
        Ok(base.add_command_buffer(&self.inner))
    }
}

// TODO: we're not necessarily a secondary command buffer
unsafe impl<L, P> SecondaryCommandBuffer for AutobarriersCommandBuffer<L, P>
    where L: CommandsList, P: CommandPool
{
    #[inline]
    fn inner(&self) -> vk::CommandBuffer {
        self.inner.internal_object()
    }

    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }

    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.inner.commands_list().append(builder);
    }
}

struct WrappedCommandsList<L>(L, Arc<Device>);
unsafe impl<L> CommandsList for WrappedCommandsList<L> where L: CommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        let mut sink = Sink {
            output: builder,
            device: &self.1,
            accesses: HashMap::new(),
            pending_accesses: HashMap::new(),
            pending_commands: Vec::new(),
        };

        self.0.append(&mut sink);
        sink.flush();
    }
}

// Helper object for AutobarriersCommandBuffer. Implementation detail.
//
// This object is created in a local scope when the command is built, and destroyed after all the
// commands have been passed through.
struct Sink<'c: 'o, 'o> {
    output: &'o mut CommandsListSink<'c>,
    device: &'o Arc<Device>,
    accesses: HashMap<u64, SmallVec<[Element<'c>; 2]>>,
    pending_accesses: HashMap<u64, SmallVec<[Element<'c>; 2]>>,
    pending_commands: Vec<Box<CommandsListSinkCaller<'c> + 'c>>,
}

impl<'c: 'o, 'o> Sink<'c, 'o> {
    fn flush(&mut self) {
        let mut pipeline_barrier = PipelineBarrierBuilder::new();

        for (key, accesses) in self.pending_accesses.drain() {
            let prev_accesses = match self.accesses.entry(key) {
                Entry::Occupied(e) => e.into_mut(),     // TODO: for images, need to transition from initial layout
                Entry::Vacant(e) => { e.insert(accesses); continue; }
            };

            for prev_access in mem::replace(prev_accesses, SmallVec::new()).into_iter() {
                let mut found_conflict = false;

                for access in accesses.iter() {
                    match (&prev_access.inner, &access.inner) {
                        (&ElementInner::Buffer { buffer: old_buffer, offset: old_offset,
                                                 size: old_size, write: old_write },
                         &ElementInner::Buffer { buffer: new_buffer, offset: new_offset,
                                                 size: new_size, write: new_write }) =>
                        {
                            if !old_buffer.conflicts_buffer(old_offset, old_size, old_write,
                                                            new_buffer, new_offset, new_size,
                                                            new_write)
                            {
                                continue;
                            }

                            found_conflict = true;

                            if !old_write {
                                unsafe {
                                    pipeline_barrier.add_execution_dependency(prev_access.stages,
                                                                              access.stages, true);
                                }
                            } else {
                                let real_offset = cmp::min(old_offset, new_offset);
                                let real_size = if old_offset < new_offset {
                                    cmp::max(old_size, new_size + (new_offset - old_offset))
                                } else {
                                    cmp::max(new_size, old_size + (old_offset - new_offset))
                                };

                                unsafe {
                                    pipeline_barrier.add_buffer_memory_barrier(old_buffer, prev_access.stages,
                                                                               prev_access.access, access.stages,
                                                                               access.access, true, None,
                                                                               real_offset, real_size);
                                }
                            }
                        },

                        (&ElementInner::Image { image: self_image, first_layer: self_fl, num_layers: self_nl,
                                                first_mipmap: self_fm, num_mipmaps: self_nm,
                                                write: self_write, layout: self_layout },
                         &ElementInner::Image { image: other_image, first_layer: other_fl, num_layers: other_nl,
                                                first_mipmap: other_fm, num_mipmaps: other_nm,
                                                write: other_write, layout: other_layout }) =>
                        {
                            if self_layout == other_layout &&
                                !self_image.conflicts_image(self_fl, self_nl, self_fm, self_nm, self_write,
                                                            other_image, other_fl, other_nl, other_fm, other_nm,
                                                            other_write)
                            {
                                continue;
                            }

                            found_conflict = true;
                            
                            if !self_write && self_layout == other_layout {
                                unsafe {
                                    pipeline_barrier.add_execution_dependency(prev_access.stages,
                                                                              access.stages, true);
                                }
                            } else {
                                let real_first_layer = cmp::min(self_fl, other_fl);
                                let real_num_layers = if self_fl < other_fl {
                                    cmp::max(self_nl, other_nl + (other_fl - self_fl))
                                } else {
                                    cmp::max(other_nl, self_nl + (self_fl - other_fl))
                                };

                                let real_first_mipmap = cmp::min(self_fm, other_fm);
                                let real_num_mipmaps = if self_fm < other_fm {
                                    cmp::max(self_nm, other_nm + (other_fm - self_fm))
                                } else {
                                    cmp::max(other_nm, self_nm + (self_fm - other_fm))
                                };

                                unsafe {
                                    pipeline_barrier.add_image_memory_barrier(self_image, real_first_layer .. real_first_layer + real_num_layers,
                                                                              real_first_mipmap .. real_first_mipmap + real_num_mipmaps, prev_access.stages,
                                                                              prev_access.access, access.stages,
                                                                              access.access, true, None,
                                                                              self_layout, other_layout);
                                }
                            }
                        },

                        (&ElementInner::Buffer { buffer, offset, size, write: self_write },
                         &ElementInner::Image { image, first_layer, num_layers, first_mipmap, num_mipmaps,
                                                write: other_write, .. }) |
                        (&ElementInner::Image { image, first_layer, num_layers, first_mipmap, num_mipmaps,
                                                write: other_write, .. },
                         &ElementInner::Buffer { buffer, offset, size, write: self_write }) =>
                        {
                            if !buffer.conflicts_image(offset, size, self_write, image, first_layer, num_layers,
                                                       first_mipmap, num_mipmaps, other_write)
                            {
                                continue;
                            }
                            
                            //found_conflict = true;    // there's a warning if we uncomment that
                            unimplemented!()        // TODO:
                        },
                    }
                }

                if !found_conflict {
                    prev_accesses.push(prev_access);
                }
            }

            for access in accesses.into_iter() {
                prev_accesses.push(access);
            }
        }

        pipeline_barrier.append_to(self.output);
        for cmd in self.pending_commands.drain(..) {
            self.output.add_command(cmd);
        }
    }
}

impl<'c: 'o, 'o> CommandsListSink<'c> for Sink<'c, 'o> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device
    }

    #[inline]
    fn add_command(&mut self, f: Box<CommandsListSinkCaller<'c> + 'c>) {
        self.pending_commands.push(f);
    }

    #[inline]
    fn add_buffer_transition(&mut self, buffer: &'c Buffer, offset: usize, size: usize, write: bool,
                             stages: PipelineStages, access: AccessFlagBits)
    {
        let key = buffer.conflict_key(offset, size, write);

        let element = Element {
            stages: stages,
            access: access,
            inner: ElementInner::Buffer {
                buffer: buffer,
                offset: offset,
                size: size,
                write: write,
            },
        };

        if self.pending_accesses.get(&key).map(|l| l.iter().any(|e| e.conflicts(&element))).unwrap_or(false) {
            self.flush();
        }

        self.pending_accesses.entry(key).or_insert_with(|| SmallVec::new()).push(element);
    }

    #[inline]
    fn add_image_transition(&mut self, image: &'c Image, first_layer: u32, num_layers: u32,
                            first_mipmap: u32, num_mipmaps: u32, write: bool, layout: Layout,
                            stages: PipelineStages, access: AccessFlagBits)
    {
        let key = image.conflict_key(first_layer, num_layers, first_mipmap, num_mipmaps, write);

        let element = Element {
            stages: stages,
            access: access,
            inner: ElementInner::Image {
                image: image,
                first_layer: first_layer,
                num_layers: num_layers,
                first_mipmap: first_mipmap,
                num_mipmaps: num_mipmaps,
                write: write,
                layout: layout,
            },
        };

        if self.pending_accesses.get(&key).map(|l| l.iter().any(|e| e.conflicts(&element))).unwrap_or(false) {
            self.flush();
        }

        self.pending_accesses.entry(key).or_insert_with(|| SmallVec::new()).push(element);
    }

    #[inline]
    fn add_image_transition_notification(&mut self, _: &Image, _: u32, _: u32, _: u32,
                                         _: u32, _: Layout, _: PipelineStages, _: AccessFlagBits)
    {
        // FIXME: unimplemented
    }
}

#[derive(Clone)]
struct Element<'a> {
    stages: PipelineStages,
    access: AccessFlagBits,
    inner: ElementInner<'a>,
}

impl<'a> Element<'a> {
    fn conflicts(&self, other: &Element<'a>) -> bool {
        match (&self.inner, &other.inner) {
            (&ElementInner::Buffer { buffer: self_buffer, offset: self_offset, size: self_size,
                                     write: self_write },
             &ElementInner::Buffer { buffer: other_buffer, offset: other_offset, size: other_size,
                                     write: other_write }) =>
            {
                self_buffer.conflicts_buffer(self_offset, self_size, self_write, other_buffer,
                                             other_offset, other_size, other_write)
            },

            (&ElementInner::Buffer { buffer, offset, size, write: self_write },
             &ElementInner::Image { image, first_layer, num_layers, first_mipmap, num_mipmaps,
                                    write: other_write, .. }) |
            (&ElementInner::Image { image, first_layer, num_layers, first_mipmap, num_mipmaps,
                                    write: other_write, .. },
             &ElementInner::Buffer { buffer, offset, size, write: self_write }) =>
            {
                buffer.conflicts_image(offset, size, self_write, image, first_layer, num_layers,
                                       first_mipmap, num_mipmaps, other_write)
            },

            (&ElementInner::Image { image: self_image, first_layer: self_fl, num_layers: self_nl,
                                    first_mipmap: self_fm, num_mipmaps: self_nm,
                                    write: self_write, layout: self_layout },
             &ElementInner::Image { image: other_image, first_layer: other_fl, num_layers: other_nl,
                                    first_mipmap: other_fm, num_mipmaps: other_nm,
                                    write: other_write, layout: other_layout }) =>
            {
                self_layout != other_layout ||
                    self_image.conflicts_image(self_fl, self_nl, self_fm, self_nm, self_write,
                                               other_image, other_fl, other_nl, other_fm, other_nm,
                                               other_write)
            },
        }
    }
}

#[derive(Clone)]
enum ElementInner<'a> {
    Buffer {
        buffer: &'a Buffer,
        offset: usize,
        size: usize,
        write: bool,
    },
    Image {
        image: &'a Image,
        first_layer: u32,
        num_layers: u32,
        first_mipmap: u32,
        num_mipmaps: u32,
        write: bool,
        layout: Layout,
    },
}
