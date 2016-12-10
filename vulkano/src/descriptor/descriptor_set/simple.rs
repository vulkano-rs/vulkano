// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferViewRef;
use command_buffer::cmd::CommandsListSink;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::TrackedDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::DescriptorPoolAlloc;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor_set::DescriptorWrite;
use descriptor::descriptor_set::StdDescriptorPool;
use descriptor::pipeline_layout::PipelineLayoutRef;
use device::Device;
use image::ImageView;
use image::sys::Layout;
use sync::AccessFlagBits;
use sync::PipelineStages;

/// A simple immutable descriptor set.
///
/// It is named "simple" because creating such a descriptor set allocates from a pool, and because
/// it can't be modified once created. It is sufficient for most usages, but in some situations
/// you may wish to use something more optimized instead.
///
/// In order to build a `SimpleDescriptorSet`, you need to use a `SimpleDescriptorSetBuilder`. But
/// the easiest way is to use the `simple_descriptor_set!` macro.
///
/// The template parameter of the `SimpleDescriptorSet` is very complex, and you shouldn't try to
/// express it explicitely. If you want to store your descriptor set in a struct or in a `Vec` for
/// example, you are encouraged to turn `SimpleDescriptorSet` into a `Box<DescriptorSet>` or a
/// `Arc<DescriptorSet>`.
///
/// # Example
// TODO:
pub struct SimpleDescriptorSet<R, P = Arc<StdDescriptorPool>> where P: DescriptorPool {
    inner: P::Alloc,
    resources: R,
    layout: Arc<UnsafeDescriptorSetLayout>
}

impl<R, P> SimpleDescriptorSet<R, P> where P: DescriptorPool {
    /// Returns the layout used to create this descriptor set.
    #[inline]
    pub fn set_layout(&self) -> &Arc<UnsafeDescriptorSetLayout> {
        &self.layout
    }
}

unsafe impl<R, P> DescriptorSet for SimpleDescriptorSet<R, P> where P: DescriptorPool {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.inner()
    }
}

unsafe impl<R, P> TrackedDescriptorSet for SimpleDescriptorSet<R, P>
    where R: SimpleDescriptorSetResourcesCollection,
          P: DescriptorPool
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        self.resources.add_transition(sink)
    }
}

/// Builds a descriptor set in the form of a `SimpleDescriptorSet` object.
// TODO: more doc
#[macro_export]
macro_rules! simple_descriptor_set {
    ($layout:expr, $set_num:expr, {$($name:ident: $val:expr),*$(,)*}) => ({
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetBuilder;
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetBufferExt;

        let builder = SimpleDescriptorSetBuilder::new($layout, $set_num);

        $(
            let builder = $val.add_me(builder, stringify!($name));
        )*

        builder.build()
    });
}

/// Prototype of a `SimpleDescriptorSet`.
///
/// > **Note**: You are encouraged to use the `simple_descriptor_set!` macro instead of
/// > manipulating these internals.
///
/// # Example
// TODO: example here
pub struct SimpleDescriptorSetBuilder<L, R> {
    layout: L,
    set_id: usize,
    writes: Vec<DescriptorWrite>,
    resources: R,
}

impl<L> SimpleDescriptorSetBuilder<L, ()> where L: PipelineLayoutRef {
    /// Builds a new prototype for a `SimpleDescriptorSet`. Requires a reference to a pipeline
    /// layout, and the id of the set within the layout.
    ///
    /// # Panic
    ///
    /// - Panics if the set id is out of range.
    ///
    pub fn new(layout: L, set_id: usize) -> SimpleDescriptorSetBuilder<L, ()> {
        assert!(layout.desc().num_sets() > set_id);

        SimpleDescriptorSetBuilder {
            layout: layout,
            set_id: set_id,
            writes: Vec::new(),
            resources: (),
        }
    }
}

impl<L, R> SimpleDescriptorSetBuilder<L, R> where L: PipelineLayoutRef {
    /// Builds a `SimpleDescriptorSet` from the builder.
    pub fn build(self) -> SimpleDescriptorSet<R, Arc<StdDescriptorPool>> {
        // TODO: check that we filled everything
        let pool = Device::standard_descriptor_pool(self.layout.device());
        let set_layout = self.layout.descriptor_set_layout(self.set_id).unwrap().clone();       // FIXME: error

        let set = unsafe {
            let mut set = pool.alloc(&set_layout).unwrap();      // FIXME: error
            set.inner_mut().write(pool.device(), self.writes.into_iter());
            set
        };

        SimpleDescriptorSet {
            inner: set,
            resources: self.resources,
            layout: set_layout,
        }
    }
}

/// Trait implemented on buffer values that can be appended to a simple descriptor set builder.
pub unsafe trait SimpleDescriptorSetBufferExt<L, R> {
    type Out;

    // TODO: return Result
    fn add_me(self, i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>;
}

unsafe impl<L, R, T> SimpleDescriptorSetBufferExt<L, R> for T
    where T: Buffer, L: PipelineLayoutRef
{
    type Out = (R, SimpleDescriptorSetBuf<T>);

    fn add_me(self, mut i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>
    {
        let (set_id, binding_id) = i.layout.desc().descriptor_by_name(name).unwrap();    // TODO: Result instead
        assert_eq!(set_id, i.set_id);       // TODO: Result instead
        let desc = i.layout.desc().descriptor(set_id, binding_id).unwrap();     // TODO: Result instead

        assert!(desc.array_count == 1);     // not implemented
        i.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::UniformBuffer => unsafe {
                DescriptorWrite::uniform_buffer(binding_id as u32, &self)
            },
            DescriptorType::StorageBuffer => unsafe {
                DescriptorWrite::storage_buffer(binding_id as u32, &self)
            },
            _ => panic!()
        });

        SimpleDescriptorSetBuilder {
            layout: i.layout,
            set_id: i.set_id,
            writes: i.writes,
            resources: (i.resources, SimpleDescriptorSetBuf {
                buffer: self,
                write: !desc.readonly,
                stage: PipelineStages::none(),      // FIXME:
                access: AccessFlagBits::none(),     // FIXME:
            })
        }
    }
}

/// Internal trait related to the `SimpleDescriptorSet` system.
pub unsafe trait SimpleDescriptorSetResourcesCollection {
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>);
}

unsafe impl SimpleDescriptorSetResourcesCollection for () {
    #[inline]
    fn add_transition<'a>(&'a self, _: &mut CommandsListSink<'a>) {
    }
}

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetBuf<B> {
    buffer: B,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

unsafe impl<B> SimpleDescriptorSetResourcesCollection for SimpleDescriptorSetBuf<B>
    where B: Buffer
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        // TODO: wrong values
        let stages = PipelineStages {
            compute_shader: true,
            all_graphics: true,
            .. PipelineStages::none()
        };
        
        let access = AccessFlagBits {
            uniform_read: true,
            shader_read: true,
            shader_write: true,
            .. AccessFlagBits::none()
        };

        sink.add_buffer_transition(&self.buffer, 0, self.buffer.size(), self.write, stages, access);
    }
}

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetBufView<V> where V: BufferViewRef {
    view: V,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

unsafe impl<V> SimpleDescriptorSetResourcesCollection for SimpleDescriptorSetBufView<V>
    where V: BufferViewRef, V::Buffer: Buffer
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        // TODO: wrong values
        let stages = PipelineStages {
            compute_shader: true,
            all_graphics: true,
            .. PipelineStages::none()
        };
        
        let access = AccessFlagBits {
            uniform_read: true,
            shader_read: true,
            shader_write: true,
            .. AccessFlagBits::none()
        };

        sink.add_buffer_transition(self.view.view().buffer(), 0, self.view.view().buffer().size(),
                                   self.write, stages, access);
    }
}

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetImg<I> {
    image: I,
    write: bool,
    first_mipmap: u32,
    num_mipmaps: u32,
    first_layer: u32,
    num_layers: u32,
    layout: Layout,
    stage: PipelineStages,
    access: AccessFlagBits,
}

unsafe impl<I> SimpleDescriptorSetResourcesCollection for SimpleDescriptorSetImg<I>
    where I: ImageView
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        // TODO: wrong values
        let stages = PipelineStages {
            compute_shader: true,
            all_graphics: true,
            .. PipelineStages::none()
        };
        
        let access = AccessFlagBits {
            uniform_read: true,
            input_attachment_read: true,
            shader_read: true,
            shader_write: true,
            .. AccessFlagBits::none()
        };

        // FIXME: adjust layers & mipmaps with the view's parameters
        sink.add_image_transition(self.image.parent(), self.first_layer, self.num_layers,
                                  self.first_mipmap, self.num_mipmaps, self.write,
                                  self.layout, stages, access);
    }
}

unsafe impl<A, B> SimpleDescriptorSetResourcesCollection for (A, B)
    where A: SimpleDescriptorSetResourcesCollection,
          B: SimpleDescriptorSetResourcesCollection
{
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
        self.0.add_transition(sink);
        self.1.add_transition(sink);
    }
}
