// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use buffer::BufferAccess;
use buffer::BufferViewRef;
use buffer::Buffer;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::DescriptorPoolAlloc;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor_set::DescriptorWrite;
use descriptor::descriptor_set::StdDescriptorPool;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use image::Image;
use image::IntoImageView;
use image::sys::Layout;
use sampler::Sampler;
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

    #[inline]
    fn buffers_list<'a>(&'a self) -> Box<Iterator<Item = &'a BufferAccess> + 'a> {
        unimplemented!()
    }

    #[inline]
    fn images_list<'a>(&'a self) -> Box<Iterator<Item = &'a Image> + 'a> {
        unimplemented!()
    }
}

unsafe impl<R, P> DescriptorSetDesc for SimpleDescriptorSet<R, P> where P: DescriptorPool {
    #[inline]
    fn num_bindings(&self) -> usize {
        unimplemented!()        // FIXME:
    }

    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        unimplemented!()        // FIXME:
    }
}

/// Builds a descriptor set in the form of a `SimpleDescriptorSet` object.
// TODO: more doc
#[macro_export]
macro_rules! simple_descriptor_set {
    ($layout:expr, $set_num:expr, {$($name:ident: $val:expr),*$(,)*}) => ({
        #[allow(unused_imports)]
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetBuilder;
        #[allow(unused_imports)]
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetBufferExt;
        #[allow(unused_imports)]
        use $crate::descriptor::descriptor_set::SimpleDescriptorSetImageExt;

        // We build an empty `SimpleDescriptorSetBuilder` struct, then adds each element one by
        // one. When done, we call `build()` on the builder.

        let builder = SimpleDescriptorSetBuilder::new($layout, $set_num);

        $(
            // Here `$val` can be either a buffer or an image. However we can't create an extension
            // trait for both buffers and image, because `impl<T: Image> ExtTrait for T {}` would
            // conflict with `impl<T: BufferAccess> ExtTrait for T {}`.
            //
            // Therefore we use a trick: we create two traits, one for buffers
            // (`SimpleDescriptorSetBufferExt`) and one for images (`SimpleDescriptorSetImageExt`),
            // that both have a method named `add_me`. We import these two traits in scope and
            // call `add_me` on the value, letting Rust dispatch to the right trait. A compilation
            // error will happen if `$val` is both a buffer and an image.
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
/// The template parameter `L` is the pipeline layout to use, and the template parameter `R` is
/// a complex unspecified type that represents the list of resources.
///
/// # Example
// TODO: example here
pub struct SimpleDescriptorSetBuilder<L, R> {
    // The pipeline layout.
    layout: L,
    // Id of the set within the pipeline layout.
    set_id: usize,
    // The writes to perform on a descriptor set in order to put the resources in it.
    writes: Vec<DescriptorWrite>,
    // Holds the resources alive.
    resources: R,
}

impl<L> SimpleDescriptorSetBuilder<L, ()> where L: PipelineLayoutAbstract {
    /// Builds a new prototype for a `SimpleDescriptorSet`. Requires a reference to a pipeline
    /// layout, and the id of the set within the layout.
    ///
    /// # Panic
    ///
    /// - Panics if the set id is out of range.
    ///
    pub fn new(layout: L, set_id: usize) -> SimpleDescriptorSetBuilder<L, ()> {
        assert!(layout.desc().num_sets() > set_id);

        let cap = layout.desc().num_bindings_in_set(set_id).unwrap_or(0);

        SimpleDescriptorSetBuilder {
            layout: layout,
            set_id: set_id,
            writes: Vec::with_capacity(cap),
            resources: (),
        }
    }
}

impl<L, R> SimpleDescriptorSetBuilder<L, R> where L: PipelineLayoutAbstract {
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

/// Trait implemented on buffers so that they can be appended to a simple descriptor set builder.
pub unsafe trait SimpleDescriptorSetBufferExt<L, R> {
    /// The new type of the template parameter `R` of the builder.
    type Out;

    /// Appends the buffer to the `SimpleDescriptorSetBuilder`.
    // TODO: return Result
    fn add_me(self, i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>;
}

unsafe impl<L, R, T> SimpleDescriptorSetBufferExt<L, R> for T
    where T: Buffer, L: PipelineLayoutAbstract
{
    type Out = (R, SimpleDescriptorSetBuf<T::Target>);

    fn add_me(self, mut i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>
    {
        let buffer = self.into_buffer();

        let (set_id, binding_id) = i.layout.desc().descriptor_by_name(name).unwrap();    // TODO: Result instead
        assert_eq!(set_id, i.set_id);       // TODO: Result instead
        let desc = i.layout.desc().descriptor(set_id, binding_id).unwrap();     // TODO: Result instead

        assert!(desc.array_count == 1);     // not implemented
        i.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::UniformBuffer => unsafe {
                DescriptorWrite::uniform_buffer(binding_id as u32, 0, &buffer)
            },
            DescriptorType::StorageBuffer => unsafe {
                DescriptorWrite::storage_buffer(binding_id as u32, 0, &buffer)
            },
            _ => panic!()
        });

        SimpleDescriptorSetBuilder {
            layout: i.layout,
            set_id: i.set_id,
            writes: i.writes,
            resources: (i.resources, SimpleDescriptorSetBuf {
                buffer: buffer,
                write: !desc.readonly,
                stage: PipelineStages::none(),      // FIXME:
                access: AccessFlagBits::none(),     // FIXME:
            })
        }
    }
}

/// Trait implemented on images so that they can be appended to a simple descriptor set builder.
pub unsafe trait SimpleDescriptorSetImageExt<L, R> {
    /// The new type of the template parameter `R` of the builder.
    type Out;

    /// Appends the image to the `SimpleDescriptorSetBuilder`.
    // TODO: return Result
    fn add_me(self, i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>;
}

unsafe impl<L, R, T> SimpleDescriptorSetImageExt<L, R> for T
    where T: IntoImageView, L: PipelineLayoutAbstract
{
    type Out = (R, SimpleDescriptorSetImg<T::Target>);

    fn add_me(self, mut i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>
    {
        let image_view = self.into_image_view();

        let (set_id, binding_id) = i.layout.desc().descriptor_by_name(name).unwrap();    // TODO: Result instead
        assert_eq!(set_id, i.set_id);       // TODO: Result instead
        let desc = i.layout.desc().descriptor(set_id, binding_id).unwrap();     // TODO: Result instead

        assert!(desc.array_count == 1);     // not implemented
        i.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::SampledImage => {
                DescriptorWrite::sampled_image(binding_id as u32, 0, &image_view)
            },
            DescriptorType::StorageImage => {
                DescriptorWrite::storage_image(binding_id as u32, 0, &image_view)
            },
            DescriptorType::InputAttachment => {
                DescriptorWrite::input_attachment(binding_id as u32, 0, &image_view)
            },
            _ => panic!()
        });

        SimpleDescriptorSetBuilder {
            layout: i.layout,
            set_id: i.set_id,
            writes: i.writes,
            resources: (i.resources, SimpleDescriptorSetImg {
                image: image_view,
                sampler: None,
                write: !desc.readonly,
                first_mipmap: 0,            // FIXME:
                num_mipmaps: 1,         // FIXME:
                first_layer: 0,         // FIXME:
                num_layers: 1,          // FIXME:
                layout: Layout::General,            // FIXME:
                stage: PipelineStages::none(),          // FIXME:
                access: AccessFlagBits::none(),         // FIXME:
            })
        }
    }
}

unsafe impl<L, R, T> SimpleDescriptorSetImageExt<L, R> for (T, Arc<Sampler>)
    where T: IntoImageView, L: PipelineLayoutAbstract
{
    type Out = (R, SimpleDescriptorSetImg<T::Target>);

    fn add_me(self, mut i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>
    {
        let image_view = self.0.into_image_view();

        let (set_id, binding_id) = i.layout.desc().descriptor_by_name(name).unwrap();    // TODO: Result instead
        assert_eq!(set_id, i.set_id);       // TODO: Result instead
        let desc = i.layout.desc().descriptor(set_id, binding_id).unwrap();     // TODO: Result instead

        assert!(desc.array_count == 1);     // not implemented
        i.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::CombinedImageSampler => {
                DescriptorWrite::combined_image_sampler(binding_id as u32, 0, &self.1, &image_view)
            },
            _ => panic!()
        });

        SimpleDescriptorSetBuilder {
            layout: i.layout,
            set_id: i.set_id,
            writes: i.writes,
            resources: (i.resources, SimpleDescriptorSetImg {
                image: image_view,
                sampler: Some(self.1),
                write: !desc.readonly,
                first_mipmap: 0,            // FIXME:
                num_mipmaps: 1,         // FIXME:
                first_layer: 0,         // FIXME:
                num_layers: 1,          // FIXME:
                layout: Layout::General,            // FIXME:
                stage: PipelineStages::none(),          // FIXME:
                access: AccessFlagBits::none(),         // FIXME:
            })
        }
    }
}

// TODO: DRY
unsafe impl<L, R, T> SimpleDescriptorSetImageExt<L, R> for Vec<(T, Arc<Sampler>)>
    where T: IntoImageView, L: PipelineLayoutAbstract
{
    type Out = (R, Vec<SimpleDescriptorSetImg<T::Target>>);

    fn add_me(self, mut i: SimpleDescriptorSetBuilder<L, R>, name: &str)
              -> SimpleDescriptorSetBuilder<L, Self::Out>
    {
        let (set_id, binding_id) = i.layout.desc().descriptor_by_name(name).unwrap();    // TODO: Result instead
        assert_eq!(set_id, i.set_id);       // TODO: Result instead
        let desc = i.layout.desc().descriptor(set_id, binding_id).unwrap();     // TODO: Result instead

        assert_eq!(desc.array_count as usize, self.len());     // not implemented

        let mut imgs = Vec::new();
        for (num, (img, sampler)) in self.into_iter().enumerate() {
            let image_view = img.into_image_view();

            i.writes.push(match desc.ty.ty().unwrap() {
                DescriptorType::CombinedImageSampler => {
                    DescriptorWrite::combined_image_sampler(binding_id as u32, num as u32,
                                                            &sampler, &image_view)
                },
                _ => panic!()
            });

            imgs.push(SimpleDescriptorSetImg {
                image: image_view,
                sampler: Some(sampler),
                write: !desc.readonly,
                first_mipmap: 0,            // FIXME:
                num_mipmaps: 1,         // FIXME:
                first_layer: 0,         // FIXME:
                num_layers: 1,          // FIXME:
                layout: Layout::General,            // FIXME:
                stage: PipelineStages::none(),          // FIXME:
                access: AccessFlagBits::none(),         // FIXME:
            });
        }

        SimpleDescriptorSetBuilder {
            layout: i.layout,
            set_id: i.set_id,
            writes: i.writes,
            resources: (i.resources, imgs),
        }
    }
}

/*
/// Internal trait related to the `SimpleDescriptorSet` system.
pub unsafe trait SimpleDescriptorSetResourcesCollection {
    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>);
}

unsafe impl SimpleDescriptorSetResourcesCollection for () {
    #[inline]
    fn add_transition<'a>(&'a self, _: &mut CommandsListSink<'a>) {
    }
}*/

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetBuf<B> {
    buffer: B,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

/*unsafe impl<B> SimpleDescriptorSetResourcesCollection for SimpleDescriptorSetBuf<B>
    where B: BufferAccess
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
}*/

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetBufView<V> where V: BufferViewRef {
    view: V,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

/*unsafe impl<V> SimpleDescriptorSetResourcesCollection for SimpleDescriptorSetBufView<V>
    where V: BufferViewRef, V::BufferAccess: BufferAccess
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
}*/

/// Internal object related to the `SimpleDescriptorSet` system.
pub struct SimpleDescriptorSetImg<I> {
    image: I,
    sampler: Option<Arc<Sampler>>,
    write: bool,
    first_mipmap: u32,
    num_mipmaps: u32,
    first_layer: u32,
    num_layers: u32,
    layout: Layout,
    stage: PipelineStages,
    access: AccessFlagBits,
}

/*unsafe impl<I> SimpleDescriptorSetResourcesCollection for SimpleDescriptorSetImg<I>
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
}*/
