// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::sync::Arc;

use buffer::BufferAccess;
use buffer::BufferViewRef;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::DescriptorPoolAlloc;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor_set::DescriptorWrite;
use descriptor::descriptor_set::StdDescriptorPoolAlloc;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use image::ImageAccess;
use image::ImageLayout;
use image::ImageViewAccess;
use sampler::Sampler;
use sync::AccessFlagBits;
use sync::PipelineStages;
use OomError;

/// An immutable descriptor set that is expected to be long-lived.
///
/// Creating a persistent descriptor set allocates from a pool, and can't be modified once created.
/// You are therefore encouraged to create them at initialization and not the during
/// performance-critical paths.
///
/// > **Note**: You can control of the pool that is used to create the descriptor set, if you wish
/// > so. By creating a implementation of the `DescriptorPool` trait that doesn't perform any
/// > actual allocation, you can skip this allocation and make it acceptable to use a persistent
/// > descriptor set in performance-critical paths..
///
/// The template parameter of the `PersistentDescriptorSet` is complex, and you shouldn't try to
/// express it explicitely. If you want to store your descriptor set in a struct or in a `Vec` for
/// example, you are encouraged to turn the `PersistentDescriptorSet` into a `Box<DescriptorSet>`
/// or a `Arc<DescriptorSet>`.
///
/// # Example
// TODO:
pub struct PersistentDescriptorSet<R, P = StdDescriptorPoolAlloc> {
    inner: P,
    resources: R,
    layout: Arc<UnsafeDescriptorSetLayout>
}

impl PersistentDescriptorSet<()> {
    /// Starts the process of building a `PersistentDescriptorSet`. Returns a builder.
    ///
    /// # Panic
    ///
    /// - Panics if the set id is out of range.
    ///
    pub fn start<Pl>(layout: Pl, set_id: usize) -> PersistentDescriptorSetBuilder<Pl, ()>
        where Pl: PipelineLayoutAbstract
    {
        assert!(layout.num_sets() > set_id);

        let cap = layout.num_bindings_in_set(set_id).unwrap_or(0);

        PersistentDescriptorSetBuilder {
            layout: layout,
            set_id: set_id,
            binding_id: 0,
            writes: Vec::with_capacity(cap),
            resources: (),
        }
    }
}

unsafe impl<R, P> DescriptorSet for PersistentDescriptorSet<R, P> where P: DescriptorPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.inner()
    }

    #[inline]
    fn buffers_list<'a>(&'a self) -> Box<Iterator<Item = &'a BufferAccess> + 'a> {
        unimplemented!()
    }

    #[inline]
    fn images_list<'a>(&'a self) -> Box<Iterator<Item = &'a ImageAccess> + 'a> {
        unimplemented!()
    }
}

// TODO: is DescriptorSetDesc really necessary?
unsafe impl<R, P> DescriptorSetDesc for PersistentDescriptorSet<R, P> {
    #[inline]
    fn num_bindings(&self) -> usize {
        unimplemented!()        // FIXME:
    }

    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        unimplemented!()        // FIXME:
    }
}

/// Prototype of a `PersistentDescriptorSet`.
///
/// The template parameter `L` is the pipeline layout to use, and the template parameter `R` is
/// an unspecified type that represents the list of resources.
///
/// See the docs of `PersistentDescriptorSet` for an example.
pub struct PersistentDescriptorSetBuilder<L, R> {
    // The pipeline layout.
    layout: L,
    // Id of the set within the pipeline layout.
    set_id: usize,
    // Binding currently being filled.
    binding_id: usize,
    // The writes to perform on a descriptor set in order to put the resources in it.
    writes: Vec<DescriptorWrite>,
    // Holds the resources alive.
    resources: R,
}

impl<L, R> PersistentDescriptorSetBuilder<L, R>
    where L: PipelineLayoutAbstract
{
    /// Builds a `PersistentDescriptorSet` from the builder.
    #[inline]
    pub fn build(self) -> Result<PersistentDescriptorSet<R, StdDescriptorPoolAlloc>, PersistentDescriptorSetBuildError> {
        let pool = Device::standard_descriptor_pool(self.layout.device());
        self.build_with_pool(pool)
    }

    /// Builds a `PersistentDescriptorSet` from the builder.
    pub fn build_with_pool<P>(self, pool: P)
                              -> Result<PersistentDescriptorSet<R, P::Alloc>, PersistentDescriptorSetBuildError>
        where P: DescriptorPool
    {
        let expected_desc = self.layout.num_bindings_in_set(self.set_id).unwrap();

        if expected_desc > self.binding_id {
            return Err(PersistentDescriptorSetBuildError::MissingDescriptors {
                expected: expected_desc as u32,
                obtained: self.binding_id as u32,
            });
        }

        debug_assert_eq!(expected_desc, self.binding_id);

        let set_layout = self.layout.descriptor_set_layout(self.set_id)
            .expect("Unable to get the descriptor set layout")
            .clone();

        let set = unsafe {
            let mut set = pool.alloc(&set_layout)?;
            set.inner_mut().write(pool.device(), self.writes.into_iter());
            set
        };

        Ok(PersistentDescriptorSet {
            inner: set,
            resources: self.resources,
            layout: set_layout,
        })
    }

    /// Call this function if the next element of the set is an array in order to set the value of
    /// each element.
    ///
    /// Returns an error if the descriptor is empty.
    ///
    /// This function can be called even if the descriptor isn't an array, and it is valid to enter
    /// the "array", add one element, then leave.
    #[inline]
    pub fn enter_array(self) -> Result<PersistentDescriptorSetBuilderArray<L, R>, PersistentDescriptorSetError> {
        let desc = match self.layout.descriptor(self.set_id, self.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        Ok(PersistentDescriptorSetBuilderArray {
            builder: self,
            desc,
            array_element: 0,
        })
    }

    /// Skips the current descriptor if it is empty.
    #[inline]
    pub fn add_empty(mut self) -> Result<PersistentDescriptorSetBuilder<L, R>, PersistentDescriptorSetError> {
        match self.layout.descriptor(self.set_id, self.binding_id) {
            None => (),
            Some(desc) => return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                expected: desc.ty.ty().unwrap()
            }),
        }

        self.binding_id += 1;
        Ok(self)
    }

    /// Binds a buffer as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    #[inline]
    pub fn add_buffer<T>(self, buffer: T)
        -> Result<PersistentDescriptorSetBuilder<L, (R, PersistentDescriptorSetBuf<T>)>, PersistentDescriptorSetError>
        where T: BufferAccess
    {
        self.enter_array()?
            .add_buffer(buffer)?
            .leave_array()
    }

    /// Binds an image view as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    #[inline]
    pub fn add_image<T>(self, image_view: T)
        -> Result<PersistentDescriptorSetBuilder<L, (R, PersistentDescriptorSetImg<T>)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        self.enter_array()?
            .add_image(image_view)?
            .leave_array()
    }

    /// Binds an image view with a sampler as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    #[inline]
    pub fn add_sampled_image<T>(self, image_view: T, sampler: Arc<Sampler>)
        -> Result<PersistentDescriptorSetBuilder<L, ((R, PersistentDescriptorSetImg<T>), PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        self.enter_array()?
            .add_sampled_image(image_view, sampler)?
            .leave_array()
    }

    /// Binds a sampler as the next descriptor.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    #[inline]
    pub fn add_sampler(self, sampler: Arc<Sampler>)
        -> Result<PersistentDescriptorSetBuilder<L, (R, PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
    {
        self.enter_array()?
            .add_sampler(sampler)?
            .leave_array()
    }
}

/// Same as `PersistentDescriptorSetBuilder`, but we're in an array.
pub struct PersistentDescriptorSetBuilderArray<L, R> {
    // The original builder.
    builder: PersistentDescriptorSetBuilder<L, R>,
    // Current array elements.
    array_element: usize,
    // Description of the descriptor.
    desc: DescriptorDesc,
}

impl<L, R> PersistentDescriptorSetBuilderArray<L, R> where L: PipelineLayoutAbstract {
    /// Leaves the array. Call this once you added all the elements of the array.
    pub fn leave_array(mut self) -> Result<PersistentDescriptorSetBuilder<L, R>, PersistentDescriptorSetError> {
        if self.desc.array_count > self.array_element as u32 {
            return Err(PersistentDescriptorSetError::MissingArrayElements {
                expected: self.desc.array_count,
                obtained: self.array_element as u32,
            });
        }

        debug_assert_eq!(self.desc.array_count, self.array_element as u32);

        self.builder.binding_id += 1;
        Ok(self.builder)
    }

    /// Binds a buffer as the next element in the array.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    pub fn add_buffer<T>(mut self, buffer: T)
        -> Result<PersistentDescriptorSetBuilderArray<L, (R, PersistentDescriptorSetBuf<T>)>, PersistentDescriptorSetError>
        where T: BufferAccess
    {
        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        self.builder.writes.push(match self.desc.ty.ty().unwrap() {
            DescriptorType::UniformBuffer => unsafe {
                DescriptorWrite::uniform_buffer(self.builder.binding_id as u32, self.array_element as u32, &buffer)
            },
            DescriptorType::StorageBuffer => unsafe {
                DescriptorWrite::storage_buffer(self.builder.binding_id as u32, self.array_element as u32, &buffer)
            },
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy { expected: ty });
            },
        });

        let readonly = self.desc.readonly;

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                set_id: self.builder.set_id,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (self.builder.resources, PersistentDescriptorSetBuf {
                    buffer: buffer,
                    write: !readonly,
                    stage: PipelineStages::none(),      // FIXME:
                    access: AccessFlagBits::none(),     // FIXME:
                })
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds an image view as the next element in the array.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    pub fn add_image<T>(mut self, image_view: T)
        -> Result<PersistentDescriptorSetBuilderArray<L, (R, PersistentDescriptorSetImg<T>)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        let desc = match self.builder.layout.descriptor(self.builder.set_id, self.builder.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        self.builder.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::SampledImage => {
                DescriptorWrite::sampled_image(self.builder.binding_id as u32, self.array_element as u32, &image_view)
            },
            DescriptorType::StorageImage => {
                DescriptorWrite::storage_image(self.builder.binding_id as u32, self.array_element as u32, &image_view)
            },
            DescriptorType::InputAttachment => {
                DescriptorWrite::input_attachment(self.builder.binding_id as u32, self.array_element as u32, &image_view)
            },
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy { expected: ty });
            },
        });

        let readonly = self.desc.readonly;

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                set_id: self.builder.set_id,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (self.builder.resources, PersistentDescriptorSetImg {
                    image: image_view,
                    write: !readonly,
                    first_mipmap: 0,            // FIXME:
                    num_mipmaps: 1,         // FIXME:
                    first_layer: 0,         // FIXME:
                    num_layers: 1,          // FIXME:
                    layout: ImageLayout::General,            // FIXME:
                    stage: PipelineStages::none(),          // FIXME:
                    access: AccessFlagBits::none(),         // FIXME:
                })
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds an image view with a sampler as the next element in the array.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    pub fn add_sampled_image<T>(mut self, image_view: T, sampler: Arc<Sampler>)
        -> Result<PersistentDescriptorSetBuilderArray<L, ((R, PersistentDescriptorSetImg<T>), PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        let desc = match self.builder.layout.descriptor(self.builder.set_id, self.builder.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        if !image_view.can_be_sampled(&sampler) {
            return Err(PersistentDescriptorSetError::IncompatibleImageViewSampler);
        }

        self.builder.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::CombinedImageSampler => {
                DescriptorWrite::combined_image_sampler(self.builder.binding_id as u32, self.array_element as u32, &sampler, &image_view)
            },
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy { expected: ty });
            },
        });

        let readonly = self.desc.readonly;

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                set_id: self.builder.set_id,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: ((self.builder.resources, PersistentDescriptorSetImg {
                    image: image_view,
                    write: !readonly,
                    first_mipmap: 0,            // FIXME:
                    num_mipmaps: 1,         // FIXME:
                    first_layer: 0,         // FIXME:
                    num_layers: 1,          // FIXME:
                    layout: ImageLayout::General,            // FIXME:
                    stage: PipelineStages::none(),          // FIXME:
                    access: AccessFlagBits::none(),         // FIXME:
                }), PersistentDescriptorSetSampler {
                    sampler: sampler,
                }),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds a sampler as the next element in the array.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    pub fn add_sampler(mut self, sampler: Arc<Sampler>)
        -> Result<PersistentDescriptorSetBuilderArray<L, (R, PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
    {
        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        let desc = match self.builder.layout.descriptor(self.builder.set_id, self.builder.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        self.builder.writes.push(match desc.ty.ty().unwrap() {
            DescriptorType::Sampler => {
                DescriptorWrite::sampler(self.builder.binding_id as u32, self.array_element as u32, &sampler)
            },
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy { expected: ty });
            },
        });

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                set_id: self.builder.set_id,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (self.builder.resources, PersistentDescriptorSetSampler {
                    sampler: sampler,
                }),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetBuf<B> {
    buffer: B,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetBufView<V>
    where V: BufferViewRef
{
    view: V,
    write: bool,
    stage: PipelineStages,
    access: AccessFlagBits,
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetImg<I> {
    image: I,
    write: bool,
    first_mipmap: u32,
    num_mipmaps: u32,
    first_layer: u32,
    num_layers: u32,
    layout: ImageLayout,
    stage: PipelineStages,
    access: AccessFlagBits,
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetSampler {
    sampler: Arc<Sampler>,
}

/// Error related to the persistent descriptor set.
#[derive(Debug, Clone)]
pub enum PersistentDescriptorSetError {
    /// Expected one type of resource but got another.
    WrongDescriptorTy {
        /// The expected descriptor type.
        expected: DescriptorType,
    },

    /// Expected nothing.
    EmptyExpected,

    /// Tried to add too many elements to an array.
    ArrayOutOfBounds,

    /// Didn't fill all the elements of an array before leaving.
    MissingArrayElements {
        /// Number of expected elements.
        expected: u32,
        /// Number of elements that were added.
        obtained: u32,
    },

    /// The image view isn't compatible with the sampler.
    IncompatibleImageViewSampler,
}

impl error::Error for PersistentDescriptorSetError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            PersistentDescriptorSetError::WrongDescriptorTy { .. } => {
                "expected one type of resource but got another"
            },
            PersistentDescriptorSetError::EmptyExpected => {
                "expected an empty descriptor but got something"
            },
            PersistentDescriptorSetError::ArrayOutOfBounds => {
                "tried to add too many elements to an array"
            },
            PersistentDescriptorSetError::MissingArrayElements { .. } => {
                "didn't fill all the elements of an array before leaving"
            },
            PersistentDescriptorSetError::IncompatibleImageViewSampler => {
                "the image view isn't compatible with the sampler"
            },
        }
    }
}

impl fmt::Display for PersistentDescriptorSetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

/// Error when building a persistent descriptor set.
#[derive(Debug, Clone)]
pub enum PersistentDescriptorSetBuildError {
    /// Out of memory.
    OomError(OomError),

    /// Didn't fill all the descriptors before building.
    MissingDescriptors {
        /// Number of expected descriptors.
        expected: u32,
        /// Number of descriptors that were added.
        obtained: u32,
    },
}

impl error::Error for PersistentDescriptorSetBuildError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            PersistentDescriptorSetBuildError::MissingDescriptors { .. } => {
                "didn't fill all the descriptors before building"
            },
            PersistentDescriptorSetBuildError::OomError(_) => {
                "not enough memory available"
            },
        }
    }
}

impl From<OomError> for PersistentDescriptorSetBuildError {
    #[inline]
    fn from(err: OomError) -> PersistentDescriptorSetBuildError {
        PersistentDescriptorSetBuildError::OomError(err)
    }
}

impl fmt::Display for PersistentDescriptorSetBuildError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
