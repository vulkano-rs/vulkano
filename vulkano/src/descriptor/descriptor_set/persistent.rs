// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use buffer::BufferAccess;
use buffer::BufferViewRef;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::DescriptorDescTy;
use descriptor::descriptor::DescriptorImageDesc;
use descriptor::descriptor::DescriptorImageDescArray;
use descriptor::descriptor::DescriptorImageDescDimensions;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::DescriptorPoolAlloc;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;
use descriptor::descriptor_set::DescriptorWrite;
use descriptor::descriptor_set::StdDescriptorPoolAlloc;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use device::Device;
use device::DeviceOwned;
use format::Format;
use image::ImageViewAccess;
use sampler::Sampler;
use OomError;
use VulkanObject;

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
/// express it explicitly. If you want to store your descriptor set in a struct or in a `Vec` for
/// example, you are encouraged to turn the `PersistentDescriptorSet` into a `Box<DescriptorSet>`
/// or a `Arc<DescriptorSet>`.
///
/// # Example
// TODO:
pub struct PersistentDescriptorSet<R, P = StdDescriptorPoolAlloc> {
    inner: P,
    resources: R,
    layout: Arc<UnsafeDescriptorSetLayout>,
}

impl PersistentDescriptorSet<()> {
    /// Starts the process of building a `PersistentDescriptorSet`. Returns a builder.
    ///
    /// # Panic
    ///
    /// - Panics if the set id is out of range.
    ///
    pub fn start(layout: Arc<UnsafeDescriptorSetLayout>) -> PersistentDescriptorSetBuilder<()> {
        let cap = layout.num_bindings();

        PersistentDescriptorSetBuilder {
            layout: layout,
            binding_id: 0,
            writes: Vec::with_capacity(cap),
            resources: (),
        }
    }
}

unsafe impl<R, P> DescriptorSet for PersistentDescriptorSet<R, P>
where
    P: DescriptorPoolAlloc,
    R: PersistentDescriptorSetResources,
{
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.inner()
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        self.resources.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        self.resources.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.resources.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        self.resources.image(index)
    }
}

unsafe impl<R, P> DescriptorSetDesc for PersistentDescriptorSet<R, P> {
    #[inline]
    fn num_bindings(&self) -> usize {
        self.layout.num_bindings()
    }

    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        self.layout.descriptor(binding)
    }
}

unsafe impl<R, P> DeviceOwned for PersistentDescriptorSet<R, P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

impl<R, P> PartialEq for PersistentDescriptorSet<R, P>
where
    P: DescriptorPoolAlloc,
    R: PersistentDescriptorSetResources,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl<R, P> Eq for PersistentDescriptorSet<R, P>
where
    P: DescriptorPoolAlloc,
    R: PersistentDescriptorSetResources,
{
}

impl<R, P> Hash for PersistentDescriptorSet<R, P>
where
    P: DescriptorPoolAlloc,
    R: PersistentDescriptorSetResources,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}

/// Prototype of a `PersistentDescriptorSet`.
///
/// The template parameter `R` is an unspecified type that represents the list of resources.
///
/// See the docs of `PersistentDescriptorSet` for an example.
pub struct PersistentDescriptorSetBuilder<R> {
    // The descriptor set layout.
    layout: Arc<UnsafeDescriptorSetLayout>,
    // Binding currently being filled.
    binding_id: usize,
    // The writes to perform on a descriptor set in order to put the resources in it.
    writes: Vec<DescriptorWrite>,
    // Holds the resources alive.
    resources: R,
}

// TODO: lots of checks are still missing, see the docs of
//       VkDescriptorImageInfo and VkWriteDescriptorSet

impl<R> PersistentDescriptorSetBuilder<R> {
    /// Builds a `PersistentDescriptorSet` from the builder.
    #[inline]
    pub fn build(
        self,
    ) -> Result<PersistentDescriptorSet<R, StdDescriptorPoolAlloc>, PersistentDescriptorSetBuildError>
    {
        let mut pool = Device::standard_descriptor_pool(self.layout.device());
        self.build_with_pool(&mut pool)
    }

    /// Builds a `PersistentDescriptorSet` from the builder.
    ///
    /// # Panic
    ///
    /// Panics if the pool doesn't have the same device as the descriptor set layout.
    ///
    pub fn build_with_pool<P>(
        self,
        pool: &mut P,
    ) -> Result<PersistentDescriptorSet<R, P::Alloc>, PersistentDescriptorSetBuildError>
    where
        P: ?Sized + DescriptorPool,
    {
        assert_eq!(
            self.layout.device().internal_object(),
            pool.device().internal_object()
        );

        let expected_desc = self.layout.num_bindings();

        if expected_desc > self.binding_id {
            return Err(PersistentDescriptorSetBuildError::MissingDescriptors {
                expected: expected_desc as u32,
                obtained: self.binding_id as u32,
            });
        }

        debug_assert_eq!(expected_desc, self.binding_id);

        let set = unsafe {
            let mut set = pool.alloc(&self.layout)?;
            set.inner_mut()
                .write(pool.device(), self.writes.into_iter());
            set
        };

        Ok(PersistentDescriptorSet {
            inner: set,
            resources: self.resources,
            layout: self.layout,
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
    pub fn enter_array(
        self,
    ) -> Result<PersistentDescriptorSetBuilderArray<R>, PersistentDescriptorSetError> {
        let desc = match self.layout.descriptor(self.binding_id) {
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
    pub fn add_empty(
        mut self,
    ) -> Result<PersistentDescriptorSetBuilder<R>, PersistentDescriptorSetError> {
        match self.layout.descriptor(self.binding_id) {
            None => (),
            Some(desc) => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                    expected: desc.ty.ty().unwrap(),
                })
            }
        }

        self.binding_id += 1;
        Ok(self)
    }

    /// Binds a buffer as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer doesn't have the same device as the descriptor set layout.
    ///
    #[inline]
    pub fn add_buffer<T>(
        self,
        buffer: T,
    ) -> Result<
        PersistentDescriptorSetBuilder<(R, PersistentDescriptorSetBuf<T>)>,
        PersistentDescriptorSetError,
    >
    where
        T: BufferAccess,
    {
        self.enter_array()?.add_buffer(buffer)?.leave_array()
    }

    /// Binds a buffer view as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer view doesn't have the same device as the descriptor set layout.
    ///
    pub fn add_buffer_view<T>(
        self,
        view: T,
    ) -> Result<
        PersistentDescriptorSetBuilder<(R, PersistentDescriptorSetBufView<T>)>,
        PersistentDescriptorSetError,
    >
    where
        T: BufferViewRef,
    {
        self.enter_array()?.add_buffer_view(view)?.leave_array()
    }

    /// Binds an image view as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image view doesn't have the same device as the descriptor set layout.
    ///
    #[inline]
    pub fn add_image<T>(
        self,
        image_view: T,
    ) -> Result<
        PersistentDescriptorSetBuilder<(R, PersistentDescriptorSetImg<T>)>,
        PersistentDescriptorSetError,
    >
    where
        T: ImageViewAccess,
    {
        self.enter_array()?.add_image(image_view)?.leave_array()
    }

    /// Binds an image view with a sampler as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image view or the sampler doesn't have the same device as the descriptor set
    /// layout.
    ///
    #[inline]
    pub fn add_sampled_image<T>(
        self,
        image_view: T,
        sampler: Arc<Sampler>,
    ) -> Result<
        PersistentDescriptorSetBuilder<(
            (R, PersistentDescriptorSetImg<T>),
            PersistentDescriptorSetSampler,
        )>,
        PersistentDescriptorSetError,
    >
    where
        T: ImageViewAccess,
    {
        self.enter_array()?
            .add_sampled_image(image_view, sampler)?
            .leave_array()
    }

    /// Binds a sampler as the next descriptor.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the sampler doesn't have the same device as the descriptor set layout.
    ///
    #[inline]
    pub fn add_sampler(
        self,
        sampler: Arc<Sampler>,
    ) -> Result<
        PersistentDescriptorSetBuilder<(R, PersistentDescriptorSetSampler)>,
        PersistentDescriptorSetError,
    > {
        self.enter_array()?.add_sampler(sampler)?.leave_array()
    }
}

/// Same as `PersistentDescriptorSetBuilder`, but we're in an array.
pub struct PersistentDescriptorSetBuilderArray<R> {
    // The original builder.
    builder: PersistentDescriptorSetBuilder<R>,
    // Current array elements.
    array_element: usize,
    // Description of the descriptor.
    desc: DescriptorDesc,
}

impl<R> PersistentDescriptorSetBuilderArray<R> {
    /// Leaves the array. Call this once you added all the elements of the array.
    pub fn leave_array(
        mut self,
    ) -> Result<PersistentDescriptorSetBuilder<R>, PersistentDescriptorSetError> {
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
    ///
    /// # Panic
    ///
    /// Panics if the buffer doesn't have the same device as the descriptor set layout.
    ///
    pub fn add_buffer<T>(
        mut self,
        buffer: T,
    ) -> Result<
        PersistentDescriptorSetBuilderArray<(R, PersistentDescriptorSetBuf<T>)>,
        PersistentDescriptorSetError,
    >
    where
        T: BufferAccess,
    {
        assert_eq!(
            self.builder.layout.device().internal_object(),
            buffer.inner().buffer.device().internal_object()
        );

        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        self.builder.writes.push(match self.desc.ty {
            DescriptorDescTy::Buffer(ref buffer_desc) => {
                // Note that the buffer content is not checked. This is technically not unsafe as
                // long as the data in the buffer has no invalid memory representation (ie. no
                // bool, no enum, no pointer, no str) and as long as the robust buffer access
                // feature is enabled.
                // TODO: this is not checked ^

                // TODO: eventually shouldn't be an assert ; for now robust_buffer_access is always
                //       enabled so this assert should never fail in practice, but we put it anyway
                //       in case we forget to adjust this code
                assert!(
                    self.builder
                        .layout
                        .device()
                        .enabled_features()
                        .robust_buffer_access
                );

                if buffer_desc.storage {
                    if !buffer.inner().buffer.usage_storage_buffer() {
                        return Err(PersistentDescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::storage_buffer(
                            self.builder.binding_id as u32,
                            self.array_element as u32,
                            &buffer,
                        )
                    }
                } else {
                    if !buffer.inner().buffer.usage_uniform_buffer() {
                        return Err(PersistentDescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::uniform_buffer(
                            self.builder.binding_id as u32,
                            self.array_element as u32,
                            &buffer,
                        )
                    }
                }
            }
            ref d => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                    expected: d.ty().unwrap(),
                });
            }
        });

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (
                    self.builder.resources,
                    PersistentDescriptorSetBuf {
                        buffer: buffer,
                        descriptor_num: self.builder.binding_id as u32,
                    },
                ),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds a buffer view as the next element in the array.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer view doesn't have the same device as the descriptor set layout.
    ///
    pub fn add_buffer_view<T>(
        mut self,
        view: T,
    ) -> Result<
        PersistentDescriptorSetBuilderArray<(R, PersistentDescriptorSetBufView<T>)>,
        PersistentDescriptorSetError,
    >
    where
        T: BufferViewRef,
    {
        assert_eq!(
            self.builder.layout.device().internal_object(),
            view.view().device().internal_object()
        );

        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        self.builder.writes.push(match self.desc.ty {
            DescriptorDescTy::TexelBuffer { storage, .. } => {
                if storage {
                    // TODO: storage_texel_buffer_atomic

                    if !view.view().storage_texel_buffer() {
                        return Err(PersistentDescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageTexelBuffer,
                        ));
                    }

                    DescriptorWrite::storage_texel_buffer(
                        self.builder.binding_id as u32,
                        self.array_element as u32,
                        view.view(),
                    )
                } else {
                    if !view.view().uniform_texel_buffer() {
                        return Err(PersistentDescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformTexelBuffer,
                        ));
                    }

                    DescriptorWrite::uniform_texel_buffer(
                        self.builder.binding_id as u32,
                        self.array_element as u32,
                        view.view(),
                    )
                }
            }
            ref d => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                    expected: d.ty().unwrap(),
                });
            }
        });

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (
                    self.builder.resources,
                    PersistentDescriptorSetBufView {
                        view: view,
                        descriptor_num: self.builder.binding_id as u32,
                    },
                ),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds an image view as the next element in the array.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image view doesn't have the same device as the descriptor set layout.
    ///
    pub fn add_image<T>(
        mut self,
        image_view: T,
    ) -> Result<
        PersistentDescriptorSetBuilderArray<(R, PersistentDescriptorSetImg<T>)>,
        PersistentDescriptorSetError,
    >
    where
        T: ImageViewAccess,
    {
        assert_eq!(
            self.builder.layout.device().internal_object(),
            image_view.parent().inner().image.device().internal_object()
        );

        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        let desc = match self.builder.layout.descriptor(self.builder.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        self.builder.writes.push(match desc.ty {
            DescriptorDescTy::Image(ref desc) => {
                image_match_desc(&image_view, &desc)?;

                if desc.sampled {
                    DescriptorWrite::sampled_image(
                        self.builder.binding_id as u32,
                        self.array_element as u32,
                        &image_view,
                    )
                } else {
                    DescriptorWrite::storage_image(
                        self.builder.binding_id as u32,
                        self.array_element as u32,
                        &image_view,
                    )
                }
            }
            DescriptorDescTy::InputAttachment {
                multisampled,
                array_layers,
            } => {
                if !image_view.parent().inner().image.usage_input_attachment() {
                    return Err(PersistentDescriptorSetError::MissingImageUsage(
                        MissingImageUsage::InputAttachment,
                    ));
                }

                if multisampled && image_view.samples() == 1 {
                    return Err(PersistentDescriptorSetError::ExpectedMultisampled);
                } else if !multisampled && image_view.samples() != 1 {
                    return Err(PersistentDescriptorSetError::UnexpectedMultisampled);
                }

                let image_layers = image_view.dimensions().array_layers();

                match array_layers {
                    DescriptorImageDescArray::NonArrayed => {
                        if image_layers != 1 {
                            return Err(PersistentDescriptorSetError::ArrayLayersMismatch {
                                expected: 1,
                                obtained: image_layers,
                            });
                        }
                    }
                    DescriptorImageDescArray::Arrayed {
                        max_layers: Some(max_layers),
                    } => {
                        if image_layers > max_layers {
                            // TODO: is this correct? "max" layers? or is it in fact min layers?
                            return Err(PersistentDescriptorSetError::ArrayLayersMismatch {
                                expected: max_layers,
                                obtained: image_layers,
                            });
                        }
                    }
                    DescriptorImageDescArray::Arrayed { max_layers: None } => {}
                };

                DescriptorWrite::input_attachment(
                    self.builder.binding_id as u32,
                    self.array_element as u32,
                    &image_view,
                )
            }
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                    expected: ty.ty().unwrap(),
                });
            }
        });

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (
                    self.builder.resources,
                    PersistentDescriptorSetImg {
                        image: image_view,
                        descriptor_num: self.builder.binding_id as u32,
                    },
                ),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds an image view with a sampler as the next element in the array.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image or the sampler doesn't have the same device as the descriptor set layout.
    ///
    pub fn add_sampled_image<T>(
        mut self,
        image_view: T,
        sampler: Arc<Sampler>,
    ) -> Result<
        PersistentDescriptorSetBuilderArray<(
            (R, PersistentDescriptorSetImg<T>),
            PersistentDescriptorSetSampler,
        )>,
        PersistentDescriptorSetError,
    >
    where
        T: ImageViewAccess,
    {
        assert_eq!(
            self.builder.layout.device().internal_object(),
            image_view.parent().inner().image.device().internal_object()
        );
        assert_eq!(
            self.builder.layout.device().internal_object(),
            sampler.device().internal_object()
        );

        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        let desc = match self.builder.layout.descriptor(self.builder.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        if !image_view.can_be_sampled(&sampler) {
            return Err(PersistentDescriptorSetError::IncompatibleImageViewSampler);
        }

        self.builder.writes.push(match desc.ty {
            DescriptorDescTy::CombinedImageSampler(ref desc) => {
                image_match_desc(&image_view, &desc)?;
                DescriptorWrite::combined_image_sampler(
                    self.builder.binding_id as u32,
                    self.array_element as u32,
                    &sampler,
                    &image_view,
                )
            }
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                    expected: ty.ty().unwrap(),
                });
            }
        });

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (
                    (
                        self.builder.resources,
                        PersistentDescriptorSetImg {
                            image: image_view,
                            descriptor_num: self.builder.binding_id as u32,
                        },
                    ),
                    PersistentDescriptorSetSampler { sampler: sampler },
                ),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }

    /// Binds a sampler as the next element in the array.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the sampler doesn't have the same device as the descriptor set layout.
    ///
    pub fn add_sampler(
        mut self,
        sampler: Arc<Sampler>,
    ) -> Result<
        PersistentDescriptorSetBuilderArray<(R, PersistentDescriptorSetSampler)>,
        PersistentDescriptorSetError,
    > {
        assert_eq!(
            self.builder.layout.device().internal_object(),
            sampler.device().internal_object()
        );

        if self.array_element as u32 >= self.desc.array_count {
            return Err(PersistentDescriptorSetError::ArrayOutOfBounds);
        }

        let desc = match self.builder.layout.descriptor(self.builder.binding_id) {
            Some(d) => d,
            None => return Err(PersistentDescriptorSetError::EmptyExpected),
        };

        self.builder.writes.push(match desc.ty {
            DescriptorDescTy::Sampler => DescriptorWrite::sampler(
                self.builder.binding_id as u32,
                self.array_element as u32,
                &sampler,
            ),
            ty => {
                return Err(PersistentDescriptorSetError::WrongDescriptorTy {
                    expected: ty.ty().unwrap(),
                });
            }
        });

        Ok(PersistentDescriptorSetBuilderArray {
            builder: PersistentDescriptorSetBuilder {
                layout: self.builder.layout,
                binding_id: self.builder.binding_id,
                writes: self.builder.writes,
                resources: (
                    self.builder.resources,
                    PersistentDescriptorSetSampler { sampler: sampler },
                ),
            },
            desc: self.desc,
            array_element: self.array_element + 1,
        })
    }
}

// Checks whether an image view matches the descriptor.
fn image_match_desc<I>(
    image_view: &I,
    desc: &DescriptorImageDesc,
) -> Result<(), PersistentDescriptorSetError>
where
    I: ?Sized + ImageViewAccess,
{
    if desc.sampled && !image_view.parent().inner().image.usage_sampled() {
        return Err(PersistentDescriptorSetError::MissingImageUsage(
            MissingImageUsage::Sampled,
        ));
    } else if !desc.sampled && !image_view.parent().inner().image.usage_storage() {
        return Err(PersistentDescriptorSetError::MissingImageUsage(
            MissingImageUsage::Storage,
        ));
    }

    let image_view_ty = DescriptorImageDescDimensions::from_dimensions(image_view.dimensions());
    if image_view_ty != desc.dimensions {
        return Err(PersistentDescriptorSetError::ImageViewTypeMismatch {
            expected: desc.dimensions,
            obtained: image_view_ty,
        });
    }

    if let Some(format) = desc.format {
        if image_view.format() != format {
            return Err(PersistentDescriptorSetError::ImageViewFormatMismatch {
                expected: format,
                obtained: image_view.format(),
            });
        }
    }

    if desc.multisampled && image_view.samples() == 1 {
        return Err(PersistentDescriptorSetError::ExpectedMultisampled);
    } else if !desc.multisampled && image_view.samples() != 1 {
        return Err(PersistentDescriptorSetError::UnexpectedMultisampled);
    }

    let image_layers = image_view.dimensions().array_layers();

    match desc.array_layers {
        DescriptorImageDescArray::NonArrayed => {
            // TODO: when a non-array is expected, can we pass an image view that is in fact an
            // array with one layer? need to check
            if image_layers != 1 {
                return Err(PersistentDescriptorSetError::ArrayLayersMismatch {
                    expected: 1,
                    obtained: image_layers,
                });
            }
        }
        DescriptorImageDescArray::Arrayed {
            max_layers: Some(max_layers),
        } => {
            if image_layers > max_layers {
                // TODO: is this correct? "max" layers? or is it in fact min layers?
                return Err(PersistentDescriptorSetError::ArrayLayersMismatch {
                    expected: max_layers,
                    obtained: image_layers,
                });
            }
        }
        DescriptorImageDescArray::Arrayed { max_layers: None } => {}
    };

    Ok(())
}

pub unsafe trait PersistentDescriptorSetResources {
    fn num_buffers(&self) -> usize;
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)>;
    fn num_images(&self) -> usize;
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)>;
}

unsafe impl PersistentDescriptorSetResources for () {
    #[inline]
    fn num_buffers(&self) -> usize {
        0
    }

    #[inline]
    fn buffer(&self, _: usize) -> Option<(&dyn BufferAccess, u32)> {
        None
    }

    #[inline]
    fn num_images(&self) -> usize {
        0
    }

    #[inline]
    fn image(&self, _: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        None
    }
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetBuf<B> {
    buffer: B,
    descriptor_num: u32,
}

unsafe impl<R, B> PersistentDescriptorSetResources for (R, PersistentDescriptorSetBuf<B>)
where
    R: PersistentDescriptorSetResources,
    B: BufferAccess,
{
    #[inline]
    fn num_buffers(&self) -> usize {
        self.0.num_buffers() + 1
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        if let Some(buf) = self.0.buffer(index) {
            Some(buf)
        } else if index == self.0.num_buffers() {
            Some((&self.1.buffer, self.1.descriptor_num))
        } else {
            None
        }
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.0.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        self.0.image(index)
    }
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetBufView<V>
where
    V: BufferViewRef,
{
    view: V,
    descriptor_num: u32,
}

unsafe impl<R, V> PersistentDescriptorSetResources for (R, PersistentDescriptorSetBufView<V>)
where
    R: PersistentDescriptorSetResources,
    V: BufferViewRef,
{
    #[inline]
    fn num_buffers(&self) -> usize {
        self.0.num_buffers() + 1
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        if let Some(buf) = self.0.buffer(index) {
            Some(buf)
        } else if index == self.0.num_buffers() {
            Some((self.1.view.view().buffer(), self.1.descriptor_num))
        } else {
            None
        }
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.0.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        self.0.image(index)
    }
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetImg<I> {
    image: I,
    descriptor_num: u32,
}

unsafe impl<R, I> PersistentDescriptorSetResources for (R, PersistentDescriptorSetImg<I>)
where
    R: PersistentDescriptorSetResources,
    I: ImageViewAccess,
{
    #[inline]
    fn num_buffers(&self) -> usize {
        self.0.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        self.0.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.0.num_images() + 1
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        if let Some(img) = self.0.image(index) {
            Some(img)
        } else if index == self.0.num_images() {
            Some((&self.1.image, self.1.descriptor_num))
        } else {
            None
        }
    }
}

/// Internal object related to the `PersistentDescriptorSet` system.
pub struct PersistentDescriptorSetSampler {
    sampler: Arc<Sampler>,
}

unsafe impl<R> PersistentDescriptorSetResources for (R, PersistentDescriptorSetSampler)
where
    R: PersistentDescriptorSetResources,
{
    #[inline]
    fn num_buffers(&self) -> usize {
        self.0.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        self.0.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.0.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        self.0.image(index)
    }
}

// Part of the PersistentDescriptorSetError for the case
// of missing usage on a buffer.
#[derive(Debug, Clone)]
pub enum MissingBufferUsage {
    StorageBuffer,
    UniformBuffer,
    StorageTexelBuffer,
    UniformTexelBuffer,
}

// Part of the PersistentDescriptorSetError for the case
// of missing usage on an image.
#[derive(Debug, Clone)]
pub enum MissingImageUsage {
    InputAttachment,
    Sampled,
    Storage,
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

    /// The buffer is missing the correct usage.
    MissingBufferUsage(MissingBufferUsage),

    /// The image is missing the correct usage.
    MissingImageUsage(MissingImageUsage),

    /// Expected a multisampled image, but got a single-sampled image.
    ExpectedMultisampled,

    /// Expected a single-sampled image, but got a multisampled image.
    UnexpectedMultisampled,

    /// The number of array layers of an image doesn't match what was expected.
    ArrayLayersMismatch {
        /// Number of expected array layers for the image.
        expected: u32,
        /// Number of array layers of the image that was added.
        obtained: u32,
    },

    /// The format of an image view doesn't match what was expected.
    ImageViewFormatMismatch {
        /// Expected format.
        expected: Format,
        /// Format of the image view that was passed.
        obtained: Format,
    },

    /// The type of an image view doesn't match what was expected.
    ImageViewTypeMismatch {
        /// Expected type.
        expected: DescriptorImageDescDimensions,
        /// Type of the image view that was passed.
        obtained: DescriptorImageDescDimensions,
    },
}

impl error::Error for PersistentDescriptorSetError {}

impl fmt::Display for PersistentDescriptorSetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                PersistentDescriptorSetError::WrongDescriptorTy { .. } => {
                    "expected one type of resource but got another"
                }
                PersistentDescriptorSetError::EmptyExpected => {
                    "expected an empty descriptor but got something"
                }
                PersistentDescriptorSetError::ArrayOutOfBounds => {
                    "tried to add too many elements to an array"
                }
                PersistentDescriptorSetError::MissingArrayElements { .. } => {
                    "didn't fill all the elements of an array before leaving"
                }
                PersistentDescriptorSetError::IncompatibleImageViewSampler => {
                    "the image view isn't compatible with the sampler"
                }
                PersistentDescriptorSetError::MissingBufferUsage { .. } => {
                    "the buffer is missing the correct usage"
                }
                PersistentDescriptorSetError::MissingImageUsage { .. } => {
                    "the image is missing the correct usage"
                }
                PersistentDescriptorSetError::ExpectedMultisampled => {
                    "expected a multisampled image, but got a single-sampled image"
                }
                PersistentDescriptorSetError::UnexpectedMultisampled => {
                    "expected a single-sampled image, but got a multisampled image"
                }
                PersistentDescriptorSetError::ArrayLayersMismatch { .. } => {
                    "the number of array layers of an image doesn't match what was expected"
                }
                PersistentDescriptorSetError::ImageViewFormatMismatch { .. } => {
                    "the format of an image view doesn't match what was expected"
                }
                PersistentDescriptorSetError::ImageViewTypeMismatch { .. } => {
                    "the type of an image view doesn't match what was expected"
                }
            }
        )
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

impl error::Error for PersistentDescriptorSetBuildError {}

impl From<OomError> for PersistentDescriptorSetBuildError {
    #[inline]
    fn from(err: OomError) -> PersistentDescriptorSetBuildError {
        PersistentDescriptorSetBuildError::OomError(err)
    }
}

impl fmt::Display for PersistentDescriptorSetBuildError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                PersistentDescriptorSetBuildError::MissingDescriptors { .. } => {
                    "didn't fill all the descriptors before building"
                }
                PersistentDescriptorSetBuildError::OomError(_) => "not enough memory available",
            }
        )
    }
}
