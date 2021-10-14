// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::resources::DescriptorSetResources;
use super::DescriptorSetError;
use super::MissingBufferUsage;
use super::MissingImageUsage;
use crate::buffer::{BufferAccess, BufferView};
use crate::descriptor_set::layout::{DescriptorDesc, DescriptorDescImage, DescriptorDescTy};
use crate::descriptor_set::sys::DescriptorWrite;
use crate::descriptor_set::DescriptorSetLayout;
use crate::device::{Device, DeviceOwned};
use crate::image::view::ImageViewType;
use crate::image::{ImageViewAbstract, SampleCount};
use crate::sampler::Sampler;
use crate::VulkanObject;
use std::sync::Arc;

struct BuilderDescriptor {
    desc: Option<DescriptorDesc>,
    array_element: u32,
}

/// A builder for constructing a new descriptor set.
pub struct DescriptorSetBuilder {
    layout: Arc<DescriptorSetLayout>,
    in_array: bool,
    descriptors: Vec<BuilderDescriptor>,
    cur_binding: u32,
    resources: DescriptorSetResources,
    desc_writes: Vec<DescriptorWrite>,
    poisoned: bool,
}

impl DescriptorSetBuilder {
    /// Starts the process of building a descriptor set. Returns a builder.
    ///
    /// # Panic
    ///
    /// - Panics if the set id is out of range.
    pub fn start(layout: Arc<DescriptorSetLayout>) -> Self {
        let mut descriptors = Vec::with_capacity(layout.num_bindings() as usize);
        let mut desc_writes_capacity = 0;
        let mut t_num_bufs = 0;
        let mut t_num_imgs = 0;
        let mut t_num_samplers = 0;

        for binding_i in 0..layout.num_bindings() {
            if let Some(desc) = layout.descriptor(binding_i) {
                let descriptor_count = desc.descriptor_count as usize;
                let (num_bufs, num_imgs, num_samplers) = match desc.ty {
                    DescriptorDescTy::Sampler { .. } => (0, 0, 1),
                    DescriptorDescTy::CombinedImageSampler { .. } => (0, 1, 1),
                    DescriptorDescTy::SampledImage { .. } => (0, 1, 0),
                    DescriptorDescTy::InputAttachment { .. } => (0, 1, 0),
                    DescriptorDescTy::StorageImage { .. } => (0, 1, 0),
                    DescriptorDescTy::UniformTexelBuffer { .. } => (1, 0, 0),
                    DescriptorDescTy::StorageTexelBuffer { .. } => (1, 0, 0),
                    DescriptorDescTy::UniformBuffer => (1, 0, 0),
                    DescriptorDescTy::StorageBuffer => (1, 0, 0),
                    DescriptorDescTy::UniformBufferDynamic => (1, 0, 0),
                    DescriptorDescTy::StorageBufferDynamic => (1, 0, 0),
                };

                t_num_bufs += num_bufs * descriptor_count;
                t_num_imgs += num_imgs * descriptor_count;
                t_num_samplers += num_samplers * descriptor_count;
                desc_writes_capacity += descriptor_count;

                descriptors.push(BuilderDescriptor {
                    desc: Some(desc),
                    array_element: 0,
                });
            } else {
                descriptors.push(BuilderDescriptor {
                    desc: None,
                    array_element: 0,
                });
            }
        }

        Self {
            layout,
            in_array: false,
            cur_binding: 0,
            descriptors,
            resources: DescriptorSetResources::new(t_num_bufs, t_num_imgs, t_num_samplers),
            desc_writes: Vec::with_capacity(desc_writes_capacity),
            poisoned: false,
        }
    }

    /// Finalizes the building process and returns the generated output.
    pub fn build(self) -> Result<DescriptorSetBuilderOutput, DescriptorSetError> {
        if self.poisoned {
            return Err(DescriptorSetError::BuilderPoisoned);
        }

        if self.cur_binding != self.descriptors.len() as u32 {
            Err(DescriptorSetError::DescriptorsMissing {
                expected: self.descriptors.len() as u32,
                obtained: self.cur_binding,
            })
        } else {
            Ok(DescriptorSetBuilderOutput {
                layout: self.layout,
                writes: self.desc_writes,
                resources: self.resources,
            })
        }
    }

    fn poison_on_err(
        &mut self,
        func: impl FnOnce(&mut Self) -> Result<(), DescriptorSetError>,
    ) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            return Err(DescriptorSetError::BuilderPoisoned);
        }

        match func(self) {
            Ok(()) => Ok(self),
            Err(err) => {
                self.poisoned = true;
                Err(err)
            }
        }
    }

    /// Call this function if the next element of the set is an array in order to set the value of
    /// each element.
    ///
    /// This function can be called even if the descriptor isn't an array, and it is valid to enter
    /// the "array", add one element, then leave.
    pub fn enter_array(&mut self) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            if builder.in_array {
                Err(DescriptorSetError::AlreadyInArray)
            } else if builder.cur_binding >= builder.descriptors.len() as u32 {
                Err(DescriptorSetError::TooManyDescriptors)
            } else if builder.descriptors[builder.cur_binding as usize]
                .desc
                .is_none()
            {
                Err(DescriptorSetError::DescriptorIsEmpty)
            } else {
                builder.in_array = true;
                Ok(())
            }
        })
    }

    /// Leaves the array. Call this once you added all the elements of the array.
    pub fn leave_array(&mut self) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            if !builder.in_array {
                Err(DescriptorSetError::NotInArray)
            } else {
                let descriptor = &builder.descriptors[builder.cur_binding as usize];
                let inner_desc = match descriptor.desc.as_ref() {
                    Some(some) => some,
                    None => unreachable!(),
                };

                if inner_desc.variable_count {
                    if descriptor.array_element > inner_desc.descriptor_count {
                        return Err(DescriptorSetError::ArrayTooManyDescriptors {
                            capacity: inner_desc.descriptor_count,
                            obtained: descriptor.array_element,
                        });
                    }
                } else if descriptor.array_element != inner_desc.descriptor_count {
                    return Err(DescriptorSetError::ArrayLengthMismatch {
                        expected: inner_desc.descriptor_count,
                        obtained: descriptor.array_element,
                    });
                }

                builder.in_array = false;
                builder.cur_binding += 1;
                Ok(())
            }
        })
    }

    /// Skips the current descriptor if it is empty.
    pub fn add_empty(&mut self) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            // Should be unreahable as enter_array prevents entering an array for empty descriptors.
            assert!(!builder.in_array);

            if builder.descriptors[builder.cur_binding as usize]
                .desc
                .is_some()
            {
                return Err(DescriptorSetError::WrongDescriptorType);
            }

            builder.cur_binding += 1;
            Ok(())
        })
    }

    /// Binds a buffer as the next descriptor or array element.
    pub fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess + 'static>,
    ) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            if buffer.inner().buffer.device().internal_object()
                != builder.layout.device().internal_object()
            {
                return Err(DescriptorSetError::ResourceWrongDevice);
            }

            let leave_array = if !builder.in_array {
                builder.enter_array()?;
                true
            } else {
                false
            };

            let descriptor = &mut builder.descriptors[builder.cur_binding as usize];
            let inner_desc = match descriptor.desc.as_ref() {
                Some(some) => some,
                None => return Err(DescriptorSetError::WrongDescriptorType),
            };

            // Note that the buffer content is not checked. This is technically not unsafe as
            // long as the data in the buffer has no invalid memory representation (ie. no
            // bool, no enum, no pointer, no str) and as long as the robust buffer access
            // feature is enabled.
            // TODO: this is not checked ^

            // TODO: eventually shouldn't be an assert ; for now robust_buffer_access is always
            //       enabled so this assert should never fail in practice, but we put it anyway
            //       in case we forget to adjust this code

            builder.desc_writes.push(match inner_desc.ty {
                DescriptorDescTy::StorageBuffer | DescriptorDescTy::StorageBufferDynamic => {
                    assert!(
                        builder
                            .layout
                            .device()
                            .enabled_features()
                            .robust_buffer_access
                    );

                    if !buffer.inner().buffer.usage().storage_buffer {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::storage_buffer(
                            builder.cur_binding,
                            descriptor.array_element,
                            [&buffer],
                        )
                    }
                }
                DescriptorDescTy::UniformBuffer => {
                    assert!(
                        builder
                            .layout
                            .device()
                            .enabled_features()
                            .robust_buffer_access
                    );

                    if !buffer.inner().buffer.usage().uniform_buffer {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::uniform_buffer(
                            builder.cur_binding,
                            descriptor.array_element,
                            [&buffer],
                        )
                    }
                }
                DescriptorDescTy::UniformBufferDynamic => {
                    assert!(
                        builder
                            .layout
                            .device()
                            .enabled_features()
                            .robust_buffer_access
                    );

                    if !buffer.inner().buffer.usage().uniform_buffer {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::dynamic_uniform_buffer(
                            builder.cur_binding,
                            descriptor.array_element,
                            [&buffer],
                        )
                    }
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            });

            builder.resources.add_buffer(builder.cur_binding, buffer);
            descriptor.array_element += 1;

            if leave_array {
                builder.leave_array()?;
            }

            Ok(())
        })
    }

    /// Binds a buffer view as the next descriptor or array element.
    pub fn add_buffer_view<B>(
        &mut self,
        view: Arc<BufferView<B>>,
    ) -> Result<&mut Self, DescriptorSetError>
    where
        B: BufferAccess + 'static,
    {
        self.poison_on_err(|builder| {
            if view.device().internal_object() != builder.layout.device().internal_object() {
                return Err(DescriptorSetError::ResourceWrongDevice);
            }

            let leave_array = if !builder.in_array {
                builder.enter_array()?;
                true
            } else {
                false
            };

            let descriptor = &mut builder.descriptors[builder.cur_binding as usize];
            let inner_desc = match descriptor.desc.as_ref() {
                Some(some) => some,
                None => return Err(DescriptorSetError::WrongDescriptorType),
            };

            builder.desc_writes.push(match inner_desc.ty {
                DescriptorDescTy::StorageTexelBuffer { .. } => {
                    // TODO: storage_texel_buffer_atomic

                    if !view.storage_texel_buffer() {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageTexelBuffer,
                        ));
                    }

                    DescriptorWrite::storage_texel_buffer(
                        builder.cur_binding,
                        descriptor.array_element,
                        [&*view],
                    )
                }
                DescriptorDescTy::UniformTexelBuffer { .. } => {
                    if !view.uniform_texel_buffer() {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformTexelBuffer,
                        ));
                    }

                    DescriptorWrite::uniform_texel_buffer(
                        builder.cur_binding,
                        descriptor.array_element,
                        [&*view],
                    )
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            });

            builder.resources.add_buffer_view(builder.cur_binding, view);
            descriptor.array_element += 1;

            if leave_array {
                builder.leave_array()?;
            }

            Ok(())
        })
    }

    /// Binds an image view as the next descriptor or array element.
    pub fn add_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + 'static>,
    ) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            if image_view.image().inner().image.device().internal_object()
                != builder.layout.device().internal_object()
            {
                return Err(DescriptorSetError::ResourceWrongDevice);
            }

            let leave_array = if !builder.in_array {
                builder.enter_array()?;
                true
            } else {
                false
            };

            let descriptor = &mut builder.descriptors[builder.cur_binding as usize];
            let inner_desc = match descriptor.desc.as_ref() {
                Some(some) => some,
                None => return Err(DescriptorSetError::WrongDescriptorType),
            };

            builder.desc_writes.push(match &inner_desc.ty {
                DescriptorDescTy::CombinedImageSampler {
                    image_desc,
                    immutable_samplers,
                } if !immutable_samplers.is_empty() => {
                    if !image_view.image().inner().image.usage().sampled {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::Sampled,
                        ));
                    }

                    if !image_view
                        .can_be_sampled(&immutable_samplers[descriptor.array_element as usize])
                    {
                        return Err(DescriptorSetError::IncompatibleImageViewSampler);
                    }

                    image_match_desc(&image_view, image_desc)?;

                    DescriptorWrite::combined_image_sampler(
                        builder.cur_binding,
                        descriptor.array_element,
                        [(None, &image_view)],
                    )
                }
                DescriptorDescTy::SampledImage { ref image_desc, .. } => {
                    if !image_view.image().inner().image.usage().sampled {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::Sampled,
                        ));
                    }

                    image_match_desc(&image_view, image_desc)?;

                    DescriptorWrite::sampled_image(
                        builder.cur_binding,
                        descriptor.array_element,
                        [&image_view],
                    )
                }
                DescriptorDescTy::StorageImage { ref image_desc, .. } => {
                    if !image_view.image().inner().image.usage().storage {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::Storage,
                        ));
                    }

                    image_match_desc(&image_view, image_desc)?;

                    if !image_view.component_mapping().is_identity() {
                        return Err(DescriptorSetError::NotIdentitySwizzled);
                    }

                    DescriptorWrite::storage_image(
                        builder.cur_binding,
                        descriptor.array_element,
                        [&image_view],
                    )
                }
                DescriptorDescTy::InputAttachment { multisampled } => {
                    if !image_view.image().inner().image.usage().input_attachment {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::InputAttachment,
                        ));
                    }

                    if !image_view.component_mapping().is_identity() {
                        return Err(DescriptorSetError::NotIdentitySwizzled);
                    }

                    if *multisampled && image_view.image().samples() == SampleCount::Sample1 {
                        return Err(DescriptorSetError::ExpectedMultisampled);
                    } else if !multisampled && image_view.image().samples() != SampleCount::Sample1
                    {
                        return Err(DescriptorSetError::UnexpectedMultisampled);
                    }

                    let image_layers = image_view.array_layers();
                    let num_layers = image_layers.end - image_layers.start;

                    if image_view.ty().is_arrayed() {
                        return Err(DescriptorSetError::UnexpectedArrayed);
                    }

                    DescriptorWrite::input_attachment(
                        builder.cur_binding,
                        descriptor.array_element,
                        [&image_view],
                    )
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            });

            descriptor.array_element += 1;
            builder.resources.add_image(builder.cur_binding, image_view);

            if leave_array {
                builder.leave_array()?;
            }

            Ok(())
        })
    }

    /// Binds an image view with a sampler as the next descriptor or array element.
    ///
    /// If the descriptor set layout contains immutable samplers for this descriptor, use
    /// `add_image` instead.
    pub fn add_sampled_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + 'static>,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            if image_view.image().inner().image.device().internal_object()
                != builder.layout.device().internal_object()
            {
                return Err(DescriptorSetError::ResourceWrongDevice);
            }

            if sampler.device().internal_object() != builder.layout.device().internal_object() {
                return Err(DescriptorSetError::ResourceWrongDevice);
            }

            if !image_view.image().inner().image.usage().sampled {
                return Err(DescriptorSetError::MissingImageUsage(
                    MissingImageUsage::Sampled,
                ));
            }

            if !image_view.can_be_sampled(&sampler) {
                return Err(DescriptorSetError::IncompatibleImageViewSampler);
            }

            let leave_array = if !builder.in_array {
                builder.enter_array()?;
                true
            } else {
                false
            };

            let descriptor = &mut builder.descriptors[builder.cur_binding as usize];
            let inner_desc = match descriptor.desc.as_ref() {
                Some(some) => some,
                None => return Err(DescriptorSetError::WrongDescriptorType),
            };

            builder.desc_writes.push(match &inner_desc.ty {
                DescriptorDescTy::CombinedImageSampler {
                    image_desc,
                    immutable_samplers,
                } => {
                    if !immutable_samplers.is_empty() {
                        return Err(DescriptorSetError::SamplerIsImmutable);
                    }

                    image_match_desc(&image_view, image_desc)?;

                    DescriptorWrite::combined_image_sampler(
                        builder.cur_binding,
                        descriptor.array_element,
                        [(Some(&sampler), &image_view)],
                    )
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            });

            descriptor.array_element += 1;
            builder.resources.add_image(builder.cur_binding, image_view);
            builder.resources.add_sampler(builder.cur_binding, sampler);

            if leave_array {
                builder.leave_array()?;
            }

            Ok(())
        })
    }

    /// Binds a sampler as the next descriptor or array element.
    pub fn add_sampler(&mut self, sampler: Arc<Sampler>) -> Result<&mut Self, DescriptorSetError> {
        self.poison_on_err(|builder| {
            if sampler.device().internal_object() != builder.layout.device().internal_object() {
                return Err(DescriptorSetError::ResourceWrongDevice);
            }

            let leave_array = if !builder.in_array {
                builder.enter_array()?;
                true
            } else {
                false
            };

            let descriptor = &mut builder.descriptors[builder.cur_binding as usize];
            let inner_desc = match descriptor.desc.as_ref() {
                Some(some) => some,
                None => return Err(DescriptorSetError::WrongDescriptorType),
            };

            builder.desc_writes.push(match &inner_desc.ty {
                DescriptorDescTy::Sampler { immutable_samplers } => {
                    if !immutable_samplers.is_empty() {
                        return Err(DescriptorSetError::SamplerIsImmutable);
                    }

                    DescriptorWrite::sampler(
                        builder.cur_binding,
                        descriptor.array_element,
                        [&sampler],
                    )
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            });

            descriptor.array_element += 1;
            builder.resources.add_sampler(builder.cur_binding, sampler);

            if leave_array {
                builder.leave_array()?;
            }

            Ok(())
        })
    }
}

unsafe impl DeviceOwned for DescriptorSetBuilder {
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

/// The output of the descriptor set builder.
///
/// This is not a descriptor set yet, but can be used to write the descriptors to one.
pub struct DescriptorSetBuilderOutput {
    layout: Arc<DescriptorSetLayout>,
    writes: Vec<DescriptorWrite>,
    resources: DescriptorSetResources,
}

impl DescriptorSetBuilderOutput {
    /// Returns the descriptor set layout.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.layout
    }

    /// Returns the resources used by the output.
    #[inline]
    pub fn resources(&self) -> &DescriptorSetResources {
        &self.resources
    }

    /// Returns the descriptor writes.
    #[inline]
    pub fn writes(&self) -> &[DescriptorWrite] {
        &self.writes
    }
}

impl From<DescriptorSetBuilderOutput>
    for (
        Arc<DescriptorSetLayout>,
        Vec<DescriptorWrite>,
        DescriptorSetResources,
    )
{
    #[inline]
    fn from(output: DescriptorSetBuilderOutput) -> Self {
        (output.layout, output.writes, output.resources)
    }
}

// Checks whether an image view matches the descriptor.
fn image_match_desc<I>(image_view: &I, desc: &DescriptorDescImage) -> Result<(), DescriptorSetError>
where
    I: ?Sized + ImageViewAbstract,
{
    if image_view.ty() != desc.view_type
        && match desc.view_type {
            ImageViewType::Dim1dArray => image_view.ty() != ImageViewType::Dim1d,
            ImageViewType::Dim2dArray => image_view.ty() != ImageViewType::Dim2d,
            ImageViewType::CubeArray => image_view.ty() != ImageViewType::Cube,
            _ => true,
        }
    {
        return Err(DescriptorSetError::ImageViewTypeMismatch {
            expected: desc.view_type,
            obtained: image_view.ty(),
        });
    }

    if let Some(format) = desc.format {
        if image_view.format() != format {
            return Err(DescriptorSetError::ImageViewFormatMismatch {
                expected: format,
                obtained: image_view.format(),
            });
        }
    }

    if desc.multisampled && image_view.image().samples() == SampleCount::Sample1 {
        return Err(DescriptorSetError::ExpectedMultisampled);
    } else if !desc.multisampled && image_view.image().samples() != SampleCount::Sample1 {
        return Err(DescriptorSetError::UnexpectedMultisampled);
    }

    Ok(())
}
