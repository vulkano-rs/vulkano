// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::{BufferAccess, BufferView};
use crate::descriptor_set::layout::{DescriptorDesc, DescriptorType};
use crate::descriptor_set::sys::DescriptorWrite;
use crate::descriptor_set::{
    DescriptorSetError, DescriptorSetLayout, MissingBufferUsage, MissingImageUsage,
};
use crate::device::{Device, DeviceOwned};
use crate::image::ImageViewAbstract;
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
    variable_descriptor_count: u32,
    writes: Vec<DescriptorWrite>,

    cur_binding: u32,
    descriptors: Vec<BuilderDescriptor>,
    in_array: bool,
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

        for binding_i in 0..layout.num_bindings() {
            if let Some(desc) = layout.descriptor(binding_i) {
                desc_writes_capacity += desc.descriptor_count as usize;
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
            variable_descriptor_count: 0,
            writes: Vec::with_capacity(desc_writes_capacity),

            cur_binding: 0,
            descriptors,
            in_array: false,
            poisoned: false,
        }
    }

    /// Finalizes the building process and returns the generated output.
    pub fn build(self) -> Result<DescriptorSetBuilderOutput, DescriptorSetError> {
        if self.poisoned {
            return Err(DescriptorSetError::BuilderPoisoned);
        }

        if self.in_array {
            return Err(DescriptorSetError::InArray);
        }

        if self.cur_binding != self.descriptors.len() as u32 {
            return Err(DescriptorSetError::DescriptorsMissing {
                expected: self.descriptors.len() as u32,
                obtained: self.cur_binding,
            });
        } else {
            Ok(DescriptorSetBuilderOutput {
                layout: self.layout,
                writes: self.writes,
                variable_descriptor_count: self.variable_descriptor_count,
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
                Err(DescriptorSetError::InArray)
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

                    debug_assert!(builder.cur_binding as usize == builder.descriptors.len() - 1);
                    debug_assert!(builder.variable_descriptor_count == 0);
                    builder.variable_descriptor_count = descriptor.array_element;
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
            // Should be unreachable as enter_array prevents entering an array for empty descriptors.
            debug_assert!(!builder.in_array);

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
        buffer: Arc<dyn BufferAccess>,
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

            match inner_desc.ty {
                DescriptorType::StorageBuffer | DescriptorType::StorageBufferDynamic => {
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
                }
                DescriptorType::UniformBuffer => {
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
                }
                DescriptorType::UniformBufferDynamic => {
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
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            }

            unsafe {
                builder.writes.push(DescriptorWrite::buffer(
                    builder.cur_binding,
                    descriptor.array_element,
                    [buffer],
                ));
            }

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

            match inner_desc.ty {
                DescriptorType::StorageTexelBuffer => {
                    // TODO: storage_texel_buffer_atomic

                    if !view.storage_texel_buffer() {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageTexelBuffer,
                        ));
                    }
                }
                DescriptorType::UniformTexelBuffer => {
                    if !view.uniform_texel_buffer() {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformTexelBuffer,
                        ));
                    }
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            }

            unsafe {
                builder.writes.push(DescriptorWrite::buffer_view(
                    builder.cur_binding,
                    descriptor.array_element,
                    [view as Arc<_>],
                ));
            }

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
        image_view: Arc<dyn ImageViewAbstract>,
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

            match &inner_desc.ty {
                DescriptorType::CombinedImageSampler
                    if !inner_desc.immutable_samplers.is_empty() =>
                {
                    if !image_view.image().inner().image.usage().sampled {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::Sampled,
                        ));
                    }

                    if !image_view.can_be_sampled(
                        &inner_desc.immutable_samplers[descriptor.array_element as usize],
                    ) {
                        return Err(DescriptorSetError::IncompatibleImageViewSampler);
                    }
                }
                DescriptorType::SampledImage => {
                    if !image_view.image().inner().image.usage().sampled {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::Sampled,
                        ));
                    }
                }
                DescriptorType::StorageImage => {
                    if !image_view.image().inner().image.usage().storage {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::Storage,
                        ));
                    }

                    if !image_view.component_mapping().is_identity() {
                        return Err(DescriptorSetError::NotIdentitySwizzled);
                    }
                }
                DescriptorType::InputAttachment => {
                    if !image_view.image().inner().image.usage().input_attachment {
                        return Err(DescriptorSetError::MissingImageUsage(
                            MissingImageUsage::InputAttachment,
                        ));
                    }

                    if !image_view.component_mapping().is_identity() {
                        return Err(DescriptorSetError::NotIdentitySwizzled);
                    }

                    let image_layers = image_view.array_layers();
                    let num_layers = image_layers.end - image_layers.start;

                    if image_view.ty().is_arrayed() {
                        return Err(DescriptorSetError::UnexpectedArrayed);
                    }
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            }

            unsafe {
                builder.writes.push(DescriptorWrite::image_view(
                    builder.cur_binding,
                    descriptor.array_element,
                    [image_view],
                ));
            }

            descriptor.array_element += 1;

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
        image_view: Arc<dyn ImageViewAbstract>,
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

            match &inner_desc.ty {
                DescriptorType::CombinedImageSampler => {
                    if !inner_desc.immutable_samplers.is_empty() {
                        return Err(DescriptorSetError::SamplerIsImmutable);
                    }
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            }

            unsafe {
                builder.writes.push(DescriptorWrite::image_view_sampler(
                    builder.cur_binding,
                    descriptor.array_element,
                    [(image_view, sampler)],
                ));
            }

            descriptor.array_element += 1;

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

            match &inner_desc.ty {
                DescriptorType::Sampler => {
                    if !inner_desc.immutable_samplers.is_empty() {
                        return Err(DescriptorSetError::SamplerIsImmutable);
                    }
                }
                _ => return Err(DescriptorSetError::WrongDescriptorType),
            }

            unsafe {
                builder.writes.push(DescriptorWrite::sampler(
                    builder.cur_binding,
                    descriptor.array_element,
                    [sampler],
                ));
            }

            descriptor.array_element += 1;

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
    variable_descriptor_count: u32,
    writes: Vec<DescriptorWrite>,
}

impl DescriptorSetBuilderOutput {
    /// Returns the descriptor set layout.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.layout
    }

    /// Returns the number of variable descriptors.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.variable_descriptor_count
    }

    /// Returns the descriptor writes.
    #[inline]
    pub fn writes(&self) -> &[DescriptorWrite] {
        &self.writes
    }
}
