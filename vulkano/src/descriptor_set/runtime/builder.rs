use super::bound::BoundResources;
use super::{MissingBufferUsage, MissingImageUsage, RuntimeDescriptorSetError};
use crate::descriptor_set::layout::DescriptorDesc;
use crate::descriptor_set::layout::DescriptorDescTy;
use crate::descriptor_set::layout::DescriptorImageDesc;
use crate::descriptor_set::layout::DescriptorImageDescArray;
use crate::descriptor_set::layout::DescriptorImageDescDimensions;
use crate::descriptor_set::sys::DescriptorWrite;
use crate::descriptor_set::BufferAccess;
use crate::descriptor_set::DescriptorSetLayout;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::ImageViewAbstract;
use crate::image::SampleCount;
use crate::sampler::Sampler;
use crate::VulkanObject;
use std::sync::Arc;

struct RuntimeDescriptor {
    desc: DescriptorDesc,
    writes: Vec<DescriptorWrite>,
}

pub struct RuntimeDescriptorSetBuilder {
    device: Arc<Device>,
    in_array: bool,
    descriptors: Vec<RuntimeDescriptor>,
    cur_binding: usize,
    bound_resources: BoundResources,
}

impl RuntimeDescriptorSetBuilder {
    pub(super) fn with_rt_desc_capacity(
        layout: Arc<DescriptorSetLayout>,
        capacity: usize,
    ) -> Result<Self, RuntimeDescriptorSetError> {
        let device = layout.device().clone();
        let mut descriptors = Vec::with_capacity(layout.num_bindings());

        for binding_i in 0..layout.num_bindings() {
            let desc = layout.descriptor(binding_i).unwrap();
            let writes = match desc.array_count {
                0 => {
                    if binding_i != layout.num_bindings() - 1 {
                        return Err(RuntimeDescriptorSetError::RuntimeArrayMustBeLast);
                    } else {
                        Vec::with_capacity(capacity)
                    }
                }
                c => Vec::with_capacity(c as usize),
            };

            descriptors.push(RuntimeDescriptor { desc, writes });
        }

        Ok(Self {
            device,
            in_array: false,
            cur_binding: 0,
            descriptors,
            bound_resources: BoundResources::new(),
        })
    }

    pub fn build(self) -> Result<(), RuntimeDescriptorSetError> {
        if self.cur_binding != self.descriptors.len() {
            Err(RuntimeDescriptorSetError::DescriptorsMissing {
                expected: self.descriptors.len(),
                obtained: self.cur_binding,
            })
        } else {
            Ok(())
        }
    }

    pub fn enter_array(&mut self) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.in_array {
            Err(RuntimeDescriptorSetError::AlreadyInArray)
        } else if self.cur_binding >= self.descriptors.len() {
            Err(RuntimeDescriptorSetError::TooManyDescriptors)
        } else {
            self.in_array = true;
            Ok(self)
        }
    }

    pub fn leave_array(&mut self) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if !self.in_array {
            Err(RuntimeDescriptorSetError::NotInArray)
        } else {
            let descriptor = &self.descriptors[self.cur_binding];

            if descriptor.desc.array_count != 0
                && descriptor.writes.len() != descriptor.desc.array_count as usize
            {
                return Err(RuntimeDescriptorSetError::ArrayLengthMismatch {
                    expected: descriptor.desc.array_count as usize,
                    obtained: descriptor.writes.len(),
                });
            }

            self.in_array = false;
            self.cur_binding += 1;
            Ok(self)
        }
    }

    pub fn add_empty(&mut self) -> Result<&mut Self, RuntimeDescriptorSetError> {
        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        // TODO:

        if leave_array {
            self.leave_array()
        } else {
            Ok(self)
        }
    }

    pub fn add_buffer<T>(
        &mut self,
        buffer: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if buffer.inner().buffer.device().internal_object() != self.device.internal_object() {
            return Err(RuntimeDescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let array_element = descriptor.writes.len() as u32;

        match descriptor.desc.ty {
            DescriptorDescTy::Buffer(ref buffer_desc) => {
                descriptor.writes.push(if buffer_desc.storage {
                    if !buffer.inner().buffer.usage().storage_buffer {
                        return Err(RuntimeDescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::storage_buffer(
                            self.cur_binding as u32,
                            array_element,
                            &buffer,
                        )
                    }
                } else {
                    if !buffer.inner().buffer.usage().uniform_buffer {
                        return Err(RuntimeDescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformBuffer,
                        ));
                    }

                    if buffer_desc.dynamic.unwrap_or(false) {
                        unsafe {
                            DescriptorWrite::dynamic_uniform_buffer(
                                self.cur_binding as u32,
                                array_element,
                                &buffer,
                            )
                        }
                    } else {
                        unsafe {
                            DescriptorWrite::uniform_buffer(
                                self.cur_binding as u32,
                                array_element,
                                &buffer,
                            )
                        }
                    }
                });

                self.bound_resources
                    .add_buffer(self.cur_binding as u32, buffer);
            }
            _ => return Err(RuntimeDescriptorSetError::WrongDescriptorType),
        }

        if leave_array {
            self.leave_array()
        } else {
            Ok(self)
        }
    }

    pub fn add_image<T>(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if image_view.image().inner().image.device().internal_object()
            != self.device.internal_object()
        {
            return Err(RuntimeDescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let array_element = descriptor.writes.len() as u32;

        match descriptor.desc.ty {
            DescriptorDescTy::Image(ref desc) => {
                image_match_desc(&image_view, &desc)?;

                descriptor.writes.push(if desc.sampled {
                    DescriptorWrite::sampled_image(
                        self.cur_binding as u32,
                        array_element,
                        &image_view,
                    )
                } else if !image_view.component_mapping().is_identity() {
                    return Err(RuntimeDescriptorSetError::NotIdentitySwizzled);
                } else {
                    DescriptorWrite::storage_image(
                        self.cur_binding as u32,
                        array_element,
                        &image_view,
                    )
                });
            }
            DescriptorDescTy::InputAttachment {
                multisampled,
                array_layers,
            } => {
                if !image_view.image().inner().image.usage().input_attachment {
                    return Err(RuntimeDescriptorSetError::MissingImageUsage(
                        MissingImageUsage::InputAttachment,
                    ));
                }

                if !image_view.component_mapping().is_identity() {
                    return Err(RuntimeDescriptorSetError::NotIdentitySwizzled);
                }

                if multisampled && image_view.image().samples() == SampleCount::Sample1 {
                    return Err(RuntimeDescriptorSetError::ExpectedMultisampled);
                } else if !multisampled && image_view.image().samples() != SampleCount::Sample1 {
                    return Err(RuntimeDescriptorSetError::UnexpectedMultisampled);
                }

                let image_layers = image_view.array_layers();
                let num_layers = image_layers.end - image_layers.start;

                match array_layers {
                    DescriptorImageDescArray::NonArrayed => {
                        if num_layers != 1 {
                            return Err(RuntimeDescriptorSetError::ArrayLayersMismatch {
                                expected: 1,
                                obtained: num_layers,
                            });
                        }
                    }
                    DescriptorImageDescArray::Arrayed {
                        max_layers: Some(max_layers),
                    } => {
                        if num_layers > max_layers {
                            // TODO: is this correct? "max" layers? or is it in fact min layers?
                            return Err(RuntimeDescriptorSetError::ArrayLayersMismatch {
                                expected: max_layers,
                                obtained: num_layers,
                            });
                        }
                    }
                    DescriptorImageDescArray::Arrayed { max_layers: None } => {}
                }

                descriptor.writes.push(DescriptorWrite::input_attachment(
                    self.cur_binding as u32,
                    array_element,
                    &image_view,
                ));
            }
            _ => return Err(RuntimeDescriptorSetError::WrongDescriptorType),
        }

        self.bound_resources
            .add_image(self.cur_binding as u32, image_view);

        if leave_array {
            self.leave_array()
        } else {
            Ok(self)
        }
    }

    pub fn add_sampled_image<T>(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if image_view.image().inner().image.device().internal_object()
            != self.device.internal_object()
        {
            return Err(RuntimeDescriptorSetError::ResourceWrongDevice);
        }

        if sampler.device().internal_object() != self.device.internal_object() {
            return Err(RuntimeDescriptorSetError::ResourceWrongDevice);
        }

        if !image_view.can_be_sampled(&sampler) {
            return Err(RuntimeDescriptorSetError::IncompatibleImageViewSampler);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let array_element = descriptor.writes.len() as u32;

        match descriptor.desc.ty {
            DescriptorDescTy::CombinedImageSampler(ref desc) => {
                image_match_desc(&image_view, &desc)?;

                descriptor
                    .writes
                    .push(DescriptorWrite::combined_image_sampler(
                        self.cur_binding as u32,
                        array_element,
                        &sampler,
                        &image_view,
                    ));
            }
            _ => return Err(RuntimeDescriptorSetError::WrongDescriptorType),
        }

        self.bound_resources
            .add_image(self.cur_binding as u32, image_view);
        self.bound_resources
            .add_sampler(self.cur_binding as u32, sampler);

        if leave_array {
            self.leave_array()
        } else {
            Ok(self)
        }
    }

    pub fn add_sampler(
        &mut self,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if sampler.device().internal_object() != self.device.internal_object() {
            return Err(RuntimeDescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let array_element = descriptor.writes.len() as u32;

        match descriptor.desc.ty {
            DescriptorDescTy::Sampler => {
                descriptor.writes.push(DescriptorWrite::sampler(
                    self.cur_binding as u32,
                    array_element,
                    &sampler,
                ));
            }
            _ => return Err(RuntimeDescriptorSetError::WrongDescriptorType),
        }

        self.bound_resources
            .add_sampler(self.cur_binding as u32, sampler);

        if leave_array {
            self.leave_array()
        } else {
            Ok(self)
        }
    }
}

// Checks whether an image view matches the descriptor.
fn image_match_desc<I>(
    image_view: &I,
    desc: &DescriptorImageDesc,
) -> Result<(), RuntimeDescriptorSetError>
where
    I: ?Sized + ImageViewAbstract,
{
    if desc.sampled && !image_view.image().inner().image.usage().sampled {
        return Err(RuntimeDescriptorSetError::MissingImageUsage(
            MissingImageUsage::Sampled,
        ));
    } else if !desc.sampled && !image_view.image().inner().image.usage().storage {
        return Err(RuntimeDescriptorSetError::MissingImageUsage(
            MissingImageUsage::Storage,
        ));
    }

    let image_view_ty = DescriptorImageDescDimensions::from_image_view_type(image_view.ty());

    if image_view_ty != desc.dimensions {
        return Err(RuntimeDescriptorSetError::ImageViewTypeMismatch {
            expected: desc.dimensions,
            obtained: image_view_ty,
        });
    }

    if let Some(format) = desc.format {
        if image_view.format() != format {
            return Err(RuntimeDescriptorSetError::ImageViewFormatMismatch {
                expected: format,
                obtained: image_view.format(),
            });
        }
    }

    if desc.multisampled && image_view.image().samples() == SampleCount::Sample1 {
        return Err(RuntimeDescriptorSetError::ExpectedMultisampled);
    } else if !desc.multisampled && image_view.image().samples() != SampleCount::Sample1 {
        return Err(RuntimeDescriptorSetError::UnexpectedMultisampled);
    }

    let image_layers = image_view.array_layers();
    let num_layers = image_layers.end - image_layers.start;

    match desc.array_layers {
        DescriptorImageDescArray::NonArrayed => {
            // TODO: when a non-array is expected, can we pass an image view that is in fact an
            // array with one layer? need to check
            let required_layers = if desc.dimensions == DescriptorImageDescDimensions::Cube {
                6
            } else {
                1
            };

            if num_layers != required_layers {
                return Err(RuntimeDescriptorSetError::ArrayLayersMismatch {
                    expected: 1,
                    obtained: num_layers,
                });
            }
        }
        DescriptorImageDescArray::Arrayed {
            max_layers: Some(max_layers),
        } => {
            let required_layers = if desc.dimensions == DescriptorImageDescDimensions::Cube {
                max_layers * 6
            } else {
                max_layers
            };

            // TODO: is this correct? "max" layers? or is it in fact min layers?
            if num_layers > required_layers {
                return Err(RuntimeDescriptorSetError::ArrayLayersMismatch {
                    expected: max_layers,
                    obtained: num_layers,
                });
            }
        }
        DescriptorImageDescArray::Arrayed { max_layers: None } => {}
    };

    Ok(())
}
