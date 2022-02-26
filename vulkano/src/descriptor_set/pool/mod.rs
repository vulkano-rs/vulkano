// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pool from which descriptor sets can be allocated.

pub use self::{
    standard::StdDescriptorPool,
    sys::{
        DescriptorPoolAllocError, DescriptorSetAllocateInfo, UnsafeDescriptorPool,
        UnsafeDescriptorPoolCreateInfo,
    },
};
use super::{layout::DescriptorSetLayout, sys::UnsafeDescriptorSet, DescriptorType};
use crate::{device::DeviceOwned, OomError};
use fnv::FnvHashMap;
use std::{cmp, ops};

pub mod standard;
mod sys;

/// A pool from which descriptor sets can be allocated.
///
/// Since the destructor of `Alloc` must free the descriptor set, this trait is usually implemented
/// on `Arc<T>` or `&'a T` and not `T` directly, so that the `Alloc` object can hold the pool.
pub unsafe trait DescriptorPool: DeviceOwned {
    /// Object that represented an allocated descriptor set.
    ///
    /// The destructor of this object should free the descriptor set.
    type Alloc: DescriptorPoolAlloc;

    /// Allocates a descriptor set.
    fn alloc(
        &mut self,
        layout: &DescriptorSetLayout,
        variable_descriptor_count: u32,
    ) -> Result<Self::Alloc, OomError>;
}

/// An allocated descriptor set.
pub trait DescriptorPoolAlloc: Send + Sync {
    /// Returns the inner unsafe descriptor set object.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the inner unsafe descriptor set object.
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet;
}

macro_rules! descriptors_count {
    ($($name:ident,)+) => (
        /// Number of available descriptors slots in a pool.
        ///
        /// # Example
        ///
        /// ```
        /// use vulkano::descriptor_set::pool::DescriptorsCount;
        ///
        /// let _descriptors = DescriptorsCount {
        ///     uniform_buffer: 10,
        ///     input_attachment: 5,
        ///     .. DescriptorsCount::zero()
        /// };
        /// ```
        ///
        #[derive(Debug, Copy, Clone)]
        pub struct DescriptorsCount {
            $(
                pub $name: u32,
            )+
        }

        impl DescriptorsCount {
            /// Returns a `DescriptorsCount` object with all fields set to 0.
            #[inline]
            pub fn zero() -> DescriptorsCount {
                DescriptorsCount {
                    $(
                        $name: 0,
                    )+
                }
            }

            /// Returns the total number of descriptors.
            #[inline]
            pub fn total(&self) -> u32 {
                [$(self.$name,)+].into_iter().sum()
            }

            /// Adds one descriptor of the given type to the count.
            #[inline]
            pub fn add_one(&mut self, ty: DescriptorType) {
                self.add_num(ty, 1);
            }

            /// Adds `num` descriptors of the given type to the count.
            #[inline]
            pub fn add_num(&mut self, ty: DescriptorType, num: u32) {
                match ty {
                    DescriptorType::Sampler => self.sampler += num,
                    DescriptorType::CombinedImageSampler => self.combined_image_sampler += num,
                    DescriptorType::SampledImage => self.sampled_image += num,
                    DescriptorType::StorageImage => self.storage_image += num,
                    DescriptorType::UniformTexelBuffer => self.uniform_texel_buffer += num,
                    DescriptorType::StorageTexelBuffer => self.storage_texel_buffer += num,
                    DescriptorType::UniformBuffer => self.uniform_buffer += num,
                    DescriptorType::StorageBuffer => self.storage_buffer += num,
                    DescriptorType::UniformBufferDynamic => self.uniform_buffer_dynamic += num,
                    DescriptorType::StorageBufferDynamic => self.storage_buffer_dynamic += num,
                    DescriptorType::InputAttachment => self.input_attachment += num,
                };
            }
        }

        impl From<DescriptorsCount> for FnvHashMap<DescriptorType, u32> {
            #[inline]
            fn from(val: DescriptorsCount) -> Self {
                let mut result = FnvHashMap::with_capacity_and_hasher(
                    val.total() as usize,
                    Default::default(),
                );

                if val.sampler != 0 {
                    result.insert(DescriptorType::Sampler, val.sampler);
                }

                if val.combined_image_sampler != 0 {
                    result.insert(DescriptorType::CombinedImageSampler, val.combined_image_sampler);
                }

                if val.sampled_image != 0 {
                    result.insert(DescriptorType::SampledImage, val.sampled_image);
                }

                if val.storage_image != 0 {
                    result.insert(DescriptorType::StorageImage, val.storage_image);
                }

                if val.uniform_texel_buffer != 0 {
                    result.insert(DescriptorType::UniformTexelBuffer, val.uniform_texel_buffer);
                }

                if val.storage_texel_buffer != 0 {
                    result.insert(DescriptorType::StorageTexelBuffer, val.storage_texel_buffer);
                }

                if val.uniform_buffer != 0 {
                    result.insert(DescriptorType::UniformBuffer, val.uniform_buffer);
                }

                if val.storage_buffer != 0 {
                    result.insert(DescriptorType::StorageBuffer, val.storage_buffer);
                }

                if val.uniform_buffer_dynamic != 0 {
                    result.insert(DescriptorType::UniformBufferDynamic, val.uniform_buffer_dynamic);
                }

                if val.storage_buffer_dynamic != 0 {
                    result.insert(DescriptorType::StorageBufferDynamic, val.storage_buffer_dynamic);
                }

                if val.input_attachment != 0 {
                    result.insert(DescriptorType::InputAttachment, val.input_attachment);
                }

                result
            }
        }

        impl cmp::PartialEq for DescriptorsCount {
            #[inline]
            fn eq(&self, other: &DescriptorsCount) -> bool {
                self.partial_cmp(other) == Some(cmp::Ordering::Equal)
            }
        }

        impl cmp::Eq for DescriptorsCount {
        }

        impl cmp::PartialOrd for DescriptorsCount {
            fn partial_cmp(&self, other: &DescriptorsCount) -> Option<cmp::Ordering> {
                if $(self.$name > other.$name)&&+ {
                    Some(cmp::Ordering::Greater)
                } else if $(self.$name < other.$name)&&+ {
                    Some(cmp::Ordering::Less)
                } else if $(self.$name == other.$name)&&+ {
                    Some(cmp::Ordering::Equal)
                } else {
                    None
                }
            }

            fn le(&self, other: &DescriptorsCount) -> bool {
                $(self.$name <= other.$name)&&+
            }

            fn ge(&self, other: &DescriptorsCount) -> bool {
                $(self.$name >= other.$name)&&+
            }
        }

        impl ops::Sub for DescriptorsCount {
            type Output = DescriptorsCount;

            #[inline]
            fn sub(self, rhs: DescriptorsCount) -> DescriptorsCount {
                DescriptorsCount {
                    $(
                        $name: self.$name - rhs.$name,
                    )+
                }
            }
        }

        impl ops::SubAssign for DescriptorsCount {
            #[inline]
            fn sub_assign(&mut self, rhs: DescriptorsCount) {
                $(
                    self.$name -= rhs.$name;
                )+
            }
        }

        impl ops::Add for DescriptorsCount {
            type Output = DescriptorsCount;

            #[inline]
            fn add(self, rhs: DescriptorsCount) -> DescriptorsCount {
                DescriptorsCount {
                    $(
                        $name: self.$name + rhs.$name,
                    )+
                }
            }
        }

        impl ops::AddAssign for DescriptorsCount {
            #[inline]
            fn add_assign(&mut self, rhs: DescriptorsCount) {
                $(
                    self.$name += rhs.$name;
                )+
            }
        }

        impl ops::Mul<u32> for DescriptorsCount {
            type Output = DescriptorsCount;

            #[inline]
            fn mul(self, rhs: u32) -> DescriptorsCount {
                DescriptorsCount {
                    $(
                        $name: self.$name * rhs,
                    )+
                }
            }
        }

        impl ops::MulAssign<u32> for DescriptorsCount {
            #[inline]
            fn mul_assign(&mut self, rhs: u32) {
                $(
                    self.$name *= rhs;
                )+
            }
        }
    );
}

descriptors_count! {
    uniform_buffer,
    storage_buffer,
    uniform_buffer_dynamic,
    storage_buffer_dynamic,
    uniform_texel_buffer,
    storage_texel_buffer,
    sampled_image,
    storage_image,
    sampler,
    combined_image_sampler,
    input_attachment,
}
