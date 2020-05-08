// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::cmp;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ops;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use buffer::BufferAccess;
use buffer::BufferInner;
use buffer::BufferView;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use device::Device;
use device::DeviceOwned;
use image::ImageViewAccess;
use sampler::Sampler;

use OomError;
use VulkanObject;
use check_errors;
use vk;

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
    fn alloc(&mut self, layout: &UnsafeDescriptorSetLayout) -> Result<Self::Alloc, OomError>;
}

/// An allocated descriptor set.
pub trait DescriptorPoolAlloc {
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
        /// use vulkano::descriptor::descriptor_set::DescriptorsCount;
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

            /// Adds one descriptor of the given type to the count.
            #[inline]
            pub fn add_one(&mut self, ty: DescriptorType) {
                match ty {
                    DescriptorType::Sampler => self.sampler += 1,
                    DescriptorType::CombinedImageSampler => self.combined_image_sampler += 1,
                    DescriptorType::SampledImage => self.sampled_image += 1,
                    DescriptorType::StorageImage => self.storage_image += 1,
                    DescriptorType::UniformTexelBuffer => self.uniform_texel_buffer += 1,
                    DescriptorType::StorageTexelBuffer => self.storage_texel_buffer += 1,
                    DescriptorType::UniformBuffer => self.uniform_buffer += 1,
                    DescriptorType::StorageBuffer => self.storage_buffer += 1,
                    DescriptorType::UniformBufferDynamic => self.uniform_buffer_dynamic += 1,
                    DescriptorType::StorageBufferDynamic => self.storage_buffer_dynamic += 1,
                    DescriptorType::InputAttachment => self.input_attachment += 1,
                };
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

/// Pool from which descriptor sets are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
pub struct UnsafeDescriptorPool {
    pool: vk::DescriptorPool,
    device: Arc<Device>,
}

impl UnsafeDescriptorPool {
    /// Initializes a new pool.
    ///
    /// Initializes a pool whose capacity is given by `count` and `max_sets`. At most `count`
    /// descriptors or `max_sets` descriptor sets can be allocated at once with this pool.
    ///
    /// If `free_descriptor_set_bit` is `true`, then individual descriptor sets can be free'd from
    /// the pool. Otherwise you must reset or destroy the whole pool at once.
    ///
    /// # Panic
    ///
    /// - Panics if all the descriptors count are 0.
    /// - Panics if `max_sets` is 0.
    ///
    pub fn new(device: Arc<Device>, count: &DescriptorsCount, max_sets: u32,
               free_descriptor_set_bit: bool)
               -> Result<UnsafeDescriptorPool, OomError> {
        let vk = device.pointers();

        assert_ne!(max_sets, 0, "The maximum number of sets can't be 0");

        let mut pool_sizes: SmallVec<[_; 10]> = SmallVec::new();

        macro_rules! elem {
            ($field:ident, $ty:expr) => (
                if count.$field >= 1 {
                    pool_sizes.push(vk::DescriptorPoolSize {
                        ty: $ty,
                        descriptorCount: count.$field,
                    });
                }
            );
        }

        elem!(uniform_buffer, vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        elem!(storage_buffer, vk::DESCRIPTOR_TYPE_STORAGE_BUFFER);
        elem!(uniform_buffer_dynamic,
              vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
        elem!(storage_buffer_dynamic,
              vk::DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
        elem!(uniform_texel_buffer,
              vk::DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER);
        elem!(storage_texel_buffer,
              vk::DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
        elem!(sampled_image, vk::DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        elem!(storage_image, vk::DESCRIPTOR_TYPE_STORAGE_IMAGE);
        elem!(sampler, vk::DESCRIPTOR_TYPE_SAMPLER);
        elem!(combined_image_sampler,
              vk::DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        elem!(input_attachment, vk::DESCRIPTOR_TYPE_INPUT_ATTACHMENT);

        assert!(!pool_sizes.is_empty(),
                "All the descriptors count of a pool are 0");

        let pool = unsafe {
            let infos = vk::DescriptorPoolCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: if free_descriptor_set_bit {
                    vk::DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
                } else {
                    0
                },
                maxSets: max_sets,
                poolSizeCount: pool_sizes.len() as u32,
                pPoolSizes: pool_sizes.as_ptr(),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(vk.CreateDescriptorPool(device.internal_object(),
                                                 &infos,
                                                 ptr::null(),
                                                 output.as_mut_ptr()))?;
            output.assume_init()
        };

        Ok(UnsafeDescriptorPool {
               pool: pool,
               device: device.clone(),
           })
    }

    /// Allocates descriptor sets from the pool, one for each layout.
    /// Returns an iterator to the allocated sets, or an error.
    ///
    /// The `FragmentedPool` errors often can't be prevented. If the function returns this error,
    /// you should just create a new pool.
    ///
    /// # Panic
    ///
    /// - Panics if one of the layouts wasn't created with the same device as the pool.
    ///
    /// # Safety
    ///
    /// See also the `new` function.
    ///
    /// - The total descriptors of the layouts must fit in the pool.
    /// - The total number of descriptor sets allocated from the pool must not overflow the pool.
    /// - You must ensure that the allocated descriptor sets are no longer in use when the pool
    ///   is destroyed, as destroying the pool is equivalent to freeing all the sets.
    ///
    #[inline]
    pub unsafe fn alloc<'l, I>(&mut self, layouts: I)
                               -> Result<UnsafeDescriptorPoolAllocIter, DescriptorPoolAllocError>
        where I: IntoIterator<Item = &'l UnsafeDescriptorSetLayout>
    {
        let layouts: SmallVec<[_; 8]> = layouts
            .into_iter()
            .map(|l| {
                     assert_eq!(self.device.internal_object(),
                                l.device().internal_object(),
                                "Tried to allocate from a pool with a set layout of a different \
                                 device");
                     l.internal_object()
                 })
            .collect();

        self.alloc_impl(&layouts)
    }

    // Actual implementation of `alloc`. Separated so that it is not inlined.
    unsafe fn alloc_impl(&mut self, layouts: &SmallVec<[vk::DescriptorSetLayout; 8]>)
                         -> Result<UnsafeDescriptorPoolAllocIter, DescriptorPoolAllocError> {
        let num = layouts.len();

        if num == 0 {
            return Ok(UnsafeDescriptorPoolAllocIter { sets: vec![].into_iter() });
        }

        let infos = vk::DescriptorSetAllocateInfo {
            sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext: ptr::null(),
            descriptorPool: self.pool,
            descriptorSetCount: layouts.len() as u32,
            pSetLayouts: layouts.as_ptr(),
        };

        let mut output = Vec::with_capacity(num);

        let vk = self.device.pointers();
        let ret =
            vk.AllocateDescriptorSets(self.device.internal_object(), &infos, output.as_mut_ptr());

        // According to the specs, because `VK_ERROR_FRAGMENTED_POOL` was added after version
        // 1.0 of Vulkan, any negative return value except out-of-memory errors must be
        // considered as a fragmented pool error.
        match ret {
            vk::ERROR_OUT_OF_HOST_MEMORY => {
                return Err(DescriptorPoolAllocError::OutOfHostMemory);
            },
            vk::ERROR_OUT_OF_DEVICE_MEMORY => {
                return Err(DescriptorPoolAllocError::OutOfDeviceMemory);
            },
            vk::ERROR_OUT_OF_POOL_MEMORY_KHR => {
                return Err(DescriptorPoolAllocError::OutOfPoolMemory);
            },
            c if (c as i32) < 0 => {
                return Err(DescriptorPoolAllocError::FragmentedPool);
            },
            _ => (),
        };

        output.set_len(num);

        Ok(UnsafeDescriptorPoolAllocIter { sets: output.into_iter() })
    }

    /// Frees some descriptor sets.
    ///
    /// Note that it is not mandatory to free sets. Destroying or resetting the pool destroys all
    /// the descriptor sets.
    ///
    /// # Safety
    ///
    /// - The pool must have been created with `free_descriptor_set_bit` set to `true`.
    /// - The descriptor sets must have been allocated from the pool.
    /// - The descriptor sets must not be free'd twice.
    /// - The descriptor sets must not be in use by the GPU.
    ///
    #[inline]
    pub unsafe fn free<I>(&mut self, descriptor_sets: I) -> Result<(), OomError>
        where I: IntoIterator<Item = UnsafeDescriptorSet>
    {
        let sets: SmallVec<[_; 8]> = descriptor_sets.into_iter().map(|s| s.set).collect();
        if !sets.is_empty() {
            self.free_impl(&sets)
        } else {
            Ok(())
        }
    }

    // Actual implementation of `free`. Separated so that it is not inlined.
    unsafe fn free_impl(&mut self, sets: &SmallVec<[vk::DescriptorSet; 8]>)
                        -> Result<(), OomError> {
        let vk = self.device.pointers();
        check_errors(vk.FreeDescriptorSets(self.device.internal_object(),
                                           self.pool,
                                           sets.len() as u32,
                                           sets.as_ptr()))?;
        Ok(())
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    pub unsafe fn reset(&mut self) -> Result<(), OomError> {
        let vk = self.device.pointers();
        check_errors(vk.ResetDescriptorPool(self.device.internal_object(),
                                            self.pool,
                                            0 /* reserved flags */))?;
        Ok(())
    }
}

unsafe impl DeviceOwned for UnsafeDescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl fmt::Debug for UnsafeDescriptorPool {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("UnsafeDescriptorPool")
            .field("raw", &self.pool)
            .field("device", &self.device)
            .finish()
    }
}

impl Drop for UnsafeDescriptorPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorPool(self.device.internal_object(), self.pool, ptr::null());
        }
    }
}

/// Error that can be returned when creating a device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DescriptorPoolAllocError {
    /// There is no memory available on the host (ie. the CPU, RAM, etc.).
    OutOfHostMemory,
    /// There is no memory available on the device (ie. video memory).
    OutOfDeviceMemory,
    /// Allocation has failed because the pool is too fragmented.
    FragmentedPool,
    /// There is no more space available in the descriptor pool.
    OutOfPoolMemory,
}

impl error::Error for DescriptorPoolAllocError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DescriptorPoolAllocError::OutOfHostMemory => {
                "no memory available on the host"
            },
            DescriptorPoolAllocError::OutOfDeviceMemory => {
                "no memory available on the graphical device"
            },
            DescriptorPoolAllocError::FragmentedPool => {
                "allocation has failed because the pool is too fragmented"
            },
            DescriptorPoolAllocError::OutOfPoolMemory => {
                "there is no more space available in the descriptor pool"
            },
        }
    }
}

impl fmt::Display for DescriptorPoolAllocError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

/// Iterator to the descriptor sets allocated from an unsafe descriptor pool.
#[derive(Debug)]
pub struct UnsafeDescriptorPoolAllocIter {
    sets: VecIntoIter<vk::DescriptorSet>,
}

impl Iterator for UnsafeDescriptorPoolAllocIter {
    type Item = UnsafeDescriptorSet;

    #[inline]
    fn next(&mut self) -> Option<UnsafeDescriptorSet> {
        self.sets.next().map(|s| UnsafeDescriptorSet { set: s })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.sets.size_hint()
    }
}

impl ExactSizeIterator for UnsafeDescriptorPoolAllocIter {
}

/// Low-level descriptor set.
///
/// Contrary to most other objects in this library, this one doesn't free itself automatically and
/// doesn't hold the pool or the device it is associated to.
/// Instead it is an object meant to be used with the `UnsafeDescriptorPool`.
pub struct UnsafeDescriptorSet {
    set: vk::DescriptorSet,
}

impl UnsafeDescriptorSet {
    // TODO: add copying from other descriptor sets
    //       add a `copy` method that just takes a copy, and an `update` method that takes both
    //       writes and copies and that actually performs the operation

    /// Modifies a descriptor set. Doesn't check that the writes or copies are correct, and
    /// doesn't check whether the descriptor set is in use.
    ///
    /// **Important**: You must ensure that the `UnsafeDescriptorSetLayout` object is alive before
    /// updating a descriptor set.
    ///
    /// # Safety
    ///
    /// - The `Device` must be the device the pool of this set was created with.
    /// - The `UnsafeDescriptorSetLayout` object this set was created with must be alive.
    /// - Doesn't verify that the things you write in the descriptor set match its layout.
    /// - Doesn't keep the resources alive. You have to do that yourself.
    /// - Updating a descriptor set obeys synchronization rules that aren't checked here. Once a
    ///   command buffer contains a pointer/reference to a descriptor set, it is illegal to write
    ///   to it.
    ///
    pub unsafe fn write<I>(&mut self, device: &Device, writes: I)
        where I: Iterator<Item = DescriptorWrite>
    {
        let vk = device.pointers();

        // In this function, we build 4 arrays: one array of image descriptors (image_descriptors),
        // one for buffer descriptors (buffer_descriptors), one for buffer view descriptors
        // (buffer_views_descriptors), and one for the final list of writes (raw_writes).
        // Only the final list is passed to Vulkan, but it will contain pointers to the first three
        // lists in `pImageInfo`, `pBufferInfo` and `pTexelBufferView`.
        //
        // In order to handle that, we start by writing null pointers as placeholders in the final
        // writes, and we store in `raw_writes_img_infos`, `raw_writes_buf_infos` and
        // `raw_writes_buf_view_infos` the offsets of the pointers compared to the start of the
        // list.
        // Once we have finished iterating all the writes requested by the user, we modify
        // `raw_writes` to point to the correct locations.

        let mut buffer_descriptors: SmallVec<[_; 64]> = SmallVec::new();
        let mut image_descriptors: SmallVec<[_; 64]> = SmallVec::new();
        let mut buffer_views_descriptors: SmallVec<[_; 64]> = SmallVec::new();

        let mut raw_writes: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_img_infos: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_buf_infos: SmallVec<[_; 64]> = SmallVec::new();
        let mut raw_writes_buf_view_infos: SmallVec<[_; 64]> = SmallVec::new();

        for indiv_write in writes {
            // Since the `DescriptorWrite` objects are built only through functions, we know for
            // sure that it's impossible to have an empty descriptor write.
            debug_assert!(!indiv_write.inner.is_empty());

            // The whole struct thats written here is valid, except for pImageInfo, pBufferInfo
            // and pTexelBufferView which are placeholder values.
            raw_writes.push(vk::WriteDescriptorSet {
                                sType: vk::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                pNext: ptr::null(),
                                dstSet: self.set,
                                dstBinding: indiv_write.binding,
                                dstArrayElement: indiv_write.first_array_element,
                                descriptorCount: indiv_write.inner.len() as u32,
                                descriptorType: indiv_write.ty() as u32,
                                pImageInfo: ptr::null(),
                                pBufferInfo: ptr::null(),
                                pTexelBufferView: ptr::null(),
                            });

            match indiv_write.inner[0] {
                DescriptorWriteInner::Sampler(_) |
                DescriptorWriteInner::CombinedImageSampler(_, _, _) |
                DescriptorWriteInner::SampledImage(_, _) |
                DescriptorWriteInner::StorageImage(_, _) |
                DescriptorWriteInner::InputAttachment(_, _) => {
                    raw_writes_img_infos.push(Some(image_descriptors.len()));
                    raw_writes_buf_infos.push(None);
                    raw_writes_buf_view_infos.push(None);
                },
                DescriptorWriteInner::UniformBuffer(_, _, _) |
                DescriptorWriteInner::StorageBuffer(_, _, _) |
                DescriptorWriteInner::DynamicUniformBuffer(_, _, _) |
                DescriptorWriteInner::DynamicStorageBuffer(_, _, _) => {
                    raw_writes_img_infos.push(None);
                    raw_writes_buf_infos.push(Some(buffer_descriptors.len()));
                    raw_writes_buf_view_infos.push(None);
                },
                DescriptorWriteInner::UniformTexelBuffer(_) |
                DescriptorWriteInner::StorageTexelBuffer(_) => {
                    raw_writes_img_infos.push(None);
                    raw_writes_buf_infos.push(None);
                    raw_writes_buf_view_infos.push(Some(buffer_views_descriptors.len()));
                },
            }

            for elem in indiv_write.inner.iter() {
                match *elem {
                    DescriptorWriteInner::UniformBuffer(buffer, offset, size) |
                    DescriptorWriteInner::DynamicUniformBuffer(buffer, offset, size) => {
                        buffer_descriptors.push(vk::DescriptorBufferInfo {
                                                    buffer: buffer,
                                                    offset: offset as u64,
                                                    range: size as u64,
                                                });
                    },
                    DescriptorWriteInner::StorageBuffer(buffer, offset, size) |
                    DescriptorWriteInner::DynamicStorageBuffer(buffer, offset, size) => {
                        buffer_descriptors.push(vk::DescriptorBufferInfo {
                                                    buffer: buffer,
                                                    offset: offset as u64,
                                                    range: size as u64,
                                                });
                    },
                    DescriptorWriteInner::Sampler(sampler) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                                                   sampler: sampler,
                                                   imageView: 0,
                                                   imageLayout: 0,
                                               });
                    },
                    DescriptorWriteInner::CombinedImageSampler(sampler, view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                                                   sampler: sampler,
                                                   imageView: view,
                                                   imageLayout: layout,
                                               });
                    },
                    DescriptorWriteInner::StorageImage(view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                                                   sampler: 0,
                                                   imageView: view,
                                                   imageLayout: layout,
                                               });
                    },
                    DescriptorWriteInner::SampledImage(view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                                                   sampler: 0,
                                                   imageView: view,
                                                   imageLayout: layout,
                                               });
                    },
                    DescriptorWriteInner::InputAttachment(view, layout) => {
                        image_descriptors.push(vk::DescriptorImageInfo {
                                                   sampler: 0,
                                                   imageView: view,
                                                   imageLayout: layout,
                                               });
                    },
                    DescriptorWriteInner::UniformTexelBuffer(view) |
                    DescriptorWriteInner::StorageTexelBuffer(view) => {
                        buffer_views_descriptors.push(view);
                    },
                }
            }
        }

        // Now that `image_descriptors`, `buffer_descriptors` and `buffer_views_descriptors` are
        // entirely filled and will never move again, we can fill the pointers in `raw_writes`.
        for (i, write) in raw_writes.iter_mut().enumerate() {
            write.pImageInfo = match raw_writes_img_infos[i] {
                Some(off) => image_descriptors.as_ptr().offset(off as isize),
                None => ptr::null(),
            };

            write.pBufferInfo = match raw_writes_buf_infos[i] {
                Some(off) => buffer_descriptors.as_ptr().offset(off as isize),
                None => ptr::null(),
            };

            write.pTexelBufferView = match raw_writes_buf_view_infos[i] {
                Some(off) => buffer_views_descriptors.as_ptr().offset(off as isize),
                None => ptr::null(),
            };
        }

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if !raw_writes.is_empty() {
            vk.UpdateDescriptorSets(device.internal_object(),
                                    raw_writes.len() as u32,
                                    raw_writes.as_ptr(),
                                    0,
                                    ptr::null());
        }
    }
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Object = vk::DescriptorSet;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_DESCRIPTOR_SET;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl fmt::Debug for UnsafeDescriptorSet {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan descriptor set {:?}>", self.set)
    }
}

/// Represents a single write entry to a descriptor set.
///
/// Use the various constructors to build a `DescriptorWrite`. While it is safe to build a
/// `DescriptorWrite`, it is unsafe to actually use it to write to a descriptor set.
// TODO: allow binding whole arrays at once
pub struct DescriptorWrite {
    binding: u32,
    first_array_element: u32,
    inner: SmallVec<[DescriptorWriteInner; 1]>,
}

#[derive(Debug, Clone)]
enum DescriptorWriteInner {
    Sampler(vk::Sampler),
    StorageImage(vk::ImageView, vk::ImageLayout),
    SampledImage(vk::ImageView, vk::ImageLayout),
    CombinedImageSampler(vk::Sampler, vk::ImageView, vk::ImageLayout),
    UniformTexelBuffer(vk::BufferView),
    StorageTexelBuffer(vk::BufferView),
    UniformBuffer(vk::Buffer, usize, usize),
    StorageBuffer(vk::Buffer, usize, usize),
    DynamicUniformBuffer(vk::Buffer, usize, usize),
    DynamicStorageBuffer(vk::Buffer, usize, usize),
    InputAttachment(vk::ImageView, vk::ImageLayout),
}

macro_rules! smallvec {
    ($elem:expr) => ({ let mut s = SmallVec::new(); s.push($elem); s });
}

impl DescriptorWrite {
    #[inline]
    pub fn storage_image<I>(binding: u32, array_element: u32, image: &I) -> DescriptorWrite
        where I: ImageViewAccess
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!({
                                 let layout = image.descriptor_set_storage_image_layout() as u32;
                                 DescriptorWriteInner::StorageImage(image.inner().internal_object(),
                                                                    layout)
                             }),
        }
    }

    #[inline]
    pub fn sampler(binding: u32, array_element: u32, sampler: &Arc<Sampler>) -> DescriptorWrite {
        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::Sampler(sampler.internal_object())),
        }
    }

    #[inline]
    pub fn sampled_image<I>(binding: u32, array_element: u32, image: &I) -> DescriptorWrite
        where I: ImageViewAccess
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!({
                                 let layout = image.descriptor_set_sampled_image_layout() as u32;
                                 DescriptorWriteInner::SampledImage(image.inner().internal_object(),
                                                                    layout)
                             }),
        }
    }

    #[inline]
    pub fn combined_image_sampler<I>(binding: u32, array_element: u32, sampler: &Arc<Sampler>,
                                     image: &I)
                                     -> DescriptorWrite
        where I: ImageViewAccess
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!({
                                 let layout =
                                     image.descriptor_set_combined_image_sampler_layout() as u32;
                                 DescriptorWriteInner::CombinedImageSampler(sampler
                                                                                .internal_object(),
                                                                            image
                                                                                .inner()
                                                                                .internal_object(),
                                                                            layout)
                             }),
        }
    }

    #[inline]
    pub fn uniform_texel_buffer<'a, F, B>(binding: u32, array_element: u32, view: &BufferView<F, B>)
                                          -> DescriptorWrite
        where B: BufferAccess
    {
        assert!(view.uniform_texel_buffer());

        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::UniformTexelBuffer(view.internal_object())),
        }
    }

    #[inline]
    pub fn storage_texel_buffer<'a, F, B>(binding: u32, array_element: u32, view: &BufferView<F, B>)
                                          -> DescriptorWrite
        where B: BufferAccess
    {
        assert!(view.storage_texel_buffer());

        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::StorageTexelBuffer(view.internal_object())),
        }
    }

    #[inline]
    pub unsafe fn uniform_buffer<B>(binding: u32, array_element: u32, buffer: &B) -> DescriptorWrite
        where B: BufferAccess
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(offset %
                             buffer
                                 .device()
                                 .physical_device()
                                 .limits()
                                 .min_uniform_buffer_offset_alignment() as
                                 usize,
                         0);
        debug_assert!(size <=
                          buffer
                              .device()
                              .physical_device()
                              .limits()
                              .max_uniform_buffer_range() as usize);

        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!({
                                 DescriptorWriteInner::UniformBuffer(buffer.internal_object(),
                                                                     offset,
                                                                     size)
                             }),
        }
    }

    #[inline]
    pub unsafe fn storage_buffer<B>(binding: u32, array_element: u32, buffer: &B) -> DescriptorWrite
        where B: BufferAccess
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(offset %
                             buffer
                                 .device()
                                 .physical_device()
                                 .limits()
                                 .min_storage_buffer_offset_alignment() as
                                 usize,
                         0);
        debug_assert!(size <=
                          buffer
                              .device()
                              .physical_device()
                              .limits()
                              .max_storage_buffer_range() as usize);

        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!({
                                 DescriptorWriteInner::StorageBuffer(buffer.internal_object(),
                                                                     offset,
                                                                     size)
                             }),
        }
    }

    #[inline]
    pub unsafe fn dynamic_uniform_buffer<B>(binding: u32, array_element: u32, buffer: &B)
                                            -> DescriptorWrite
        where B: BufferAccess
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(offset %
                             buffer
                                 .device()
                                 .physical_device()
                                 .limits()
                                 .min_uniform_buffer_offset_alignment() as
                                 usize,
                         0);
        debug_assert!(size <=
                          buffer
                              .device()
                              .physical_device()
                              .limits()
                              .max_uniform_buffer_range() as usize);

        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::DynamicUniformBuffer(buffer.internal_object(),
                                                                        offset,
                                                                        size)),
        }
    }

    #[inline]
    pub unsafe fn dynamic_storage_buffer<B>(binding: u32, array_element: u32, buffer: &B)
                                            -> DescriptorWrite
        where B: BufferAccess
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        debug_assert_eq!(offset %
                             buffer
                                 .device()
                                 .physical_device()
                                 .limits()
                                 .min_storage_buffer_offset_alignment() as
                                 usize,
                         0);
        debug_assert!(size <=
                          buffer
                              .device()
                              .physical_device()
                              .limits()
                              .max_storage_buffer_range() as usize);

        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!(DescriptorWriteInner::DynamicStorageBuffer(buffer.internal_object(),
                                                                        offset,
                                                                        size)),
        }
    }

    #[inline]
    pub fn input_attachment<I>(binding: u32, array_element: u32, image: &I) -> DescriptorWrite
        where I: ImageViewAccess
    {
        DescriptorWrite {
            binding: binding,
            first_array_element: array_element,
            inner: smallvec!({
                                 let layout = image.descriptor_set_input_attachment_layout() as u32;
                                 DescriptorWriteInner::InputAttachment(image
                                                                           .inner()
                                                                           .internal_object(),
                                                                       layout)
                             }),
        }
    }

    /// Returns the type corresponding to this write.
    #[inline]
    pub fn ty(&self) -> DescriptorType {
        match self.inner[0] {
            DescriptorWriteInner::Sampler(_) => DescriptorType::Sampler,
            DescriptorWriteInner::CombinedImageSampler(_, _, _) =>
                DescriptorType::CombinedImageSampler,
            DescriptorWriteInner::SampledImage(_, _) => DescriptorType::SampledImage,
            DescriptorWriteInner::StorageImage(_, _) => DescriptorType::StorageImage,
            DescriptorWriteInner::UniformTexelBuffer(_) => DescriptorType::UniformTexelBuffer,
            DescriptorWriteInner::StorageTexelBuffer(_) => DescriptorType::StorageTexelBuffer,
            DescriptorWriteInner::UniformBuffer(_, _, _) => DescriptorType::UniformBuffer,
            DescriptorWriteInner::StorageBuffer(_, _, _) => DescriptorType::StorageBuffer,
            DescriptorWriteInner::DynamicUniformBuffer(_, _, _) =>
                DescriptorType::UniformBufferDynamic,
            DescriptorWriteInner::DynamicStorageBuffer(_, _, _) =>
                DescriptorType::StorageBufferDynamic,
            DescriptorWriteInner::InputAttachment(_, _) => DescriptorType::InputAttachment,
        }
    }
}

#[cfg(test)]
mod tests {
    use descriptor::descriptor::DescriptorBufferDesc;
    use descriptor::descriptor::DescriptorDesc;
    use descriptor::descriptor::DescriptorDescTy;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::DescriptorsCount;
    use descriptor::descriptor_set::UnsafeDescriptorPool;
    use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    use std::iter;

    #[test]
    fn pool_create() {
        let (device, _) = gfx_dev_and_queue!();
        let desc = DescriptorsCount {
            uniform_buffer: 1,
            ..DescriptorsCount::zero()
        };

        let _ = UnsafeDescriptorPool::new(device, &desc, 10, false).unwrap();
    }

    #[test]
    fn zero_max_set() {
        let (device, _) = gfx_dev_and_queue!();
        let desc = DescriptorsCount {
            uniform_buffer: 1,
            ..DescriptorsCount::zero()
        };

        assert_should_panic!("The maximum number of sets can't be 0", {
            let _ = UnsafeDescriptorPool::new(device, &desc, 0, false);
        });
    }

    #[test]
    fn zero_descriptors() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!("All the descriptors count of a pool are 0", {
            let _ = UnsafeDescriptorPool::new(device, &DescriptorsCount::zero(), 10, false);
        });
    }

    #[test]
    fn basic_alloc() {
        let (device, _) = gfx_dev_and_queue!();

        let layout = DescriptorDesc {
            ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                                             dynamic: Some(false),
                                             storage: false,
                                         }),
            array_count: 1,
            stages: ShaderStages::all_graphics(),
            readonly: true,
        };

        let set_layout = UnsafeDescriptorSetLayout::new(device.clone(), iter::once(Some(layout)))
            .unwrap();

        let desc = DescriptorsCount {
            uniform_buffer: 10,
            ..DescriptorsCount::zero()
        };

        let mut pool = UnsafeDescriptorPool::new(device, &desc, 10, false).unwrap();
        unsafe {
            let sets = pool.alloc(iter::once(&set_layout)).unwrap();
            assert_eq!(sets.count(), 1);
        }
    }

    #[test]
    fn alloc_diff_device() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let layout = DescriptorDesc {
            ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                                             dynamic: Some(false),
                                             storage: false,
                                         }),
            array_count: 1,
            stages: ShaderStages::all_graphics(),
            readonly: true,
        };

        let set_layout = UnsafeDescriptorSetLayout::new(device1, iter::once(Some(layout))).unwrap();

        let desc = DescriptorsCount {
            uniform_buffer: 10,
            ..DescriptorsCount::zero()
        };

        assert_should_panic!("Tried to allocate from a pool with a set layout \
                              of a different device",
                             {
                                 let mut pool =
                                     UnsafeDescriptorPool::new(device2, &desc, 10, false).unwrap();

                                 unsafe {
                                     let _ = pool.alloc(iter::once(&set_layout));
                                 }
                             });
    }

    #[test]
    fn alloc_zero() {
        let (device, _) = gfx_dev_and_queue!();

        let desc = DescriptorsCount {
            uniform_buffer: 1,
            ..DescriptorsCount::zero()
        };

        let mut pool = UnsafeDescriptorPool::new(device, &desc, 1, false).unwrap();
        unsafe {
            let sets = pool.alloc(iter::empty()).unwrap();
            assert_eq!(sets.count(), 0);
        }
    }
}
