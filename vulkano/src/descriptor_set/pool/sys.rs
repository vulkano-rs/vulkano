// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::descriptor_set::pool::DescriptorsCount;
use crate::descriptor_set::UnsafeDescriptorSet;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

/// Pool from which descriptor sets are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
pub struct UnsafeDescriptorPool {
    pool: ash::vk::DescriptorPool,
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
    pub fn new(
        device: Arc<Device>,
        count: &DescriptorsCount,
        max_sets: u32,
        free_descriptor_set_bit: bool,
    ) -> Result<UnsafeDescriptorPool, OomError> {
        let fns = device.fns();

        assert_ne!(max_sets, 0, "The maximum number of sets can't be 0");

        let mut pool_sizes: SmallVec<[_; 10]> = SmallVec::new();

        macro_rules! elem {
            ($field:ident, $ty:expr) => {
                if count.$field >= 1 {
                    pool_sizes.push(ash::vk::DescriptorPoolSize {
                        ty: $ty,
                        descriptor_count: count.$field,
                    });
                }
            };
        }

        elem!(uniform_buffer, ash::vk::DescriptorType::UNIFORM_BUFFER);
        elem!(storage_buffer, ash::vk::DescriptorType::STORAGE_BUFFER);
        elem!(
            uniform_buffer_dynamic,
            ash::vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        );
        elem!(
            storage_buffer_dynamic,
            ash::vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        );
        elem!(
            uniform_texel_buffer,
            ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER
        );
        elem!(
            storage_texel_buffer,
            ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER
        );
        elem!(sampled_image, ash::vk::DescriptorType::SAMPLED_IMAGE);
        elem!(storage_image, ash::vk::DescriptorType::STORAGE_IMAGE);
        elem!(sampler, ash::vk::DescriptorType::SAMPLER);
        elem!(
            combined_image_sampler,
            ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        );
        elem!(input_attachment, ash::vk::DescriptorType::INPUT_ATTACHMENT);

        assert!(
            !pool_sizes.is_empty(),
            "All the descriptors count of a pool are 0"
        );

        let pool = unsafe {
            let infos = ash::vk::DescriptorPoolCreateInfo {
                flags: if free_descriptor_set_bit {
                    ash::vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                } else {
                    ash::vk::DescriptorPoolCreateFlags::empty()
                },
                max_sets: max_sets,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_descriptor_pool(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(UnsafeDescriptorPool {
            pool,
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
    pub unsafe fn alloc<'l, I>(
        &mut self,
        layouts: I,
    ) -> Result<UnsafeDescriptorPoolAllocIter, DescriptorPoolAllocError>
    where
        I: IntoIterator<Item = &'l DescriptorSetLayout>,
    {
        let layouts: SmallVec<[_; 8]> = layouts
            .into_iter()
            .map(|l| {
                assert_eq!(
                    self.device.internal_object(),
                    l.device().internal_object(),
                    "Tried to allocate from a pool with a set layout of a different \
                                 device"
                );
                l.internal_object()
            })
            .collect();

        self.alloc_impl(&layouts)
    }

    // Actual implementation of `alloc`. Separated so that it is not inlined.
    unsafe fn alloc_impl(
        &mut self,
        layouts: &SmallVec<[ash::vk::DescriptorSetLayout; 8]>,
    ) -> Result<UnsafeDescriptorPoolAllocIter, DescriptorPoolAllocError> {
        let num = layouts.len();

        if num == 0 {
            return Ok(UnsafeDescriptorPoolAllocIter {
                sets: vec![].into_iter(),
            });
        }

        let infos = ash::vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let mut output = Vec::with_capacity(num);

        let fns = self.device.fns();
        let ret = fns.v1_0.allocate_descriptor_sets(
            self.device.internal_object(),
            &infos,
            output.as_mut_ptr(),
        );

        // According to the specs, because `VK_ERROR_FRAGMENTED_POOL` was added after version
        // 1.0 of Vulkan, any negative return value except out-of-memory errors must be
        // considered as a fragmented pool error.
        match ret {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                return Err(DescriptorPoolAllocError::OutOfHostMemory);
            }
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                return Err(DescriptorPoolAllocError::OutOfDeviceMemory);
            }
            ash::vk::Result::ERROR_OUT_OF_POOL_MEMORY_KHR => {
                return Err(DescriptorPoolAllocError::OutOfPoolMemory);
            }
            c if c.as_raw() < 0 => {
                return Err(DescriptorPoolAllocError::FragmentedPool);
            }
            _ => (),
        };

        output.set_len(num);

        Ok(UnsafeDescriptorPoolAllocIter {
            sets: output.into_iter(),
        })
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
    where
        I: IntoIterator<Item = UnsafeDescriptorSet>,
    {
        let sets: SmallVec<[_; 8]> = descriptor_sets
            .into_iter()
            .map(|s| s.internal_object())
            .collect();
        if !sets.is_empty() {
            self.free_impl(&sets)
        } else {
            Ok(())
        }
    }

    // Actual implementation of `free`. Separated so that it is not inlined.
    unsafe fn free_impl(
        &mut self,
        sets: &SmallVec<[ash::vk::DescriptorSet; 8]>,
    ) -> Result<(), OomError> {
        let fns = self.device.fns();
        check_errors(fns.v1_0.free_descriptor_sets(
            self.device.internal_object(),
            self.pool,
            sets.len() as u32,
            sets.as_ptr(),
        ))?;
        Ok(())
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    pub unsafe fn reset(&mut self) -> Result<(), OomError> {
        let fns = self.device.fns();
        check_errors(fns.v1_0.reset_descriptor_pool(
            self.device.internal_object(),
            self.pool,
            ash::vk::DescriptorPoolResetFlags::empty(),
        ))?;
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
            let fns = self.device.fns();
            fns.v1_0
                .destroy_descriptor_pool(self.device.internal_object(), self.pool, ptr::null());
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

impl error::Error for DescriptorPoolAllocError {}

impl fmt::Display for DescriptorPoolAllocError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                DescriptorPoolAllocError::OutOfHostMemory => "no memory available on the host",
                DescriptorPoolAllocError::OutOfDeviceMemory => {
                    "no memory available on the graphical device"
                }
                DescriptorPoolAllocError::FragmentedPool => {
                    "allocation has failed because the pool is too fragmented"
                }
                DescriptorPoolAllocError::OutOfPoolMemory => {
                    "there is no more space available in the descriptor pool"
                }
            }
        )
    }
}

/// Iterator to the descriptor sets allocated from an unsafe descriptor pool.
#[derive(Debug)]
pub struct UnsafeDescriptorPoolAllocIter {
    sets: VecIntoIter<ash::vk::DescriptorSet>,
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

impl ExactSizeIterator for UnsafeDescriptorPoolAllocIter {}

#[cfg(test)]
mod tests {
    use crate::descriptor_set::layout::DescriptorBufferDesc;
    use crate::descriptor_set::layout::DescriptorDesc;
    use crate::descriptor_set::layout::DescriptorDescTy;
    use crate::descriptor_set::layout::DescriptorSetDesc;
    use crate::descriptor_set::layout::DescriptorSetLayout;
    use crate::descriptor_set::pool::DescriptorsCount;
    use crate::descriptor_set::pool::UnsafeDescriptorPool;
    use crate::pipeline::shader::ShaderStages;
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

        let set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetDesc::new(iter::once(Some(layout))),
        )
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

        let set_layout =
            DescriptorSetLayout::new(device1, DescriptorSetDesc::new(iter::once(Some(layout))))
                .unwrap();

        let desc = DescriptorsCount {
            uniform_buffer: 10,
            ..DescriptorsCount::zero()
        };

        assert_should_panic!(
            "Tried to allocate from a pool with a set layout \
                              of a different device",
            {
                let mut pool = UnsafeDescriptorPool::new(device2, &desc, 10, false).unwrap();

                unsafe {
                    let _ = pool.alloc(iter::once(&set_layout));
                }
            }
        );
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
