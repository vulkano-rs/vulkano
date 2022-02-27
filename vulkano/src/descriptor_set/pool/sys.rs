// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    check_errors,
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorType},
        sys::UnsafeDescriptorSet,
    },
    device::{Device, DeviceOwned},
    OomError, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    collections::HashMap,
    error, fmt,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ptr,
    sync::Arc,
};

/// Pool that descriptors are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
#[derive(Debug)]
pub struct UnsafeDescriptorPool {
    handle: ash::vk::DescriptorPool,
    device: Arc<Device>,

    max_sets: u32,
    pool_sizes: HashMap<DescriptorType, u32>,
    can_free_descriptor_sets: bool,
}

impl UnsafeDescriptorPool {
    /// Creates a new `UnsafeDescriptorPool`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.max_sets` is `0`.
    /// - Panics if `create_info.pool_sizes` is empty.
    /// - Panics if `create_info.pool_sizes` contains a descriptor type with a count of `0`.
    pub fn new(
        device: Arc<Device>,
        create_info: UnsafeDescriptorPoolCreateInfo,
    ) -> Result<UnsafeDescriptorPool, OomError> {
        let UnsafeDescriptorPoolCreateInfo {
            max_sets,
            pool_sizes,
            can_free_descriptor_sets,
            _ne: _,
        } = create_info;

        // VUID-VkDescriptorPoolCreateInfo-maxSets-00301
        assert!(max_sets != 0);

        // VUID-VkDescriptorPoolCreateInfo-poolSizeCount-arraylength
        assert!(!pool_sizes.is_empty());

        let handle = {
            let pool_sizes: SmallVec<[_; 8]> = pool_sizes
                .iter()
                .map(|(&ty, &descriptor_count)| {
                    // VUID-VkDescriptorPoolSize-descriptorCount-00302
                    assert!(descriptor_count != 0);

                    ash::vk::DescriptorPoolSize {
                        ty: ty.into(),
                        descriptor_count,
                    }
                })
                .collect();

            let mut flags = ash::vk::DescriptorPoolCreateFlags::empty();

            if can_free_descriptor_sets {
                flags |= ash::vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
            }

            let create_info = ash::vk::DescriptorPoolCreateInfo {
                flags,
                max_sets,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };

            unsafe {
                let fns = device.fns();
                let mut output = MaybeUninit::uninit();
                check_errors(fns.v1_0.create_descriptor_pool(
                    device.internal_object(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            }
        };

        Ok(UnsafeDescriptorPool {
            handle,
            device,

            max_sets,
            pool_sizes,
            can_free_descriptor_sets,
        })
    }

    /// Returns the maximum number of sets that can be allocated from the pool.
    #[inline]
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }

    /// Returns the number of descriptors of each type that the pool was created with.
    #[inline]
    pub fn pool_sizes(&self) -> &HashMap<DescriptorType, u32> {
        &self.pool_sizes
    }

    /// Returns whether the descriptor sets allocated from the pool can be individually freed.
    #[inline]
    pub fn can_free_descriptor_sets(&self) -> bool {
        self.can_free_descriptor_sets
    }

    /// Allocates descriptor sets from the pool, one for each element in `create_info`.
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
    pub unsafe fn allocate_descriptor_sets<'a>(
        &mut self,
        allocate_info: impl IntoIterator<Item = DescriptorSetAllocateInfo<'a>>,
    ) -> Result<impl ExactSizeIterator<Item = UnsafeDescriptorSet>, DescriptorPoolAllocError> {
        let (layouts, variable_descriptor_counts): (SmallVec<[_; 1]>, SmallVec<[_; 1]>) =
            allocate_info
                .into_iter()
                .map(|info| {
                    assert_eq!(
                        self.device.internal_object(),
                        info.layout.device().internal_object(),
                    );
                    debug_assert!(!info.layout.push_descriptor());
                    debug_assert!(
                        info.variable_descriptor_count <= info.layout.variable_descriptor_count()
                    );

                    (
                        info.layout.internal_object(),
                        info.variable_descriptor_count,
                    )
                })
                .unzip();

        let output = if layouts.len() == 0 {
            vec![]
        } else {
            let variable_desc_count_alloc_info = if (self.device.api_version() >= Version::V1_2
                || self.device.enabled_extensions().ext_descriptor_indexing)
                && variable_descriptor_counts.iter().any(|c| *c != 0)
            {
                Some(ash::vk::DescriptorSetVariableDescriptorCountAllocateInfo {
                    descriptor_set_count: layouts.len() as u32,
                    p_descriptor_counts: variable_descriptor_counts.as_ptr(),
                    ..Default::default()
                })
            } else {
                None
            };

            let infos = ash::vk::DescriptorSetAllocateInfo {
                descriptor_pool: self.handle,
                descriptor_set_count: layouts.len() as u32,
                p_set_layouts: layouts.as_ptr(),
                p_next: if let Some(next) = variable_desc_count_alloc_info.as_ref() {
                    next as *const _ as *const _
                } else {
                    ptr::null()
                },
                ..Default::default()
            };

            let mut output = Vec::with_capacity(layouts.len());

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

            output.set_len(layouts.len());
            output
        };

        Ok(output
            .into_iter()
            .map(|handle| UnsafeDescriptorSet::new(handle)))
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
    pub unsafe fn free_descriptor_sets<I>(&mut self, descriptor_sets: I) -> Result<(), OomError>
    where
        I: IntoIterator<Item = UnsafeDescriptorSet>,
    {
        let sets: SmallVec<[_; 8]> = descriptor_sets
            .into_iter()
            .map(|s| s.internal_object())
            .collect();
        if !sets.is_empty() {
            let fns = self.device.fns();
            check_errors(fns.v1_0.free_descriptor_sets(
                self.device.internal_object(),
                self.handle,
                sets.len() as u32,
                sets.as_ptr(),
            ))?;
        }

        Ok(())
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    pub unsafe fn reset(&mut self) -> Result<(), OomError> {
        let fns = self.device.fns();
        check_errors(fns.v1_0.reset_descriptor_pool(
            self.device.internal_object(),
            self.handle,
            ash::vk::DescriptorPoolResetFlags::empty(),
        ))?;
        Ok(())
    }
}

impl Drop for UnsafeDescriptorPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0.destroy_descriptor_pool(
                self.device.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl VulkanObject for UnsafeDescriptorPool {
    type Object = ash::vk::DescriptorPool;

    #[inline]
    fn internal_object(&self) -> Self::Object {
        self.handle
    }
}

unsafe impl DeviceOwned for UnsafeDescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for UnsafeDescriptorPool {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for UnsafeDescriptorPool {}

impl Hash for UnsafeDescriptorPool {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Parameters to create a new `UnsafeDescriptorPool`.
#[derive(Clone, Debug)]
pub struct UnsafeDescriptorPoolCreateInfo {
    /// The maximum number of descriptor sets that can be allocated from the pool.
    ///
    /// The default value is `0`, which must be overridden.
    pub max_sets: u32,

    /// The number of descriptors of each type to allocate for the pool.
    ///
    /// The default value is empty, which must be overridden.
    pub pool_sizes: HashMap<DescriptorType, u32>,

    /// Whether individual descriptor sets can be freed from the pool. Otherwise you must reset or
    /// destroy the whole pool at once.
    ///
    /// The default value is `false`.
    pub can_free_descriptor_sets: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for UnsafeDescriptorPoolCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            max_sets: 0,
            pool_sizes: HashMap::default(),
            can_free_descriptor_sets: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters to allocate a new `UnsafeDescriptorSet` from an `UnsafeDescriptorPool`.
#[derive(Clone, Debug)]
pub struct DescriptorSetAllocateInfo<'a> {
    /// The descriptor set layout to create the set for.
    pub layout: &'a DescriptorSetLayout,

    /// For layouts with a variable-count binding, the number of descriptors to allocate for that
    /// binding. This should be 0 for layouts that don't have a variable-count binding.
    pub variable_descriptor_count: u32,
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

#[cfg(test)]
mod tests {
    use super::{UnsafeDescriptorPool, UnsafeDescriptorPoolCreateInfo};
    use crate::{
        descriptor_set::{
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            pool::DescriptorSetAllocateInfo,
        },
        shader::ShaderStages,
    };

    #[test]
    fn pool_create() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = UnsafeDescriptorPool::new(
            device,
            UnsafeDescriptorPoolCreateInfo {
                max_sets: 10,
                pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_max_set() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = UnsafeDescriptorPool::new(
                device,
                UnsafeDescriptorPoolCreateInfo {
                    max_sets: 0,
                    pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                    ..Default::default()
                },
            );
        });
    }

    #[test]
    fn zero_descriptors() {
        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let _ = UnsafeDescriptorPool::new(
                device,
                UnsafeDescriptorPoolCreateInfo {
                    max_sets: 10,
                    ..Default::default()
                },
            );
        });
    }

    #[test]
    fn basic_alloc() {
        let (device, _) = gfx_dev_and_queue!();

        let set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::all_graphics(),
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        let mut pool = UnsafeDescriptorPool::new(
            device,
            UnsafeDescriptorPoolCreateInfo {
                max_sets: 10,
                pool_sizes: [(DescriptorType::UniformBuffer, 10)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap();
        unsafe {
            let sets = pool
                .allocate_descriptor_sets([DescriptorSetAllocateInfo {
                    layout: set_layout.as_ref(),
                    variable_descriptor_count: 0,
                }])
                .unwrap();
            assert_eq!(sets.count(), 1);
        }
    }

    #[test]
    fn alloc_diff_device() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set_layout = DescriptorSetLayout::new(
            device1,
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::all_graphics(),
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        assert_should_panic!({
            let mut pool = UnsafeDescriptorPool::new(
                device2,
                UnsafeDescriptorPoolCreateInfo {
                    max_sets: 10,
                    pool_sizes: [(DescriptorType::UniformBuffer, 10)].into_iter().collect(),
                    ..Default::default()
                },
            )
            .unwrap();

            unsafe {
                let _ = pool.allocate_descriptor_sets([DescriptorSetAllocateInfo {
                    layout: set_layout.as_ref(),
                    variable_descriptor_count: 0,
                }]);
            }
        });
    }

    #[test]
    fn alloc_zero() {
        let (device, _) = gfx_dev_and_queue!();

        let mut pool = UnsafeDescriptorPool::new(
            device,
            UnsafeDescriptorPoolCreateInfo {
                max_sets: 1,
                pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap();
        unsafe {
            let sets = pool.allocate_descriptor_sets([]).unwrap();
            assert_eq!(sets.count(), 0);
        }
    }
}
