// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorType},
        sys::UnsafeDescriptorSet,
    },
    device::{Device, DeviceOwned},
    macros::impl_id_counter,
    OomError, Version, VulkanError, VulkanObject,
};
use ahash::HashMap;
use smallvec::SmallVec;
use std::{
    cell::Cell,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

/// Pool that descriptors are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
#[derive(Debug)]
pub struct DescriptorPool {
    handle: ash::vk::DescriptorPool,
    device: Arc<Device>,
    id: NonZeroU64,

    max_sets: u32,
    pool_sizes: HashMap<DescriptorType, u32>,
    can_free_descriptor_sets: bool,
    // Unimplement `Sync`, as Vulkan descriptor pools are not thread safe.
    _marker: PhantomData<Cell<ash::vk::DescriptorPool>>,
}

impl DescriptorPool {
    /// Creates a new `UnsafeDescriptorPool`.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.max_sets` is `0`.
    /// - Panics if `create_info.pool_sizes` is empty.
    /// - Panics if `create_info.pool_sizes` contains a descriptor type with a count of `0`.
    pub fn new(
        device: Arc<Device>,
        create_info: DescriptorPoolCreateInfo,
    ) -> Result<DescriptorPool, OomError> {
        let DescriptorPoolCreateInfo {
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
                (fns.v1_0.create_descriptor_pool)(
                    device.handle(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;
                output.assume_init()
            }
        };

        Ok(DescriptorPool {
            handle,
            device,
            id: Self::next_id(),
            max_sets,
            pool_sizes,
            can_free_descriptor_sets,
            _marker: PhantomData,
        })
    }

    /// Creates a new `UnsafeDescriptorPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::DescriptorPool,
        create_info: DescriptorPoolCreateInfo,
    ) -> DescriptorPool {
        let DescriptorPoolCreateInfo {
            max_sets,
            pool_sizes,
            can_free_descriptor_sets,
            _ne: _,
        } = create_info;

        DescriptorPool {
            handle,
            device,
            id: Self::next_id(),
            max_sets,
            pool_sizes,
            can_free_descriptor_sets,
            _marker: PhantomData,
        }
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
    /// # Panics
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
    pub unsafe fn allocate_descriptor_sets<'a>(
        &self,
        allocate_info: impl IntoIterator<Item = DescriptorSetAllocateInfo<'a>>,
    ) -> Result<impl ExactSizeIterator<Item = UnsafeDescriptorSet>, DescriptorPoolAllocError> {
        let (layouts, variable_descriptor_counts): (SmallVec<[_; 1]>, SmallVec<[_; 1]>) =
            allocate_info
                .into_iter()
                .map(|info| {
                    assert_eq!(self.device.handle(), info.layout.device().handle(),);
                    debug_assert!(!info.layout.push_descriptor());
                    debug_assert!(
                        info.variable_descriptor_count <= info.layout.variable_descriptor_count()
                    );

                    (info.layout.handle(), info.variable_descriptor_count)
                })
                .unzip();

        let output = if layouts.is_empty() {
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
            let ret = (fns.v1_0.allocate_descriptor_sets)(
                self.device.handle(),
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

        Ok(output.into_iter().map(UnsafeDescriptorSet::new))
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
    pub unsafe fn free_descriptor_sets(
        &self,
        descriptor_sets: impl IntoIterator<Item = UnsafeDescriptorSet>,
    ) -> Result<(), OomError> {
        let sets: SmallVec<[_; 8]> = descriptor_sets.into_iter().map(|s| s.handle()).collect();
        if !sets.is_empty() {
            let fns = self.device.fns();
            (fns.v1_0.free_descriptor_sets)(
                self.device.handle(),
                self.handle,
                sets.len() as u32,
                sets.as_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    #[inline]
    pub unsafe fn reset(&self) -> Result<(), OomError> {
        let fns = self.device.fns();
        (fns.v1_0.reset_descriptor_pool)(
            self.device.handle(),
            self.handle,
            ash::vk::DescriptorPoolResetFlags::empty(),
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }
}

impl Drop for DescriptorPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_descriptor_pool)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for DescriptorPool {
    type Handle = ash::vk::DescriptorPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(DescriptorPool);

/// Parameters to create a new `UnsafeDescriptorPool`.
#[derive(Clone, Debug)]
pub struct DescriptorPoolCreateInfo {
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

impl Default for DescriptorPoolCreateInfo {
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

impl Error for DescriptorPoolAllocError {}

impl Display for DescriptorPoolAllocError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
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
    use super::{DescriptorPool, DescriptorPoolCreateInfo};
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

        let _ = DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
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
            let _ = DescriptorPool::new(
                device,
                DescriptorPoolCreateInfo {
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
            let _ = DescriptorPool::new(
                device,
                DescriptorPoolCreateInfo {
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

        let pool = DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
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
            let pool = DescriptorPool::new(
                device2,
                DescriptorPoolCreateInfo {
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

        let pool = DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
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
