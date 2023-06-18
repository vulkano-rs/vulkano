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
        layout::{DescriptorSetLayout, DescriptorSetLayoutCreateFlags, DescriptorType},
        sys::UnsafeDescriptorSet,
    },
    device::{Device, DeviceOwned},
    macros::{impl_id_counter, vulkan_bitflags},
    Requires, RequiresAllOf, RequiresOneOf, RuntimeError, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ahash::HashMap;
use smallvec::SmallVec;
use std::{cell::Cell, marker::PhantomData, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// Pool that descriptors are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
#[derive(Debug)]
pub struct DescriptorPool {
    handle: ash::vk::DescriptorPool,
    device: Arc<Device>,
    id: NonZeroU64,

    flags: DescriptorPoolCreateFlags,
    max_sets: u32,
    pool_sizes: HashMap<DescriptorType, u32>,
    max_inline_uniform_block_bindings: u32,

    // Unimplement `Sync`, as Vulkan descriptor pools are not thread safe.
    _marker: PhantomData<Cell<ash::vk::DescriptorPool>>,
}

impl DescriptorPool {
    /// Creates a new `DescriptorPool`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: DescriptorPoolCreateInfo,
    ) -> Result<DescriptorPool, VulkanError> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: DescriptorPoolCreateInfo,
    ) -> Result<DescriptorPool, RuntimeError> {
        let &DescriptorPoolCreateInfo {
            flags,
            max_sets,
            ref pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = &create_info;

        let pool_sizes_vk: SmallVec<[_; 8]> = pool_sizes
            .iter()
            .map(|(&ty, &descriptor_count)| ash::vk::DescriptorPoolSize {
                ty: ty.into(),
                descriptor_count,
            })
            .collect();

        let mut create_info_vk = ash::vk::DescriptorPoolCreateInfo {
            flags: flags.into(),
            max_sets,
            pool_size_count: pool_sizes_vk.len() as u32,
            p_pool_sizes: pool_sizes_vk.as_ptr(),
            ..Default::default()
        };

        let mut inline_uniform_block_create_info_vk = None;

        if max_inline_uniform_block_bindings != 0 {
            let next = inline_uniform_block_create_info_vk.insert(
                ash::vk::DescriptorPoolInlineUniformBlockCreateInfo {
                    max_inline_uniform_block_bindings,
                    ..Default::default()
                },
            );

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_descriptor_pool)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        unsafe { Ok(Self::from_handle(device, handle, create_info)) }
    }

    fn validate_new(
        device: &Device,
        create_info: &DescriptorPoolCreateInfo,
    ) -> Result<(), ValidationError> {
        // VUID-vkCreateDescriptorPool-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    /// Creates a new `DescriptorPool` from a raw object handle.
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
            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = create_info;

        DescriptorPool {
            handle,
            device,
            id: Self::next_id(),

            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,

            _marker: PhantomData,
        }
    }

    /// Returns the flags that the descriptor pool was created with.
    #[inline]
    pub fn flags(&self) -> DescriptorPoolCreateFlags {
        self.flags
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

    /// Returns the maximum number of [`DescriptorType::InlineUniformBlock`] bindings that can
    /// be allocated from the descriptor pool.
    #[inline]
    pub fn max_inline_uniform_block_bindings(&self) -> u32 {
        self.max_inline_uniform_block_bindings
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
    ) -> Result<impl ExactSizeIterator<Item = UnsafeDescriptorSet>, RuntimeError> {
        let (layouts, variable_descriptor_counts): (SmallVec<[_; 1]>, SmallVec<[_; 1]>) =
            allocate_info
                .into_iter()
                .map(|info| {
                    assert_eq!(self.device.handle(), info.layout.device().handle(),);
                    debug_assert!(!info
                        .layout
                        .flags()
                        .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR));
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
            (fns.v1_0.allocate_descriptor_sets)(self.device.handle(), &infos, output.as_mut_ptr())
                .result()
                .map_err(RuntimeError::from)
                .map_err(|err| match err {
                    RuntimeError::OutOfHostMemory
                    | RuntimeError::OutOfDeviceMemory
                    | RuntimeError::OutOfPoolMemory => err,
                    // According to the specs, because `VK_ERROR_FRAGMENTED_POOL` was added after
                    // version 1.0 of Vulkan, any negative return value except out-of-memory errors
                    // must be considered as a fragmented pool error.
                    _ => RuntimeError::FragmentedPool,
                })?;

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
    ) -> Result<(), RuntimeError> {
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
            .map_err(RuntimeError::from)?;
        }

        Ok(())
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    #[inline]
    pub unsafe fn reset(&self) -> Result<(), RuntimeError> {
        let fns = self.device.fns();
        (fns.v1_0.reset_descriptor_pool)(
            self.device.handle(),
            self.handle,
            ash::vk::DescriptorPoolResetFlags::empty(),
        )
        .result()
        .map_err(RuntimeError::from)?;

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
    /// Additional properties of the descriptor pool.
    ///
    /// The default value is empty.
    pub flags: DescriptorPoolCreateFlags,

    /// The maximum number of descriptor sets that can be allocated from the pool.
    ///
    /// The default value is `0`, which must be overridden.
    pub max_sets: u32,

    /// The number of descriptors of each type to allocate for the pool.
    ///
    /// If the descriptor type is [`DescriptorType::InlineUniformBlock`], then the value is the
    /// number of bytes to allocate for such descriptors. The value must then be a multiple of 4.
    ///
    /// The default value is empty, which must be overridden.
    pub pool_sizes: HashMap<DescriptorType, u32>,

    /// The maximum number of [`DescriptorType::InlineUniformBlock`] bindings that can be allocated
    /// from the descriptor pool.
    ///
    /// If this is not 0, the device API version must be at least 1.3, or the
    /// [`khr_inline_uniform_block`](crate::device::DeviceExtensions::ext_inline_uniform_block)
    /// extension must be enabled on the device.
    ///
    /// The default value is 0.
    pub max_inline_uniform_block_bindings: u32,

    pub _ne: crate::NonExhaustive,
}

impl Default for DescriptorPoolCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: DescriptorPoolCreateFlags::empty(),
            max_sets: 0,
            pool_sizes: HashMap::default(),
            max_inline_uniform_block_bindings: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl DescriptorPoolCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            max_sets,
            ref pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkDescriptorPoolCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if max_sets == 0 {
            return Err(ValidationError {
                context: "max_sets".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkDescriptorPoolCreateInfo-maxSets-00301"],
                ..Default::default()
            });
        }

        if pool_sizes.is_empty() {
            return Err(ValidationError {
                context: "pool_sizes".into(),
                problem: "is empty".into(),
                // vuids?
                ..Default::default()
            });
        }

        // VUID-VkDescriptorPoolCreateInfo-pPoolSizes-parameter
        for (&descriptor_type, &pool_size) in pool_sizes.iter() {
            flags
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "pool_sizes".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-type-parameter"],
                    ..ValidationError::from_requirement(err)
                })?;

            if pool_size == 0 {
                return Err(ValidationError {
                    context: format!("pool_sizes[DescriptorType::{:?}]", descriptor_type).into(),
                    problem: "is zero".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-descriptorCount-00302"],
                    ..Default::default()
                });
            }

            if descriptor_type == DescriptorType::InlineUniformBlock {
                return Err(ValidationError {
                    context: "pool_sizes[DescriptorType::InlineUniformBlock]".into(),
                    problem: "is not a multiple of 4".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-type-02218"],
                    ..Default::default()
                });
            }
        }

        if max_inline_uniform_block_bindings != 0
            && !(device.api_version() >= Version::V1_3
                || device.enabled_extensions().ext_inline_uniform_block)
        {
            return Err(ValidationError {
                context: "max_inline_uniform_block_bindings".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("ext_inline_uniform_block")]),
                ]),
                // vuids?
                ..Default::default()
            });
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a descriptor pool.
    DescriptorPoolCreateFlags = DescriptorPoolCreateFlags(u32);

    /// Individual descriptor sets can be freed from the pool. Otherwise you must reset or
    /// destroy the whole pool at once.
    FREE_DESCRIPTOR_SET = FREE_DESCRIPTOR_SET,

    /* TODO: enable
    // TODO: document
    UPDATE_AFTER_BIND = UPDATE_AFTER_BIND {
        api_version: V1_2,
        device_extensions: [ext_descriptor_indexing],
    }, */

    /* TODO: enable
    // TODO: document
    HOST_ONLY = HOST_ONLY_EXT {
        device_extensions: [ext_mutable_descriptor_type, valve_mutable_descriptor_type],
    }, */
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

        DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 0,
                pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap_err();
    }

    #[test]
    fn zero_descriptors() {
        let (device, _) = gfx_dev_and_queue!();

        DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 10,
                ..Default::default()
            },
        )
        .unwrap_err();
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
