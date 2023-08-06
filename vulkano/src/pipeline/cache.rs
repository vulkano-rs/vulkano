// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Cache the pipeline objects to disk for faster reloads.
//!
//! A pipeline cache is an opaque type that allow you to cache your graphics and compute
//! pipelines on the disk.
//!
//! You can create either an empty cache or a cache from some initial data. Whenever you create a
//! graphics or compute pipeline, you have the possibility to pass a reference to that cache.
//! The Vulkan implementation will then look in the cache for an existing entry, or add one if it
//! doesn't exist.
//!
//! Once that is done, you can extract the data from the cache and store it. See the documentation
//! of [`get_data`](crate::pipeline::cache::PipelineCache::get_data) for example of how to store the data
//! on the disk, and [`new`](crate::pipeline::cache::PipelineCache::new) for how to reload it.

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// Opaque cache that contains pipeline objects.
///
/// See [the documentation of the module](crate::pipeline::cache) for more info.
#[derive(Debug)]
pub struct PipelineCache {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    handle: ash::vk::PipelineCache,
    id: NonZeroU64,

    flags: PipelineCacheCreateFlags,
}

impl PipelineCache {
    /// Builds a new pipeline cache.
    ///
    /// # Safety
    ///
    /// - The data in `create_info.initial_data` must be valid data that was previously retrieved
    ///   using [`get_data`](PipelineCache::get_data).
    ///
    /// # Examples
    ///
    /// This example loads a cache from a file, if it exists.
    /// See [`get_data`](#method.get_data) for how to store the data in a file.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use vulkano::device::Device;
    /// use std::fs::File;
    /// use std::io::Read;
    /// use vulkano::pipeline::cache::{PipelineCache, PipelineCacheCreateInfo};
    /// # let device: Arc<Device> = return;
    ///
    /// let initial_data = {
    ///     let file = File::open("pipeline_cache.bin");
    ///     if let Ok(mut file) = file {
    ///         let mut data = Vec::new();
    ///         if let Ok(_) = file.read_to_end(&mut data) {
    ///             data
    ///         } else {
    ///             Vec::new()
    ///         }
    ///     } else {
    ///         Vec::new()
    ///     }
    /// };
    ///
    /// // This is unsafe because there is no way to be sure that the file contains valid data.
    /// let cache = unsafe {
    ///     PipelineCache::new(
    ///         device.clone(),
    ///         PipelineCacheCreateInfo {
    ///             initial_data,
    ///             ..Default::default()
    ///         }
    ///     ).unwrap()
    /// };
    /// ```
    #[inline]
    pub unsafe fn new(
        device: Arc<Device>,
        create_info: PipelineCacheCreateInfo,
    ) -> Result<Arc<PipelineCache>, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        Ok(Self::new_unchecked(device, create_info)?)
    }

    fn validate_new(
        device: &Device,
        create_info: &PipelineCacheCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: PipelineCacheCreateInfo,
    ) -> Result<Arc<PipelineCache>, VulkanError> {
        let &PipelineCacheCreateInfo {
            flags,
            ref initial_data,
            _ne: _,
        } = &create_info;

        let infos = ash::vk::PipelineCacheCreateInfo {
            flags: flags.into(),
            initial_data_size: initial_data.len(),
            p_initial_data: if initial_data.is_empty() {
                ptr::null()
            } else {
                initial_data.as_ptr() as _
            },
            ..Default::default()
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_pipeline_cache)(
                device.handle(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Creates a new `PipelineCache` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::PipelineCache,
        create_info: PipelineCacheCreateInfo,
    ) -> Arc<PipelineCache> {
        let PipelineCacheCreateInfo {
            flags,
            initial_data: _,
            _ne: _,
        } = create_info;

        Arc::new(PipelineCache {
            device: InstanceOwnedDebugWrapper(device),
            handle,
            id: Self::next_id(),

            flags,
        })
    }

    /// Returns the flags that the pipeline cache was created with.
    pub fn flags(&self) -> PipelineCacheCreateFlags {
        self.flags
    }

    /// Obtains the data from the cache.
    ///
    /// This data can be stored and then reloaded and passed to `PipelineCache::new`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs;
    /// use std::fs::File;
    /// use std::io::Write;
    /// # use std::sync::Arc;
    /// # use vulkano::pipeline::cache::PipelineCache;
    ///
    /// # let cache: Arc<PipelineCache> = return;
    /// // If an error happens (eg. no permission for the file) we simply skip storing the cache.
    /// if let Ok(data) = cache.get_data() {
    ///     if let Ok(mut file) = File::create("pipeline_cache.bin.tmp") {
    ///         if let Ok(_) = file.write_all(&data) {
    ///             let _ = fs::rename("pipeline_cache.bin.tmp", "pipeline_cache.bin");
    ///         } else {
    ///             let _ = fs::remove_file("pipeline_cache.bin.tmp");
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn get_data(&self) -> Result<Vec<u8>, VulkanError> {
        let fns = self.device.fns();

        let data = unsafe {
            loop {
                let mut count = 0;
                (fns.v1_0.get_pipeline_cache_data)(
                    self.device.handle(),
                    self.handle,
                    &mut count,
                    ptr::null_mut(),
                )
                .result()
                .map_err(VulkanError::from)?;

                let mut data: Vec<u8> = Vec::with_capacity(count);
                let result = (fns.v1_0.get_pipeline_cache_data)(
                    self.device.handle(),
                    self.handle,
                    &mut count,
                    data.as_mut_ptr() as *mut _,
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        data.set_len(count);
                        break data;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            }
        };

        Ok(data)
    }

    /// Merges other pipeline caches into this one.
    ///
    /// It is `self` that is modified here. The pipeline caches passed as parameter are untouched.
    ///
    // FIXME: vkMergePipelineCaches is not thread safe for the destination cache
    // TODO: write example
    pub fn merge<'a>(
        &self,
        src_caches: impl IntoIterator<Item = &'a PipelineCache>,
    ) -> Result<(), Validated<VulkanError>> {
        let src_caches: SmallVec<[_; 8]> = src_caches.into_iter().collect();
        self.validate_merge(&src_caches)?;

        unsafe { Ok(self.merge_unchecked(src_caches)?) }
    }

    fn validate_merge(&self, src_caches: &[&PipelineCache]) -> Result<(), Box<ValidationError>> {
        for (index, &src_cache) in src_caches.iter().enumerate() {
            if src_cache == self {
                return Err(Box::new(ValidationError {
                    context: format!("src_caches[{}]", index).into(),
                    problem: "equals `self`".into(),
                    vuids: &["VUID-vkMergePipelineCaches-dstCache-00770"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn merge_unchecked<'a>(
        &self,
        src_caches: impl IntoIterator<Item = &'a PipelineCache>,
    ) -> Result<(), VulkanError> {
        let src_caches_vk: SmallVec<[_; 8]> =
            src_caches.into_iter().map(VulkanObject::handle).collect();

        let fns = self.device.fns();
        (fns.v1_0.merge_pipeline_caches)(
            self.device.handle(),
            self.handle,
            src_caches_vk.len() as u32,
            src_caches_vk.as_ptr(),
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }
}

impl Drop for PipelineCache {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_pipeline_cache)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for PipelineCache {
    type Handle = ash::vk::PipelineCache;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for PipelineCache {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(PipelineCache);

/// Parameters to create a new `PipelineCache`.
#[derive(Clone, Debug)]
pub struct PipelineCacheCreateInfo {
    /// Additional properties of the pipeline cache.
    ///
    /// The default value is empty.
    pub flags: PipelineCacheCreateFlags,

    /// The initial data to provide to the cache.
    ///
    /// If this is not empty, then the data must have been previously retrieved by calling
    /// [`PipelineCache::get_data`].
    ///
    /// The data passed to this function will most likely be blindly trusted by the Vulkan
    /// implementation. Therefore you can easily crash your application or the system by passing
    /// wrong data.
    ///
    /// The default value is empty.
    pub initial_data: Vec<u8>,

    pub _ne: crate::NonExhaustive,
}

impl Default for PipelineCacheCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: PipelineCacheCreateFlags::empty(),
            initial_data: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl PipelineCacheCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            initial_data: _,
            _ne,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineCacheCreateInfo-flags-parameter"])
        })?;

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a pipeline cache.
    PipelineCacheCreateFlags = PipelineCacheCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    EXTERNALLY_SYNCHRONIZED = EXTERNALLY_SYNCHRONIZED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_pipeline_creation_cache_control)]),
    ]), */
}

#[cfg(test)]
mod tests {
    use crate::{
        pipeline::{
            cache::PipelineCache, compute::ComputePipelineCreateInfo,
            layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, PipelineLayout,
            PipelineShaderStageCreateInfo,
        },
        shader::{ShaderModule, ShaderModuleCreateInfo},
    };

    #[test]
    fn merge_self_forbidden() {
        let (device, _queue) = gfx_dev_and_queue!();
        let pipeline = unsafe { PipelineCache::new(device, Default::default()).unwrap() };
        match pipeline.merge([pipeline.as_ref()]) {
            Err(_) => (),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn cache_returns_same_data() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = unsafe { PipelineCache::new(device.clone(), Default::default()).unwrap() };

        let cs = unsafe {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u32; 48] = [
                119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734, 2, 4, 0,
                3, 131320, 5, 65789, 65592,
            ];
            let module =
                ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&MODULE)).unwrap();
            module.entry_point("main").unwrap()
        };

        let _pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device,
                Some(cache.clone()),
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let cache_data = cache.get_data().unwrap();
        let second_data = cache.get_data().unwrap();

        assert_eq!(cache_data, second_data);
    }

    #[test]
    fn cache_returns_different_data() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = unsafe { PipelineCache::new(device.clone(), Default::default()).unwrap() };

        let _first_pipeline = {
            let cs = unsafe {
                /*
                 * #version 450
                 * void main() {
                 * }
                 */
                const MODULE: [u32; 48] = [
                    119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                    808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1,
                    1, 196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734,
                    2, 4, 0, 3, 131320, 5, 65789, 65592,
                ];
                let module =
                    ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&MODULE))
                        .unwrap();
                module.entry_point("main").unwrap()
            };

            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                Some(cache.clone()),
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let cache_data = cache.get_data().unwrap();

        let _second_pipeline = {
            let cs = unsafe {
                /*
                 * #version 450
                 *
                 * void main() {
                 *     uint idx = gl_GlobalInvocationID.x;
                 * }
                 */
                const MODULE: [u32; 108] = [
                    119734787, 65536, 524298, 16, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                    808793134, 0, 196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 1,
                    1, 1, 196611, 2, 450, 262149, 4, 1852399981, 0, 196613, 8, 7890025, 524293, 11,
                    1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 262215, 11, 11,
                    28, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262167, 9, 6,
                    3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6,
                    327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 327745, 13, 14, 11, 12, 262205,
                    6, 15, 14, 196670, 8, 15, 65789, 65592,
                ];
                let module =
                    ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&MODULE))
                        .unwrap();
                module.entry_point("main").unwrap()
            };

            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device,
                Some(cache.clone()),
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let second_data = cache.get_data().unwrap();

        if cache_data.is_empty() {
            assert_eq!(cache_data, second_data);
        } else {
            assert_ne!(cache_data, second_data);
        }
    }

    #[test]
    fn cache_data_does_not_change() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = unsafe { PipelineCache::new(device.clone(), Default::default()).unwrap() };

        let cs = unsafe {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u32; 48] = [
                119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734, 2, 4, 0,
                3, 131320, 5, 65789, 65592,
            ];
            let module =
                ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&MODULE)).unwrap();
            module.entry_point("main").unwrap()
        };

        let _first_pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(cs.clone());
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                Some(cache.clone()),
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let cache_data = cache.get_data().unwrap();

        let _second_pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device,
                Some(cache.clone()),
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let second_data = cache.get_data().unwrap();

        assert_eq!(cache_data, second_data);
    }
}
