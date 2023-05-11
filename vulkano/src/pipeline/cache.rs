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
//! on the disk, and [`with_data`](crate::pipeline::cache::PipelineCache::with_data) for how to reload it.

use crate::{
    device::{Device, DeviceOwned},
    OomError, RuntimeError, VulkanObject,
};
use std::{mem::MaybeUninit, ptr, sync::Arc};

/// Opaque cache that contains pipeline objects.
///
/// See [the documentation of the module](crate::pipeline::cache) for more info.
#[derive(Debug)]
pub struct PipelineCache {
    device: Arc<Device>,
    handle: ash::vk::PipelineCache,
}

impl PipelineCache {
    /// Builds a new pipeline cache from existing data. The data must have been previously obtained
    /// with [`get_data`](#method.get_data).
    ///
    /// The data passed to this function will most likely be blindly trusted by the Vulkan
    /// implementation. Therefore you can easily crash your application or the system by passing
    /// wrong data. Hence why this function is unsafe.
    ///
    /// # Examples
    ///
    /// This example loads a cache from a file, if it exists.
    /// See [`get_data`](#method.get_data) for how to store the data in a file.
    /// TODO: there's a header in the cached data that must be checked ; talk about this
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use vulkano::device::Device;
    /// use std::fs::File;
    /// use std::io::Read;
    /// use vulkano::pipeline::cache::PipelineCache;
    /// # let device: Arc<Device> = return;
    ///
    /// let data = {
    ///     let file = File::open("pipeline_cache.bin");
    ///     if let Ok(mut file) = file {
    ///         let mut data = Vec::new();
    ///         if let Ok(_) = file.read_to_end(&mut data) {
    ///             Some(data)
    ///         } else {
    ///             None
    ///         }
    ///     } else {
    ///         None
    ///     }
    /// };
    ///
    /// let cache = if let Some(data) = data {
    ///     // This is unsafe because there is no way to be sure that the file contains valid data.
    ///     unsafe { PipelineCache::with_data(device.clone(), &data).unwrap() }
    /// } else {
    ///     PipelineCache::empty(device.clone()).unwrap()
    /// };
    /// ```
    #[inline]
    pub unsafe fn with_data(
        device: Arc<Device>,
        initial_data: &[u8],
    ) -> Result<Arc<PipelineCache>, OomError> {
        PipelineCache::new_impl(device, Some(initial_data))
    }

    /// Builds a new empty pipeline cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use vulkano::device::Device;
    /// use vulkano::pipeline::cache::PipelineCache;
    /// # let device: Arc<Device> = return;
    /// let cache = PipelineCache::empty(device.clone()).unwrap();
    /// ```
    #[inline]
    pub fn empty(device: Arc<Device>) -> Result<Arc<PipelineCache>, OomError> {
        unsafe { PipelineCache::new_impl(device, None) }
    }

    // Actual implementation of the constructor.
    unsafe fn new_impl(
        device: Arc<Device>,
        initial_data: Option<&[u8]>,
    ) -> Result<Arc<PipelineCache>, OomError> {
        let fns = device.fns();

        let cache = {
            let infos = ash::vk::PipelineCacheCreateInfo {
                flags: ash::vk::PipelineCacheCreateFlags::empty(),
                initial_data_size: initial_data.map(|d| d.len()).unwrap_or(0),
                p_initial_data: initial_data
                    .map(|d| d.as_ptr() as *const _)
                    .unwrap_or(ptr::null()),
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_pipeline_cache)(
                device.handle(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(PipelineCache {
            device: device.clone(),
            handle: cache,
        }))
    }

    /// Merges other pipeline caches into this one.
    ///
    /// It is `self` that is modified here. The pipeline caches passed as parameter are untouched.
    ///
    /// # Panics
    ///
    /// - Panics if `self` is included in the list of other pipelines.
    ///
    // FIXME: vkMergePipelineCaches is not thread safe for the destination cache
    // TODO: write example
    pub fn merge<'a>(
        &self,
        pipelines: impl IntoIterator<Item = &'a &'a Arc<PipelineCache>>,
    ) -> Result<(), OomError> {
        unsafe {
            let fns = self.device.fns();

            let pipelines = pipelines
                .into_iter()
                .map(|pipeline| {
                    assert!(&***pipeline as *const _ != self as *const _);
                    pipeline.handle
                })
                .collect::<Vec<_>>();

            (fns.v1_0.merge_pipeline_caches)(
                self.device.handle(),
                self.handle,
                pipelines.len() as u32,
                pipelines.as_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;

            Ok(())
        }
    }

    /// Obtains the data from the cache.
    ///
    /// This data can be stored and then reloaded and passed to `PipelineCache::with_data`.
    ///
    /// # Examples
    ///
    /// This example stores the data of a pipeline cache on the disk.
    /// See [`with_data`](#method.with_data) for how to reload it.
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
    pub fn get_data(&self) -> Result<Vec<u8>, OomError> {
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
                .map_err(RuntimeError::from)?;

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
                    err => return Err(RuntimeError::from(err).into()),
                }
            }
        };

        Ok(data)
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

#[cfg(test)]
mod tests {
    use crate::{
        pipeline::{
            cache::PipelineCache, compute::ComputePipelineCreateInfo,
            layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, PipelineLayout,
        },
        shader::{PipelineShaderStageCreateInfo, ShaderModule},
    };

    #[test]
    fn merge_self_forbidden() {
        let (device, _queue) = gfx_dev_and_queue!();
        let pipeline = PipelineCache::empty(device).unwrap();
        assert_should_panic!({
            pipeline.merge(&[&pipeline]).unwrap();
        });
    }

    #[test]
    fn cache_returns_same_data() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::empty(device.clone()).unwrap();

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
            let module = ShaderModule::from_words(device.clone(), &MODULE).unwrap();
            module.entry_point("main").unwrap()
        };

        let _pipeline = {
            let stage = PipelineShaderStageCreateInfo::entry_point(cs);
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

        let cache = PipelineCache::empty(device.clone()).unwrap();

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
                let module = ShaderModule::from_words(device.clone(), &MODULE).unwrap();
                module.entry_point("main").unwrap()
            };

            let stage = PipelineShaderStageCreateInfo::entry_point(cs);
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
                let module = ShaderModule::from_words(device.clone(), &MODULE).unwrap();
                module.entry_point("main").unwrap()
            };

            let stage = PipelineShaderStageCreateInfo::entry_point(cs);
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

        let cache = PipelineCache::empty(device.clone()).unwrap();

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
            let module = ShaderModule::from_words(device.clone(), &MODULE).unwrap();
            module.entry_point("main").unwrap()
        };

        let _first_pipeline = {
            let stage = PipelineShaderStageCreateInfo::entry_point(cs.clone());
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
            let stage = PipelineShaderStageCreateInfo::entry_point(cs);
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
