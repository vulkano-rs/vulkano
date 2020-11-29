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
//! of [`get_data`](struct.PipelineCache.html#method.get_data) for example of how to store the data
//! on the disk, and [`with_data`](struct.PipelineCache.html#method.with_data) for how to reload it.

use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

use device::Device;

use check_errors;
use vk;
use OomError;
use VulkanObject;

/// Opaque cache that contains pipeline objects.
///
/// See [the documentation of the module](index.html) for more info.
pub struct PipelineCache {
    device: Arc<Device>,
    cache: vk::PipelineCache,
}

impl PipelineCache {
    /// Builds a new pipeline cache from existing data. The data must have been previously obtained
    /// with [`get_data`](#method.get_data).
    ///
    /// The data passed to this function will most likely be blindly trusted by the Vulkan
    /// implementation. Therefore you can easily crash your application or the system by passing
    /// wrong data. Hence why this function is unsafe.
    ///
    /// # Example
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
    ///         } else { None }
    ///     } else { None }
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
    /// # Example
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
        let vk = device.pointers();

        let cache = {
            let infos = vk::PipelineCacheCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                initialDataSize: initial_data.map(|d| d.len()).unwrap_or(0),
                pInitialData: initial_data
                    .map(|d| d.as_ptr() as *const _)
                    .unwrap_or(ptr::null()),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(vk.CreatePipelineCache(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(PipelineCache {
            device: device.clone(),
            cache: cache,
        }))
    }

    /// Merges other pipeline caches into this one.
    ///
    /// It is `self` that is modified here. The pipeline caches passed as parameter are untouched.
    ///
    /// # Panic
    ///
    /// - Panics if `self` is included in the list of other pipelines.
    ///
    // FIXME: vkMergePipelineCaches is not thread safe for the destination cache
    // TODO: write example
    pub fn merge<'a, I>(&self, pipelines: I) -> Result<(), OomError>
    where
        I: IntoIterator<Item = &'a &'a Arc<PipelineCache>>,
    {
        unsafe {
            let vk = self.device.pointers();

            let pipelines = pipelines
                .into_iter()
                .map(|pipeline| {
                    assert!(&***pipeline as *const _ != &*self as *const _);
                    pipeline.cache
                })
                .collect::<Vec<_>>();

            check_errors(vk.MergePipelineCaches(
                self.device.internal_object(),
                self.cache,
                pipelines.len() as u32,
                pipelines.as_ptr(),
            ))?;

            Ok(())
        }
    }

    /// Obtains the data from the cache.
    ///
    /// This data can be stored and then reloaded and passed to `PipelineCache::with_data`.
    ///
    /// # Example
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
    pub fn get_data(&self) -> Result<Vec<u8>, OomError> {
        unsafe {
            let vk = self.device.pointers();

            let mut num = 0;
            check_errors(vk.GetPipelineCacheData(
                self.device.internal_object(),
                self.cache,
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut data: Vec<u8> = Vec::with_capacity(num as usize);
            check_errors(vk.GetPipelineCacheData(
                self.device.internal_object(),
                self.cache,
                &mut num,
                data.as_mut_ptr() as *mut _,
            ))?;
            data.set_len(num as usize);

            Ok(data)
        }
    }
}

unsafe impl VulkanObject for PipelineCache {
    type Object = vk::PipelineCache;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_PIPELINE_CACHE;

    #[inline]
    fn internal_object(&self) -> vk::PipelineCache {
        self.cache
    }
}

impl Drop for PipelineCache {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipelineCache(self.device.internal_object(), self.cache, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::CStr, sync::Arc};

    use descriptor::descriptor::DescriptorDesc;
    use descriptor::pipeline_layout::PipelineLayoutDesc;
    use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    use pipeline::cache::PipelineCache;
    use pipeline::shader::ShaderModule;
    use pipeline::ComputePipeline;

    #[test]
    fn merge_self_forbidden() {
        let (device, queue) = gfx_dev_and_queue!();
        let pipeline = PipelineCache::empty(device).unwrap();
        assert_should_panic!({
            pipeline.merge(&[&pipeline]).unwrap();
        });
    }

    #[test]
    fn cache_returns_same_data() {
        let (device, queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::empty(device.clone()).unwrap();

        let module = unsafe {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u8; 192] = [
                3, 2, 35, 7, 0, 0, 1, 0, 10, 0, 8, 0, 6, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0, 0,
                0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100, 46, 52, 53, 48, 0,
                0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 5, 0, 5, 0, 0, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 16, 0, 6, 0, 4, 0, 0, 0, 17, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 5, 0, 4, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2,
                0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 248, 0, 2, 0,
                5, 0, 0, 0, 253, 0, 1, 0, 56, 0, 1, 0,
            ];
            ShaderModule::new(device.clone(), &MODULE).unwrap()
        };

        let shader = unsafe {
            #[derive(Clone)]
            struct Layout;
            unsafe impl PipelineLayoutDesc for Layout {
                fn num_sets(&self) -> usize {
                    0
                }

                fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                    None
                }

                fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                    None
                }

                fn num_push_constants_ranges(&self) -> usize {
                    0
                }

                fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                    None
                }
            }
            static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // "main"
            module.compute_entry_point(CStr::from_ptr(NAME.as_ptr() as *const _), Layout)
        };

        let pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &shader, &(), Some(cache.clone())).unwrap(),
        );

        let cache_data = cache.get_data().unwrap();
        let second_data = cache.get_data().unwrap();

        assert_eq!(cache_data, second_data);
    }

    #[test]
    fn cache_returns_different_data() {
        let (device, queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::empty(device.clone()).unwrap();

        let first_module = unsafe {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u8; 192] = [
                3, 2, 35, 7, 0, 0, 1, 0, 10, 0, 8, 0, 6, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0, 0,
                0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100, 46, 52, 53, 48, 0,
                0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 5, 0, 5, 0, 0, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 16, 0, 6, 0, 4, 0, 0, 0, 17, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 5, 0, 4, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2,
                0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 248, 0, 2, 0,
                5, 0, 0, 0, 253, 0, 1, 0, 56, 0, 1, 0,
            ];
            ShaderModule::new(device.clone(), &MODULE).unwrap()
        };

        let first_shader = unsafe {
            #[derive(Clone)]
            struct Layout;
            unsafe impl PipelineLayoutDesc for Layout {
                fn num_sets(&self) -> usize {
                    0
                }

                fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                    None
                }

                fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                    None
                }

                fn num_push_constants_ranges(&self) -> usize {
                    0
                }

                fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                    None
                }
            }
            static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // "main"
            first_module.compute_entry_point(CStr::from_ptr(NAME.as_ptr() as *const _), Layout)
        };

        let second_module = unsafe {
            /*
             * #version 450
             *
             * void main() {
             *     uint idx = gl_GlobalInvocationID.x;
             * }
             */
            const SECOND_MODULE: [u8; 432] = [
                3, 2, 35, 7, 0, 0, 1, 0, 10, 0, 8, 0, 16, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0,
                0, 0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100, 46, 52, 53, 48,
                0, 0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 6, 0, 5, 0, 0, 0, 4, 0, 0,
                0, 109, 97, 105, 110, 0, 0, 0, 0, 11, 0, 0, 0, 16, 0, 6, 0, 4, 0, 0, 0, 17, 0, 0,
                0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 5, 0,
                4, 0, 4, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 5, 0, 3, 0, 8, 0, 0, 0, 105, 100,
                120, 0, 5, 0, 8, 0, 11, 0, 0, 0, 103, 108, 95, 71, 108, 111, 98, 97, 108, 73, 110,
                118, 111, 99, 97, 116, 105, 111, 110, 73, 68, 0, 0, 0, 71, 0, 4, 0, 11, 0, 0, 0,
                11, 0, 0, 0, 28, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2, 0,
                0, 0, 21, 0, 4, 0, 6, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 32, 0, 4, 0, 7, 0, 0, 0, 7,
                0, 0, 0, 6, 0, 0, 0, 23, 0, 4, 0, 9, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 32, 0, 4, 0,
                10, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 59, 0, 4, 0, 10, 0, 0, 0, 11, 0, 0, 0, 1, 0,
                0, 0, 43, 0, 4, 0, 6, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 4, 0, 13, 0, 0, 0,
                1, 0, 0, 0, 6, 0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
                0, 248, 0, 2, 0, 5, 0, 0, 0, 59, 0, 4, 0, 7, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 65,
                0, 5, 0, 13, 0, 0, 0, 14, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0, 61, 0, 4, 0, 6, 0, 0,
                0, 15, 0, 0, 0, 14, 0, 0, 0, 62, 0, 3, 0, 8, 0, 0, 0, 15, 0, 0, 0, 253, 0, 1, 0,
                56, 0, 1, 0,
            ];
            ShaderModule::new(device.clone(), &SECOND_MODULE).unwrap()
        };

        let second_shader = unsafe {
            #[derive(Clone)]
            struct Layout;
            unsafe impl PipelineLayoutDesc for Layout {
                fn num_sets(&self) -> usize {
                    0
                }

                fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                    None
                }

                fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                    None
                }

                fn num_push_constants_ranges(&self) -> usize {
                    0
                }

                fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                    None
                }
            }
            static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // "main"
            second_module.compute_entry_point(CStr::from_ptr(NAME.as_ptr() as *const _), Layout)
        };

        let pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &first_shader, &(), Some(cache.clone())).unwrap(),
        );

        let cache_data = cache.get_data().unwrap();

        let second_pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &second_shader, &(), Some(cache.clone())).unwrap(),
        );

        let second_data = cache.get_data().unwrap();

        if cache_data.is_empty() {
            assert_eq!(cache_data, second_data);
        } else {
            assert_ne!(cache_data, second_data);
        }
    }

    #[test]
    fn cache_data_does_not_change() {
        let (device, queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::empty(device.clone()).unwrap();

        let module = unsafe {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u8; 192] = [
                3, 2, 35, 7, 0, 0, 1, 0, 10, 0, 8, 0, 6, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0, 0,
                0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100, 46, 52, 53, 48, 0,
                0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 5, 0, 5, 0, 0, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 16, 0, 6, 0, 4, 0, 0, 0, 17, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 5, 0, 4, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2,
                0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 248, 0, 2, 0,
                5, 0, 0, 0, 253, 0, 1, 0, 56, 0, 1, 0,
            ];
            ShaderModule::new(device.clone(), &MODULE).unwrap()
        };

        let shader = unsafe {
            #[derive(Clone)]
            struct Layout;
            unsafe impl PipelineLayoutDesc for Layout {
                fn num_sets(&self) -> usize {
                    0
                }

                fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                    None
                }

                fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                    None
                }

                fn num_push_constants_ranges(&self) -> usize {
                    0
                }

                fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                    None
                }
            }
            static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // "main"
            module.compute_entry_point(CStr::from_ptr(NAME.as_ptr() as *const _), Layout)
        };

        let pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &shader, &(), Some(cache.clone())).unwrap(),
        );

        let cache_data = cache.get_data().unwrap();

        let second_pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &shader, &(), Some(cache.clone())).unwrap(),
        );

        let second_data = cache.get_data().unwrap();

        assert_eq!(cache_data, second_data);
    }
}
