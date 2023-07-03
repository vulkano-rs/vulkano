// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Image storage (1D, 2D, 3D, arrays, etc.) and image views.
//!
//! An *image* is a region of memory whose purpose is to store multi-dimensional data. Its
//! most common use is to store a 2D array of color pixels (in other words an *image* in
//! everyday language), but it can also be used to store arbitrary data.
//!
//! The advantage of using an image compared to a buffer is that the memory layout is optimized
//! for locality. When reading a specific pixel of an image, reading the nearby pixels is really
//! fast. Most implementations have hardware dedicated to reading from images if you access them
//! through a sampler.
//!
//! # Properties of an image
//!
//! TODO
//!
//! # Images and image views
//!
//! There is a distinction between *images* and *image views*. As its name suggests, an image
//! view describes how the GPU must interpret the image.
//!
//! Transfer and memory operations operate on images themselves, while reading/writing an image
//! operates on image views. You can create multiple image views from the same image.
//!
//! # High-level wrappers
//!
//! In the vulkano library, images that have memory bound to them are represented by [`Image`]. You
//! can [create an `Image` directly] by providing a memory allocator and all the info for the image
//! and allocation you want to create. This should satisify most use cases. The more low-level use
//! cases such as importing memory for the image are described below.
//!
//! You can create an [`ImageView`] from any `Image`.
//!
//! # Low-level information
//!
//! [`RawImage`] is the low-level wrapper around a `VkImage`, which has no memory bound to it. You
//! can [create a `RawImage`] similarly to `Image` except that you don't provide any info about the
//! allocation. That way, you can [bind memory to it] however you wish, including:
//!
//! - Binding [`DeviceMemory`] you [allocated yourself], for instance with specific export handle
//!   types.
//! - Binding [imported] `DeviceMemory`.
//!
//! You can [create a `MemoryAlloc` from `DeviceMemory`] if you want to bind its own block of
//! memory to an image.
//!
//! [`ImageView`]: crate::image::view::ImageView
//! [create an `Image` directly]: Image::new
//! [create a `RawImage`]: RawImage::new
//! [bind memory to it]: RawImage::bind_memory
//! [`DeviceMemory`]: crate::memory::DeviceMemory
//! [allocated yourself]: crate::memory::DeviceMemory::allocate
//! [imported]: crate::memory::DeviceMemory::import
//! [create a `MemoryAlloc` from `DeviceMemory`]: MemoryAlloc::new

pub use self::sys::ImageCreateInfo;
pub use self::{
    aspect::{ImageAspect, ImageAspects},
    layout::ImageLayout,
    usage::ImageUsage,
};
use self::{
    sys::{ImageError, RawImage},
    view::ImageViewType,
};
use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    format::{Format, FormatFeatures},
    macros::{vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
    memory::{
        allocator::{AllocationCreateInfo, MemoryAlloc, MemoryAllocator},
        is_aligned, DedicatedAllocation, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        ExternalMemoryProperties, MemoryRequirements,
    },
    range_map::RangeMap,
    swapchain::Swapchain,
    sync::{future::AccessError, AccessConflict, CurrentAccess, Sharing},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    cmp,
    hash::{Hash, Hasher},
    iter::{FusedIterator, Peekable},
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

mod aspect;
mod layout;
pub mod sampler;
pub mod sys;
mod usage;
pub mod view;

/// A multi-dimensioned storage for texel data.
///
/// Unlike [`RawImage`], an `Image` has memory backing it, and can be used normally.
///
/// See also [the module-level documentation] for more information about images.
///
/// [the module-level documentation]: self
#[derive(Debug)]
pub struct Image {
    inner: RawImage,
    memory: ImageMemory,

    aspect_list: SmallVec<[ImageAspect; 4]>,
    aspect_size: DeviceSize,
    mip_level_size: DeviceSize,
    range_size: DeviceSize,

    state: Mutex<ImageState>,
    layout: ImageLayout,
    is_layout_initialized: AtomicBool,
}

/// The type of backing memory that an image can have.
#[derive(Debug)]
pub enum ImageMemory {
    /// The image is backed by normal memory, bound with [`bind_memory`].
    ///
    /// [`bind_memory`]: RawImage::bind_memory
    Normal(SmallVec<[MemoryAlloc; 3]>),

    /// The image is backed by sparse memory, bound with [`bind_sparse`].
    ///
    /// [`bind_sparse`]: crate::device::QueueGuard::bind_sparse
    Sparse(Vec<SparseImageMemoryRequirements>),

    /// The image is backed by memory owned by a [`Swapchain`].
    Swapchain {
        swapchain: Arc<Swapchain>,
        image_index: u32,
    },
}

impl Image {
    /// Creates a new uninitialized `Image`.
    pub fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        image_info: ImageCreateInfo,
        allocation_info: AllocationCreateInfo,
    ) -> Result<Arc<Self>, ImageError> {
        // TODO: adjust the code below to make this safe
        assert!(!image_info.flags.intersects(ImageCreateFlags::DISJOINT));

        let allocation_type = image_info.tiling.into();
        let raw_image = RawImage::new(allocator.device().clone(), image_info)?;
        let requirements = raw_image.memory_requirements()[0];

        let allocation = unsafe {
            allocator.allocate_unchecked(
                requirements,
                allocation_type,
                allocation_info,
                Some(DedicatedAllocation::Image(&raw_image)),
            )
        }?;

        debug_assert!(is_aligned(
            allocation.offset(),
            requirements.layout.alignment(),
        ));
        debug_assert!(allocation.size() == requirements.layout.size());

        let image =
            unsafe { raw_image.bind_memory_unchecked([allocation]) }.map_err(|(err, _, _)| err)?;

        Ok(Arc::new(image))
    }

    fn from_raw(inner: RawImage, memory: ImageMemory, layout: ImageLayout) -> Self {
        let aspects = inner.format().unwrap().aspects();
        let aspect_list: SmallVec<[ImageAspect; 4]> = aspects.into_iter().collect();
        let mip_level_size = inner.dimensions().array_layers() as DeviceSize;
        let aspect_size = mip_level_size * inner.mip_levels() as DeviceSize;
        let range_size = aspect_list.len() as DeviceSize * aspect_size;
        let state = Mutex::new(ImageState::new(range_size, inner.initial_layout()));

        Image {
            inner,
            memory,

            aspect_list,
            aspect_size,
            mip_level_size,
            range_size,

            state,
            is_layout_initialized: AtomicBool::new(false),
            layout,
        }
    }

    pub(crate) unsafe fn from_swapchain(
        handle: ash::vk::Image,
        swapchain: Arc<Swapchain>,
        image_index: u32,
    ) -> Self {
        let create_info = ImageCreateInfo {
            flags: ImageCreateFlags::empty(),
            dimensions: ImageDimensions::Dim2d {
                width: swapchain.image_extent()[0],
                height: swapchain.image_extent()[1],
                array_layers: swapchain.image_array_layers(),
            },
            format: Some(swapchain.image_format()),
            initial_layout: ImageLayout::Undefined,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: swapchain.image_usage(),
            stencil_usage: swapchain.image_usage(),
            sharing: swapchain.image_sharing().clone(),
            ..Default::default()
        };

        Self::from_raw(
            RawImage::from_handle_with_destruction(
                swapchain.device().clone(),
                handle,
                create_info,
                false,
            ),
            ImageMemory::Swapchain {
                swapchain,
                image_index,
            },
            ImageLayout::PresentSrc,
        )
    }

    /// Returns the type of memory that is backing this image.
    #[inline]
    pub fn memory(&self) -> &ImageMemory {
        &self.memory
    }

    /// Returns the memory requirements for this image.
    ///
    /// - If the image is a swapchain image, this returns a slice with a length of 0.
    /// - If `self.flags().disjoint` is not set, this returns a slice with a length of 1.
    /// - If `self.flags().disjoint` is set, this returns a slice with a length equal to
    ///   `self.format().unwrap().planes().len()`.
    #[inline]
    pub fn memory_requirements(&self) -> &[MemoryRequirements] {
        self.inner.memory_requirements()
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.inner.flags()
    }

    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> ImageDimensions {
        self.inner.dimensions()
    }

    /// Returns the image's format.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.inner.format()
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.inner.format_features()
    }

    /// Returns the number of mipmap levels in the image.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.inner.mip_levels()
    }

    /// Returns the initial layout of the image.
    #[inline]
    pub fn initial_layout(&self) -> ImageLayout {
        self.inner.initial_layout()
    }

    /// Returns the number of samples for the image.
    #[inline]
    pub fn samples(&self) -> SampleCount {
        self.inner.samples()
    }

    /// Returns the tiling of the image.
    #[inline]
    pub fn tiling(&self) -> ImageTiling {
        self.inner.tiling()
    }

    /// Returns the usage the image was created with.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.inner.usage()
    }

    /// Returns the stencil usage the image was created with.
    #[inline]
    pub fn stencil_usage(&self) -> ImageUsage {
        self.inner.stencil_usage()
    }

    /// Returns the sharing the image was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        self.inner.sharing()
    }

    /// Returns the external memory handle types that are supported with this image.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.inner.external_memory_handle_types()
    }

    /// Returns an `ImageSubresourceLayers` covering the first mip level of the image. All aspects
    /// of the image are selected, or `plane0` if the image is multi-planar.
    #[inline]
    pub fn subresource_layers(&self) -> ImageSubresourceLayers {
        self.inner.subresource_layers()
    }

    /// Returns an `ImageSubresourceRange` covering the whole image. If the image is multi-planar,
    /// only the `color` aspect is selected.
    #[inline]
    pub fn subresource_range(&self) -> ImageSubresourceRange {
        self.inner.subresource_range()
    }

    /// Queries the memory layout of a single subresource of the image.
    ///
    /// Only images with linear tiling are supported, if they do not have a format with both a
    /// depth and a stencil format. Images with optimal tiling have an opaque image layout that is
    /// not suitable for direct memory accesses, and likewise for combined depth/stencil formats.
    /// Multi-planar formats are supported, but you must specify one of the planes as the `aspect`,
    /// not [`ImageAspect::Color`].
    ///
    /// The layout is invariant for each image. However it is not cached, as this would waste
    /// memory in the case of non-linear-tiling images. You are encouraged to store the layout
    /// somewhere in order to avoid calling this semi-expensive function at every single memory
    /// access.
    #[inline]
    pub fn subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<SubresourceLayout, ImageError> {
        self.inner
            .subresource_layout(aspect, mip_level, array_layer)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn subresource_layout_unchecked(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> SubresourceLayout {
        self.inner
            .subresource_layout_unchecked(aspect, mip_level, array_layer)
    }

    pub(crate) fn range_size(&self) -> DeviceSize {
        self.range_size
    }

    /// Returns an iterator over subresource ranges.
    ///
    /// In ranges, the subresources are "flattened" to `DeviceSize`, where each index in the range
    /// is a single array layer. The layers are arranged hierarchically: aspects at the top level,
    /// with the mip levels in that aspect, and the array layers in that mip level.
    pub(crate) fn iter_ranges(
        &self,
        subresource_range: ImageSubresourceRange,
    ) -> SubresourceRangeIterator {
        assert!(self
            .format()
            .unwrap()
            .aspects()
            .contains(subresource_range.aspects));
        assert!(subresource_range.mip_levels.end <= self.mip_levels());
        assert!(subresource_range.array_layers.end <= self.dimensions().array_layers());

        SubresourceRangeIterator::new(
            subresource_range,
            &self.aspect_list,
            self.aspect_size,
            self.mip_levels(),
            self.mip_level_size,
            self.dimensions().array_layers(),
        )
    }

    pub(crate) fn range_to_subresources(
        &self,
        mut range: Range<DeviceSize>,
    ) -> ImageSubresourceRange {
        debug_assert!(!range.is_empty());
        debug_assert!(range.end <= self.range_size);

        if range.end - range.start > self.aspect_size {
            debug_assert!(range.start % self.aspect_size == 0);
            debug_assert!(range.end % self.aspect_size == 0);

            let start_aspect_num = (range.start / self.aspect_size) as usize;
            let end_aspect_num = (range.end / self.aspect_size) as usize;

            ImageSubresourceRange {
                aspects: self.aspect_list[start_aspect_num..end_aspect_num]
                    .iter()
                    .copied()
                    .collect(),
                mip_levels: 0..self.mip_levels(),
                array_layers: 0..self.dimensions().array_layers(),
            }
        } else {
            let aspect_num = (range.start / self.aspect_size) as usize;
            range.start %= self.aspect_size;
            range.end %= self.aspect_size;

            // Wraparound
            if range.end == 0 {
                range.end = self.aspect_size;
            }

            if range.end - range.start > self.mip_level_size {
                debug_assert!(range.start % self.mip_level_size == 0);
                debug_assert!(range.end % self.mip_level_size == 0);

                let start_mip_level = (range.start / self.mip_level_size) as u32;
                let end_mip_level = (range.end / self.mip_level_size) as u32;

                ImageSubresourceRange {
                    aspects: self.aspect_list[aspect_num].into(),
                    mip_levels: start_mip_level..end_mip_level,
                    array_layers: 0..self.dimensions().array_layers(),
                }
            } else {
                let mip_level = (range.start / self.mip_level_size) as u32;
                range.start %= self.mip_level_size;
                range.end %= self.mip_level_size;

                // Wraparound
                if range.end == 0 {
                    range.end = self.mip_level_size;
                }

                let start_array_layer = range.start as u32;
                let end_array_layer = range.end as u32;

                ImageSubresourceRange {
                    aspects: self.aspect_list[aspect_num].into(),
                    mip_levels: mip_level..mip_level + 1,
                    array_layers: start_array_layer..end_array_layer,
                }
            }
        }
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, ImageState> {
        self.state.lock()
    }

    pub(crate) fn initial_layout_requirement(&self) -> ImageLayout {
        self.layout
    }

    pub(crate) fn final_layout_requirement(&self) -> ImageLayout {
        self.layout
    }

    pub(crate) unsafe fn layout_initialized(&self) {
        match &self.memory {
            ImageMemory::Normal(..) | ImageMemory::Sparse(..) => {
                self.is_layout_initialized.store(true, Ordering::Release);
            }
            ImageMemory::Swapchain {
                swapchain,
                image_index,
            } => {
                swapchain.image_layout_initialized(*image_index);
            }
        }
    }

    pub(crate) fn is_layout_initialized(&self) -> bool {
        match &self.memory {
            ImageMemory::Normal(..) | ImageMemory::Sparse(..) => {
                self.is_layout_initialized.load(Ordering::Acquire)
            }
            ImageMemory::Swapchain {
                swapchain,
                image_index,
            } => swapchain.is_image_layout_initialized(*image_index),
        }
    }
}

unsafe impl VulkanObject for Image {
    type Handle = ash::vk::Image;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for Image {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl PartialEq for Image {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// The current state of an image.
#[derive(Debug)]
pub(crate) struct ImageState {
    ranges: RangeMap<DeviceSize, ImageRangeState>,
}

impl ImageState {
    fn new(size: DeviceSize, initial_layout: ImageLayout) -> Self {
        ImageState {
            ranges: [(
                0..size,
                ImageRangeState {
                    current_access: CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    },
                    layout: initial_layout,
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn check_cpu_read(&self, range: Range<DeviceSize>) -> Result<(), AccessConflict> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive { .. } => return Err(AccessConflict::HostWrite),
                CurrentAccess::GpuExclusive { .. } => return Err(AccessConflict::DeviceWrite),
                CurrentAccess::Shared { .. } => (),
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => {
                    *cpu_reads += 1;
                }
                _ => unreachable!("Image is being written by the CPU or GPU"),
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::Shared { cpu_reads, .. } => *cpu_reads -= 1,
                _ => unreachable!("Image was not locked for CPU read"),
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn check_cpu_write(&self, range: Range<DeviceSize>) -> Result<(), AccessConflict> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::CpuExclusive => return Err(AccessConflict::HostWrite),
                CurrentAccess::GpuExclusive { .. } => return Err(AccessConflict::DeviceWrite),
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                CurrentAccess::Shared { cpu_reads, .. } if *cpu_reads > 0 => {
                    return Err(AccessConflict::HostRead)
                }
                CurrentAccess::Shared { .. } => return Err(AccessConflict::DeviceRead),
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_write_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            state.current_access = CurrentAccess::CpuExclusive;
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn cpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::CpuExclusive => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads: 0,
                    }
                }
                _ => unreachable!("Image was not locked for CPU write"),
            }
        }
    }

    pub(crate) fn check_gpu_read(
        &self,
        range: Range<DeviceSize>,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared { .. } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }

            if expected_layout != ImageLayout::Undefined && state.layout != expected_layout {
                return Err(AccessError::UnexpectedImageLayout {
                    allowed: state.layout,
                    requested: expected_layout,
                });
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_read_lock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. }
                | CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads += 1,
                _ => unreachable!("Image is being written by the CPU"),
            }
        }
    }

    pub(crate) unsafe fn gpu_read_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_reads, .. } => *gpu_reads -= 1,
                CurrentAccess::Shared { gpu_reads, .. } => *gpu_reads -= 1,
                _ => unreachable!("Buffer was not locked for GPU read"),
            }
        }
    }

    pub(crate) fn check_gpu_write(
        &self,
        range: Range<DeviceSize>,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        for (_range, state) in self.ranges.range(&range) {
            match &state.current_access {
                CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads: 0,
                } => (),
                _ => return Err(AccessError::AlreadyInUse),
            }

            if expected_layout != ImageLayout::Undefined && state.layout != expected_layout {
                return Err(AccessError::UnexpectedImageLayout {
                    allowed: state.layout,
                    requested: expected_layout,
                });
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn gpu_write_lock(
        &mut self,
        range: Range<DeviceSize>,
        destination_layout: ImageLayout,
    ) {
        debug_assert!(!matches!(
            destination_layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized
        ));

        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes += 1,
                &mut CurrentAccess::Shared {
                    cpu_reads: 0,
                    gpu_reads,
                } => {
                    state.current_access = CurrentAccess::GpuExclusive {
                        gpu_reads,
                        gpu_writes: 1,
                    }
                }
                _ => unreachable!("Image is being accessed by the CPU"),
            }

            state.layout = destination_layout;
        }
    }

    pub(crate) unsafe fn gpu_write_unlock(&mut self, range: Range<DeviceSize>) {
        self.ranges.split_at(&range.start);
        self.ranges.split_at(&range.end);

        for (_range, state) in self.ranges.range_mut(&range) {
            match &mut state.current_access {
                &mut CurrentAccess::GpuExclusive {
                    gpu_reads,
                    gpu_writes: 1,
                } => {
                    state.current_access = CurrentAccess::Shared {
                        cpu_reads: 0,
                        gpu_reads,
                    }
                }
                CurrentAccess::GpuExclusive { gpu_writes, .. } => *gpu_writes -= 1,
                _ => unreachable!("Image was not locked for GPU write"),
            }
        }
    }
}

/// The current state of a specific subresource range in an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageRangeState {
    current_access: CurrentAccess,
    layout: ImageLayout,
}

#[derive(Clone)]
pub(crate) struct SubresourceRangeIterator {
    next_fn: fn(&mut Self) -> Option<Range<DeviceSize>>,
    image_aspect_size: DeviceSize,
    image_mip_level_size: DeviceSize,
    mip_levels: Range<u32>,
    array_layers: Range<u32>,

    aspect_nums: Peekable<smallvec::IntoIter<[usize; 4]>>,
    current_aspect_num: Option<usize>,
    current_mip_level: u32,
}

impl SubresourceRangeIterator {
    fn new(
        subresource_range: ImageSubresourceRange,
        image_aspect_list: &[ImageAspect],
        image_aspect_size: DeviceSize,
        image_mip_levels: u32,
        image_mip_level_size: DeviceSize,
        image_array_layers: u32,
    ) -> Self {
        assert!(!subresource_range.mip_levels.is_empty());
        assert!(!subresource_range.array_layers.is_empty());

        let next_fn = if subresource_range.array_layers.start != 0
            || subresource_range.array_layers.end != image_array_layers
        {
            Self::next_some_layers
        } else if subresource_range.mip_levels.start != 0
            || subresource_range.mip_levels.end != image_mip_levels
        {
            Self::next_some_levels_all_layers
        } else {
            Self::next_all_levels_all_layers
        };

        let mut aspect_nums = subresource_range
            .aspects
            .into_iter()
            .map(|aspect| image_aspect_list.iter().position(|&a| a == aspect).unwrap())
            .collect::<SmallVec<[usize; 4]>>()
            .into_iter()
            .peekable();
        assert!(aspect_nums.len() != 0);
        let current_aspect_num = aspect_nums.next();
        let current_mip_level = subresource_range.mip_levels.start;

        Self {
            next_fn,
            image_aspect_size,
            image_mip_level_size,
            mip_levels: subresource_range.mip_levels,
            array_layers: subresource_range.array_layers,

            aspect_nums,
            current_aspect_num,
            current_mip_level,
        }
    }

    /// Used when the requested range contains only a subset of the array layers in the image.
    /// The iterator returns one range for each mip level and aspect, each covering the range of
    /// array layers of that mip level and aspect.
    fn next_some_layers(&mut self) -> Option<Range<DeviceSize>> {
        self.current_aspect_num.map(|aspect_num| {
            let mip_level_offset = aspect_num as DeviceSize * self.image_aspect_size
                + self.current_mip_level as DeviceSize * self.image_mip_level_size;
            self.current_mip_level += 1;

            if self.current_mip_level >= self.mip_levels.end {
                self.current_mip_level = self.mip_levels.start;
                self.current_aspect_num = self.aspect_nums.next();
            }

            let start = mip_level_offset + self.array_layers.start as DeviceSize;
            let end = mip_level_offset + self.array_layers.end as DeviceSize;
            start..end
        })
    }

    /// Used when the requested range contains all array layers in the image, but not all mip
    /// levels. The iterator returns one range for each aspect, each covering all layers of the
    /// range of mip levels of that aspect.
    fn next_some_levels_all_layers(&mut self) -> Option<Range<DeviceSize>> {
        self.current_aspect_num.map(|aspect_num| {
            let aspect_offset = aspect_num as DeviceSize * self.image_aspect_size;
            self.current_aspect_num = self.aspect_nums.next();

            let start =
                aspect_offset + self.mip_levels.start as DeviceSize * self.image_mip_level_size;
            let end = aspect_offset + self.mip_levels.end as DeviceSize * self.image_mip_level_size;
            start..end
        })
    }

    /// Used when the requested range contains all array layers and mip levels in the image.
    /// The iterator returns one range for each series of adjacent aspect numbers, each covering
    /// all mip levels and all layers of those aspects. If the range contains the whole image, then
    /// exactly one range is returned since all aspect numbers will be adjacent.
    fn next_all_levels_all_layers(&mut self) -> Option<Range<DeviceSize>> {
        self.current_aspect_num.map(|aspect_num_start| {
            self.current_aspect_num = self.aspect_nums.next();
            let mut aspect_num_end = aspect_num_start + 1;

            while self.current_aspect_num == Some(aspect_num_end) {
                self.current_aspect_num = self.aspect_nums.next();
                aspect_num_end += 1;
            }

            let start = aspect_num_start as DeviceSize * self.image_aspect_size;
            let end = aspect_num_end as DeviceSize * self.image_aspect_size;
            start..end
        })
    }
}

impl Iterator for SubresourceRangeIterator {
    type Item = Range<DeviceSize>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.next_fn)(self)
    }
}

impl FusedIterator for SubresourceRangeIterator {}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that can be set when creating a new image.
    ImageCreateFlags = ImageCreateFlags(u32);

    /* TODO: enable
    /// The image will be backed by sparse memory binding (through queue commands) instead of
    /// regular binding (through [`bind_memory`]).
    ///
    /// The [`sparse_binding`] feature must be enabled on the device.
    ///
    /// [`bind_memory`]: sys::RawImage::bind_memory
    /// [`sparse_binding`]: crate::device::Features::sparse_binding
    SPARSE_BINDING = SPARSE_BINDING,*/

    /* TODO: enable
    /// The image can be used without being fully resident in memory at the time of use.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// Depending on the image dimensions, either the [`sparse_residency_image2_d`] or the
    /// [`sparse_residency_image3_d`] feature must be enabled on the device.
    /// For a multisampled image, the one of the features [`sparse_residency2_samples`],
    /// [`sparse_residency4_samples`], [`sparse_residency8_samples`] or
    /// [`sparse_residency16_samples`], corresponding to the sample count of the image, must
    /// be enabled on the device.
    ///
    /// [`sparse_binding`]: crate::device::Features::sparse_binding
    /// [`sparse_residency_image2_d`]: crate::device::Features::sparse_residency_image2_d
    /// [`sparse_residency_image2_3`]: crate::device::Features::sparse_residency_image3_d
    /// [`sparse_residency2_samples`]: crate::device::Features::sparse_residency2_samples
    /// [`sparse_residency4_samples`]: crate::device::Features::sparse_residency4_samples
    /// [`sparse_residency8_samples`]: crate::device::Features::sparse_residency8_samples
    /// [`sparse_residency16_samples`]: crate::device::Features::sparse_residency16_samples
    SPARSE_RESIDENCY = SPARSE_RESIDENCY,*/

    /* TODO: enable
    /// The buffer's memory can alias with another image or a different part of the same image.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_aliased`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_aliased`]: crate::device::Features::sparse_residency_aliased
    SPARSE_ALIASED = SPARSE_ALIASED,*/

    /// For non-multi-planar formats, whether an image view wrapping the image can have a
    /// different format.
    ///
    /// For multi-planar formats, whether an image view wrapping the image can be created from a
    /// single plane of the image.
    MUTABLE_FORMAT = MUTABLE_FORMAT,

    /// For 2D images, whether an image view of type [`ImageViewType::Cube`] or
    /// [`ImageViewType::CubeArray`] can be created from the image.
    ///
    /// [`ImageViewType::Cube`]: crate::image::view::ImageViewType::Cube
    /// [`ImageViewType::CubeArray`]: crate::image::view::ImageViewType::CubeArray
    CUBE_COMPATIBLE = CUBE_COMPATIBLE,

    /* TODO: enable
    // TODO: document
    ALIAS = ALIAS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_bind_memory2)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SPLIT_INSTANCE_BIND_REGIONS = SPLIT_INSTANCE_BIND_REGIONS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /// For 3D images, whether an image view of type [`ImageViewType::Dim2d`] or
    /// [`ImageViewType::Dim2dArray`] can be created from the image.
    ///
    /// On [portability subset] devices, the [`image_view2_d_on3_d_image`] feature must be enabled
    /// on the device.
    ///
    /// [`ImageViewType::Dim2d`]: crate::image::view::ImageViewType::Dim2d
    /// [`ImageViewType::Dim2dArray`]: crate::image::view::ImageViewType::Dim2dArray
    /// [portability subset]: crate::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`image_view2_d_on3_d_image`]: crate::device::Features::image_view2_d_on3_d_image
    ARRAY_2D_COMPATIBLE = TYPE_2D_ARRAY_COMPATIBLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance1)]),
    ]),

    /// For images with a compressed format, whether an image view with an uncompressed
    /// format can be created from the image, where each texel in the view will correspond to a
    /// compressed texel block in the image.
    ///
    /// Requires `mutable_format`.
    BLOCK_TEXEL_VIEW_COMPATIBLE = BLOCK_TEXEL_VIEW_COMPATIBLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance2)]),
    ]),

    /* TODO: enable
    // TODO: document
    EXTENDED_USAGE = EXTENDED_USAGE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance2)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),*/

    /// For images with a multi-planar format, whether each plane will have its memory bound
    /// separately, rather than having a single memory binding for the whole image.
    DISJOINT = DISJOINT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /* TODO: enable
    // TODO: document
    CORNER_SAMPLED = CORNER_SAMPLED_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_corner_sampled_image)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SAMPLE_LOCATIONS_COMPATIBLE_DEPTH = SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_sample_locations)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SUBSAMPLED = SUBSAMPLED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED = MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_multisampled_render_to_single_sampled)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    TYPE_2D_VIEW_COMPATIBLE = TYPE_2D_VIEW_COMPATIBLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_2d_view_of_3d)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FRAGMENT_DENSITY_MAP_OFFSET = FRAGMENT_DENSITY_MAP_OFFSET_QCOM
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(qcom_fragment_density_map_offset)]),
    ]),*/
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`SampleCount`] values.
    SampleCounts impl {
        /// Returns the maximum sample count in `self`.
        #[inline]
        pub const fn max_count(self) -> SampleCount {
            if self.intersects(SampleCounts::SAMPLE_64) {
                SampleCount::Sample64
            } else if self.intersects(SampleCounts::SAMPLE_32) {
                SampleCount::Sample32
            } else if self.intersects(SampleCounts::SAMPLE_16) {
                SampleCount::Sample16
            } else if self.intersects(SampleCounts::SAMPLE_8) {
                SampleCount::Sample8
            } else if self.intersects(SampleCounts::SAMPLE_4) {
                SampleCount::Sample4
            } else if self.intersects(SampleCounts::SAMPLE_2) {
                SampleCount::Sample2
            } else {
                SampleCount::Sample1
            }
        }
    },

    /// The number of samples per texel of an image.
    SampleCount,

    = SampleCountFlags(u32);

    /// 1 sample per texel.
    SAMPLE_1, Sample1 = TYPE_1,

    /// 2 samples per texel.
    SAMPLE_2, Sample2 = TYPE_2,

    /// 4 samples per texel.
    SAMPLE_4, Sample4 = TYPE_4,

    /// 8 samples per texel.
    SAMPLE_8, Sample8 = TYPE_8,

    /// 16 samples per texel.
    SAMPLE_16, Sample16 = TYPE_16,

    /// 32 samples per texel.
    SAMPLE_32, Sample32 = TYPE_32,

    /// 64 samples per texel.
    SAMPLE_64, Sample64 = TYPE_64,
}

impl From<SampleCount> for u32 {
    #[inline]
    fn from(value: SampleCount) -> Self {
        value as u32
    }
}

impl TryFrom<u32> for SampleCount {
    type Error = ();

    #[inline]
    fn try_from(val: u32) -> Result<Self, Self::Error> {
        match val {
            1 => Ok(Self::Sample1),
            2 => Ok(Self::Sample2),
            4 => Ok(Self::Sample4),
            8 => Ok(Self::Sample8),
            16 => Ok(Self::Sample16),
            32 => Ok(Self::Sample32),
            64 => Ok(Self::Sample64),
            _ => Err(()),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    ImageType = ImageType(i32);

    // TODO: document
    Dim1d = TYPE_1D,

    // TODO: document
    Dim2d = TYPE_2D,

    // TODO: document
    Dim3d = TYPE_3D,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    ImageTiling = ImageTiling(i32);

    // TODO: document
    Optimal = OPTIMAL,

    // TODO: document
    Linear = LINEAR,

    // TODO: document
    DrmFormatModifier = DRM_FORMAT_MODIFIER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_drm_format_modifier)]),
    ]),
}

/// The dimensions of an image.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageDimensions {
    Dim1d {
        width: u32,
        array_layers: u32,
    },
    Dim2d {
        width: u32,
        height: u32,
        array_layers: u32,
    },
    Dim3d {
        width: u32,
        height: u32,
        depth: u32,
    },
}

impl ImageDimensions {
    #[inline]
    pub fn width(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { width, .. } => width,
            ImageDimensions::Dim2d { width, .. } => width,
            ImageDimensions::Dim3d { width, .. } => width,
        }
    }

    #[inline]
    pub fn height(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { .. } => 1,
            ImageDimensions::Dim2d { height, .. } => height,
            ImageDimensions::Dim3d { height, .. } => height,
        }
    }

    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { .. } => 1,
            ImageDimensions::Dim2d { .. } => 1,
            ImageDimensions::Dim3d { depth, .. } => depth,
        }
    }

    #[inline]
    pub fn width_height_depth(&self) -> [u32; 3] {
        [self.width(), self.height(), self.depth()]
    }

    #[inline]
    pub fn array_layers(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { array_layers, .. } => array_layers,
            ImageDimensions::Dim2d { array_layers, .. } => array_layers,
            ImageDimensions::Dim3d { .. } => 1,
        }
    }

    /// Returns the total number of texels for an image of these dimensions.
    #[inline]
    pub fn num_texels(&self) -> u32 {
        self.width() * self.height() * self.depth() * self.array_layers()
    }

    #[inline]
    pub fn image_type(&self) -> ImageType {
        match *self {
            ImageDimensions::Dim1d { .. } => ImageType::Dim1d,
            ImageDimensions::Dim2d { .. } => ImageType::Dim2d,
            ImageDimensions::Dim3d { .. } => ImageType::Dim3d,
        }
    }

    /// Returns the maximum number of mipmap levels for these image dimensions.
    ///
    /// The returned value is always at least 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use vulkano::image::ImageDimensions;
    ///
    /// let dims = ImageDimensions::Dim2d {
    ///     width: 32,
    ///     height: 50,
    ///     array_layers: 1,
    /// };
    ///
    /// assert_eq!(dims.max_mip_levels(), 6);
    /// ```
    #[inline]
    pub fn max_mip_levels(&self) -> u32 {
        // This calculates `log2(max(width, height, depth)) + 1` using fast integer operations.
        let max = match *self {
            ImageDimensions::Dim1d { width, .. } => width,
            ImageDimensions::Dim2d { width, height, .. } => width | height,
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => width | height | depth,
        };
        32 - max.leading_zeros()
    }

    /// Returns the dimensions of the `level`th mipmap level. If `level` is 0, then the dimensions
    /// are left unchanged.
    ///
    /// Returns `None` if `level` is superior or equal to `max_mip_levels()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vulkano::image::ImageDimensions;
    ///
    /// let dims = ImageDimensions::Dim2d {
    ///     width: 963,
    ///     height: 256,
    ///     array_layers: 1,
    /// };
    ///
    /// assert_eq!(dims.mip_level_dimensions(0), Some(dims));
    /// assert_eq!(dims.mip_level_dimensions(1), Some(ImageDimensions::Dim2d {
    ///     width: 481,
    ///     height: 128,
    ///     array_layers: 1,
    /// }));
    /// assert_eq!(dims.mip_level_dimensions(6), Some(ImageDimensions::Dim2d {
    ///     width: 15,
    ///     height: 4,
    ///     array_layers: 1,
    /// }));
    /// assert_eq!(dims.mip_level_dimensions(9), Some(ImageDimensions::Dim2d {
    ///     width: 1,
    ///     height: 1,
    ///     array_layers: 1,
    /// }));
    /// assert_eq!(dims.mip_level_dimensions(11), None);
    /// ```
    ///
    /// # Panics
    ///
    /// - In debug mode, panics if `width`, `height` or `depth` is equal to 0. In release, returns
    ///   an unspecified value.
    #[inline]
    pub fn mip_level_dimensions(&self, level: u32) -> Option<ImageDimensions> {
        if level == 0 {
            return Some(*self);
        }

        if level >= self.max_mip_levels() {
            return None;
        }

        Some(match *self {
            ImageDimensions::Dim1d {
                width,
                array_layers,
            } => {
                debug_assert_ne!(width, 0);
                ImageDimensions::Dim1d {
                    array_layers,
                    width: cmp::max(1, width >> level),
                }
            }

            ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            } => {
                debug_assert_ne!(width, 0);
                debug_assert_ne!(height, 0);
                ImageDimensions::Dim2d {
                    width: cmp::max(1, width >> level),
                    height: cmp::max(1, height >> level),
                    array_layers,
                }
            }

            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => {
                debug_assert_ne!(width, 0);
                debug_assert_ne!(height, 0);
                ImageDimensions::Dim3d {
                    width: cmp::max(1, width >> level),
                    height: cmp::max(1, height >> level),
                    depth: cmp::max(1, depth >> level),
                }
            }
        })
    }
}

/// One or more subresources of an image, spanning a single mip level, that should be accessed by a
/// command.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageSubresourceLayers {
    /// Selects the aspects that will be included.
    ///
    /// The value must not be empty, and must not include any of the `memory_plane` aspects.
    /// The `color` aspect cannot be selected together any of with the `plane` aspects.
    pub aspects: ImageAspects,

    /// Selects mip level that will be included.
    pub mip_level: u32,

    /// Selects the range of array layers that will be included.
    ///
    /// The range must not be empty.
    pub array_layers: Range<u32>,
}

impl ImageSubresourceLayers {
    /// Returns an `ImageSubresourceLayers` from the given image parameters, covering the first
    /// mip level of the image. All aspects of the image are selected, or `plane0` if the image
    /// is multi-planar.
    #[inline]
    pub fn from_parameters(format: Format, array_layers: u32) -> Self {
        Self {
            aspects: {
                let aspects = format.aspects();

                if aspects.intersects(ImageAspects::PLANE_0) {
                    ImageAspects::PLANE_0
                } else {
                    aspects
                }
            },
            mip_level: 0,
            array_layers: 0..array_layers,
        }
    }
}

impl From<ImageSubresourceLayers> for ash::vk::ImageSubresourceLayers {
    #[inline]
    fn from(val: ImageSubresourceLayers) -> Self {
        Self {
            aspect_mask: val.aspects.into(),
            mip_level: val.mip_level,
            base_array_layer: val.array_layers.start,
            layer_count: val.array_layers.end - val.array_layers.start,
        }
    }
}

impl From<&ImageSubresourceLayers> for ash::vk::ImageSubresourceLayers {
    #[inline]
    fn from(val: &ImageSubresourceLayers) -> Self {
        Self {
            aspect_mask: val.aspects.into(),
            mip_level: val.mip_level,
            base_array_layer: val.array_layers.start,
            layer_count: val.array_layers.end - val.array_layers.start,
        }
    }
}

/// One or more subresources of an image that should be accessed by a command.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageSubresourceRange {
    /// Selects the aspects that will be included.
    ///
    /// The value must not be empty, and must not include any of the `memory_plane` aspects.
    /// The `color` aspect cannot be selected together any of with the `plane` aspects.
    pub aspects: ImageAspects,

    /// Selects the range of the mip levels that will be included.
    ///
    /// The range must not be empty.
    pub mip_levels: Range<u32>,

    /// Selects the range of array layers that will be included.
    ///
    /// The range must not be empty.
    pub array_layers: Range<u32>,
}

impl ImageSubresourceRange {
    /// Returns an `ImageSubresourceRange` from the given image parameters, covering the whole
    /// image. If the image is multi-planar, only the `color` aspect is selected.
    #[inline]
    pub fn from_parameters(format: Format, mip_levels: u32, array_layers: u32) -> Self {
        Self {
            aspects: format.aspects()
                - (ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2),
            mip_levels: 0..mip_levels,
            array_layers: 0..array_layers,
        }
    }
}

impl From<ImageSubresourceRange> for ash::vk::ImageSubresourceRange {
    #[inline]
    fn from(val: ImageSubresourceRange) -> Self {
        Self {
            aspect_mask: val.aspects.into(),
            base_mip_level: val.mip_levels.start,
            level_count: val.mip_levels.end - val.mip_levels.start,
            base_array_layer: val.array_layers.start,
            layer_count: val.array_layers.end - val.array_layers.start,
        }
    }
}

impl From<ImageSubresourceLayers> for ImageSubresourceRange {
    #[inline]
    fn from(val: ImageSubresourceLayers) -> Self {
        Self {
            aspects: val.aspects,
            mip_levels: val.mip_level..val.mip_level + 1,
            array_layers: val.array_layers,
        }
    }
}

/// Describes the memory layout of a single subresource of an image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceLayout {
    /// The number of bytes from the start of the memory to the start of the queried subresource.
    pub offset: DeviceSize,

    /// The total number of bytes for the queried subresource.
    pub size: DeviceSize,

    /// The number of bytes between two texels or two blocks in adjacent rows.
    pub row_pitch: DeviceSize,

    /// For images with more than one array layer, the number of bytes between two texels or two
    /// blocks in adjacent array layers.
    pub array_pitch: Option<DeviceSize>,

    /// For 3D images, the number of bytes between two texels or two blocks in adjacent depth
    /// layers.
    pub depth_pitch: Option<DeviceSize>,
}

/// The image configuration to query in
/// [`PhysicalDevice::image_format_properties`](crate::device::physical::PhysicalDevice::image_format_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageFormatInfo {
    /// The `flags` that the image will have.
    ///
    /// The default value is [`ImageCreateFlags::empty()`].
    pub flags: ImageCreateFlags,

    /// The `format` that the image will have.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The dimension type that the image will have.
    ///
    /// The default value is [`ImageType::Dim2d`].
    pub image_type: ImageType,

    /// The `tiling` that the image will have.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    /// The `usage` that the image will have.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub usage: ImageUsage,

    /// The `stencil_usage` that the image will have.
    ///
    /// If `stencil_usage` is empty or if `format` does not have both a depth and a stencil aspect,
    /// then it is automatically set to equal `usage`.
    ///
    /// If after this, `stencil_usage` does not equal `usage`,
    /// then the physical device API version must be at least 1.2, or the
    /// [`ext_separate_stencil_usage`](crate::device::DeviceExtensions::ext_separate_stencil_usage)
    /// extension must be supported by the physical device.
    ///
    /// The default value is [`ImageUsage::empty()`].
    pub stencil_usage: ImageUsage,

    /// An external memory handle type that will be imported to or exported from the image.
    ///
    /// This is needed to retrieve the
    /// [`external_memory_properties`](ImageFormatProperties::external_memory_properties) value,
    /// and the physical device API version must be at least 1.1 or the
    /// [`khr_external_memory_capabilities`](crate::instance::InstanceExtensions::khr_external_memory_capabilities)
    /// extension must be enabled on the instance.
    ///
    /// The default value is `None`.
    pub external_memory_handle_type: Option<ExternalMemoryHandleType>,

    /// The image view type that will be created from the image.
    ///
    /// This is needed to retrieve the
    /// [`filter_cubic`](ImageFormatProperties::filter_cubic) and
    /// [`filter_cubic_minmax`](ImageFormatProperties::filter_cubic_minmax) values, and the
    /// [`ext_filter_cubic`](crate::device::DeviceExtensions::ext_filter_cubic) extension must be
    /// supported on the physical device.
    ///
    /// The default value is `None`.
    pub image_view_type: Option<ImageViewType>,

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageFormatInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: ImageCreateFlags::empty(),
            format: None,
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            stencil_usage: ImageUsage::empty(),
            external_memory_handle_type: None,
            image_view_type: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl ImageFormatInfo {
    pub(crate) fn validate(&self, physical_device: &PhysicalDevice) -> Result<(), ValidationError> {
        let &Self {
            flags,
            format,
            image_type,
            tiling,
            usage,
            mut stencil_usage,
            external_memory_handle_type,
            image_view_type,
            _ne: _,
        } = self;

        flags
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        let format = format.ok_or(ValidationError {
            context: "format".into(),
            problem: "is `None`".into(),
            vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-format-parameter"],
            ..Default::default()
        })?;
        let aspects = format.aspects();

        format
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "format".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-format-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        image_type
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "image_type".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-imageType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        tiling
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "tiling".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-tiling-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        usage
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "usage".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-usage-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if usage.is_empty() {
            return Err(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-usage-requiredbitmask"],
                ..Default::default()
            });
        }

        let has_separate_stencil_usage = if stencil_usage.is_empty()
            || !aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            stencil_usage = usage;
            false
        } else {
            stencil_usage == usage
        };

        if has_separate_stencil_usage {
            if !(physical_device.api_version() >= Version::V1_2
                || physical_device
                    .supported_extensions()
                    .ext_separate_stencil_usage)
            {
                return Err(ValidationError {
                    problem: "`stencil_usage` is `Some`, and `format` has both a depth and a \
                        stencil aspect"
                        .into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("ext_separate_stencil_usage")]),
                    ]),
                    ..Default::default()
                });
            }

            stencil_usage
                .validate_physical_device(physical_device)
                .map_err(|err| ValidationError {
                    context: "stencil_usage".into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-stencilUsage-parameter"],
                    ..ValidationError::from_requirement(err)
                })?;

            if stencil_usage.is_empty() {
                return Err(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "is empty".into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-usage-requiredbitmask"],
                    ..Default::default()
                });
            }
        }

        if let Some(handle_type) = external_memory_handle_type {
            if !(physical_device.api_version() >= Version::V1_1
                || physical_device
                    .instance()
                    .enabled_extensions()
                    .khr_external_memory_capabilities)
            {
                return Err(ValidationError {
                    problem: "`external_memory_handle_type` is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::InstanceExtension(
                            "khr_external_memory_capabilities",
                        )]),
                    ]),
                    ..Default::default()
                });
            }

            handle_type
                .validate_physical_device(physical_device)
                .map_err(|err| ValidationError {
                    context: "handle_type".into(),
                    vuids: &["VUID-VkPhysicalDeviceExternalImageFormatInfo-handleType-parameter"],
                    ..ValidationError::from_requirement(err)
                })?;
        }

        if let Some(image_view_type) = image_view_type {
            if !physical_device.supported_extensions().ext_filter_cubic {
                return Err(ValidationError {
                    problem: "`image_view_type` is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_filter_cubic",
                    )])]),
                    ..Default::default()
                });
            }

            image_view_type
                .validate_physical_device(physical_device)
                .map_err(|err| ValidationError {
                    context: "image_view_type".into(),
                    vuids: &[
                        "VUID-VkPhysicalDeviceImageViewImageFormatInfoEXT-imageViewType-parameter",
                    ],
                    ..ValidationError::from_requirement(err)
                })?;
        }

        // TODO: VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02313
        // Currently there is nothing in Vulkano for for adding a VkImageFormatListCreateInfo.

        Ok(())
    }
}

/// The properties that are supported by a physical device for images of a certain type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImageFormatProperties {
    /// The maximum dimensions.
    pub max_extent: [u32; 3],

    /// The maximum number of mipmap levels.
    pub max_mip_levels: u32,

    /// The maximum number of array layers.
    pub max_array_layers: u32,

    /// The supported sample counts.
    pub sample_counts: SampleCounts,

    /// The maximum total size of an image, in bytes. This is guaranteed to be at least
    /// 0x80000000.
    pub max_resource_size: DeviceSize,

    /// The properties for external memory.
    /// This will be [`ExternalMemoryProperties::default()`] if `external_handle_type` was `None`.
    pub external_memory_properties: ExternalMemoryProperties,

    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    pub filter_cubic: bool,

    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max) `reduction_mode`.
    pub filter_cubic_minmax: bool,
}

impl From<ash::vk::ImageFormatProperties> for ImageFormatProperties {
    #[inline]
    fn from(props: ash::vk::ImageFormatProperties) -> Self {
        Self {
            max_extent: [
                props.max_extent.width,
                props.max_extent.height,
                props.max_extent.depth,
            ],
            max_mip_levels: props.max_mip_levels,
            max_array_layers: props.max_array_layers,
            sample_counts: props.sample_counts.into(),
            max_resource_size: props.max_resource_size,
            external_memory_properties: Default::default(),
            filter_cubic: false,
            filter_cubic_minmax: false,
        }
    }
}

/// The image configuration to query in
/// [`PhysicalDevice::sparse_image_format_properties`](crate::device::physical::PhysicalDevice::sparse_image_format_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SparseImageFormatInfo {
    /// The `format` that the image will have.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The dimension type that the image will have.
    ///
    /// The default value is [`ImageType::Dim2d`].
    pub image_type: ImageType,

    /// The `samples` that the image will have.
    ///
    /// The default value is `SampleCount::Sample1`.
    pub samples: SampleCount,

    /// The `usage` that the image will have.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub usage: ImageUsage,

    /// The `tiling` that the image will have.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    pub _ne: crate::NonExhaustive,
}

impl Default for SparseImageFormatInfo {
    #[inline]
    fn default() -> Self {
        Self {
            format: None,
            image_type: ImageType::Dim2d,
            samples: SampleCount::Sample1,
            usage: ImageUsage::empty(),
            tiling: ImageTiling::Optimal,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SparseImageFormatInfo {
    pub(crate) fn validate(&self, physical_device: &PhysicalDevice) -> Result<(), ValidationError> {
        let &Self {
            format,
            image_type,
            samples,
            usage,
            tiling,
            _ne: _,
        } = self;

        let format = format.ok_or(ValidationError {
            context: "format".into(),
            problem: "is `None`".into(),
            vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-format-parameter"],
            ..Default::default()
        })?;

        format
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "format".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-format-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        image_type
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "image_type".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-type-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        samples
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "samples".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-samples-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        usage
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "usage".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-usage-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if usage.is_empty() {
            return Err(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-usage-requiredbitmask"],
                ..Default::default()
            });
        }

        tiling
            .validate_physical_device(physical_device)
            .map_err(|err| ValidationError {
                context: "tiling".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-tiling-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        // VUID-VkPhysicalDeviceSparseImageFormatInfo2-samples-01095
        // TODO:

        Ok(())
    }
}

/// The properties that are supported by a physical device for sparse images of a certain type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SparseImageFormatProperties {
    /// The aspects of the image that the properties apply to.
    pub aspects: ImageAspects,

    /// The size of the sparse image block, in texels or compressed texel blocks.
    ///
    /// If `flags.nonstandard_block_size` is set, then these values do not match the standard
    /// sparse block dimensions for the given format.
    pub image_granularity: [u32; 3],

    /// Additional information about the sparse image.
    pub flags: SparseImageFormatFlags,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying information about a sparse resource.
    SparseImageFormatFlags = SparseImageFormatFlags(u32);

    /// The image uses a single mip tail region for all array layers, instead of one mip tail region
    /// per array layer.
    SINGLE_MIPTAIL = SINGLE_MIPTAIL,

    /// The image's mip tail region begins with the first mip level whose dimensions are not an
    /// integer multiple of the corresponding sparse image block dimensions.
    ALIGNED_MIP_SIZE = ALIGNED_MIP_SIZE,

    /// The image uses non-standard sparse image block dimensions.
    NONSTANDARD_BLOCK_SIZE = NONSTANDARD_BLOCK_SIZE,
}

/// Requirements for binding memory to a sparse image.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SparseImageMemoryRequirements {
    /// The properties of the image format.
    pub format_properties: SparseImageFormatProperties,

    /// The first mip level at which image subresources are included in the mip tail region.
    pub image_mip_tail_first_lod: u32,

    /// The size in bytes of the mip tail region. This value is guaranteed to be a multiple of the
    /// sparse block size in bytes.
    ///
    /// If `format_properties.flags.single_miptail` is set, then this is the size of the whole
    /// mip tail. Otherwise it is the size of the mip tail of a single array layer.
    pub image_mip_tail_size: DeviceSize,

    /// The memory offset that must be used to bind the mip tail region.
    pub image_mip_tail_offset: DeviceSize,

    /// If `format_properties.flags.single_miptail` is not set, specifies the stride between
    /// the mip tail regions of each array layer.
    pub image_mip_tail_stride: Option<DeviceSize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_mip_levels() {
        let dims = ImageDimensions::Dim2d {
            width: 2,
            height: 1,
            array_layers: 1,
        };
        assert_eq!(dims.max_mip_levels(), 2);

        let dims = ImageDimensions::Dim2d {
            width: 2,
            height: 3,
            array_layers: 1,
        };
        assert_eq!(dims.max_mip_levels(), 2);

        let dims = ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 1,
        };
        assert_eq!(dims.max_mip_levels(), 10);
    }

    #[test]
    fn mip_level_dimensions() {
        let dims = ImageDimensions::Dim2d {
            width: 283,
            height: 175,
            array_layers: 1,
        };
        assert_eq!(dims.mip_level_dimensions(0), Some(dims));
        assert_eq!(
            dims.mip_level_dimensions(1),
            Some(ImageDimensions::Dim2d {
                width: 141,
                height: 87,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(2),
            Some(ImageDimensions::Dim2d {
                width: 70,
                height: 43,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(3),
            Some(ImageDimensions::Dim2d {
                width: 35,
                height: 21,
                array_layers: 1,
            })
        );

        assert_eq!(
            dims.mip_level_dimensions(4),
            Some(ImageDimensions::Dim2d {
                width: 17,
                height: 10,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(5),
            Some(ImageDimensions::Dim2d {
                width: 8,
                height: 5,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(6),
            Some(ImageDimensions::Dim2d {
                width: 4,
                height: 2,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(7),
            Some(ImageDimensions::Dim2d {
                width: 2,
                height: 1,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(8),
            Some(ImageDimensions::Dim2d {
                width: 1,
                height: 1,
                array_layers: 1,
            })
        );
        assert_eq!(dims.mip_level_dimensions(9), None);
    }
}
