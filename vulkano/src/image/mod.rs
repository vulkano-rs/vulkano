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
//! You can [create a `ResourceMemory` from `DeviceMemory`] if you want to bind its own block of
//! memory to an image.
//!
//! [`ImageView`]: crate::image::view::ImageView
//! [create an `Image` directly]: Image::new
//! [create a `RawImage`]: RawImage::new
//! [bind memory to it]: RawImage::bind_memory
//! [`DeviceMemory`]: crate::memory::DeviceMemory
//! [allocated yourself]: crate::memory::DeviceMemory::allocate
//! [imported]: crate::memory::DeviceMemory::import
//! [create a `ResourceMemory` from `DeviceMemory`]: ResourceMemory::new_dedicated

pub use self::{aspect::*, layout::*, sys::ImageCreateInfo, usage::*};
use self::{sys::RawImage, view::ImageViewType};
use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    format::{Format, FormatFeatures},
    macros::{vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
    memory::{
        allocator::{AllocationCreateInfo, MemoryAllocator, MemoryAllocatorError},
        DedicatedAllocation, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        ExternalMemoryProperties, MemoryRequirements, ResourceMemory,
    },
    range_map::RangeMap,
    swapchain::Swapchain,
    sync::{future::AccessError, AccessConflict, CurrentAccess, Sharing},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    cmp::max,
    error::Error,
    fmt::{Display, Formatter},
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
#[non_exhaustive]
pub enum ImageMemory {
    /// The image is backed by normal memory, bound with [`bind_memory`].
    ///
    /// [`bind_memory`]: RawImage::bind_memory
    Normal(SmallVec<[ResourceMemory; 4]>),

    /// The image is backed by sparse memory, bound with [`bind_sparse`].
    ///
    /// [`bind_sparse`]: crate::device::QueueGuard::bind_sparse
    Sparse(Vec<SparseImageMemoryRequirements>),

    /// The image is backed by memory owned by a [`Swapchain`].
    Swapchain {
        swapchain: Arc<Swapchain>,
        image_index: u32,
    },

    /// The image is backed by external memory not managed by vulkano.
    External,
}

impl Image {
    /// Creates a new uninitialized `Image`.
    pub fn new(
        allocator: Arc<dyn MemoryAllocator>,
        create_info: ImageCreateInfo,
        allocation_info: AllocationCreateInfo,
    ) -> Result<Arc<Self>, Validated<AllocateImageError>> {
        // TODO: adjust the code below to make this safe
        assert!(!create_info.flags.intersects(ImageCreateFlags::DISJOINT));

        let allocation_type = create_info.tiling.into();
        let raw_image =
            RawImage::new(allocator.device().clone(), create_info).map_err(|err| match err {
                Validated::Error(err) => Validated::Error(AllocateImageError::CreateImage(err)),
                Validated::ValidationError(err) => err.into(),
            })?;
        let requirements = raw_image.memory_requirements()[0];

        let allocation = allocator
            .allocate(
                requirements,
                allocation_type,
                allocation_info,
                Some(DedicatedAllocation::Image(&raw_image)),
            )
            .map_err(AllocateImageError::AllocateMemory)?;
        let allocation = unsafe { ResourceMemory::from_allocation(allocator, allocation) };

        let image = raw_image.bind_memory([allocation]).map_err(|(err, _, _)| {
            err.map(AllocateImageError::BindMemory)
                .map_validation(|err| err.add_context("RawImage::bind_memory"))
        })?;

        Ok(Arc::new(image))
    }

    fn from_raw(inner: RawImage, memory: ImageMemory, layout: ImageLayout) -> Self {
        let aspects = inner.format().aspects();
        let aspect_list: SmallVec<[ImageAspect; 4]> = aspects.into_iter().collect();
        let mip_level_size = inner.array_layers() as DeviceSize;
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
    ) -> Result<Self, VulkanError> {
        // Per https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html#_description
        let create_info = ImageCreateInfo {
            flags: swapchain.flags().into(),
            image_type: ImageType::Dim2d,
            format: swapchain.image_format(),
            view_formats: swapchain.image_view_formats().to_vec(),
            extent: [swapchain.image_extent()[0], swapchain.image_extent()[1], 1],
            array_layers: swapchain.image_array_layers(),
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: swapchain.image_usage(),
            stencil_usage: None,
            sharing: swapchain.image_sharing().clone(),
            initial_layout: ImageLayout::Undefined,
            drm_format_modifiers: Vec::new(),
            drm_format_modifier_plane_layouts: Vec::new(),
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            _ne: crate::NonExhaustive(()),
        };

        Ok(Self::from_raw(
            RawImage::from_handle_with_destruction(
                swapchain.device().clone(),
                handle,
                create_info,
                false,
            )?,
            ImageMemory::Swapchain {
                swapchain,
                image_index,
            },
            ImageLayout::PresentSrc,
        ))
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
    ///   `self.format().planes().len()`.
    #[inline]
    pub fn memory_requirements(&self) -> &[MemoryRequirements] {
        self.inner.memory_requirements()
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.inner.flags()
    }

    /// Returns the image type of the image.
    #[inline]
    pub fn image_type(&self) -> ImageType {
        self.inner.image_type()
    }

    /// Returns the image's format.
    #[inline]
    pub fn format(&self) -> Format {
        self.inner.format()
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.inner.format_features()
    }

    /// Returns the formats that an image view created from this image can have.
    #[inline]
    pub fn view_formats(&self) -> &[Format] {
        self.inner.view_formats()
    }

    /// Returns the extent of the image.
    #[inline]
    pub fn extent(&self) -> [u32; 3] {
        self.inner.extent()
    }

    /// Returns the number of array layers in the image.
    #[inline]
    pub fn array_layers(&self) -> u32 {
        self.inner.array_layers()
    }

    /// Returns the number of mip levels in the image.
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
    pub fn stencil_usage(&self) -> Option<ImageUsage> {
        self.inner.stencil_usage()
    }

    /// Returns the sharing the image was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        self.inner.sharing()
    }

    /// If `self.tiling()` is `ImageTiling::DrmFormatModifier`, returns the DRM format modifier
    /// of the image, and the number of memory planes.
    /// This was either provided in [`ImageCreateInfo::drm_format_modifiers`], or if
    /// multiple modifiers were provided, selected from the list by the Vulkan implementation.
    #[inline]
    pub fn drm_format_modifier(&self) -> Option<(u64, u32)> {
        self.inner.drm_format_modifier()
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
    ) -> Result<SubresourceLayout, Box<ValidationError>> {
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
        assert!(self.format().aspects().contains(subresource_range.aspects));
        assert!(subresource_range.mip_levels.end <= self.mip_levels());
        assert!(subresource_range.array_layers.end <= self.array_layers());

        SubresourceRangeIterator::new(
            subresource_range,
            &self.aspect_list,
            self.aspect_size,
            self.mip_levels(),
            self.mip_level_size,
            self.array_layers(),
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
                array_layers: 0..self.array_layers(),
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
                    array_layers: 0..self.array_layers(),
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
            ImageMemory::Normal(..) | ImageMemory::Sparse(..) | ImageMemory::External => {
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
            ImageMemory::Normal(..) | ImageMemory::Sparse(..) | ImageMemory::External => {
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

/// Error that can happen when allocating a new image.
#[derive(Clone, Debug)]
pub enum AllocateImageError {
    CreateImage(VulkanError),
    AllocateMemory(MemoryAllocatorError),
    BindMemory(VulkanError),
}

impl Error for AllocateImageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CreateImage(err) => Some(err),
            Self::AllocateMemory(err) => Some(err),
            Self::BindMemory(err) => Some(err),
        }
    }
}

impl Display for AllocateImageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateImage(_) => write!(f, "creating the image failed"),
            Self::AllocateMemory(_) => write!(f, "allocating memory for the image failed"),
            Self::BindMemory(_) => write!(f, "binding memory to the image failed"),
        }
    }
}

impl From<AllocateImageError> for Validated<AllocateImageError> {
    fn from(err: AllocateImageError) -> Self {
        Self::Error(err)
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

    /// Flags specifying additional properties of an image.
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
    /// Depending on the image type, either the [`sparse_residency_image2_d`] or the
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
    /// 2D image views created from 3D images with this flag cannot be written to a
    /// descriptor set and accessed in shaders, but can be used as a framebuffer attachment.
    /// To write such an image view to a descriptor set, use the [`DIM2D_VIEW_COMPATIBLE`] flag.
    ///
    /// On [portability subset] devices, the [`image_view2_d_on3_d_image`] feature must be enabled
    /// on the device.
    ///
    /// [`ImageViewType::Dim2d`]: crate::image::view::ImageViewType::Dim2d
    /// [`ImageViewType::Dim2dArray`]: crate::image::view::ImageViewType::Dim2dArray
    /// [`DIM2D_VIEW_COMPATIBLE`]: ImageCreateFlags::DIM2D_VIEW_COMPATIBLE
    /// [portability subset]: crate::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`image_view2_d_on3_d_image`]: crate::device::Features::image_view2_d_on3_d_image
    DIM2D_ARRAY_COMPATIBLE = TYPE_2D_ARRAY_COMPATIBLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance1)]),
    ]),

    /// For images with a compressed format, whether an image view with an uncompressed
    /// format can be created from the image, where each texel in the view will correspond to a
    /// compressed texel block in the image.
    ///
    /// Requires `MUTABLE_FORMAT`.
    BLOCK_TEXEL_VIEW_COMPATIBLE = BLOCK_TEXEL_VIEW_COMPATIBLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance2)]),
    ]),

    /// If `MUTABLE_FORMAT` is also enabled, allows specifying a `usage` for the image that is not
    /// supported by the `format` of the image, as long as there is a format that does support the
    /// usage, that an image view created from the image can have.
    EXTENDED_USAGE = EXTENDED_USAGE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance2)]),
    ]),

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

    /// For 3D images, whether an image view of type [`ImageViewType::Dim2d`]
    /// (but not [`ImageViewType::Dim2dArray`]) can be created from the image.
    ///
    /// Unlike [`DIM2D_ARRAY_COMPATIBLE`], 2D image views created from 3D images with this flag
    /// can be written to a descriptor set and accessed in shaders. To do this,
    /// a feature must also be enabled, depending on the descriptor type:
    /// - [`image2_d_view_of3_d`] for storage images.
    /// - [`sampler2_d_view_of3_d`] for sampled images.
    ///
    /// [`ImageViewType::Dim2d`]: crate::image::view::ImageViewType::Dim2d
    /// [`DIM2D_ARRAY_COMPATIBLE`]: ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE
    /// [`image2_d_view_of3_d`]: crate::device::Features::image2_d_view_of3_d
    /// [`sampler2_d_view_of3_d`]: crate::device::Features::sampler2_d_view_of3_d
    DIM2D_VIEW_COMPATIBLE = TYPE_2D_VIEW_COMPATIBLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_2d_view_of_3d)]),
    ]),

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
    SampleCounts,

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

impl SampleCounts {
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

    /// The basic dimensionality of an image.
    ImageType = ImageType(i32);

    /// A one-dimensional image, consisting of only a width, with a height and depth of 1.
    Dim1d = TYPE_1D,

    /// A two-dimensional image, consisting of a width and height, with a depth of 1.
    Dim2d = TYPE_2D,

    /// A three-dimensional image, consisting of a width, height and depth.
    Dim3d = TYPE_3D,
}

vulkan_enum! {
    #[non_exhaustive]

    /// The arrangement of texels or texel blocks in an image.
    ImageTiling = ImageTiling(i32);

    /// The arrangement is optimized for access in an implementation-defined way.
    ///
    /// This layout is opaque to the user, and cannot be queried. Data can only be read from or
    /// written to the image by using Vulkan commands, such as copy commands.
    Optimal = OPTIMAL,

    /// The texels are laid out in row-major order. This allows easy access by the user, but
    /// is much slower for the device, so it should be used only in specific situations that call
    /// for it.
    ///
    /// You can query the layout by calling [`Image::subresource_layout`].
    Linear = LINEAR,

    /// The tiling is defined by a Linux DRM format modifier associated with the image.
    ///
    /// You can query the layout by calling [`Image::subresource_layout`].
    DrmFormatModifier = DRM_FORMAT_MODIFIER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_drm_format_modifier)]),
    ]),
}

/// Returns the maximum number of mipmap levels for the given image extent.
///
/// The returned value is always at least 1.
///
/// # Examples
///
/// ```
/// use vulkano::image::max_mip_levels;
///
/// assert_eq!(max_mip_levels([32, 50, 1]), 6);
/// ```
#[inline]
pub fn max_mip_levels(extent: [u32; 3]) -> u32 {
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#resources-image-mip-level-sizing
    //
    // This calculates `floor(log2(max(width, height, depth))) + 1` using fast integer operations.
    32 - (extent[0] | extent[1] | extent[2]).leading_zeros()
}

/// Returns the extent of the `level`th mipmap level.
/// If `level` is 0, then it returns `extent` back unchanged.
///
/// Returns `None` if `level` is not less than `max_mip_levels(extent)`.
///
/// # Examples
///
/// ```
/// use vulkano::image::mip_level_extent;
///
/// let extent = [963, 256, 1];
///
/// assert_eq!(mip_level_extent(extent, 0), Some(extent));
/// assert_eq!(mip_level_extent(extent, 1), Some([481, 128, 1]));
/// assert_eq!(mip_level_extent(extent, 6), Some([15, 4, 1]));
/// assert_eq!(mip_level_extent(extent, 9), Some([1, 1, 1]));
/// assert_eq!(mip_level_extent(extent, 11), None);
/// ```
///
/// # Panics
///
/// - In debug mode, panics if `extent` contains 0.
///   In release, returns an unspecified value.
#[inline]
pub fn mip_level_extent(extent: [u32; 3], level: u32) -> Option<[u32; 3]> {
    if level == 0 {
        return Some(extent);
    }

    if level >= max_mip_levels(extent) {
        return None;
    }

    Some(extent.map(|x| {
        debug_assert!(x != 0);
        max(1, x >> level)
    }))
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
    /// mip level of the image. All aspects of the image are selected, or `PLANE_0` if the image
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

impl ImageSubresourceLayers {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            aspects,
            mip_level: _,
            ref array_layers,
        } = self;

        aspects.validate_device(device).map_err(|err| {
            err.add_context("aspects")
                .set_vuids(&["VUID-VkImageSubresourceLayers-aspectMask-parameter"])
        })?;

        if aspects.is_empty() {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageSubresourceLayers-aspectMask-requiredbitmask"],
                ..Default::default()
            }));
        }

        if aspects.intersects(ImageAspects::COLOR)
            && aspects.intersects(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "contains both `ImageAspects::COLOR`, and either `ImageAspects::DEPTH` \
                    or `ImageAspects::STENCIL`"
                    .into(),
                vuids: &["VUID-VkImageSubresourceLayers-aspectMask-00167"],
                ..Default::default()
            }));
        }

        if aspects.intersects(ImageAspects::METADATA) {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "contains `ImageAspects::METADATA`".into(),
                vuids: &["VUID-VkImageSubresourceLayers-aspectMask-00168"],
                ..Default::default()
            }));
        }

        if aspects.intersects(
            ImageAspects::MEMORY_PLANE_0
                | ImageAspects::MEMORY_PLANE_1
                | ImageAspects::MEMORY_PLANE_2
                | ImageAspects::MEMORY_PLANE_3,
        ) {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "contains `ImageAspects::MEMORY_PLANE_0`, \
                    `ImageAspects::MEMORY_PLANE_1`, `ImageAspects::MEMORY_PLANE_2` or \
                    `ImageAspects::MEMORY_PLANE_3`"
                    .into(),
                vuids: &["VUID-VkImageSubresourceLayers-aspectMask-02247"],
                ..Default::default()
            }));
        }

        if array_layers.is_empty() {
            return Err(Box::new(ValidationError {
                context: "array_layers".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageSubresourceLayers-layerCount-01700"],
                ..Default::default()
            }));
        }

        Ok(())
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            aspects,
            ref mip_levels,
            ref array_layers,
        } = self;

        aspects.validate_device(device).map_err(|err| {
            err.add_context("aspects")
                .set_vuids(&["VUID-VkImageSubresourceRange-aspectMask-parameter"])
        })?;

        if aspects.is_empty() {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageSubresourceRange-aspectMask-requiredbitmask"],
                ..Default::default()
            }));
        }

        if mip_levels.is_empty() {
            return Err(Box::new(ValidationError {
                context: "mip_levels".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageSubresourceRange-levelCount-01720"],
                ..Default::default()
            }));
        }

        if array_layers.is_empty() {
            return Err(Box::new(ValidationError {
                context: "array_layers".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageSubresourceRange-layerCount-01721"],
                ..Default::default()
            }));
        }

        if aspects.intersects(ImageAspects::COLOR)
            && aspects
                .intersects(ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2)
        {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "contains both `ImageAspects::COLOR`, and one of \
                    `ImageAspects::PLANE_0`, `ImageAspects::PLANE_1` or `ImageAspects::PLANE_2`"
                    .into(),
                vuids: &["VUID-VkImageSubresourceRange-aspectMask-01670"],
                ..Default::default()
            }));
        }

        if aspects.intersects(
            ImageAspects::MEMORY_PLANE_0
                | ImageAspects::MEMORY_PLANE_1
                | ImageAspects::MEMORY_PLANE_2
                | ImageAspects::MEMORY_PLANE_3,
        ) {
            return Err(Box::new(ValidationError {
                context: "aspects".into(),
                problem: "contains `ImageAspects::MEMORY_PLANE_0`, \
                    `ImageAspects::MEMORY_PLANE_1`, `ImageAspects::MEMORY_PLANE_2` or \
                    `ImageAspects::MEMORY_PLANE_3`"
                    .into(),
                vuids: &["VUID-VkImageSubresourceLayers-aspectMask-02247"],
                ..Default::default()
            }));
        }

        Ok(())
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
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

    /// The image view formats that will be allowed for the image.
    ///
    /// If this is not empty, then the physical device API version must be at least 1.2, or the
    /// [`khr_image_format_list`] extension must be supported by the physical device.
    ///
    /// The default value is empty.
    ///
    /// [`khr_image_format_list`]: crate::device::DeviceExtensions::khr_image_format_list
    pub view_formats: Vec<Format>,

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

    /// The `stencil_usage` that the image will have, if different from the regular `usage`.
    ///
    /// If this is `Some`, then the physical device API version must be at least 1.2, or the
    /// [`ext_separate_stencil_usage`](crate::device::DeviceExtensions::ext_separate_stencil_usage)
    /// extension must be supported by the physical device.
    ///
    /// The default value is `None`.
    pub stencil_usage: Option<ImageUsage>,

    /// The Linux DRM format modifier information to query.
    ///
    /// If this is `Some`, then the
    /// [`ext_image_drm_format_modifier`](crate::device::DeviceExtensions::ext_image_drm_format_modifier)
    /// extension must be supported by the physical device.
    ///
    /// The default value is `None`.
    pub drm_format_modifier_info: Option<ImageDrmFormatModifierInfo>,

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
            format: Format::UNDEFINED,
            view_formats: Vec::new(),
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            stencil_usage: None,
            drm_format_modifier_info: None,
            external_memory_handle_type: None,
            image_view_type: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl ImageFormatInfo {
    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            format,
            ref view_formats,
            image_type,
            tiling,
            usage,
            stencil_usage,
            ref drm_format_modifier_info,
            external_memory_handle_type,
            image_view_type,
            _ne: _,
        } = self;

        flags
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("flags")
                    .set_vuids(&["VUID-VkPhysicalDeviceImageFormatInfo2-flags-parameter"])
            })?;

        format
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("format")
                    .set_vuids(&["VUID-VkPhysicalDeviceImageFormatInfo2-format-parameter"])
            })?;

        image_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("image_type")
                    .set_vuids(&["VUID-VkPhysicalDeviceImageFormatInfo2-imageType-parameter"])
            })?;

        tiling
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("tiling")
                    .set_vuids(&["VUID-VkPhysicalDeviceImageFormatInfo2-tiling-parameter"])
            })?;

        usage
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("usage")
                    .set_vuids(&["VUID-VkPhysicalDeviceImageFormatInfo2-usage-parameter"])
            })?;

        if usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-usage-requiredbitmask"],
                ..Default::default()
            }));
        }

        if let Some(stencil_usage) = stencil_usage {
            if !(physical_device.api_version() >= Version::V1_2
                || physical_device
                    .supported_extensions()
                    .ext_separate_stencil_usage)
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("ext_separate_stencil_usage")]),
                    ]),
                    ..Default::default()
                }));
            }

            stencil_usage
                .validate_physical_device(physical_device)
                .map_err(|err| {
                    err.add_context("stencil_usage")
                        .set_vuids(&["VUID-VkImageStencilUsageCreateInfo-stencilUsage-parameter"])
                })?;

            if stencil_usage.is_empty() {
                return Err(Box::new(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "is empty".into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-usage-requiredbitmask"],
                    ..Default::default()
                }));
            }

            if stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                && !(stencil_usage
                    - (ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT))
                    .is_empty()
            {
                return Err(Box::new(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "contains `ImageUsage::TRANSIENT_ATTACHMENT`, but also contains \
                        usages other than `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or \
                        `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-stencilUsage-02539"],
                    ..Default::default()
                }));
            }
        }

        if !view_formats.is_empty() {
            if !(physical_device.api_version() >= Version::V1_2
                || physical_device.supported_extensions().khr_image_format_list)
            {
                return Err(Box::new(ValidationError {
                    context: "view_formats".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_image_format_list")]),
                    ]),
                    ..Default::default()
                }));
            }

            for (index, view_format) in view_formats.iter().enumerate() {
                view_format
                    .validate_physical_device(physical_device)
                    .map_err(|err| {
                        err.add_context(format!("view_formats[{}]", index))
                            .set_vuids(&["VUID-VkImageFormatListCreateInfo-pViewFormats-parameter"])
                    })?;
            }
        }

        if let Some(drm_format_modifier_info) = drm_format_modifier_info {
            if !physical_device
                .supported_extensions()
                .ext_image_drm_format_modifier
            {
                return Err(Box::new(ValidationError {
                    context: "drm_format_modifier_info".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_image_drm_format_modifier",
                    )])]),
                    ..Default::default()
                }));
            }

            drm_format_modifier_info
                .validate(physical_device)
                .map_err(|err| err.add_context("drm_format_modifier_info"))?;

            if tiling != ImageTiling::DrmFormatModifier {
                return Err(Box::new(ValidationError {
                    problem: "`drm_format_modifier_info` is `Some` but \
                        `tiling` is not `ImageTiling::DrmFormatModifier`"
                        .into(),
                    vuids: &[" VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02249"],
                    ..Default::default()
                }));
            }

            if flags.intersects(ImageCreateFlags::MUTABLE_FORMAT) && view_formats.is_empty() {
                return Err(Box::new(ValidationError {
                    problem: "`tiling` is `ImageTiling::DrmFormatModifier`, and \
                        `flags` contains `ImageCreateFlags::MUTABLE_FORMAT`, but \
                        `view_formats` is empty"
                        .into(),
                    vuids: &["VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02313"],
                    ..Default::default()
                }));
            }
        } else if tiling == ImageTiling::DrmFormatModifier {
            return Err(Box::new(ValidationError {
                problem: "`tiling` is `ImageTiling::DrmFormatModifier`, but \
                    `drm_format_modifier_info` is `None`"
                    .into(),
                vuids: &[" VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02249"],
                ..Default::default()
            }));
        }

        if let Some(handle_type) = external_memory_handle_type {
            if !(physical_device.api_version() >= Version::V1_1
                || physical_device
                    .instance()
                    .enabled_extensions()
                    .khr_external_memory_capabilities)
            {
                return Err(Box::new(ValidationError {
                    problem: "`external_memory_handle_type` is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::InstanceExtension(
                            "khr_external_memory_capabilities",
                        )]),
                    ]),
                    ..Default::default()
                }));
            }

            handle_type
                .validate_physical_device(physical_device)
                .map_err(|err| {
                    err.add_context("handle_type").set_vuids(&[
                        "VUID-VkPhysicalDeviceExternalImageFormatInfo-handleType-parameter",
                    ])
                })?;
        }

        if let Some(image_view_type) = image_view_type {
            if !physical_device.supported_extensions().ext_filter_cubic {
                return Err(Box::new(ValidationError {
                    problem: "`image_view_type` is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_filter_cubic",
                    )])]),
                    ..Default::default()
                }));
            }

            image_view_type
                .validate_physical_device(physical_device)
                .map_err(|err| {
                    err.add_context("image_view_type").set_vuids(&[
                        "VUID-VkPhysicalDeviceImageViewImageFormatInfoEXT-imageViewType-parameter",
                    ])
                })?;
        }

        Ok(())
    }
}

/// The image's DRM format modifier configuration to query in
/// [`PhysicalDevice::image_format_properties`](crate::device::physical::PhysicalDevice::image_format_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageDrmFormatModifierInfo {
    /// The DRM format modifier to query.
    ///
    /// The default value is 0.
    pub drm_format_modifier: u64,

    /// Whether the image can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<SmallVec<[u32; 4]>>,

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageDrmFormatModifierInfo {
    #[inline]
    fn default() -> Self {
        Self {
            drm_format_modifier: 0,
            sharing: Sharing::Exclusive,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl ImageDrmFormatModifierInfo {
    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            drm_format_modifier: _,
            ref sharing,
            _ne,
        } = self;

        match sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                if queue_family_indices.len() < 2 {
                    return Err(Box::new(ValidationError {
                        context: "sharing".into(),
                        problem: "is `Sharing::Concurrent`, but contains less than 2 elements"
                            .into(),
                        vuids: &[
                            "VUID-VkPhysicalDeviceImageDrmFormatModifierInfoEXT-sharingMode-02315",
                        ],
                        ..Default::default()
                    }));
                }

                let queue_family_count = physical_device.queue_family_properties().len() as u32;

                for (index, &queue_family_index) in queue_family_indices.iter().enumerate() {
                    if queue_family_indices[..index].contains(&queue_family_index) {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_indices".into(),
                            problem: format!(
                                "the queue family index in the list at index {} is contained in \
                                the list more than once",
                                index,
                            )
                            .into(),
                            vuids: &[" VUID-VkPhysicalDeviceImageDrmFormatModifierInfoEXT-sharingMode-02316"],
                            ..Default::default()
                        }));
                    }

                    if queue_family_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: format!("sharing[{}]", index).into(),
                            problem: "is not less than the number of queue families in the device"
                                .into(),
                            vuids: &[" VUID-VkPhysicalDeviceImageDrmFormatModifierInfoEXT-sharingMode-02316"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }
}

/// The properties that are supported by a physical device for images of a certain type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImageFormatProperties {
    /// The maximum image extent.
    pub max_extent: [u32; 3],

    /// The maximum number of mip levels.
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
    /// a [`Cubic`](crate::image::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    pub filter_cubic: bool,

    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::image::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::image::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::image::sampler::SamplerReductionMode::Max) `reduction_mode`.
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
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

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
            format: Format::UNDEFINED,
            image_type: ImageType::Dim2d,
            samples: SampleCount::Sample1,
            usage: ImageUsage::empty(),
            tiling: ImageTiling::Optimal,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SparseImageFormatInfo {
    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            format,
            image_type,
            samples,
            usage,
            tiling,
            _ne: _,
        } = self;

        format
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("format")
                    .set_vuids(&["VUID-VkPhysicalDeviceSparseImageFormatInfo2-format-parameter"])
            })?;

        image_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("image_type")
                    .set_vuids(&["VUID-VkPhysicalDeviceSparseImageFormatInfo2-type-parameter"])
            })?;

        samples
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("samples")
                    .set_vuids(&["VUID-VkPhysicalDeviceSparseImageFormatInfo2-samples-parameter"])
            })?;

        usage
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("usage")
                    .set_vuids(&["VUID-VkPhysicalDeviceSparseImageFormatInfo2-usage-parameter"])
            })?;

        if usage.is_empty() {
            return Err(Box::new(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPhysicalDeviceSparseImageFormatInfo2-usage-requiredbitmask"],
                ..Default::default()
            }));
        }

        tiling
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("tiling")
                    .set_vuids(&["VUID-VkPhysicalDeviceSparseImageFormatInfo2-tiling-parameter"])
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
    #[test]
    fn max_mip_levels() {
        assert_eq!(super::max_mip_levels([2, 1, 1]), 2);
        assert_eq!(super::max_mip_levels([2, 3, 1]), 2);
        assert_eq!(super::max_mip_levels([512, 512, 1]), 10);
    }

    #[test]
    fn mip_level_size() {
        let extent = [283, 175, 1];
        assert_eq!(super::mip_level_extent(extent, 0), Some(extent));
        assert_eq!(super::mip_level_extent(extent, 1), Some([141, 87, 1]));
        assert_eq!(super::mip_level_extent(extent, 2), Some([70, 43, 1]));
        assert_eq!(super::mip_level_extent(extent, 3), Some([35, 21, 1]));
        assert_eq!(super::mip_level_extent(extent, 4), Some([17, 10, 1]));
        assert_eq!(super::mip_level_extent(extent, 5), Some([8, 5, 1]));
        assert_eq!(super::mip_level_extent(extent, 6), Some([4, 2, 1]));
        assert_eq!(super::mip_level_extent(extent, 7), Some([2, 1, 1]));
        assert_eq!(super::mip_level_extent(extent, 8), Some([1, 1, 1]));
        assert_eq!(super::mip_level_extent(extent, 9), None);
    }
}
