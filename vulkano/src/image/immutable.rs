// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferUsage;
use crate::buffer::CpuAccessibleBuffer;
use crate::buffer::TypedBufferAccess;
use crate::command_buffer::AutoCommandBufferBuilder;
use crate::command_buffer::CommandBufferExecFuture;
use crate::command_buffer::CommandBufferUsage;
use crate::command_buffer::PrimaryAutoCommandBuffer;
use crate::command_buffer::PrimaryCommandBuffer;
use crate::device::physical::QueueFamily;
use crate::device::Device;
use crate::device::Queue;
use crate::format::Format;
use crate::format::Pixel;
use crate::image::sys::ImageCreationError;
use crate::image::sys::UnsafeImage;
use crate::image::traits::ImageAccess;
use crate::image::traits::ImageContent;
use crate::image::ImageCreateFlags;
use crate::image::ImageDescriptorLayouts;
use crate::image::ImageDimensions;
use crate::image::ImageInner;
use crate::image::ImageLayout;
use crate::image::ImageUsage;
use crate::image::MipmapsCount;
use crate::memory::pool::AllocFromRequirementsFilter;
use crate::memory::pool::AllocLayout;
use crate::memory::pool::MappingRequirement;
use crate::memory::pool::MemoryPool;
use crate::memory::pool::MemoryPoolAlloc;
use crate::memory::pool::PotentialDedicatedAllocation;
use crate::memory::pool::StdMemoryPoolAlloc;
use crate::memory::DedicatedAlloc;
use crate::sampler::Filter;
use crate::sync::AccessError;
use crate::sync::NowFuture;
use crate::sync::Sharing;
use smallvec::SmallVec;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Image whose purpose is to be used for read-only purposes. You can write to the image once,
/// but then you must only ever read from it.
// TODO: type (2D, 3D, array, etc.) as template parameter
#[derive(Debug)]
pub struct ImmutableImage<A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    image: UnsafeImage,
    dimensions: ImageDimensions,
    memory: A,
    format: Format,
    initialized: AtomicBool,
    layout: ImageLayout,
}

/// Image whose purpose is to access only a part of one image, for any kind of access
/// We define a part of one image here by a level of mipmap, or a layer of an array
/// The image attribute must be an implementation of ImageAccess
/// The mip_levels_access must be a range showing which mipmaps will be accessed
/// The array_layers_access must be a range showing which layers will be accessed
/// The layout must be the layout of the image at the beginning and at the end of the command buffer
pub struct SubImage {
    image: Arc<dyn ImageAccess>,
    mip_levels_access: std::ops::Range<u32>,
    array_layers_access: std::ops::Range<u32>,
    layout: ImageLayout,
}

impl SubImage {
    pub fn new(
        image: Arc<dyn ImageAccess>,
        mip_level: u32,
        mip_level_count: u32,
        layer_level: u32,
        layer_level_count: u32,
        layout: ImageLayout,
    ) -> Arc<SubImage> {
        debug_assert!(mip_level + mip_level_count <= image.mip_levels());
        debug_assert!(layer_level + layer_level_count <= image.dimensions().array_layers());

        let last_level = mip_level + mip_level_count;
        let mip_levels_access = mip_level..last_level;

        let last_level = layer_level + layer_level_count;
        let array_layers_access = layer_level..last_level;

        Arc::new(SubImage {
            image,
            mip_levels_access,
            array_layers_access,
            layout,
        })
    }
}

// Must not implement Clone, as that would lead to multiple `used` values.
pub struct ImmutableImageInitialization<A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    image: Arc<ImmutableImage<A>>,
    used: AtomicBool,
    mip_levels_access: std::ops::Range<u32>,
    array_layers_access: std::ops::Range<u32>,
}

fn has_mipmaps(mipmaps: MipmapsCount) -> bool {
    match mipmaps {
        MipmapsCount::One => false,
        MipmapsCount::Log2 => true,
        MipmapsCount::Specific(x) => x > 1,
    }
}

fn generate_mipmaps<L>(
    cbb: &mut AutoCommandBufferBuilder<L>,
    image: Arc<dyn ImageAccess>,
    dimensions: ImageDimensions,
    layout: ImageLayout,
) {
    for level in 1..image.mip_levels() {
        let [xs, ys, ds] = dimensions
            .mipmap_dimensions(level - 1)
            .unwrap()
            .width_height_depth();
        let [xd, yd, dd] = dimensions
            .mipmap_dimensions(level)
            .unwrap()
            .width_height_depth();

        let src = SubImage::new(
            image.clone(),
            level - 1,
            1,
            0,
            dimensions.array_layers(),
            layout,
        );

        let dst = SubImage::new(
            image.clone(),
            level,
            1,
            0,
            dimensions.array_layers(),
            layout,
        );

        cbb.blit_image(
            src,                               //source
            [0, 0, 0],                         //source_top_left
            [xs as i32, ys as i32, ds as i32], //source_bottom_right
            0,                                 //source_base_array_layer
            level - 1,                         //source_mip_level
            dst,                               //destination
            [0, 0, 0],                         //destination_top_left
            [xd as i32, yd as i32, dd as i32], //destination_bottom_right
            0,                                 //destination_base_array_layer
            level,                             //destination_mip_level
            1,                                 //layer_count
            Filter::Linear,                    //filter
        )
        .expect("failed to blit a mip map to image!");
    }
}

impl ImmutableImage {
    #[deprecated(note = "use ImmutableImage::uninitialized instead")]
    #[inline]
    pub fn new<'a, I>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: Format,
        queue_families: I,
    ) -> Result<Arc<ImmutableImage>, ImageCreationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
    {
        #[allow(deprecated)]
        ImmutableImage::with_mipmaps(
            device,
            dimensions,
            format,
            MipmapsCount::One,
            queue_families,
        )
    }

    #[deprecated(note = "use ImmutableImage::uninitialized instead")]
    #[inline]
    pub fn with_mipmaps<'a, I, M>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: Format,
        mip_levels: M,
        queue_families: I,
    ) -> Result<Arc<ImmutableImage>, ImageCreationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
        M: Into<MipmapsCount>,
    {
        let usage = ImageUsage {
            transfer_source: true, // for blits
            transfer_destination: true,
            sampled: true,
            ..ImageUsage::none()
        };

        let flags = ImageCreateFlags::none();

        let (image, _) = ImmutableImage::uninitialized(
            device,
            dimensions,
            format,
            mip_levels,
            usage,
            flags,
            ImageLayout::ShaderReadOnlyOptimal,
            queue_families,
        )?;
        image.initialized.store(true, Ordering::Relaxed); // Allow uninitialized access for backwards compatibility
        Ok(image)
    }

    /// Builds an uninitialized immutable image.
    ///
    /// Returns two things: the image, and a special access that should be used for the initial upload to the image.
    pub fn uninitialized<'a, I, M>(
        device: Arc<Device>,
        dimensions: ImageDimensions,
        format: Format,
        mip_levels: M,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        layout: ImageLayout,
        queue_families: I,
    ) -> Result<(Arc<ImmutableImage>, Arc<ImmutableImageInitialization>), ImageCreationError>
    where
        I: IntoIterator<Item = QueueFamily<'a>>,
        M: Into<MipmapsCount>,
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let image = UnsafeImage::start(device.clone())
            .flags(flags)
            .dimensions(dimensions)
            .format(format)
            .mip_levels(mip_levels)
            .sharing(if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            })
            .usage(usage)
            .build()?;

        let mem_reqs = image.memory_requirements();
        let memory = MemoryPool::alloc_from_requirements(
            &Device::standard_pool(&device),
            &mem_reqs,
            AllocLayout::Optimal,
            MappingRequirement::DoNotMap,
            DedicatedAlloc::Image(&image),
            |t| {
                if t.is_device_local() {
                    AllocFromRequirementsFilter::Preferred
                } else {
                    AllocFromRequirementsFilter::Allowed
                }
            },
        )?;
        debug_assert!((memory.offset() % mem_reqs.alignment) == 0);
        unsafe {
            image.bind_memory(memory.memory(), memory.offset())?;
        }

        let image = Arc::new(ImmutableImage {
            image,
            memory,
            dimensions,
            format,
            initialized: AtomicBool::new(false),
            layout,
        });

        let init = Arc::new(ImmutableImageInitialization {
            image: image.clone(),
            used: AtomicBool::new(false),
            mip_levels_access: 0..image.mip_levels(),
            array_layers_access: 0..image.dimensions().array_layers(),
        });

        Ok((image, init))
    }

    /// Construct an ImmutableImage from the contents of `iter`.
    #[inline]
    pub fn from_iter<Px, I>(
        iter: I,
        dimensions: ImageDimensions,
        mip_levels: MipmapsCount,
        format: Format,
        queue: Arc<Queue>,
    ) -> Result<
        (
            Arc<Self>,
            CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
        ),
        ImageCreationError,
    >
    where
        Px: Pixel + Send + Sync + Clone + 'static,
        I: IntoIterator<Item = Px>,
        I::IntoIter: ExactSizeIterator,
    {
        let source = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage::transfer_source(),
            false,
            iter,
        )?;
        ImmutableImage::from_buffer(source, dimensions, mip_levels, format, queue)
    }

    /// Construct an ImmutableImage containing a copy of the data in `source`.
    pub fn from_buffer<B, Px>(
        source: Arc<B>,
        dimensions: ImageDimensions,
        mip_levels: MipmapsCount,
        format: Format,
        queue: Arc<Queue>,
    ) -> Result<
        (
            Arc<Self>,
            CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
        ),
        ImageCreationError,
    >
    where
        B: TypedBufferAccess<Content = [Px]> + 'static,
        Px: Pixel + Send + Sync + Clone + 'static,
    {
        let need_to_generate_mipmaps = has_mipmaps(mip_levels);
        let usage = ImageUsage {
            transfer_destination: true,
            transfer_source: need_to_generate_mipmaps,
            sampled: true,
            ..ImageUsage::none()
        };
        let flags = ImageCreateFlags::none();
        let layout = ImageLayout::ShaderReadOnlyOptimal;

        let (image, initializer) = ImmutableImage::uninitialized(
            source.device().clone(),
            dimensions,
            format,
            mip_levels,
            usage,
            flags,
            layout,
            source.device().active_queue_families(),
        )?;

        let init = SubImage::new(initializer, 0, 1, 0, 1, ImageLayout::ShaderReadOnlyOptimal);

        let mut cbb = AutoCommandBufferBuilder::primary(
            source.device().clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )?;
        cbb.copy_buffer_to_image_dimensions(
            source,
            init,
            [0, 0, 0],
            dimensions.width_height_depth(),
            0,
            dimensions.array_layers(),
            0,
        )
        .unwrap();

        if need_to_generate_mipmaps {
            generate_mipmaps(
                &mut cbb,
                image.clone(),
                image.dimensions,
                ImageLayout::ShaderReadOnlyOptimal,
            );
        }

        let cb = cbb.build().unwrap();

        let future = match cb.execute(queue) {
            Ok(f) => f,
            Err(e) => unreachable!("{:?}", e),
        };

        image.initialized.store(true, Ordering::Relaxed);

        Ok((image, future))
    }
}

unsafe impl<A> ImageAccess for ImmutableImage<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn inner(&self) -> ImageInner {
        ImageInner {
            image: &self.image,
            first_layer: 0,
            num_layers: self.image.dimensions().array_layers() as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: self.image.mip_levels() as usize,
        }
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        self.layout
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        self.layout
    }

    #[inline]
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        Some(ImageDescriptorLayouts {
            storage_image: ImageLayout::General,
            combined_image_sampler: self.layout,
            sampled_image: self.layout,
            input_attachment: self.layout,
        })
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(
        &self,
        exclusive_access: bool,
        uninitialized_safe: bool,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        if expected_layout != self.layout && expected_layout != ImageLayout::Undefined {
            return Err(AccessError::UnexpectedImageLayout {
                requested: expected_layout,
                allowed: self.layout,
            });
        }

        if exclusive_access {
            return Err(AccessError::ExclusiveDenied);
        }

        if !self.initialized.load(Ordering::Relaxed) {
            return Err(AccessError::BufferNotInitialized);
        }

        Ok(())
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {}

    #[inline]
    unsafe fn unlock(&self, new_layout: Option<ImageLayout>) {
        debug_assert!(new_layout.is_none());
    }

    #[inline]
    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        0..self.mip_levels()
    }

    #[inline]
    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        0..self.dimensions().array_layers()
    }
}

unsafe impl<P, A> ImageContent<P> for ImmutableImage<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

unsafe impl ImageAccess for SubImage {
    #[inline]
    fn inner(&self) -> ImageInner {
        self.image.inner()
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        self.image.initial_layout_requirement()
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        self.image.final_layout_requirement()
    }

    #[inline]
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        None
    }

    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        self.mip_levels_access.clone()
    }

    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        self.array_layers_access.clone()
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.conflict_key()
    }

    #[inline]
    fn try_gpu_lock(
        &self,
        exclusive_access: bool,
        uninitialized_safe: bool,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        if expected_layout != self.layout && expected_layout != ImageLayout::Undefined {
            return Err(AccessError::UnexpectedImageLayout {
                requested: expected_layout,
                allowed: self.layout,
            });
        }

        Ok(())
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        self.image.increase_gpu_lock()
    }

    #[inline]
    unsafe fn unlock(&self, new_layout: Option<ImageLayout>) {
        self.image.unlock(new_layout)
    }
}

impl<A> PartialEq for ImmutableImage<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl<A> Eq for ImmutableImage<A> where A: MemoryPoolAlloc {}

impl<A> Hash for ImmutableImage<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

unsafe impl<A> ImageAccess for ImmutableImageInitialization<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn inner(&self) -> ImageInner {
        self.image.inner()
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        ImageLayout::Undefined
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        self.image.layout
    }

    #[inline]
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        None
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.image.key()
    }

    #[inline]
    fn try_gpu_lock(
        &self,
        _: bool,
        uninitialized_safe: bool,
        expected_layout: ImageLayout,
    ) -> Result<(), AccessError> {
        if expected_layout != ImageLayout::Undefined {
            return Err(AccessError::UnexpectedImageLayout {
                requested: expected_layout,
                allowed: ImageLayout::Undefined,
            });
        }

        if self.image.initialized.load(Ordering::Relaxed) {
            return Err(AccessError::AlreadyInUse);
        }

        // FIXME: Mipmapped textures require multiple writes to initialize
        if !self
            .used
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .unwrap_or_else(|e| e)
        {
            Ok(())
        } else {
            Err(AccessError::AlreadyInUse)
        }
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        debug_assert!(self.used.load(Ordering::Relaxed));
    }

    #[inline]
    unsafe fn unlock(&self, new_layout: Option<ImageLayout>) {
        assert_eq!(new_layout, Some(self.image.layout));
        self.image.initialized.store(true, Ordering::Relaxed);
    }

    #[inline]
    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        self.mip_levels_access.clone()
    }

    #[inline]
    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        self.array_layers_access.clone()
    }
}

impl<A> PartialEq for ImmutableImageInitialization<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl<A> Eq for ImmutableImageInitialization<A> where A: MemoryPoolAlloc {}

impl<A> Hash for ImmutableImageInitialization<A>
where
    A: MemoryPoolAlloc,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}
