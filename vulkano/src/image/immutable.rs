// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use buffer::BufferAccess;
use buffer::BufferUsage;
use buffer::CpuAccessibleBuffer;
use buffer::TypedBufferAccess;
use command_buffer::AutoCommandBuffer;
use command_buffer::AutoCommandBufferBuilder;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecFuture;
use device::Device;
use device::Queue;
use format::AcceptsPixels;
use format::Format;
use format::FormatDesc;
use image::sys::ImageCreationError;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::traits::ImageAccess;
use image::traits::ImageContent;
use image::traits::ImageViewAccess;
use image::Dimensions;
use image::ImageInner;
use image::ImageLayout;
use image::ImageUsage;
use image::MipmapsCount;
use instance::QueueFamily;
use memory::pool::AllocFromRequirementsFilter;
use memory::pool::AllocLayout;
use memory::pool::MappingRequirement;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::PotentialDedicatedAllocation;
use memory::pool::StdMemoryPoolAlloc;
use memory::DedicatedAlloc;
use sampler::Filter;
use sync::AccessError;
use sync::NowFuture;
use sync::Sharing;

/// Image whose purpose is to be used for read-only purposes. You can write to the image once,
/// but then you must only ever read from it.
// TODO: type (2D, 3D, array, etc.) as template parameter
#[derive(Debug)]
pub struct ImmutableImage<F, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    image: UnsafeImage,
    view: UnsafeImageView,
    dimensions: Dimensions,
    memory: A,
    format: F,
    initialized: AtomicBool,
    layout: ImageLayout,
}


/// Image whose purpose is to access only a part of one image, for any kind of access
/// We define a part of one image here by a level of mipmap, or a layer of an array
/// The image attribute must be an implementation of ImageAccess
/// The mip_levels_access must be a range showing which mipmaps will be accessed
/// The layer_levels_access must be a range showing which layers will be accessed
/// The layout must be the layout of the image at the beginning and at the end of the command buffer
pub struct SubImage {
    image: Arc<dyn ImageAccess + Sync + Send>,
    mip_levels_access: std::ops::Range<u32>,
    layer_levels_access: std::ops::Range<u32>,
    layout: ImageLayout,
}

impl SubImage
{
    pub fn new(
        image: Arc<dyn ImageAccess + Sync + Send>,
        mip_level: u32,
        mip_level_count: u32,
        layer_level: u32,
        layer_level_count: u32,
        layout: ImageLayout,
    ) -> Arc<SubImage> {
        debug_assert!(mip_level + mip_level_count <= image.mipmap_levels());
        debug_assert!(layer_level + layer_level_count <= image.dimensions().array_layers());

        let last_level = mip_level + mip_level_count;
        let mip_levels_access = mip_level..last_level;

        let last_level = layer_level + layer_level_count;
        let layer_levels_access = layer_level..last_level;

        Arc::new(SubImage {
            image,
            mip_levels_access,
            layer_levels_access,
            layout: ImageLayout::ShaderReadOnlyOptimal,
        })
    }
}

// Must not implement Clone, as that would lead to multiple `used` values.
pub struct ImmutableImageInitialization<F, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
    image: Arc<ImmutableImage<F, A>>,
    used: AtomicBool,
    mip_levels_access: std::ops::Range<u32>,
    layer_levels_access: std::ops::Range<u32>,
}

fn has_mipmaps(mipmaps: MipmapsCount) -> bool {
    match mipmaps {
        MipmapsCount::One => false,
        MipmapsCount::Log2 => true,
        MipmapsCount::Specific(x) => x > 1
    }
}

fn generate_mipmaps<Img>(
    cbb: &mut AutoCommandBufferBuilder,
    image: Arc<Img>,
    dimensions: Dimensions,
    layout: ImageLayout,
) where
    Img: ImageAccess + Send + Sync + 'static,
{
    let img_dim = dimensions.to_image_dimensions();
    for level in 1..image.mipmap_levels() {
        let [xs, ys, ds] = img_dim
            .mipmap_dimensions(level - 1)
            .unwrap()
            .width_height_depth();
        let [xd, yd, dd] = img_dim
            .mipmap_dimensions(level)
            .unwrap()
            .width_height_depth();

        let src = SubImage::new(
            image.clone(),
            level - 1,
            1,
            0,
            img_dim.array_layers(),
            layout,
        );

        let dst = SubImage::new(image.clone(), level, 1, 0, img_dim.array_layers(), layout);

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

impl<F> ImmutableImage<F> {
    #[deprecated(note = "use ImmutableImage::uninitialized instead")]
    #[inline]
    pub fn new<'a, I>(
        device: Arc<Device>,
        dimensions: Dimensions,
        format: F,
        queue_families: I,
    ) -> Result<Arc<ImmutableImage<F>>, ImageCreationError>
    where
        F: FormatDesc,
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
        dimensions: Dimensions,
        format: F,
        mipmaps: M,
        queue_families: I,
    ) -> Result<Arc<ImmutableImage<F>>, ImageCreationError>
    where
        F: FormatDesc,
        I: IntoIterator<Item = QueueFamily<'a>>,
        M: Into<MipmapsCount>,
    {
        let usage = ImageUsage {
            transfer_source: true, // for blits
            transfer_destination: true,
            sampled: true,
            ..ImageUsage::none()
        };

        let (image, _) = ImmutableImage::uninitialized(
            device,
            dimensions,
            format,
            mipmaps,
            usage,
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
        dimensions: Dimensions,
        format: F,
        mipmaps: M,
        usage: ImageUsage,
        layout: ImageLayout,
        queue_families: I,
    ) -> Result<(Arc<ImmutableImage<F>>, ImmutableImageInitialization<F>), ImageCreationError>
    where
        F: FormatDesc,
        I: IntoIterator<Item = QueueFamily<'a>>,
        M: Into<MipmapsCount>,
    {
        let queue_families = queue_families
            .into_iter()
            .map(|f| f.id())
            .collect::<SmallVec<[u32; 4]>>();

        let (image, mem_reqs) = unsafe {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            UnsafeImage::new(
                device.clone(),
                usage,
                format.format(),
                dimensions.to_image_dimensions(),
                1,
                mipmaps,
                sharing,
                false,
                false,
            )?
        };

        let mem = MemoryPool::alloc_from_requirements(
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
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        unsafe {
            image.bind_memory(mem.memory(), mem.offset())?;
        }

        let view = unsafe {
            UnsafeImageView::raw(
                &image,
                dimensions.to_view_type(),
                0..image.mipmap_levels(),
                0..image.dimensions().array_layers(),
            )?
        };

        let image = Arc::new(ImmutableImage {
            image: image,
            view: view,
            memory: mem,
            dimensions: dimensions,
            format: format,
            initialized: AtomicBool::new(false),
            layout: layout,
        });

        let init = ImmutableImageInitialization {
            image: image.clone(),
            used: AtomicBool::new(false),
            mip_levels_access: 0..image.mipmap_levels(),
            layer_levels_access: 0..image.dimensions().array_layers(),
        };

        Ok((image, init))
    }

    /// Construct an ImmutableImage from the contents of `iter`.
    #[inline]
    pub fn from_iter<P, I>(
        iter: I,
        dimensions: Dimensions,
        mipmaps: MipmapsCount,
        format: F,
        queue: Arc<Queue>,
    ) -> Result<
        (
            Arc<Self>,
            CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
        ),
        ImageCreationError,
    >
    where
        P: Send + Sync + Clone + 'static,
        F: FormatDesc + AcceptsPixels<P> + 'static + Send + Sync,
        I: ExactSizeIterator<Item = P>,
        Format: AcceptsPixels<P>,
    {
        let source = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage::transfer_source(),
            false,
            iter,
        )?;
        ImmutableImage::from_buffer(source, dimensions, mipmaps, format, queue)
    }

    /// Construct an ImmutableImage containing a copy of the data in `source`.
    pub fn from_buffer<B, P>(
        source: B,
        dimensions: Dimensions,
        mipmaps: MipmapsCount,
        format: F,
        queue: Arc<Queue>,
    ) -> Result<
        (
            Arc<Self>,
            CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
        ),
        ImageCreationError,
    >
    where
        B: BufferAccess + TypedBufferAccess<Content = [P]> + 'static + Clone + Send + Sync,
        P: Send + Sync + Clone + 'static,
        F: FormatDesc + AcceptsPixels<P> + 'static + Send + Sync,
        Format: AcceptsPixels<P>,
    {
        let need_to_generate_mipmaps = has_mipmaps(mipmaps);
        let usage = ImageUsage {
            transfer_destination: true,
            transfer_source: need_to_generate_mipmaps,
            sampled: true,
            ..ImageUsage::none()
        };
        let layout = ImageLayout::ShaderReadOnlyOptimal;

        let (image, initializer) = ImmutableImage::uninitialized(
            source.device().clone(),
            dimensions,
            format,
            mipmaps,
            usage,
            layout,
            source.device().active_queue_families(),
        )?;

        let init = SubImage::new(
            Arc::new(initializer),
            0,
            1,
            0,
            1,
            ImageLayout::ShaderReadOnlyOptimal,
        );

        let mut cbb = AutoCommandBufferBuilder::new(source.device().clone(), queue.family())?;
        cbb.copy_buffer_to_image_dimensions(
            source,
            init,
            [0, 0, 0],
            dimensions.width_height_depth(),
            0,
            dimensions.array_layers_with_cube(),
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
            Err(e) => unreachable!("{:?}", e)
        };

        image.initialized.store(true, Ordering::Relaxed);

        Ok((image, future))
    }
}

impl<F, A> ImmutableImage<F, A> {
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    /// Returns the number of mipmap levels of the image.
    #[inline]
    pub fn mipmap_levels(&self) -> u32 {
        self.image.mipmap_levels()
    }
}

unsafe impl<F, A> ImageAccess for ImmutableImage<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn inner(&self) -> ImageInner {
        ImageInner {
            image: &self.image,
            first_layer: 0,
            num_layers: self.image.dimensions().array_layers() as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: self.image.mipmap_levels() as usize,
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
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        self.conflict_key() == other.conflict_key() // TODO:
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.key()
    }

    #[inline]
    fn try_gpu_lock(
        &self,
        exclusive_access: bool,
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
    fn current_miplevels_access(&self) -> std::ops::Range<u32> {
        0..self.mipmap_levels()
    }

    #[inline]
    fn current_layer_levels_access(&self) -> std::ops::Range<u32> {
        0..self.dimensions().array_layers()
    }
}

unsafe impl<P, F, A> ImageContent<P> for ImmutableImage<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

unsafe impl<F: 'static, A> ImageViewAccess for ImmutableImage<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn parent(&self) -> &dyn ImageAccess {
        self
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        &self.view
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self) -> ImageLayout {
        self.layout
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout {
        self.layout
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> ImageLayout {
        self.layout
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> ImageLayout {
        self.layout
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}

unsafe impl ImageAccess for SubImage
{
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
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        self.conflict_key() == other.conflict_key()
            && self.current_miplevels_access() == other.current_miplevels_access()
            && self.current_layer_levels_access() == other.current_layer_levels_access()
    }

    fn current_miplevels_access(&self) -> std::ops::Range<u32> {
        self.mip_levels_access.clone()
    }

    fn current_layer_levels_access(&self) -> std::ops::Range<u32> {
        self.layer_levels_access.clone()
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.conflict_key()
    }

    #[inline]
    fn try_gpu_lock(
        &self,
        exclusive_access: bool,
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

impl<F, A> PartialEq for ImmutableImage<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ImageAccess::inner(self) == ImageAccess::inner(other)
    }
}

impl<F, A> Eq for ImmutableImage<F, A> where F: 'static + Send + Sync {}

impl<F, A> Hash for ImmutableImage<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        ImageAccess::inner(self).hash(state);
    }
}

unsafe impl<F, A> ImageAccess for ImmutableImageInitialization<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn inner(&self) -> ImageInner {
        ImageAccess::inner(&self.image)
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
    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
        false
    }

    #[inline]
    fn conflicts_image(&self, other: &dyn ImageAccess) -> bool {
        self.conflict_key() == other.conflict_key() // TODO:
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.image.key()
    }

    #[inline]
    fn try_gpu_lock(&self, _: bool, expected_layout: ImageLayout) -> Result<(), AccessError> {
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
        if !self.used.compare_and_swap(false, true, Ordering::Relaxed) {
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
    fn current_miplevels_access(&self) -> std::ops::Range<u32> {
        self.mip_levels_access.clone()
    }

    #[inline]
    fn current_layer_levels_access(&self) -> std::ops::Range<u32> {
        self.layer_levels_access.clone()
    }
}

impl<F, A> PartialEq for ImmutableImageInitialization<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ImageAccess::inner(self) == ImageAccess::inner(other)
    }
}

impl<F, A> Eq for ImmutableImageInitialization<F, A> where F: 'static + Send + Sync {}

impl<F, A> Hash for ImmutableImageInitialization<F, A>
where
    F: 'static + Send + Sync,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        ImageAccess::inner(self).hash(state);
    }
}
