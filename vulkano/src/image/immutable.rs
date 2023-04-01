// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    sys::{Image, RawImage},
    traits::ImageContent,
    ImageAccess, ImageCreateFlags, ImageDescriptorLayouts, ImageDimensions, ImageError, ImageInner,
    ImageLayout, ImageSubresourceLayers, ImageUsage, MipmapsCount,
};
use crate::{
    buffer::{Buffer, BufferAllocateInfo, BufferContents, BufferError, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        BufferImageCopy, CommandBufferBeginError, CopyBufferToImageInfo, ImageBlit,
    },
    device::{Device, DeviceOwned},
    format::Format,
    image::sys::ImageCreateInfo,
    memory::{
        allocator::{
            AllocationCreateInfo, AllocationCreationError, AllocationType,
            MemoryAllocatePreference, MemoryAllocator, MemoryUsage,
        },
        is_aligned, DedicatedAllocation,
    },
    sampler::Filter,
    sync::Sharing,
    DeviceSize, VulkanError,
};
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Image whose purpose is to be used for read-only purposes. You can write to the image once,
/// but then you must only ever read from it.
// TODO: type (2D, 3D, array, etc.) as template parameter
#[derive(Debug)]
pub struct ImmutableImage {
    inner: Arc<Image>,
    layout: ImageLayout,
}

fn has_mipmaps(mipmaps: MipmapsCount) -> bool {
    match mipmaps {
        MipmapsCount::One => false,
        MipmapsCount::Log2 => true,
        MipmapsCount::Specific(x) => x > 1,
    }
}

fn generate_mipmaps<L, Cba>(
    cbb: &mut AutoCommandBufferBuilder<L, Cba>,
    image: Arc<dyn ImageAccess>,
    dimensions: ImageDimensions,
    _layout: ImageLayout,
) where
    Cba: CommandBufferAllocator,
{
    for level in 1..image.mip_levels() {
        let src_size = dimensions
            .mip_level_dimensions(level - 1)
            .unwrap()
            .width_height_depth();
        let dst_size = dimensions
            .mip_level_dimensions(level)
            .unwrap()
            .width_height_depth();

        cbb.blit_image(BlitImageInfo {
            regions: [ImageBlit {
                src_subresource: ImageSubresourceLayers {
                    mip_level: level - 1,
                    ..image.subresource_layers()
                },
                src_offsets: [[0; 3], src_size],
                dst_subresource: ImageSubresourceLayers {
                    mip_level: level,
                    ..image.subresource_layers()
                },
                dst_offsets: [[0; 3], dst_size],
                ..Default::default()
            }]
            .into(),
            filter: Filter::Linear,
            ..BlitImageInfo::images(image.clone(), image.clone())
        })
        .expect("failed to blit a mip map to image!");
    }
}

impl ImmutableImage {
    /// Builds an uninitialized immutable image.
    ///
    /// Returns two things: the image, and a special access that should be used for the initial
    /// upload to the image.
    pub fn uninitialized(
        allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: ImageDimensions,
        format: Format,
        mip_levels: impl Into<MipmapsCount>,
        usage: ImageUsage,
        flags: ImageCreateFlags,
        layout: ImageLayout,
        queue_family_indices: impl IntoIterator<Item = u32>,
    ) -> Result<(Arc<ImmutableImage>, Arc<ImmutableImageInitialization>), ImmutableImageCreationError>
    {
        let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();
        assert!(!flags.intersects(ImageCreateFlags::DISJOINT)); // TODO: adjust the code below to make this safe

        let raw_image = RawImage::new(
            allocator.device().clone(),
            ImageCreateInfo {
                flags,
                dimensions,
                format: Some(format),
                mip_levels: match mip_levels.into() {
                    MipmapsCount::Specific(num) => num,
                    MipmapsCount::Log2 => dimensions.max_mip_levels(),
                    MipmapsCount::One => 1,
                },
                usage,
                sharing: if queue_family_indices.len() >= 2 {
                    Sharing::Concurrent(queue_family_indices)
                } else {
                    Sharing::Exclusive
                },
                ..Default::default()
            },
        )?;
        let requirements = raw_image.memory_requirements()[0];
        let create_info = AllocationCreateInfo {
            requirements,
            allocation_type: AllocationType::NonLinear,
            usage: MemoryUsage::DeviceOnly,
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
            ..Default::default()
        };

        match unsafe { allocator.allocate_unchecked(create_info) } {
            Ok(alloc) => {
                debug_assert!(is_aligned(alloc.offset(), requirements.layout.alignment()));
                debug_assert!(alloc.size() == requirements.layout.size());

                let inner = Arc::new(unsafe {
                    raw_image
                        .bind_memory_unchecked([alloc])
                        .map_err(|(err, _, _)| err)?
                });

                let image = Arc::new(ImmutableImage { inner, layout });

                let init = Arc::new(ImmutableImageInitialization {
                    image: image.clone(),
                });

                Ok((image, init))
            }
            Err(err) => Err(err.into()),
        }
    }

    /// Construct an ImmutableImage from the contents of `iter`.
    ///
    /// This is a convenience function, equivalent to creating a `CpuAccessibleBuffer`, writing
    /// `iter` to it, then calling [`from_buffer`](ImmutableImage::from_buffer) to copy the data
    /// over.
    pub fn from_iter<Px, I, L, A>(
        allocator: &(impl MemoryAllocator + ?Sized),
        iter: I,
        dimensions: ImageDimensions,
        mip_levels: MipmapsCount,
        format: Format,
        command_buffer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> Result<Arc<Self>, ImmutableImageCreationError>
    where
        Px: BufferContents,
        I: IntoIterator<Item = Px>,
        I::IntoIter: ExactSizeIterator,
        A: CommandBufferAllocator,
    {
        let source = Buffer::from_iter(
            allocator,
            BufferAllocateInfo {
                buffer_usage: BufferUsage::TRANSFER_SRC,
                memory_usage: MemoryUsage::Upload,
                ..Default::default()
            },
            iter,
        )
        .map_err(|err| match err {
            BufferError::AllocError(err) => err,
            // We don't use sparse-binding, concurrent sharing or external memory, therefore the
            // other errors can't happen.
            _ => unreachable!(),
        })?;

        ImmutableImage::from_buffer(
            allocator,
            source,
            dimensions,
            mip_levels,
            format,
            command_buffer_builder,
        )
    }

    /// Construct an ImmutableImage containing a copy of the data in `source`.
    ///
    /// This is a convenience function, equivalent to calling
    /// [`uninitialized`](ImmutableImage::uninitialized) with the queue family index of
    /// `command_buffer_builder`, then recording a `copy_buffer_to_image` command to
    /// `command_buffer_builder`.
    ///
    /// `command_buffer_builder` can then be used to record other commands, built, and executed as
    /// normal. If it is not executed, the image contents will be left undefined.
    pub fn from_buffer<L, A>(
        allocator: &(impl MemoryAllocator + ?Sized),
        source: Subbuffer<impl ?Sized>,
        dimensions: ImageDimensions,
        mip_levels: MipmapsCount,
        format: Format,
        command_buffer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> Result<Arc<Self>, ImmutableImageCreationError>
    where
        A: CommandBufferAllocator,
    {
        let region = BufferImageCopy {
            image_subresource: ImageSubresourceLayers::from_parameters(
                format,
                dimensions.array_layers(),
            ),
            image_extent: dimensions.width_height_depth(),
            ..Default::default()
        };
        let required_size = region.buffer_copy_size(format);

        if source.size() < required_size {
            return Err(ImmutableImageCreationError::SourceTooSmall {
                source_size: source.size(),
                required_size,
            });
        }

        let need_to_generate_mipmaps = has_mipmaps(mip_levels);
        let usage = ImageUsage::TRANSFER_DST
            | ImageUsage::SAMPLED
            | if need_to_generate_mipmaps {
                ImageUsage::TRANSFER_SRC
            } else {
                ImageUsage::empty()
            };
        let flags = ImageCreateFlags::empty();
        let layout = ImageLayout::ShaderReadOnlyOptimal;

        let (image, initializer) = ImmutableImage::uninitialized(
            allocator,
            dimensions,
            format,
            mip_levels,
            usage,
            flags,
            layout,
            source
                .device()
                .active_queue_family_indices()
                .iter()
                .copied(),
        )?;

        command_buffer_builder
            .copy_buffer_to_image(CopyBufferToImageInfo {
                regions: smallvec![region],
                ..CopyBufferToImageInfo::buffer_image(source, initializer)
            })
            .unwrap();

        if need_to_generate_mipmaps {
            generate_mipmaps(
                command_buffer_builder,
                image.clone(),
                image.inner.dimensions(),
                ImageLayout::ShaderReadOnlyOptimal,
            );
        }

        Ok(image)
    }
}

unsafe impl DeviceOwned for ImmutableImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl ImageAccess for ImmutableImage {
    #[inline]
    fn inner(&self) -> ImageInner<'_> {
        ImageInner {
            image: &self.inner,
            first_layer: 0,
            num_layers: self.inner.dimensions().array_layers(),
            first_mipmap_level: 0,
            num_mipmap_levels: self.inner.mip_levels(),
        }
    }

    #[inline]
    fn is_layout_initialized(&self) -> bool {
        true
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
}

unsafe impl<P> ImageContent<P> for ImmutableImage {
    fn matches_format(&self) -> bool {
        true // FIXME:
    }
}

impl PartialEq for ImmutableImage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for ImmutableImage {}

impl Hash for ImmutableImage {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

// Must not implement Clone, as that would lead to multiple `used` values.
pub struct ImmutableImageInitialization {
    image: Arc<ImmutableImage>,
}

unsafe impl DeviceOwned for ImmutableImageInitialization {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.image.device()
    }
}

unsafe impl ImageAccess for ImmutableImageInitialization {
    #[inline]
    fn inner(&self) -> ImageInner<'_> {
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
}

impl PartialEq for ImmutableImageInitialization {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for ImmutableImageInitialization {}

impl Hash for ImmutableImageInitialization {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

/// Error that can happen when creating an `ImmutableImage`.
#[derive(Clone, Debug)]
pub enum ImmutableImageCreationError {
    ImageCreationError(ImageError),
    AllocError(AllocationCreationError),
    CommandBufferBeginError(CommandBufferBeginError),

    /// The size of the provided source data is less than the required size for an image with the
    /// given format and dimensions.
    SourceTooSmall {
        source_size: DeviceSize,
        required_size: DeviceSize,
    },
}

impl Error for ImmutableImageCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ImageCreationError(err) => Some(err),
            Self::AllocError(err) => Some(err),
            Self::CommandBufferBeginError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for ImmutableImageCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::ImageCreationError(err) => err.fmt(f),
            Self::AllocError(err) => err.fmt(f),
            Self::CommandBufferBeginError(err) => err.fmt(f),
            Self::SourceTooSmall {
                source_size,
                required_size,
            } => write!(
                f,
                "the size of the provided source data ({} bytes) is less than the required size \
                for an image of the given format and dimensions ({} bytes)",
                source_size, required_size,
            ),
        }
    }
}

impl From<ImageError> for ImmutableImageCreationError {
    fn from(err: ImageError) -> Self {
        Self::ImageCreationError(err)
    }
}

impl From<AllocationCreationError> for ImmutableImageCreationError {
    fn from(err: AllocationCreationError) -> Self {
        Self::AllocError(err)
    }
}

impl From<VulkanError> for ImmutableImageCreationError {
    fn from(err: VulkanError) -> Self {
        Self::AllocError(err.into())
    }
}

impl From<CommandBufferBeginError> for ImmutableImageCreationError {
    fn from(err: CommandBufferBeginError) -> Self {
        Self::CommandBufferBeginError(err)
    }
}
