// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{
        auto::ClearDepthStencilImageError,
        commands::transfer::{is_overlapping_ranges, is_overlapping_regions},
        synced::{Command, Resource, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, BlitImageError,
        ClearColorImageError,
    },
    device::{Device, DeviceOwned},
    format::{ClearValue, NumericType},
    image::{
        ImageAccess, ImageAspects, ImageDimensions, ImageLayout, ImageSubresourceRange, SampleCount,
    },
    sampler::Filter,
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    VulkanObject,
};
use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};
use std::{error, fmt, sync::Arc};

/// # Commands that operate on images.
///
/// Unlike transfer commands, these require a graphics queue, except for `clear_color_image`, which
/// can also be called on a compute queue.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Adds a command that blits an image to another.
    ///
    /// A *blit* is similar to an image copy operation, except that the portion of the image that
    /// is transferred can be resized. You choose an area of the source and an area of the
    /// destination, and the implementation will resize the area of the source so that it matches
    /// the size of the area of the destination before writing it.
    ///
    /// Blit operations have several restrictions:
    ///
    /// - Blit operations are only allowed on queue families that support graphics operations.
    /// - The format of the source and destination images must support blit operations, which
    ///   depends on the Vulkan implementation. Vulkan guarantees that some specific formats must
    ///   always be supported. See tables 52 to 61 of the specifications.
    /// - Only single-sampled images are allowed.
    /// - You can only blit between two images whose formats belong to the same type. The types
    ///   are: floating-point, signed integers, unsigned integers, depth-stencil.
    /// - If you blit between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - If you blit between depth, stencil or depth-stencil images, only the `Nearest` filter is
    ///   allowed.
    /// - For two-dimensional images, the Z coordinate must be 0 for the top-left offset and 1 for
    ///   the bottom-right offset. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the blit will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panic
    ///
    /// - Panics if the source or the destination was not created with `device`.
    ///
    pub fn blit_image(
        &mut self,
        source: Arc<dyn ImageAccess>,
        source_top_left: [i32; 3],
        source_bottom_right: [i32; 3],
        source_base_array_layer: u32,
        source_mip_level: u32,
        destination: Arc<dyn ImageAccess>,
        destination_top_left: [i32; 3],
        destination_bottom_right: [i32; 3],
        destination_base_array_layer: u32,
        destination_mip_level: u32,
        layer_count: u32,
        filter: Filter,
    ) -> Result<&mut Self, BlitImageError> {
        unsafe {
            if !self.queue_family().supports_graphics() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;

            check_blit_image(
                self.device(),
                source.as_ref(),
                source_top_left,
                source_bottom_right,
                source_base_array_layer,
                source_mip_level,
                destination.as_ref(),
                destination_top_left,
                destination_bottom_right,
                destination_base_array_layer,
                destination_mip_level,
                layer_count,
                filter,
            )?;

            let blit = UnsafeCommandBufferBuilderImageBlit {
                // TODO:
                aspects: if source.format().aspects().color {
                    ImageAspects {
                        color: true,
                        ..ImageAspects::none()
                    }
                } else {
                    unimplemented!()
                },
                source_mip_level,
                destination_mip_level,
                source_base_array_layer,
                destination_base_array_layer,
                layer_count,
                source_top_left,
                source_bottom_right,
                destination_top_left,
                destination_bottom_right,
            };

            // TODO: Allow choosing layouts, but note that only Transfer*Optimal and General are
            // valid.
            if source.conflict_key() == destination.conflict_key() {
                // since we are blitting from the same image, we must use the same layout
                self.inner.blit_image(
                    source,
                    ImageLayout::General,
                    destination,
                    ImageLayout::General,
                    [blit],
                    filter,
                )?;
            } else {
                self.inner.blit_image(
                    source,
                    ImageLayout::TransferSrcOptimal,
                    destination,
                    ImageLayout::TransferDstOptimal,
                    [blit],
                    filter,
                )?;
            }

            Ok(self)
        }
    }

    /// Adds a command that clears all the layers and mipmap levels of a color image with a
    /// specific value.
    ///
    /// # Panic
    ///
    /// Panics if `color` is not a color value.
    ///
    pub fn clear_color_image(
        &mut self,
        image: Arc<dyn ImageAccess>,
        color: ClearValue,
    ) -> Result<&mut Self, ClearColorImageError> {
        let array_layers = image.dimensions().array_layers();
        let mip_levels = image.mip_levels();

        self.clear_color_image_dimensions(image, 0, array_layers, 0, mip_levels, color)
    }

    /// Adds a command that clears a color image with a specific value.
    ///
    /// # Panic
    ///
    /// - Panics if `color` is not a color value.
    ///
    pub fn clear_color_image_dimensions(
        &mut self,
        image: Arc<dyn ImageAccess>,
        base_array_layer: u32,
        layer_count: u32,
        base_mip_level: u32,
        level_count: u32,
        color: ClearValue,
    ) -> Result<&mut Self, ClearColorImageError> {
        unsafe {
            if !self.queue_family().supports_graphics() && !self.queue_family().supports_compute() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_clear_color_image(
                self.device(),
                image.as_ref(),
                base_array_layer,
                layer_count,
                base_mip_level,
                level_count,
            )?;

            match color {
                ClearValue::Float(_) | ClearValue::Int(_) | ClearValue::Uint(_) => {}
                _ => panic!("The clear color is not a color value"),
            };

            let region = UnsafeCommandBufferBuilderColorImageClear {
                base_mip_level,
                level_count,
                base_array_layer,
                layer_count,
            };

            // TODO: let choose layout
            self.inner.clear_color_image(
                image,
                ImageLayout::TransferDstOptimal,
                color,
                [region],
            )?;
            Ok(self)
        }
    }

    /// Adds a command that clears all the layers of a depth / stencil image with a
    /// specific value.
    ///
    /// # Panic
    ///
    /// Panics if `clear_value` is not a depth / stencil value.
    ///
    pub fn clear_depth_stencil_image(
        &mut self,
        image: Arc<dyn ImageAccess>,
        clear_value: ClearValue,
    ) -> Result<&mut Self, ClearDepthStencilImageError> {
        let layers = image.dimensions().array_layers();

        self.clear_depth_stencil_image_dimensions(image, 0, layers, clear_value)
    }

    /// Adds a command that clears a depth / stencil image with a specific value.
    ///
    /// # Panic
    ///
    /// - Panics if `clear_value` is not a depth / stencil value.
    ///
    pub fn clear_depth_stencil_image_dimensions(
        &mut self,
        image: Arc<dyn ImageAccess>,
        base_array_layer: u32,
        layer_count: u32,
        clear_value: ClearValue,
    ) -> Result<&mut Self, ClearDepthStencilImageError> {
        unsafe {
            if !self.queue_family().supports_graphics() && !self.queue_family().supports_compute() {
                return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
            }

            self.ensure_outside_render_pass()?;
            check_clear_depth_stencil_image(
                self.device(),
                image.as_ref(),
                base_array_layer,
                layer_count,
            )?;

            let (clear_depth, clear_stencil) = match clear_value {
                ClearValue::Depth(_) => (true, false),
                ClearValue::Stencil(_) => (false, true),
                ClearValue::DepthStencil(_) => (true, true),
                _ => panic!("The clear value is not a depth / stencil value"),
            };

            let region = UnsafeCommandBufferBuilderDepthStencilImageClear {
                base_array_layer,
                layer_count,
                clear_depth,
                clear_stencil,
            };

            // TODO: let choose layout
            self.inner.clear_depth_stencil_image(
                image,
                ImageLayout::TransferDstOptimal,
                clear_value,
                [region],
            )?;
            Ok(self)
        }
    }
}

/// Checks whether a blit image command is valid.
///
/// Note that this doesn't check whether `layer_count` is equal to 0. TODO: change that?
///
/// # Panic
///
/// - Panics if the source or the destination was not created with `device`.
///
fn check_blit_image<S, D>(
    device: &Device,
    source: &S,
    source_top_left: [i32; 3],
    source_bottom_right: [i32; 3],
    source_base_array_layer: u32,
    source_mip_level: u32,
    destination: &D,
    destination_top_left: [i32; 3],
    destination_bottom_right: [i32; 3],
    destination_base_array_layer: u32,
    destination_mip_level: u32,
    layer_count: u32,
    filter: Filter,
) -> Result<(), CheckBlitImageError>
where
    S: ?Sized + ImageAccess,
    D: ?Sized + ImageAccess,
{
    let source_inner = source.inner();
    let destination_inner = destination.inner();

    assert_eq!(
        source_inner.image.device().internal_object(),
        device.internal_object()
    );
    assert_eq!(
        destination_inner.image.device().internal_object(),
        device.internal_object()
    );

    if !source_inner.image.usage().transfer_source {
        return Err(CheckBlitImageError::MissingTransferSourceUsage);
    }

    if !destination_inner.image.usage().transfer_destination {
        return Err(CheckBlitImageError::MissingTransferDestinationUsage);
    }

    if !source_inner.image.format_features().blit_src {
        return Err(CheckBlitImageError::SourceFormatNotSupported);
    }

    if !destination_inner.image.format_features().blit_dst {
        return Err(CheckBlitImageError::DestinationFormatNotSupported);
    }

    if source.samples() != SampleCount::Sample1 || destination.samples() != SampleCount::Sample1 {
        return Err(CheckBlitImageError::UnexpectedMultisampled);
    }

    if let (Some(source_type), Some(destination_type)) = (
        source.format().type_color(),
        destination.format().type_color(),
    ) {
        let types_should_be_same = source_type == NumericType::UINT
            || destination_type == NumericType::UINT
            || source_type == NumericType::SINT
            || destination_type == NumericType::SINT;
        if types_should_be_same && (source_type != destination_type) {
            return Err(CheckBlitImageError::IncompatibleFormatTypes {
                source_type,
                destination_type,
            });
        }
    } else {
        if source.format() != destination.format() {
            return Err(CheckBlitImageError::DepthStencilFormatMismatch);
        }

        if filter != Filter::Nearest {
            return Err(CheckBlitImageError::DepthStencilNearestMandatory);
        }
    }

    let source_dimensions = match source.dimensions().mip_level_dimensions(source_mip_level) {
        Some(d) => d,
        None => return Err(CheckBlitImageError::SourceCoordinatesOutOfRange),
    };

    let destination_dimensions = match destination
        .dimensions()
        .mip_level_dimensions(destination_mip_level)
    {
        Some(d) => d,
        None => return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange),
    };

    if source_base_array_layer + layer_count > source_dimensions.array_layers() {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if destination_base_array_layer + layer_count > destination_dimensions.array_layers() {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if source_top_left[0] < 0 || source_top_left[0] > source_dimensions.width() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_top_left[1] < 0 || source_top_left[1] > source_dimensions.height() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_top_left[2] < 0 || source_top_left[2] > source_dimensions.depth() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_bottom_right[0] < 0 || source_bottom_right[0] > source_dimensions.width() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_bottom_right[1] < 0 || source_bottom_right[1] > source_dimensions.height() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if source_bottom_right[2] < 0 || source_bottom_right[2] > source_dimensions.depth() as i32 {
        return Err(CheckBlitImageError::SourceCoordinatesOutOfRange);
    }

    if destination_top_left[0] < 0
        || destination_top_left[0] > destination_dimensions.width() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_top_left[1] < 0
        || destination_top_left[1] > destination_dimensions.height() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_top_left[2] < 0
        || destination_top_left[2] > destination_dimensions.depth() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_bottom_right[0] < 0
        || destination_bottom_right[0] > destination_dimensions.width() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_bottom_right[1] < 0
        || destination_bottom_right[1] > destination_dimensions.height() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_bottom_right[2] < 0
        || destination_bottom_right[2] > destination_dimensions.depth() as i32
    {
        return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange);
    }

    match source_dimensions {
        ImageDimensions::Dim1d { .. } => {
            if source_top_left[1] != 0 || source_bottom_right[1] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
            if source_top_left[2] != 0 || source_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            if source_top_left[2] != 0 || source_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim3d { .. } => {}
    }

    match destination_dimensions {
        ImageDimensions::Dim1d { .. } => {
            if destination_top_left[1] != 0 || destination_bottom_right[1] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
            if destination_top_left[2] != 0 || destination_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            if destination_top_left[2] != 0 || destination_bottom_right[2] != 1 {
                return Err(CheckBlitImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim3d { .. } => {}
    }

    if source.conflict_key() == destination.conflict_key() {
        if source_mip_level == destination_mip_level
            && is_overlapping_ranges(
                source_base_array_layer as u64,
                layer_count as u64,
                destination_base_array_layer as u64,
                layer_count as u64,
            )
        {
            // we get the top left coordinate of the source in relation to the resulting image,
            // because in blit we can do top_left = [100, 100] and bottom_right = [0, 0]
            // which would result in flipped image and thats ok, but we can't use these values to compute
            // extent, because it would result in negative size.
            let mut source_render_top_left = [0; 3];
            let mut source_extent = [0; 3];
            let mut destination_render_top_left = [0; 3];
            let mut destination_extent = [0; 3];
            for i in 0..3 {
                if source_top_left[i] < source_bottom_right[i] {
                    source_render_top_left[i] = source_top_left[i];
                    source_extent[i] = (source_bottom_right[i] - source_top_left[i]) as u32;
                } else {
                    source_render_top_left[i] = source_bottom_right[i];
                    source_extent[i] = (source_top_left[i] - source_bottom_right[i]) as u32;
                }
                if destination_top_left[i] < destination_bottom_right[i] {
                    destination_render_top_left[i] = destination_top_left[i];
                    destination_extent[i] =
                        (destination_bottom_right[i] - destination_top_left[i]) as u32;
                } else {
                    destination_render_top_left[i] = destination_bottom_right[i];
                    destination_extent[i] =
                        (destination_top_left[i] - destination_bottom_right[i]) as u32;
                }
            }

            if is_overlapping_regions(
                source_render_top_left,
                source_extent,
                destination_render_top_left,
                destination_extent,
                // since both images are the same, we can use any dimensions type
                source_dimensions,
            ) {
                return Err(CheckBlitImageError::OverlappingRegions);
            }
        }
    }

    match filter {
        Filter::Nearest => (),
        Filter::Linear => {
            if !source_inner
                .image
                .format_features()
                .sampled_image_filter_linear
            {
                return Err(CheckBlitImageError::FilterFormatNotSupported);
            }
        }
        Filter::Cubic => {
            if !device.enabled_extensions().ext_filter_cubic {
                return Err(CheckBlitImageError::ExtensionNotEnabled {
                    extension: "ext_filter_cubic",
                    reason: "the specified filter was Cubic",
                });
            }

            if !source_inner
                .image
                .format_features()
                .sampled_image_filter_cubic
            {
                return Err(CheckBlitImageError::FilterFormatNotSupported);
            }

            if !matches!(source.dimensions(), ImageDimensions::Dim2d { .. }) {
                return Err(CheckBlitImageError::FilterDimensionalityNotSupported);
            }
        }
    }

    Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckBlitImageError {
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },

    /// The chosen filter type does not support the dimensionality of the source image.
    FilterDimensionalityNotSupported,
    /// The chosen filter type does not support the format of the source image.
    FilterFormatNotSupported,
    /// The source is missing the transfer source usage.
    MissingTransferSourceUsage,
    /// The destination is missing the transfer destination usage.
    MissingTransferDestinationUsage,
    /// The format of the source image doesn't support blit operations.
    SourceFormatNotSupported,
    /// The format of the destination image doesn't support blit operations.
    DestinationFormatNotSupported,
    /// You must use the nearest filter when blitting depth/stencil images.
    DepthStencilNearestMandatory,
    /// The format of the source and destination must be equal when blitting depth/stencil images.
    DepthStencilFormatMismatch,
    /// The types of the source format and the destination format aren't compatible.
    IncompatibleFormatTypes {
        source_type: NumericType,
        destination_type: NumericType,
    },
    /// Blitting between multisampled images is forbidden.
    UnexpectedMultisampled,
    /// The offsets, array layers and/or mipmap levels are out of range in the source image.
    SourceCoordinatesOutOfRange,
    /// The offsets, array layers and/or mipmap levels are out of range in the destination image.
    DestinationCoordinatesOutOfRange,
    /// The top-left and/or bottom-right coordinates are incompatible with the image type.
    IncompatibleRangeForImageType,
    /// The source and destination regions are overlapping.
    OverlappingRegions,
}

impl error::Error for CheckBlitImageError {}

impl fmt::Display for CheckBlitImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
            Self::FilterDimensionalityNotSupported => write!(
                fmt,
                "the chosen filter type does not support the dimensionality of the source image"
            ),
            Self::FilterFormatNotSupported => write!(
                fmt,
                "the chosen filter type does not support the format of the source image"
            ),
            Self::MissingTransferSourceUsage => {
                write!(fmt, "the source is missing the transfer source usage")
            }
            Self::MissingTransferDestinationUsage => {
                write!(
                    fmt,
                    "the destination is missing the transfer destination usage"
                )
            }
            Self::SourceFormatNotSupported => {
                write!(
                    fmt,
                    "the format of the source image doesn't support blit operations"
                )
            }
            Self::DestinationFormatNotSupported => {
                write!(
                    fmt,
                    "the format of the destination image doesn't support blit operations"
                )
            }
            Self::DepthStencilNearestMandatory => {
                write!(
                    fmt,
                    "you must use the nearest filter when blitting depth/stencil images"
                )
            }
            Self::DepthStencilFormatMismatch => {
                write!(fmt, "the format of the source and destination must be equal when blitting depth/stencil images")
            }
            Self::IncompatibleFormatTypes { .. } => {
                write!(
                    fmt,
                    "the types of the source format and the destination format aren't compatible"
                )
            }
            Self::UnexpectedMultisampled => {
                write!(fmt, "blitting between multisampled images is forbidden")
            }
            Self::SourceCoordinatesOutOfRange => {
                write!(fmt, "the offsets, array layers and/or mipmap levels are out of range in the source image")
            }
            Self::DestinationCoordinatesOutOfRange => {
                write!(fmt, "the offsets, array layers and/or mipmap levels are out of range in the destination image")
            }
            Self::IncompatibleRangeForImageType => {
                write!(fmt, "the top-left and/or bottom-right coordinates are incompatible with the image type")
            }
            Self::OverlappingRegions => {
                write!(fmt, "the source and destination regions are overlapping")
            }
        }
    }
}

/// Checks whether a clear color image command is valid.
///
/// # Panic
///
/// - Panics if the destination was not created with `device`.
///
fn check_clear_color_image<I>(
    device: &Device,
    image: &I,
    base_array_layer: u32,
    layer_count: u32,
    base_mip_level: u32,
    level_count: u32,
) -> Result<(), CheckClearColorImageError>
where
    I: ?Sized + ImageAccess,
{
    assert_eq!(
        image.inner().image.device().internal_object(),
        device.internal_object()
    );

    if !image.inner().image.usage().transfer_destination {
        return Err(CheckClearColorImageError::MissingTransferUsage);
    }

    if base_array_layer + layer_count > image.dimensions().array_layers() {
        return Err(CheckClearColorImageError::OutOfRange);
    }

    if base_mip_level + level_count > image.mip_levels() {
        return Err(CheckClearColorImageError::OutOfRange);
    }

    Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckClearColorImageError {
    /// The image is missing the transfer destination usage.
    MissingTransferUsage,
    /// The array layers and mipmap levels are out of range.
    OutOfRange,
}

impl error::Error for CheckClearColorImageError {}

impl fmt::Display for CheckClearColorImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckClearColorImageError::MissingTransferUsage => {
                    "the image is missing the transfer destination usage"
                }
                CheckClearColorImageError::OutOfRange => {
                    "the array layers and mipmap levels are out of range"
                }
            }
        )
    }
}

/// Checks whether a clear depth / stencil image command is valid.
///
/// # Panic
///
/// - Panics if the destination was not created with `device`.
///
fn check_clear_depth_stencil_image<I>(
    device: &Device,
    image: &I,
    first_layer: u32,
    num_layers: u32,
) -> Result<(), CheckClearDepthStencilImageError>
where
    I: ?Sized + ImageAccess,
{
    assert_eq!(
        image.inner().image.device().internal_object(),
        device.internal_object()
    );

    if !image.inner().image.usage().transfer_destination {
        return Err(CheckClearDepthStencilImageError::MissingTransferUsage);
    }

    if first_layer + num_layers > image.dimensions().array_layers() {
        return Err(CheckClearDepthStencilImageError::OutOfRange);
    }

    Ok(())
}

/// Error that can happen from `check_clear_depth_stencil_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckClearDepthStencilImageError {
    /// The image is missing the transfer destination usage.
    MissingTransferUsage,
    /// The array layers are out of range.
    OutOfRange,
}

impl error::Error for CheckClearDepthStencilImageError {}

impl fmt::Display for CheckClearDepthStencilImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckClearDepthStencilImageError::MissingTransferUsage => {
                    "the image is missing the transfer destination usage"
                }
                CheckClearDepthStencilImageError::OutOfRange => {
                    "the array layers are out of range"
                }
            }
        )
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image<R>(
        &mut self,
        source: Arc<dyn ImageAccess>,
        source_layout: ImageLayout,
        destination: Arc<dyn ImageAccess>,
        destination_layout: ImageLayout,
        regions: R,
        filter: Filter,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = UnsafeCommandBufferBuilderImageBlit> + Send + Sync + 'static,
    {
        struct Cmd<R> {
            source: Arc<dyn ImageAccess>,
            source_layout: ImageLayout,
            destination: Arc<dyn ImageAccess>,
            destination_layout: ImageLayout,
            regions: Mutex<Option<R>>,
            filter: Filter,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = UnsafeCommandBufferBuilderImageBlit> + Send + Sync,
        {
            fn name(&self) -> &'static str {
                "blit_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.blit_image(
                    self.source.as_ref(),
                    self.source_layout,
                    self.destination.as_ref(),
                    self.destination_layout,
                    self.regions.lock().take().unwrap(),
                    self.filter,
                );
            }
        }

        // if its the same image in source and destination, we need to lock it once
        let source_key = (
            source.conflict_key(),
            source.current_mip_levels_access(),
            source.current_array_layers_access(),
        );
        let destination_key = (
            destination.conflict_key(),
            destination.current_mip_levels_access(),
            destination.current_array_layers_access(),
        );

        let resources: SmallVec<[_; 2]> = if source_key == destination_key {
            smallvec![(
                "source_and_destination".into(),
                Resource::Image {
                    image: source.clone(),
                    subresource_range: ImageSubresourceRange {
                        // TODO:
                        aspects: source.format().aspects(),
                        mip_levels: source.current_mip_levels_access(),
                        array_layers: source.current_array_layers_access(),
                    },
                    memory: PipelineMemoryAccess {
                        stages: PipelineStages {
                            transfer: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlags {
                            transfer_read: true,
                            transfer_write: true,
                            ..AccessFlags::none()
                        },
                        exclusive: true,
                    },
                    start_layout: ImageLayout::General,
                    end_layout: ImageLayout::General,
                },
            )]
        } else {
            smallvec![
                (
                    "source".into(),
                    Resource::Image {
                        image: source.clone(),
                        subresource_range: ImageSubresourceRange {
                            // TODO:
                            aspects: source.format().aspects(),
                            mip_levels: source.current_mip_levels_access(),
                            array_layers: source.current_array_layers_access(),
                        },
                        memory: PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlags {
                                transfer_read: true,
                                ..AccessFlags::none()
                            },
                            exclusive: false,
                        },
                        start_layout: source_layout,
                        end_layout: source_layout,
                    },
                ),
                (
                    "destination".into(),
                    Resource::Image {
                        image: destination.clone(),
                        subresource_range: ImageSubresourceRange {
                            // TODO:
                            aspects: destination.format().aspects(),
                            mip_levels: destination.current_mip_levels_access(),
                            array_layers: destination.current_array_layers_access(),
                        },
                        memory: PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlags {
                                transfer_write: true,
                                ..AccessFlags::none()
                            },
                            exclusive: true,
                        },
                        start_layout: destination_layout,
                        end_layout: destination_layout,
                    },
                ),
            ]
        };

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            source,
            source_layout,
            destination,
            destination_layout,
            regions: Mutex::new(Some(regions)),
            filter,
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_color_image<R>(
        &mut self,
        image: Arc<dyn ImageAccess>,
        layout: ImageLayout,
        color: ClearValue,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = UnsafeCommandBufferBuilderColorImageClear> + Send + Sync + 'static,
    {
        struct Cmd<R> {
            image: Arc<dyn ImageAccess>,
            layout: ImageLayout,
            color: ClearValue,
            regions: Mutex<Option<R>>,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = UnsafeCommandBufferBuilderColorImageClear>
                + Send
                + Sync
                + 'static,
        {
            fn name(&self) -> &'static str {
                "clear_color_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_color_image(
                    self.image.as_ref(),
                    self.layout,
                    self.color,
                    self.regions.lock().take().unwrap(),
                );
            }
        }

        let resources = [(
            "target".into(),
            Resource::Image {
                image: image.clone(),
                subresource_range: ImageSubresourceRange {
                    // TODO:
                    aspects: image.format().aspects(),
                    mip_levels: image.current_mip_levels_access(),
                    array_layers: image.current_array_layers_access(),
                },
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        transfer: true,
                        ..PipelineStages::none()
                    },
                    access: AccessFlags {
                        transfer_write: true,
                        ..AccessFlags::none()
                    },
                    exclusive: true,
                },
                start_layout: layout,
                end_layout: layout,
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            image,
            layout,
            color,
            regions: Mutex::new(Some(regions)),
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }

    /// Calls `vkCmdClearDepthStencilImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_depth_stencil_image<R>(
        &mut self,
        image: Arc<dyn ImageAccess>,
        layout: ImageLayout,
        clear_value: ClearValue,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = UnsafeCommandBufferBuilderDepthStencilImageClear>
            + Send
            + Sync
            + 'static,
    {
        struct Cmd<R> {
            image: Arc<dyn ImageAccess>,
            layout: ImageLayout,
            clear_value: ClearValue,
            regions: Mutex<Option<R>>,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = UnsafeCommandBufferBuilderDepthStencilImageClear>
                + Send
                + Sync
                + 'static,
        {
            fn name(&self) -> &'static str {
                "clear_depth_stencil_image"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_depth_stencil_image(
                    self.image.as_ref(),
                    self.layout,
                    self.clear_value,
                    self.regions.lock().take().unwrap(),
                );
            }
        }

        let resources = [(
            "target".into(),
            Resource::Image {
                image: image.clone(),
                subresource_range: ImageSubresourceRange {
                    // TODO:
                    aspects: image.format().aspects(),
                    mip_levels: image.current_mip_levels_access(),
                    array_layers: image.current_array_layers_access(),
                },
                memory: PipelineMemoryAccess {
                    stages: PipelineStages {
                        transfer: true,
                        ..PipelineStages::none()
                    },
                    access: AccessFlags {
                        transfer_write: true,
                        ..AccessFlags::none()
                    },
                    exclusive: true,
                },
                start_layout: layout,
                end_layout: layout,
            },
        )];

        for resource in &resources {
            self.check_resource_conflicts(resource)?;
        }

        self.commands.push(Box::new(Cmd {
            image,
            layout,
            clear_value,
            regions: Mutex::new(Some(regions)),
        }));

        for resource in resources {
            self.add_resource(resource);
        }

        Ok(())
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image(
        &mut self,
        source: &dyn ImageAccess,
        source_layout: ImageLayout,
        destination: &dyn ImageAccess,
        destination_layout: ImageLayout,
        regions: impl IntoIterator<Item = UnsafeCommandBufferBuilderImageBlit>,
        filter: Filter,
    ) {
        let source_aspects = source.format().aspects();

        if let (Some(source_type), Some(destination_type)) = (
            source.format().type_color(),
            destination.format().type_color(),
        ) {
            debug_assert!(
                (source_type == NumericType::UINT) == (destination_type == NumericType::UINT)
            );
            debug_assert!(
                (source_type == NumericType::SINT) == (destination_type == NumericType::SINT)
            );
        } else {
            debug_assert!(source.format() == destination.format());
            debug_assert!(filter == Filter::Nearest);
        }

        debug_assert_eq!(source.samples(), SampleCount::Sample1);
        let source = source.inner();
        debug_assert!(source.image.format_features().blit_src);
        debug_assert!(source.image.usage().transfer_source);
        debug_assert!(
            source_layout == ImageLayout::General
                || source_layout == ImageLayout::TransferSrcOptimal
        );

        debug_assert_eq!(destination.samples(), SampleCount::Sample1);
        let destination = destination.inner();
        debug_assert!(destination.image.format_features().blit_dst);
        debug_assert!(destination.image.usage().transfer_destination);
        debug_assert!(
            destination_layout == ImageLayout::General
                || destination_layout == ImageLayout::TransferDstOptimal
        );

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|blit| {
                // TODO: not everything is checked here
                debug_assert!(
                    blit.source_base_array_layer + blit.layer_count <= source.num_layers as u32
                );
                debug_assert!(
                    blit.destination_base_array_layer + blit.layer_count
                        <= destination.num_layers as u32
                );
                debug_assert!(blit.source_mip_level < destination.num_mipmap_levels as u32);
                debug_assert!(blit.destination_mip_level < destination.num_mipmap_levels as u32);

                if blit.layer_count == 0 {
                    return None;
                }

                Some(ash::vk::ImageBlit {
                    src_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: blit.aspects.into(),
                        mip_level: blit.source_mip_level,
                        base_array_layer: blit.source_base_array_layer + source.first_layer as u32,
                        layer_count: blit.layer_count,
                    },
                    src_offsets: [
                        ash::vk::Offset3D {
                            x: blit.source_top_left[0],
                            y: blit.source_top_left[1],
                            z: blit.source_top_left[2],
                        },
                        ash::vk::Offset3D {
                            x: blit.source_bottom_right[0],
                            y: blit.source_bottom_right[1],
                            z: blit.source_bottom_right[2],
                        },
                    ],
                    dst_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: blit.aspects.into(),
                        mip_level: blit.destination_mip_level,
                        base_array_layer: blit.destination_base_array_layer
                            + destination.first_layer as u32,
                        layer_count: blit.layer_count,
                    },
                    dst_offsets: [
                        ash::vk::Offset3D {
                            x: blit.destination_top_left[0],
                            y: blit.destination_top_left[1],
                            z: blit.destination_top_left[2],
                        },
                        ash::vk::Offset3D {
                            x: blit.destination_bottom_right[0],
                            y: blit.destination_bottom_right[1],
                            z: blit.destination_bottom_right[2],
                        },
                    ],
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_blit_image(
            self.handle,
            source.image.internal_object(),
            source_layout.into(),
            destination.image.internal_object(),
            destination_layout.into(),
            regions.len() as u32,
            regions.as_ptr(),
            filter.into(),
        );
    }

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    // TODO: ClearValue could be more precise
    pub unsafe fn clear_color_image(
        &mut self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        color: ClearValue,
        regions: impl IntoIterator<Item = UnsafeCommandBufferBuilderColorImageClear>,
    ) {
        let image_aspects = image.format().aspects();
        debug_assert!(image_aspects.color && !image_aspects.plane0);
        debug_assert!(image.format().compression().is_none());

        let image = image.inner();
        debug_assert!(image.image.usage().transfer_destination);
        debug_assert!(layout == ImageLayout::General || layout == ImageLayout::TransferDstOptimal);

        let color = match color {
            ClearValue::Float(val) => ash::vk::ClearColorValue { float32: val },
            ClearValue::Int(val) => ash::vk::ClearColorValue { int32: val },
            ClearValue::Uint(val) => ash::vk::ClearColorValue { uint32: val },
            _ => ash::vk::ClearColorValue { float32: [0.0; 4] },
        };

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|region| {
                debug_assert!(
                    region.layer_count + region.base_array_layer <= image.num_layers as u32
                );
                debug_assert!(
                    region.level_count + region.base_mip_level <= image.num_mipmap_levels as u32
                );

                if region.layer_count == 0 || region.level_count == 0 {
                    return None;
                }

                Some(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_mip_level: region.base_mip_level + image.first_mipmap_level as u32,
                    level_count: region.level_count,
                    base_array_layer: region.base_array_layer + image.first_layer as u32,
                    layer_count: region.layer_count,
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_clear_color_image(
            self.handle,
            image.image.internal_object(),
            layout.into(),
            &color,
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdClearDepthStencilImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_depth_stencil_image(
        &mut self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        clear_value: ClearValue,
        regions: impl IntoIterator<Item = UnsafeCommandBufferBuilderDepthStencilImageClear>,
    ) {
        let image_aspects = image.format().aspects();
        debug_assert!((image_aspects.depth || image_aspects.stencil) && !image_aspects.plane0);
        debug_assert!(image.format().compression().is_none());

        let image = image.inner();
        debug_assert!(image.image.usage().transfer_destination);
        debug_assert!(layout == ImageLayout::General || layout == ImageLayout::TransferDstOptimal);

        let clear_value = match clear_value {
            ClearValue::Depth(val) => ash::vk::ClearDepthStencilValue {
                depth: val,
                stencil: 0,
            },
            ClearValue::Stencil(val) => ash::vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: val,
            },
            ClearValue::DepthStencil((depth, stencil)) => {
                ash::vk::ClearDepthStencilValue { depth, stencil }
            }
            _ => ash::vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        };

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|region| {
                debug_assert!(
                    region.layer_count + region.base_array_layer <= image.num_layers as u32
                );

                if region.layer_count == 0 {
                    return None;
                }

                let mut aspect_mask = ash::vk::ImageAspectFlags::empty();
                if region.clear_depth {
                    aspect_mask |= ash::vk::ImageAspectFlags::DEPTH;
                }
                if region.clear_stencil {
                    aspect_mask |= ash::vk::ImageAspectFlags::STENCIL;
                }

                if aspect_mask.is_empty() {
                    return None;
                }

                Some(ash::vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: region.base_array_layer + image.first_layer as u32,
                    layer_count: region.layer_count,
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_clear_depth_stencil_image(
            self.handle,
            image.image.internal_object(),
            layout.into(),
            &clear_value,
            regions.len() as u32,
            regions.as_ptr(),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageBlit {
    pub aspects: ImageAspects,
    pub source_mip_level: u32,
    pub destination_mip_level: u32,
    pub source_base_array_layer: u32,
    pub destination_base_array_layer: u32,
    pub layer_count: u32,
    pub source_top_left: [i32; 3],
    pub source_bottom_right: [i32; 3],
    pub destination_top_left: [i32; 3],
    pub destination_bottom_right: [i32; 3],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderColorImageClear {
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderDepthStencilImageClear {
    pub base_array_layer: u32,
    pub layer_count: u32,
    pub clear_stencil: bool,
    pub clear_depth: bool,
}
