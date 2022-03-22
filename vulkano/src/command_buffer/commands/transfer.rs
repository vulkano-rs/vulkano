// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::{BufferAccess, BufferContents, BufferInner, TypedBufferAccess},
    command_buffer::{
        synced::{Command, KeyTy, SyncCommandBufferBuilder, SyncCommandBufferBuilderError},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder, CopyBufferError, CopyBufferImageError, CopyImageError,
        FillBufferError, UpdateBufferError,
    },
    device::{Device, DeviceOwned},
    format::{Format, NumericType},
    image::{ImageAccess, ImageAspect, ImageAspects, ImageDimensions, ImageLayout, SampleCount},
    sync::{AccessFlags, PipelineMemoryAccess, PipelineStages},
    DeviceSize, SafeDeref, VulkanObject,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{cmp, error, fmt, mem::size_of_val, sync::Arc};

/// # Commands to transfer data to a resource, either from the host or from another resource.
///
/// These commands can be called on a transfer queue, in addition to a compute or graphics queue.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Adds a command that copies from a buffer to another.
    ///
    /// This command will copy from the source to the destination. If their size is not equal, then
    /// the amount of data copied is equal to the smallest of the two.
    #[inline]
    pub fn copy_buffer<S, D, T>(
        &mut self,
        source: Arc<S>,
        destination: Arc<D>,
    ) -> Result<&mut Self, CopyBufferError>
    where
        S: TypedBufferAccess<Content = T> + 'static,
        D: TypedBufferAccess<Content = T> + 'static,
        T: ?Sized,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            let copy_size = cmp::min(source.size(), destination.size());
            check_copy_buffer(
                self.device(),
                source.as_ref(),
                destination.as_ref(),
                0,
                0,
                copy_size,
            )?;
            self.inner
                .copy_buffer(source, destination, [(0, 0, copy_size)])?;
            Ok(self)
        }
    }

    /// Adds a command that copies a range from the source to the destination buffer.
    /// Panics if out of bounds.
    #[inline]
    pub fn copy_buffer_dimensions<S, D, T>(
        &mut self,
        source: Arc<S>,
        source_offset: DeviceSize,
        destination: Arc<D>,
        destination_offset: DeviceSize,
        count: DeviceSize,
    ) -> Result<&mut Self, CopyBufferError>
    where
        S: TypedBufferAccess<Content = [T]> + 'static,
        D: TypedBufferAccess<Content = [T]> + 'static,
    {
        self.ensure_outside_render_pass()?;
        let size = std::mem::size_of::<T>() as DeviceSize;

        let source_offset = source_offset * size;
        let destination_offset = destination_offset * size;
        let copy_size = count * size;

        check_copy_buffer(
            self.device(),
            source.as_ref(),
            destination.as_ref(),
            source_offset,
            destination_offset,
            copy_size,
        )?;

        unsafe {
            self.inner.copy_buffer(
                source,
                destination,
                [(source_offset, destination_offset, copy_size)],
            )?;
        }
        Ok(self)
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image(
        &mut self,
        source: Arc<dyn BufferAccess>,
        destination: Arc<dyn ImageAccess>,
    ) -> Result<&mut Self, CopyBufferImageError> {
        self.ensure_outside_render_pass()?;

        let dims = destination.dimensions().width_height_depth();
        self.copy_buffer_to_image_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from a buffer to an image.
    pub fn copy_buffer_to_image_dimensions(
        &mut self,
        source: Arc<dyn BufferAccess>,
        destination: Arc<dyn ImageAccess>,
        offset: [u32; 3],
        size: [u32; 3],
        base_array_layer: u32,
        layer_count: u32,
        mip_level: u32,
    ) -> Result<&mut Self, CopyBufferImageError> {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_buffer_image(
                self.device(),
                source.as_ref(),
                destination.as_ref(),
                CheckCopyBufferImageTy::BufferToImage,
                offset,
                size,
                base_array_layer,
                layer_count,
                mip_level,
            )?;

            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_aspect: if destination.format().aspects().color {
                    ImageAspect::Color
                } else {
                    unimplemented!()
                },
                image_mip_level: mip_level,
                image_base_array_layer: base_array_layer,
                image_layer_count: layer_count,
                image_offset: [offset[0] as i32, offset[1] as i32, offset[2] as i32],
                image_extent: size,
            };

            self.inner.copy_buffer_to_image(
                source,
                destination,
                ImageLayout::TransferDstOptimal, // TODO: let choose layout
                [copy],
            )?;
            Ok(self)
        }
    }

    /// Adds a command that writes the content of a buffer.
    ///
    /// This function is similar to the `memset` function in C. The `data` parameter is a number
    /// that will be repeatedly written through the entire buffer.
    ///
    /// > **Note**: This function is technically safe because buffers can only contain integers or
    /// > floating point numbers, which are always valid whatever their memory representation is.
    /// > But unless your buffer actually contains only 32-bits integers, you are encouraged to use
    /// > this function only for zeroing the content of a buffer by passing `0` for the data.
    // TODO: not safe because of signalling NaNs
    #[inline]
    pub fn fill_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess>,
        data: u32,
    ) -> Result<&mut Self, FillBufferError> {
        unsafe {
            self.ensure_outside_render_pass()?;
            check_fill_buffer(self.device(), buffer.as_ref())?;
            self.inner.fill_buffer(buffer, data);
            Ok(self)
        }
    }

    /// Adds a command that writes data to a buffer.
    ///
    /// If `data` is larger than the buffer, only the part of `data` that fits is written. If the
    /// buffer is larger than `data`, only the start of the buffer is written.
    #[inline]
    pub fn update_buffer<B, D, Dd>(
        &mut self,
        buffer: Arc<B>,
        data: Dd,
    ) -> Result<&mut Self, UpdateBufferError>
    where
        B: TypedBufferAccess<Content = D> + 'static,
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        unsafe {
            self.ensure_outside_render_pass()?;
            check_update_buffer(self.device(), buffer.as_ref(), data.deref())?;

            let size_of_data = size_of_val(data.deref()) as DeviceSize;
            if buffer.size() >= size_of_data {
                self.inner.update_buffer(buffer, data);
            } else {
                unimplemented!() // TODO:
                                 //self.inner.update_buffer(buffer.slice(0 .. size_of_data), data);
            }

            Ok(self)
        }
    }

    /// Adds a command that copies an image to another.
    ///
    /// Copy operations have several restrictions:
    ///
    /// - Copy operations are only allowed on queue families that support transfer, graphics, or
    ///   compute operations.
    /// - The number of samples in the source and destination images must be equal.
    /// - The size of the uncompressed element format of the source image must be equal to the
    ///   compressed element format of the destination.
    /// - If you copy between depth, stencil or depth-stencil images, the format of both images
    ///   must match exactly.
    /// - For two-dimensional images, the Z coordinate must be 0 for the image offsets and 1 for
    ///   the extent. Same for the Y coordinate for one-dimensional images.
    /// - For non-array images, the base array layer must be 0 and the number of layers must be 1.
    ///
    /// If `layer_count` is greater than 1, the copy will happen between each individual layer as
    /// if they were separate images.
    ///
    /// # Panic
    ///
    /// - Panics if the source or the destination was not created with `device`.
    ///
    pub fn copy_image(
        &mut self,
        source: Arc<dyn ImageAccess>,
        source_offset: [i32; 3],
        source_base_array_layer: u32,
        source_mip_level: u32,
        destination: Arc<dyn ImageAccess>,
        destination_offset: [i32; 3],
        destination_base_array_layer: u32,
        destination_mip_level: u32,
        extent: [u32; 3],
        layer_count: u32,
    ) -> Result<&mut Self, CopyImageError> {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_image(
                self.device(),
                source.as_ref(),
                source_offset,
                source_base_array_layer,
                source_mip_level,
                destination.as_ref(),
                destination_offset,
                destination_base_array_layer,
                destination_mip_level,
                extent,
                layer_count,
            )?;

            let source_aspects = source.format().aspects();
            let destination_aspects = destination.format().aspects();
            let copy = UnsafeCommandBufferBuilderImageCopy {
                // TODO: Allowing choosing a subset of the image aspects, but note that if color
                // is included, neither depth nor stencil may.
                aspects: ImageAspects {
                    color: source_aspects.color,
                    depth: !source_aspects.color
                        && source_aspects.depth
                        && destination_aspects.depth,
                    stencil: !source_aspects.color
                        && source_aspects.stencil
                        && destination_aspects.stencil,
                    ..ImageAspects::none()
                },
                source_mip_level,
                destination_mip_level,
                source_base_array_layer,
                destination_base_array_layer,
                layer_count,
                source_offset,
                destination_offset,
                extent,
            };

            // TODO: Allow choosing layouts, but note that only Transfer*Optimal and General are
            // valid.
            if source.conflict_key() == destination.conflict_key() {
                // since we are copying from the same image, we must use the same layout
                self.inner.copy_image(
                    source,
                    ImageLayout::General,
                    destination,
                    ImageLayout::General,
                    [copy],
                )?;
            } else {
                self.inner.copy_image(
                    source,
                    ImageLayout::TransferSrcOptimal,
                    destination,
                    ImageLayout::TransferDstOptimal,
                    [copy],
                )?;
            }
            Ok(self)
        }
    }

    /// Adds a command that copies from an image to a buffer.
    // The data layout of the image on the gpu is opaque, as in, it is non of our business how the gpu stores the image.
    // This does not matter since the act of copying the image into a buffer converts it to linear form.
    pub fn copy_image_to_buffer(
        &mut self,
        source: Arc<dyn ImageAccess>,
        destination: Arc<dyn BufferAccess>,
    ) -> Result<&mut Self, CopyBufferImageError> {
        self.ensure_outside_render_pass()?;

        let dims = source.dimensions().width_height_depth();
        self.copy_image_to_buffer_dimensions(source, destination, [0, 0, 0], dims, 0, 1, 0)
    }

    /// Adds a command that copies from an image to a buffer.
    pub fn copy_image_to_buffer_dimensions(
        &mut self,
        source: Arc<dyn ImageAccess>,
        destination: Arc<dyn BufferAccess>,
        offset: [u32; 3],
        size: [u32; 3],
        base_array_layer: u32,
        layer_count: u32,
        mip_level: u32,
    ) -> Result<&mut Self, CopyBufferImageError> {
        unsafe {
            self.ensure_outside_render_pass()?;

            check_copy_buffer_image(
                self.device(),
                destination.as_ref(),
                source.as_ref(),
                CheckCopyBufferImageTy::ImageToBuffer,
                offset,
                size,
                base_array_layer,
                layer_count,
                mip_level,
            )?;

            let source_aspects = source.format().aspects();
            let copy = UnsafeCommandBufferBuilderBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                // TODO: Allow the user to choose aspect
                image_aspect: if source_aspects.color {
                    ImageAspect::Color
                } else if source_aspects.depth {
                    ImageAspect::Depth
                } else if source_aspects.stencil {
                    ImageAspect::Stencil
                } else {
                    unimplemented!()
                },
                image_mip_level: mip_level,
                image_base_array_layer: base_array_layer,
                image_layer_count: layer_count,
                image_offset: [offset[0] as i32, offset[1] as i32, offset[2] as i32],
                image_extent: size,
            };

            self.inner.copy_image_to_buffer(
                source,
                ImageLayout::TransferSrcOptimal,
                destination, // TODO: let choose layout
                [copy],
            )?;
            Ok(self)
        }
    }
}

/// Checks whether a copy buffer command is valid.
///
/// # Panic
///
/// - Panics if the source and destination were not created with `device`.
///
fn check_copy_buffer(
    device: &Device,
    source: &dyn BufferAccess,
    destination: &dyn BufferAccess,
    source_offset: DeviceSize,
    destination_offset: DeviceSize,
    size: DeviceSize,
) -> Result<(), CheckCopyBufferError> {
    assert_eq!(
        source.inner().buffer.device().internal_object(),
        device.internal_object()
    );
    assert_eq!(
        destination.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !source.inner().buffer.usage().transfer_source {
        return Err(CheckCopyBufferError::SourceMissingTransferUsage);
    }

    if !destination.inner().buffer.usage().transfer_destination {
        return Err(CheckCopyBufferError::DestinationMissingTransferUsage);
    }

    if source_offset + size > source.size() {
        return Err(CheckCopyBufferError::SourceOutOfBounds);
    }

    if destination_offset + size > destination.size() {
        return Err(CheckCopyBufferError::DestinationOutOfBounds);
    }

    if source.conflict_key() == destination.conflict_key()
        && is_overlapping_ranges(source_offset, size, destination_offset, size)
    {
        return Err(CheckCopyBufferError::OverlappingRanges);
    }

    Ok(())
}

/// Error that can happen from `check_copy_buffer`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyBufferError {
    /// The source buffer is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination buffer is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The source and destination ranges are overlapping.
    OverlappingRanges,
    /// The source range is out of bounds.
    SourceOutOfBounds,
    /// The destination range is out of bounds.
    DestinationOutOfBounds,
}

impl error::Error for CheckCopyBufferError {}

impl fmt::Display for CheckCopyBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckCopyBufferError::SourceMissingTransferUsage => {
                    "the source buffer is missing the transfer source usage"
                }
                CheckCopyBufferError::DestinationMissingTransferUsage => {
                    "the destination buffer is missing the transfer destination usage"
                }
                CheckCopyBufferError::OverlappingRanges =>
                    "the source and destination ranges are overlapping",
                CheckCopyBufferError::SourceOutOfBounds => "the source range is out of bounds",
                CheckCopyBufferError::DestinationOutOfBounds => {
                    "the destination range is out of bounds"
                }
            }
        )
    }
}

/// Checks whether a fill buffer command is valid.
///
/// # Panic
///
/// - Panics if the buffer not created with `device`.
///
fn check_fill_buffer<B>(device: &Device, buffer: &B) -> Result<(), CheckFillBufferError>
where
    B: ?Sized + BufferAccess,
{
    assert_eq!(
        buffer.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !buffer.inner().buffer.usage().transfer_destination {
        return Err(CheckFillBufferError::BufferMissingUsage);
    }

    if buffer.inner().offset % 4 != 0 {
        return Err(CheckFillBufferError::WrongAlignment);
    }

    Ok(())
}

/// Error that can happen when attempting to add a `fill_buffer` command.
#[derive(Debug, Copy, Clone)]
pub enum CheckFillBufferError {
    /// The "transfer destination" usage must be enabled on the buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
}

impl error::Error for CheckFillBufferError {}

impl fmt::Display for CheckFillBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckFillBufferError::BufferMissingUsage => {
                    "the transfer destination usage must be enabled on the buffer"
                }
                CheckFillBufferError::WrongAlignment =>
                    "the offset or size are not aligned to 4 bytes",
            }
        )
    }
}

/// Checks whether an update buffer command is valid.
///
/// # Panic
///
/// - Panics if the buffer not created with `device`.
///
fn check_update_buffer<D>(
    device: &Device,
    buffer: &dyn BufferAccess,
    data: &D,
) -> Result<(), CheckUpdateBufferError>
where
    D: ?Sized,
{
    assert_eq!(
        buffer.inner().buffer.device().internal_object(),
        device.internal_object()
    );

    if !buffer.inner().buffer.usage().transfer_destination {
        return Err(CheckUpdateBufferError::BufferMissingUsage);
    }

    if buffer.inner().offset % 4 != 0 {
        return Err(CheckUpdateBufferError::WrongAlignment);
    }

    let size = buffer.size().min(size_of_val(data) as DeviceSize);

    if size % 4 != 0 {
        return Err(CheckUpdateBufferError::WrongAlignment);
    }

    if size > 65536 {
        return Err(CheckUpdateBufferError::DataTooLarge);
    }

    Ok(())
}

/// Error that can happen when attempting to add an `update_buffer` command.
#[derive(Debug, Copy, Clone)]
pub enum CheckUpdateBufferError {
    /// The "transfer destination" usage must be enabled on the buffer.
    BufferMissingUsage,
    /// The data or size must be 4-bytes aligned.
    WrongAlignment,
    /// The data must not be larger than 64k bytes.
    DataTooLarge,
}

impl error::Error for CheckUpdateBufferError {}

impl fmt::Display for CheckUpdateBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckUpdateBufferError::BufferMissingUsage => {
                    "the transfer destination usage must be enabled on the buffer"
                }
                CheckUpdateBufferError::WrongAlignment => {
                    "the offset or size are not aligned to 4 bytes"
                }
                CheckUpdateBufferError::DataTooLarge => "data is too large",
            }
        )
    }
}

/// Checks whether a copy image command is valid.
///
/// Note that this doesn't check whether `layer_count` is equal to 0. TODO: change that?
///
/// # Panic
///
/// - Panics if the source or the destination was not created with `device`.
///
fn check_copy_image<S, D>(
    device: &Device,
    source: &S,
    source_offset: [i32; 3],
    source_base_array_layer: u32,
    source_mip_level: u32,
    destination: &D,
    destination_offset: [i32; 3],
    destination_base_array_layer: u32,
    destination_mip_level: u32,
    extent: [u32; 3],
    layer_count: u32,
) -> Result<(), CheckCopyImageError>
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
        return Err(CheckCopyImageError::MissingTransferSourceUsage);
    }

    if !destination_inner.image.usage().transfer_destination {
        return Err(CheckCopyImageError::MissingTransferDestinationUsage);
    }

    if source.samples() != destination.samples() {
        return Err(CheckCopyImageError::SampleCountMismatch);
    }

    if let (Some(source_type), Some(destination_type)) = (
        source.format().type_color(),
        destination.format().type_color(),
    ) {
        // TODO: The correct check here is that the uncompressed element size of the source is
        // equal to the compressed element size of the destination.  However, format doesn't
        // currently expose this information, so to be safe, we simply disallow compressed formats.
        if source.format().compression().is_some()
            || destination.format().compression().is_some()
            || (source.format().block_size() != destination.format().block_size())
        {
            return Err(CheckCopyImageError::SizeIncompatibleFormatTypes {
                source_type,
                destination_type,
            });
        }
    } else {
        if source.format() != destination.format() {
            return Err(CheckCopyImageError::DepthStencilFormatMismatch);
        }
    }

    let source_dimensions = match source.dimensions().mip_level_dimensions(source_mip_level) {
        Some(d) => d,
        None => return Err(CheckCopyImageError::SourceCoordinatesOutOfRange),
    };

    let destination_dimensions = match destination
        .dimensions()
        .mip_level_dimensions(destination_mip_level)
    {
        Some(d) => d,
        None => return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange),
    };

    if source_base_array_layer + layer_count > source_dimensions.array_layers() {
        return Err(CheckCopyImageError::SourceCoordinatesOutOfRange);
    }

    if destination_base_array_layer + layer_count > destination_dimensions.array_layers() {
        return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange);
    }

    if source_offset[0] < 0 || source_offset[0] as u32 + extent[0] > source_dimensions.width() {
        return Err(CheckCopyImageError::SourceCoordinatesOutOfRange);
    }

    if source_offset[1] < 0 || source_offset[1] as u32 + extent[1] > source_dimensions.height() {
        return Err(CheckCopyImageError::SourceCoordinatesOutOfRange);
    }

    if source_offset[2] < 0 || source_offset[2] as u32 + extent[2] > source_dimensions.depth() {
        return Err(CheckCopyImageError::SourceCoordinatesOutOfRange);
    }

    if destination_offset[0] < 0
        || destination_offset[0] as u32 + extent[0] > destination_dimensions.width()
    {
        return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_offset[1] < 0
        || destination_offset[1] as u32 + extent[1] > destination_dimensions.height()
    {
        return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange);
    }

    if destination_offset[2] < 0
        || destination_offset[2] as u32 + extent[2] > destination_dimensions.depth()
    {
        return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange);
    }

    match source_dimensions {
        ImageDimensions::Dim1d { .. } => {
            if source_offset[1] != 0 || extent[1] != 1 {
                return Err(CheckCopyImageError::IncompatibleRangeForImageType);
            }
            if source_offset[2] != 0 || extent[2] != 1 {
                return Err(CheckCopyImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            if source_offset[2] != 0 || extent[2] != 1 {
                return Err(CheckCopyImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim3d { .. } => {}
    }

    match destination_dimensions {
        ImageDimensions::Dim1d { .. } => {
            if destination_offset[1] != 0 || extent[1] != 1 {
                return Err(CheckCopyImageError::IncompatibleRangeForImageType);
            }
            if destination_offset[2] != 0 || extent[2] != 1 {
                return Err(CheckCopyImageError::IncompatibleRangeForImageType);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            if destination_offset[2] != 0 || extent[2] != 1 {
                return Err(CheckCopyImageError::IncompatibleRangeForImageType);
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
            // since both images are the same, we can use any dimensions type
            && is_overlapping_regions(source_offset, extent, destination_offset, extent, source_dimensions)
        {
            return Err(CheckCopyImageError::OverlappingRegions);
        }
    }

    Ok(())
}

/// Error that can happen from `check_copy_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyImageError {
    /// The source is missing the transfer source usage.
    MissingTransferSourceUsage,
    /// The destination is missing the transfer destination usage.
    MissingTransferDestinationUsage,
    /// The number of samples in the source and destination do not match.
    SampleCountMismatch,
    /// The format of the source and destination must be equal when copying depth/stencil images.
    DepthStencilFormatMismatch,
    /// The types of the source format and the destination format aren't size-compatible.
    SizeIncompatibleFormatTypes {
        source_type: NumericType,
        destination_type: NumericType,
    },
    /// The offsets, array layers and/or mipmap levels are out of range in the source image.
    SourceCoordinatesOutOfRange,
    /// The offsets, array layers and/or mipmap levels are out of range in the destination image.
    DestinationCoordinatesOutOfRange,
    /// The offsets or extent are incompatible with the image type.
    IncompatibleRangeForImageType,
    /// The source and destination regions are overlapping.
    OverlappingRegions,
}

impl error::Error for CheckCopyImageError {}

impl fmt::Display for CheckCopyImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckCopyImageError::MissingTransferSourceUsage => {
                    "the source is missing the transfer source usage"
                }
                CheckCopyImageError::MissingTransferDestinationUsage => {
                    "the destination is missing the transfer destination usage"
                }
                CheckCopyImageError::SampleCountMismatch => {
                    "the number of samples in the source and destination do not match"
                }
                CheckCopyImageError::DepthStencilFormatMismatch => {
                    "the format of the source and destination must be equal when copying \
                 depth/stencil images"
                }
                CheckCopyImageError::SizeIncompatibleFormatTypes { .. } => {
                    "the types of the source format and the destination format aren't size-compatible"
                }
                CheckCopyImageError::SourceCoordinatesOutOfRange => {
                    "the offsets, array layers and/or mipmap levels are out of range in the source \
                 image"
                }
                CheckCopyImageError::DestinationCoordinatesOutOfRange => {
                    "the offsets, array layers and/or mipmap levels are out of range in the \
                 destination image"
                }
                CheckCopyImageError::IncompatibleRangeForImageType => {
                    "the offsets or extent are incompatible with the image type"
                }
                CheckCopyImageError::OverlappingRegions => {
                    "the source and destination regions are overlapping"
                }
            }
        )
    }
}

/// Type of operation to check.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CheckCopyBufferImageTy {
    BufferToImage,
    ImageToBuffer,
}

/// Checks whether a copy buffer-image command is valid. Can check both buffer-to-image copies and
/// image-to-buffer copies.
///
/// # Panic
///
/// - Panics if the buffer and image were not created with `device`.
///
fn check_copy_buffer_image(
    device: &Device,
    buffer: &dyn BufferAccess,
    image: &dyn ImageAccess,
    ty: CheckCopyBufferImageTy,
    image_offset: [u32; 3],
    image_size: [u32; 3],
    image_first_layer: u32,
    image_num_layers: u32,
    image_mipmap: u32,
) -> Result<(), CheckCopyBufferImageError> {
    let buffer_inner = buffer.inner();
    let image_inner = image.inner();

    assert_eq!(
        buffer_inner.buffer.device().internal_object(),
        device.internal_object()
    );
    assert_eq!(
        image_inner.image.device().internal_object(),
        device.internal_object()
    );

    match ty {
        CheckCopyBufferImageTy::BufferToImage => {
            if !buffer_inner.buffer.usage().transfer_source {
                return Err(CheckCopyBufferImageError::SourceMissingTransferUsage);
            }
            if !image_inner.image.usage().transfer_destination {
                return Err(CheckCopyBufferImageError::DestinationMissingTransferUsage);
            }
        }
        CheckCopyBufferImageTy::ImageToBuffer => {
            if !image_inner.image.usage().transfer_source {
                return Err(CheckCopyBufferImageError::SourceMissingTransferUsage);
            }
            if !buffer_inner.buffer.usage().transfer_destination {
                return Err(CheckCopyBufferImageError::DestinationMissingTransferUsage);
            }
        }
    }

    if image.samples() != SampleCount::Sample1 {
        return Err(CheckCopyBufferImageError::UnexpectedMultisampled);
    }

    let image_dimensions = match image.dimensions().mip_level_dimensions(image_mipmap) {
        Some(d) => d,
        None => return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange),
    };

    if image_first_layer + image_num_layers > image_dimensions.array_layers() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    if image_offset[0] + image_size[0] > image_dimensions.width() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    if image_offset[1] + image_size[1] > image_dimensions.height() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    if image_offset[2] + image_size[2] > image_dimensions.depth() {
        return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
    }

    match image.dimensions() {
        ImageDimensions::Dim1d { .. } => {
            // VUID-vkCmdCopyBufferToImage-srcImage-00199
            if image_offset[1] != 0 || image_size[1] != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }

            // VUID-vkCmdCopyBufferToImage-srcImage-00201
            if image_offset[2] != 0 || image_size[2] != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }
        }
        ImageDimensions::Dim2d { .. } => {
            // VUID-vkCmdCopyBufferToImage-srcImage-00201
            if image_offset[2] != 0 || image_size[2] != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }
        }
        ImageDimensions::Dim3d { .. } => {
            // VUID-vkCmdCopyBufferToImage-baseArrayLayer-00213
            if image_first_layer != 0 || image_num_layers != 1 {
                return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange);
            }
        }
    }

    let required_size = required_size_for_format(image.format(), image_size, image_num_layers);
    if required_size > buffer.size() {
        return Err(CheckCopyBufferImageError::BufferTooSmall {
            required_size,
            actual_size: buffer.size(),
        });
    }

    // TODO: check memory overlap?

    Ok(())
}

/// Computes the minimum required len in elements for buffer with image data in specified
/// format of specified size.
fn required_size_for_format(format: Format, extent: [u32; 3], layer_count: u32) -> DeviceSize {
    let num_blocks = extent
        .into_iter()
        .zip(format.block_extent())
        .map(|(extent, block_extent)| {
            let extent = extent as DeviceSize;
            let block_extent = block_extent as DeviceSize;
            (extent + block_extent - 1) / block_extent
        })
        .product::<DeviceSize>()
        * layer_count as DeviceSize;
    let block_size = format
        .block_size()
        .expect("this format cannot accept pixels");
    num_blocks * block_size
}

/// Error that can happen from `check_copy_buffer_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyBufferImageError {
    /// The source buffer or image is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination buffer or image is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The source and destination are overlapping.
    OverlappingRanges,
    /// The image must not be multisampled.
    UnexpectedMultisampled,
    /// The image coordinates are out of range.
    ImageCoordinatesOutOfRange,
    /// The buffer is too small for the copy operation.
    BufferTooSmall {
        /// Required size of the buffer.
        required_size: DeviceSize,
        /// Actual size of the buffer.
        actual_size: DeviceSize,
    },
}

impl error::Error for CheckCopyBufferImageError {}

impl fmt::Display for CheckCopyBufferImageError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckCopyBufferImageError::SourceMissingTransferUsage => {
                    "the source buffer is missing the transfer source usage"
                }
                CheckCopyBufferImageError::DestinationMissingTransferUsage => {
                    "the destination buffer is missing the transfer destination usage"
                }
                CheckCopyBufferImageError::OverlappingRanges => {
                    "the source and destination are overlapping"
                }
                CheckCopyBufferImageError::UnexpectedMultisampled => {
                    "the image must not be multisampled"
                }
                CheckCopyBufferImageError::ImageCoordinatesOutOfRange => {
                    "the image coordinates are out of range"
                }
                CheckCopyBufferImageError::BufferTooSmall { .. } => {
                    "the buffer is too small for the copy operation"
                }
            }
        )
    }
}

/// Checks whether the range `source`..`source + size` is overlapping with the range `destination`..`destination + size`.
/// TODO: add unit tests
pub(super) fn is_overlapping_ranges(
    source: u64,
    source_size: u64,
    destination: u64,
    destination_size: u64,
) -> bool {
    (destination < source + source_size) && (source < destination + destination_size)
}

/// Checks whether there is an overlap between the source and destination regions.
/// The `image_dim` is used to determine the number of dimentions and not the image size.
/// TODO: add unit tests
pub(super) fn is_overlapping_regions(
    source_offset: [i32; 3],
    source_extent: [u32; 3],
    destination_offset: [i32; 3],
    destination_extent: [u32; 3],
    image_dim: ImageDimensions,
) -> bool {
    let dim = match image_dim {
        ImageDimensions::Dim1d { .. } => 1,
        ImageDimensions::Dim2d { .. } => 2,
        ImageDimensions::Dim3d { .. } => 3,
    };
    let mut result = true;
    // for 1d, it will check x only, for 2d x and y, and so on...
    for i in 0..dim {
        result &= is_overlapping_ranges(
            source_offset[i] as u64,
            source_extent[i] as u64,
            destination_offset[i] as u64,
            destination_extent[i] as u64,
        );
    }
    result
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer<R>(
        &mut self,
        source: Arc<dyn BufferAccess>,
        destination: Arc<dyn BufferAccess>,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = (DeviceSize, DeviceSize, DeviceSize)> + Send + Sync + 'static,
    {
        struct Cmd<R> {
            source: Arc<dyn BufferAccess>,
            destination: Arc<dyn BufferAccess>,
            regions: Mutex<Option<R>>,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = (DeviceSize, DeviceSize, DeviceSize)> + Send + Sync,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyBuffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_buffer(
                    self.source.as_ref(),
                    self.destination.as_ref(),
                    self.regions.lock().take().unwrap(),
                );
            }
        }

        let mut resources: SmallVec<[_; 2]> = SmallVec::new();

        // if its the same image in source and destination, we need to lock it once
        if source.conflict_key() == destination.conflict_key() {
            resources.push((
                KeyTy::Buffer(source.clone()),
                "source_and_destination".into(),
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            transfer: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlags {
                            transfer_read: true,
                            transfer_write: true,
                            ..AccessFlags::none()
                        },
                        exclusive: false,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            ));
        } else {
            resources.extend([
                (
                    KeyTy::Buffer(source.clone()),
                    "source".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
                (
                    KeyTy::Buffer(destination.clone()),
                    "destination".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
            ]);
        }

        self.append_command(
            Cmd {
                source,
                destination,
                regions: Mutex::new(Some(regions)),
            },
            resources,
        )?;

        Ok(())
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image<R>(
        &mut self,
        source: Arc<dyn BufferAccess>,
        destination: Arc<dyn ImageAccess>,
        destination_layout: ImageLayout,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync + 'static,
    {
        struct Cmd<R> {
            source: Arc<dyn BufferAccess>,
            destination: Arc<dyn ImageAccess>,
            destination_layout: ImageLayout,
            regions: Mutex<Option<R>>,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyBufferToImage"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_buffer_to_image(
                    self.source.as_ref(),
                    self.destination.as_ref(),
                    self.destination_layout,
                    self.regions.lock().take().unwrap(),
                );
            }
        }

        self.append_command(
            Cmd {
                source: source.clone(),
                destination: destination.clone(),
                destination_layout,
                regions: Mutex::new(Some(regions)),
            },
            [
                (
                    KeyTy::Buffer(source),
                    "source".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
                (
                    KeyTy::Image(destination),
                    "destination".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        destination_layout,
                        destination_layout,
                    )),
                ),
            ],
        )?;

        Ok(())
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer(&mut self, buffer: Arc<dyn BufferAccess>, data: u32) {
        struct Cmd {
            buffer: Arc<dyn BufferAccess>,
            data: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdFillBuffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.fill_buffer(self.buffer.as_ref(), self.data);
            }
        }

        self.append_command(
            Cmd {
                buffer: buffer.clone(),
                data,
            },
            [(
                KeyTy::Buffer(buffer),
                "destination".into(),
                Some((
                    PipelineMemoryAccess {
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
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )
        .unwrap();
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<D, Dd>(&mut self, buffer: Arc<dyn BufferAccess>, data: Dd)
    where
        D: BufferContents + ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        struct Cmd<Dd> {
            buffer: Arc<dyn BufferAccess>,
            data: Dd,
        }

        impl<D, Dd> Command for Cmd<Dd>
        where
            D: BufferContents + ?Sized,
            Dd: SafeDeref<Target = D> + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdUpdateBuffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.update_buffer(self.buffer.as_ref(), self.data.deref());
            }
        }

        self.append_command(
            Cmd {
                buffer: buffer.clone(),
                data,
            },
            [(
                KeyTy::Buffer(buffer),
                "destination".into(),
                Some((
                    PipelineMemoryAccess {
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
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )
        .unwrap();
    }

    /// Calls `vkCmdCopyImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image<R>(
        &mut self,
        source: Arc<dyn ImageAccess>,
        source_layout: ImageLayout,
        destination: Arc<dyn ImageAccess>,
        destination_layout: ImageLayout,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = UnsafeCommandBufferBuilderImageCopy> + Send + Sync + 'static,
    {
        struct Cmd<R> {
            source: Arc<dyn ImageAccess>,
            source_layout: ImageLayout,
            destination: Arc<dyn ImageAccess>,
            destination_layout: ImageLayout,
            regions: Mutex<Option<R>>,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = UnsafeCommandBufferBuilderImageCopy> + Send + Sync,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyImage"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_image(
                    self.source.as_ref(),
                    self.source_layout,
                    self.destination.as_ref(),
                    self.destination_layout,
                    self.regions.lock().take().unwrap(),
                );
            }
        }

        let mut resources: SmallVec<[_; 2]> = SmallVec::new();

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
        if source_key == destination_key {
            resources.push((
                KeyTy::Image(source.clone()),
                "source_and_destination".into(),
                Some((
                    PipelineMemoryAccess {
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
                    // TODO: should, we take the layout as parameter? if so, which? source or destination?
                    ImageLayout::General,
                    ImageLayout::General,
                )),
            ));
        } else {
            resources.extend([
                (
                    KeyTy::Image(source.clone()),
                    "source".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        source_layout,
                        source_layout,
                    )),
                ),
                (
                    KeyTy::Image(destination.clone()),
                    "destination".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        destination_layout,
                        destination_layout,
                    )),
                ),
            ]);
        }

        self.append_command(
            Cmd {
                source,
                source_layout,
                destination,
                destination_layout,
                regions: Mutex::new(Some(regions)),
            },
            resources,
        )?;

        Ok(())
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer<R>(
        &mut self,
        source: Arc<dyn ImageAccess>,
        source_layout: ImageLayout,
        destination: Arc<dyn BufferAccess>,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        R: IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync + 'static,
    {
        struct Cmd<R> {
            source: Arc<dyn ImageAccess>,
            source_layout: ImageLayout,
            destination: Arc<dyn BufferAccess>,
            regions: Mutex<Option<R>>,
        }

        impl<R> Command for Cmd<R>
        where
            R: IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyImageToBuffer"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_image_to_buffer(
                    self.source.as_ref(),
                    self.source_layout,
                    self.destination.as_ref(),
                    self.regions.lock().take().unwrap(),
                );
            }
        }

        self.append_command(
            Cmd {
                source: source.clone(),
                destination: destination.clone(),
                source_layout,
                regions: Mutex::new(Some(regions)),
            },
            [
                (
                    KeyTy::Image(source),
                    "source".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        source_layout,
                        source_layout,
                    )),
                ),
                (
                    KeyTy::Buffer(destination),
                    "destination".into(),
                    Some((
                        PipelineMemoryAccess {
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
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
            ],
        )?;

        Ok(())
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer(
        &mut self,
        source: &dyn BufferAccess,
        destination: &dyn BufferAccess,
        regions: impl IntoIterator<Item = (DeviceSize, DeviceSize, DeviceSize)>,
    ) {
        // TODO: debug assert that there's no overlap in the destinations?

        let source = source.inner();
        debug_assert!(source.offset < source.buffer.size());
        debug_assert!(source.buffer.usage().transfer_source);

        let destination = destination.inner();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage().transfer_destination);

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .map(|(sr, de, sz)| ash::vk::BufferCopy {
                src_offset: sr + source.offset,
                dst_offset: de + destination.offset,
                size: sz,
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_copy_buffer(
            self.handle,
            source.buffer.internal_object(),
            destination.buffer.internal_object(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        source: &dyn BufferAccess,
        destination: &dyn ImageAccess,
        destination_layout: ImageLayout,
        regions: impl IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>,
    ) {
        let source = source.inner();
        debug_assert!(source.offset < source.buffer.size());
        debug_assert!(source.buffer.usage().transfer_source);

        debug_assert_eq!(destination.samples(), SampleCount::Sample1);
        let destination = destination.inner();
        debug_assert!(destination.image.usage().transfer_destination);
        debug_assert!(
            destination_layout == ImageLayout::General
                || destination_layout == ImageLayout::TransferDstOptimal
        );

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .map(|copy| {
                debug_assert!(copy.image_layer_count <= destination.num_layers as u32);
                debug_assert!(copy.image_mip_level < destination.num_mipmap_levels as u32);

                ash::vk::BufferImageCopy {
                    buffer_offset: source.offset + copy.buffer_offset,
                    buffer_row_length: copy.buffer_row_length,
                    buffer_image_height: copy.buffer_image_height,
                    image_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.image_aspect.into(),
                        mip_level: copy.image_mip_level + destination.first_mipmap_level as u32,
                        base_array_layer: copy.image_base_array_layer
                            + destination.first_layer as u32,
                        layer_count: copy.image_layer_count,
                    },
                    image_offset: ash::vk::Offset3D {
                        x: copy.image_offset[0],
                        y: copy.image_offset[1],
                        z: copy.image_offset[2],
                    },
                    image_extent: ash::vk::Extent3D {
                        width: copy.image_extent[0],
                        height: copy.image_extent[1],
                        depth: copy.image_extent[2],
                    },
                }
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_copy_buffer_to_image(
            self.handle,
            source.buffer.internal_object(),
            destination.image.internal_object(),
            destination_layout.into(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer(&mut self, buffer: &dyn BufferAccess, data: u32) {
        let fns = self.device.fns();

        let size = buffer.size();

        let (buffer_handle, offset) = {
            let BufferInner {
                buffer: buffer_inner,
                offset,
            } = buffer.inner();
            debug_assert!(buffer_inner.usage().transfer_destination);
            debug_assert_eq!(offset % 4, 0);
            (buffer_inner.internal_object(), offset)
        };

        fns.v1_0
            .cmd_fill_buffer(self.handle, buffer_handle, offset, size, data);
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<D>(&mut self, buffer: &dyn BufferAccess, data: &D)
    where
        D: BufferContents + ?Sized,
    {
        let fns = self.device.fns();

        let size = buffer.size();
        debug_assert_eq!(size % 4, 0);
        debug_assert!(size <= 65536);
        debug_assert!(size <= size_of_val(data) as DeviceSize);

        let (buffer_handle, offset) = {
            let BufferInner {
                buffer: buffer_inner,
                offset,
            } = buffer.inner();
            debug_assert!(buffer_inner.usage().transfer_destination);
            debug_assert_eq!(offset % 4, 0);
            (buffer_inner.internal_object(), offset)
        };

        fns.v1_0.cmd_update_buffer(
            self.handle,
            buffer_handle,
            offset,
            size,
            data.as_bytes().as_ptr() as *const _,
        );
    }

    /// Calls `vkCmdCopyImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image(
        &mut self,
        source: &dyn ImageAccess,
        source_layout: ImageLayout,
        destination: &dyn ImageAccess,
        destination_layout: ImageLayout,
        regions: impl IntoIterator<Item = UnsafeCommandBufferBuilderImageCopy>,
    ) {
        // TODO: The correct check here is that the uncompressed element size of the source is
        // equal to the compressed element size of the destination.
        debug_assert!(
            source.format().compression().is_some()
                || destination.format().compression().is_some()
                || source.format().block_size() == destination.format().block_size()
        );

        // Depth/Stencil formats are required to match exactly.
        let source_aspects = source.format().aspects();
        debug_assert!(
            !source_aspects.depth && !source_aspects.stencil
                || source.format() == destination.format()
        );

        debug_assert_eq!(source.samples(), destination.samples());
        let source = source.inner();
        debug_assert!(source.image.usage().transfer_source);
        debug_assert!(
            source_layout == ImageLayout::General
                || source_layout == ImageLayout::TransferSrcOptimal
        );

        let destination = destination.inner();
        debug_assert!(destination.image.usage().transfer_destination);
        debug_assert!(
            destination_layout == ImageLayout::General
                || destination_layout == ImageLayout::TransferDstOptimal
        );

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .filter_map(|copy| {
                // TODO: not everything is checked here
                debug_assert!(
                    copy.source_base_array_layer + copy.layer_count <= source.num_layers as u32
                );
                debug_assert!(
                    copy.destination_base_array_layer + copy.layer_count
                        <= destination.num_layers as u32
                );
                debug_assert!(copy.source_mip_level < destination.num_mipmap_levels as u32);
                debug_assert!(copy.destination_mip_level < destination.num_mipmap_levels as u32);

                if copy.layer_count == 0 {
                    return None;
                }

                Some(ash::vk::ImageCopy {
                    src_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.aspects.into(),
                        mip_level: copy.source_mip_level,
                        base_array_layer: copy.source_base_array_layer + source.first_layer as u32,
                        layer_count: copy.layer_count,
                    },
                    src_offset: ash::vk::Offset3D {
                        x: copy.source_offset[0],
                        y: copy.source_offset[1],
                        z: copy.source_offset[2],
                    },
                    dst_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.aspects.into(),
                        mip_level: copy.destination_mip_level,
                        base_array_layer: copy.destination_base_array_layer
                            + destination.first_layer as u32,
                        layer_count: copy.layer_count,
                    },
                    dst_offset: ash::vk::Offset3D {
                        x: copy.destination_offset[0],
                        y: copy.destination_offset[1],
                        z: copy.destination_offset[2],
                    },
                    extent: ash::vk::Extent3D {
                        width: copy.extent[0],
                        height: copy.extent[1],
                        depth: copy.extent[2],
                    },
                })
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_copy_image(
            self.handle,
            source.image.internal_object(),
            source_layout.into(),
            destination.image.internal_object(),
            destination_layout.into(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        source: &dyn ImageAccess,
        source_layout: ImageLayout,
        destination: &dyn BufferAccess,
        regions: impl IntoIterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>,
    ) {
        debug_assert_eq!(source.samples(), SampleCount::Sample1);
        let source = source.inner();
        debug_assert!(source.image.usage().transfer_source);
        debug_assert!(
            source_layout == ImageLayout::General
                || source_layout == ImageLayout::TransferSrcOptimal
        );

        let destination = destination.inner();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage().transfer_destination);

        let regions: SmallVec<[_; 8]> = regions
            .into_iter()
            .map(|copy| {
                debug_assert!(copy.image_layer_count <= source.num_layers as u32);
                debug_assert!(copy.image_mip_level < source.num_mipmap_levels as u32);

                ash::vk::BufferImageCopy {
                    buffer_offset: destination.offset + copy.buffer_offset,
                    buffer_row_length: copy.buffer_row_length,
                    buffer_image_height: copy.buffer_image_height,
                    image_subresource: ash::vk::ImageSubresourceLayers {
                        aspect_mask: copy.image_aspect.into(),
                        mip_level: copy.image_mip_level + source.first_mipmap_level as u32,
                        base_array_layer: copy.image_base_array_layer + source.first_layer as u32,
                        layer_count: copy.image_layer_count,
                    },
                    image_offset: ash::vk::Offset3D {
                        x: copy.image_offset[0],
                        y: copy.image_offset[1],
                        z: copy.image_offset[2],
                    },
                    image_extent: ash::vk::Extent3D {
                        width: copy.image_extent[0],
                        height: copy.image_extent[1],
                        depth: copy.image_extent[2],
                    },
                }
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.v1_0.cmd_copy_image_to_buffer(
            self.handle,
            source.image.internal_object(),
            source_layout.into(),
            destination.buffer.internal_object(),
            regions.len() as u32,
            regions.as_ptr(),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderBufferImageCopy {
    pub buffer_offset: DeviceSize,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_aspect: ImageAspect,
    pub image_mip_level: u32,
    pub image_base_array_layer: u32,
    pub image_layer_count: u32,
    pub image_offset: [i32; 3],
    pub image_extent: [u32; 3],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageCopy {
    pub aspects: ImageAspects,
    pub source_mip_level: u32,
    pub destination_mip_level: u32,
    pub source_base_array_layer: u32,
    pub destination_base_array_layer: u32,
    pub layer_count: u32,
    pub source_offset: [i32; 3],
    pub destination_offset: [i32; 3],
    pub extent: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::{BufferUsage, CpuAccessibleBuffer},
        format::Format,
    };

    #[test]
    fn test_required_len_for_format() {
        // issue #1292
        assert_eq!(
            required_size_for_format(Format::BC1_RGB_UNORM_BLOCK, [2048, 2048, 1], 1),
            2097152
        );
        // other test cases
        assert_eq!(
            required_size_for_format(Format::R8G8B8A8_UNORM, [2048, 2048, 1], 1),
            16777216
        );
        assert_eq!(
            required_size_for_format(Format::R4G4_UNORM_PACK8, [512, 512, 1], 1),
            262144
        );
        assert_eq!(
            required_size_for_format(Format::R8G8B8_USCALED, [512, 512, 1], 1),
            786432
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::R32G32_UINT, [512, 512, 1], 1),
            2097152
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_8x8_UNORM_BLOCK, [512, 512, 1], 1),
            65536
        );
        assert_eq!(
            required_size_for_format(Format::ASTC_12x12_SRGB_BLOCK, [512, 512, 1], 1),
            29584
        );
    }

    #[test]
    fn missing_usage() {
        let (device, queue) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            0u32,
        )
        .unwrap();

        match check_fill_buffer(&device, buffer.as_ref()) {
            Err(CheckFillBufferError::BufferMissingUsage) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn wrong_device() {
        let (dev1, queue) = gfx_dev_and_queue!();
        let (dev2, _) = gfx_dev_and_queue!();
        let buffer = CpuAccessibleBuffer::from_data(dev1, BufferUsage::all(), false, 0u32).unwrap();

        assert_should_panic!({
            let _ = check_fill_buffer(&dev2, buffer.as_ref());
        });
    }
}
