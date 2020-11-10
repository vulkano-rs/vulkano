// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This module contains the `ensure_image_view_compatible` function, which verifies whether
//! an image view can be used as a render pass attachment.

use format::Format;
use framebuffer::RenderPassDesc;
use image::ImageViewAccess;
use std::error;
use std::fmt;

/// Checks whether the given image view is allowed to be the nth attachment of the given render
/// pass.
///
/// # Panic
///
/// Panics if the attachment number is out of range.
// TODO: add a specializable trait instead, that uses this function
// TODO: ImageView instead of ImageViewAccess?
pub fn ensure_image_view_compatible<Rp, I>(
    render_pass: &Rp,
    attachment_num: usize,
    image: &I,
) -> Result<(), IncompatibleRenderPassAttachmentError>
where
    Rp: ?Sized + RenderPassDesc,
    I: ?Sized + ImageViewAccess,
{
    let attachment_desc = render_pass
        .attachment_desc(attachment_num)
        .expect("Attachment num out of range");

    if image.format() != attachment_desc.format {
        return Err(IncompatibleRenderPassAttachmentError::FormatMismatch {
            expected: attachment_desc.format,
            obtained: image.format(),
        });
    }

    if image.samples() != attachment_desc.samples {
        return Err(IncompatibleRenderPassAttachmentError::SamplesMismatch {
            expected: attachment_desc.samples,
            obtained: image.samples(),
        });
    }

    if !image.identity_swizzle() {
        return Err(IncompatibleRenderPassAttachmentError::NotIdentitySwizzled);
    }

    for subpass_num in 0..render_pass.num_subpasses() {
        let subpass = render_pass
            .subpass_desc(subpass_num)
            .expect("Subpass num out of range ; wrong RenderPassDesc trait impl");

        if subpass
            .color_attachments
            .iter()
            .any(|&(n, _)| n == attachment_num)
        {
            debug_assert!(image.parent().has_color()); // Was normally checked by the render pass.
            if !image.parent().inner().image.usage_color_attachment() {
                return Err(IncompatibleRenderPassAttachmentError::MissingColorAttachmentUsage);
            }
        }

        if let Some((ds, _)) = subpass.depth_stencil {
            if ds == attachment_num {
                // Was normally checked by the render pass.
                debug_assert!(image.parent().has_depth() || image.parent().has_stencil());
                if !image
                    .parent()
                    .inner()
                    .image
                    .usage_depth_stencil_attachment()
                {
                    return Err(
                        IncompatibleRenderPassAttachmentError::MissingDepthStencilAttachmentUsage,
                    );
                }
            }
        }

        if subpass
            .input_attachments
            .iter()
            .any(|&(n, _)| n == attachment_num)
        {
            if !image.parent().inner().image.usage_input_attachment() {
                return Err(IncompatibleRenderPassAttachmentError::MissingInputAttachmentUsage);
            }
        }
    }

    // TODO: consider forbidding LoadOp::Load if image is transient

    // TODO: are all image layouts allowed? check this

    Ok(())
}

/// Error that can happen when an image is not compatible with a render pass attachment slot.
#[derive(Copy, Clone, Debug)]
pub enum IncompatibleRenderPassAttachmentError {
    /// The image format expected by the render pass doesn't match the actual format of
    /// the image.
    FormatMismatch {
        /// Format expected by the render pass.
        expected: Format,
        /// Format of the image.
        obtained: Format,
    },

    /// The number of samples expected by the render pass doesn't match the number of samples of
    /// the image.
    SamplesMismatch {
        /// Number of samples expected by the render pass.
        expected: u32,
        /// Number of samples of the image.
        obtained: u32,
    },

    /// The image view has a component swizzle that is different from identity.
    NotIdentitySwizzled,

    /// The image is used as a color attachment but is missing the color attachment usage.
    MissingColorAttachmentUsage,

    /// The image is used as a depth/stencil attachment but is missing the depth-stencil attachment
    /// usage.
    MissingDepthStencilAttachmentUsage,

    /// The image is used as an input attachment but is missing the input attachment usage.
    MissingInputAttachmentUsage,
}

impl error::Error for IncompatibleRenderPassAttachmentError {}

impl fmt::Display for IncompatibleRenderPassAttachmentError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                IncompatibleRenderPassAttachmentError::FormatMismatch { .. } => {
                    "mismatch between the format expected by the render pass and the actual format"
                }
                IncompatibleRenderPassAttachmentError::SamplesMismatch { .. } => {
                    "mismatch between the number of samples expected by the render pass and the actual \
                 number of samples"
                }
                IncompatibleRenderPassAttachmentError::NotIdentitySwizzled => {
                    "the image view does not use identity swizzling"
                }
                IncompatibleRenderPassAttachmentError::MissingColorAttachmentUsage => {
                    "the image is used as a color attachment but is missing the color attachment usage"
                }
                IncompatibleRenderPassAttachmentError::MissingDepthStencilAttachmentUsage => {
                    "the image is used as a depth/stencil attachment but is missing the depth-stencil \
                 attachment usage"
                }
                IncompatibleRenderPassAttachmentError::MissingInputAttachmentUsage => {
                    "the image is used as an input attachment but is missing the input \
                 attachment usage"
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_image_view_compatible;
    use super::IncompatibleRenderPassAttachmentError;
    use format::Format;
    use framebuffer::EmptySinglePassRenderPassDesc;
    use image::AttachmentImage;

    #[test]
    fn basic_ok() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        let img = AttachmentImage::new(device, [128, 128], Format::R8G8B8A8Unorm).unwrap();

        ensure_image_view_compatible(&rp, 0, &img).unwrap();
    }

    #[test]
    fn format_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R16G16Sfloat,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        let img = AttachmentImage::new(device, [128, 128], Format::R8G8B8A8Unorm).unwrap();

        match ensure_image_view_compatible(&rp, 0, &img) {
            Err(IncompatibleRenderPassAttachmentError::FormatMismatch {
                expected: Format::R16G16Sfloat,
                obtained: Format::R8G8B8A8Unorm,
            }) => (),
            e => panic!("{:?}", e),
        }
    }

    #[test]
    fn attachment_out_of_range() {
        let (device, _) = gfx_dev_and_queue!();

        let rp = EmptySinglePassRenderPassDesc;
        let img = AttachmentImage::new(device, [128, 128], Format::R8G8B8A8Unorm).unwrap();

        assert_should_panic!("Attachment num out of range", {
            let _ = ensure_image_view_compatible(&rp, 0, &img);
        });
    }

    // TODO: more tests
}
