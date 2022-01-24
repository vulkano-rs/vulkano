use std::sync::Arc;

use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageUsage};

use crate::VulkanContext;

/// The storage for all the screen sized buffers needed for rendering,
/// To simplify the code, we'll discard and re-create the upper abstractions (`ImageView`s)
/// along with their underlying buffers (`AttachmentImage`s) upon surface resize,
/// but you could actually allocate long-lived "large enough" `AttachmentImage`s
/// and just re-create the `ImageView`s to reduce the overhead,
/// if saving video memory is a lower priority than interactive resizing.
pub struct GBuffers {
    /// The buffer for world-space coordinates,
    /// in a Z-up Y-back right-handed coordinate system (to match Blender).
    pub position_buffer: Arc<ImageView<AttachmentImage>>,
    /// The buffer for world-space surface normals.
    pub normal_buffer: Arc<ImageView<AttachmentImage>>,
    /// The buffer for "base color" defined in materials.
    /// Sometimes also called "diffuse" or "albedo", but "base color" is the most appropriate name.
    pub base_color_buffer: Arc<ImageView<AttachmentImage>>,
    /// The buffer for object ID.
    /// Included to demonstrate how to pass integers around.
    pub id_buffer: Arc<ImageView<AttachmentImage>>,
    /// The final HDR image presented to screen, before post-processing.
    /// Maybe a better name is "radiance buffer?" I haven't really learned physics...
    pub composite_buffer: Arc<ImageView<AttachmentImage>>,

    /// The buffer used for depth testing.
    pub depth_buffer: Arc<ImageView<AttachmentImage>>,
}

impl GBuffers {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        dimensions: [u32; 2],
    ) -> anyhow::Result<Arc<Self>> {
        // "Image usage" declaration for all the buffers other than depth buffer.
        // Needed for creating a new attachment image.
        let standard_usage = ImageUsage {
            color_attachment: true, // Required by being drawn onto.
            input_attachment: true, // Required by being read in a multi-subpass render pass.
            sampled: true,          // Required by being read in another render pass.
            // Note: you can also use `storage: true` for the same purpose,
            // but you would lose the free sRGB decoding when transferring colors with 8bpc formats
            // (and you shouldn't use linear 8bpc RGB for colors, guess why isn't sRGB linear?)
            // which isn't as simple as some websites claims it to be (no, gamma 2.2 != sRGB).
            // You'll also lose free clamping for pixels on screen edges.
            // Last of all, you would have to manually specify the image formats in the shaders,
            // which can be a headache for prototyping and refactoring.
            ..ImageUsage::none()
        };
        // Usage for depth buffer.
        let depth_usage = ImageUsage {
            depth_stencil_attachment: true, // Required by being the depth test buffer.
            transient_attachment: true,     // Isn't required by anything, but it tells the GPU
            // we won't need this buffer outside of this particular render pass,
            // so it may improves performance.
            ..ImageUsage::none()
        };

        // Boilerplate code
        // You can write a macro if this list ever grows too long...
        let position_buffer = ImageView::new(
            // An "attachment image" is meant to be used in the graphics pipeline.
            AttachmentImage::with_usage(
                vk.device().clone(),
                dimensions.clone(),
                Format::R32G32B32A32_SFLOAT, // 32-bit float is necessary for positions.
                standard_usage,
            )?,
        )?;
        let normal_buffer = ImageView::new(AttachmentImage::with_usage(
            vk.device().clone(),
            dimensions.clone(),
            Format::R32G32B32A32_SFLOAT, /* Although there are ways to stuff normals into two components... */
            standard_usage,
        )?)?;
        let base_color_buffer = ImageView::new(AttachmentImage::with_usage(
            vk.device().clone(),
            dimensions.clone(),
            Format::R8G8B8A8_SRGB, // Since we're just copying colors from 8bpc sRGB PNGs.
            standard_usage,
        )?)?;
        let id_buffer = ImageView::new(AttachmentImage::with_usage(
            vk.device().clone(),
            dimensions.clone(),
            Format::R32_UINT, // Actually not overkill if you want to render e.g. a detailed forest.
            standard_usage,
        )?)?;
        let composite_buffer = ImageView::new(AttachmentImage::with_usage(
            vk.device().clone(),
            dimensions.clone(),
            Format::R16G16B16A16_SFLOAT, // To prevent banding after tone mapping.
            standard_usage,
        )?)?;
        let depth_buffer = ImageView::new(AttachmentImage::with_usage(
            vk.device().clone(),
            dimensions.clone(),
            Format::D32_SFLOAT, // I hate Z-fighting. Note: D24 is presumably faster on Nvidia cards.
            depth_usage, // <-- Don't forget about this line's difference after all the ctrl-v's!
        )?)?;

        Ok(Arc::new(Self {
            position_buffer,
            normal_buffer,
            base_color_buffer,
            id_buffer,
            composite_buffer,
            depth_buffer,
        }))
    }
}

/// The values used to clear the buffers on each redraw.
pub const CLEAR_VALUES: [ClearValue; 6] = [
    ClearValue::Float([0., 0., 0., 0.]), // Position
    ClearValue::Float([0., 0., 0., 0.]), // Normal
    ClearValue::Float([0., 0., 0., 0.]), // Base color
    ClearValue::Uint([0, 0, 0, 0]), /* ID, note how single-component buffers will still have 4-component clears */
    ClearValue::Float([0., 0., 0., 0.]), // Composite color
    // IMPORTANT: note how the depth is cleared with 1, the farthest possible depth by default.
    // This is ridiculously hard to debug if you have accidentally set it to 0,
    // since RenderDoc won't tell you which fragments are discarded by which tests.
    ClearValue::Depth(1.), // Depth
];
