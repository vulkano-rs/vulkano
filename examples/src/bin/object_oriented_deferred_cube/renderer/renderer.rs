use std::sync::Arc;
use std::time::Duration;

use anyhow::bail;

use vulkano::format::Format;
use vulkano::image::ImageUsage;

use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{ColorSpace, CompositeAlpha, PresentMode, Swapchain};
use vulkano::sync::GpuFuture;

use crate::renderer::asset_library::AssetLibrary;
use crate::renderer::gbuffers::GBuffers;
use crate::renderer::passes::main_pass::main_pass::MainPass;
use crate::renderer::passes::post_pass::post_pass::PostPass;
use crate::scene::scene::Scene;
use crate::VulkanContext;

/// The object wrapping everything needed to render a frame,
/// bound to a particular combination `VulkanContext`, `AssetLibrary` and window size,
/// so it needs to be re-created whenever any of those changes.
/// For what the hell the undocumented children are, try ctrl+clicking their types in an IDE.
pub struct Renderer<TWindow: Send + Sync + 'static> {
    vk: VulkanContext<TWindow>,
    library: Arc<AssetLibrary>,

    /// A swapchain is the candidate buffers for actual frames painted to the screen.
    swapchain: Arc<Swapchain<TWindow>>,
    // A viewport specifies in which area of the output buffers should be drawn on.
    // Since we're making the G-buffers' sizes identical to the window,
    // we'll reuse it everywhere in the entire process.
    viewport: Viewport,

    gbuffers: Arc<GBuffers>,
    main_pass: MainPass,
    post_pass: PostPass,
}

impl<TWindow: Send + Sync + 'static> Renderer<TWindow> {
    pub fn new(
        vk: VulkanContext<TWindow>,
        library: Arc<AssetLibrary>,
        size: [u32; 2],
        present_mode: PresentMode,
        format: Format,
        color_space: ColorSpace,
    ) -> anyhow::Result<Self> {
        // Create the swapchain.
        let (swapchain, images) = {
            let capabilities = vk.surface().capabilities(vk.device().physical_device())?;
            match Swapchain::start(vk.device(), vk.surface())
                .num_images(
                    // We'll try to use a number of two, or else grab whatever minimum the device supports.
                    // AFAIK you shouldn't want to change this value either;
                    // To prevent tearing while having maximum framerate,
                    // just set the present mode to something else instead of using triple buffering.
                    2.clamp(
                        capabilities.min_image_count,
                        capabilities.max_image_count.unwrap_or(u32::MAX),
                    ),
                )
                .format(format)
                .color_space(color_space)
                .dimensions(size.into())
                .present_mode(present_mode)
                // So we can draw onto the swapchain (yes, that must be explicit too)
                .usage(ImageUsage::color_attachment())
                // Limiting the images' access to a single queue family.
                // This function can accept a single queue and smartly obtain its family.
                .sharing_mode(&vk.main_queue())
                // Composite alpha is useful if you're writing a desktop widget,
                // or a fancy splash screen like the Adobe products have.
                // Otherwise you should just set this to Opaque.
                .composite_alpha(CompositeAlpha::Opaque)
                .build()
            {
                Ok(x) => x,
                Err(e) => bail!("{}", e),
            }
        };

        // Create a viewport that covers the whole drawable area of the window.
        let viewport = Viewport {
            origin: [0., 0.],
            dimensions: [size[0] as f32, size[1] as f32],
            depth_range: 0f32..1f32,
        };

        let gbuffers = GBuffers::new(vk.clone(), size.clone())?;
        let main_pass = MainPass::new(vk.clone(), gbuffers.clone(), library.clone())?;
        let post_pass = PostPass::new(vk.clone(), gbuffers.clone(), images, swapchain.format())?;

        Ok(Self {
            vk,
            library,
            swapchain,
            viewport,
            gbuffers,
            main_pass,
            post_pass,
        })
    }

    /// Submits the current render.
    /// This method blocks the calling thread until the frame is finished.
    pub fn render(&self, scene: &Scene) -> anyhow::Result<()> {
        // Acquire the index of the next available image in the swapchain.
        let (swapchain_index, suboptimal, swapchain_acquire_future) =
            match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(x) => x,
                Err(e) => bail!("Failed to acquire next swapchain image: {}", e),
            };
        // If `suboptimal` is true, it means the swapchain can still be used,
        // but you should recreate it ASAP if possible, because some graphics settings have changed.
        if suboptimal {
            // We'll simply fail here.
            // TODO This may cause a period of (if not permanent) graphics freeze in real world;
            // Please don't do this.
            bail!("Suboptimal swapchain");
        }

        let main_pass = self.main_pass.build_command_buffer(
            self.vk.main_queue(),
            self.viewport.clone(),
            &scene,
        )?;
        let post_pass = self.post_pass.build_command_buffer(
            self.vk.main_queue(),
            self.viewport.clone(),
            swapchain_index,
        )?;
        swapchain_acquire_future
            .then_execute(self.vk.main_queue(), main_pass)?
            .then_execute(self.vk.main_queue(), post_pass)?
            .then_swapchain_present(
                self.vk.main_queue(),
                self.swapchain.clone(),
                swapchain_index,
            )
            .then_signal_fence_and_flush()?
            .wait(Some(Duration::from_secs(10)))?; // TODO change or remove the hardcoded timeout
        Ok(())
    }
}
