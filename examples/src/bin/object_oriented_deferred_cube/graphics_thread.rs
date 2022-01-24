use std::sync::mpsc::Receiver;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;

use anyhow::anyhow;
use log::{error, info};

use vulkano::format::Format;
use vulkano::swapchain::{ColorSpace, PresentMode};

use crate::renderer::renderer::Renderer;
use crate::scene::scene::Scene;
use crate::{AssetLibrary, VulkanContext};

/// Create dedicated thread for graphics submission,
/// because blocking the main thread (which is used for collecting user input)
/// for the next GPU frame is usually not a good idea,
/// especially in video games and creative applications.
///
/// Caution: do not use std::thread::sleep for limiting framerate.
/// Most OSs (for example Windows) aren't real-time,
/// meaning you might sleep a whole lot more than you'd expect.
/// Just use FIFO, or do a spin-wait with std::time::Instant in other present modes.
pub fn graphics_thread<TWindow: Send + Sync + 'static>(
    vk: VulkanContext<TWindow>,
    library: Arc<AssetLibrary>,
    size: [u32; 2],
    exit_event: Receiver<()>,
    resize_event: Receiver<[u32; 2]>,
    scene: Arc<RwLock<Scene>>,
) -> anyhow::Result<JoinHandle<()>> {
    // The present mode is related to what the average users call "vertical syncing".
    // Fifo is vertical syncing, and is guaranteed to be available on valid Vulkan drivers;
    // Mailbox is the fastest you can get without tearing.
    // The other two are usually undesirable and only acts as a compromise
    // when Mailbox isn't available and the user wants less latency.
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPresentModeKHR.html
    // https://vulkan.gpuinfo.org/listsurfacepresentmodes.php
    // (Vulkano doesn't support VK_KHR_shared_presentable_image ATM.)
    // We'll just hard code Fifo here.
    let present_mode = PresentMode::Fifo;

    // The format and color space pair is... hard to explain,
    // but most of the time you will want some sort of nonlinear sRGB as of Jan 2022
    // (or extended linear sRGB for HDR output on Windows)
    // and a format that matches the CURRENT MONITOR the application is on
    // (although B8G8R8A8 seems to be always working for me
    // even if I'm using a 10-bit monitor on Windows 10 + Nvidia with HDR output enabled).
    //
    // Another pitfall is the driver might have support format/color space pairs it didn't list:
    // R8G8B8A8_SRGB works for me even if it's not reported by the driver.
    //
    // Q: What's the difference between _UNORM and _SRGB when they are all paired with SRGB_NONLINEAR_KHR?
    // A: _UNORM is only useful if you are copying sRGB colors from a sRGB buffer
    //    directly to the swapchain with direct image access (GLSL imageLoad/Store),
    //    or somehow generating already sRGB-encoded colors inside the shader.
    //    You should NEVER do the sRGB encoding in shaders yourself
    //    if you don't understand what are you doing;
    //    gamma 2.2 != sRGB, and letting the implementation do the encoding is supposedly faster.
    //
    // List of (driver-reported) platform coverage: https://vulkan.gpuinfo.org/listsurfaceformats.php
    //
    // Here we just find the first 8bpc sRGB pair the surface claims to support.
    // You might want to try creating the swapchain first,
    // then fallback to choosing something from this list.
    let (format, color_space) = vk
        .surface()
        .capabilities(vk.device().physical_device())?
        .supported_formats
        .iter()
        .filter(|(format, color_space)| {
            (*format == Format::B8G8R8A8_SRGB || *format == Format::R8G8B8A8_SRGB)
                && *color_space == ColorSpace::SrgbNonLinear
        })
        .next()
        .ok_or_else(|| anyhow!("Surface does not support 8bpc sRGB"))?
        .clone();

    let renderer = Renderer::new(
        vk.clone(),
        library.clone(),
        size,
        present_mode,
        format,
        color_space,
    )?;

    Ok(std::thread::spawn(move || {
        // Current window size.
        let mut size = size;

        // We'll clear this value in case of error and try to re-create it.
        // This saves code and allows the user to easily quit
        // when the renderer can never successfully be re-created.
        let mut renderer = Some(renderer);
        while exit_event.try_recv().is_err() {
            // If the window size changed, the renderer needs to be re-created.
            if let Some(new_size) = resize_event.try_iter().last() {
                size = new_size;
                renderer = None;
            }

            if renderer.is_none() {
                info!("Re-creating renderer with size {:?}", size);
                match Renderer::new(
                    vk.clone(),
                    library.clone(),
                    size,
                    present_mode,
                    format,
                    color_space,
                ) {
                    Ok(x) => {
                        renderer = Some(x);
                        info!("Re-created renderer");
                    }
                    Err(e) => {
                        error!("Re-creation of renderer failed: {}", e);
                        continue;
                    }
                }
            }
            let scene = match scene.read() {
                Ok(x) => x,
                Err(e) => {
                    error!("Scene lock poisoned: {:?}", e);
                    break;
                }
            };
            // Blocking happens on the .render() call!
            if let Err(e) = renderer.as_ref().unwrap().render(&scene) {
                error!("Render failed: {:?}", e);
                renderer = None;
            }
        }
    }))
}
