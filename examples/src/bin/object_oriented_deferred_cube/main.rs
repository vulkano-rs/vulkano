use std::sync::{Arc, RwLock};

use log::error;
use simple_logger::SimpleLogger;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use scene::scene::Scene;
use vulkano::device::{DeviceExtensions, Features};
use vulkano::instance::InstanceExtensions;
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;

use crate::graphics_thread::graphics_thread;
use crate::renderer::asset_library::AssetLibrary;

use crate::vulkan_context::VulkanContext;

mod asset;
mod graphics_thread;
mod renderer;
pub mod scene;
mod scene_generation;
mod vulkan_context;
mod vulkan_instance;

fn main() {
    // Set up logging.
    SimpleLogger::new().with_colors(true).init().unwrap();

    // Create the Instance.
    // For an explanation on this, check out vulkan_instance.rs.
    let (instance, _debug_callback) = vulkan_instance::new(
        Version::V1_2, // This is the current latest version.
        false,         // TODO Change to true to have more fancy debug info!
        InstanceExtensions::none(),
        vec![],
    )
    .unwrap();

    // Create the Window with winit.
    // If you for some reason don't want to use winit, I also recommend SDL;
    // You can find how to create a Vulkano-compatible window with SDL in their README:
    // https://github.com/Rust-SDL2/rust-sdl2
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone()) // This black magic method is added by the vulkano_win crate.
        .unwrap();

    // Set up the Vulkan context.
    // Read vulkan_context.rs for details.
    let vk = VulkanContext::new(
        instance.clone(),
        surface,
        DeviceExtensions::none(),
        Features::none(),
    )
    .unwrap();

    // Create the asset library.
    let library = AssetLibrary::new(
        vk.clone(),
        (1..=6)
            .into_iter()
            .map(|i| (i.to_string(), scene_generation::cube(i.to_string())))
            .collect(),
        vec![
            ("1".to_string(), include_bytes!("resources/1.png").to_vec()),
            ("2".to_string(), include_bytes!("resources/2.png").to_vec()),
            ("3".to_string(), include_bytes!("resources/3.png").to_vec()),
            ("4".to_string(), include_bytes!("resources/4.png").to_vec()),
            ("5".to_string(), include_bytes!("resources/5.png").to_vec()),
            ("6".to_string(), include_bytes!("resources/6.png").to_vec()),
        ]
        .into_iter()
        .collect(),
        128,
    )
    .unwrap();

    // Generate the scene.
    let scene = Arc::new(RwLock::new(scene_generation::generate_scene()));

    // The channel used to tell the graphics thread to stop.
    // You might want check out crossbeam's multiple receiver channel in a real app.
    let (exit_event_sender, exit_event) = std::sync::mpsc::channel();
    // The channel used to tell the graphics thread the new window size.
    // although winit's Window is Clone so we can get it on the graphics thread,
    // other libraries' window type might not be.
    let (resize_event_sender, resize_event) = std::sync::mpsc::channel();
    // Actual graphics loop is here.
    let mut graphics_thread = Some(
        graphics_thread(
            vk.clone(),
            library.clone(),
            vk.surface().window().inner_size().into(),
            exit_event,
            resize_event,
            scene,
        )
        .unwrap(),
    );

    // The input loop.
    event_loop.run(move |event, _, control_flow| match event {
        // So the user could use the X button to quit.
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
            if exit_event_sender.send(()).is_ok() {
                if let Err(_e) = graphics_thread.take().unwrap().join() {
                    error!("Graphics thread panicked");
                }
            }
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            if let Err(e) = resize_event_sender.send(vk.surface().window().inner_size().into()) {
                error!("Can't send resize event: {}", e);
            }
        }
        _ => (),
    });
}
