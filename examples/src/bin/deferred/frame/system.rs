// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::Matrix4;
use cgmath::SquareMatrix;
use cgmath::Vector3;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::command_buffer::SubpassContents;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::AttachmentImage;
use vulkano::image::ImageAccess;
use vulkano::image::ImageUsage;
use vulkano::image::ImageViewAccess;
use vulkano::sync::GpuFuture;

use crate::frame::ambient_lighting_system::AmbientLightingSystem;
use crate::frame::directional_lighting_system::DirectionalLightingSystem;
use crate::frame::point_lighting_system::PointLightingSystem;

/// System that contains the necessary facilities for rendering a single frame.
pub struct FrameSystem {
    // Queue to use to render everything.
    gfx_queue: Arc<Queue>,

    // Render pass used for the drawing. See the `new` method for the actual render pass content.
    // We need to keep it in `FrameSystem` because we may want to recreate the intermediate buffers
    // in of a change in the dimensions.
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,

    // Intermediate render target that will contain the albedo of each pixel of the scene.
    diffuse_buffer: Arc<AttachmentImage>,
    // Intermediate render target that will contain the normal vector in world coordinates of each
    // pixel of the scene.
    // The normal vector is the vector perpendicular to the surface of the object at this point.
    normals_buffer: Arc<AttachmentImage>,
    // Intermediate render target that will contain the depth of each pixel of the scene.
    // This is a traditional depth buffer. `0.0` means "near", and `1.0` means "far".
    depth_buffer: Arc<AttachmentImage>,

    // Will allow us to add an ambient lighting to a scene during the second subpass.
    ambient_lighting_system: AmbientLightingSystem,
    // Will allow us to add a directional light to a scene during the second subpass.
    directional_lighting_system: DirectionalLightingSystem,
    // Will allow us to add a spot light source to a scene during the second subpass.
    point_lighting_system: PointLightingSystem,
}

impl FrameSystem {
    /// Initializes the frame system.
    ///
    /// Should be called at initialization, as it can take some time to build.
    ///
    /// - `gfx_queue` is the queue that will be used to perform the main rendering.
    /// - `final_output_format` is the format of the image that will later be passed to the
    ///   `frame()` method. We need to know that in advance. If that format ever changes, we have
    ///   to create a new `FrameSystem`.
    ///
    pub fn new(gfx_queue: Arc<Queue>, final_output_format: Format) -> FrameSystem {
        // Creating the render pass.
        //
        // The render pass has two subpasses. In the first subpass, we draw all the objects of the
        // scene. Note that it is not the `FrameSystem` that is responsible for the drawing,
        // instead it only provides an API that allows the user to do so.
        //
        // The drawing of the objects will write to the `diffuse`, `normals` and `depth`
        // attachments.
        //
        // Then in the second subpass, we read these three attachments as input attachments and
        // draw to `final_color`. Each draw operation performed in this second subpass has its
        // value added to `final_color` and not replaced, thanks to blending.
        //
        // > **Warning**: If the red, green or blue component of the final image goes over `1.0`
        // > then it will be clamped. For example a pixel of `[2.0, 1.0, 1.0]` (which is red) will
        // > be clamped to `[1.0, 1.0, 1.0]` (which is white) instead of being converted to
        // > `[1.0, 0.5, 0.5]` as desired. In a real-life application you want to use an additional
        // > intermediate image with a floating-point format, then perform additional passes to
        // > convert all the colors in the correct range. These techniques are known as HDR and
        // > tone mapping.
        //
        // Input attachments are a special kind of way to read images. You can only read from them
        // from a fragment shader, and you can only read the pixel corresponding to the pixel
        // currently being processed by the fragment shader. If you want to read from attachments
        // but can't deal with these restrictions, then you should create multiple render passes
        // instead.
        let render_pass = Arc::new(
            vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
                attachments: {
                    // The image that will contain the final rendering (in this example the swapchain
                    // image, but it could be another image).
                    final_color: {
                        load: Clear,
                        store: Store,
                        format: final_output_format,
                        samples: 1,
                    },
                    // Will be bound to `self.diffuse_buffer`.
                    diffuse: {
                        load: Clear,
                        store: DontCare,
                        format: Format::A2B10G10R10UnormPack32,
                        samples: 1,
                    },
                    // Will be bound to `self.normals_buffer`.
                    normals: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R16G16B16A16Sfloat,
                        samples: 1,
                    },
                    // Will be bound to `self.depth_buffer`.
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                passes: [
                    // Write to the diffuse, normals and depth attachments.
                    {
                        color: [diffuse, normals],
                        depth_stencil: {depth},
                        input: []
                    },
                    // Apply lighting by reading these three attachments and writing to `final_color`.
                    {
                        color: [final_color],
                        depth_stencil: {},
                        input: [diffuse, normals, depth]
                    }
                ]
            )
            .unwrap(),
        );

        // For now we create three temporary images with a dimension of 1 by 1 pixel.
        // These images will be replaced the first time we call `frame()`.
        // TODO: use shortcut provided in vulkano 0.6
        let atch_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };
        let diffuse_buffer = AttachmentImage::with_usage(
            gfx_queue.device().clone(),
            [1, 1],
            Format::A2B10G10R10UnormPack32,
            atch_usage,
        )
        .unwrap();
        let normals_buffer = AttachmentImage::with_usage(
            gfx_queue.device().clone(),
            [1, 1],
            Format::R16G16B16A16Sfloat,
            atch_usage,
        )
        .unwrap();
        let depth_buffer = AttachmentImage::with_usage(
            gfx_queue.device().clone(),
            [1, 1],
            Format::D16Unorm,
            atch_usage,
        )
        .unwrap();

        // Initialize the three lighting systems.
        // Note that we need to pass to them the subpass where they will be executed.
        let lighting_subpass = Subpass::from(render_pass.clone(), 1).unwrap();
        let ambient_lighting_system =
            AmbientLightingSystem::new(gfx_queue.clone(), lighting_subpass.clone());
        let directional_lighting_system =
            DirectionalLightingSystem::new(gfx_queue.clone(), lighting_subpass.clone());
        let point_lighting_system = PointLightingSystem::new(gfx_queue.clone(), lighting_subpass);

        FrameSystem {
            gfx_queue,
            render_pass: render_pass as Arc<_>,
            diffuse_buffer,
            normals_buffer,
            depth_buffer,
            ambient_lighting_system,
            directional_lighting_system,
            point_lighting_system,
        }
    }

    /// Returns the subpass of the render pass where the rendering should write info to gbuffers.
    ///
    /// Has two outputs: the diffuse color (3 components) and the normals in world coordinates
    /// (3 components). Also has a depth attachment.
    ///
    /// This method is necessary in order to initialize the pipelines that will draw the objects
    /// of the scene.
    #[inline]
    pub fn deferred_subpass(&self) -> Subpass<Arc<dyn RenderPassAbstract + Send + Sync>> {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }

    /// Starts drawing a new frame.
    ///
    /// - `before_future` is the future after which the main rendering should be executed.
    /// - `final_image` is the image we are going to draw to.
    /// - `world_to_framebuffer` is the matrix that will be used to convert from 3D coordinates in
    ///   the world into 2D coordinates on the framebuffer.
    ///
    pub fn frame<F, I>(
        &mut self,
        before_future: F,
        final_image: I,
        world_to_framebuffer: Matrix4<f32>,
    ) -> Frame
    where
        F: GpuFuture + 'static,
        I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static,
    {
        // First of all we recreate `self.diffuse_buffer`, `self.normals_buffer` and
        // `self.depth_buffer` if their dimensions doesn't match the dimensions of the final image.
        let img_dims = ImageAccess::dimensions(&final_image).width_height();
        if ImageAccess::dimensions(&self.diffuse_buffer).width_height() != img_dims {
            // TODO: use shortcut provided in vulkano 0.6
            let atch_usage = ImageUsage {
                transient_attachment: true,
                input_attachment: true,
                ..ImageUsage::none()
            };

            // Note that we create "transient" images here. This means that the content of the
            // image is only defined when within a render pass. In other words you can draw to
            // them in a subpass then read them in another subpass, but as soon as you leave the
            // render pass their content becomes undefined.
            self.diffuse_buffer = AttachmentImage::with_usage(
                self.gfx_queue.device().clone(),
                img_dims,
                Format::A2B10G10R10UnormPack32,
                atch_usage,
            )
            .unwrap();
            self.normals_buffer = AttachmentImage::with_usage(
                self.gfx_queue.device().clone(),
                img_dims,
                Format::R16G16B16A16Sfloat,
                atch_usage,
            )
            .unwrap();
            self.depth_buffer = AttachmentImage::with_usage(
                self.gfx_queue.device().clone(),
                img_dims,
                Format::D16Unorm,
                atch_usage,
            )
            .unwrap();
        }

        // Build the framebuffer. The image must be attached in the same order as they were defined
        // with the `ordered_passes_renderpass!` macro.
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(final_image.clone())
                .unwrap()
                .add(self.diffuse_buffer.clone())
                .unwrap()
                .add(self.normals_buffer.clone())
                .unwrap()
                .add(self.depth_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        // Start the command buffer builder that will be filled throughout the frame handling.
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
        )
        .unwrap();
        command_buffer_builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![
                    [0.0, 0.0, 0.0, 0.0].into(),
                    [0.0, 0.0, 0.0, 0.0].into(),
                    [0.0, 0.0, 0.0, 0.0].into(),
                    1.0f32.into(),
                ],
            )
            .unwrap();

        Frame {
            system: self,
            before_main_cb_future: Some(Box::new(before_future)),
            framebuffer,
            num_pass: 0,
            command_buffer_builder: Some(command_buffer_builder),
            world_to_framebuffer,
        }
    }
}

/// Represents the active process of rendering a frame.
///
/// This struct mutably borrows the `FrameSystem`.
pub struct Frame<'a> {
    // The `FrameSystem`.
    system: &'a mut FrameSystem,

    // The active pass we are in. This keeps track of the step we are in.
    // - If `num_pass` is 0, then we haven't start anything yet.
    // - If `num_pass` is 1, then we have finished drawing all the objects of the scene.
    // - If `num_pass` is 2, then we have finished applying lighting.
    // - Otherwise the frame is finished.
    // In a more complex application you can have dozens of passes, in which case you probably
    // don't want to document them all here.
    num_pass: u8,

    // Future to wait upon before the main rendering.
    before_main_cb_future: Option<Box<dyn GpuFuture>>,
    // Framebuffer that was used when starting the render pass.
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
    // The command buffer builder that will be built during the lifetime of this object.
    command_buffer_builder: Option<AutoCommandBufferBuilder>,
    // Matrix that was passed to `frame()`.
    world_to_framebuffer: Matrix4<f32>,
}

impl<'a> Frame<'a> {
    /// Returns an enumeration containing the next pass of the rendering.
    pub fn next_pass<'f>(&'f mut self) -> Option<Pass<'f, 'a>> {
        // This function reads `num_pass` increments its value, and returns a struct corresponding
        // to that pass that the user will be able to manipulate in order to customize the pass.
        match {
            let current_pass = self.num_pass;
            self.num_pass += 1;
            current_pass
        } {
            0 => {
                // If we are in the pass 0 then we haven't start anything yet.
                // We already called `begin_render_pass` (in the `frame()` method), and that's the
                // state we are in.
                // We return an object that will allow the user to draw objects on the scene.
                Some(Pass::Deferred(DrawPass { frame: self }))
            }

            1 => {
                // If we are in pass 1 then we have finished drawing the objects on the scene.
                // Going to the next subpass.
                self.command_buffer_builder
                    .as_mut()
                    .unwrap()
                    .next_subpass(SubpassContents::SecondaryCommandBuffers)
                    .unwrap();

                // And returning an object that will allow the user to apply lighting to the scene.
                Some(Pass::Lighting(LightingPass { frame: self }))
            }

            2 => {
                // If we are in pass 2 then we have finished applying lighting.
                // We take the builder, call `end_render_pass()`, and then `build()` it to obtain
                // an actual command buffer.
                self.command_buffer_builder
                    .as_mut()
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = self.command_buffer_builder.take().unwrap().build().unwrap();

                // Extract `before_main_cb_future` and append the command buffer execution to it.
                let after_main_cb = self
                    .before_main_cb_future
                    .take()
                    .unwrap()
                    .then_execute(self.system.gfx_queue.clone(), command_buffer)
                    .unwrap();
                // We obtain `after_main_cb`, which we give to the user.
                Some(Pass::Finished(Box::new(after_main_cb)))
            }

            // If the pass is over 2 then the frame is in the finished state and can't do anything
            // more.
            _ => None,
        }
    }
}

/// Struct provided to the user that allows them to customize or handle the pass.
pub enum Pass<'f, 's: 'f> {
    /// We are in the pass where we draw objects on the scene. The `DrawPass` allows the user to
    /// draw the objects.
    Deferred(DrawPass<'f, 's>),

    /// We are in the pass where we add lighting to the scene. The `LightingPass` allows the user
    /// to add light sources.
    Lighting(LightingPass<'f, 's>),

    /// The frame has been fully prepared, and here is the future that will perform the drawing
    /// on the image.
    Finished(Box<dyn GpuFuture>),
}

/// Allows the user to draw objects on the scene.
pub struct DrawPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> DrawPass<'f, 's> {
    /// Appends a command that executes a secondary command buffer that performs drawing.
    #[inline]
    pub fn execute<C>(&mut self, command_buffer: C)
    where
        C: CommandBuffer + Send + Sync + 'static,
    {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        // TODO: ^
        unsafe {
            self.frame
                .command_buffer_builder
                .as_mut()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap();
        }
    }

    /// Returns the dimensions in pixels of the viewport.
    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        let dims = self.frame.framebuffer.dimensions();
        [dims[0], dims[1]]
    }

    /// Returns the 4x4 matrix that turns world coordinates into 2D coordinates on the framebuffer.
    #[allow(dead_code)]
    #[inline]
    pub fn world_to_framebuffer_matrix(&self) -> Matrix4<f32> {
        self.frame.world_to_framebuffer
    }
}

/// Allows the user to apply lighting on the scene.
pub struct LightingPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> LightingPass<'f, 's> {
    /// Applies an ambient lighting to the scene.
    ///
    /// All the objects will be colored with an intensity of `color`.
    pub fn ambient_light(&mut self, color: [f32; 3]) {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        // TODO: ^
        unsafe {
            let dims = self.frame.framebuffer.dimensions();
            let command_buffer = self.frame.system.ambient_lighting_system.draw(
                [dims[0], dims[1]],
                self.frame.system.diffuse_buffer.clone(),
                color,
            );
            self.frame
                .command_buffer_builder
                .as_mut()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap();
        }
    }

    /// Applies an directional lighting to the scene.
    ///
    /// All the objects will be colored with an intensity varying between `[0, 0, 0]` and `color`,
    /// depending on the dot product of their normal and `direction`.
    pub fn directional_light(&mut self, direction: Vector3<f32>, color: [f32; 3]) {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        // TODO: ^
        unsafe {
            let dims = self.frame.framebuffer.dimensions();
            let command_buffer = self.frame.system.directional_lighting_system.draw(
                [dims[0], dims[1]],
                self.frame.system.diffuse_buffer.clone(),
                self.frame.system.normals_buffer.clone(),
                direction,
                color,
            );
            self.frame
                .command_buffer_builder
                .as_mut()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap();
        }
    }

    /// Applies a spot lighting to the scene.
    ///
    /// All the objects will be colored with an intensity varying between `[0, 0, 0]` and `color`,
    /// depending on their distance with `position`. Objects that aren't facing `position` won't
    /// receive any light.
    pub fn point_light(&mut self, position: Vector3<f32>, color: [f32; 3]) {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        // TODO: ^
        unsafe {
            let dims = self.frame.framebuffer.dimensions();
            let command_buffer = {
                self.frame.system.point_lighting_system.draw(
                    [dims[0], dims[1]],
                    self.frame.system.diffuse_buffer.clone(),
                    self.frame.system.normals_buffer.clone(),
                    self.frame.system.depth_buffer.clone(),
                    self.frame.world_to_framebuffer.invert().unwrap(),
                    position,
                    color,
                )
            };

            self.frame
                .command_buffer_builder
                .as_mut()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap();
        }
    }
}
