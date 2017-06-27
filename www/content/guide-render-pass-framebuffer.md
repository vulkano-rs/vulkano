# Render passes

In the previous section, we created a window and asked the GPU to fill its surface with a color.
However our ultimate goal is to draw some shapes on that surface, not just clear it.

In order to fully optimize and parallelize commands execution, we can't just add ask the GPU
to draw a shape whenever we want. Instead we first have to enter "rendering mode" by entering
what is called a *render pass*, then draw, and then leave the render pass.

In this section we are just going to enter a render pass and leave it immediately. This is not
very useful per se, but it will serve as a foundation for the next tutorial, which is about
drawing a triangle.

## What is a render pass?

The term "render pass" describes two things:

- It designates the "rendering mode" we have to enter before we can add drawing commands to
  a command buffer.

- It also designates a kind of object that describes this rendering mode.

Entering a render pass (as in "the rendering mode") requires passing a render pass object.

## Creating a render pass

For the moment, the only thing we want to do is draw some color to an image that corresponds to
our window. This is the most simple case possible, and we only need to provide two informations
to a render pass: the format of the images of our swapchain, and the fact that we don't use
multisampling (which is an advanced anti-aliasing technique).

However complex games can use render passes in very complex ways, with multiple subpasses and
multiple attachments, and with various micro-optimizations. In order to accomodate for these
complex usages, vulkano's API to create a render pass is a bit particular.

TODO: provide a simpler way in vulkano to do that?

    mod render_pass {
        use vulkano::format::Format;
         
        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        }
    }
     
    let render_pass = render_pass::CustomRenderPass::new(&device, &render_pass::Formats {
        color: (images[0].format(), 1)
    }).unwrap();

## Entering the render pass

A render pass only describes the format and the way we load and store the image we are going to
draw upon. It is enough to initialize all the objects we need.

But before we can draw, we also need to indicate the actual list of attachments. This is done
by creating a *framebuffer*.

Creating a framebuffer is typically done as part of the rendering process. Although it is not a
bad idea to keep the framebuffer objects alive between frames, but it won't kill your
performances to create and destroy a few framebuffer objects during each frame.

    let framebuffer = {
        let image = &images[image_num];
        let dimensions = [image.dimensions()[0], image.dimensions()[1], 1];
        Framebuffer::new(&render_pass, dimensions, render_pass::AList {
            color: image
        }).unwrap()
    };

We are now ready the enter drawing mode!

This is done by calling the `draw_inline` function on the primary command buffer builder.
This function takes as parameter the render pass object, the framebuffer, and a struct that
contains the colors to fill the attachments with.

This struct is created by the `single_pass_renderpass!` macro and contains one field for
each attachment that was defined with `load: Clear`.

Clearing our attachment has exactly the same effect as `clear_color_foo`, except that this
time it is done by the rendering engine.

    let command_buffer = PrimaryCommandBufferBuilder::new(&cb_pool)
        .draw_inline(&render_pass, &framebuffer, render_pass::ClearValues {
            color: [0.0, 0.0, 1.0, 1.0]
        })
        .draw_end()
        .build();

We enter the render pass and immediately leave it afterward. In the next section, we are going
to insert a function call between `draw_inline` and `draw_end`.
