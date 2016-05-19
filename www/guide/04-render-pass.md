---
layout: page
title: "Tutorial 4: render passes"
---

# Render passes

In the previous section, we created a window and asked the GPU to fill its surface with a color.
However our ultimate goal is to draw some shapes on that surface, not just clear it.

In order to fully optimize and parallelize commands execution, we can't just add ask the GPU
to draw a shape whenever we want. Instead we first have to enter "rendering mode" by entering
a *render pass*, then draw, and then leave the render pass.

This will serve as a foundation for the next tutorial, which is about drawing a triangle.

## What is a render pass?

The term "render pass" describes two things:

- It designates the "rendering mode" we have to enter before we can add drawing commands to
  a command buffer.

- It also designates a kind of object that describes this rendering mode.

In this section, we are going to create a render pass object, and then modify our command buffer
to enter the render pass.

## Creating a render pass

In this tutorial, the only thing we want to do is draw to a window. This is the most simple case
possible, and we only need to provide two informations to a render pass: the format of the images
of our swapchain, and the fact that we don't use multisampling (which is an advanced anti-aliasing
technique).

However complex games can use render passes in very complex ways, with multiple subpasses and
multiple attachments, and with various micro-optimizations. In order to accomodate for these
complex usages, vulkano's API to create a render pass is a bit complex.

{% highlight rust %}
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
{% endhighlight %}

A render pass only describes the format and the way we load and store the image we are going to
draw upon. However we also need to indicate the actual list of attachments.

{% highlight rust %}
let framebuffers = images.iter().map(|image| {
    let dimensions = [image.dimensions()[0], image.dimensions()[1], 1];
    Framebuffer::new(&render_pass, dimensions, render_pass::AList {
        color: image
    }).unwrap()
}).collect::<Vec<_>>();
{% endhighlight %}

{% highlight rust %}
let command_buffer = PrimaryCommandBufferBuilder::new(&cb_pool)
    .draw_inline(&render_pass, &framebuffers[image_num], render_pass::ClearValues {
        color: [0.0, 0.0, 1.0, 1.0]
    })
    .draw_end()
    .build();
{% endhighlight %}
