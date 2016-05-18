---
layout: page
title: "Tutorial 5: the first triangle"
---

# The first triangle

With some exceptions, Vulkan doesn't provide any function to easily draw shapes. There is
no draw_rectangle, draw_cube or draw_text function for example. Instead everything is handled
the same way: through the graphics pipeline. It doesn't matter whether you draw a simple
triangle or a 3D model with thousands of polygons and advanced shadowing techniques, everything
uses the same mechanics.

This is the point where the learning curve becomes very steep, as you need to learn how the
graphics pipeline works even if you just want to draw a single triangle. However once you have
passed that step, it will become easier to understand the rest.

Before we can draw a triangle, we need to prepare two things during the initialization:

- A shape that describes our triangle.
- A graphics pipeline object that will be executed by the GPU.

## Shape

A shape represents the geometry of an object. When you think "geometry", you may think of squares,
circles, etc., but in graphics programming the only shapes that we are going to manipulate are
triangles (note: tessellation unlocks the possibility to use other polygons, but this is an
advanced topic).

Here is an example of an object's shape. As you can see, it is made of hundreds of triangles and
only triangles.

TODO: The famous Utah Teapot

Each triangle is made of three vertices, which means that a shape is just a collection of vertices
linked together to form triangles. The first step to describe a shape like this with vulkano is to
create a struct named `Vertex` (the actual name doesn't matter) whose purpose is to describe each
individual vertex. Our collection of vertices can later be represented by a collection of `Vertex`
objects.

{% highlight rust %}
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl_vertex!(Vertex, position);
{% endhighlight %}

Our struct contains a position field which we will use to store the position of each vertex on
the window. Being a true vectorial renderer, Vulkan doesn't use coordinates in pixels. Instead it
considers that the window has a width and a height of 2 units, and that the origin is at the
center of the window.

TODO: The windows coordinates system

When we give positions to Vulkan, we need to use this coordinate system. Let's pick a shape for
our triangle, for example this one:

TODO: Finding the coordinates of our triangle

Which translates into this code:

{% highlight rust %}
let vertex1 = Vertex { position: [-0.5,  0.5] };
let vertex2 = Vertex { position: [ 0.0, -0.5] };
let vertex3 = Vertex { position: [ 0.5,  0.25] };
{% endhighlight %}

Since we are going to pass this data to the video card, we have to put it in a buffer. This is
done in the same way as we did earlier.

{% highlight rust %}
{% endhighlight %}

## The graphics pipeline

### Shaders

In the 1990s, drawing an object with a video card consisted in sending a shape alongside with
various parameters like the color, lightning direction, fog distance, etc. Over time these
parameters became too limiting for game creators, and in the year 2000s a more flexible system
was introduced with what are called shaders. A few years later, all these parameters were removed
and totally replaced with shaders.

In order to draw a triangle, you will need some basic understanding about how the drawing process
(also called the pipeline) works.

TODO: The graphics pipeline

The list of coordinates at the left of the schema represents the vertices of the shape that we
have created earlier. When we will ask the GPU to draw this shape, it will first execute what is
called a vertex shader, once for each vertex (that means three times here). A vertex shader is
a small program whose purpose is to tell the GPU what the screen coordinates of each vertex is.

Then the GPU builds our triangle and determines which pixels of the screen are inside of it. It
will then execute a fragment shader once for each of these pixels. A fragment shader is a small
program whose purpose is to tell the GPU what the color of each pixel needs to be.

The tricky part is that we need to write the vertex and fragment shaders. To do so, we are going
to write it using a programming language named GLSL, which is very similar to the C programming
language. Teaching you GLSL would be a bit too complicated for now, so I will just give you the
source codes. Here is the source code that we will use for the vertex shader:

{% highlight glsl %}
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
{% endhighlight %}

When we defined the `Vertex` struct in our shape, we created a field named position which
contains the position of our vertex. But contrary to what I let you think, this struct doesn't
contain the actual position of the vertex but only a attribute whose value is passed to the vertex
shader. Vulkan doesn't care about the name of the attribute, all it does is passing its value
to the vertex shader. The `in vec2 position;` line of our shader is here to declare that we are
expected to be passed an attribute named position whose type is `vec2` (which corresponds to
`[f32; 2]` in Rust).

The main function of our shader is called once per vertex, which means three times for our
triangle. The first time, the value of position will be `[-0.5, -0.5]`, the second time it will
be `[0, 0.5]`, and the third time `[0.5, -0.25]`. It is in this function that we actually tell
Vulkan what the position of our vertex is, thanks to the `gl_Position = vec4(position, 0.0, 1.0);`
line. We need to do a small conversion because Vulkan doesn't expect two-dimensional coordinates,
but four-dimensional coordinates (the reason for this will be covered in a later tutorial).

The second shader is called the fragment shader (sometimes also named pixel shader in other APIs).

{% highlight glsl %}
#version 450

layout(location = 0) out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
{% endhighlight %}

This source code is very similar to our vertex shader above. This time the `main` function is
executed once per pixel and has to return the color of this pixel, which we do with the
`color = vec4(1.0, 0.0, 0.0, 1.0);` line. Just like when clearing the image, we need to pass the
red, green, blue and alpha components of the pixel. Here we are returning an opaque red color.
In a real application you most likely want to return different values depending on the pixel,
but this will be covered in later tutorials.

### Compiling the shaders

Before we can pass our shaders to Vulkan, we have to compile them in a format named **SPIR-V**.

{% highlight rust %}
mod vs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/triangle_vs.glsl")} }
let vs = vs::Shader::load(&device).expect("failed to create shader module");
mod fs { include!{concat!(env!("OUT_DIR"), "/shaders/src/bin/triangle_fs.glsl")} }
let fs = fs::Shader::load(&device).expect("failed to create shader module");
{% endhighlight %}

### Building the graphics pipeline object

{% highlight rust %}
use vulkano::descriptor::pipeline_layout::EmptyPipeline;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineParams;
use vulkano::pipeline::blend::Blend;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::input_assembly::InputAssembly;
use vulkano::pipeline::multisample::Multisample;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::ViewportsState;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::viewport::Scissor;

let pipeline = GraphicsPipeline::new(&device, GraphicsPipelineParams {
    vertex_input: SingleBufferDefinition::new(),
    vertex_shader: vs.main_entry_point(),
    input_assembly: InputAssembly::triangle_list(),
    geometry_shader: None,

    viewport: ViewportsState::Fixed {
        data: vec![(
            Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0 .. 1.0,
                dimensions: [images[0].dimensions()[0] as f32,
                             images[0].dimensions()[1] as f32],
            },
            Scissor::irrelevant()
        )],
    },

    raster: Default::default(),
    multisample: Multisample::disabled(),
    fragment_shader: fs.main_entry_point(),
    depth_stencil: DepthStencil::disabled(),
    blend: Blend::pass_through(),
    layout: &EmptyPipeline::new(&device).unwrap(),
    render_pass: Subpass::from(&render_pass, 0).unwrap(),
}).unwrap();
{% endhighlight %}

This big struct contains all the parameters required to describe the draw operation to Vulkan.

## Drawing

Now that we have prepared our shape and graphics pipeline object, we can finally draw this
triangle!

Remember the target object? We will need to use it to start a draw operation.

Starting a draw operation needs several things: a source of vertices (here we use our vertex_buffer), a source of indices (we use our indices variable), a program, the program's uniforms, and some draw parameters. We will explain what uniforms and draw parameters are in the next tutorials, but for the moment we will just ignore them by passing a EmptyUniforms marker and by building the default draw parameters.

target.draw(&vertex_buffer, &indices, &program, &glium::uniforms::EmptyUniforms,
            &Default::default()).unwrap();

The "draw command" designation could make you think that drawing is a heavy operation that takes a lot of time. In reality drawing a triangle takes less than a few microseconds, and if everything goes well you should see a nice little triangle:
