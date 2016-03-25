# Differences between Vulkan and vulkano

This page lists the features that vulkano brings over Vulkan, and the design decisions of vulkano
that you have to be aware of compared to Vulkan.

## Arcs usage

If you use vulkano, you quickly notice that all `Foo::new` functions return an `Arc<Foo>` instead
of a `Foo`.

Take this code for example:

```rust
pub struct SpriteDrawSystem<R> {
    cb: Arc<SecondaryGraphicsCommandBuffer<R>>,
}

impl<R> SpriteDrawSystem<R> {
    pub fn new(queue: &Arc<Queue>, subpass: Subpass<R>) -> SpriteDrawSystem<R> {
        .. build the descriptor set layouts ..
        .. build the descriptor pool ..
        .. build the pipeline layout ..
        .. build the descriptor sets ..
        .. build the shader modules ..
        .. build the pipeline ..
        .. build the command buffer ..

        SpriteDrawSystem { cb: .. }
    }

    #[inline]
    pub fn draw(&self, out: PrimaryCommandBufferBuilderSecondaryDraw<R>)
                -> PrimaryCommandBufferBuilderSecondaryDraw<R>
    {
        out.execute_commands(&self.cb)
    }
}
```

## Buffer strong typing

In vulkano, buffers are strongly-typed. For example, creating an index buffer is done like this:

```rust
use vulkano::buffer::Buffer;
use vulkano::buffer::Usage as BufferUsage;
use vulkano::memory::DeviceLocal;

let index_buffer = Buffer::<[u16], _>::array(&device, 128 /* number of elements */,
                                             &BufferUsage::all(), DeviceLocal, &queue).unwrap();
```

Notice that we explicitely tell vulkano that the content of the buffer is of type `[u16]`.

## Image strong typing

Just like buffers, images also use strong typing.

## Choosing between compile-time checks and runtime checks

Whenever you build a render pass, you have to pass an object that describes the layout of the
render pass.

## Keeping command buffers alive

In Vulkan, care must be taken to not destroy command buffers that are still in use by the GPU.
In vulkano, submitting a command buffer to a queue returns a `Submission` object that holds
an `Arc` to the command buffer that has been submitted. Destroying this object will block
until it is known that the command buffer is no longer in use.

The best way to deal with `Submission` objects is to store them in a `Vec`. You can either return
the `Submission` objects from functions that produce them, and then store them all on the stack
around your main loop. Or you can store the objects within another long-lived object, for example
in your `Scene` or your `SpritesDrawingSystem`.

If you keep pushing submission objects in your `Vec` you will ultimately run out of memory.
Cleaning the objects that are no longer needed can be done like this:

```rust
submissions.retain(|s| !s.destroying_would_block());
```
