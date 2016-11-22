---
layout: page
title: "Tutorial 2: the first operation"
---

# The first operation

Now that we have chosen a physical device, it is time to ask it to do something.

But before we start, a few things need to be explained. Just like it is possible to use multiple
threads in your program running on the CPU, it is also possible to run multiple operations in
parallel on the physical device. The Vulkan equivalent of a CPU core is a *queue*.

When we ask the device to perform an operation, we have to submit the command to a specific queue.
Some queues support only graphical operations, some others support only compute operations, and
some others support both.

The reason why this is important is that at initialization we need to tell the device which queues
we are going to use.

## Creating a device

A `Device` object is an open channel of communication with a physical device. It is probably the
most important object of the Vulkan API.

{% highlight rust %}
let (device, mut queues) = {
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        .. DeviceExtensions::none()
    };

    Device::new(&physical, physical.supported_features(), &device_ext, None,
                [(queue, 0.5)].iter().cloned()).expect("failed to create device")
};
{% endhighlight %}

We now have an open channel of communication with a Vulkan device!

In the rest of this article, we are going to ask the device to copy data from a buffer to
another. Copying data is an operation that you do very often in Vulkan, so let's get used
to it early.

## Creating buffers

To do so, let's create two buffers first: one source and one destination. There are multiple
ways to create a buffer in vulkano, but for now we're going to use a `CpuAccessibleBuffer`.

{% highlight rust %}
let source = CpuAccessibleBuffer::array(&device, 3, &BufferUsage::all(), Some(queue.family()))
                                    .expect("failed to create buffer");
let destination = CpuAccessibleBuffer::array(&device, 3, &BufferUsage::all(), Some(queue.family()))
                                    .expect("failed to create buffer");
{% endhighlight %}

Creating a buffer in Vulkan requires passing several informations.

The first parameter is the device to use. Most objects in Vulkan and in vulkano are linked to a
specific device, and only objects that belong to the same device can interact with each other.
Most of the time you will only have one `Device` object alive, so it's not a problem.

The second parameter is present only because we use `CpuAccessibleBuffer::array` and corresponds
to the capacity of the array in number of elements.

The third parameter tells the Vulkan implementation in which ways the buffer is going to be used.
Thanks to this, the implementation may be capable of performing some optimizations. Here we just
pass a dummy value, but in a real code you should indicate.

The final parameter is the queue family which are going to perform operations on the buffer.

## Copying

In Vulkan you can't just submit a command to a queue. Instead you must create a *command buffer*
which contains one or more commands, and then submit the command buffer.

That sounds complicated, but it is not:

{% highlight rust %}
let cmd = PrimaryCommandBuffer::new().copy(&source, &destination).build();
{% endhighlight %}

We now have our command buffer! It is ready to be executed. The last thing we need to do is
submit it to a queue for execution.

{% highlight rust %}

{% endhighlight %}

Note: there are several things that we can do in a more optimal way.

This code asks the GPU to execute the command buffer that contains our copy command and waits for
it to be finished.
