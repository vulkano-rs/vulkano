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
let source = CpuAccessibleBuffer::new(&device, 3, &BufferUsage::all(), Some(queue.family()))
                                    .expect("failed to create buffer");
let destination = CpuAccessibleBuffer::new(&device, 3, &BufferUsage::all(), Some(queue.family()))
                                    .expect("failed to create buffer");
{% endhighlight %}

## Copying

In Vulkan you can't just submit a command to a queue. Instead you must create a *command buffer*
which contains one or more commands, and submit the command buffer.

{% highlight rust %}
let cb_pool = CommandPool::new(&device);
{% endhighlight %}

{% highlight rust %}
let cmd = PrimaryCommandBuffer::new().copy(&source, &destination).build();
{% endhighlight %}

We now have our command buffer! The last thing we need to do is submit it to a
queue for execution.

{% highlight rust %}

{% endhighlight %}

Note: there are several things that we can do in a more optimal way.

This code asks the GPU to execute the command buffer that contains our copy command and waits for
it to be finished.
