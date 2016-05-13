---
layout: page
title: "Tutorial 2: the first operation"
---

# The first operation

## Creating a device

Now that we have chosen a physical device, we can create a `Device` object.

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

Now that we 

{% highlight rust %}
let cb_pool = CommandPool::new(&device);
{% endhighlight %}
