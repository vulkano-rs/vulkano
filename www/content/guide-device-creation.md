# Creating a device

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

    let (device, mut queues) = {
        Device::new(&physical, physical.supported_features(), &DeviceExtensions::none(), None,
                    [(queue, 0.5)].iter().cloned()).expect("failed to create device")
    };

We now have an open channel of communication with a Vulkan device!

In the rest of this article, we are going to ask the device to copy data from a buffer to
another. Copying data is an operation that you do very often in Vulkan, so let's get used
to it early.
