# Initialization

## Creating an instance

Before you can start using any function from Vulkan and vulkano, the first thing to do is to create
an *instance*. Creating an instance tries to load Vulkan from the system and reads the list of
available implementations.

Creating an instance takes three optional parameters which aren't going to cover for now. You can
check [the documentation of `Instance`](https://docs.rs/vulkano/0.4/vulkano/instance/struct.Instance.html)
for more information.

    use vulkano::instance::Instance;
    use vulkano::instance::InstanceExtensions;
     
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");

Like many other functions in vulkano, creating an instance returns a `Result`. If Vulkan is not
available on the system, this result will contain an error. For the sake of this example we call
`expect` on the `Result`, which prints a message to stderr and terminates the application if it
contains an error. In a real game or application you should handle that situation in a nicer way,
for example by opening a dialog box with an explanation. This is out of scope of this guide.

Before going further you should try your code by running:

    cargo run

## Enumerating physical devices

The machine you run your program on may have multiple devices that support Vulkan. Before we can
ask a video card to perform some operations, we have to enumerate all the *physical device*s that
support Vulkan and choose which one we are going to use.

A physical device can be a dedicated graphics card, but also an integrated graphics processor
or a software implementation. It can be basically anything that allows running Vulkan operations.

As of the writing of this guide, it is not yet possible to use multiple devices simultaneously
in an efficient way (eg. SLI/Crossfire). You *can* use multiple devices simultaneously in the same
program, but there is not much point in doing so because you cannot share anything between them.
Consequently the best thing to do is to chose one physical device which is going to run everything:

    use vulkano::instance::PhysicalDevice;
     
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

The `enumerate` function returns an iterator to the list of available physical devices.
We call `next` on it to return the first device, if any. Note that the first device is not
necessarily the best device. In a real program you probably want to leave the choice to the user.

Keep in mind that the list of physical devices can be empty. This happens if Vulkan is installed
on the system, but none of the physical devices are capable of supporting Vulkan. In a real-world
application you are encouraged to handle this situation properly.
