---
layout: page
title: "Tutorial 1: getting started"
---

# Getting started

{% highlight toml %}
vulkano = "0.1"
{% endhighlight %}

## Creating an instance

The first thing any Vulkan program should do is create an instance. Creating an instance checks
whether Vulkan is supported on the system and loads the list of available devices from the
environment.

{% highlight rust %}
// Put this at the top of the file.
use vulkano::instance::Instance;

// Put this inside the main function.
let instance = Instance::new(None, &Default::default(), None)
                        .expect("failed to create instance");
{% endhighlight %}

There are three optional parameters that we can pass to the `new` functions: a description of your
application, a list of extensions to enable, and a list of layers to enable. We don't need any of
these for the moment.

Like many other functions in vulkano, creating an instance returns a `Result`. If Vulkan is not
installed on the system, this result will contain an error. For the sake of this example we call
`expect` on the `Result`, which prints a message to stderr and terminates the application. In a
real game or application you should handle that situation in a nicer way, for example by opening
a dialog box with an explanation.

You can now try your code by running:

{% highlight bash %}
cargo run
{% endhighlight %}

## Enumerating physical devices

The machine you run your program on may have multiple devices that support Vulkan. Before we can
ask a video card to perform some operations, we have to enumerate all the physical devices that
support Vulkan and choose which one we are going to use.

As of the writing of this tutorial, it is not possible to share resources between multiple
physical devices. The consequence is that you would probably gain nothing from using multiple
devices at once. At the moment everybody chooses the "best" device and uses it exclusively, like
this:

{% highlight rust %}
use vulkano::instance::PhysicalDevice;

let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
{% endhighlight %}

The `enumerate` function returns an iterator to the list of available physical devices.
We call `next` on it to return the first device, if any. Note that the first device is not
necessarily the best device. In a real program you probably want to choose in a better way
or leave the choice to the user.

It is possible for this iterator to be empty, in which case the code above will panic. This
happens if Vulkan is installed on the system, but none of the physical devices are capable
of supporting Vulkan.
