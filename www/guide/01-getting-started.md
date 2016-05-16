---
layout: page
title: "Tutorial 1: getting started"
---

# Getting started

{% highlight toml %}
vulkano = "0.1"
{% endhighlight %}


{% highlight rust %}
extern crate vulkano;

use vulkano::instance::Instance;

fn main() {
    let instance = Instance::new(None, &Default::default(), None)
                            .expect("failed to create instance");
}
{% endhighlight %}

This code does the first thing any Vulkan program should do: create an instance.

You can now try your code by running:

{% highlight bash %}
cargo run
{% endhighlight %}

# Enumerating physical devices
