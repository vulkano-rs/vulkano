# Introduction

Welcome to the vulkano guide!

This guide is going to cover the basics of Vulkan and vulkano. It won't cover graphics programming.
This guide also assumes that you are familiar with the Rust programming language.

As with all Rust libraries, add this entry in your Cargo.toml:

    vulkano = "0.4"

And this at your crate root:

    #[macro_use]
    extern crate vulkano;
