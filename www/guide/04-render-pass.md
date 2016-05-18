---
layout: page
title: "Tutorial 4: render passes"
---

# Render passes

In the previous section, we created a window and asked the GPU to fill its surface with a color.
However our ultimate goal is to draw some shapes on that surface, not just clear it.

In order to fully optimize and parallelize commands execution, we can't just add ask the GPU
to draw a shape whenever we want. Instead we first have to enter "rendering mode" by entering
a *render pass*, then draw, and then leave the render pass.

This will serve as a foundation for the next tutorial, which is about drawing a triangle.

## What is a render pass?

A render pass describes how .
