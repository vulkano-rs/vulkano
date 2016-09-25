// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;
use std::collections::HashMap;

use buffer::sys::UnsafeBuffer;
use image::sys::UnsafeImage;

use VulkanObject;
use vk;

// TODO: rework this

pub struct StatesManager {
    buffers: HashMap<(vk::Buffer, u32), Box<Any>>,
    images: HashMap<(vk::Image, u32), Box<Any>>,
}

impl StatesManager {
    pub fn new() -> StatesManager {
        StatesManager {
            buffers: HashMap::new(),
            images: HashMap::new(),
        }
    }

    pub fn remove_buffer<T>(&mut self, buffer: &UnsafeBuffer, subkey: u32) -> Option<T>
        where T: Any
    {
        match self.buffers.remove(&(buffer.internal_object(), subkey)) {
            Some(s) => Some(*s.downcast().expect("Wrong buffer state")),
            None => None
        }
    }

    pub fn buffer_or<T, F>(&mut self, buffer: &UnsafeBuffer, subkey: u32, default: F) -> &mut T
        where T: Any, F: FnOnce() -> T
    {
        self.buffers.entry((buffer.internal_object(), subkey))
                    .or_insert_with(|| Box::new(default()) as Box<_>)
                    .downcast_mut().expect("Wrong buffer state")
    }

    pub fn remove_image<T>(&mut self, image: &UnsafeImage, subkey: u32) -> Option<T>
        where T: Any
    {
        match self.images.remove(&(image.internal_object(), subkey)) {
            Some(s) => Some(*s.downcast().expect("Wrong image state")),
            None => None
        }
    }

    pub fn image_or<T, F>(&mut self, image: &UnsafeImage, subkey: u32, default: F) -> &mut T
        where T: Any, F: FnOnce() -> T
    {
        self.images.entry((image.internal_object(), subkey))
                   .or_insert_with(|| Box::new(default()) as Box<_>)
                   .downcast_mut().expect("Wrong image state")
    }
}
