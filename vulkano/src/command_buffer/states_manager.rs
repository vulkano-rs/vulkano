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

use VulkanObject;

// TODO: rework this

pub struct StatesManager {
    buffers: HashMap<(vk::Buffer, u32), Box<Any>>,
    images: HashMap<(vk::Image, u32), Box<Any>>,
}

impl StatesManager {
    pub fn buffer_or<T, F>(&mut self, buffer: &UnsafeBuffer, subkey: u32, default: F)
        where T: Any, F: FnOnce() -> T
    {
        self.buffers.entry((buffer.internal_object(), subkey))
                    .or_insert_with(default)
                    .downcast_mut().expect("Wrong buffer state")
    }

    pub fn remove_image<T>(&mut self, image: &UnsafeImage, subkey: u32) -> Option<T>
        where T: Any
    {
        self.images.remove(&(image.internal_object(), subkey))
    }

    pub fn image_or<T, F>(&mut self, image: &UnsafeImage, subkey: u32, default: F)
        where T: Any, F: FnOnce() -> T
    {
        self.images.entry((image.internal_object(), subkey))
                   .or_insert_with(default)
                   .downcast_mut().expect("Wrong image state")
    }
}
