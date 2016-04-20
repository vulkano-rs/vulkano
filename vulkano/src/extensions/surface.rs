// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


use std::sync::Arc;

use super::instance::{
    InstanceExtensions,
    InstanceExtension,
    InstanceExtensionDescriptor,
};

use instance::Instance;
use vk;
use swapchain::Surface;


pub struct SurfaceExtension {
    instance: Arc<Instance>,
}

impl InstanceExtension for SurfaceExtension {
    instance_extension_descriptor!("VK_KHR_surface", []);
    
    fn new(extensions: Arc<InstanceExtensions>) -> Arc<SurfaceExtension> {
        if !extensions.is_supported(Self::descriptor()) {
            panic!("Surface extension cannot be created because it is not supported.");
        }
        
        Arc::new(SurfaceExtension {
            instance: extensions.instance(),
        })
    }
    
    fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
}

impl SurfaceExtension {
    pub unsafe fn create_surface(&self, vk_surface: vk::SurfaceKHR) -> Arc<Surface> {
        // TODO
        // or create something like "SurfaceCreator" in PlatformSurfaceExtension which can be passed to Surface::new
        unimplemented!();
    }
}


pub trait PlatformSurfaceExtension: InstanceExtension {
    fn surface_extension(&self) -> Arc<SurfaceExtension>;
}


pub struct Win32SurfaceExtension {
    instance: Arc<Instance>,
    surface_extension: Arc<SurfaceExtension>,
}

impl InstanceExtension for Win32SurfaceExtension {
    instance_extension_descriptor!("VK_KHR_win32_surface", [SurfaceExtension]);
    
    fn new(extensions: Arc<InstanceExtensions>) -> Arc<Win32SurfaceExtension> {
        if !extensions.is_supported(Self::descriptor()) {
            panic!("Win32 surface extension cannot be created because it is not supported.");
        }
        
        Arc::new(Win32SurfaceExtension {
            instance: extensions.instance(),
            surface_extension: SurfaceExtension::new(extensions),
        })
    }
    
    fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
}

impl PlatformSurfaceExtension for Win32SurfaceExtension {
    fn surface_extension(&self) -> Arc<SurfaceExtension> {
        self.surface_extension.clone()
    }
}

impl Win32SurfaceExtension {
    pub fn create_surface<T, U>(hinstance: *const T, hwnd: *const U) -> Arc<Surface> {
        unimplemented!();
        // TODO: create surface: vk::SurfaceKHR, then call
        // self.surface_extension.create_surface(surface)
    }
}