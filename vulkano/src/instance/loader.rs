// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Vulkan implementation loading system.
//!
//! Before vulkano can do anything, it first needs to find an implementation of Vulkan. A Vulkan
//! implementation is defined as a single `vkGetInstanceProcAddr` function, which can be accessed
//! through the `Loader` trait.
//!
//! This module provides various implementations of the `Loader` trait.
//!
//! Once you have a struct that implements `Loader`, you can create a `FunctionPointers` struct
//! from it and use this `FunctionPointers` struct to build an `Instance`.
//!
//! By default vulkano will use the `auto_loader()` function, which tries to automatically load
//! a Vulkan implementation from the system.

use shared_library;
use std::error;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::os::raw::c_char;
use std::os::raw::c_void;
use std::path::Path;

use SafeDeref;
use vk;

/// Implemented on objects that grant access to a Vulkan implementation.
pub unsafe trait Loader {
    /// Calls the `vkGetInstanceProcAddr` function. The parameters are the same.
    ///
    /// The returned function must stay valid for as long as `self` is alive.
    fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                              -> extern "system" fn() -> ();
}

unsafe impl<T> Loader for T
    where T: SafeDeref,
          T::Target: Loader
{
    #[inline]
    fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                              -> extern "system" fn() -> () {
        (**self).get_instance_proc_addr(instance, name)
    }
}

/// Implementation of `Loader` that loads Vulkan from a dynamic library.
pub struct DynamicLibraryLoader {
    vk_lib: shared_library::dynamic_library::DynamicLibrary,
    get_proc_addr: extern "system" fn(instance: vk::Instance, pName: *const c_char)
                                      -> extern "system" fn() -> (),
}

impl DynamicLibraryLoader {
    /// Tries to load the dynamic library at the given path, and tries to
    /// load `vkGetInstanceProcAddr` in it.
    ///
    /// # Safety
    ///
    /// - The dynamic library must be a valid Vulkan implementation.
    ///
    pub unsafe fn new<P>(path: P) -> Result<DynamicLibraryLoader, LoadingError>
        where P: AsRef<Path>
    {
        let vk_lib = shared_library::dynamic_library::DynamicLibrary::open(Some(path.as_ref()))
            .map_err(LoadingError::LibraryLoadFailure)?;

        let get_proc_addr = {
            let ptr: *mut c_void =
                vk_lib
                    .symbol("vkGetInstanceProcAddr")
                    .map_err(|_| {
                                 LoadingError::MissingEntryPoint("vkGetInstanceProcAddr".to_owned())
                             })?;
            mem::transmute(ptr)
        };

        Ok(DynamicLibraryLoader {
               vk_lib,
               get_proc_addr,
           })
    }
}

unsafe impl Loader for DynamicLibraryLoader {
    #[inline]
    fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                              -> extern "system" fn() -> () {
        (self.get_proc_addr)(instance, name)
    }
}

/// Wraps around a loader and contains function pointers.
pub struct FunctionPointers<L> {
    loader: L,
    entry_points: vk::EntryPoints,
}

impl<L> FunctionPointers<L> {
    /// Loads some global function pointer from the loader.
    pub fn new(loader: L) -> FunctionPointers<L>
        where L: Loader
    {
        let entry_points = vk::EntryPoints::load(|name| unsafe { mem::transmute(loader.get_instance_proc_addr(0, name.as_ptr())) });

        FunctionPointers {
            loader,
            entry_points,
        }
    }

    /// Returns the collection of Vulkan entry points from the Vulkan loader.
    #[inline]
    pub(crate) fn entry_points(&self) -> &vk::EntryPoints {
        &self.entry_points
    }

    /// Calls `get_instance_proc_addr` on the underlying loader.
    #[inline]
    pub fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                                  -> extern "system" fn() -> ()
        where L: Loader
    {
        self.loader.get_instance_proc_addr(instance, name)
    }
}

/// Expression that returns a loader that assumes that Vulkan is linked to the executable you're
/// compiling.
///
/// If you use this macro, you must linked to a library that provides the `vkGetInstanceProcAddr`
/// symbol.
///
/// This is provided as a macro and not as a regular function, because the macro contains an
/// `extern {}` block.
// TODO: should this be unsafe?
#[macro_export]
macro_rules! statically_linked_vulkan_loader {
    () => ({
        extern "C" {
            fn vkGetInstanceProcAddr(instance: vk::Instance, pName: *const c_char)
                                    -> vk::PFN_vkVoidFunction;
        }

        struct StaticallyLinkedVulkanLoader;
        unsafe impl Loader for StaticallyLinkedVulkanLoader {
            fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                                    -> extern "system" fn() -> () {
                unsafe { vkGetInstanceProcAddr(instance, name) }
            }
        }

        StaticallyLinkedVulkanLoader
    })
}

/// Returns the default `FunctionPointers` for this system.
///
/// This function tries to auto-guess where to find the Vulkan implementation, and loads it in a
/// `lazy_static!`. The content of the lazy_static is then returned, or an error if we failed to
/// load Vulkan.
pub fn auto_loader(
    )
    -> Result<&'static FunctionPointers<Box<Loader + Send + Sync>>, LoadingError>
{
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(non_snake_case)]
    fn def_loader_impl() -> Result<Box<Loader + Send + Sync>, LoadingError> {
        let loader = statically_linked_vulkan_loader!();
        Ok(Box::new(loader))
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn def_loader_impl() -> Result<Box<Loader + Send + Sync>, LoadingError> {
        #[cfg(windows)]
        fn get_path() -> &'static Path {
            Path::new("vulkan-1.dll")
        }
        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        fn get_path() -> &'static Path {
            Path::new("libvulkan.so.1")
        }
        #[cfg(target_os = "android")]
        fn get_path() -> &'static Path {
            Path::new("libvulkan.so")
        }

        let loader = unsafe { DynamicLibraryLoader::new(get_path())? };

        Ok(Box::new(loader))
    }

    lazy_static! {
        static ref DEFAULT_LOADER: Result<FunctionPointers<Box<Loader + Send + Sync>>, LoadingError> = {
            def_loader_impl().map(FunctionPointers::new)
        };
    }

    match DEFAULT_LOADER.deref() {
        &Ok(ref ptr) => Ok(ptr),
        &Err(ref err) => Err(err.clone()),
    }
}

/// Error that can happen when loading the Vulkan loader.
#[derive(Debug, Clone)]
pub enum LoadingError {
    /// Failed to load the Vulkan shared library.
    LibraryLoadFailure(String), // TODO: meh for error type, but this needs changes in shared_library

    /// One of the entry points required to be supported by the Vulkan implementation is missing.
    MissingEntryPoint(String),
}

impl error::Error for LoadingError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            LoadingError::LibraryLoadFailure(_) => {
                "failed to load the Vulkan shared library"
            },
            LoadingError::MissingEntryPoint(_) => {
                "one of the entry points required to be supported by the Vulkan implementation \
                 is missing"
            },
        }
    }

    /*#[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            LoadingError::LibraryLoadFailure(ref err) => Some(err),
            _ => None
        }
    }*/
}

impl fmt::Display for LoadingError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

#[cfg(test)]
mod tests {
    use instance::loader::DynamicLibraryLoader;
    use instance::loader::LoadingError;

    #[test]
    fn dl_open_error() {
        unsafe {
            match DynamicLibraryLoader::new("_non_existing_library.void") {
                Err(LoadingError::LibraryLoadFailure(_)) => (),
                _ => panic!(),
            }
        }
    }
}
