// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::os::raw::c_char;
use std::os::raw::c_void;

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
        let entry_points = vk::EntryPoints::load(|name| {
            unsafe {
                mem::transmute(loader.get_instance_proc_addr(0, name.as_ptr()))
            }
        });

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

/// Returns the default `FunctionPointers` for this system.
///
/// This function tries to auto-guess where to find the Vulkan implementation, and loads it in a
/// `lazy_static!`. The content of the lazy_static is then returned, or an error if we failed to
/// load Vulkan.
pub fn default_function_pointers()
    -> Result<&'static FunctionPointers<Box<Loader + Send + Sync>>, LoadingError>
{
    lazy_static! {
        static ref DEFAULT_LOADER: Result<FunctionPointers<Box<Loader + Send + Sync>>, LoadingError> = {
            def_loader_impl().map(FunctionPointers::new)
        };
    }

    match DEFAULT_LOADER.deref() {
        &Ok(ref ptr) => Ok(ptr),
        &Err(ref err) => Err(err.clone())
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(non_snake_case)]
fn def_loader_impl() -> Result<Box<Loader + Send + Sync>, LoadingError> {
    extern "C" {
        fn vkGetInstanceProcAddr(instance: vk::Instance, pName: *const c_char)
                                 -> vk::PFN_vkVoidFunction;
    }

    struct LoaderImpl;
    unsafe impl Loader for LoaderImpl {
        fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                                  -> extern "system" fn() -> () {
            unsafe { vkGetInstanceProcAddr(instance, name) }
        }
    }

    Ok(Box::new(LoaderImpl))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn def_loader_impl() -> Result<Box<Loader + Send + Sync>, LoadingError> {
    use std::path::Path;
    use shared_library;

    let vk_lib = {
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
        let path = get_path();

        shared_library::dynamic_library::DynamicLibrary::open(Some(path))
            .map_err(LoadingError::LibraryLoadFailure)?
    };

    let get_proc_addr = unsafe {
        let ptr: *mut c_void = vk_lib
            .symbol("GetInstanceProcAddr")
            .map_err(|_| LoadingError::MissingEntryPoint("GetInstanceProcAddr".to_owned()))?;
        mem::transmute(ptr)
    };

    struct LoaderImpl {
        vk_lib: shared_library::dynamic_library::DynamicLibrary,
        get_proc_addr: extern "system" fn(instance: vk::Instance, pName: *const c_char)
                                          -> extern "system" fn() -> (),
    }

    unsafe impl Loader for LoaderImpl {
        #[inline]
        fn get_instance_proc_addr(&self, instance: vk::Instance, name: *const c_char)
                                  -> extern "system" fn() -> () {
            (self.get_proc_addr)(instance, name)
        }
    }

    Ok(Box::new(LoaderImpl {
                    vk_lib,
                    get_proc_addr,
                }))
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
