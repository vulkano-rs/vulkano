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
use std::path::Path;
use std::ptr;

use shared_library;
use vk;

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn load_static() -> Result<vk::Static, LoadingError> {
    use std::os::raw::c_char;

    extern {
        fn vkGetInstanceProcAddr(instance: vk::Instance, pName: *const c_char)
                                 -> vk::PFN_vkVoidFunction;
    }

    extern "system" fn wrapper(instance: vk::Instance, pName: *const c_char)
                               -> vk::PFN_vkVoidFunction
    {
        unsafe {
            vkGetInstanceProcAddr(instance, pName)
        }
    }

    Ok(vk::Static {
        GetInstanceProcAddr: wrapper,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn load_static() -> Result<vk::Static, LoadingError> {
    lazy_static! {
        static ref VK_LIB: Result<shared_library::dynamic_library::DynamicLibrary, LoadingError> = {
            #[cfg(windows)] fn get_path() -> &'static Path { Path::new("vulkan-1.dll") }
            #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))] fn get_path() -> &'static Path { Path::new("libvulkan.so.1") }
            #[cfg(target_os = "android")] fn get_path() -> &'static Path { Path::new("libvulkan.so") }
            let path = get_path();
            shared_library::dynamic_library::DynamicLibrary::open(Some(path))
                                        .map_err(|err| LoadingError::LibraryLoadFailure(err))
        };
    }

    match *VK_LIB {
        Ok(ref lib) => {
            let mut err = None;
            let result = vk::Static::load(|name| unsafe {
                let name = name.to_str().unwrap();
                match lib.symbol(name) {
                    Ok(s) => s,
                    Err(_) => {     // TODO: return error?
                        err = Some(LoadingError::MissingEntryPoint(name.to_owned()));
                        ptr::null()
                    }
                }
            });

            if let Some(err) = err {
                Err(err)
            } else {
                Ok(result)
            }
        },
        Err(ref err) => Err(err.clone()),
    }
}

lazy_static! {
    static ref VK_STATIC: Result<vk::Static, LoadingError> = load_static();

    static ref VK_ENTRY: Result<vk::EntryPoints, LoadingError> = {
        match *VK_STATIC {
            Ok(ref lib) => {
                // At this point we assume that if one of the functions fails to load, it is an
                // implementation bug and not a real-life situation that could be handled by
                // an error.
                Ok(vk::EntryPoints::load(|name| unsafe {
                    mem::transmute(lib.GetInstanceProcAddr(0, name.as_ptr()))
                }))
            },
            Err(ref err) => Err(err.clone()),
        }
    };
}

/// Returns the collection of static functions from the Vulkan loader, or an error if failed to
/// open the loader.
pub fn static_functions() -> Result<&'static vk::Static, LoadingError> {
    VK_STATIC.as_ref().map_err(|err| err.clone())
}

/// Returns the collection of Vulkan entry points from the Vulkan loader, or an error if failed to
/// open the loader.
pub fn entry_points() -> Result<&'static vk::EntryPoints, LoadingError> {
    VK_ENTRY.as_ref().map_err(|err| err.clone())
}

/// Error that can happen when loading the Vulkan loader.
#[derive(Debug, Clone)]
pub enum LoadingError {
    /// Failed to load the Vulkan shared library.
    LibraryLoadFailure(String),         // TODO: meh for error type, but this needs changes in shared_library

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
