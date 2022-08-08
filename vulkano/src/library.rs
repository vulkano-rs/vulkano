// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Vulkan library loading system.
//!
//! Before Vulkano can do anything, it first needs to find a library containing an implementation
//! of Vulkan. A Vulkan implementation is defined as a single `vkGetInstanceProcAddr` function,
//! which can be accessed through the `Loader` trait.
//!
//! This module provides various implementations of the `Loader` trait.
//!
//! Once you have a type that implements `Loader`, you can create a `VulkanLibrary`
//! from it and use this `VulkanLibrary` struct to build an `Instance`.

pub use crate::fns::EntryFunctions;
use crate::{
    check_errors,
    instance::{InstanceExtensions, LayerProperties},
    Error, OomError, SafeDeref, Success, Version,
};
use libloading::{Error as LibloadingError, Library};
use std::{error, ffi::CStr, fmt, mem::transmute, os::raw::c_char, path::Path, ptr, sync::Arc};

/// A loaded library containing a valid Vulkan implementation.
#[derive(Debug)]
pub struct VulkanLibrary {
    loader: Box<dyn Loader>,
    fns: EntryFunctions,

    api_version: Version,
    supported_extensions: InstanceExtensions,
}

impl VulkanLibrary {
    /// Loads the default Vulkan library for this system.
    pub fn new() -> Result<Arc<Self>, LoadingError> {
        #[cfg(target_os = "ios")]
        #[allow(non_snake_case)]
        fn def_loader_impl() -> Result<Box<dyn Loader>, LoadingError> {
            let loader = crate::statically_linked_vulkan_loader!();
            Ok(Box::new(loader))
        }

        #[cfg(not(target_os = "ios"))]
        fn def_loader_impl() -> Result<Box<dyn Loader>, LoadingError> {
            #[cfg(windows)]
            fn get_path() -> &'static Path {
                Path::new("vulkan-1.dll")
            }
            #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
            fn get_path() -> &'static Path {
                Path::new("libvulkan.so.1")
            }
            #[cfg(target_os = "macos")]
            fn get_path() -> &'static Path {
                Path::new("libvulkan.1.dylib")
            }
            #[cfg(target_os = "android")]
            fn get_path() -> &'static Path {
                Path::new("libvulkan.so")
            }

            let loader = unsafe { DynamicLibraryLoader::new(get_path())? };

            Ok(Box::new(loader))
        }

        def_loader_impl().and_then(VulkanLibrary::with_loader)
    }

    /// Loads a custom Vulkan library.
    pub fn with_loader<L>(loader: L) -> Result<Arc<Self>, LoadingError>
    where
        L: Loader + 'static,
    {
        let fns = EntryFunctions::load(|name| unsafe {
            loader
                .get_instance_proc_addr(ash::vk::Instance::null(), name.as_ptr())
                .map_or(ptr::null(), |func| func as _)
        });
        // Per the Vulkan spec:
        // If the vkGetInstanceProcAddr returns NULL for vkEnumerateInstanceVersion, it is a
        // Vulkan 1.0 implementation. Otherwise, the application can call vkEnumerateInstanceVersion
        // to determine the version of Vulkan.
        let api_version = unsafe {
            let name = CStr::from_bytes_with_nul_unchecked(b"vkEnumerateInstanceVersion\0");
            let func = loader.get_instance_proc_addr(ash::vk::Instance::null(), name.as_ptr());

            if let Some(func) = func {
                let func: ash::vk::PFN_vkEnumerateInstanceVersion = transmute(func);
                let mut api_version = 0;
                check_errors(func(&mut api_version))?;
                Version::from(api_version)
            } else {
                Version {
                    major: 1,
                    minor: 0,
                    patch: 0,
                }
            }
        };

        let supported_extensions = unsafe {
            let extension_properties = loop {
                let mut count = 0;
                check_errors((fns.v1_0.enumerate_instance_extension_properties)(
                    ptr::null(),
                    &mut count,
                    ptr::null_mut(),
                ))?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = check_errors((fns.v1_0.enumerate_instance_extension_properties)(
                    ptr::null(),
                    &mut count,
                    properties.as_mut_ptr(),
                ))?;

                if !matches!(result, Success::Incomplete) {
                    properties.set_len(count as usize);
                    break properties;
                }
            };

            InstanceExtensions::from(
                extension_properties
                    .iter()
                    .map(|property| CStr::from_ptr(property.extension_name.as_ptr())),
            )
        };

        Ok(Arc::new(VulkanLibrary {
            loader: Box::new(loader),
            fns,
            api_version,
            supported_extensions,
        }))
    }

    /// Returns pointers to the raw global Vulkan functions of the library.
    #[inline]
    pub fn fns(&self) -> &EntryFunctions {
        &self.fns
    }

    /// Returns the highest Vulkan version that is supported for instances.
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the extensions that are supported by this Vulkan library.
    #[inline]
    pub fn supported_extensions(&self) -> &InstanceExtensions {
        &self.supported_extensions
    }

    /// Returns the list of layers that are available when creating an instance.
    ///
    /// On success, this function returns an iterator that produces
    /// [`LayerProperties`](crate::instance::LayerProperties) objects. In order to enable a layer,
    /// you need to pass its name (returned by `LayerProperties::name()`) when creating the
    /// [`Instance`](crate::instance::Instance).
    ///
    /// > **Note**: The available layers may change between successive calls to this function, so
    /// > each call may return different results. It is possible that one of the layers enumerated
    /// > here is no longer available when you create the `Instance`. This will lead to an error
    /// > when calling `Instance::new`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::VulkanLibrary;
    ///
    /// let library = VulkanLibrary::new().unwrap();
    ///
    /// for layer in library.layer_properties().unwrap() {
    ///     println!("Available layer: {}", layer.name());
    /// }
    /// ```
    pub fn layer_properties(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = LayerProperties>, OomError> {
        let fns = self.fns();

        let layer_properties = unsafe {
            loop {
                let mut count = 0;
                check_errors((fns.v1_0.enumerate_instance_layer_properties)(
                    &mut count,
                    ptr::null_mut(),
                ))?;

                let mut properties = Vec::with_capacity(count as usize);
                let result = check_errors({
                    (fns.v1_0.enumerate_instance_layer_properties)(
                        &mut count,
                        properties.as_mut_ptr(),
                    )
                })?;

                if !matches!(result, Success::Incomplete) {
                    properties.set_len(count as usize);
                    break properties;
                }
            }
        };

        Ok(layer_properties
            .into_iter()
            .map(|p| LayerProperties { props: p }))
    }

    /// Calls `get_instance_proc_addr` on the underlying loader.
    #[inline]
    pub unsafe fn get_instance_proc_addr(
        &self,
        instance: ash::vk::Instance,
        name: *const c_char,
    ) -> ash::vk::PFN_vkVoidFunction {
        self.loader.get_instance_proc_addr(instance, name)
    }
}

/// Implemented on objects that grant access to a Vulkan implementation.
pub unsafe trait Loader: Send + Sync {
    /// Calls the `vkGetInstanceProcAddr` function. The parameters are the same.
    ///
    /// The returned function must stay valid for as long as `self` is alive.
    unsafe fn get_instance_proc_addr(
        &self,
        instance: ash::vk::Instance,
        name: *const c_char,
    ) -> ash::vk::PFN_vkVoidFunction;
}

unsafe impl<T> Loader for T
where
    T: SafeDeref + Send + Sync,
    T::Target: Loader,
{
    #[inline]
    unsafe fn get_instance_proc_addr(
        &self,
        instance: ash::vk::Instance,
        name: *const c_char,
    ) -> ash::vk::PFN_vkVoidFunction {
        (**self).get_instance_proc_addr(instance, name)
    }
}

impl fmt::Debug for dyn Loader {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

/// Implementation of `Loader` that loads Vulkan from a dynamic library.
pub struct DynamicLibraryLoader {
    vk_lib: Library,
    get_instance_proc_addr: ash::vk::PFN_vkGetInstanceProcAddr,
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
    where
        P: AsRef<Path>,
    {
        let vk_lib = Library::new(path.as_ref()).map_err(LoadingError::LibraryLoadFailure)?;

        let get_instance_proc_addr = *vk_lib
            .get(b"vkGetInstanceProcAddr")
            .map_err(LoadingError::LibraryLoadFailure)?;

        Ok(DynamicLibraryLoader {
            vk_lib,
            get_instance_proc_addr,
        })
    }
}

unsafe impl Loader for DynamicLibraryLoader {
    #[inline]
    unsafe fn get_instance_proc_addr(
        &self,
        instance: ash::vk::Instance,
        name: *const c_char,
    ) -> ash::vk::PFN_vkVoidFunction {
        (self.get_instance_proc_addr)(instance, name)
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
    () => {{
        extern "C" {
            fn vkGetInstanceProcAddr(
                instance: ash::vk::Instance,
                pName: *const c_char,
            ) -> ash::vk::PFN_vkVoidFunction;
        }

        struct StaticallyLinkedVulkanLoader;
        unsafe impl Loader for StaticallyLinkedVulkanLoader {
            unsafe fn get_instance_proc_addr(
                &self,
                instance: ash::vk::Instance,
                name: *const c_char,
            ) -> ash::vk::PFN_vkVoidFunction {
                unsafe { vkGetInstanceProcAddr(instance, name) }
            }
        }

        StaticallyLinkedVulkanLoader
    }};
}

/// Error that can happen when loading a Vulkan library.
#[derive(Debug)]
pub enum LoadingError {
    /// Failed to load the Vulkan shared library.
    LibraryLoadFailure(LibloadingError),

    /// Not enough memory.
    OomError(OomError),
}

impl error::Error for LoadingError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            //Self::LibraryLoadFailure(ref err) => Some(err),
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for LoadingError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::LibraryLoadFailure(_) => "failed to load the Vulkan shared library",
                Self::OomError(_) => "not enough memory available",
            }
        )
    }
}

impl From<Error> for LoadingError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            err @ Error::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DynamicLibraryLoader, LoadingError};

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
