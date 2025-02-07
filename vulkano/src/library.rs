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
    instance::{InstanceExtensions, LayerProperties},
    ExtensionProperties, SafeDeref, Version, VulkanError,
};
use ash::vk;
use libloading::{Error as LibloadingError, Library};
use std::{
    error::Error,
    ffi::{CStr, CString},
    fmt::{Debug, Display, Error as FmtError, Formatter},
    mem::transmute,
    os::raw::c_char,
    path::Path,
    ptr,
    sync::Arc,
};

/// A loaded library containing a valid Vulkan implementation.
#[derive(Debug)]
pub struct VulkanLibrary {
    loader: Box<dyn Loader>,
    fns: EntryFunctions,

    api_version: Version,
    extension_properties: Vec<ExtensionProperties>,
    supported_extensions: InstanceExtensions,
}

impl VulkanLibrary {
    /// Loads the default Vulkan library for this system.
    pub fn new() -> Result<Arc<Self>, LoadingError> {
        #[cfg(any(target_os = "ios", target_os = "tvos"))]
        #[allow(non_snake_case)]
        fn def_loader_impl() -> Result<Box<dyn Loader>, LoadingError> {
            let loader = crate::statically_linked_vulkan_loader!();

            Ok(Box::new(loader))
        }

        #[cfg(not(any(target_os = "ios", target_os = "tvos")))]
        fn def_loader_impl() -> Result<Box<dyn Loader>, LoadingError> {
            #[cfg(windows)]
            const PATHS: [&str; 1] = ["vulkan-1.dll"];
            #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
            const PATHS: [&str; 1] = ["libvulkan.so.1"];
            #[cfg(target_os = "macos")]
            const PATHS: [&str; 6] = [
                "libvulkan.dylib",
                "libvulkan.1.dylib",
                "libMoltenVK.dylib",
                "vulkan.framework/vulkan",
                "MoltenVK.framework/MoltenVK",
                // Stock macOS no longer has `/usr/local/lib` in `LD_LIBRARY_PATH` like it used to,
                // but libraries (including MoltenVK installed through the Vulkan SDK) are still
                // installed here. Try the absolute path as a last resort.
                "/usr/local/lib/libvulkan.dylib",
            ];
            #[cfg(target_os = "android")]
            const PATHS: [&str; 2] = ["libvulkan.so.1", "libvulkan.so"];

            let mut err: Option<LoadingError> = None;

            for path in PATHS {
                match unsafe { DynamicLibraryLoader::new(path) } {
                    Ok(library) => return Ok(Box::new(library)),
                    Err(e) => err = Some(e),
                }
            }

            Err(err.unwrap())
        }

        def_loader_impl().and_then(VulkanLibrary::with_loader)
    }

    /// Loads a custom Vulkan library.
    pub fn with_loader(loader: impl Loader + 'static) -> Result<Arc<Self>, LoadingError> {
        let fns = EntryFunctions::load(|name| {
            unsafe { loader.get_instance_proc_addr(vk::Instance::null(), name.as_ptr()) }
                .map_or(ptr::null(), |func| func as _)
        });

        let api_version = unsafe { Self::get_api_version(&loader) }?;
        let extension_properties = unsafe { Self::get_extension_properties(&fns, None) }?;
        let supported_extensions = extension_properties
            .iter()
            .map(|property| property.extension_name.as_str())
            .collect();

        Ok(Arc::new(VulkanLibrary {
            loader: Box::new(loader),
            fns,
            api_version,
            extension_properties,
            supported_extensions,
        }))
    }

    unsafe fn get_api_version(loader: &impl Loader) -> Result<Version, VulkanError> {
        // Per the Vulkan spec:
        // If the vkGetInstanceProcAddr returns NULL for vkEnumerateInstanceVersion, it is a
        // Vulkan 1.0 implementation. Otherwise, the application can call vkEnumerateInstanceVersion
        // to determine the version of Vulkan.

        let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"vkEnumerateInstanceVersion\0") };
        let func = unsafe { loader.get_instance_proc_addr(vk::Instance::null(), name.as_ptr()) };

        let version = if let Some(func) = func {
            let func: vk::PFN_vkEnumerateInstanceVersion = unsafe { transmute(func) };
            let mut api_version = 0;
            unsafe { func(&mut api_version) }
                .result()
                .map_err(VulkanError::from)?;
            Version::from(api_version)
        } else {
            Version {
                major: 1,
                minor: 0,
                patch: 0,
            }
        };

        Ok(version)
    }

    unsafe fn get_extension_properties(
        fns: &EntryFunctions,
        layer: Option<&str>,
    ) -> Result<Vec<ExtensionProperties>, VulkanError> {
        let layer_vk = layer.map(|layer| CString::new(layer).unwrap());

        loop {
            let mut count = 0;
            unsafe {
                (fns.v1_0.enumerate_instance_extension_properties)(
                    layer_vk
                        .as_ref()
                        .map_or(ptr::null(), |layer| layer.as_ptr()),
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut output = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.v1_0.enumerate_instance_extension_properties)(
                    layer_vk
                        .as_ref()
                        .map_or(ptr::null(), |layer| layer.as_ptr()),
                    &mut count,
                    output.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { output.set_len(count as usize) };
                    return Ok(output.into_iter().map(Into::into).collect());
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        }
    }

    /// Returns pointers to the raw global Vulkan functions of the library.
    #[inline]
    pub fn fns(&self) -> &EntryFunctions {
        &self.fns
    }

    /// Returns the highest Vulkan version that is supported for instances.
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the extension properties reported by the core library.
    #[inline]
    pub fn extension_properties(&self) -> &[ExtensionProperties] {
        &self.extension_properties
    }

    /// Returns the extensions that are supported by the core library.
    #[inline]
    pub fn supported_extensions(&self) -> &InstanceExtensions {
        &self.supported_extensions
    }

    /// Returns the list of layers that are available when creating an instance.
    ///
    /// On success, this function returns an iterator that produces
    /// [`LayerProperties`] objects. In order to enable a layer,
    /// you need to pass its name (returned by `LayerProperties::name()`) when creating the
    /// [`Instance`](crate::instance::Instance).
    ///
    /// > **Note**: The available layers may change between successive calls to this function, so
    /// > each call may return different results. It is possible that one of the layers enumerated
    /// > here is no longer available when you create the `Instance`. This will lead to an error
    /// > when calling `Instance::new`.
    ///
    /// # Examples
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
    ) -> Result<impl ExactSizeIterator<Item = LayerProperties>, VulkanError> {
        let fns = self.fns();

        let layer_properties = loop {
            let mut count = 0;
            unsafe { (fns.v1_0.enumerate_instance_layer_properties)(&mut count, ptr::null_mut()) }
                .result()
                .map_err(VulkanError::from)?;

            let mut properties = Vec::with_capacity(count as usize);
            let result = unsafe {
                (fns.v1_0.enumerate_instance_layer_properties)(&mut count, properties.as_mut_ptr())
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { properties.set_len(count as usize) };
                    break properties;
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok(layer_properties
            .into_iter()
            .map(|p| LayerProperties { props: p }))
    }

    /// Returns the extension properties that are reported by the given layer.
    #[inline]
    pub fn layer_extension_properties(
        &self,
        layer: &str,
    ) -> Result<Vec<ExtensionProperties>, VulkanError> {
        unsafe { Self::get_extension_properties(&self.fns, Some(layer)) }
    }

    /// Returns the extensions that are supported by the given layer.
    #[inline]
    pub fn supported_layer_extensions(
        &self,
        layer: &str,
    ) -> Result<InstanceExtensions, VulkanError> {
        Ok(self
            .layer_extension_properties(layer)?
            .iter()
            .map(|property| property.extension_name.as_str())
            .collect())
    }

    /// Returns the union of the extensions that are supported by the core library and all
    /// the given layers.
    #[inline]
    pub fn supported_extensions_with_layers<'a>(
        &self,
        layers: impl IntoIterator<Item = &'a str>,
    ) -> Result<InstanceExtensions, VulkanError> {
        layers
            .into_iter()
            .try_fold(self.supported_extensions, |extensions, layer| {
                self.supported_layer_extensions(layer)
                    .map(|layer_extensions| extensions.union(&layer_extensions))
            })
    }

    /// Calls `get_instance_proc_addr` on the underlying loader.
    #[inline]
    pub unsafe fn get_instance_proc_addr(
        &self,
        instance: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        unsafe { self.loader.get_instance_proc_addr(instance, name) }
    }
}

/// Implemented on objects that grant access to a Vulkan implementation.
pub unsafe trait Loader: Send + Sync {
    /// Calls the `vkGetInstanceProcAddr` function. The parameters are the same.
    ///
    /// The returned function must stay valid for as long as `self` is alive.
    unsafe fn get_instance_proc_addr(
        &self,
        instance: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction;
}

unsafe impl<T> Loader for T
where
    T: SafeDeref + Send + Sync,
    T::Target: Loader,
{
    unsafe fn get_instance_proc_addr(
        &self,
        instance: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        unsafe { (**self).get_instance_proc_addr(instance, name) }
    }
}

impl Debug for dyn Loader {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("Loader").finish_non_exhaustive()
    }
}

/// Implementation of `Loader` that loads Vulkan from a dynamic library.
pub struct DynamicLibraryLoader {
    _vk_lib: Library,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
}

impl DynamicLibraryLoader {
    /// Tries to load the dynamic library at the given path, and tries to
    /// load `vkGetInstanceProcAddr` in it.
    ///
    /// # Safety
    ///
    /// - The dynamic library must be a valid Vulkan implementation.
    pub unsafe fn new(path: impl AsRef<Path>) -> Result<DynamicLibraryLoader, LoadingError> {
        let vk_lib =
            unsafe { Library::new(path.as_ref()) }.map_err(LoadingError::LibraryLoadFailure)?;

        let get_instance_proc_addr = *unsafe { vk_lib.get(b"vkGetInstanceProcAddr") }
            .map_err(LoadingError::LibraryLoadFailure)?;

        Ok(DynamicLibraryLoader {
            _vk_lib: vk_lib,
            get_instance_proc_addr,
        })
    }
}

unsafe impl Loader for DynamicLibraryLoader {
    #[inline]
    unsafe fn get_instance_proc_addr(
        &self,
        instance: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction {
        unsafe { (self.get_instance_proc_addr)(instance, name) }
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
                instance: vk::Instance,
                pName: *const c_char,
            ) -> vk::PFN_vkVoidFunction;
        }

        struct StaticallyLinkedVulkanLoader;
        unsafe impl Loader for StaticallyLinkedVulkanLoader {
            unsafe fn get_instance_proc_addr(
                &self,
                instance: vk::Instance,
                name: *const c_char,
            ) -> vk::PFN_vkVoidFunction {
                vkGetInstanceProcAddr(instance, name)
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

    /// The Vulkan driver returned an error and was unable to complete the operation.
    VulkanError(VulkanError),
}

impl Error for LoadingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            //Self::LibraryLoadFailure(err) => Some(err),
            Self::VulkanError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for LoadingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::LibraryLoadFailure(_) => write!(f, "failed to load the Vulkan shared library"),
            Self::VulkanError(err) => write!(f, "a runtime error occurred: {err}"),
        }
    }
}

impl From<VulkanError> for LoadingError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::{DynamicLibraryLoader, LoadingError};

    #[test]
    fn dl_open_error() {
        match unsafe { DynamicLibraryLoader::new("_non_existing_library.void") } {
            Err(LoadingError::LibraryLoadFailure(_)) => (),
            _ => panic!(),
        }
    }
}
