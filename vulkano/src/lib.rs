//! 
//! # Brief summary of Vulkan
//!
//! - The `Instance` object is the API entry point. It is the first object you must create before
//!   starting to use Vulkan.
//!
//! - The `PhysicalDevice` object represents an implementation of Vulkan available on the system
//!   (eg. a graphics card, a CPU implementation, multiple graphics card working together, etc.).
//!   Physical devices can be enumerated from an instance with `PhysicalDevice::enumerate()`.
//!
//! - Once you have chosen a physical device to use, you must a `Device` object from it. The
//!   `Device` is another very important object, as it represents an open channel of
//!   communicaton with the physical device.
//!
//! - `Buffer`s and `Image`s can be used to store data on memory accessible from the GPU (or
//!   Vulkan implementation). Buffers are usually used to store vertices, lights, etc. or
//!   arbitrary data, while images are used to store textures or multi-dimensional data.
//!
//! - In order to show something on the screen, you need a `Swapchain`. A `Swapchain` contains a
//!   special `Image` that corresponds to the content of the window or the monitor. When you
//!   *present* a swapchain, the content of that special image is shown on the screen.
//!
//! - `ComputePipeline`s and `GraphicsPipeline`s describe the way the GPU must perform a certain
//!   operation. `Shader`s are programs that the GPU will execute as part of a pipeline.
//!
//! - `RenderPass`es and `Framebuffer`s describe on which attachments the implementation must draw
//!   on. They are only used for graphical operations.
//!
//! - In order to ask the GPU to do something, you must create a `CommandBuffer`. A `CommandBuffer`
//!   contains a list of commands that the GPU must perform. This can include copies between
//!   buffers, compute operations, or graphics operations. For the work to start, the
//!   `CommandBuffer` must then be submitted to a `Queue`, which is obtained when you create
//!   the `Device`.
//!

//#![warn(missing_docs)]        // TODO: activate
#![allow(dead_code)]            // TODO: remove
#![allow(unused_variables)]     // TODO: remove

#[macro_use]
extern crate lazy_static;
extern crate shared_library;

#[macro_use]
mod tests;

mod features;
mod version;

pub mod buffer;
pub mod command_buffer;
pub mod device;
pub mod formats;
pub mod framebuffer;
pub mod image;
pub mod instance;
pub mod memory;
pub mod pipeline;
//pub mod query;
pub mod sampler;
pub mod shader;
pub mod swapchain;
pub mod sync;

use std::error;
use std::fmt;
use std::mem;
use std::path::Path;

mod vk {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_snake_case)]
    #![allow(non_camel_case_types)]
    include!(concat!(env!("OUT_DIR"), "/vk_bindings.rs"));
}

lazy_static! {
    static ref VK_LIB: shared_library::dynamic_library::DynamicLibrary = {
        #[cfg(windows)] fn get_path() -> &'static Path { Path::new("vulkan-1.dll") }
        #[cfg(unix)] fn get_path() -> &'static Path { Path::new("libvulkan-1.so") }
        let path = get_path();
        shared_library::dynamic_library::DynamicLibrary::open(Some(path)).unwrap()
    };

    static ref VK_STATIC: vk::Static = {
        vk::Static::load(|name| unsafe {
            VK_LIB.symbol(name.to_str().unwrap()).unwrap()      // TODO: error handling
        })
    };

    static ref VK_ENTRY: vk::EntryPoints = {
        vk::EntryPoints::load(|name| unsafe {
            mem::transmute(VK_STATIC.GetInstanceProcAddr(0, name.as_ptr()))
        })
    };
}

/// Gives access to the internals of an object.
trait VulkanObject {
    /// The type of the object.
    type Object;

    /// Returns a reference to the object.
    fn internal_object(&self) -> Self::Object;
}

/// Gives access to the Vulkan function pointers stored in this object.
trait VulkanPointers {
    /// The struct that provides access to the function pointers.
    type Pointers;

    // Returns a reference to the pointers.
    fn pointers(&self) -> &Self::Pointers;
}

/// Error type returned by most Vulkan functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OomError {
    /// There is no memory available on the host (ie. the CPU, RAM, etc.).
    OutOfHostMemory,
    /// There is no memory available on the device (ie. video memory).
    OutOfDeviceMemory,
}

impl error::Error for OomError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            OomError::OutOfHostMemory => "no memory available on the host",
            OomError::OutOfDeviceMemory => "no memory available on the graphical device",
        }
    }
}

impl fmt::Display for OomError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for OomError {
    #[inline]
    fn from(err: Error) -> OomError {
        match err {
            Error::OutOfHostMemory => OomError::OutOfHostMemory,
            Error::OutOfDeviceMemory => OomError::OutOfDeviceMemory,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

/// All possible success codes returned by any Vulkan function.
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
enum Success {
    Success = vk::SUCCESS,
    NotReady = vk::NOT_READY,
    Timeout = vk::TIMEOUT,
    EventSet = vk::EVENT_SET,
    EventReset = vk::EVENT_RESET,
    Incomplete = vk::INCOMPLETE,
    Suboptimal = vk::SUBOPTIMAL_KHR,
}

/// All possible errors returned by any Vulkan function.
///
/// This type is not public. Instead all public error types should implement `From<Error>` and
/// panic for error code that arent supposed to happen.
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
enum Error {
    OutOfHostMemory = vk::ERROR_OUT_OF_HOST_MEMORY,
    OutOfDeviceMemory = vk::ERROR_OUT_OF_DEVICE_MEMORY,
    InitializationFailed = vk::ERROR_INITIALIZATION_FAILED,
    DeviceLost = vk::ERROR_DEVICE_LOST,
    MemoryMapFailed = vk::ERROR_MEMORY_MAP_FAILED,
    LayerNotPresent = vk::ERROR_LAYER_NOT_PRESENT,
    ExtensionNotPresent = vk::ERROR_EXTENSION_NOT_PRESENT,
    FeatureNotPresent = vk::ERROR_FEATURE_NOT_PRESENT,
    IncompatibleDriver = vk::ERROR_INCOMPATIBLE_DRIVER,
    TooManyObjects = vk::ERROR_TOO_MANY_OBJECTS,
    FormatNotSupported = vk::ERROR_FORMAT_NOT_SUPPORTED,
    SurfaceLost = vk::ERROR_SURFACE_LOST_KHR,
    NativeWindowInUse = vk::ERROR_NATIVE_WINDOW_IN_USE_KHR,
    OutOfDate = vk::ERROR_OUT_OF_DATE_KHR,
    IncompatibleDisplay = vk::ERROR_INCOMPATIBLE_DISPLAY_KHR,
    ValidationFailed = vk::ERROR_VALIDATION_FAILED_EXT,
}

/// Checks whether the result returned correctly.
fn check_errors(result: vk::Result) -> Result<Success, Error> {
    match result {
        vk::SUCCESS => Ok(Success::Success),
        vk::NOT_READY => Ok(Success::NotReady),
        vk::TIMEOUT => Ok(Success::Timeout),
        vk::EVENT_SET => Ok(Success::EventSet),
        vk::EVENT_RESET => Ok(Success::EventReset),
        vk::INCOMPLETE => Ok(Success::Incomplete),
        vk::ERROR_OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory),
        vk::ERROR_OUT_OF_DEVICE_MEMORY => Err(Error::OutOfDeviceMemory),
        vk::ERROR_INITIALIZATION_FAILED => Err(Error::InitializationFailed),
        vk::ERROR_DEVICE_LOST => Err(Error::DeviceLost),
        vk::ERROR_MEMORY_MAP_FAILED => Err(Error::MemoryMapFailed),
        vk::ERROR_LAYER_NOT_PRESENT => Err(Error::LayerNotPresent),
        vk::ERROR_EXTENSION_NOT_PRESENT => Err(Error::ExtensionNotPresent),
        vk::ERROR_FEATURE_NOT_PRESENT => Err(Error::FeatureNotPresent),
        vk::ERROR_INCOMPATIBLE_DRIVER => Err(Error::IncompatibleDriver),
        vk::ERROR_TOO_MANY_OBJECTS => Err(Error::TooManyObjects),
        vk::ERROR_FORMAT_NOT_SUPPORTED => Err(Error::FormatNotSupported),
        vk::ERROR_SURFACE_LOST_KHR => Err(Error::SurfaceLost),
        vk::ERROR_NATIVE_WINDOW_IN_USE_KHR => Err(Error::NativeWindowInUse),
        vk::SUBOPTIMAL_KHR => Ok(Success::Suboptimal),
        vk::ERROR_OUT_OF_DATE_KHR => Err(Error::OutOfDate),
        vk::ERROR_INCOMPATIBLE_DISPLAY_KHR => Err(Error::IncompatibleDisplay),
        vk::ERROR_VALIDATION_FAILED_EXT => Err(Error::ValidationFailed),
        _ => unreachable!("Unexpected error code returned by Vulkan")
    }
}
