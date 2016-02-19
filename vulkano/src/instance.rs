//! API entry point.
//! 
//! Creating an instance initializes everything and allows you to:
//! 
//!  - Enumerate physical devices.
//!  - Enumerate monitors.
//!  - Create surfaces.
//!
//! # Application info
//! 
//! When you create an instance, you have the possibility to pass an `ApplicationInfo` struct. This
//! struct contains various information about your application, most notably its name and engine.
//! 
//! Passing such a structure allows for example the driver to let the user configure the driver's
//! behavior for your application alone through a control panel.
//!
//! # Enumerating physical devices
//! 
//! After you have created an instance, the next step is to enumerate the physical devices that
//! are available on the system with `PhysicalDevice::enumerate()`.
//!
//! When choosing which physical device to use, keep in mind that physical devices may or may not
//! be able to draw to a certain surface (ie. to a window or a monitor). See the `swapchain`
//! module for more info.
//!
//! A physical device can designate a video card, an integrated chip, but also multiple video
//! cards working together. Once you have chosen a physical device, you can create a `Device`
//! from it. See the `device` module for more info.
//!
use std::error;
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::mem;
use std::os::raw::{c_void, c_char};
use std::ptr;
use std::sync::Arc;

//use alloc::Alloc;
use check_errors;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;
use VK_ENTRY;
use VK_STATIC;

pub use features::Features;
pub use version::Version;

/// An instance of a Vulkan context. This is the main object that should be created by an
/// application before everything else.
pub struct Instance {
    instance: vk::Instance,
    debug_report: Option<vk::DebugReportCallbackEXT>,
    //alloc: Option<Box<Alloc + Send + Sync>>,
    physical_devices: Vec<PhysicalDeviceInfos>,
    vk: vk::InstancePointers,
}

impl Instance {
    /// Initializes a new instance of Vulkan.
    // TODO: if no allocator is specified by the user, use Rust's allocator instead of leaving
    //       the choice to Vulkan
    pub fn new<'a, L>(app_infos: Option<&ApplicationInfo>, layers: L)
                      -> Result<Arc<Instance>, InstanceCreationError>
        where L: IntoIterator<Item = &'a &'a str>
    {
        // Building the CStrings from the `str`s within `app_infos`.
        // They need to be created ahead of time, since we pass pointers to them.
        let app_infos_strings = if let Some(app_infos) = app_infos {
            Some((
                CString::new(app_infos.application_name).unwrap(),
                CString::new(app_infos.engine_name).unwrap()
            ))
        } else {
            None
        };

        // Building the `vk::ApplicationInfo` if required.
        let app_infos = if let Some(app_infos) = app_infos {
            Some(vk::ApplicationInfo {
                sType: vk::STRUCTURE_TYPE_APPLICATION_INFO,
                pNext: ptr::null(),
                pApplicationName: app_infos_strings.as_ref().unwrap().0.as_ptr(),
                applicationVersion: app_infos.application_version,
                pEngineName: app_infos_strings.as_ref().unwrap().1.as_ptr(),
                engineVersion: app_infos.engine_version,
                apiVersion: Version { major: 1, minor: 0, patch: 0 }.into_vulkan_version(), // TODO: 
            })

        } else {
            None
        };

        let layers = layers.into_iter().map(|&layer| {
            // FIXME: check whether each layer is supported
            CString::new(layer).unwrap()
        }).collect::<Vec<_>>();
        let layers = layers.iter().map(|layer| {
            layer.as_ptr()
        }).collect::<Vec<_>>();

        let extensions = ["VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report"].iter().map(|&ext| {
            // FIXME: check whether each extension is supported
            CString::new(ext).unwrap()
        }).collect::<Vec<_>>();
        let extensions = extensions.iter().map(|extension| {
            extension.as_ptr()
        }).collect::<Vec<_>>();

        // Creating the Vulkan instance.
        let instance = unsafe {
            let mut output = mem::uninitialized();
            let infos = vk::InstanceCreateInfo {
                sType: vk::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                pApplicationInfo: if let Some(app) = app_infos.as_ref() {
                    app as *const _
                } else {
                    ptr::null()
                },
                enabledLayerCount: layers.len() as u32,
                ppEnabledLayerNames: layers.as_ptr(),
                enabledExtensionCount: extensions.len() as u32,
                ppEnabledExtensionNames: extensions.as_ptr(),
            };

            try!(check_errors(VK_ENTRY.CreateInstance(&infos, ptr::null(), &mut output)));
            output
        };

        // Loading the function pointers of the newly-created instance.
        let vk = vk::InstancePointers::load(|name| unsafe {
            mem::transmute(VK_STATIC.GetInstanceProcAddr(instance, name.as_ptr()))
        });

        // Creating the debug report callback.
        // TODO: should be optional
        let debug_report = unsafe {
            extern "system" fn callback(_: vk::DebugReportFlagsEXT, _: vk::DebugReportObjectTypeEXT,
                                        _: u64, _: usize, _: i32, layer_prefix: *const c_char,
                                        message: *const c_char, _: *mut c_void) -> u32
            {
                unsafe {
                    let message = CStr::from_ptr(message).to_str()
                                                    .expect("debug callback message not utf-8");
                    println!("Debug callback message: {:?}", message);
                    vk::DEBUG_REPORT_ERROR_NONE_EXT
                }
            }

            let infos = vk::DebugReportCallbackCreateInfoEXT {
                sType: vk::STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT,
                pNext: ptr::null(),
                flags: 0,   // reserved
                pfnCallback: callback,
                pUserData: ptr::null_mut(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDebugReportCallbackEXT(instance, &infos,
                                                              ptr::null(), &mut output)));
            output
        };

        // Enumerating all physical devices.
        let physical_devices: Vec<vk::PhysicalDevice> = unsafe {
            let mut num = mem::uninitialized();
            try!(check_errors(vk.EnumeratePhysicalDevices(instance, &mut num, ptr::null_mut())));

            let mut devices = Vec::with_capacity(num as usize);
            try!(check_errors(vk.EnumeratePhysicalDevices(instance, &mut num,
                                                          devices.as_mut_ptr())));
            devices.set_len(num as usize);
            devices
        };

        // Getting the properties of all physical devices.
        let physical_devices = {
            let mut output = Vec::with_capacity(physical_devices.len());

            for device in physical_devices.into_iter() {
                let properties: vk::PhysicalDeviceProperties = unsafe {
                    let mut output = mem::uninitialized();
                    vk.GetPhysicalDeviceProperties(device, &mut output);
                    output
                };

                let queue_families = unsafe {
                    let mut num = mem::uninitialized();
                    vk.GetPhysicalDeviceQueueFamilyProperties(device, &mut num, ptr::null_mut());

                    let mut families = Vec::with_capacity(num as usize);
                    vk.GetPhysicalDeviceQueueFamilyProperties(device, &mut num,
                                                              families.as_mut_ptr());
                    families.set_len(num as usize);
                    families
                };

                let memory: vk::PhysicalDeviceMemoryProperties = unsafe {
                    let mut output = mem::uninitialized();
                    vk.GetPhysicalDeviceMemoryProperties(device, &mut output);
                    output
                };

                let available_features: vk::PhysicalDeviceFeatures = unsafe {
                    let mut output = mem::uninitialized();
                    vk.GetPhysicalDeviceFeatures(device, &mut output);
                    output
                };

                output.push(PhysicalDeviceInfos {
                    device: device,
                    properties: properties,
                    memory: memory,
                    queue_families: queue_families,
                    available_features: Features::from(available_features),
                });
            }
            output
        };

        Ok(Arc::new(Instance {
            instance: instance,
            debug_report: Some(debug_report),
            //alloc: None,
            physical_devices: physical_devices,
            vk: vk,
        }))
    }

    /*/// Same as `new`, but provides an allocator that will be used by the Vulkan library whenever
    /// it needs to allocate memory on the host.
    ///
    /// Note that this allocator can be overriden when you create a `Device`, a `MemoryPool`, etc.
    pub fn with_alloc(app_infos: Option<&ApplicationInfo>, alloc: Box<Alloc + Send + Sync>) -> Arc<Instance> {
        unimplemented!()
    }*/
}

impl fmt::Debug for Instance {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan instance>")
    }
}

impl VulkanObject for Instance {
    type Object = vk::Instance;

    #[inline]
    fn internal_object(&self) -> vk::Instance {
        self.instance
    }
}

impl VulkanPointers for Instance {
    type Pointers = vk::InstancePointers;

    #[inline]
    fn pointers(&self) -> &vk::InstancePointers {
        &self.vk
    }
}

impl Drop for Instance {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_report) = self.debug_report {
                self.vk.DestroyDebugReportCallbackEXT(self.instance, debug_report, ptr::null());
            }

            self.vk.DestroyInstance(self.instance, ptr::null());
        }
    }
}

/// Information that can be given to the Vulkan driver so that it can identify your application.
pub struct ApplicationInfo<'a> {
    /// Name of the application.
    pub application_name: &'a str,
    /// An opaque number that contains the version number of the application.
    pub application_version: u32,
    /// Name of the engine used to power the application.
    pub engine_name: &'a str,
    /// An opaque number that contains the version number of the engine.
    pub engine_version: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum InstanceCreationError {
    OutOfHostMemory = vk::ERROR_OUT_OF_HOST_MEMORY,
    OutOfDeviceMemory = vk::ERROR_OUT_OF_DEVICE_MEMORY,
    InitializationFailed = vk::ERROR_INITIALIZATION_FAILED,
    LayerNotPresent = vk::ERROR_LAYER_NOT_PRESENT,
    ExtensionNotPresent = vk::ERROR_EXTENSION_NOT_PRESENT,
    IncompatibleDriver = vk::ERROR_INCOMPATIBLE_DRIVER,
}

impl error::Error for InstanceCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            InstanceCreationError::OutOfHostMemory => "no memory available on the host",
            InstanceCreationError::OutOfDeviceMemory => "no memory available on the graphical device",
            InstanceCreationError::InitializationFailed => "initialization failed",
            InstanceCreationError::LayerNotPresent => "layer not present",
            InstanceCreationError::ExtensionNotPresent => "extension not present",
            InstanceCreationError::IncompatibleDriver => "incompatible driver",
        }
    }
}

impl fmt::Display for InstanceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for InstanceCreationError {
    #[inline]
    fn from(err: Error) -> InstanceCreationError {
        match err {
            Error::OutOfHostMemory => InstanceCreationError::OutOfHostMemory,
            Error::OutOfDeviceMemory => InstanceCreationError::OutOfDeviceMemory,
            Error::InitializationFailed => InstanceCreationError::InitializationFailed,
            Error::LayerNotPresent => InstanceCreationError::LayerNotPresent,
            Error::ExtensionNotPresent => InstanceCreationError::ExtensionNotPresent,
            Error::IncompatibleDriver => InstanceCreationError::IncompatibleDriver,
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

/// Queries the list of layers that are available when creating an instance.
pub fn layers_list() -> Result<Vec<LayerProperties>, OomError> {
    unsafe {
        let mut num = mem::uninitialized();
        try!(check_errors(VK_ENTRY.EnumerateInstanceLayerProperties(&mut num, ptr::null_mut())));

        let mut layers: Vec<vk::LayerProperties> = Vec::with_capacity(num as usize);
        try!(check_errors(VK_ENTRY.EnumerateInstanceLayerProperties(&mut num, layers.as_mut_ptr())));
        layers.set_len(num as usize);

        Ok(layers.into_iter().map(|layer| {
            LayerProperties { props: layer }
        }).collect())
    }
}

/// Properties of an available layer.
pub struct LayerProperties {
    props: vk::LayerProperties,
}

impl LayerProperties {
    /// Returns the name of the layer.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.props.layerName.as_ptr()).to_str().unwrap() }
    }

    /// Returns a description of the layer.
    #[inline]
    pub fn description(&self) -> &str {
        unsafe { CStr::from_ptr(self.props.description.as_ptr()).to_str().unwrap() }
    }

    /// Returns the version of Vulkan supported by this layer.
    #[inline]
    pub fn vulkan_version(&self) -> Version {
        Version::from_vulkan_version(self.props.specVersion)
    }

    /// Returns an implementation-specific version number for this layer.
    #[inline]
    pub fn implementation_version(&self) -> u32 {
        self.props.implementationVersion
    }
}

struct PhysicalDeviceInfos {
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    queue_families: Vec<vk::QueueFamilyProperties>,
    memory: vk::PhysicalDeviceMemoryProperties,
    available_features: Features,
}

/// Represents one of the available devices on this machine.
#[derive(Debug, Clone)]
pub struct PhysicalDevice {
    instance: Arc<Instance>,
    device: usize,
}

impl PhysicalDevice {
    /// Returns an iterator that enumerates the physical devices available.
    #[inline]
    pub fn enumerate(instance: &Arc<Instance>) -> PhysicalDevicesIter {
        PhysicalDevicesIter {
            instance: instance,
            current_id: 0,
        }
    }

    /// Returns the instance corresponding to this physical device.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the human-readable name of the device.
    #[inline]
    pub fn name(&self) -> String {  // FIXME: for some reason this panicks if you use a `&str`
        unsafe {
            let val = self.infos().properties.deviceName;
            let val = CStr::from_ptr(val.as_ptr());
            val.to_str().expect("physical device name contained non-UTF8 characters").to_owned()
        }
    }

    /// Returns the type of the device.
    #[inline]
    pub fn ty(&self) -> PhysicalDeviceType {
        match self.instance.physical_devices[self.device].properties.deviceType {
            vk::PHYSICAL_DEVICE_TYPE_OTHER => PhysicalDeviceType::Other,
            vk::PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => PhysicalDeviceType::IntegratedGpu,
            vk::PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => PhysicalDeviceType::DiscreteGpu,
            vk::PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => PhysicalDeviceType::VirtualGpu,
            vk::PHYSICAL_DEVICE_TYPE_CPU => PhysicalDeviceType::Cpu,
            _ => panic!("Unrecognized Vulkan device type")
        }
    }

    /// Returns the version of Vulkan supported by this device.
    #[inline]
    pub fn api_version(&self) -> Version {
        let val = self.infos().properties.apiVersion;
        Version::from_vulkan_version(val)
    }

    /// Returns the Vulkan features that are supported by this physical device.
    #[inline]
    pub fn supported_features(&self) -> &Features {
        &self.infos().available_features
    }

    /// Builds an iterator that enumerates all the queue families on this physical device.
    #[inline]
    pub fn queue_families(&self) -> QueueFamiliesIter {
        QueueFamiliesIter {
            physical_device: self,
            current_id: 0,
        }
    }

    /// Returns the queue family with the given index, or `None` if out of range.
    #[inline]
    pub fn queue_family_by_id(&self, id: u32) -> Option<QueueFamily> {
        if (id as usize) < self.infos().queue_families.len() {
            Some(QueueFamily {
                physical_device: self,
                id: id,
            })

        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory types on this physical device.
    #[inline]
    pub fn memory_types(&self) -> MemoryTypesIter {
        MemoryTypesIter {
            physical_device: self,
            current_id: 0,
        }
    }

    /// Returns the memory type with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_type_by_id(&self, id: u32) -> Option<MemoryType> {
        if id < self.infos().memory.memoryTypeCount {
            Some(MemoryType {
                physical_device: self,
                id: id,
            })

        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory heaps on this physical device.
    #[inline]
    pub fn memory_heaps(&self) -> MemoryHeapsIter {
        MemoryHeapsIter {
            physical_device: self,
            current_id: 0,
        }
    }

    /// Returns the memory heap with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_heap_by_id(&self, id: u32) -> Option<MemoryHeap> {
        if id < self.infos().memory.memoryHeapCount {
            Some(MemoryHeap {
                physical_device: self,
                id: id,
            })

        } else {
            None
        }
    }

    /// Returns an opaque number representing the version of the driver of this device.
    #[inline]
    pub fn driver_version(&self) -> u32 {
        self.infos().properties.driverVersion
    }

    /// Returns the PCI ID of the device.
    #[inline]
    pub fn pci_device_id(&self) -> u32 {
        self.infos().properties.deviceID
    }

    /// Returns the PCI ID of the vendor.
    #[inline]
    pub fn pci_vendor_id(&self) -> u32 {
        self.infos().properties.vendorID
    }

    /// Returns a unique identifier for the device.
    #[inline]
    pub fn uuid(&self) -> &[u8; 16] {   // must be equal to vk::UUID_SIZE
        &self.infos().properties.pipelineCacheUUID
    }

    /// Internal function to make it easier to get the infos of this device.
    #[inline]
    fn infos(&self) -> &PhysicalDeviceInfos {
        &self.instance.physical_devices[self.device]
    }
}

impl VulkanObject for PhysicalDevice {
    type Object = vk::PhysicalDevice;

    #[inline]
    fn internal_object(&self) -> vk::PhysicalDevice {
        self.infos().device
    }
}

/// Iterator for all the physical devices available on hardware.
#[derive(Debug, Clone)]
pub struct PhysicalDevicesIter<'a> {
    instance: &'a Arc<Instance>,
    current_id: usize,
}

impl<'a> Iterator for PhysicalDevicesIter<'a> {
    type Item = PhysicalDevice;

    #[inline]
    fn next(&mut self) -> Option<PhysicalDevice> {
        if self.current_id >= self.instance.physical_devices.len() {
            return None;
        }

        let dev = PhysicalDevice {
            instance: self.instance.clone(),
            device: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }
}

/// Type of a physical device.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PhysicalDeviceType {
    /// The device is an integrated GPU.
    IntegratedGpu = 1,
    /// The device is a discrete GPU.
    DiscreteGpu = 2,
    /// The device is a virtual GPU.
    VirtualGpu = 3,
    /// The device is a CPU.
    Cpu = 4,
    /// The device is something else.
    Other = 0,
}

/// Represents a queue family in a physical device.
///
/// A queue family is group of one or multiple queues. All queues of one family have the same
/// characteristics.
#[derive(Debug, Copy, Clone)]
pub struct QueueFamily<'a> {
    physical_device: &'a PhysicalDevice,
    id: u32,
}

impl<'a> QueueFamily<'a> {
    /// Returns the physical device associated to this queue family.
    #[inline]
    pub fn physical_device(&self) -> &'a PhysicalDevice {
        self.physical_device
    }

    /// Returns the identifier of this queue family within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the number of queues that belong to this family.
    ///
    /// Guaranteed to be at least 1 (or else that family wouldn't exist).
    #[inline]
    pub fn queues_count(&self) -> usize {
        self.physical_device.infos().queue_families[self.id as usize].queueCount as usize
    }

    /// Returns true if queues of this family can execute graphics operations.
    #[inline]
    pub fn supports_graphics(&self) -> bool {
        (self.flags() & vk::QUEUE_GRAPHICS_BIT) != 0
    }

    /// Returns true if queues of this family can execute compute operations.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        (self.flags() & vk::QUEUE_COMPUTE_BIT) != 0
    }

    /// Returns true if queues of this family can execute transfer operations.
    #[inline]
    pub fn supports_transfers(&self) -> bool {
        (self.flags() & vk::QUEUE_TRANSFER_BIT) != 0
    }

    /// Returns true if queues of this family can execute sparse resources binding operations.
    #[inline]
    pub fn supports_sparse_binding(&self) -> bool {
        (self.flags() & vk::QUEUE_SPARSE_BINDING_BIT) != 0
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> u32 {
        self.physical_device.infos().queue_families[self.id as usize].queueFlags
    }
}

/// Iterator for all the queue families available on a physical device.
#[derive(Debug, Clone)]
pub struct QueueFamiliesIter<'a> {
    physical_device: &'a PhysicalDevice,
    current_id: u32,
}

impl<'a> Iterator for QueueFamiliesIter<'a> {
    type Item = QueueFamily<'a>;

    #[inline]
    fn next(&mut self) -> Option<QueueFamily<'a>> {
        if self.current_id as usize >= self.physical_device.infos().queue_families.len() {
            return None;
        }

        let dev = QueueFamily {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().queue_families.len();
        let remain = len - self.current_id as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for QueueFamiliesIter<'a> {}

/// Represents a memory type in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryType<'a> {
    physical_device: &'a PhysicalDevice,
    id: u32,
}

impl<'a> MemoryType<'a> {
    /// Returns the physical device associated to this memory type.
    #[inline]
    pub fn physical_device(&self) -> &'a PhysicalDevice {
        self.physical_device
    }

    /// Returns the identifier of this memory type within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the heap that corresponds to this memory type.
    #[inline]
    pub fn heap(&self) -> MemoryHeap<'a> {
        let heap_id = self.physical_device.infos().memory.memoryTypes[self.id as usize].heapIndex;
        MemoryHeap { physical_device: self.physical_device, id: heap_id }
    }

    /// Returns true if the memory type is located on the device, which means that it's the most
    /// efficient for GPU accesses.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
    }

    /// Returns true if the memory type can be accessed by the host.
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0
    }

    /// Returns true if modifications made by the host or the GPU on this memory type are
    /// instantaneously visible to the other party. False means that changes have to be flushed.
    ///
    /// You don't need to worry about this, as this library handles that for you.
    #[inline]
    pub fn is_host_coherent(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0
    }

    /// Returns true if memory of this memory type is cached by the host. Host memory accesses to
    /// cached memory is faster than for uncached memory. However you are not guaranteed that it
    /// is coherent.
    #[inline]
    pub fn is_host_cached(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_CACHED_BIT) != 0
    }

    /// Returns true if allocations made to this memory type is lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation.
    ///
    /// Memory of this type can only be used on images created with a certain flag. Memory of this
    /// type is never host-visible.
    #[inline]
    pub fn is_lazily_allocated(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> u32 {
        self.physical_device.infos().memory.memoryTypes[self.id as usize].propertyFlags
    }
}

/// Iterator for all the memory types available on a physical device.
#[derive(Debug, Clone)]
pub struct MemoryTypesIter<'a> {
    physical_device: &'a PhysicalDevice,
    current_id: u32,
}

impl<'a> Iterator for MemoryTypesIter<'a> {
    type Item = MemoryType<'a>;

    #[inline]
    fn next(&mut self) -> Option<MemoryType<'a>> {
        if self.current_id >= self.physical_device.infos().memory.memoryTypeCount {
            return None;
        }

        let dev = MemoryType {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().memory.memoryTypeCount;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryTypesIter<'a> {}

/// Represents a memory heap in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryHeap<'a> {
    physical_device: &'a PhysicalDevice,
    id: u32,
}

impl<'a> MemoryHeap<'a> {
    /// Returns the physical device associated to this memory heap.
    #[inline]
    pub fn physical_device(&self) -> &'a PhysicalDevice {
        self.physical_device
    }

    /// Returns the identifier of this memory heap within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the size in bytes on this heap.
    #[inline]
    pub fn size(&self) -> usize {
        self.physical_device.infos().memory.memoryHeaps[self.id as usize].size as usize
    }

    /// Returns true if the heap is local to the GPU.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        let flags = self.physical_device.infos().memory.memoryHeaps[self.id as usize].flags;
        (flags & vk::MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
    }
}

/// Iterator for all the memory heaps available on a physical device.
#[derive(Debug, Clone)]
pub struct MemoryHeapsIter<'a> {
    physical_device: &'a PhysicalDevice,
    current_id: u32,
}

impl<'a> Iterator for MemoryHeapsIter<'a> {
    type Item = MemoryHeap<'a>;

    #[inline]
    fn next(&mut self) -> Option<MemoryHeap<'a>> {
        if self.current_id >= self.physical_device.infos().memory.memoryHeapCount {
            return None;
        }

        let dev = MemoryHeap {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().memory.memoryHeapCount;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryHeapsIter<'a> {}

#[cfg(test)]
mod tests {
    use instance;

    #[test]
    fn create_instance() {
        let _ = instance::Instance::new(None, None);
    }

    #[test]
    fn layers_list() {
        let _ = instance::layers_list();
    }

    #[test]
    fn queue_family_by_id() {
        let instance = match instance::Instance::new(None, None) {
            Ok(i) => i, Err(_) => return
        };

        let phys = match instance::PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return
        };

        let queue_family = match phys.queue_families().next() {
            Some(q) => q,
            None => return
        };

        let by_id = phys.queue_family_by_id(queue_family.id()).unwrap();
        assert_eq!(by_id.id(), queue_family.id());
    }
}
