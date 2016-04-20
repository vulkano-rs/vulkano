// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


use std::ffi::{ CString, CStr };
use std::os::raw::c_char;
use std::collections::HashSet;
use std::hash::{ Hash, Hasher };
use std::borrow::Borrow;
use std::ptr;
use std::sync::Arc;

use instance::Instance;
use vk;
use VK_ENTRY;
use check_errors;


pub struct InstanceExtensionSetBuilder {
    supported_layers: HashSet<InstanceLayerInfo>,
    supported_core_extensions: HashSet<InstanceExtensionInfo>,
    supported_extensions: HashSet<InstanceExtensionInfo>,
    loaded_layers: HashSet<InstanceLayerInfo>,
    loaded_extensions: HashSet<InstanceExtensionInfo>,
}

impl InstanceExtensionSetBuilder {
    pub fn new() -> InstanceExtensionSetBuilder {
        let supported_layers = unsafe {
            let mut num = 0;
            check_errors(VK_ENTRY.EnumerateInstanceLayerProperties(
                &mut num, ptr::null_mut())).unwrap();

            let mut properties: Vec<vk::LayerProperties> = Vec::with_capacity(num as usize);
            check_errors(VK_ENTRY.EnumerateInstanceLayerProperties(
                &mut num, properties.as_mut_ptr())).unwrap();
            properties.set_len(num as usize);
            
            properties.iter().map(|p| InstanceLayerInfo::new(
                CStr::from_ptr(p.layerName.as_ptr()).to_string_lossy().into_owned(),
                CStr::from_ptr(p.description.as_ptr()).to_string_lossy().into_owned()
            )).collect()
        };
        
        let supported_core_extensions = unsafe {
            supported_extensions_raw(ptr::null())
        };
        let supported_extensions = supported_core_extensions.clone();
        
        InstanceExtensionSetBuilder {
            supported_layers: supported_layers,
            supported_core_extensions: supported_core_extensions,
            supported_extensions: supported_extensions,
            loaded_layers: HashSet::new(),
            loaded_extensions: HashSet::new(),
        }
    }
    
    pub fn supported_layers(&self) -> HashSet<InstanceLayerInfo> {
        self.supported_layers.clone()
    }
    
    pub fn supported_core_extensions(&self) -> HashSet<InstanceExtensionInfo> {
        self.supported_core_extensions.clone()
    }
    
    pub fn supported_extensions(&self) -> HashSet<InstanceExtensionInfo> {
        self.supported_extensions.clone()
    }
    
    pub fn loaded_layers(&self) -> HashSet<InstanceLayerInfo> {
        self.loaded_layers.clone()
    }
    
    pub fn loaded_extensions(&self) -> HashSet<InstanceExtensionInfo> {
        self.loaded_extensions.clone()
    }
    
    pub fn supported_layer_info(&self, layer: &InstanceLayerDescriptor) -> Option<InstanceLayerInfo> {
        self.supported_layers.get(layer.name()).and_then(|v| Some(v.clone()))
    }
    
    pub fn supported_extension_info(&self, extension: &InstanceExtensionDescriptor) -> Option<InstanceExtensionInfo> {
        self.supported_extensions.get(extension.name()).and_then(|v| Some(v.clone()))
    }
    
    pub fn add_layer(mut self, layer: &InstanceLayerDescriptor) -> Self {
        let layer_info = self.supported_layer_info(layer)
            .expect("Tried to add unsupported instance layer!");
        self.loaded_layers.insert(layer_info.clone());
        
        for extension in layer_info.supported_extensions() {
            self.supported_extensions.insert(extension);
        }
        self
    }
    
    pub fn add_extension(mut self, extension: &InstanceExtensionDescriptor) -> Self {
        let extension_info = self.supported_extension_info(extension)
            .expect("Tried to add unsupported instance extension!");
        self.loaded_extensions.insert(extension_info.clone());
        
        for dependency in extension.dependencies() {
            self = self.add_extension(&dependency);
        }
        self
    }
    
    pub fn build(&self) -> InstanceExtensionSet {
        InstanceExtensionSet::new(
            self.loaded_layers.iter().cloned().collect(),
            self.loaded_extensions.iter().cloned().collect())
    }
}


// An instance of this is passed to Instance::new()
pub struct InstanceExtensionSet {
    layers: HashSet<InstanceLayerInfo>,
    extensions: HashSet<InstanceExtensionInfo>,
}

impl InstanceExtensionSet {
    fn new(layers: HashSet<InstanceLayerInfo>, extensions: HashSet<InstanceExtensionInfo>)
        -> InstanceExtensionSet
    {
        InstanceExtensionSet {
            layers: layers,
            extensions: extensions,
        }
    }
    
    pub fn layers(&self) -> HashSet<InstanceLayerInfo> {
        self.layers.clone()
    }
    
    pub fn extensions(&self) -> HashSet<InstanceExtensionInfo> {
        self.extensions.clone()
    }
}


// Retrieved from instance.extensions()
pub struct InstanceExtensions {
    instance: Arc<Instance>,
    layers: HashSet<InstanceLayerInfo>,
    extensions: HashSet<InstanceExtensionInfo>,
}

impl InstanceExtensions {
    // make this pub(instance::instance) once pub(restricted) syntax is stable
    pub fn new(instance: Arc<Instance>, extension_set: InstanceExtensionSet)
        -> InstanceExtensions
    {
        InstanceExtensions {
            instance: instance,
            layers: extension_set.layers(),
            extensions: extension_set.extensions(),
        }
    }
    
    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
    
    pub fn layers(&self) -> HashSet<InstanceLayerInfo> {
        self.layers.clone()
    }
    
    pub fn extensions(&self) -> HashSet<InstanceExtensionInfo> {
        self.extensions.clone()
    }
    
    pub fn is_supported(&self, extension: InstanceExtensionDescriptor) -> bool {
        self.extensions.contains(extension.name())
    }
}


#[derive(Clone)]
pub struct InstanceLayerInfo {
    // TODO: Versions?
    name: String,
    description: String,
}

impl InstanceLayerInfo {
    fn new(name: String, description: String) -> InstanceLayerInfo {
        InstanceLayerInfo {
            name: name,
            description: description,
        }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
    
    pub fn description(&self) -> &str {
        &self.description
    }
}

impl InstanceLayerInfo {
    pub fn supported_extensions(&self) -> HashSet<InstanceExtensionInfo> {
        unsafe {
            let layer_name = CString::new(self.name.clone()).unwrap();
            supported_extensions_raw(layer_name.as_ptr())
        }
    }
}

impl PartialEq for InstanceLayerInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl PartialEq<InstanceLayerDescriptor> for InstanceLayerInfo {
    fn eq(&self, other: &InstanceLayerDescriptor) -> bool {
        self.name == other.name
    }
}

impl Eq for InstanceLayerInfo {}

impl Hash for InstanceLayerInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl Borrow<str> for InstanceLayerInfo {
    fn borrow(&self) -> &str {
        &self.name
    }
}


#[derive(Clone)]
pub struct InstanceLayerDescriptor {
    name: String,
}

impl InstanceLayerDescriptor {
    pub fn new(name: String) -> InstanceLayerDescriptor {
        InstanceLayerDescriptor {
            name: name,
        }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
}


#[derive(Clone)]
pub struct InstanceExtensionInfo {
    name: String,
    // TODO: version
}

impl InstanceExtensionInfo {
    fn new(name: String) -> InstanceExtensionInfo {
        InstanceExtensionInfo {
            name: name,
        }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl PartialEq for InstanceExtensionInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for InstanceExtensionInfo {}

impl Hash for InstanceExtensionInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl Borrow<str> for InstanceExtensionInfo {
    fn borrow(&self) -> &str {
        &self.name
    }
}


#[derive(Clone)]
pub struct InstanceExtensionDescriptor {
    name: String,
    dependencies: Vec<InstanceExtensionDescriptor>,
}

impl InstanceExtensionDescriptor {
    pub fn new(name: String, dependencies: Vec<InstanceExtensionDescriptor>)
               -> InstanceExtensionDescriptor
    {
        InstanceExtensionDescriptor {
            name: name,
            dependencies: dependencies,
        }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
    
    pub fn dependencies(&self) -> Vec<InstanceExtensionDescriptor> {
        // TODO: return iterator instead
        self.dependencies.clone()
    }
}


pub trait InstanceExtension {
    fn descriptor() -> InstanceExtensionDescriptor;
    
    fn new(extensions: Arc<InstanceExtensions>) -> Arc<Self>;
    
    fn instance(&self) -> Arc<Instance>;
}


#[macro_export]
macro_rules! instance_extension_descriptor {
    ( $ext_name:expr, [$($dep:ident),*] ) => {
        fn descriptor() -> InstanceExtensionDescriptor {
            InstanceExtensionDescriptor::new(
                $ext_name.to_string(),
                vec![$( $dep::descriptor() ),*]
            )
        }
    };
}


unsafe fn supported_extensions_raw(layer_name: *const c_char)
    -> HashSet<InstanceExtensionInfo>
{
    let mut num = 0;
    check_errors(VK_ENTRY.EnumerateInstanceExtensionProperties(
        layer_name, &mut num, ptr::null_mut())).unwrap();

    let mut properties: Vec<vk::ExtensionProperties> = Vec::with_capacity(num as usize);
    check_errors(VK_ENTRY.EnumerateInstanceExtensionProperties(
        layer_name, &mut num, properties.as_mut_ptr())).unwrap();
    properties.set_len(num as usize);
    
    properties.iter().map(|p| InstanceExtensionInfo {
        name: CStr::from_ptr(p.extensionName.as_ptr()).to_string_lossy().into_owned(),
    }).collect()
}