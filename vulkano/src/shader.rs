use std::borrow::Cow;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::ffi::CStr;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Contains SPIR-V code with one or more entry points.
///
/// Note that it is advised to wrap around a `ShaderModule` with a struct that is different for
/// each shader.
pub struct ShaderModule {
    device: Arc<Device>,
    module: vk::ShaderModule,
}

impl ShaderModule {
    /// Builds a new shader module from SPIR-V.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated.
    /// - The SPIR-V code may require some features that are not enabled. This isn't checked by
    ///   this function either.
    ///
    pub unsafe fn new(device: &Arc<Device>, spirv: &[u8])
                      -> Result<Arc<ShaderModule>, OomError>
    {
        let vk = device.pointers();

        assert!((spirv.len() % 4) == 0);

        let module = {
            let infos = vk::ShaderModuleCreateInfo {
                sType: vk::STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                codeSize: spirv.len(),
                pCode: spirv.as_ptr() as *const _,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateShaderModule(device.internal_object(), &infos,
                                                    ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(ShaderModule {
            device: device.clone(),
            module: module,
        }))
    }

    pub unsafe fn vertex_shader_entry_point<'a, V, L>(&'a self, name: &'a CStr,
                                                      attributes: Vec<(u32, Cow<'static, str>)>)
                                                      -> VertexShaderEntryPoint<'a, V, L>
    {
        VertexShaderEntryPoint {
            module: self,
            name: name,
            marker: PhantomData,
            attributes: attributes,
        }
    }

    /// Gets access to an entry point contained in this module.
    ///
    /// This is purely a *logical* operation. It returns a struct that *represents* the entry
    /// point but doesn't actually do anything.
    ///
    /// # Safety
    ///
    /// - The user must check that the entry point exists in the module, as this is not checked
    ///   by Vulkan.
    /// - Calling this function also determines the template parameters associated to the
    ///   `EntryPoint` struct. Therefore care must be taken that the values there are correct.
    ///
    pub unsafe fn fragment_shader_entry_point<'a, F>(&'a self, name: &'a CStr)
                                                    -> FragmentShaderEntryPoint<'a, F>
    {
        FragmentShaderEntryPoint {
            module: self,
            name: name,
            marker: PhantomData,
        }
    }
}

unsafe impl VulkanObject for ShaderModule {
    type Object = vk::ShaderModule;

    #[inline]
    fn internal_object(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for ShaderModule {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyShaderModule(self.device.internal_object(), self.module, ptr::null());
        }
    }
}

pub struct VertexShaderEntryPoint<'a, V, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    marker: PhantomData<(V, L)>,
    attributes: Vec<(u32, Cow<'static, str>)>,
}

impl<'a, V, L> VertexShaderEntryPoint<'a, V, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    // TODO: change API
    #[inline]
    pub fn attributes(&self) -> &[(u32, Cow<'static, str>)] {
        &self.attributes
    }
}

pub struct ComputeShaderEntryPoint<'a, D, S, P> {
    module: &'a ShaderModule,
    name: &'a CStr,
    marker: PhantomData<(D, S, P)>
}

pub struct FragmentShaderEntryPoint<'a, F> {
    module: &'a ShaderModule,
    name: &'a CStr,
    marker: PhantomData<F>
}

impl<'a, F> FragmentShaderEntryPoint<'a, F> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }
}

/// Trait to describe structs that contain specialization data for shaders.
///
/// It is implemented on `()` for shaders that don't have any specialization constant.
pub unsafe trait SpecializationConstants {
    /// Returns descriptors of the struct's layout.
    fn descriptors() -> &'static [SpecializationMapEntry];
}

unsafe impl SpecializationConstants for () {
    #[inline]
    fn descriptors() -> &'static [SpecializationMapEntry] {
        &[]
    }
}

/// Describes an invidiual constant to set in the shader. Also a field in the struct.
// Has the same memory representation as a `VkSpecializationMapEntry`.
#[repr(C)]
pub struct SpecializationMapEntry {
    /// Identifier of the constant in the shader that corresponds to this field.
    pub constant_id: u32,
    /// Offset within this struct for the data.
    pub offset: u32,
    /// Size of the data in bytes.
    pub size: usize,
}

/// Trait to describe structs that contain push constants for shaders.
///
/// It is implemented on `()` for shaders that don't have any push constant.
pub unsafe trait PushConstants {
    // TODO: 
}

unsafe impl PushConstants for () {
}
