// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::borrow::Cow;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::ffi::CStr;

use pipeline::input_assembly::PrimitiveTopology;

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

    pub unsafe fn vertex_shader_entry_point<'a, S, V, L>(&'a self, name: &'a CStr, layout: L,
                                                         attributes: Vec<(u32, Cow<'static, str>)>)
                                                         -> VertexShaderEntryPoint<'a, S, V, L>
    {
        VertexShaderEntryPoint {
            module: self,
            name: name,
            layout: layout,
            attributes: attributes,
            marker: PhantomData,
        }
    }

    pub unsafe fn tess_control_shader_entry_point<'a, S, I, O, L>(&'a self, name: &'a CStr, layout: L)
                                                              -> TessControlShaderEntryPoint<'a, S, I, O, L>
    {
        TessControlShaderEntryPoint {
            module: self,
            name: name,
            layout: layout,
            marker: PhantomData,
        }
    }

    pub unsafe fn tess_evaluation_shader_entry_point<'a, S, I, O, L>(&'a self, name: &'a CStr, layout: L)
                                                                     -> TessEvaluationShaderEntryPoint<'a, S, I, O, L>
    {
        TessEvaluationShaderEntryPoint {
            module: self,
            name: name,
            layout: layout,
            marker: PhantomData,
        }
    }

    pub unsafe fn geometry_shader_entry_point<'a, S, I, O, L>(&'a self, name: &'a CStr, primitives: GeometryShaderExecutionMode, layout: L)
                                                              -> GeometryShaderEntryPoint<'a, S, I, O, L>
    {
        GeometryShaderEntryPoint {
            module: self,
            name: name,
            layout: layout,
            primitives: primitives,
            marker: PhantomData,
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
    pub unsafe fn fragment_shader_entry_point<'a, S, F, L>(&'a self, name: &'a CStr, layout: L)
                                                           -> FragmentShaderEntryPoint<'a, S, F, L>
    {
        FragmentShaderEntryPoint {
            module: self,
            name: name,
            layout: layout,
            marker: PhantomData,
        }
    }

    #[inline]
    pub unsafe fn compute_shader_entry_point<'a, S, L>(&'a self, name: &'a CStr, layout: L)
                                                       -> ComputeShaderEntryPoint<'a, S, L>
    {
        ComputeShaderEntryPoint {
            module: self,
            name: name,
            layout: layout,
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

pub struct VertexShaderEntryPoint<'a, S, V, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    attributes: Vec<(u32, Cow<'static, str>)>,
    layout: L,
    marker: PhantomData<(S, V)>,
}

impl<'a, S, V, L> VertexShaderEntryPoint<'a, S, V, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }

    // TODO: change API
    #[inline]
    pub fn attributes(&self) -> &[(u32, Cow<'static, str>)] {
        &self.attributes
    }
}

pub struct TessControlShaderEntryPoint<'a, S, I, O, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    layout: L,
    marker: PhantomData<(S, I, O)>,
}

impl<'a, S, I, O, L> TessControlShaderEntryPoint<'a, S, I, O, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

pub struct TessEvaluationShaderEntryPoint<'a, S, I, O, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    layout: L,
    marker: PhantomData<(S, I, O)>,
}

impl<'a, S, I, O, L> TessEvaluationShaderEntryPoint<'a, S, I, O, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

pub struct GeometryShaderEntryPoint<'a, S, I, O, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    layout: L,
    primitives: GeometryShaderExecutionMode,
    marker: PhantomData<(S, I, O)>,
}

impl<'a, S, I, O, L> GeometryShaderEntryPoint<'a, S, I, O, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    /// Returns the kind of primitives expected by the geometry shader.
    #[inline]
    pub fn primitives(&self) -> GeometryShaderExecutionMode {
        self.primitives
    }

    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

/// Declares which type of primitives are expected by the geometry shader.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GeometryShaderExecutionMode {
    Points,
    Lines,
    LinesWithAdjacency,
    Triangles,
    TrianglesWithAdjacency,
}

impl GeometryShaderExecutionMode {
    /// Returns true if the given primitive topology can be used with this execution mode.
    #[inline]
    pub fn matches(&self, input: PrimitiveTopology) -> bool {
        match (*self, input) {
            (GeometryShaderExecutionMode::Points, PrimitiveTopology::PointList) => true,
            (GeometryShaderExecutionMode::Lines, PrimitiveTopology::LineList) => true,
            (GeometryShaderExecutionMode::Lines, PrimitiveTopology::LineStrip) => true,
            (GeometryShaderExecutionMode::LinesWithAdjacency,
             PrimitiveTopology::LineListWithAdjacency) => true,
            (GeometryShaderExecutionMode::LinesWithAdjacency,
             PrimitiveTopology::LineStripWithAdjacency) => true,
            (GeometryShaderExecutionMode::Triangles, PrimitiveTopology::TriangleList) => true,
            (GeometryShaderExecutionMode::Triangles, PrimitiveTopology::TriangleStrip) => true,
            (GeometryShaderExecutionMode::Triangles, PrimitiveTopology::TriangleFan) => true,
            (GeometryShaderExecutionMode::TrianglesWithAdjacency,
             PrimitiveTopology::TriangleListWithAdjancecy) => true,
            (GeometryShaderExecutionMode::TrianglesWithAdjacency,
             PrimitiveTopology::TriangleStripWithAdjacency) => true,
            _ => false,
        }
    }
}

pub struct FragmentShaderEntryPoint<'a, S, F, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    layout: L,
    marker: PhantomData<(S, F)>,
}

impl<'a, S, F, L> FragmentShaderEntryPoint<'a, S, F, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

pub struct ComputeShaderEntryPoint<'a, S, L> {
    module: &'a ShaderModule,
    name: &'a CStr,
    layout: L,
    marker: PhantomData<S>,
}

impl<'a, S, L> ComputeShaderEntryPoint<'a, S, L> {
    #[inline]
    pub fn module(&self) -> &'a ShaderModule {
        self.module
    }

    #[inline]
    pub fn name(&self) -> &'a CStr {
        self.name
    }

    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

pub unsafe trait ShaderInterfaceDef {
}

pub unsafe trait PossibleMatchShaderInterface<I>: ShaderInterfaceDef where I: ShaderInterfaceDef {
    fn matches(&self, other: &I) -> bool;
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

/// Describes an indiviual constant to set in the shader. Also a field in the struct.
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
