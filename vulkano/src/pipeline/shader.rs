// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Stage of a graphics pipeline.
//!
//! In Vulkan, shaders are grouped in *shader modules*. Each shader module is built from SPIR-V
//! code and can contain one or more entry points. Note that for the moment the official
//! GLSL-to-SPIR-V compiler does not support multiple entry points.
//!
//! The vulkano library does not provide any functionality that checks and introspects the SPIR-V
//! code, therefore the whole shader-related API is unsafe. You are encouraged to use the
//! `vulkano-shaders` crate that will generate Rust code that wraps around vulkano's shaders API.

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::device::Device;
use crate::format::Format;
use crate::pipeline::input_assembly::PrimitiveTopology;
use crate::pipeline::layout::PipelineLayoutPcRange;
use crate::sync::PipelineStages;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::error;
use std::ffi::CStr;
use std::fmt;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::BitOr;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

/// Contains SPIR-V code with one or more entry points.
///
/// Note that it is advised to wrap around a `ShaderModule` with a struct that is different for
/// each shader.
#[derive(Debug)]
pub struct ShaderModule {
    // The module.
    module: ash::vk::ShaderModule,
    // Pointer to the device.
    device: Arc<Device>,
}

impl ShaderModule {
    /// Builds a new shader module from SPIR-V bytes.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated.
    /// - The SPIR-V code may require some features that are not enabled. This isn't checked by
    ///   this function either.
    ///
    pub unsafe fn new(device: Arc<Device>, spirv: &[u8]) -> Result<Arc<ShaderModule>, OomError> {
        debug_assert!((spirv.len() % 4) == 0);
        Self::from_ptr(device, spirv.as_ptr() as *const _, spirv.len())
    }

    /// Builds a new shader module from SPIR-V 32-bit words.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated.
    /// - The SPIR-V code may require some features that are not enabled. This isn't checked by
    ///   this function either.
    ///
    pub unsafe fn from_words(
        device: Arc<Device>,
        spirv: &[u32],
    ) -> Result<Arc<ShaderModule>, OomError> {
        Self::from_ptr(device, spirv.as_ptr(), spirv.len() * mem::size_of::<u32>())
    }

    /// Builds a new shader module from SPIR-V.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated.
    /// - The SPIR-V code may require some features that are not enabled. This isn't checked by
    ///   this function either.
    ///
    unsafe fn from_ptr(
        device: Arc<Device>,
        spirv: *const u32,
        spirv_len: usize,
    ) -> Result<Arc<ShaderModule>, OomError> {
        let module = {
            let infos = ash::vk::ShaderModuleCreateInfo {
                flags: ash::vk::ShaderModuleCreateFlags::empty(),
                code_size: spirv_len,
                p_code: spirv,
                ..Default::default()
            };

            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_shader_module(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(ShaderModule {
            module: module,
            device: device,
        }))
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
    /// - The input, output and layout must correctly describe the input, output and layout used
    ///   by this stage.
    ///
    pub unsafe fn graphics_entry_point<'a, D>(
        &'a self,
        name: &'a CStr,
        descriptor_set_layout_descs: D,
        push_constant_range: Option<PipelineLayoutPcRange>,
        spec_constants: &'static [SpecializationMapEntry],
        input: ShaderInterface,
        output: ShaderInterface,
        ty: GraphicsShaderType,
    ) -> GraphicsEntryPoint<'a>
    where
        D: IntoIterator<Item = DescriptorSetDesc>,
    {
        GraphicsEntryPoint {
            module: self,
            name,
            descriptor_set_layout_descs: descriptor_set_layout_descs.into_iter().collect(),
            push_constant_range,
            spec_constants,
            input,
            output,
            ty,
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
    /// - The layout must correctly describe the layout used by this stage.
    ///
    #[inline]
    pub unsafe fn compute_entry_point<'a, D>(
        &'a self,
        name: &'a CStr,
        descriptor_set_layout_descs: D,
        push_constant_range: Option<PipelineLayoutPcRange>,
        spec_constants: &'static [SpecializationMapEntry],
    ) -> ComputeEntryPoint<'a>
    where
        D: IntoIterator<Item = DescriptorSetDesc>,
    {
        ComputeEntryPoint {
            module: self,
            name,
            descriptor_set_layout_descs: descriptor_set_layout_descs.into_iter().collect(),
            push_constant_range,
            spec_constants,
        }
    }
}

unsafe impl VulkanObject for ShaderModule {
    type Object = ash::vk::ShaderModule;

    #[inline]
    fn internal_object(&self) -> ash::vk::ShaderModule {
        self.module
    }
}

impl Drop for ShaderModule {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_shader_module(self.device.internal_object(), self.module, ptr::null());
        }
    }
}

pub unsafe trait EntryPointAbstract {
    /// Returns the module this entry point comes from.
    fn module(&self) -> &ShaderModule;

    /// Returns the name of the entry point.
    fn name(&self) -> &CStr;

    /// Returns a description of the descriptor set layouts.
    fn descriptor_set_layout_descs(&self) -> &[DescriptorSetDesc];

    /// Returns the push constant ranges.
    fn push_constant_range(&self) -> &Option<PipelineLayoutPcRange>;

    /// Returns the layout of the specialization constants.
    fn spec_constants(&self) -> &[SpecializationMapEntry];
}

/// Represents a shader entry point in a shader module.
///
/// Can be obtained by calling `entry_point()` on the shader module.
#[derive(Clone, Debug)]
pub struct GraphicsEntryPoint<'a> {
    module: &'a ShaderModule,
    name: &'a CStr,

    descriptor_set_layout_descs: SmallVec<[DescriptorSetDesc; 16]>,
    push_constant_range: Option<PipelineLayoutPcRange>,
    spec_constants: &'static [SpecializationMapEntry],
    input: ShaderInterface,
    output: ShaderInterface,
    ty: GraphicsShaderType,
}

impl<'a> GraphicsEntryPoint<'a> {
    /// Returns the input attributes used by the shader stage.
    #[inline]
    pub fn input(&self) -> &ShaderInterface {
        &self.input
    }

    /// Returns the output attributes used by the shader stage.
    #[inline]
    pub fn output(&self) -> &ShaderInterface {
        &self.output
    }

    /// Returns the type of shader.
    #[inline]
    pub fn ty(&self) -> GraphicsShaderType {
        self.ty
    }
}

unsafe impl<'a> EntryPointAbstract for GraphicsEntryPoint<'a> {
    #[inline]
    fn module(&self) -> &ShaderModule {
        self.module
    }

    #[inline]
    fn name(&self) -> &CStr {
        self.name
    }

    #[inline]
    fn descriptor_set_layout_descs(&self) -> &[DescriptorSetDesc] {
        &self.descriptor_set_layout_descs
    }

    #[inline]
    fn push_constant_range(&self) -> &Option<PipelineLayoutPcRange> {
        &self.push_constant_range
    }

    #[inline]
    fn spec_constants(&self) -> &[SpecializationMapEntry] {
        self.spec_constants
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GraphicsShaderType {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry(GeometryShaderExecutionMode),
    Fragment,
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
            (
                GeometryShaderExecutionMode::LinesWithAdjacency,
                PrimitiveTopology::LineListWithAdjacency,
            ) => true,
            (
                GeometryShaderExecutionMode::LinesWithAdjacency,
                PrimitiveTopology::LineStripWithAdjacency,
            ) => true,
            (GeometryShaderExecutionMode::Triangles, PrimitiveTopology::TriangleList) => true,
            (GeometryShaderExecutionMode::Triangles, PrimitiveTopology::TriangleStrip) => true,
            (GeometryShaderExecutionMode::Triangles, PrimitiveTopology::TriangleFan) => true,
            (
                GeometryShaderExecutionMode::TrianglesWithAdjacency,
                PrimitiveTopology::TriangleListWithAdjacency,
            ) => true,
            (
                GeometryShaderExecutionMode::TrianglesWithAdjacency,
                PrimitiveTopology::TriangleStripWithAdjacency,
            ) => true,
            _ => false,
        }
    }
}

/// Represents the entry point of a compute shader in a shader module.
///
/// Can be obtained by calling `compute_shader_entry_point()` on the shader module.
#[derive(Debug, Clone)]
pub struct ComputeEntryPoint<'a> {
    module: &'a ShaderModule,
    name: &'a CStr,
    descriptor_set_layout_descs: SmallVec<[DescriptorSetDesc; 16]>,
    push_constant_range: Option<PipelineLayoutPcRange>,
    spec_constants: &'static [SpecializationMapEntry],
}

unsafe impl<'a> EntryPointAbstract for ComputeEntryPoint<'a> {
    #[inline]
    fn module(&self) -> &ShaderModule {
        self.module
    }

    #[inline]
    fn name(&self) -> &CStr {
        self.name
    }

    #[inline]
    fn descriptor_set_layout_descs(&self) -> &[DescriptorSetDesc] {
        &self.descriptor_set_layout_descs
    }

    #[inline]
    fn push_constant_range(&self) -> &Option<PipelineLayoutPcRange> {
        &self.push_constant_range
    }

    #[inline]
    fn spec_constants(&self) -> &[SpecializationMapEntry] {
        self.spec_constants
    }
}

/// Type that contains the definition of an interface between two shader stages, or between
/// the outside and a shader stage.
#[derive(Clone, Debug)]
pub struct ShaderInterface {
    elements: Vec<ShaderInterfaceEntry>,
}

impl ShaderInterface {
    /// Constructs a new `ShaderInterface`.
    ///
    /// # Safety
    ///
    /// - Must only provide one entry per location.
    /// - The format of each element must not be larger than 128 bits.
    // TODO: could this be made safe?
    #[inline]
    pub unsafe fn new_unchecked(elements: Vec<ShaderInterfaceEntry>) -> ShaderInterface {
        ShaderInterface { elements }
    }

    /// Creates a description of an empty shader interface.
    pub const fn empty() -> ShaderInterface {
        ShaderInterface {
            elements: Vec::new(),
        }
    }

    /// Returns a slice containing the elements of the interface.
    #[inline]
    pub fn elements(&self) -> &[ShaderInterfaceEntry] {
        self.elements.as_ref()
    }

    /// Checks whether the interface is potentially compatible with another one.
    ///
    /// Returns `Ok` if the two interfaces are compatible.
    pub fn matches(&self, other: &ShaderInterface) -> Result<(), ShaderInterfaceMismatchError> {
        if self.elements().len() != other.elements().len() {
            return Err(ShaderInterfaceMismatchError::ElementsCountMismatch {
                self_elements: self.elements().len() as u32,
                other_elements: other.elements().len() as u32,
            });
        }

        for a in self.elements() {
            for loc in a.location.clone() {
                let b = match other
                    .elements()
                    .iter()
                    .find(|e| loc >= e.location.start && loc < e.location.end)
                {
                    None => {
                        return Err(ShaderInterfaceMismatchError::MissingElement { location: loc })
                    }
                    Some(b) => b,
                };

                if a.format != b.format {
                    return Err(ShaderInterfaceMismatchError::FormatMismatch {
                        location: loc,
                        self_format: a.format,
                        other_format: b.format,
                    });
                }

                // TODO: enforce this?
                /*match (a.name, b.name) {
                    (Some(ref an), Some(ref bn)) => if an != bn { return false },
                    _ => ()
                };*/
            }
        }

        // Note: since we check that the number of elements is the same, we don't need to iterate
        // over b's elements.

        Ok(())
    }
}

/// Entry of a shader interface definition.
#[derive(Debug, Clone)]
pub struct ShaderInterfaceEntry {
    /// Range of locations covered by the element.
    pub location: Range<u32>,
    /// Format of a each location of the element.
    pub format: Format,
    /// Name of the element, or `None` if the name is unknown.
    pub name: Option<Cow<'static, str>>,
}

/// Error that can happen when the interface mismatches between two shader stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderInterfaceMismatchError {
    /// The number of elements is not the same between the two shader interfaces.
    ElementsCountMismatch {
        /// Number of elements in the first interface.
        self_elements: u32,
        /// Number of elements in the second interface.
        other_elements: u32,
    },

    /// An element is missing from one of the interfaces.
    MissingElement {
        /// Location of the missing element.
        location: u32,
    },

    /// The format of an element does not match.
    FormatMismatch {
        /// Location of the element that mismatches.
        location: u32,
        /// Format in the first interface.
        self_format: Format,
        /// Format in the second interface.
        other_format: Format,
    },
}

impl error::Error for ShaderInterfaceMismatchError {}

impl fmt::Display for ShaderInterfaceMismatchError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                ShaderInterfaceMismatchError::ElementsCountMismatch { .. } => {
                    "the number of elements mismatches"
                }
                ShaderInterfaceMismatchError::MissingElement { .. } => "an element is missing",
                ShaderInterfaceMismatchError::FormatMismatch { .. } => {
                    "the format of an element does not match"
                }
            }
        )
    }
}

/// Trait for types that contain specialization data for shaders.
///
/// Shader modules can contain what is called *specialization constants*. They are the same as
/// constants except that their values can be defined when you create a compute pipeline or a
/// graphics pipeline. Doing so is done by passing a type that implements the
/// `SpecializationConstants` trait and that stores the values in question. The `descriptors()`
/// method of this trait indicates how to grab them.
///
/// Boolean specialization constants must be stored as 32bits integers, where `0` means `false` and
/// any non-zero value means `true`. Integer and floating-point specialization constants are
/// stored as their Rust equivalent.
///
/// This trait is implemented on `()` for shaders that don't have any specialization constant.
///
/// Note that it is the shader module that chooses which type that implements
/// `SpecializationConstants` it is possible to pass when creating the pipeline, through [the
/// `EntryPointAbstract` trait](trait.EntryPointAbstract.html). Therefore there is generally no
/// point to implement this trait yourself, unless you are also writing your own implementation of
/// `EntryPointAbstract`.
///
/// # Example
///
/// ```rust
/// use vulkano::pipeline::shader::SpecializationConstants;
/// use vulkano::pipeline::shader::SpecializationMapEntry;
///
/// #[repr(C)]      // `#[repr(C)]` guarantees that the struct has a specific layout
/// struct MySpecConstants {
///     my_integer_constant: i32,
///     a_boolean: u32,
///     floating_point: f32,
/// }
///
/// unsafe impl SpecializationConstants for MySpecConstants {
///     fn descriptors() -> &'static [SpecializationMapEntry] {
///         static DESCRIPTORS: [SpecializationMapEntry; 3] = [
///             SpecializationMapEntry {
///                 constant_id: 0,
///                 offset: 0,
///                 size: 4,
///             },
///             SpecializationMapEntry {
///                 constant_id: 1,
///                 offset: 4,
///                 size: 4,
///             },
///             SpecializationMapEntry {
///                 constant_id: 2,
///                 offset: 8,
///                 size: 4,
///             },
///         ];
///
///         &DESCRIPTORS
///     }
/// }
/// ```
///
/// # Safety
///
/// - The `SpecializationMapEntry` returned must contain valid offsets and sizes.
/// - The size of each `SpecializationMapEntry` must match the size of the corresponding constant
///   (`4` for booleans).
///
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

/// Describes an individual constant to set in the shader. Also a field in the struct.
// Implementation note: has the same memory representation as a `VkSpecializationMapEntry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SpecializationMapEntry {
    /// Identifier of the constant in the shader that corresponds to this field.
    ///
    /// For SPIR-V, this must be the value of the `SpecId` decoration applied to the specialization
    /// constant.
    /// For GLSL, this must be the value of `N` in the `layout(constant_id = N)` attribute applied
    /// to a constant.
    pub constant_id: u32,

    /// Offset within the struct where the data can be found.
    pub offset: u32,

    /// Size of the data in bytes. Must match the size of the constant (`4` for booleans).
    pub size: usize,
}

/// Describes a set of shader stages.
// TODO: add example with BitOr
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ShaderStages {
    pub vertex: bool,
    pub tessellation_control: bool,
    pub tessellation_evaluation: bool,
    pub geometry: bool,
    pub fragment: bool,
    pub compute: bool,
}

impl ShaderStages {
    /// Creates a `ShaderStages` struct will all stages set to `true`.
    // TODO: add example
    #[inline]
    pub const fn all() -> ShaderStages {
        ShaderStages {
            vertex: true,
            tessellation_control: true,
            tessellation_evaluation: true,
            geometry: true,
            fragment: true,
            compute: true,
        }
    }

    /// Creates a `ShaderStages` struct will all stages set to `false`.
    // TODO: add example
    #[inline]
    pub const fn none() -> ShaderStages {
        ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: false,
        }
    }

    /// Creates a `ShaderStages` struct with all graphics stages set to `true`.
    // TODO: add example
    #[inline]
    pub const fn all_graphics() -> ShaderStages {
        ShaderStages {
            vertex: true,
            tessellation_control: true,
            tessellation_evaluation: true,
            geometry: true,
            fragment: true,
            compute: false,
        }
    }

    /// Creates a `ShaderStages` struct with the compute stage set to `true`.
    // TODO: add example
    #[inline]
    pub const fn compute() -> ShaderStages {
        ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: true,
        }
    }

    /// Checks whether we have more stages enabled than `other`.
    // TODO: add example
    #[inline]
    pub const fn ensure_superset_of(
        &self,
        other: &ShaderStages,
    ) -> Result<(), ShaderStagesSupersetError> {
        if (self.vertex || !other.vertex)
            && (self.tessellation_control || !other.tessellation_control)
            && (self.tessellation_evaluation || !other.tessellation_evaluation)
            && (self.geometry || !other.geometry)
            && (self.fragment || !other.fragment)
            && (self.compute || !other.compute)
        {
            Ok(())
        } else {
            Err(ShaderStagesSupersetError::NotSuperset)
        }
    }

    /// Checks whether any of the stages in `self` are also present in `other`.
    // TODO: add example
    #[inline]
    pub const fn intersects(&self, other: &ShaderStages) -> bool {
        (self.vertex && other.vertex)
            || (self.tessellation_control && other.tessellation_control)
            || (self.tessellation_evaluation && other.tessellation_evaluation)
            || (self.geometry && other.geometry)
            || (self.fragment && other.fragment)
            || (self.compute && other.compute)
    }
}

impl From<ShaderStages> for ash::vk::ShaderStageFlags {
    #[inline]
    fn from(val: ShaderStages) -> Self {
        let mut result = ash::vk::ShaderStageFlags::empty();
        if val.vertex {
            result |= ash::vk::ShaderStageFlags::VERTEX;
        }
        if val.tessellation_control {
            result |= ash::vk::ShaderStageFlags::TESSELLATION_CONTROL;
        }
        if val.tessellation_evaluation {
            result |= ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION;
        }
        if val.geometry {
            result |= ash::vk::ShaderStageFlags::GEOMETRY;
        }
        if val.fragment {
            result |= ash::vk::ShaderStageFlags::FRAGMENT;
        }
        if val.compute {
            result |= ash::vk::ShaderStageFlags::COMPUTE;
        }
        result
    }
}

impl From<ash::vk::ShaderStageFlags> for ShaderStages {
    #[inline]
    fn from(val: ash::vk::ShaderStageFlags) -> Self {
        Self {
            vertex: val.intersects(ash::vk::ShaderStageFlags::VERTEX),
            tessellation_control: val.intersects(ash::vk::ShaderStageFlags::TESSELLATION_CONTROL),
            tessellation_evaluation: val
                .intersects(ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION),
            geometry: val.intersects(ash::vk::ShaderStageFlags::GEOMETRY),
            fragment: val.intersects(ash::vk::ShaderStageFlags::FRAGMENT),
            compute: val.intersects(ash::vk::ShaderStageFlags::COMPUTE),
        }
    }
}

impl BitOr for ShaderStages {
    type Output = ShaderStages;

    #[inline]
    fn bitor(self, other: ShaderStages) -> ShaderStages {
        ShaderStages {
            vertex: self.vertex || other.vertex,
            tessellation_control: self.tessellation_control || other.tessellation_control,
            tessellation_evaluation: self.tessellation_evaluation || other.tessellation_evaluation,
            geometry: self.geometry || other.geometry,
            fragment: self.fragment || other.fragment,
            compute: self.compute || other.compute,
        }
    }
}

impl From<ShaderStages> for PipelineStages {
    #[inline]
    fn from(stages: ShaderStages) -> PipelineStages {
        PipelineStages {
            vertex_shader: stages.vertex,
            tessellation_control_shader: stages.tessellation_control,
            tessellation_evaluation_shader: stages.tessellation_evaluation,
            geometry_shader: stages.geometry,
            fragment_shader: stages.fragment,
            compute_shader: stages.compute,
            ..PipelineStages::none()
        }
    }
}

/// Error when checking that a `ShaderStages` object is a superset of another.
#[derive(Debug, Clone)]
pub enum ShaderStagesSupersetError {
    NotSuperset,
}

impl error::Error for ShaderStagesSupersetError {}

impl fmt::Display for ShaderStagesSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                ShaderStagesSupersetError::NotSuperset => "shader stages not a superset",
            }
        )
    }
}
