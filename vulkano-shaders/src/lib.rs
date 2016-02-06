use std::io::Error as IoError;
use std::io::Read;

pub use parse::ParseError;

mod enums;
mod parse;

pub fn reflect<R>(name: &str, mut spirv: R) -> Result<String, Error>
    where R: Read
{
    let mut data = Vec::new();
    try!(spirv.read_to_end(&mut data));

    // now parsing the document
    let doc = try!(parse::parse_spirv(&data));

    let mut output = String::new();

    {
        // contains the data that was passed as input to this function
        let spirv_data = data.iter().map(|&byte| byte.to_string())
                             .collect::<Vec<String>>()
                             .join(", ");

        // writing the header
        output.push_str(&format!(r#"
struct {name} {{
    shader: ::std::sync::Arc<::vulkano::shader::ShaderModule>,
}}

impl {name} {{
    /// Loads the shader in Vulkan as a `ShaderModule`.
    #[inline]
    pub fn load(device: &::std::sync::Arc<::vulkano::device::Device>) -> {name} {{

        "#, name = name));

        // checking whether each required capability is supported by the vulkan implementation
        for i in doc.instructions.iter() {
            if let &parse::Instruction::Capability(ref cap) = i {
                if let Some(cap) = capability_name(cap) {
                    output.push_str(&format!(r#"
                        if !device.enabled_features().{cap} {{
                            panic!("capability not supported")  // FIXME: error
                            //return Err(CapabilityNotSupported);
                        }}"#, cap = cap));
                }
            }
        }

        // follow-up of the header
        output.push_str(&format!(r#"
        unsafe {{
            let data = [{spirv_data}];

            {name} {{
                shader: ::vulkano::shader::ShaderModule::new(device, &data).unwrap()    // FIXME: try!()
            }}
        }}
    }}

    /// Returns the module that was created.
    #[inline]
    pub fn module(&self) -> &::std::sync::Arc<::vulkano::shader::ShaderModule> {{
        &self.shader
    }}
        "#, name = name, spirv_data = spirv_data));

        // writing one method for each entry point of this module
        for instruction in doc.instructions.iter() {
            if let &parse::Instruction::EntryPoint { .. } = instruction {
                output.push_str(&write_entry_point(&doc, instruction));
            }
        }

        // footer
        output.push_str(&format!(r#"
}}
        "#));
    }

    // TODO: remove
    println!("{:#?}", doc);

    Ok(output)
}

#[derive(Debug)]
pub enum Error {
    IoError(IoError),
    ParseError(ParseError),
}

impl From<IoError> for Error {
    #[inline]
    fn from(err: IoError) -> Error {
        Error::IoError(err)
    }
}

impl From<ParseError> for Error {
    #[inline]
    fn from(err: ParseError) -> Error {
        Error::ParseError(err)
    }
}

fn write_entry_point(doc: &parse::Spirv, instruction: &parse::Instruction) -> String {
    let (execution, ep_name, interface) = match instruction {
        &parse::Instruction::EntryPoint { ref execution, id, ref name, ref interface } => {
            (execution, name, interface)
        },
        _ => unreachable!()
    };

    let (ty, f_name) = match *execution {
        enums::ExecutionModel::ExecutionModelVertex => {
            let mut input_types = Vec::new();

            // TODO: sort types by location

            for interface in interface.iter() {
                for i in doc.instructions.iter() {
                    match i {
                        &parse::Instruction::Variable { result_type_id, result_id,
                                    storage_class: enums::StorageClass::StorageClassInput, .. }
                                    if &result_id == interface =>
                        {
                            input_types.push(type_from_id(doc, result_type_id));
                        },
                        _ => ()
                    }
                }
            }

            let t = format!("::vulkano::shader::VertexShaderEntryPoint<({input})>",
                            input = input_types.join(", ") + ",");
            (t, "vertex_shader_entry_point")
        },

        enums::ExecutionModel::ExecutionModelTessellationControl => {
            (format!("::vulkano::shader::TessControlShaderEntryPoint"), "")
        },

        enums::ExecutionModel::ExecutionModelTessellationEvaluation => {
            (format!("::vulkano::shader::TessEvaluationShaderEntryPoint"), "")
        },

        enums::ExecutionModel::ExecutionModelGeometry => {
            (format!("::vulkano::shader::GeometryShaderEntryPoint"), "")
        },

        enums::ExecutionModel::ExecutionModelFragment => {
            let mut output_types = Vec::new();

            for interface in interface.iter() {
                for i in doc.instructions.iter() {
                    match i {
                        &parse::Instruction::Variable { result_type_id, result_id,
                                    storage_class: enums::StorageClass::StorageClassOutput, .. }
                                    if &result_id == interface =>
                        {
                            output_types.push(type_from_id(doc, result_type_id));
                        },
                        _ => ()
                    }
                }
            }

            let t = format!("::vulkano::shader::FragmentShaderEntryPoint<({output})>",
                            output = output_types.join(", ") + ",");
            (t, "fragment_shader_entry_point")
        },

        enums::ExecutionModel::ExecutionModelGLCompute => {
            (format!("::vulkano::shader::ComputeShaderEntryPoint"), "compute_shader_entry_point")
        },

        enums::ExecutionModel::ExecutionModelKernel => panic!("Kernels are not supported"),
    };

    format!(r#"
    /// Returns a logical struct describing the entry point named `{ep_name}`.
    #[inline]
    pub fn {ep_name}_entry_point(&self) -> {ty} {{
        unsafe {{
            static NAME: [u8; {ep_name_lenp1}] = [{encoded_ep_name}, 0];
            self.shader.{f_name}(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _))
        }}
    }}
            "#, ep_name = ep_name, ep_name_lenp1 = ep_name.chars().count() + 1, ty = ty,
                encoded_ep_name = ep_name.chars().map(|c| (c as u32).to_string())
                                         .collect::<Vec<String>>().join(", "),
                f_name = f_name)
}

fn type_from_id(doc: &parse::Spirv, searched: u32) -> String {
    for instruction in doc.instructions.iter() {
        match instruction {
            &parse::Instruction::TypeVoid { result_id } if result_id == searched => {
                return "()".to_owned()
            },
            &parse::Instruction::TypeBool { result_id } if result_id == searched => {
                return "bool".to_owned()
            },
            &parse::Instruction::TypeInt { result_id, width, signedness } if result_id == searched => {
                return "i32".to_owned()
            },
            &parse::Instruction::TypeFloat { result_id, width } if result_id == searched => {
                return "f32".to_owned()
            },
            &parse::Instruction::TypeVector { result_id, component_id, count } if result_id == searched => {
                let t = type_from_id(doc, component_id);
                return format!("[{}; {}]", t, count);
            },
            &parse::Instruction::TypeImage { result_id, sampled_type_id, ref dim, depth, arrayed, ms, sampled, ref format, ref access } if result_id == searched => {
                return format!("{}{}Texture{:?}{}{:?}",
                    if ms { "Multisample" } else { "" },
                    if depth == Some(true) { "Depth" } else { "" },
                    dim,
                    if arrayed { "Array" } else { "" },
                    format);
            },
            &parse::Instruction::TypeSampledImage { result_id, image_type_id } if result_id == searched => {
                return type_from_id(doc, image_type_id);
            },
            &parse::Instruction::TypeArray { result_id, type_id, length_id } if result_id == searched => {
                let t = type_from_id(doc, type_id);
                let len = doc.instructions.iter().filter_map(|e| {
                    match e { &parse::Instruction::Constant { result_id, ref data, .. } if result_id == length_id => Some(data.clone()), _ => None }
                }).next().expect("failed to find array length");
                let len = len.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);
                return format!("[{}; {}]", t, len);       // FIXME:
            },
            &parse::Instruction::TypeRuntimeArray { result_id, type_id } if result_id == searched => {
                let t = type_from_id(doc, type_id);
                return format!("[{}]", t);
            },
            &parse::Instruction::TypeStruct { result_id, ref member_types } if result_id == searched => {
                let name = name_from_id(doc, result_id);
                let members = member_types.iter().enumerate().map(|(offset, &member)| {
                    let ty = type_from_id(doc, member);
                    let name = member_name_from_id(doc, result_id, offset as u32);
                    format!("\t{}: {}", name, ty)
                }).collect::<Vec<_>>();
                return format!("struct {} {{\n{}\n}}", name, members.join(",\n"));
            },
            &parse::Instruction::TypeOpaque { result_id, ref name } if result_id == searched => {
                return "<opaque>".to_owned();
            },
            &parse::Instruction::TypePointer { result_id, type_id, .. } if result_id == searched => {
                return type_from_id(doc, type_id);
            },
            _ => ()
        }
    }

    panic!("Type #{} not found", searched)
}

fn name_from_id(doc: &parse::Spirv, searched: u32) -> String {
    doc.instructions.iter().filter_map(|i| {
        if let &parse::Instruction::Name { target_id, ref name } = i {
            if target_id == searched {
                Some(name.clone())
            } else {
                None
            }
        } else {
            None
        }
    }).next().and_then(|n| if !n.is_empty() { Some(n) } else { None })
      .unwrap_or("__unnamed".to_owned())
}

fn member_name_from_id(doc: &parse::Spirv, searched: u32, searched_member: u32) -> String {
    doc.instructions.iter().filter_map(|i| {
        if let &parse::Instruction::MemberName { target_id, member, ref name } = i {
            if target_id == searched && member == searched_member {
                Some(name.clone())
            } else {
                None
            }
        } else {
            None
        }
    }).next().and_then(|n| if !n.is_empty() { Some(n) } else { None })
      .unwrap_or("__unnamed".to_owned())
}

/// Returns the name of the Vulkan something that corresponds to an `OpCapability`.
///
/// Returns `None` if irrelevant.
// TODO: this function is a draft, as the actual names may not be the same
fn capability_name(cap: &enums::Capability) -> Option<&'static str> {
    match *cap {
        enums::Capability::CapabilityMatrix => Some("Matrix"),
        enums::Capability::CapabilityShader => None,
        enums::Capability::CapabilityGeometry => Some("Geometry"),
        enums::Capability::CapabilityTessellation => Some("Tessellation"),
        enums::Capability::CapabilityAddresses => Some("Addresses"),
        enums::Capability::CapabilityLinkage => Some("Linkage"),
        enums::Capability::CapabilityKernel => Some("Kernel"),
        enums::Capability::CapabilityVector16 => Some("Vector16"),
        enums::Capability::CapabilityFloat16Buffer => Some("Float16Buffer"),
        enums::Capability::CapabilityFloat16 => Some("Float16"),
        enums::Capability::CapabilityFloat64 => Some("Float64"),
        enums::Capability::CapabilityInt64 => Some("Int64"),
        enums::Capability::CapabilityInt64Atomics => Some("Int64Atomics"),
        enums::Capability::CapabilityImageBasic => Some("ImageBasic"),
        enums::Capability::CapabilityImageReadWrite => Some("ImageReadWrite"),
        enums::Capability::CapabilityImageMipmap => Some("ImageMipmap"),
        enums::Capability::CapabilityPipes => Some("Pipes"),
        enums::Capability::CapabilityGroups => Some("Groups"),
        enums::Capability::CapabilityDeviceEnqueue => Some("DeviceEnqueue"),
        enums::Capability::CapabilityLiteralSampler => Some("LiteralSampler"),
        enums::Capability::CapabilityAtomicStorage => Some("AtomicStorage"),
        enums::Capability::CapabilityInt16 => Some("Int16"),
        enums::Capability::CapabilityTessellationPointSize => Some("TessellationPointSize"),
        enums::Capability::CapabilityGeometryPointSize => Some("GeometryPointSize"),
        enums::Capability::CapabilityImageGatherExtended => Some("ImageGatherExtended"),
        enums::Capability::CapabilityStorageImageMultisample => Some("StorageImageMultisample"),
        enums::Capability::CapabilityUniformBufferArrayDynamicIndexing => Some("UniformBufferArrayDynamicIndexing"),
        enums::Capability::CapabilitySampledImageArrayDynamicIndexing => Some("SampledImageArrayDynamicIndexing"),
        enums::Capability::CapabilityStorageBufferArrayDynamicIndexing => Some("StorageBufferArrayDynamicIndexing"),
        enums::Capability::CapabilityStorageImageArrayDynamicIndexing => Some("StorageImageArrayDynamicIndexing"),
        enums::Capability::CapabilityClipDistance => Some("ClipDistance"),
        enums::Capability::CapabilityCullDistance => Some("CullDistance"),
        enums::Capability::CapabilityImageCubeArray => Some("ImageCubeArray"),
        enums::Capability::CapabilitySampleRateShading => Some("SampleRateShading"),
        enums::Capability::CapabilityImageRect => Some("ImageRect"),
        enums::Capability::CapabilitySampledRect => Some("SampledRect"),
        enums::Capability::CapabilityGenericPointer => Some("GenericPointer"),
        enums::Capability::CapabilityInt8 => Some("Int8"),
        enums::Capability::CapabilityInputAttachment => Some("InputAttachment"),
        enums::Capability::CapabilitySparseResidency => Some("SparseResidency"),
        enums::Capability::CapabilityMinLod => Some("MinLod"),
        enums::Capability::CapabilitySampled1D => Some("Sampled1D"),
        enums::Capability::CapabilityImage1D => Some("Image1D"),
        enums::Capability::CapabilitySampledCubeArray => Some("SampledCubeArray"),
        enums::Capability::CapabilitySampledBuffer => Some("SampledBuffer"),
        enums::Capability::CapabilityImageBuffer => Some("ImageBuffer"),
        enums::Capability::CapabilityImageMSArray => Some("ImageMSArray"),
        enums::Capability::CapabilityStorageImageExtendedFormats => Some("StorageImageExtendedFormats"),
        enums::Capability::CapabilityImageQuery => Some("ImageQuery"),
        enums::Capability::CapabilityDerivativeControl => Some("DerivativeControl"),
        enums::Capability::CapabilityInterpolationFunction => Some("InterpolationFunction"),
        enums::Capability::CapabilityTransformFeedback => Some("TransformFeedback"),
        enums::Capability::CapabilityGeometryStreams => Some("GeometryStreams"),
        enums::Capability::CapabilityStorageImageReadWithoutFormat => Some("StorageImageReadWithoutFormat"),
        enums::Capability::CapabilityStorageImageWriteWithoutFormat => Some("StorageImageWriteWithoutFormat"),
    }
}
