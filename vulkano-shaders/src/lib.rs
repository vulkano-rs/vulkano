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
pub struct {name} {{
    shader: ::std::sync::Arc<::vulkano::shader::ShaderModule>,
}}

impl {name} {{
    /// Loads the shader in Vulkan as a `ShaderModule`.
    #[inline]
    pub fn load(device: &::std::sync::Arc<::vulkano::Device>) -> {name} {{

        "#, name = name));

        // checking whether each required capability is supported by the vulkan implementation
        for i in doc.instructions.iter() {
            if let &parse::Instruction::Capability(ref cap) = i {
                output.push_str(&format!(r#"
                    if !device.is_capability_supported("{cap}") {{
                        return Err(CapabilityNotSupported);
                    }}"#, cap = capability_name(cap)));
            }
        }

        // follow-up of the header
        output.push_str(&format!(r#"
        unsafe {{
            let data = [{spirv_data}];

            {name} {{
                shader: ::vulkano::shader::ShaderModule::new(device, &data)
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

    let ty = match *execution {
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

            format!("::vulkano::shader::VertexShaderEntryPoint<({input})>",
                    input = input_types.join(", ") + ",")
        },

        enums::ExecutionModel::ExecutionModelTessellationControl => {
            format!("::vulkano::shader::TessControlShaderEntryPoint")
        },

        enums::ExecutionModel::ExecutionModelTessellationEvaluation => {
            format!("::vulkano::shader::TessEvaluationShaderEntryPoint")
        },

        enums::ExecutionModel::ExecutionModelGeometry => {
            format!("::vulkano::shader::GeometryShaderEntryPoint")
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

            format!("::vulkano::shader::FragmentShaderEntryPoint<({output})>",
                    output = output_types.join(", ") + ",")
        },

        enums::ExecutionModel::ExecutionModelGLCompute => {
            format!("::vulkano::shader::ComputeShaderEntryPoint")
        },

        enums::ExecutionModel::ExecutionModelKernel => panic!("Kernels are not supported"),
    };

    format!(r#"
    /// Returns a logical struct describing the entry point named `{ep_name}`.
    #[inline]
    pub fn {ep_name}_entry_point(&self) -> {ty} {{
        unsafe {{
            static NAME: [u8; {ep_name_lenp1}] = [{encoded_ep_name}, 0];
            self.shader.entry_point(::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _))
        }}
    }}
            "#, ep_name = ep_name, ep_name_lenp1 = ep_name.chars().count() + 1, ty = ty,
                encoded_ep_name = ep_name.chars().map(|c| (c as u32).to_string())
                                         .collect::<Vec<String>>().join(", "))
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

// TODO: this function is a draft, as the actual names may not be the same
fn capability_name(cap: &enums::Capability) -> &'static str {
    match *cap {
        enums::Capability::CapabilityMatrix => "Matrix",
        enums::Capability::CapabilityShader => "Shader",
        enums::Capability::CapabilityGeometry => "Geometry",
        enums::Capability::CapabilityTessellation => "Tessellation",
        enums::Capability::CapabilityAddresses => "Addresses",
        enums::Capability::CapabilityLinkage => "Linkage",
        enums::Capability::CapabilityKernel => "Kernel",
        enums::Capability::CapabilityVector16 => "Vector16",
        enums::Capability::CapabilityFloat16Buffer => "Float16Buffer",
        enums::Capability::CapabilityFloat16 => "Float16",
        enums::Capability::CapabilityFloat64 => "Float64",
        enums::Capability::CapabilityInt64 => "Int64",
        enums::Capability::CapabilityInt64Atomics => "Int64Atomics",
        enums::Capability::CapabilityImageBasic => "ImageBasic",
        enums::Capability::CapabilityImageReadWrite => "ImageReadWrite",
        enums::Capability::CapabilityImageMipmap => "ImageMipmap",
        enums::Capability::CapabilityPipes => "Pipes",
        enums::Capability::CapabilityGroups => "Groups",
        enums::Capability::CapabilityDeviceEnqueue => "DeviceEnqueue",
        enums::Capability::CapabilityLiteralSampler => "LiteralSampler",
        enums::Capability::CapabilityAtomicStorage => "AtomicStorage",
        enums::Capability::CapabilityInt16 => "Int16",
        enums::Capability::CapabilityTessellationPointSize => "TessellationPointSize",
        enums::Capability::CapabilityGeometryPointSize => "GeometryPointSize",
        enums::Capability::CapabilityImageGatherExtended => "ImageGatherExtended",
        enums::Capability::CapabilityStorageImageMultisample => "StorageImageMultisample",
        enums::Capability::CapabilityUniformBufferArrayDynamicIndexing => "UniformBufferArrayDynamicIndexing",
        enums::Capability::CapabilitySampledImageArrayDynamicIndexing => "SampledImageArrayDynamicIndexing",
        enums::Capability::CapabilityStorageBufferArrayDynamicIndexing => "StorageBufferArrayDynamicIndexing",
        enums::Capability::CapabilityStorageImageArrayDynamicIndexing => "StorageImageArrayDynamicIndexing",
        enums::Capability::CapabilityClipDistance => "ClipDistance",
        enums::Capability::CapabilityCullDistance => "CullDistance",
        enums::Capability::CapabilityImageCubeArray => "ImageCubeArray",
        enums::Capability::CapabilitySampleRateShading => "SampleRateShading",
        enums::Capability::CapabilityImageRect => "ImageRect",
        enums::Capability::CapabilitySampledRect => "SampledRect",
        enums::Capability::CapabilityGenericPointer => "GenericPointer",
        enums::Capability::CapabilityInt8 => "Int8",
        enums::Capability::CapabilityInputAttachment => "InputAttachment",
        enums::Capability::CapabilitySparseResidency => "SparseResidency",
        enums::Capability::CapabilityMinLod => "MinLod",
        enums::Capability::CapabilitySampled1D => "Sampled1D",
        enums::Capability::CapabilityImage1D => "Image1D",
        enums::Capability::CapabilitySampledCubeArray => "SampledCubeArray",
        enums::Capability::CapabilitySampledBuffer => "SampledBuffer",
        enums::Capability::CapabilityImageBuffer => "ImageBuffer",
        enums::Capability::CapabilityImageMSArray => "ImageMSArray",
        enums::Capability::CapabilityStorageImageExtendedFormats => "StorageImageExtendedFormats",
        enums::Capability::CapabilityImageQuery => "ImageQuery",
        enums::Capability::CapabilityDerivativeControl => "DerivativeControl",
        enums::Capability::CapabilityInterpolationFunction => "InterpolationFunction",
        enums::Capability::CapabilityTransformFeedback => "TransformFeedback",
        enums::Capability::CapabilityGeometryStreams => "GeometryStreams",
        enums::Capability::CapabilityStorageImageReadWithoutFormat => "StorageImageReadWithoutFormat",
        enums::Capability::CapabilityStorageImageWriteWithoutFormat => "StorageImageWriteWithoutFormat",
    }
}
