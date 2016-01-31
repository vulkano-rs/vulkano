
pub fn parse_spirv(data: &[u8]) -> Result<Spirv, ParseError> {
    if data.len() < 20 {
        return Err(ParseError::MissingHeader);
    }

    // we need to determine whether we are in big endian order or little endian order depending
    // on the magic number at the start of the file
    let data = if data[0] == 0x07 && data[1] == 0x23 && data[2] == 0x02 && data[3] == 0x03 {
        // big endian
        data.chunks(4).map(|c| {
            ((c[0] as u32) << 24) | ((c[1] as u32) << 16) | ((c[2] as u32) << 8) | c[3] as u32
        }).collect::<Vec<_>>()

    } else if data[3] == 0x07 && data[2] == 0x23 && data[1] == 0x02 && data[0] == 0x03 {
        // little endian
        data.chunks(4).map(|c| {
            ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) | c[0] as u32
        }).collect::<Vec<_>>()

    } else {
        return Err(ParseError::MissingHeader);
    };

    parse_u32s(&data)
}

fn parse_u32s(i: &[u32]) -> Result<Spirv, ParseError> {
    if i.len() < 5 {
        return Err(ParseError::MissingHeader);
    }

    if i[0] != 0x07230203 {
        return Err(ParseError::WrongHeader);
    }

    let version = (((i[1] & 0x00ff0000) >> 16) as u8, ((i[1] & 0x0000ff00) >> 8) as u8);

    let instructions = {
        let mut ret = Vec::new();
        let mut i = &i[5..];
        while i.len() >= 1 {
            let (instruction, rest) = try!(parse_instruction(i));
            ret.push(instruction);
            i = rest;
        }
        ret
    };

    Ok(Spirv {
        version: version,
        bound: i[3],
        instructions: instructions,
    })
}

#[derive(Debug, Clone)]
pub enum ParseError {
    MissingHeader,
    WrongHeader,
    IncompleteInstruction,
    UnknownCapability(u32),
    UnknownExecutionModel(u32),
}

#[derive(Debug, Clone)]
pub struct Spirv {
    pub version: (u8, u8),
    pub bound: u32,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Unknown(u16, Vec<u32>),
    Nop,
    ExtInstImport { result_id: u32, name: String },
    EntryPoint { execution: ExecutionModel, id: u32, name: String, interface: Vec<u32> },
    Capability(Capability),
    FunctionEnd,
    Kill,
    Return,
}

fn parse_instruction(i: &[u32]) -> Result<(Instruction, &[u32]), ParseError> {
    assert!(i.len() >= 1);

    let word_count = (i[0] >> 16) as usize;
    assert!(word_count >= 1);
    let opcode = (i[0] & 0xffff) as u16;

    if i.len() < word_count {
        return Err(ParseError::IncompleteInstruction);
    }

    let opcode = try!(decode_instruction(opcode, &i[1 .. word_count]));
    Ok((opcode, &i[word_count..]))
}

fn decode_instruction(opcode: u16, operands: &[u32]) -> Result<Instruction, ParseError> {
    Ok(match opcode {
        0 => Instruction::Nop,
        11 => Instruction::ExtInstImport {
            result_id: operands[0],
            name: parse_string(&operands[1..]).0
        },
        15 => {
            let (n, r) = parse_string(&operands[2..]);
            Instruction::EntryPoint {
                execution: try!(ExecutionModel::from_num(operands[0])),
                id: operands[1],
                name: n,
                interface: r.to_owned(),
            }
        },
        17 => Instruction::Capability(try!(Capability::from_num(operands[0]))),
        56 => Instruction::FunctionEnd,
        252 => Instruction::Kill,
        253 => Instruction::Return,
        _ => Instruction::Unknown(opcode, operands.to_owned()),
    })
}

fn parse_string(data: &[u32]) -> (String, &[u32]) {
    let bytes = data.iter().flat_map(|&n| {
        let b1 = (n & 0xff) as u8;
        let b2 = ((n >> 8) & 0xff) as u8;
        let b3 = ((n >> 16) & 0xff) as u8;
        let b4 = ((n >> 24) & 0xff) as u8;
        vec![b1, b2, b3, b4].into_iter()
    }).take_while(|&b| b != 0).collect::<Vec<u8>>();

    let r = 1 + bytes.len() / 4;
    let s = String::from_utf8(bytes).expect("Shader content is not UTF-8");

    (s, &data[r..])
}

#[derive(Debug, Clone)]
pub enum Capability {
    Matrix,
    Shader,
    Geometry,
    Tessellation,
    Addresses,
    Linkage,
    Kernel,
    Vector16,
    Float16Buffer,
    Float16,
    Float64,
    Int64,
    Int64Atomics,
    ImageBasic,
    ImageReadWrite,
    ImageMipmap,
    Pipes,
    Groups,
    DeviceEnqueue,
    LiteralSampler,
    AtomicStorage,
    Int16,
    TessellationPointSize,
    GeometryPointSize,
    ImageGatherExtended,
    StorageImageMultisample,
    UniformBufferArrayDynamicIndexing,
    SampledImageArrayDynamicIndexing,
    StorageBufferArrayDynamicIndexing,
    StorageImageArrayDynamicIndexing,
    ClipDistance,
    CullDistance,
    ImageCubeArray,
    SampleRateShading,
    ImageRect,
    SampledRect,
    GenericPointer,
    Int8,
    InputAttachment,
    SparseResidency,
    MinLod,
    Sampled1D,
    Image1D,
    SampledCubeArray,
    SampledBuffer,
    ImageBuffer,
    ImageMSArray,
    StorageImageExtendedFormats,
    ImageQuery,
    DerivativeControl,
    InterpolationFunction,
    TransformFeedback,
    GeometryStreams,
    StorageImageReadWithoutFormat,
    StorageImageWriteWithoutFormat,
}

impl Capability {
    fn from_num(num: u32) -> Result<Capability, ParseError> {
        match num {
            0 => Ok(Capability::Matrix),
            1 => Ok(Capability::Shader),
            2 => Ok(Capability::Geometry),
            3 => Ok(Capability::Tessellation),
            4 => Ok(Capability::Addresses),
            5 => Ok(Capability::Linkage),
            6 => Ok(Capability::Kernel),
            7 => Ok(Capability::Vector16),
            8 => Ok(Capability::Float16Buffer),
            9 => Ok(Capability::Float16),
            10 => Ok(Capability::Float64),
            11 => Ok(Capability::Int64),
            12 => Ok(Capability::Int64Atomics),
            13 => Ok(Capability::ImageBasic),
            14 => Ok(Capability::ImageReadWrite),
            15 => Ok(Capability::ImageMipmap),
            17 => Ok(Capability::Pipes),
            18 => Ok(Capability::Groups),
            19 => Ok(Capability::DeviceEnqueue),
            20 => Ok(Capability::LiteralSampler),
            21 => Ok(Capability::AtomicStorage),
            22 => Ok(Capability::Int16),
            23 => Ok(Capability::TessellationPointSize),
            24 => Ok(Capability::GeometryPointSize),
            25 => Ok(Capability::ImageGatherExtended),
            27 => Ok(Capability::StorageImageMultisample),
            28 => Ok(Capability::UniformBufferArrayDynamicIndexing),
            29 => Ok(Capability::SampledImageArrayDynamicIndexing),
            30 => Ok(Capability::StorageBufferArrayDynamicIndexing),
            31 => Ok(Capability::StorageImageArrayDynamicIndexing),
            32 => Ok(Capability::ClipDistance),
            33 => Ok(Capability::CullDistance),
            34 => Ok(Capability::ImageCubeArray),
            35 => Ok(Capability::SampleRateShading),
            36 => Ok(Capability::ImageRect),
            37 => Ok(Capability::SampledRect),
            38 => Ok(Capability::GenericPointer),
            39 => Ok(Capability::Int8),
            40 => Ok(Capability::InputAttachment),
            41 => Ok(Capability::SparseResidency),
            42 => Ok(Capability::MinLod),
            43 => Ok(Capability::Sampled1D),
            44 => Ok(Capability::Image1D),
            45 => Ok(Capability::SampledCubeArray),
            46 => Ok(Capability::SampledBuffer),
            47 => Ok(Capability::ImageBuffer),
            48 => Ok(Capability::ImageMSArray),
            49 => Ok(Capability::StorageImageExtendedFormats),
            50 => Ok(Capability::ImageQuery),
            51 => Ok(Capability::DerivativeControl),
            52 => Ok(Capability::InterpolationFunction),
            53 => Ok(Capability::TransformFeedback),
            54 => Ok(Capability::GeometryStreams),
            55 => Ok(Capability::StorageImageReadWithoutFormat),
            56 => Ok(Capability::StorageImageWriteWithoutFormat),
            _ => Err(ParseError::UnknownCapability(num)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExecutionModel {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    GLCompute,
    Kernel,
}

impl ExecutionModel {
    fn from_num(num: u32) -> Result<ExecutionModel, ParseError> {
        match num {
            0 => Ok(ExecutionModel::Vertex),
            1 => Ok(ExecutionModel::TessellationControl),
            2 => Ok(ExecutionModel::TessellationEvaluation),
            3 => Ok(ExecutionModel::Geometry),
            4 => Ok(ExecutionModel::Fragment),
            5 => Ok(ExecutionModel::GLCompute),
            6 => Ok(ExecutionModel::Kernel),
            _ => Err(ParseError::UnknownExecutionModel(num)),
        }
    }
}

#[cfg(test)]
mod test {
    use parse;

    #[test]
    fn test() {
        let data = include_bytes!("../tests/frag.spv");
        let doc = parse::parse_spirv(data).unwrap();

        println!("{:?}", doc);
    }
}
