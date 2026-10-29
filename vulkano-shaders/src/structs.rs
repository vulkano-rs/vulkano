use crate::{bail, spirv_words_to_rspirv2_words, LinAlgType, MacroOptions, Result};
use foldhash::HashMap;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use rspirv2::{
    core::{
        inst::{
            OpConstant, OpDecorate, OpMemberDecorate, OpMemberName, OpName, OpTypeArray,
            OpTypeFloat, OpTypeInt, OpTypeMatrix, OpTypePointer, OpTypeRuntimeArray, OpTypeStruct,
            OpTypeVector,
        },
        preamble::{self, CoreInstSet, Decoration},
    },
    inst::{Inst, InstEncoding},
    module::SpirvHeader,
    operand::{IdRef, IdResult},
    slice::InstSlice,
};
use std::{cmp::Ordering, num::NonZero};
use syn::{Error, Ident, LitStr};

/// Translates all the structs that are contained in the SPIR-V document as Rust structs.
pub(super) fn generate_structs(
    options: &MacroOptions,
    lit: LitStr,
    shader_name: Option<String>,
    words: &[u32],
    type_registry: &mut TypeRegistry,
) -> Result<TokenStream> {
    if !options.generate_structs {
        return Ok(TokenStream::new());
    }

    let words = spirv_words_to_rspirv2_words(words);
    let (header, insts) = words.split_at(5);
    let header = SpirvHeader::from_slice(header);
    header.validate().unwrap();
    let insts = InstSlice::from_words(insts).map_err(|err| {
        Error::new_spanned(&lit, format_args!("failed to parse SPIR-V words: {err}"))
    })?;
    let shader = Shader {
        lit,
        name: shader_name.unwrap_or_default(),
        insts,
    };

    let mut structs = TokenStream::new();

    for op_type_struct in shader
        .insts
        .iter_inst::<OpTypeStruct>()
        .filter(|op_type_struct| has_defined_layout(&shader, op_type_struct))
    {
        let struct_ty = TypeStruct::new(&shader, &op_type_struct);

        // Register the type if needed.
        if !type_registry.register_struct(&shader, &struct_ty)? {
            continue;
        }

        if struct_ty.is_bindless_id() {
            continue;
        }

        let custom_derives = if struct_ty.size().is_some() {
            options.custom_derives.as_slice()
        } else {
            &[]
        };
        let struct_ser = Serializer(&struct_ty, options);

        structs.extend(quote! {
            #[allow(non_camel_case_types, non_snake_case)]
            #[derive(::vulkano::buffer::BufferContents #(, #custom_derives )* )]
            #[repr(C)]
            #struct_ser
        })
    }

    Ok(structs)
}

struct Shader<'a> {
    lit: LitStr,
    name: String,
    insts: &'a InstSlice<CoreInstSet>,
}

fn has_defined_layout(shader: &Shader<'_>, op_type_struct: &OpTypeStruct) -> bool {
    let mut offset_found = false;

    for op_member_decorate in
        shader
            .insts
            .iter_inst::<OpMemberDecorate>()
            .filter(|op_member_decorate| {
                op_member_decorate.structure_type.0 == op_type_struct.id_result
            })
    {
        match op_member_decorate.decoration {
            Decoration::BuiltIn(_) => {
                // Ignore the whole struct if a member is built in, which includes `gl_Position`
                // for example.
                return false;
            }
            Decoration::Offset(_) => {
                offset_found = true;
            }
            _ => {}
        }
    }

    // Some structs don't have `Offset` decorations. In that case, they are used as local variables
    // only. Ignoring these.
    offset_found
}

#[derive(Default)]
pub struct TypeRegistry {
    registered_structs: HashMap<Ident, RegisteredType>,
}

impl TypeRegistry {
    fn register_struct(&mut self, shader: &Shader<'_>, ty: &TypeStruct) -> Result<bool> {
        // Checking with registry if this struct is already registered by another shader, and if
        // their signatures match.
        if let Some(registered) = self.registered_structs.get(&ty.ident) {
            registered.validate_signatures(&shader.name, ty)?;

            // If the struct is already registered and matches this one, skip the duplicate.
            Ok(false)
        } else {
            self.registered_structs.insert(
                ty.ident.clone(),
                RegisteredType {
                    shader: shader.name.clone(),
                    ty: ty.clone(),
                },
            );

            Ok(true)
        }
    }
}

struct RegisteredType {
    shader: String,
    ty: TypeStruct,
}

impl RegisteredType {
    fn validate_signatures(&self, other_shader: &str, other_ty: &TypeStruct) -> Result {
        let (shader, struct_ident) = (&self.shader, &self.ty.ident);

        if self.ty.members.len() > other_ty.members.len() {
            let member_ident = &self.ty.members[other_ty.members.len()].ident;
            bail!(
                "shaders `{shader}` and `{other_shader}` declare structs with the same name \
                `{struct_ident}`, but the struct from shader `{shader}` contains an extra field \
                `{member_ident}`",
            );
        }

        if self.ty.members.len() < other_ty.members.len() {
            let member_ident = &other_ty.members[self.ty.members.len()].ident;
            bail!(
                "shaders `{shader}` and `{other_shader}` declare structs with the same name \
                `{struct_ident}`, but the struct from shader `{other_shader}` contains an extra \
                field `{member_ident}`",
            );
        }

        for (index, (member, other_member)) in self
            .ty
            .members
            .iter()
            .zip(other_ty.members.iter())
            .enumerate()
        {
            if member.ty != other_member.ty {
                let (member_ty, other_member_ty) = (&member.ty, &other_member.ty);
                bail!(
                    "shaders `{shader}` and `{other_shader}` declare structs with the same name \
                    `{struct_ident}`, but the struct from shader `{shader}` contains a field of \
                    type `{member_ty:?}` at index `{index}`, whereas the same struct from shader \
                    `{other_shader}` contains a field of type `{other_member_ty:?}` in the same \
                    position",
                );
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Alignment {
    A1 = 1,
    A2 = 2,
    A4 = 4,
    A8 = 8,
    A16 = 16,
    A32 = 32,
}

impl Alignment {
    fn new(alignment: usize) -> Self {
        match alignment {
            1 => Alignment::A1,
            2 => Alignment::A2,
            4 => Alignment::A4,
            8 => Alignment::A8,
            16 => Alignment::A16,
            32 => Alignment::A32,
            _ => unreachable!(),
        }
    }
}

impl PartialOrd for Alignment {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Alignment {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as usize).cmp(&(*other as usize))
    }
}

fn align_up(offset: usize, alignment: Alignment) -> usize {
    (offset + alignment as usize - 1) & !(alignment as usize - 1)
}

const OP_TYPE_BOOL_OPCODE: u16 = preamble::OP_TYPE_BOOL.opcode;
const OP_TYPE_INT_OPCODE: u16 = preamble::OP_TYPE_INT.opcode;
const OP_TYPE_FLOAT_OPCODE: u16 = preamble::OP_TYPE_FLOAT.opcode;
const OP_TYPE_POINTER_OPCODE: u16 = preamble::OP_TYPE_POINTER.opcode;
const OP_TYPE_VECTOR_OPCODE: u16 = preamble::OP_TYPE_VECTOR.opcode;
const OP_TYPE_MATRIX_OPCODE: u16 = preamble::OP_TYPE_MATRIX.opcode;
const OP_TYPE_ARRAY_OPCODE: u16 = preamble::OP_TYPE_ARRAY.opcode;
const OP_TYPE_RUNTIME_ARRAY_OPCODE: u16 = preamble::OP_TYPE_RUNTIME_ARRAY.opcode;
const OP_TYPE_STRUCT_OPCODE: u16 = preamble::OP_TYPE_STRUCT.opcode;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Type {
    Scalar(TypeScalar),
    Pointer(TypePointer),
    Vector(TypeVector),
    Matrix(TypeMatrix),
    Array(TypeArray),
    Struct(TypeStruct),
}

impl Type {
    fn new(shader: &Shader<'_>, id_ref: IdRef) -> Self {
        shader
            .insts
            .iter_ref()
            .find_map(|inst_ref| {
                // For the below instructions, the Result ID is always the first operand. So if
                // this word doesn't match the type we're searching for, it can't be it (a true
                // negative), so false negatives can't happen. There is the possibility of false
                // positives if it's not one of the below types, but worst case, we fall through in
                // the below code, which results in a true negative again. Otherwise, both the
                // Result ID and opcode must match, so we end up with a true positive. This way, we
                // don't spend any more time than necessary decoding irrelevant instructions.
                if inst_ref.to_words().get(1).copied() != Some(id_ref.0 .0) {
                    return None;
                }

                match inst_ref.opcode() {
                    OP_TYPE_BOOL_OPCODE => panic!("can't put booleans in structs"),
                    OP_TYPE_INT_OPCODE => {
                        let op_type_int = OpTypeInt::decode(*inst_ref).unwrap();
                        let type_int = TypeInt::new(&op_type_int);

                        Some(Type::Scalar(TypeScalar::Int(type_int)))
                    }
                    OP_TYPE_FLOAT_OPCODE => {
                        let op_type_float = OpTypeFloat::decode(*inst_ref).unwrap();
                        let type_float = TypeFloat::new(&op_type_float);

                        Some(Type::Scalar(TypeScalar::Float(type_float)))
                    }
                    OP_TYPE_POINTER_OPCODE => {
                        let _op_type_pointer = OpTypePointer::decode(*inst_ref).unwrap();
                        let type_pointer = TypePointer::new();

                        Some(Type::Pointer(type_pointer))
                    }
                    OP_TYPE_VECTOR_OPCODE => {
                        let op_type_vector = OpTypeVector::decode(*inst_ref).unwrap();
                        let type_vector = TypeVector::new(shader, &op_type_vector);

                        Some(Type::Vector(type_vector))
                    }
                    OP_TYPE_MATRIX_OPCODE => {
                        let op_type_matrix = OpTypeMatrix::decode(*inst_ref).unwrap();
                        let type_matrix = TypeMatrix::new(shader, &op_type_matrix);

                        Some(Type::Matrix(type_matrix))
                    }
                    OP_TYPE_ARRAY_OPCODE => {
                        let op_type_array = OpTypeArray::decode(*inst_ref).unwrap();
                        let type_array = TypeArray::new(shader, &op_type_array);

                        Some(Type::Array(type_array))
                    }
                    OP_TYPE_RUNTIME_ARRAY_OPCODE => {
                        let op_type_runtime_array = OpTypeRuntimeArray::decode(*inst_ref).unwrap();
                        let type_array = TypeArray::new_runtime(shader, &op_type_runtime_array);

                        Some(Type::Array(type_array))
                    }
                    OP_TYPE_STRUCT_OPCODE => {
                        let op_type_struct = OpTypeStruct::decode(*inst_ref).unwrap();
                        let type_struct = TypeStruct::new(shader, &op_type_struct);

                        Some(Type::Struct(type_struct))
                    }
                    _ => None,
                }
            })
            .unwrap()
    }

    fn size(&self) -> Option<usize> {
        match self {
            Self::Scalar(ty) => Some(ty.size()),
            Self::Pointer(ty) => Some(ty.size()),
            Self::Vector(ty) => Some(ty.size()),
            Self::Matrix(ty) => Some(ty.size()),
            Self::Array(ty) => ty.size(),
            Self::Struct(ty) => ty.size(),
        }
    }

    fn ident(&self) -> Option<Ident> {
        match self {
            Self::Scalar(TypeScalar::Int(ty)) => Some(ty.to_ident()),
            Self::Scalar(TypeScalar::Float(ty)) => Some(ty.to_ident()),
            Self::Pointer(_) => None,
            Self::Vector(ty) => {
                let scalar = Ident::new(
                    match &ty.component_type {
                        TypeScalar::Int(i) => i.as_str(),
                        TypeScalar::Float(f) => f.as_str(),
                    },
                    Span::call_site(),
                );
                Some(format_ident!(
                    "vec{}_{}",
                    ty.component_count as usize,
                    scalar
                ))
            }
            Self::Matrix(ty) => Some(format_ident!(
                "mat{}x{}",
                ty.column_count as usize,
                ty.row_count as usize,
            )),
            Self::Array(ty) => ty.element_type.ident(),
            Self::Struct(ty) => Some(ty.ident.clone()),
        }
    }

    fn scalar_alignment(&self) -> Alignment {
        match self {
            Self::Scalar(ty) => ty.alignment(),
            Self::Pointer(ty) => ty.alignment(),
            Self::Vector(ty) => ty.component_type.alignment(),
            Self::Matrix(ty) => ty.component_type.alignment(),
            Self::Array(ty) => ty.scalar_alignment(),
            Self::Struct(ty) => ty.scalar_alignment(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TypeScalar {
    Int(TypeInt),
    Float(TypeFloat),
}

impl TypeScalar {
    fn size(&self) -> usize {
        match self {
            Self::Int(ty) => ty.size(),
            Self::Float(ty) => ty.size(),
        }
    }

    fn alignment(&self) -> Alignment {
        match self {
            Self::Int(ty) => ty.alignment(),
            Self::Float(ty) => ty.alignment(),
        }
    }

    #[track_caller]
    fn unwrap_float(self) -> TypeFloat {
        match self {
            TypeScalar::Int(_) => {
                panic!("called `TypeScalar::unwrap_float` on an `Int` value")
            }
            TypeScalar::Float(type_float) => type_float,
        }
    }
}

impl ToTokens for TypeScalar {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Int(ty) => ty.to_tokens(tokens),
            Self::Float(ty) => ty.to_tokens(tokens),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeInt {
    width: IntWidth,
    signed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum IntWidth {
    W8 = 8,
    W16 = 16,
    W32 = 32,
    W64 = 64,
}

impl TypeInt {
    fn new(op_type_int: &OpTypeInt) -> Self {
        let width = match op_type_int.width.to_u32() {
            8 => IntWidth::W8,
            16 => IntWidth::W16,
            32 => IntWidth::W32,
            64 => IntWidth::W64,
            _ => unreachable!(),
        };

        let signed = match op_type_int.signedness.to_u32() {
            0 => false,
            1 => true,
            _ => unreachable!(),
        };

        TypeInt { width, signed }
    }

    fn size(&self) -> usize {
        self.width as usize >> 3
    }

    fn alignment(&self) -> Alignment {
        Alignment::new(self.size())
    }

    #[rustfmt::skip]
    fn as_str(&self) -> &'static str {
        match (self.width, self.signed) {
            (IntWidth::W8,  false) => "u8",
            (IntWidth::W16, false) => "u16",
            (IntWidth::W32, false) => "u32",
            (IntWidth::W64, false) => "u64",
            (IntWidth::W8,  true)  => "i8",
            (IntWidth::W16, true)  => "i16",
            (IntWidth::W32, true)  => "i32",
            (IntWidth::W64, true)  => "i64",
        }
    }

    fn to_ident(&self) -> Ident {
        Ident::new(self.as_str(), Span::call_site())
    }
}

impl ToTokens for TypeInt {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.to_ident());
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeFloat {
    width: FloatWidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum FloatWidth {
    W16 = 16,
    W32 = 32,
    W64 = 64,
}

impl TypeFloat {
    fn new(op_type_float: &OpTypeFloat) -> Self {
        let width = match op_type_float.width.to_u32() {
            16 => FloatWidth::W16,
            32 => FloatWidth::W32,
            64 => FloatWidth::W64,
            _ => unreachable!(),
        };

        TypeFloat { width }
    }

    fn size(&self) -> usize {
        self.width as usize >> 3
    }

    fn alignment(&self) -> Alignment {
        Alignment::new(self.size())
    }

    fn as_str(&self) -> &'static str {
        match self.width {
            FloatWidth::W16 => "f16",
            FloatWidth::W32 => "f32",
            FloatWidth::W64 => "f64",
        }
    }

    fn to_ident(&self) -> Ident {
        Ident::new(self.as_str(), Span::call_site())
    }
}

impl ToTokens for TypeFloat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.width == FloatWidth::W16 {
            tokens.extend(quote! { ::vulkano::half:: });
        }
        tokens.append(self.to_ident());
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypePointer;

impl TypePointer {
    fn new() -> Self {
        TypePointer
    }

    fn size(&self) -> usize {
        8
    }

    fn alignment(&self) -> Alignment {
        Alignment::A8
    }
}

impl ToTokens for TypePointer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(quote! { ::vulkano::DeviceAddress });
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeVector {
    component_type: TypeScalar,
    component_count: ComponentCount,
}

impl TypeVector {
    fn new(shader: &Shader<'_>, op_type_vector: &OpTypeVector) -> Self {
        let component_count = ComponentCount::new(op_type_vector.component_count.to_u32());

        let component_type = shader
            .insts
            .iter_ref()
            .find_map(|inst_ref| {
                // For the below instructions, the Result ID is always the first operand. So if
                // this word doesn't match the type we're searching for, it can't be it (a true
                // negative), so false negatives can't happen. There is the possibility of false
                // positives if it's not one of the below types, but worst case, we fall through in
                // the below code, which results in a true negative again. Otherwise, both the
                // Result ID and opcode must match, so we end up with a true positive. This way, we
                // don't spend any more time than necessary decoding irrelevant instructions.
                if inst_ref.to_words().get(1).copied() != Some(op_type_vector.component_type.0 .0) {
                    return None;
                }

                match inst_ref.opcode() {
                    OP_TYPE_BOOL_OPCODE => panic!("can't put booleans in structs"),
                    OP_TYPE_INT_OPCODE => {
                        let op_type_int = OpTypeInt::decode(*inst_ref).unwrap();
                        let type_int = TypeInt::new(&op_type_int);

                        Some(TypeScalar::Int(type_int))
                    }
                    OP_TYPE_FLOAT_OPCODE => {
                        let op_type_float = OpTypeFloat::decode(*inst_ref).unwrap();
                        let type_float = TypeFloat::new(&op_type_float);

                        Some(TypeScalar::Float(type_float))
                    }
                    _ => None,
                }
            })
            .unwrap();

        TypeVector {
            component_type,
            component_count,
        }
    }

    fn size(&self) -> usize {
        self.component_type.size() * self.component_count as usize
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeMatrix {
    component_type: TypeFloat,
    column_count: ComponentCount,
    row_count: ComponentCount,
    stride: usize,
    majorness: MatrixMajorness,
}

impl TypeMatrix {
    fn new(shader: &Shader<'_>, op_type_matrix: &OpTypeMatrix) -> Self {
        let column_count = ComponentCount::new(op_type_matrix.column_count.to_u32());

        let column_type = shader
            .insts
            .iter_inst::<OpTypeVector>()
            .find_map(|op_type_vector| {
                (op_type_vector.id_result == op_type_matrix.column_type.0)
                    .then(|| TypeVector::new(shader, &op_type_vector))
            })
            .unwrap();

        let component_type = column_type.component_type.unwrap_float();
        let row_count = column_type.component_count;

        // We can't know these until we get to the members and their decorations, so just use
        // defaults for now.
        let stride = component_type.size() * row_count as usize;
        let majorness = MatrixMajorness::ColumnMajor;

        TypeMatrix {
            component_type,
            column_count,
            row_count,
            stride,
            majorness,
        }
    }

    fn size(&self) -> usize {
        self.stride * self.vector_count() as usize
    }

    fn vector_size(&self) -> usize {
        self.component_type.size() * self.component_count() as usize
    }

    fn vector_count(&self) -> ComponentCount {
        match self.majorness {
            MatrixMajorness::ColumnMajor => self.column_count,
            MatrixMajorness::RowMajor => self.row_count,
        }
    }

    fn component_count(&self) -> ComponentCount {
        match self.majorness {
            MatrixMajorness::ColumnMajor => self.row_count,
            MatrixMajorness::RowMajor => self.column_count,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixMajorness {
    ColumnMajor,
    RowMajor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum ComponentCount {
    C2 = 2,
    C3 = 3,
    C4 = 4,
}

impl ComponentCount {
    fn new(count: u32) -> Self {
        match count {
            2 => ComponentCount::C2,
            3 => ComponentCount::C3,
            4 => ComponentCount::C4,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeArray {
    element_type: Box<Type>,
    length: Option<NonZero<usize>>,
    stride: usize,
}

impl TypeArray {
    fn new(shader: &Shader<'_>, op_type_array: &OpTypeArray) -> Self {
        let length = shader
            .insts
            .iter_inst::<OpConstant>()
            .find_map(|op_constant| {
                (op_constant.id_result == op_type_array.length.0).then(|| {
                    let len = op_constant.value.as_u64().unwrap();

                    NonZero::new(len.try_into().unwrap()).unwrap()
                })
            })
            .unwrap();

        Self::new_inner(
            shader,
            op_type_array.id_result,
            op_type_array.element_type,
            Some(length),
        )
    }

    fn new_runtime(shader: &Shader<'_>, op_type_runtime_array: &OpTypeRuntimeArray) -> Self {
        Self::new_inner(
            shader,
            op_type_runtime_array.id_result,
            op_type_runtime_array.element_type,
            None,
        )
    }

    fn new_inner(
        shader: &Shader<'_>,
        id_result: IdResult,
        element_type_id_ref: IdRef,
        length: Option<NonZero<usize>>,
    ) -> Self {
        let element_type = Box::new(Type::new(shader, element_type_id_ref));

        let stride = shader
            .insts
            .iter_inst::<OpDecorate>()
            .filter(|op_decorate| op_decorate.target.0 == id_result)
            .find_map(|op_decorate| match op_decorate.decoration {
                Decoration::ArrayStride(array_stride) => Some(array_stride.0 .0 as usize),
                _ => None,
            })
            .unwrap();

        TypeArray {
            element_type,
            length,
            stride,
        }
    }

    fn size(&self) -> Option<usize> {
        self.length.map(|length| self.stride * length.get())
    }

    fn scalar_alignment(&self) -> Alignment {
        self.element_type.scalar_alignment()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeStruct {
    ident: Ident,
    members: Vec<Member>,
}

impl TypeStruct {
    fn new(shader: &Shader<'_>, op_type_struct: &OpTypeStruct) -> Self {
        let ident = shader
            .insts
            .iter_inst::<OpName>()
            .find_map(|op_name| {
                (op_name.target.0 == op_type_struct.id_result)
                    .then(|| name_to_ident(op_name.name.as_str()))
            })
            .unwrap_or_else(|| format_ident!("Unnamed{}", op_type_struct.id_result.0 .0));

        let mut members = Vec::<Member>::with_capacity(op_type_struct.id_ref.len());

        for (member_index, &member_id) in op_type_struct.id_ref.iter().enumerate() {
            let member_index = member_index as u32;

            let ident = shader
                .insts
                .iter_inst::<OpMemberName>()
                .find_map(|op_member_name| {
                    (op_member_name.ty.0 == op_type_struct.id_result
                        && op_member_name.member.0 .0 == member_index)
                        .then(|| name_to_ident(op_member_name.name.as_str()))
                })
                .unwrap_or_else(|| format_ident!("unnamed{member_index}"));

            let mut ty = Type::new(shader, member_id);

            {
                // If the member is an array, then matrix decorations can be applied to it if the
                // innermost type of the array is a matrix. Else, this will stay being the type of
                // the member.
                let mut ty = &mut ty;
                while let Type::Array(TypeArray { element_type, .. }) = ty {
                    ty = element_type;
                }

                if let Type::Matrix(matrix) = ty {
                    matrix.stride = shader
                        .insts
                        .iter_inst::<OpMemberDecorate>()
                        .filter(|op_member_decorate| {
                            op_member_decorate.structure_type.0 == op_type_struct.id_result
                                && op_member_decorate.member.0 .0 == member_index
                        })
                        .find_map(|op_member_decorate| match op_member_decorate.decoration {
                            Decoration::MatrixStride(matrix_stride) => {
                                Some(matrix_stride.0 .0 as usize)
                            }
                            _ => None,
                        })
                        .unwrap();

                    matrix.majorness = shader
                        .insts
                        .iter_inst::<OpMemberDecorate>()
                        .filter(|op_member_decorate| {
                            op_member_decorate.structure_type.0 == op_type_struct.id_result
                                && op_member_decorate.member.0 .0 == member_index
                        })
                        .find_map(|op_member_decorate| match op_member_decorate.decoration {
                            Decoration::ColMajor => Some(MatrixMajorness::ColumnMajor),
                            Decoration::RowMajor => Some(MatrixMajorness::RowMajor),
                            _ => None,
                        })
                        .unwrap();
                }
            }

            let offset = shader
                .insts
                .iter_inst::<OpMemberDecorate>()
                .filter(|op_member_decorate| {
                    op_member_decorate.structure_type.0 == op_type_struct.id_result
                        && op_member_decorate.member.0 .0 == member_index
                })
                .find_map(|op_member_decorate| match op_member_decorate.decoration {
                    Decoration::Offset(offset) => Some(offset.0 .0 as usize),
                    _ => None,
                })
                .unwrap();

            members.push(Member { ident, ty, offset });
        }

        let ident = disambiguate_structured_buffer(&ident, &members).unwrap_or(ident);

        TypeStruct { ident, members }
    }

    fn size(&self) -> Option<usize> {
        self.members
            .last()
            .map(|member| {
                member
                    .ty
                    .size()
                    .map(|size| align_up(member.offset + size, self.scalar_alignment()))
            })
            .unwrap_or(Some(0))
    }

    fn scalar_alignment(&self) -> Alignment {
        self.members
            .iter()
            .map(|member| member.ty.scalar_alignment())
            .max()
            .unwrap_or(Alignment::A1)
    }

    fn is_bindless_id(&self) -> bool {
        self.members.len() == 2
            && self.members.iter().all(|member| {
                matches!(
                    member.ty,
                    Type::Scalar(TypeScalar::Int(TypeInt {
                        width: IntWidth::W32,
                        signed: false,
                    })),
                )
            })
            && matches!(
                self.ident.to_string().as_str(),
                "SamplerId"
                    | "SampledImageId"
                    | "StorageImageId"
                    | "StorageBufferId"
                    | "AccelerationStructureId",
            )
    }
}

fn name_to_ident(name: &str) -> Ident {
    // Replace chars that could potentially cause the ident to be invalid with "_". For example,
    // Rust-GPU names structs by their fully qualified Rust name (e.g., "foo::bar::MyStruct"), in
    // which the ":" is an invalid character for idents.
    let mut name = name.replace(|c: char| !(c.is_ascii_alphanumeric() || c == '_'), "_");

    if name.starts_with(|c: char| !(c.is_ascii_alphabetic() || c == '_')) {
        name.insert(0, '_');
    }

    Ident::new(&name, Span::call_site())
}

fn disambiguate_structured_buffer(ident: &Ident, members: &[Member]) -> Option<Ident> {
    let name = ident.to_string();

    if !(name.starts_with("StructuredBuffer") || name.starts_with("RWStructuredBuffer"))
        || members.len() != 1
    {
        return None;
    }

    let Type::Array(TypeArray { element_type, .. }) = &members[0].ty else {
        return None;
    };

    let element_ident = element_type.ident()?;

    Some(format_ident!(
        "{ident}_{element_ident}",
        span = ident.span()
    ))
}

#[derive(Clone, Debug)]
struct Member {
    ident: Ident,
    ty: Type,
    offset: usize,
}

impl PartialEq for Member {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty && self.offset == other.offset
    }
}

impl Eq for Member {}

trait InstSliceExt {
    fn iter_inst<T: Inst + InstEncoding>(&self) -> impl Iterator<Item = T>;
}

impl InstSliceExt for InstSlice<CoreInstSet> {
    fn iter_inst<T: Inst + InstEncoding>(&self) -> impl Iterator<Item = T> {
        self.iter_ref()
            .filter(|inst_ref| inst_ref.opcode() == T::META.opcode)
            .map(|inst_ref| T::decode(*inst_ref).unwrap())
    }
}

/// Helper for serializing a type to tokens with respect to macro options.
struct Serializer<'a, T>(&'a T, &'a MacroOptions);

impl ToTokens for Serializer<'_, Type> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match &self.0 {
            Type::Scalar(ty) => ty.to_tokens(tokens),
            Type::Pointer(ty) => ty.to_tokens(tokens),
            Type::Vector(ty) => Serializer(ty, self.1).to_tokens(tokens),
            Type::Matrix(ty) => Serializer(ty, self.1).to_tokens(tokens),
            Type::Array(ty) => Serializer(ty, self.1).to_tokens(tokens),
            Type::Struct(ty) => {
                if ty.is_bindless_id() {
                    tokens.extend(quote! { ::vulkano_taskgraph::descriptor_set:: });
                }

                tokens.append(ty.ident.clone());
            }
        }
    }
}

impl ToTokens for Serializer<'_, TypeVector> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let component_type = &self.0.component_type;
        let component_count = self.0.component_count as usize;

        match self.1.linalg_type {
            LinAlgType::Std => {
                tokens.extend(quote! { [#component_type; #component_count] });
            }
            LinAlgType::CgMath => {
                let vector = format_ident!("Vector{}", component_count);
                tokens.extend(quote! { ::cgmath::#vector<#component_type> });
            }
            LinAlgType::Nalgebra => {
                tokens.extend(quote! {
                    ::nalgebra::base::SVector<#component_type, #component_count>
                });
            }
        }
    }
}

impl ToTokens for Serializer<'_, TypeMatrix> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let component_type = &self.0.component_type;
        let vector_count = self.0.vector_count() as usize;
        let component_count = self.0.component_count() as usize;
        let majorness = self.0.majorness;

        // This can't overflow because the stride must be at least the vector size.
        let padding = self.0.stride - self.0.vector_size();

        match self.1.linalg_type {
            // cgmath only has column-major matrices. It also only has square matrices, and its 3x3
            // matrix is not padded right. Fall back to std for anything else.
            LinAlgType::CgMath
                if majorness == MatrixMajorness::ColumnMajor
                    && padding == 0
                    && vector_count == component_count =>
            {
                let matrix = format_ident!("Matrix{}", component_count);
                tokens.extend(quote! { ::cgmath::#matrix<#component_type> });
            }
            // nalgebra only has column-major matrices, and its 3xN matrices are not padded right.
            // Fall back to std for anything else.
            LinAlgType::Nalgebra if majorness == MatrixMajorness::ColumnMajor && padding == 0 => {
                tokens.extend(quote! {
                    ::nalgebra::base::SMatrix<#component_type, #component_count, #vector_count>
                });
            }
            _ => {
                let vector = Padded(quote! { [#component_type; #component_count] }, padding);
                tokens.extend(quote! { [#vector; #vector_count] });
            }
        }
    }
}

impl ToTokens for Serializer<'_, TypeArray> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let element_type = &*self.0.element_type;
        // This can't panic because array elements must be sized.
        let element_size = element_type.size().unwrap();
        // This can't overflow because the stride must be at least the element size.
        let padding = self.0.stride - element_size;

        let element_type = Padded(Serializer(element_type, self.1), padding);

        if let Some(length) = self.0.length.map(NonZero::get) {
            tokens.extend(quote! { [#element_type; #length] });
        } else {
            tokens.extend(quote! { [#element_type] });
        }
    }
}

impl ToTokens for Serializer<'_, TypeStruct> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let struct_ident = &self.0.ident;
        let member_idents = self.0.members.iter().map(|member| &member.ident);
        let mut member_types = Vec::new();

        // TODO: Replace with the `ArrayWindows` iterator once it is stabilized.
        for member_pair in self.0.members.windows(2) {
            let member = &member_pair[0];
            let next_member = &member_pair[1];

            let offset = member.offset;
            // This can't panic, because only the last member can be unsized.
            let size = member.ty.size().unwrap();
            let next_offset = next_member.offset;
            let next_alignment = next_member.ty.scalar_alignment();

            let ser = Serializer(&member.ty, self.1);

            // Wrap the member in `Padded` in order to pad the next member, if required. Note that
            // the next offset can't be larger than the aligned offset, and all offsets must be
            // aligned to the scalar alignment of its member's type.
            //
            // NOTE(Marc): The next offset must also be aligned to the scalar alignment of the
            // previous member's type. What this prevents is a situation like this:
            //
            // | 0x0 |     |     |     |     | 0x5 |
            // |-----------------------|-----|-----|
            // | u32                   |     | u8  |
            //
            // We can't pad the `u8` using a `Padded<u32, N>`, because its size will always be
            // rounded up to the nearest multiple of its alignment which is 4 in this case. We
            // could pad the `u8` field if `Padded` also had a const parameter for the number of
            // leading padding bytes, but that would make the logic here more complicated than I
            // feel is necessary for now. We can always add this functionality later if someone's
            // life depends on it.
            //
            // The reason it makes more sense to have trailing padding than leading padding is:
            //
            // 1. The above can only be achieved by setting the offset explicitly, and makes little
            //    sense from a practical standpoint. The compiler by default only adds padding if
            //    the following field has a *higher alignment*.
            // 2. Arrays and by extension matrices need trailing padding to satisfy their strides.
            //
            // For now, the first member must also start at offset 0, for simplicity. This can be
            // adjusted in the future also by adding a parameter for leading padding.
            if align_up(offset + size, next_alignment) < next_offset {
                member_types.push(Padded(ser, next_offset - offset - size).into_token_stream());
            } else {
                member_types.push(ser.into_token_stream());
            }
        }

        // Add the last field, which is excluded in the above loop (both if the number of members
        // is 1 and >= 2).
        if let Some(last) = self.0.members.last() {
            member_types.push(Serializer(&last.ty, self.1).into_token_stream());
        }

        tokens.extend(quote! {
            pub struct #struct_ident {
                #( pub #member_idents: #member_types, )*
            }
        })
    }
}

/// Helper for wrapping tokens in `Padded`. Doesn't wrap if the padding is `0`.
struct Padded<T>(T, usize);

impl<T: ToTokens> ToTokens for Padded<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = &self.0;
        let padding = self.1;

        if padding == 0 {
            ty.to_tokens(tokens);
        } else {
            tokens.extend(quote! { ::vulkano::padded::Padded<#ty, #padding> });
        }
    }
}
