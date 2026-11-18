use self::{
    reflected_map::{Annotation, Metadata, ReflectedMap},
    reflected_ref::{ReflectedRef, Type, TypeRef, TypeScalar, TypeScalarRef},
};
use crate::{
    arena::{slice_assume_init_mut, Arena},
    bail, LinAlgType, MacroOptions, Result,
};
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
        preamble::{self, CoreInstSet},
    },
    inst::{InstEncoding, InstRef},
    module::{ParseError, SpirvHeader},
    slice::InstSlice,
};
use std::{
    cmp::Ordering,
    fmt::{Arguments, Write},
    num::NonZero,
};
use syn::{Error, Ident, LitStr};

mod reflected_map;
mod reflected_ref;

const OP_NAME_OPCODE: u16 = preamble::OP_NAME.opcode;
const OP_MEMBER_NAME_OPCODE: u16 = preamble::OP_MEMBER_NAME.opcode;

const OP_TYPE_INT_OPCODE: u16 = preamble::OP_TYPE_INT.opcode;
const OP_TYPE_FLOAT_OPCODE: u16 = preamble::OP_TYPE_FLOAT.opcode;
const OP_TYPE_VECTOR_OPCODE: u16 = preamble::OP_TYPE_VECTOR.opcode;
const OP_TYPE_MATRIX_OPCODE: u16 = preamble::OP_TYPE_MATRIX.opcode;
const OP_TYPE_ARRAY_OPCODE: u16 = preamble::OP_TYPE_ARRAY.opcode;
const OP_TYPE_RUNTIME_ARRAY_OPCODE: u16 = preamble::OP_TYPE_RUNTIME_ARRAY.opcode;
const OP_TYPE_STRUCT_OPCODE: u16 = preamble::OP_TYPE_STRUCT.opcode;
const OP_TYPE_POINTER_OPCODE: u16 = preamble::OP_TYPE_POINTER.opcode;
const OP_CONSTANT_OPCODE: u16 = preamble::OP_CONSTANT.opcode;

const OP_DECORATE_OPCODE: u16 = preamble::OP_DECORATE.opcode;
const OP_MEMBER_DECORATE_OPCODE: u16 = preamble::OP_MEMBER_DECORATE.opcode;

/// Translates all the structs that are contained in the SPIR-V module to Rust structs.
pub(super) fn generate_structs<'global_arena>(
    options: &MacroOptions,
    lit: LitStr,
    shader_name: Option<&'global_arena str>,
    words: &[u32],
    type_registry: &mut TypeRegistry<'global_arena>,
) -> Result<TokenStream> {
    if !options.generate_structs {
        return Ok(TokenStream::new());
    }

    let arena = Arena::new();

    let mut decoder = Decoder::new(words, &arena).map_err(|err| {
        Error::new_spanned(&lit, format_args!("failed to parse SPIR-V words: {err}"))
    })?;

    for inst_ref in decoder.insts.iter_ref() {
        match inst_ref.opcode() {
            OP_NAME_OPCODE => decoder.decode_name(inst_ref),
            OP_MEMBER_NAME_OPCODE => decoder.decode_member_name(inst_ref),
            OP_TYPE_INT_OPCODE => decoder.decode_type_int(inst_ref),
            OP_TYPE_FLOAT_OPCODE => decoder.decode_type_float(inst_ref),
            OP_TYPE_VECTOR_OPCODE => decoder.decode_type_vector(inst_ref),
            OP_TYPE_MATRIX_OPCODE => decoder.decode_type_matrix(inst_ref),
            OP_TYPE_ARRAY_OPCODE => decoder.decode_type_array(inst_ref),
            OP_TYPE_RUNTIME_ARRAY_OPCODE => decoder.decode_type_runtime_array(inst_ref),
            OP_TYPE_STRUCT_OPCODE => decoder.decode_type_struct(inst_ref),
            OP_TYPE_POINTER_OPCODE => decoder.decode_type_pointer(inst_ref),
            OP_CONSTANT_OPCODE => decoder.decode_constant(inst_ref),
            OP_DECORATE_OPCODE => decoder.decode_decorate(inst_ref),
            OP_MEMBER_DECORATE_OPCODE => decoder.decode_member_decorate(inst_ref),
            _ => {}
        }
    }

    let mut structs = TokenStream::new();

    for ty in decoder
        .reflected
        .values()
        .filter_map(|reflected_ref| reflected_ref.to_type_struct())
    {
        // Register the type if needed.
        if let Some(shader_name) = shader_name {
            if !type_registry.register_struct(shader_name, ty)? {
                continue;
            }
        }

        if ty.to_bindless_id().is_some() {
            continue;
        }

        let custom_derives = if ty.size().is_some() {
            options.custom_derives.as_slice()
        } else {
            &[]
        };
        let ser = Serializer {
            ty,
            options,
            matrix_stride: None,
            matrix_majorness: None,
        };

        structs.extend(quote! {
            #[allow(non_camel_case_types, non_snake_case)]
            #[derive(::vulkano::buffer::BufferContents #(, #custom_derives )* )]
            #[repr(C)]
            #ser
        })
    }

    Ok(structs)
}

pub(super) struct TypeRegistry<'global_arena> {
    registered_structs: HashMap<&'global_arena str, RegisteredType<'global_arena>>,
    global_arena: &'global_arena Arena,
}

impl<'global_arena> TypeRegistry<'global_arena> {
    pub(crate) fn new(global_arena: &'global_arena Arena) -> Self {
        Self {
            registered_structs: HashMap::default(),
            global_arena,
        }
    }

    fn register_struct(
        &mut self,
        shader_name: &'global_arena str,
        ty: &TypeStruct<'_>,
    ) -> Result<bool> {
        // Check with the registry if this struct is already registered by another shader, and if
        // so, if their signatures match.
        if let Some(registered) = self.registered_structs.get(ty.name) {
            registered.validate_signatures(shader_name, ty)?;

            // If the struct is already registered and machtes this one, skip serializing the
            // duplicate.
            Ok(false)
        } else {
            let ty = ty.clone_to(self.global_arena);
            let registered = RegisteredType { shader_name, ty };
            self.registered_structs.insert(ty.name, registered);

            Ok(true)
        }
    }
}

struct RegisteredType<'global_arena> {
    shader_name: &'global_arena str,
    ty: &'global_arena TypeStruct<'global_arena>,
}

impl<'global_arena> RegisteredType<'global_arena> {
    fn validate_signatures(&self, other_shader_name: &str, other_ty: &TypeStruct<'_>) -> Result {
        let (shader_name, struct_name) = (self.shader_name, self.ty.name);

        if self.ty.members.len() > other_ty.members.len() {
            let member_name = self.ty.members[other_ty.members.len()].name;
            bail!(
                "shaders `{shader_name}` and `{other_shader_name}` declare structs with the same \
                name `{struct_name}`, but the struct from shader `{shader_name}` contains an \
                extra field `{member_name}`",
            );
        }

        if self.ty.members.len() < other_ty.members.len() {
            let member_name = other_ty.members[self.ty.members.len()].name;
            bail!(
                "shaders `{shader_name}` and `{other_shader_name}` declare structs with the same \
                name `{struct_name}`, but the struct from shader `{other_shader_name}` contains \
                an extra field `{member_name}`",
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
                let mut member_ty = String::new();
                member.ty().to_string(&mut member_ty);
                let mut other_member_ty = String::new();
                other_member.ty().to_string(&mut other_member_ty);
                bail!(
                    "shaders `{shader_name}` and `{other_shader_name}` declare structs with the \
                    same name `{struct_name}`, but the struct from shader `{shader_name}` \
                    contains a field of type `{member_ty:?}` at index `{index}`, whereas the same \
                    struct from shader `{other_shader_name}` contains a field of type \
                    `{other_member_ty:?}` in the same position",
                );
            }
        }

        Ok(())
    }
}

struct Decoder<'arena> {
    // This has a longer lifetime than `'arena`, but since both lifetimes are covariant and we
    // don't need to recover the original lifetime, it doesn't make a difference.
    insts: &'arena InstSlice<CoreInstSet>,
    reflected: ReflectedMap<'arena>,
    member_buffer: Vec<Member<'arena>>,
    string_buffer: String,
    arena: &'arena Arena,
}

impl<'arena> Decoder<'arena> {
    fn new(words: &'arena [u32], arena: &'arena Arena) -> Result<Self, ParseError> {
        let (header, insts) = words.split_at(5);
        let header = SpirvHeader::from_slice(header);
        header.validate()?;
        let insts = InstSlice::from_words(insts)?;

        let inst_count_estimate = words.len() / 8;

        Ok(Decoder {
            insts,
            reflected: ReflectedMap::with_capacity(inst_count_estimate / 8, arena),
            member_buffer: Vec::with_capacity(16),
            string_buffer: String::with_capacity(32),
            arena,
        })
    }

    fn decode_name(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_name) = OpName::decode(*inst_ref) else {
            unreachable!();
        };

        self.reflected
            .push_name(op_name.target, op_name.name.as_str());
    }

    fn decode_member_name(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_member_name) = OpMemberName::decode(*inst_ref) else {
            unreachable!();
        };

        self.reflected.push_member_name(
            op_member_name.ty,
            op_member_name.member.to_u32(),
            op_member_name.name.as_str(),
        );
    }

    fn decode_decorate(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_decorate) = OpDecorate::decode(*inst_ref) else {
            unreachable!();
        };

        let Some(annotation) = Annotation::from_decoration(op_decorate.decoration) else {
            return;
        };

        self.reflected
            .push_annotation(op_decorate.target, annotation);
    }

    fn decode_member_decorate(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_member_decorate) = OpMemberDecorate::decode(*inst_ref) else {
            unreachable!();
        };

        let Some(annotation) = Annotation::from_decoration(op_member_decorate.decoration) else {
            return;
        };

        self.reflected.push_member_annotation(
            op_member_decorate.structure_type,
            op_member_decorate.member.to_u32(),
            annotation,
        );
    }

    fn decode_type_int(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_int) = OpTypeInt::decode(*inst_ref) else {
            unreachable!();
        };

        let type_int = self.arena.allocate_sized();

        let width = match op_type_int.width.to_u32() {
            8 => IntWidth::W8,
            16 => IntWidth::W16,
            32 => IntWidth::W32,
            64 => IntWidth::W64,
            // This can't panic because int width must be 8, 16, 32 or 64.
            _ => unreachable!(),
        };

        let signed = match op_type_int.signedness.to_u32() {
            0 => false,
            1 => true,
            // This can't panic because int signedness must be 0 or 1.
            _ => unreachable!(),
        };

        let type_int = type_int.write(TypeInt { width, signed });
        self.reflected.insert(
            op_type_int.id_result,
            TypeRef::new(type_int).to_reflected_ref(),
        );
    }

    fn decode_type_float(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_float) = OpTypeFloat::decode(*inst_ref) else {
            unreachable!();
        };

        let type_float = self.arena.allocate_sized();

        let width = match op_type_float.width.to_u32() {
            16 => FloatWidth::W16,
            32 => FloatWidth::W32,
            64 => FloatWidth::W64,
            // This can't panic because float width must be 16, 32 or 64.
            _ => unreachable!(),
        };

        let type_float = type_float.write(TypeFloat { width });
        self.reflected.insert(
            op_type_float.id_result,
            TypeRef::new(type_float).to_reflected_ref(),
        );
    }

    fn decode_type_pointer(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_pointer) = OpTypePointer::decode(*inst_ref) else {
            unreachable!();
        };

        let type_pointer = &TypePointer;
        self.reflected.insert(
            op_type_pointer.id_result,
            TypeRef::new(type_pointer).to_reflected_ref(),
        );
    }

    fn decode_type_vector(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_vector) = OpTypeVector::decode(*inst_ref) else {
            unreachable!();
        };

        // We only decode the types we care about, and a vector can contain a bool type, so we have
        // to skip decoding this if we skipped decoding the component type. There's no other way
        // this can fail: operands must be declared before being used.
        let Some(component_type) = self.reflected.get(op_type_vector.component_type) else {
            return;
        };

        let type_vector = self.arena.allocate_sized();

        // This can't panic because vector component types must be scalars.
        let component_type = component_type.to_type_scalar_ref().unwrap();

        let component_count = ComponentCount::new(op_type_vector.component_count.to_u32());

        let type_vector = type_vector.write(TypeVector {
            component_type,
            component_count,
        });
        self.reflected.insert(
            op_type_vector.id_result,
            TypeRef::new(type_vector).to_reflected_ref(),
        );
    }

    fn decode_type_matrix(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_matrix) = OpTypeMatrix::decode(*inst_ref) else {
            unreachable!();
        };

        let type_matrix = self.arena.allocate_sized();

        // These two can't panic because matrix column types must be vectors and operands must be
        // declared before being used.
        let column_type = self.reflected.get(op_type_matrix.column_type).unwrap();
        let column_type = column_type.to_type_vector().unwrap();

        let component_type = match column_type.component_type() {
            TypeScalar::Float(ty) => ty,
            // This can't panic because matrix component types must be floats.
            _ => unreachable!(),
        };

        let column_count = ComponentCount::new(op_type_matrix.column_count.to_u32());

        let row_count = column_type.component_count;

        let type_matrix = type_matrix.write(TypeMatrix {
            component_type,
            column_count,
            row_count,
        });
        self.reflected.insert(
            op_type_matrix.id_result,
            TypeRef::new(type_matrix).to_reflected_ref(),
        );
    }

    fn decode_type_array(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_array) = OpTypeArray::decode(*inst_ref) else {
            unreachable!();
        };

        let Some(metadata_list) = self.reflected.metadata_list(op_type_array.id_result) else {
            return;
        };

        // We only decode the types we care about, and an array can contain an arbitrary element
        // type, so we have to skip decoding this if we skipped decoding the element type. There's
        // no other way this can fail: operands must be declared before being used.
        let Some(element_type) = self.reflected.get(op_type_array.element_type) else {
            return;
        };

        // This can't panic because array elements must be types.
        let element_type = element_type.to_type_ref().unwrap();

        let Some(stride) = metadata_list.find_array_stride() else {
            return;
        };

        let type_array = self.arena.allocate_sized();

        // These two can't panic because array lengths must be constants and operands must be
        // declared before being used.
        let length_constant = self.reflected.get(op_type_array.length).unwrap();
        let length_constant = length_constant.to_constant().unwrap();

        // This can't panic because arrays must have a length of at least 1.
        let length = NonZero::new(length_constant.value).unwrap();

        let type_array = type_array.write(TypeArray {
            element_type,
            length: Some(length),
            stride,
        });
        self.reflected.insert(
            op_type_array.id_result,
            TypeRef::new(type_array).to_reflected_ref(),
        );
    }

    fn decode_type_runtime_array(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_runtime_array) = OpTypeRuntimeArray::decode(*inst_ref) else {
            unreachable!();
        };

        let Some(metadata_list) = self
            .reflected
            .metadata_list(op_type_runtime_array.id_result)
        else {
            return;
        };

        // We only decode the types we care about, and an array can contain an arbitrary element
        // type, so we have to skip decoding this if we skipped decoding the element type. There's
        // no other way this can fail: operands must be declared before being used.
        let Some(element_type) = self.reflected.get(op_type_runtime_array.element_type) else {
            return;
        };

        // This can't panic because array elements must be types.
        let element_type = element_type.to_type_ref().unwrap();

        let Some(stride) = metadata_list.find_array_stride() else {
            return;
        };

        let type_array = self.arena.allocate_sized();
        let type_array = type_array.write(TypeArray {
            element_type,
            length: None,
            stride,
        });
        self.reflected.insert(
            op_type_runtime_array.id_result,
            TypeRef::new(type_array).to_reflected_ref(),
        );
    }

    fn decode_type_struct(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_type_struct) = OpTypeStruct::decode(*inst_ref) else {
            unreachable!();
        };

        let Some(metadata_list) = self.reflected.metadata_list(op_type_struct.id_result) else {
            return;
        };

        let mut name = "";

        self.member_buffer.clear();

        for &member_ty in &op_type_struct.id_ref {
            // We only decode the types we care about, and a member can have an arbitrary type, so
            // we have to skip decoding the struct if we skipped decoding a member's type. There's
            // no other way this can fail: operands must be declared before being used.
            let Some(ty) = self.reflected.get(member_ty) else {
                return;
            };

            // This can't panic because members must be types.
            let ty = ty.to_type_ref().unwrap();

            self.member_buffer.push(Member {
                name: "",
                ty,
                offset: u32::MAX,
                matrix_stride: None,
                matrix_majorness: None,
            });
        }

        for metadata in metadata_list.iter() {
            match metadata {
                Metadata::Name(val) => {
                    name = *val;
                }
                Metadata::MemberName(member_index, val) => {
                    let member = &mut self.member_buffer[*member_index as usize];
                    member.name = *val;
                }
                Metadata::Annotation(_) => {}
                Metadata::MemberAnnotation(member_index, annotation) => {
                    let member = &mut self.member_buffer[*member_index as usize];

                    match annotation {
                        Annotation::Offset(val) => {
                            member.offset = *val;
                        }
                        Annotation::BuiltIn => {
                            // Ignore the whole struct if a member is built in, which includes
                            // `gl_Position` for example.
                            return;
                        }
                        Annotation::ArrayStride(_) => {}
                        Annotation::MatrixStride(val) => {
                            member.matrix_stride = Some(*val);
                        }
                        Annotation::MatrixMajorness(val) => {
                            member.matrix_majorness = Some(*val);
                        }
                    }
                }
            }
        }

        for member in &self.member_buffer {
            if member.offset == u32::MAX {
                // Some structs don't have `Offset` decorations. In that case, they are used as
                // local variables only. Ignoring these.
                return;
            }
        }

        let type_struct = self.arena.allocate_sized();
        let members = self.arena.allocate_slice_copy(&self.member_buffer);

        let name = if !name.is_empty() {
            self.sanitize_name(name)
        } else {
            self.format(format_args!("Unnamed{}", op_type_struct.id_result.as_u32()))
        };
        let name = self
            .disambiguate_structured_buffer(name, members)
            .unwrap_or(name);

        for (member_index, member) in members.iter_mut().enumerate() {
            member.name = if !member.name.is_empty() {
                self.sanitize_name(member.name)
            } else {
                self.format(format_args!("unnamed{member_index}"))
            };
        }

        let type_struct = type_struct.write(TypeStruct { name, members });
        self.reflected.insert(
            op_type_struct.id_result,
            TypeRef::new(type_struct).to_reflected_ref(),
        );
    }

    fn sanitize_name(&mut self, name: &'arena str) -> &'arena str {
        self.string_buffer.clear();

        if name.starts_with(|c: char| !(c.is_ascii_alphabetic() || c == '_')) {
            self.string_buffer.push('_');
        }

        let mut prev_end = 0;

        // Replace chars that could potentially cause the ident to be invalid with "_". For
        // example, Rust-GPU names structs by their fully qualified Rust name (e.g.,
        // "foo::bar::MyStruct"), in which the ":" is an invalid character for idents.
        for (start, matched_str) in
            name.match_indices(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        {
            // SAFETY: `str::match_indices` produces valid indices.
            let unmatched_str = unsafe { name.get_unchecked(prev_end..start) };

            self.string_buffer.push_str(unmatched_str);
            self.string_buffer.push('_');
            prev_end = start + matched_str.len();
        }

        // If nothing was written to the buffer, then the original str is already a valid ident, so
        // we can simply return the original str.
        if self.string_buffer.is_empty() {
            return name;
        }

        // SAFETY: `str::match_indices` produces valid indices.
        let unmatched_str = unsafe { name.get_unchecked(prev_end..name.len()) };

        self.string_buffer.push_str(unmatched_str);

        self.arena.allocate_str_copy(&self.string_buffer)
    }

    fn format(&mut self, args: Arguments<'_>) -> &'arena str {
        self.string_buffer.clear();
        self.string_buffer.write_fmt(args).unwrap();

        self.arena.allocate_str_copy(&self.string_buffer)
    }

    fn disambiguate_structured_buffer(
        &mut self,
        name: &str,
        members: &[Member<'_>],
    ) -> Option<&'arena str> {
        let [member] = members else {
            return None;
        };

        let Type::Array(type_array) = member.ty() else {
            return None;
        };

        let (name, layout) = match name {
            "StructuredBuffer" => ("StructuredBuffer", "std430"),
            "RWStructuredBuffer" => ("RWStructuredBuffer", "std430"),
            _ => return None,
        };

        self.string_buffer.clear();
        self.string_buffer.push_str(name);
        self.string_buffer.push('_');
        type_array.element_type().to_string(&mut self.string_buffer);
        self.string_buffer.push('_');
        self.string_buffer.push_str(layout);

        Some(self.arena.allocate_str_copy(&self.string_buffer))
    }

    fn decode_constant(&mut self, inst_ref: InstRef<'_, CoreInstSet>) {
        let Ok(op_constant) = OpConstant::decode(*inst_ref) else {
            unreachable!();
        };

        let Ok(value) = op_constant.value.as_u32() else {
            return;
        };

        let constant = self.arena.allocate_sized();
        let constant = constant.write(Constant { value });
        self.reflected
            .insert(op_constant.id_result, ReflectedRef::from_constant(constant));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum Alignment {
    A1 = 1,
    A2 = 2,
    A4 = 4,
    A8 = 8,
    A16 = 16,
    A32 = 32,
}

impl Alignment {
    fn new(alignment: u32) -> Self {
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
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Alignment {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u32).cmp(&(*other as u32))
    }
}

/// # Panics
///
/// - Panics on overflow.
fn align_up(offset: u32, alignment: Alignment) -> u32 {
    (offset.checked_add(alignment as u32 - 1).unwrap()) & !(alignment as u32 - 1)
}

fn is_aligned(offset: u32, alignment: Alignment) -> bool {
    offset & (alignment as u32 - 1) == 0
}

impl<'arena> Type<'arena> {
    fn size(
        self,
        matrix_stride: Option<NonZero<u32>>,
        matrix_majorness: Option<MatrixMajorness>,
    ) -> Option<u32> {
        match self {
            Self::Int(ty) => Some(ty.size()),
            Self::Float(ty) => Some(ty.size()),
            Self::Pointer(ty) => Some(ty.size()),
            Self::Vector(ty) => Some(ty.size()),
            Self::Matrix(ty) => Some(ty.size(matrix_stride.unwrap(), matrix_majorness.unwrap())),
            Self::Array(ty) => ty.size(),
            Self::Struct(ty) => ty.size(),
        }
    }

    fn scalar_alignment(self) -> Alignment {
        match self {
            Self::Int(ty) => ty.alignment(),
            Self::Float(ty) => ty.alignment(),
            Self::Pointer(ty) => ty.alignment(),
            Self::Vector(ty) => ty.scalar_alignment(),
            Self::Matrix(ty) => ty.scalar_alignment(),
            Self::Array(ty) => ty.scalar_alignment(),
            Self::Struct(ty) => ty.scalar_alignment(),
        }
    }

    fn to_string(self, string: &mut String) {
        match self {
            Self::Int(ty) => string.push_str(ty.as_str()),
            Self::Float(ty) => string.push_str(ty.as_str()),
            Self::Pointer(ty) => string.push_str(ty.as_str()),
            Self::Vector(ty) => ty.to_string(string),
            Self::Matrix(ty) => ty.to_string(string),
            Self::Array(ty) => ty.to_string(string),
            Self::Struct(ty) => string.push_str(ty.name),
        }
    }

    fn clone_to<'other_arena>(self, other_arena: &'other_arena Arena) -> TypeRef<'other_arena> {
        match self {
            Type::Int(ty) => TypeRef::new(ty.clone_to(other_arena)),
            Type::Float(ty) => TypeRef::new(ty.clone_to(other_arena)),
            Type::Pointer(ty) => TypeRef::new(ty.clone_to(other_arena)),
            Type::Vector(ty) => TypeRef::new(ty.clone_to(other_arena)),
            Type::Matrix(ty) => TypeRef::new(ty.clone_to(other_arena)),
            Type::Array(ty) => TypeRef::new(ty.clone_to(other_arena)),
            Type::Struct(ty) => TypeRef::new(ty.clone_to(other_arena)),
        }
    }
}

impl TypeScalar<'_> {
    fn size(self) -> u32 {
        match self {
            Self::Int(ty) => ty.size(),
            Self::Float(ty) => ty.size(),
        }
    }

    fn alignment(self) -> Alignment {
        match self {
            Self::Int(ty) => ty.alignment(),
            Self::Float(ty) => ty.alignment(),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Int(ty) => ty.as_str(),
            Self::Float(ty) => ty.as_str(),
        }
    }

    fn to_ident(self) -> Ident {
        match self {
            Self::Int(ty) => ty.to_ident(),
            Self::Float(ty) => ty.to_ident(),
        }
    }

    fn clone_to<'other_arena>(
        self,
        other_arena: &'other_arena Arena,
    ) -> TypeScalarRef<'other_arena> {
        match self {
            TypeScalar::Int(ty) => TypeScalarRef::from_type_int(ty.clone_to(other_arena)),
            TypeScalar::Float(ty) => TypeScalarRef::from_type_float(ty.clone_to(other_arena)),
        }
    }
}

impl ToTokens for TypeScalar<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match *self {
            Self::Int(ty) => ty.to_tokens(tokens),
            Self::Float(ty) => ty.to_tokens(tokens),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
#[repr(align(8))]
struct TypeInt {
    width: IntWidth,
    signed: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum IntWidth {
    W8 = 8,
    W16 = 16,
    W32 = 32,
    W64 = 64,
}

impl TypeInt {
    fn size(&self) -> u32 {
        self.width as u32 >> 3
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

    fn clone_to<'other_arena>(&self, other_arena: &'other_arena Arena) -> &'other_arena TypeInt {
        let other = other_arena.allocate_sized();

        other.write(TypeInt {
            width: self.width,
            signed: self.signed,
        })
    }
}

impl ToTokens for TypeInt {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(self.to_ident());
    }
}

#[derive(Clone, PartialEq, Eq)]
#[repr(align(8))]
struct TypeFloat {
    width: FloatWidth,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum FloatWidth {
    W16 = 16,
    W32 = 32,
    W64 = 64,
}

impl TypeFloat {
    fn size(&self) -> u32 {
        self.width as u32 >> 3
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

    fn clone_to<'other_arena>(&self, other_arena: &'other_arena Arena) -> &'other_arena TypeFloat {
        let other = other_arena.allocate_sized();

        other.write(TypeFloat { width: self.width })
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

#[derive(Clone, PartialEq, Eq)]
#[repr(align(8))]
struct TypePointer;

impl TypePointer {
    fn size(&self) -> u32 {
        8
    }

    fn alignment(&self) -> Alignment {
        Alignment::A8
    }

    fn as_str(&self) -> &'static str {
        "ptr"
    }

    fn clone_to<'other_arena>(
        &self,
        _other_arena: &'other_arena Arena,
    ) -> &'other_arena TypePointer {
        &TypePointer
    }
}

impl ToTokens for TypePointer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(quote! { ::vulkano::DeviceAddress });
    }
}

#[derive(PartialEq, Eq)]
#[repr(align(8))]
struct TypeVector<'arena> {
    component_type: TypeScalarRef<'arena>,
    component_count: ComponentCount,
}

impl<'arena> TypeVector<'arena> {
    fn component_type(&self) -> TypeScalar<'arena> {
        self.component_type.to_type_scalar()
    }

    fn size(&self) -> u32 {
        self.component_type().size() * self.component_count as u32
    }

    fn scalar_alignment(&self) -> Alignment {
        self.component_type().alignment()
    }

    fn to_string(&self, string: &mut String) {
        write!(
            string,
            "{}vec{}",
            self.component_type().as_str(),
            self.component_count as u32,
        )
        .unwrap();
    }

    fn clone_to<'other_arena>(
        &self,
        other_arena: &'other_arena Arena,
    ) -> &'other_arena TypeVector<'other_arena> {
        let other = other_arena.allocate_sized();

        other.write(TypeVector {
            component_type: self.component_type().clone_to(other_arena),
            component_count: self.component_count,
        })
    }
}

#[derive(PartialEq, Eq)]
#[repr(align(8))]
struct TypeMatrix<'arena> {
    component_type: &'arena TypeFloat,
    column_count: ComponentCount,
    row_count: ComponentCount,
}

impl<'arena> TypeMatrix<'arena> {
    fn size(&self, stride: NonZero<u32>, majorness: MatrixMajorness) -> u32 {
        let vector_count = self.vector_count(majorness) as u32;

        // This can't panic because the matrix must fit in the address space.
        stride.get().checked_mul(vector_count).unwrap()
    }

    fn vector_size(&self, majorness: MatrixMajorness) -> u32 {
        self.component_type.size() * self.component_count(majorness) as u32
    }

    fn vector_count(&self, majorness: MatrixMajorness) -> ComponentCount {
        match majorness {
            MatrixMajorness::ColMajor => self.column_count,
            MatrixMajorness::RowMajor => self.row_count,
        }
    }

    fn component_count(&self, majorness: MatrixMajorness) -> ComponentCount {
        match majorness {
            MatrixMajorness::ColMajor => self.row_count,
            MatrixMajorness::RowMajor => self.column_count,
        }
    }

    fn scalar_alignment(&self) -> Alignment {
        self.component_type.alignment()
    }

    fn to_string(&self, string: &mut String) {
        write!(
            string,
            "{}mat{}x{}",
            self.component_type.as_str(),
            self.column_count as u32,
            self.row_count as u32,
        )
        .unwrap();
    }

    fn clone_to<'other_arena>(
        &self,
        other_arena: &'other_arena Arena,
    ) -> &'other_arena TypeMatrix<'other_arena> {
        let other = other_arena.allocate_sized();

        other.write(TypeMatrix {
            component_type: self.component_type.clone_to(other_arena),
            column_count: self.column_count,
            row_count: self.row_count,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixMajorness {
    ColMajor,
    RowMajor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
            // This can't panic because a component/column count must to be between 2 and 4.
            _ => unreachable!(),
        }
    }
}

#[derive(PartialEq, Eq)]
#[repr(align(8))]
struct TypeArray<'arena> {
    element_type: TypeRef<'arena>,
    length: Option<NonZero<u32>>,
    stride: NonZero<u32>,
}

impl<'arena> TypeArray<'arena> {
    fn element_type(&self) -> Type<'arena> {
        self.element_type.to_type()
    }

    fn size(&self) -> Option<u32> {
        self.length.map(|length| {
            // This can't panic because the array must fit in the address space.
            self.stride.get().checked_mul(length.get()).unwrap()
        })
    }

    fn scalar_alignment(&self) -> Alignment {
        self.element_type().scalar_alignment()
    }

    fn to_string(&self, string: &mut String) {
        if let Some(length) = self.length {
            write!(string, "arr{length}_").unwrap();
        } else {
            write!(string, "arr_").unwrap();
        }

        self.element_type().to_string(string);
    }

    fn clone_to<'other_arena>(
        &self,
        arena_other: &'other_arena Arena,
    ) -> &'other_arena TypeArray<'other_arena> {
        let other = arena_other.allocate_sized();

        other.write(TypeArray {
            element_type: self.element_type().clone_to(arena_other),
            length: self.length,
            stride: self.stride,
        })
    }
}

#[derive(PartialEq, Eq)]
#[repr(align(8))]
struct TypeStruct<'arena> {
    name: &'arena str,
    members: &'arena [Member<'arena>],
}

impl<'arena> TypeStruct<'arena> {
    fn size(&self) -> Option<u32> {
        self.members
            .last()
            .map(|member| {
                member.size().map(|size| {
                    // This can't panic because the struct must fit in the address space.
                    let end = member.offset.checked_add(size).unwrap();

                    // This can't panic because the struct must fit in the address space.
                    align_up(end, self.scalar_alignment())
                })
            })
            .unwrap_or(Some(0))
    }

    fn scalar_alignment(&self) -> Alignment {
        self.members
            .iter()
            .map(Member::scalar_alignment)
            .max()
            .unwrap_or(Alignment::A1)
    }

    fn to_bindless_id(&self) -> Option<Ident> {
        if self.members.len() != 2 {
            return None;
        }

        if !self.members.iter().all(|member| {
            matches!(
                member.ty(),
                Type::Int(TypeInt {
                    width: IntWidth::W32,
                    signed: false,
                }),
            )
        }) {
            return None;
        }

        if matches!(
            self.name,
            "SamplerId"
                | "SampledImageId"
                | "StorageImageId"
                | "StorageBufferId"
                | "AccelerationStructureId",
        ) {
            return Some(Ident::new(self.name, Span::call_site()));
        }

        if matches!(
            self.name,
            "SamplerId_std430"
                | "SampledImageId_std430"
                | "StorageImageId_std430"
                | "StorageBufferId_std430"
                | "AccelerationStructureId_std430",
        ) {
            return Some(Ident::new(
                self.name.strip_suffix("_std430").unwrap(),
                Span::call_site(),
            ));
        }

        None
    }

    fn clone_to<'other_arena>(
        &self,
        other_arena: &'other_arena Arena,
    ) -> &'other_arena TypeStruct<'other_arena> {
        let other = other_arena.allocate_sized();
        let other_name = other_arena.allocate_str_copy(self.name);
        let other_members = other_arena.allocate_slice(self.members.len());

        for (member, other_member) in self.members.iter().zip(&mut *other_members) {
            other_member.write(Member {
                name: other_arena.allocate_str_copy(member.name),
                ty: member.ty.to_type().clone_to(other_arena),
                offset: member.offset,
                matrix_stride: member.matrix_stride,
                matrix_majorness: member.matrix_majorness,
            });
        }

        // SAFETY: We have initialized all the members above.
        let other_members = unsafe { slice_assume_init_mut(other_members) };

        other.write(TypeStruct {
            name: other_name,
            members: other_members,
        })
    }
}

struct Member<'arena> {
    name: &'arena str,
    ty: TypeRef<'arena>,
    offset: u32,
    matrix_stride: Option<NonZero<u32>>,
    matrix_majorness: Option<MatrixMajorness>,
}

impl<'arena> Member<'arena> {
    fn ty(&self) -> Type<'arena> {
        self.ty.to_type()
    }

    fn size(&self) -> Option<u32> {
        self.ty().size(self.matrix_stride, self.matrix_majorness)
    }

    fn scalar_alignment(&self) -> Alignment {
        self.ty().scalar_alignment()
    }
}

impl PartialEq for Member<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty
            && self.offset == other.offset
            && self.matrix_stride == other.matrix_stride
            && self.matrix_majorness == other.matrix_majorness
    }
}

impl Eq for Member<'_> {}

#[repr(align(8))]
struct Constant {
    value: u32,
}

/// Helper for serializing a type to tokens.
struct Serializer<'a, T> {
    ty: &'a T,
    options: &'a MacroOptions,
    matrix_stride: Option<NonZero<u32>>,
    matrix_majorness: Option<MatrixMajorness>,
}

impl<'a, T> Serializer<'a, T> {
    fn project<U>(&self, ty: &'a U) -> Serializer<'a, U> {
        Serializer {
            ty,
            options: self.options,
            matrix_stride: self.matrix_stride,
            matrix_majorness: self.matrix_majorness,
        }
    }
}

impl ToTokens for Serializer<'_, Type<'_>> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match *self.ty {
            Type::Int(ty) => ty.to_tokens(tokens),
            Type::Float(ty) => ty.to_tokens(tokens),
            Type::Pointer(ty) => ty.to_tokens(tokens),
            Type::Vector(ty) => self.project(ty).to_tokens(tokens),
            Type::Matrix(ty) => self.project(ty).to_tokens(tokens),
            Type::Array(ty) => self.project(ty).to_tokens(tokens),
            Type::Struct(ty) => {
                if let Some(ident) = ty.to_bindless_id() {
                    tokens.extend(quote! { ::vulkano_taskgraph::descriptor_set::#ident });
                } else {
                    tokens.append(Ident::new(ty.name, Span::call_site()));
                }
            }
        }
    }
}

impl ToTokens for Serializer<'_, TypeVector<'_>> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let component_type_ident = self.ty.component_type().to_ident();
        let component_count = self.ty.component_count as usize;

        match self.options.linalg_type {
            LinAlgType::Std => {
                tokens.extend(quote! { [#component_type_ident; #component_count] });
            }
            LinAlgType::CgMath => {
                let vector = format_ident!("Vector{}", component_count);
                tokens.extend(quote! { ::cgmath::#vector<#component_type_ident> });
            }
            LinAlgType::Nalgebra => {
                tokens.extend(quote! {
                    ::nalgebra::base::SVector<#component_type_ident, #component_count>
                });
            }
        }
    }
}

impl ToTokens for Serializer<'_, TypeMatrix<'_>> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // These two can't panic because we only serialize types with a defined layout.
        let stride = self.matrix_stride.unwrap().get();
        let majorness = self.matrix_majorness.unwrap();

        let component_type = self.ty.component_type;
        let vector_size = self.ty.vector_size(majorness);
        let vector_count = self.ty.vector_count(majorness) as usize;
        let component_count = self.ty.component_count(majorness) as usize;

        // This can't panic because the stride must be aligned to the component type's alignment.
        assert!(is_aligned(stride, component_type.alignment()));

        // This can't panic because the stride must be at least the vector size.
        let padding = stride.checked_sub(vector_size).unwrap();

        match self.options.linalg_type {
            // cgmath only has column-major matrices. It also only has square matrices, and its 3x3
            // matrix is not padded right. Fall back to std for anything else.
            LinAlgType::CgMath
                if majorness == MatrixMajorness::ColMajor
                    && padding == 0
                    && vector_count == component_count =>
            {
                let matrix = format_ident!("Matrix{}", component_count);
                tokens.extend(quote! { ::cgmath::#matrix<#component_type> });
            }
            // nalgebra only has column-major matrices, and its 3xN matrices are not padded right.
            // Fall back to std for anything else.
            LinAlgType::Nalgebra if majorness == MatrixMajorness::ColMajor && padding == 0 => {
                tokens.extend(quote! {
                    ::nalgebra::base::SMatrix<#component_type, #component_count, #vector_count>
                });
            }
            _ => {
                let vector = quote! { [#component_type; #component_count] };
                let vector = Padded(&vector, padding);
                tokens.extend(quote! { [#vector; #vector_count] });
            }
        }
    }
}

impl ToTokens for Serializer<'_, TypeArray<'_>> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let element_type = self.ty.element_type();

        // This can't panic because we only serialize types with a defined layout.
        let element_size = element_type.size(self.matrix_stride, self.matrix_majorness);

        // This can't panic because array elements must be sized.
        let element_size = element_size.unwrap();

        let element_alignment = element_type.scalar_alignment();

        let stride = self.ty.stride.get();

        // This can't panic because the stride must be aligned to the element type's alignment.
        assert!(is_aligned(stride, element_alignment));

        // This can't panic because the stride must be at least the element size.
        let padding = stride.checked_sub(element_size).unwrap();

        let element_type = self.project(&element_type);
        let element_type = Padded(&element_type, padding);

        if let Some(length) = self.ty.length.map(NonZero::get) {
            tokens.extend(quote! { [#element_type; #length] });
        } else {
            tokens.extend(quote! { [#element_type] });
        }
    }
}

impl ToTokens for Serializer<'_, TypeStruct<'_>> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let struct_ident = Ident::new(self.ty.name, Span::call_site());
        let mut members_code = TokenStream::new();

        if let Some(first_member) = self.ty.members.first() {
            if first_member.offset != 0 {
                unimplemented!(
                    "expected struct member 0 of struct `{}` to be at offset 0",
                    self.ty.name,
                );
            }
        }

        // TODO: Replace with the `ArrayWindows` iterator once we bump the MSRV.
        for (member_index, member_pair) in self.ty.members.windows(2).enumerate() {
            let member = &member_pair[0];
            let next_member = &member_pair[1];
            let member_ty = member.ty();
            let next_member_ty = next_member.ty();

            // This can't panic because we only serialize types with a defined layout.
            let size = member_ty.size(member.matrix_stride, member.matrix_majorness);

            // This can't panic because only the last member can be unsized.
            let size = size.unwrap();

            let next_alignment = next_member_ty.scalar_alignment();

            // These two can't panic because the member must fit in the struct.
            let end = member.offset.checked_add(size).unwrap();
            let aligned_end = align_up(end, next_alignment);

            if aligned_end > next_member.offset {
                // This would be a huge pain in the ass to support, but teeeeechnically, nothing in
                // the spec forbids this. Or rather, the spec is ambiguous.
                unimplemented!(
                    "expected struct members of struct `{}` to be in order of increasing offsets",
                    self.ty.name,
                );
            }

            if !is_aligned(next_member.offset, member.scalar_alignment()) {
                unimplemented!(
                    "expected the offset of struct member {} of struct `{}` to be aligned to the \
                    previous member's scalar alignment",
                    member_index + 1,
                    self.ty.name,
                );
            }

            let ser = Serializer {
                ty: &member_ty,
                options: self.options,
                matrix_stride: member.matrix_stride,
                matrix_majorness: member.matrix_majorness,
            };

            // Pad the member such that the next member is at the right offset, if required.
            //
            // NOTE(Marc): The next offset must be aligned to the scalar alignment of the previous
            // member's type in order to prevent a situation like this:
            //
            // | 0x0 |     |     |     |     | 0x5 |
            // |-----------------------|-----|-----|
            // | u32                   |     | u8  |
            //
            // We can't pad the `u8` using a `Padded<u32, N>` because its size will always be
            // rounded up to the nearest multiple of its alignment, which is 4 in this case. We
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
            let padding = if aligned_end < next_member.offset {
                next_member.offset - member.offset - size
            } else {
                0
            };

            let ident = Ident::new(member.name, Span::call_site());
            let ser = Padded(&ser, padding);
            members_code.extend(quote! { pub #ident: #ser, });
        }

        // Add the last field, which is excluded in the above loop (both if the number of members
        // is 1 and >= 2).
        if let Some(member) = self.ty.members.last() {
            let ident = Ident::new(member.name, Span::call_site());
            let ser = Serializer {
                ty: &member.ty(),
                options: self.options,
                matrix_stride: member.matrix_stride,
                matrix_majorness: member.matrix_majorness,
            };
            members_code.extend(quote! { pub #ident: #ser, });
        }

        tokens.extend(quote! {
            pub struct #struct_ident {
                #members_code
            }
        })
    }
}

/// Helper for wrapping tokens in `Padded`. Doesn't wrap if the padding is `0`.
struct Padded<'a, T>(&'a T, u32);

impl<T: ToTokens> ToTokens for Padded<'_, T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let ty = self.0;
        let padding = self.1 as usize;

        if padding == 0 {
            ty.to_tokens(tokens);
        } else {
            tokens.extend(quote! { ::vulkano::padded::Padded<#ty, #padding> });
        }
    }
}
