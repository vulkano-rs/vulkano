use crate::{bail, LinAlgTypes, MacroOptions};
use ahash::HashMap;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use std::{cmp::Ordering, num::NonZeroUsize};
use syn::{Error, Ident, LitStr, Result};
use vulkano::shader::spirv::{Decoration, Id, Instruction, Spirv};

struct Shader {
    spirv: Spirv,
    name: String,
    source: LitStr,
}

pub(super) struct RegisteredStruct {
    members: Vec<Member>,
    shader_name: String,
}

/// Generates Rust structs from structs declared in SPIR-V bytecode.
pub(super) fn generate_structs(
    macro_options: &MacroOptions,
    spirv: Spirv,
    shader_name: String,
    shader_source: LitStr,
    registered_structs: &mut HashMap<Ident, RegisteredStruct>,
) -> Result<TokenStream> {
    let mut structs_code = TokenStream::new();

    let shader = Shader {
        spirv,
        name: shader_name,
        source: shader_source,
    };

    let structs = shader
        .spirv
        .types()
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::TypeStruct {
                result_id,
                member_types,
            } => Some((*result_id, member_types)),
            _ => None,
        })
        .filter(|&(id, _)| struct_has_defined_layout(&shader.spirv, id));

    for (struct_id, member_type_ids) in structs {
        let ty = TypeStruct::new(&shader, struct_id, member_type_ids)?;

        if let Some(registered) = registered_structs.get(&ty.ident) {
            validate_members(
                &ty.ident,
                &ty.members,
                &registered.members,
                &shader.name,
                &registered.shader_name,
            )?;
        } else {
            let custom_derives = if ty.size().is_some() {
                macro_options.custom_derives.as_slice()
            } else {
                &[]
            };

            let struct_ser = Serializer(&ty, macro_options);

            structs_code.extend(quote! {
                #[allow(non_camel_case_types, non_snake_case)]
                #[derive(::vulkano::buffer::BufferContents #(, #custom_derives )* )]
                #[repr(C)]
                #struct_ser
            });

            registered_structs.insert(
                ty.ident,
                RegisteredStruct {
                    members: ty.members,
                    shader_name: shader.name.clone(),
                },
            );
        }
    }

    Ok(structs_code)
}

fn struct_has_defined_layout(spirv: &Spirv, id: Id) -> bool {
    spirv.id(id).members().iter().all(|member_info| {
        let decorations = member_info
            .decorations()
            .iter()
            .map(|instruction| match instruction {
                Instruction::MemberDecorate { decoration, .. } => decoration,
                _ => unreachable!(),
            });

        let has_offset_decoration = decorations
            .clone()
            .any(|decoration| matches!(decoration, Decoration::Offset { .. }));

        let has_builtin_decoration = decorations
            .clone()
            .any(|decoration| matches!(decoration, Decoration::BuiltIn { .. }));

        has_offset_decoration && !has_builtin_decoration
    })
}

fn validate_members(
    ident: &Ident,
    first_members: &[Member],
    second_members: &[Member],
    first_shader: &str,
    second_shader: &str,
) -> Result<()> {
    match first_members.len().cmp(&second_members.len()) {
        Ordering::Greater => bail!(
            "the declaration of struct `{ident}` in shader `{first_shader}` has more fields than \
            the declaration in shader `{second_shader}`"
        ),
        Ordering::Less => bail!(
            "the declaration of struct `{ident}` in shader `{second_shader}` has more fields than \
            the declaration in shader `{first_shader}`"
        ),
        _ => {}
    }

    for (index, (first_member, second_member)) in
        first_members.iter().zip(second_members).enumerate()
    {
        let (first_type, second_type) = (&first_member.ty, &second_member.ty);
        if first_type != second_type {
            bail!(
                "field {index} of struct `{ident}` is of type `{first_type:?}` in shader \
                `{first_shader}` but of type `{second_type:?}` in shader `{second_shader}`"
            );
        }
    }

    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
            _ => panic!(),
        }
    }
}

fn align_up(offset: usize, alignment: Alignment) -> usize {
    (offset + alignment as usize - 1) & !(alignment as usize - 1)
}

fn is_aligned(offset: usize, alignment: Alignment) -> bool {
    offset & (alignment as usize - 1) == 0
}

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
    fn new(shader: &Shader, type_id: Id) -> Result<Self> {
        let id_info = shader.spirv.id(type_id);

        let ty = match *id_info.instruction() {
            Instruction::TypeBool { .. } => bail!(
                shader.source,
                "SPIR-V Boolean types don't have a defined layout"
            ),
            Instruction::TypeInt {
                width, signedness, ..
            } => Type::Scalar(TypeScalar::Int(TypeInt::new(shader, width, signedness)?)),
            Instruction::TypeFloat { width, .. } => {
                Type::Scalar(TypeScalar::Float(TypeFloat::new(shader, width)?))
            }
            Instruction::TypePointer { .. } => Type::Pointer(TypePointer::new(shader)?),
            Instruction::TypeVector {
                component_type,
                component_count,
                ..
            } => Type::Vector(TypeVector::new(shader, component_type, component_count)?),
            Instruction::TypeMatrix {
                column_type,
                column_count,
                ..
            } => Type::Matrix(TypeMatrix::new(shader, column_type, column_count)?),
            Instruction::TypeArray {
                element_type,
                length,
                ..
            } => Type::Array(TypeArray::new(shader, type_id, element_type, Some(length))?),
            Instruction::TypeRuntimeArray { element_type, .. } => {
                Type::Array(TypeArray::new(shader, type_id, element_type, None)?)
            }
            Instruction::TypeStruct {
                ref member_types, ..
            } => Type::Struct(TypeStruct::new(shader, type_id, member_types)?),
            _ => bail!(shader.source, "type {type_id} was not found"),
        };

        Ok(ty)
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
    fn new(shader: &Shader, width: u32, signedness: u32) -> Result<Self> {
        let width = match width {
            8 => IntWidth::W8,
            16 => IntWidth::W16,
            32 => IntWidth::W32,
            64 => IntWidth::W64,
            _ => bail!(shader.source, "integers must be 8, 16, 32, or 64-bit wide"),
        };

        let signed = match signedness {
            0 => false,
            1 => true,
            _ => bail!(shader.source, "signedness must be either 0 or 1"),
        };

        Ok(TypeInt { width, signed })
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
    fn new(shader: &Shader, width: u32) -> Result<Self> {
        let width = match width {
            16 => FloatWidth::W16,
            32 => FloatWidth::W32,
            64 => FloatWidth::W64,
            _ => bail!(shader.source, "floats must be 16, 32, or 64-bit wide"),
        };

        Ok(TypeFloat { width })
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
    fn new(_shader: &Shader) -> Result<Self> {
        Ok(TypePointer)
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
    fn new(shader: &Shader, component_type_id: Id, component_count: u32) -> Result<Self> {
        let component_count = ComponentCount::new(shader, component_count)?;

        let component_type = match *shader.spirv.id(component_type_id).instruction() {
            Instruction::TypeBool { .. } => bail!(
                shader.source,
                "SPIR-V Boolean types don't have a defined layout"
            ),
            Instruction::TypeInt {
                width, signedness, ..
            } => TypeScalar::Int(TypeInt::new(shader, width, signedness)?),
            Instruction::TypeFloat { width, .. } => {
                TypeScalar::Float(TypeFloat::new(shader, width)?)
            }
            _ => bail!(shader.source, "vector components must be scalar"),
        };

        Ok(TypeVector {
            component_type,
            component_count,
        })
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
    fn new(shader: &Shader, column_type_id: Id, column_count: u32) -> Result<Self> {
        let column_count = ComponentCount::new(shader, column_count)?;

        let (component_type, row_count) = match *shader.spirv.id(column_type_id).instruction() {
            Instruction::TypeVector {
                component_type,
                component_count,
                ..
            } => match *shader.spirv.id(component_type).instruction() {
                Instruction::TypeFloat { width, .. } => (
                    TypeFloat::new(shader, width)?,
                    ComponentCount::new(shader, component_count)?,
                ),
                _ => bail!(shader.source, "matrix components must be floats"),
            },
            _ => bail!(shader.source, "matrix columns must be vectors"),
        };

        // We can't know these until we get to the members and their decorations, so just use
        // defaults for now.
        let stride = component_type.size() * row_count as usize;
        let majorness = MatrixMajorness::ColumnMajor;

        Ok(TypeMatrix {
            component_type,
            column_count,
            row_count,
            stride,
            majorness,
        })
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
    fn new(shader: &Shader, count: u32) -> Result<Self> {
        let count = match count {
            2 => ComponentCount::C2,
            3 => ComponentCount::C3,
            4 => ComponentCount::C4,
            _ => bail!(shader.source, "component counts must be 2, 3 or 4"),
        };

        Ok(count)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TypeArray {
    element_type: Box<Type>,
    length: Option<NonZeroUsize>,
    stride: usize,
}

impl TypeArray {
    fn new(
        shader: &Shader,
        array_id: Id,
        element_type_id: Id,
        length_id: Option<Id>,
    ) -> Result<Self> {
        let element_type = Box::new(Type::new(shader, element_type_id)?);

        let length = length_id
            .map(|id| match shader.spirv.id(id).instruction() {
                Instruction::Constant { value, .. } | Instruction::SpecConstant { value, .. } => {
                    assert!(matches!(value.len(), 1 | 2));
                    let len = value.iter().rev().fold(0u64, |a, &b| (a << 32) | b as u64);

                    NonZeroUsize::new(len.try_into().unwrap()).ok_or_else(|| {
                        Error::new_spanned(&shader.source, "arrays must have a non-zero length")
                    })
                }
                _ => bail!(shader.source, "failed to find array length"),
            })
            .transpose()?;

        let mut strides =
            shader
                .spirv
                .id(array_id)
                .decorations()
                .iter()
                .filter_map(|instruction| match *instruction {
                    Instruction::Decorate {
                        decoration: Decoration::ArrayStride { array_stride },
                        ..
                    } => Some(array_stride as usize),
                    _ => None,
                });

        let stride = strides.next().ok_or_else(|| {
            Error::new_spanned(
                &shader.source,
                "arrays inside structs must have an `ArrayStride` decoration",
            )
        })?;

        if !strides.all(|s| s == stride) {
            bail!(shader.source, "found conflicting `ArrayStride` decorations");
        }

        if !is_aligned(stride, element_type.scalar_alignment()) {
            bail!(
                shader.source,
                "array strides must be aligned to the element type's alignment",
            );
        }

        let Some(element_size) = element_type.size() else {
            bail!(shader.source, "array elements must be sized");
        };

        if stride < element_size {
            bail!(shader.source, "array elements must not overlap");
        }

        Ok(TypeArray {
            element_type,
            length,
            stride,
        })
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
    fn new(shader: &Shader, struct_id: Id, member_type_ids: &[Id]) -> Result<Self> {
        let id_info = shader.spirv.id(struct_id);

        let ident = id_info
            .names()
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::Name { name, .. } => {
                    // Replace non-alphanumeric and non-ascii characters with '_' to ensure the name
                    // is a valid identifier. For example, Rust-GPU names structs by their fully
                    // qualified rust name (e.g. `foo::bar::MyStruct`) in which `:` makes it an
                    // invalid identifier.
                    let mut name =
                        name.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_");

                    if name.starts_with(|c: char| c.is_ascii_digit()) {
                        name.insert(0, '_');
                    }

                    // Fall-back to `Unnamed{Id}` if it's still invalid
                    syn::parse_str(&name).ok()
                }
                _ => None,
            })
            .unwrap_or_else(|| format_ident!("Unnamed{}", struct_id.as_raw()));

        let mut members = Vec::<Member>::with_capacity(member_type_ids.len());

        for (member_index, (&member_id, member_info)) in
            member_type_ids.iter().zip(id_info.members()).enumerate()
        {
            let ident = member_info
                .names()
                .iter()
                .find_map(|instruction| match instruction {
                    Instruction::MemberName { name, .. } => {
                        Some(Ident::new(name, Span::call_site()))
                    }
                    _ => None,
                })
                .unwrap_or_else(|| format_ident!("unnamed{member_index}"));

            let mut ty = Type::new(shader, member_id)?;

            {
                // Matrix decorations can be applied to an array if its innermost type is a matrix.
                let mut ty = &mut ty;
                while let Type::Array(TypeArray { element_type, .. }) = ty {
                    ty = element_type;
                }

                if let Type::Matrix(matrix) = ty {
                    let mut strides =
                        member_info.decorations().iter().filter_map(
                            |instruction| match *instruction {
                                Instruction::MemberDecorate {
                                    decoration: Decoration::MatrixStride { matrix_stride },
                                    ..
                                } => Some(matrix_stride as usize),
                                _ => None,
                            },
                        );

                    matrix.stride = strides.next().ok_or_else(|| {
                        Error::new_spanned(
                            &shader.source,
                            "matrices inside structs must have a `MatrixStride` decoration",
                        )
                    })?;

                    if !strides.all(|s| s == matrix.stride) {
                        bail!(
                            shader.source,
                            "found conflicting `MatrixStride` decorations",
                        );
                    }

                    if !is_aligned(matrix.stride, matrix.component_type.alignment()) {
                        bail!(
                            shader.source,
                            "matrix strides must be an integer multiple of the size of the \
                            component",
                        );
                    }

                    let mut majornesses = member_info.decorations().iter().filter_map(
                        |instruction| match *instruction {
                            Instruction::MemberDecorate {
                                decoration: Decoration::ColMajor,
                                ..
                            } => Some(MatrixMajorness::ColumnMajor),
                            Instruction::MemberDecorate {
                                decoration: Decoration::RowMajor,
                                ..
                            } => Some(MatrixMajorness::RowMajor),
                            _ => None,
                        },
                    );

                    matrix.majorness = majornesses.next().ok_or_else(|| {
                        Error::new_spanned(
                            &shader.source,
                            "matrices inside structs must have a `ColMajor` or `RowMajor` \
                            decoration",
                        )
                    })?;

                    if !majornesses.all(|m| m == matrix.majorness) {
                        bail!(
                            shader.source,
                            "found conflicting matrix majorness decorations",
                        );
                    }

                    // NOTE(Marc): It is crucial that we do this check after setting the majorness,
                    // because `TypeMatrix::vector_size` depends on it.
                    if matrix.stride < matrix.vector_size() {
                        bail!(shader.source, "matrix columns/rows must not overlap");
                    }
                }
            }

            let offset = member_info
                .decorations()
                .iter()
                .find_map(|instruction| match *instruction {
                    Instruction::MemberDecorate {
                        decoration: Decoration::Offset { byte_offset },
                        ..
                    } => Some(byte_offset as usize),
                    _ => None,
                })
                .ok_or_else(|| {
                    Error::new_spanned(
                        &shader.source,
                        "struct members must have an `Offset` decoration",
                    )
                })?;

            if !is_aligned(offset, ty.scalar_alignment()) {
                bail!(
                    shader.source,
                    "struct member offsets must be aligned to their type's alignment",
                );
            }

            if let Some(previous_member) = members.last() {
                if !is_aligned(offset, previous_member.ty.scalar_alignment()) {
                    bail!(
                        shader.source,
                        "expected struct member offset to be aligned to the preceding member \
                        type's alignment",
                    );
                }

                let last_size = previous_member.ty.size().ok_or_else(|| {
                    Error::new_spanned(
                        &shader.source,
                        "all members except the last member of a struct must be sized",
                    )
                })?;

                if previous_member.offset + last_size > offset {
                    bail!(shader.source, "struct members must not overlap");
                }
            }

            members.push(Member { ident, ty, offset });
        }

        Ok(TypeStruct { ident, members })
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

/// Helper for serializing a type as tokens according to the macro options.
struct Serializer<'a, T>(&'a T, &'a MacroOptions);

impl ToTokens for Serializer<'_, Type> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match &self.0 {
            Type::Scalar(ty) => ty.to_tokens(tokens),
            Type::Pointer(ty) => ty.to_tokens(tokens),
            Type::Vector(ty) => Serializer(ty, self.1).to_tokens(tokens),
            Type::Matrix(ty) => Serializer(ty, self.1).to_tokens(tokens),
            Type::Array(ty) => Serializer(ty, self.1).to_tokens(tokens),
            Type::Struct(ty) => tokens.append(ty.ident.clone()),
        }
    }
}

impl ToTokens for Serializer<'_, TypeVector> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let component_type = &self.0.component_type;
        let component_count = self.0.component_count as usize;

        match self.1.linalg_types {
            LinAlgTypes::Std => {
                tokens.extend(quote! { [#component_type; #component_count] });
            }
            LinAlgTypes::Cgmath => {
                let vector = format_ident!("Vector{}", component_count);
                tokens.extend(quote! { ::cgmath::#vector<#component_type> });
            }
            LinAlgTypes::Nalgebra => {
                tokens.extend(quote! { ::nalgebra::SVector<#component_type, #component_count> });
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

        match self.1.linalg_types {
            // cgmath only supports column-major square matrices, and its 3x3 matrix is not padded
            // correctly.
            LinAlgTypes::Cgmath
                if majorness == MatrixMajorness::ColumnMajor
                    && vector_count == component_count
                    && padding == 0 =>
            {
                let matrix = format_ident!("Matrix{}", component_count);
                tokens.extend(quote! { ::cgmath::#matrix<#component_type> });
            }
            // nalgebra only supports column-major matrices, and its 3xN matrices are not padded
            // correctly.
            LinAlgTypes::Nalgebra if majorness == MatrixMajorness::ColumnMajor && padding == 0 => {
                tokens.extend(quote! {
                    ::nalgebra::SMatrix<#component_type, #component_count, #vector_count>
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
        let element_type = self.0.element_type.as_ref();
        // This can't panic because array elements must be sized.
        let element_size = element_type.size().unwrap();
        // This can't overflow because the stride must be at least the element size.
        let padding = self.0.stride - element_size;

        let element_type = Padded(Serializer(element_type, self.1), padding);

        if let Some(length) = self.0.length.map(NonZeroUsize::get) {
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

/// Helper for wrapping tokens in [Padded][struct@vulkano::padded::Padded].
/// Doesn't wrap if the padding is `0`.
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
