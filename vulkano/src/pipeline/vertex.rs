// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! # Vertex sources definition
//!
//! When you create a graphics pipeline object, you need to pass an object which indicates the
//! layout of the vertex buffer(s) that will serve as input for the vertex shader. This is done
//! by passing an implementation of the `Definition` trait.
//!
//! In addition to this, the object that you pass when you create the graphics pipeline must also
//! implement the `Source` trait. This trait has a template parameter which corresponds to the
//! list of vertex buffers.
//!
//! The vulkano library provides some structs that already implement these traits.
//! The most common situation is a single vertex buffer and no instancing, in which case you can
//! pass a `SingleBufferDefinition` when you create the pipeline.
//!
//! # Implementing `Vertex`
//!
//! The implementations of the `Definition` trait that are provided by vulkano (like
//! `SingleBufferDefinition`) require you to use a buffer whose content is `[V]` where `V`
//! implements the `Vertex` trait.
//!
//! The `Vertex` trait is unsafe, but can be implemented on a struct with the `impl_vertex!`
//! macro.
//!
//! # Example
//!
//! ```ignore       // TODO:
//! # #[macro_use] extern crate vulkano
//! # fn main() {
//! # use std::sync::Arc;
//! # use vulkano::device::Device;
//! # use vulkano::device::Queue;
//! use vulkano::buffer::Buffer;
//! use vulkano::buffer::Usage as BufferUsage;
//! use vulkano::memory::HostVisible;
//! use vulkano::pipeline::vertex::;
//! # let device: Arc<Device> = unsafe { std::mem::uninitialized() };
//! # let queue: Arc<Queue> = unsafe { std::mem::uninitialized() };
//!
//! struct Vertex {
//!     position: [f32; 2]
//! }
//!
//! impl_vertex!(Vertex, position);
//!
//! let usage = BufferUsage {
//!     vertex_buffer: true,
//!     .. BufferUsage::none()
//! };
//!
//! let vertex_buffer = Buffer::<[Vertex], _>::array(&device, 128, &usage, HostVisible, &queue)
//!                                                     .expect("failed to create buffer");
//!
//! // TODO: finish example
//! # }
//! ```

use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::vec::IntoIter as VecIntoIter;

use buffer::BufferInner;
use buffer::TypedBuffer;
use format::Format;
use pipeline::shader::ShaderInterfaceDef;
use vk;

/// How the vertex source should be unrolled.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum InputRate {
    /// Each element of the source corresponds to a vertex.
    Vertex = vk::VERTEX_INPUT_RATE_VERTEX,
    /// Each element of the source corresponds to an instance.
    Instance = vk::VERTEX_INPUT_RATE_INSTANCE,
}

/// Describes an individual `Vertex`. In other words a collection of attributes that can be read
/// from a vertex shader.
///
/// At this stage, the vertex is in a "raw" format. For example a `[f32; 4]` can match both a
/// `vec4` or a `float[4]`. The way the things are binded depends on the shader.
pub unsafe trait Vertex: 'static + Send + Sync {
    /// Returns the characteristics of a vertex member by its name.
    fn member(name: &str) -> Option<VertexMemberInfo>;
}

unsafe impl Vertex for () {
    #[inline]
    fn member(_: &str) -> Option<VertexMemberInfo> {
        None
    }
}

/// Information about a member of a vertex struct.
pub struct VertexMemberInfo {
    /// Offset of the member in bytes from the start of the struct.
    pub offset: usize,
    /// Type of data. This is used to check that the interface is matching.
    pub ty: VertexMemberTy,
    /// Number of consecutive elements of that type.
    pub array_size: usize,
}

/// Type of a member of a vertex struct.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum VertexMemberTy {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl VertexMemberTy {
    /// Returns true if a combination of `(type, array_size)` matches a format.
    #[inline]
    pub fn matches(&self, array_size: usize, format: Format, num_locs: u32) -> bool {
        // TODO: implement correctly
        let my_size = match *self {
            VertexMemberTy::I8 => 1,
            VertexMemberTy::U8 => 1,
            VertexMemberTy::I16 => 2,
            VertexMemberTy::U16 => 2,
            VertexMemberTy::I32 => 4,
            VertexMemberTy::U32 => 4,
            VertexMemberTy::F32 => 4,
            VertexMemberTy::F64 => 8,
        };

        let format_size = match format.size() {
            None => return false,
            Some(s) => s,
        };

        array_size * my_size == format_size * num_locs as usize
    }
}

/// Information about a single attribute within a vertex.
/// TODO: change that API
pub struct AttributeInfo {
    /// Number of bytes between the start of a vertex and the location of attribute.
    pub offset: usize,
    /// VertexMember type of the attribute.
    pub format: Format,
}

/// Trait for types that describe the definition of the vertex input used by a graphics pipeline.
pub unsafe trait Definition<I>: 'static + Send + Sync {
    /// Iterator that returns the offset, the stride (in bytes) and input rate of each buffer.
    type BuffersIter: ExactSizeIterator<Item = (u32, usize, InputRate)>;
    /// Iterator that returns the attribute location, buffer id, and infos.
    type AttribsIter: ExactSizeIterator<Item = (u32, u32, AttributeInfo)>;

    /// Builds the vertex definition to use to link this definition to a vertex shader's input
    /// interface.
    fn definition(&self, interface: &I) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError>;
}

/// Error that can happen when the vertex definition doesn't match the input of the vertex shader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IncompatibleVertexDefinitionError {
    /// An attribute of the vertex shader is missing in the vertex source.
    MissingAttribute {
        /// Name of the missing attribute.
        attribute: String,
    },

    /// The format of an attribute does not match.
    FormatMismatch {
        /// Name of the attribute.
        attribute: String,
        /// The format in the vertex shader.
        shader: (Format, usize),
        /// The format in the vertex definition.
        definition: (VertexMemberTy, usize),
    },
}

impl error::Error for IncompatibleVertexDefinitionError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            IncompatibleVertexDefinitionError::MissingAttribute { .. } => "an attribute is missing",
            IncompatibleVertexDefinitionError::FormatMismatch { .. } => "the format of an attribute does not match",
        }
    }
}

impl fmt::Display for IncompatibleVertexDefinitionError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}


/// Extension trait of `Definition`. The `L` parameter is an acceptable vertex source for this
/// vertex definition.
pub unsafe trait Source<L> {
    /// Checks and returns the list of buffers with offsets, number of vertices and number of instances.
    // TODO: return error if problem
    // TODO: better than a Vec
    // TODO: return a struct instead
    fn decode<'l>(&self, &'l L) -> (Vec<BufferInner<'l>>, usize, usize);
}

/// Implementation of `Definition` for a single vertex buffer.
pub struct SingleBufferDefinition<T>(pub PhantomData<T>);

impl<T> SingleBufferDefinition<T> {
    #[inline]
    pub fn new() -> SingleBufferDefinition<T> {
        SingleBufferDefinition(PhantomData)
    }
}

unsafe impl<T, I> Definition<I> for SingleBufferDefinition<T>
    where T: Vertex,
          I: ShaderInterfaceDef
{
    type BuffersIter = OptionIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        let attrib = {
                let mut attribs = Vec::with_capacity(interface.elements().len());
                for e in interface.elements() {
                    let name = e.name.as_ref().unwrap();

                    let infos = match <T as Vertex>::member(name) {
                        Some(m) => m,
                        None => return Err(IncompatibleVertexDefinitionError::MissingAttribute { attribute: name.clone().into_owned() }),
                    };

                    if !infos.ty.matches(infos.array_size,
                                         e.format,
                                         e.location.end - e.location.start) {
                        return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                            attribute: name.clone().into_owned(),
                            shader: (e.format, (e.location.end - e.location.start) as usize),
                            definition: (infos.ty, infos.array_size),
                        });
                    }

                    let mut offset = infos.offset;
                    for loc in e.location.clone() {
                        attribs.push((loc,
                                      0,
                                      AttributeInfo {
                                          offset: offset,
                                          format: e.format,
                                      }));
                        offset += e.format.size().unwrap();
                    }
                }
                attribs
            }
            .into_iter();      // TODO: meh

        let buffers = Some((0, mem::size_of::<T>(), InputRate::Vertex)).into_iter();
        Ok((buffers, attrib))
    }
}

unsafe impl<'a, B, V> Source<B> for SingleBufferDefinition<V>
    where B: TypedBuffer<Content = [V]>,
          V: Vertex
{
    #[inline]
    fn decode<'l>(&self, source: &'l B) -> (Vec<BufferInner<'l>>, usize, usize) {
        (vec![source.inner()], source.len(), 1)
    }
}

/// Unstable.
// TODO: shouldn't be just `Two` but `Multi`
pub struct TwoBuffersDefinition<T, U>(pub PhantomData<(T, U)>);

impl<T, U> TwoBuffersDefinition<T, U> {
    #[inline]
    pub fn new() -> TwoBuffersDefinition<T, U> {
        TwoBuffersDefinition(PhantomData)
    }
}

unsafe impl<T, U, I> Definition<I> for TwoBuffersDefinition<T, U>
    where T: Vertex,
          U: Vertex,
          I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        let attrib = {
                let mut attribs = Vec::with_capacity(interface.elements().len());
                for e in interface.elements() {
                    let name = e.name.as_ref().unwrap();

                    let (infos, buf_offset) = if let Some(infos) = <T as Vertex>::member(name) {
                        (infos, 0)
                    } else if let Some(infos) = <U as Vertex>::member(name) {
                        (infos, 1)
                    } else {
                        return Err(IncompatibleVertexDefinitionError::MissingAttribute { attribute: name.clone().into_owned() });
                    };

                    if !infos.ty.matches(infos.array_size,
                                         e.format,
                                         e.location.end - e.location.start) {
                        return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                            attribute: name.clone().into_owned(),
                            shader: (e.format, (e.location.end - e.location.start) as usize),
                            definition: (infos.ty, infos.array_size),
                        });
                    }

                    let mut offset = infos.offset;
                    for loc in e.location.clone() {
                        attribs.push((loc,
                                      buf_offset,
                                      AttributeInfo {
                                          offset: offset,
                                          format: e.format,
                                      }));
                        offset += e.format.size().unwrap();
                    }
                }
                attribs
            }
            .into_iter();      // TODO: meh

        let buffers = vec![
            (0, mem::size_of::<T>(), InputRate::Vertex),
            (1, mem::size_of::<U>(), InputRate::Vertex)
        ]
            .into_iter();

        Ok((buffers, attrib))
    }
}

unsafe impl<'a, T, U, Bt, Bu> Source<(Bt, Bu)> for TwoBuffersDefinition<T, U>
    where T: Vertex,
          Bt: TypedBuffer<Content = [T]>,
          U: Vertex,
          Bu: TypedBuffer<Content = [U]>
{
    #[inline]
    fn decode<'l>(&self, source: &'l (Bt, Bu)) -> (Vec<BufferInner<'l>>, usize, usize) {
        let vertices = [source.0.len(), source.1.len()].iter().cloned().min().unwrap();
        (vec![source.0.inner(), source.1.inner()], vertices, 1)
    }
}

/// Unstable.
// TODO: bad way to do things
pub struct OneVertexOneInstanceDefinition<T, U>(pub PhantomData<(T, U)>);

impl<T, U> OneVertexOneInstanceDefinition<T, U> {
    #[inline]
    pub fn new() -> OneVertexOneInstanceDefinition<T, U> {
        OneVertexOneInstanceDefinition(PhantomData)
    }
}

unsafe impl<T, U, I> Definition<I> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex,
          U: Vertex,
          I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        let attrib = {
                let mut attribs = Vec::with_capacity(interface.elements().len());
                for e in interface.elements() {
                    let name = e.name.as_ref().unwrap();

                    let (infos, buf_offset) = if let Some(infos) = <T as Vertex>::member(name) {
                        (infos, 0)
                    } else if let Some(infos) = <U as Vertex>::member(name) {
                        (infos, 1)
                    } else {
                        return Err(IncompatibleVertexDefinitionError::MissingAttribute { attribute: name.clone().into_owned() });
                    };

                    if !infos.ty.matches(infos.array_size,
                                         e.format,
                                         e.location.end - e.location.start) {
                        return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                            attribute: name.clone().into_owned(),
                            shader: (e.format, (e.location.end - e.location.start) as usize),
                            definition: (infos.ty, infos.array_size),
                        });
                    }

                    let mut offset = infos.offset;
                    for loc in e.location.clone() {
                        attribs.push((loc,
                                      buf_offset,
                                      AttributeInfo {
                                          offset: offset,
                                          format: e.format,
                                      }));
                        offset += e.format.size().unwrap();
                    }
                }
                attribs
            }
            .into_iter();      // TODO: meh

        let buffers = vec![
            (0, mem::size_of::<T>(), InputRate::Vertex),
            (1, mem::size_of::<U>(), InputRate::Instance)
        ]
            .into_iter();

        Ok((buffers, attrib))
    }
}

unsafe impl<'a, T, U, Bt, Bu> Source<(Bt, Bu)> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex,
          Bt: TypedBuffer<Content = [T]>,
          U: Vertex,
          Bu: TypedBuffer<Content = [U]>
{
    #[inline]
    fn decode<'l>(&self, source: &'l (Bt, Bu)) -> (Vec<BufferInner<'l>>, usize, usize) {
        (vec![source.0.inner(), source.1.inner()], source.0.len(), source.1.len())
    }
}

/// Implements the `Vertex` trait on a struct.
// TODO: add example
#[macro_export]
macro_rules! impl_vertex {
    ($out:ident $(, $member:ident)*) => (
        #[allow(unsafe_code)]
        unsafe impl $crate::pipeline::vertex::Vertex for $out {
            #[inline(always)]
            fn member(name: &str) -> Option<$crate::pipeline::vertex::VertexMemberInfo> {
                #[allow(unused_imports)]
                use $crate::format::Format;
                use $crate::pipeline::vertex::VertexMemberInfo;
                use $crate::pipeline::vertex::VertexMemberTy;
                use $crate::pipeline::vertex::VertexMember;

                $(
                    if name == stringify!($member) {
                        let (ty, array_size) = unsafe {
                            #[inline] fn f<T: VertexMember>(_: &T) -> (VertexMemberTy, usize)
                                      { T::format() }
                            let dummy = 0usize as *const $out;
                            f(&(&*dummy).$member)
                        };

                        return Some(VertexMemberInfo {
                            offset: unsafe {
                                let dummy = 0usize as *const $out;
                                let member = (&(&*dummy).$member) as *const _;
                                member as usize
                            },

                            ty: ty,
                            array_size: array_size,
                        });
                    }
                )*

                None
            }
        }
    )
}

/// Trait for data types that can be used as vertex members. Used by the `impl_vertex!` macro.
pub unsafe trait VertexMember {
    /// Returns the format and array size of the member.
    fn format() -> (VertexMemberTy, usize);
}

unsafe impl VertexMember for i8 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::I8, 1)
    }
}

unsafe impl VertexMember for u8 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::U8, 1)
    }
}

unsafe impl VertexMember for i16 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::I16, 1)
    }
}

unsafe impl VertexMember for u16 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::U16, 1)
    }
}

unsafe impl VertexMember for i32 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::I32, 1)
    }
}

unsafe impl VertexMember for u32 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::U32, 1)
    }
}

unsafe impl VertexMember for f32 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::F32, 1)
    }
}

unsafe impl VertexMember for f64 {
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::F64, 1)
    }
}

unsafe impl<T> VertexMember for (T,)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        <T as VertexMember>::format()
    }
}

unsafe impl<T> VertexMember for (T, T)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 2)
    }
}

unsafe impl<T> VertexMember for (T, T, T)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 3)
    }
}

unsafe impl<T> VertexMember for (T, T, T, T)
    where T: VertexMember
{
    #[inline]
    fn format() -> (VertexMemberTy, usize) {
        let (ty, sz) = <T as VertexMember>::format();
        (ty, sz * 4)
    }
}

macro_rules! impl_vm_array {
    ($sz:expr) => (
        unsafe impl<T> VertexMember for [T; $sz]
            where T: VertexMember
        {
            #[inline]
            fn format() -> (VertexMemberTy, usize) {
                let (ty, sz) = <T as VertexMember>::format();
                (ty, sz * $sz)
            }
        }
    );
}

impl_vm_array!(1);
impl_vm_array!(2);
impl_vm_array!(3);
impl_vm_array!(4);
impl_vm_array!(5);
impl_vm_array!(6);
impl_vm_array!(7);
impl_vm_array!(8);
impl_vm_array!(9);
impl_vm_array!(10);
impl_vm_array!(11);
impl_vm_array!(12);
impl_vm_array!(13);
impl_vm_array!(14);
impl_vm_array!(15);
impl_vm_array!(16);
impl_vm_array!(32);
impl_vm_array!(64);
