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
//!
use std::marker::PhantomData;
use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use buffer::Buffer;
use buffer::TypedBuffer;
use format::Format;
use pipeline::shader::ShaderInterfaceDef;
use vk;

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum InputRate {
    Vertex = vk::VERTEX_INPUT_RATE_VERTEX,
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
    pub fn matches(&self, array_size: usize, format: Format) -> bool {
        true        // FIXME:
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
    // TODO: return error if problem
    fn definition(&self, interface: &I) -> (Self::BuffersIter, Self::AttribsIter);
}

/// Extension trait of `Definition`. The `L` parameter is an acceptable vertex source for this
/// vertex definition.
pub unsafe trait Source<L>: 'static + Send + Sync {
    /// Iterator used by `decode`.
    type Iter: ExactSizeIterator<Item = Arc<Buffer>>;

    /// Checks and returns the list of buffers, number of vertices and number of instances.
    // TODO: return error if problem
    fn decode(&self, L) -> (Self::Iter, usize, usize);
}

/// Implementation of `Definition` for a single vertex buffer.
pub struct SingleBufferDefinition<T>(pub PhantomData<T>);

impl<T> SingleBufferDefinition<T> {
    #[inline]
    pub fn new() -> SingleBufferDefinition<T> { SingleBufferDefinition(PhantomData) }
}

unsafe impl<T, I> Definition<I> for SingleBufferDefinition<T>
    where T: Vertex, I: ShaderInterfaceDef
{
    type BuffersIter = OptionIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I) -> (Self::BuffersIter, Self::AttribsIter) {
        let attrib = {
            let mut attribs = Vec::with_capacity(interface.elements().len());
            for e in interface.elements() {
                let infos = <T as Vertex>::member(e.name.as_ref().unwrap()).expect("missing vertex attrib");
                assert!(infos.ty.matches(infos.array_size, e.format));

                let mut offset = infos.offset;
                for loc in e.location.clone() {
                    attribs.push((loc, 0, AttributeInfo { offset: offset, format: e.format }));
                    offset += e.format.size().unwrap();
                }
            }
            attribs
        }.into_iter();      // TODO: meh

        let buffers = Some((0, mem::size_of::<T>(), InputRate::Vertex)).into_iter();
        (buffers, attrib)
    }
}

unsafe impl<'a, B, V> Source<&'a Arc<B>> for SingleBufferDefinition<V>
    where B: TypedBuffer<Content = [V]> + 'static, V: Vertex + 'static
{
    type Iter = OptionIntoIter<Arc<Buffer>>;

    #[inline]
    fn decode(&self, source: &'a Arc<B>) -> (OptionIntoIter<Arc<Buffer>>, usize, usize) {
        let iter = Some(source.clone() as Arc<_>).into_iter();
        (iter, source.len(), 1)
    }
}

/// Unstable.
// TODO: shouldn't be just `Two` but `Multi`
pub struct TwoBuffersDefinition<T, U>(pub PhantomData<(T, U)>);

impl<T, U> TwoBuffersDefinition<T, U> {
    #[inline]
    pub fn new() -> TwoBuffersDefinition<T, U> { TwoBuffersDefinition(PhantomData) }
}

unsafe impl<T, U, I> Definition<I> for TwoBuffersDefinition<T, U>
    where T: Vertex, U: Vertex, I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I) -> (Self::BuffersIter, Self::AttribsIter) {
        let attrib = {
            let mut attribs = Vec::with_capacity(interface.elements().len());
            for e in interface.elements() {
                let (infos, buf_offset) = if let Some(infos) = <T as Vertex>::member(e.name.as_ref().unwrap()) {
                    (infos, 0)
                } else if let Some(infos) = <U as Vertex>::member(e.name.as_ref().unwrap()) {
                    (infos, 1)
                } else {
                    panic!("missing vertex attrib")
                };
                assert!(infos.ty.matches(infos.array_size, e.format));

                let mut offset = infos.offset;
                for loc in e.location.clone() {
                    attribs.push((loc, buf_offset, AttributeInfo { offset: offset, format: e.format }));
                    offset += e.format.size().unwrap();
                }
            }
            attribs
        }.into_iter();      // TODO: meh

        let buffers = vec![
            (0, mem::size_of::<T>(), InputRate::Vertex),
            (1, mem::size_of::<U>(), InputRate::Vertex)
        ].into_iter();

        (buffers, attrib)
    }
}

unsafe impl<'a, T, U, Bt, Bu> Source<(&'a Arc<Bt>, &'a Arc<Bu>)> for TwoBuffersDefinition<T, U>
    where T: Vertex + 'static, Bt: TypedBuffer<Content = [T]> + 'static, T: 'static,
          U: Vertex + 'static, Bu: TypedBuffer<Content = [U]> + 'static, T: 'static
{
    type Iter = VecIntoIter<Arc<Buffer>>;

    #[inline]
    fn decode(&self, source: (&'a Arc<Bt>, &'a Arc<Bu>))
              -> (VecIntoIter<Arc<Buffer>>, usize, usize)
    {
        let iter = vec![source.0.clone() as Arc<_>, source.1.clone() as Arc<_>].into_iter();
        (iter, [source.0.len(), source.1.len()].iter().cloned().min().unwrap(), 1)
    }
}

/// Unstable.
// TODO: bad way to do things
pub struct OneVertexOneInstanceDefinition<T, U>(pub PhantomData<(T, U)>);

impl<T, U> OneVertexOneInstanceDefinition<T, U> {
    #[inline]
    pub fn new() -> OneVertexOneInstanceDefinition<T, U> { OneVertexOneInstanceDefinition(PhantomData) }
}

unsafe impl<T, U, I> Definition<I> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex, U: Vertex, I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(&self, interface: &I) -> (Self::BuffersIter, Self::AttribsIter) {
        let attrib = {
            let mut attribs = Vec::with_capacity(interface.elements().len());
            for e in interface.elements() {
                let (infos, buf_offset) = if let Some(infos) = <T as Vertex>::member(e.name.as_ref().unwrap()) {
                    (infos, 0)
                } else if let Some(infos) = <U as Vertex>::member(e.name.as_ref().unwrap()) {
                    (infos, 1)
                } else {
                    panic!("missing vertex attrib")
                };
                assert!(infos.ty.matches(infos.array_size, e.format));

                let mut offset = infos.offset;
                for loc in e.location.clone() {
                    attribs.push((loc, buf_offset, AttributeInfo { offset: offset, format: e.format }));
                    offset += e.format.size().unwrap();
                }
            }
            attribs
        }.into_iter();      // TODO: meh

        let buffers = vec![
            (0, mem::size_of::<T>(), InputRate::Vertex),
            (1, mem::size_of::<U>(), InputRate::Instance)
        ].into_iter();

        (buffers, attrib)
    }
}

unsafe impl<'a, T, U, Bt, Bu> Source<(&'a Arc<Bt>, &'a Arc<Bu>)> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex + 'static, Bt: TypedBuffer<Content = [T]> + 'static, T: 'static,
          U: Vertex + 'static, Bu: TypedBuffer<Content = [U]> + 'static, U: 'static
{
    type Iter = VecIntoIter<Arc<Buffer>>;

    #[inline]
    fn decode(&self, source: (&'a Arc<Bt>, &'a Arc<Bu>))
              -> (VecIntoIter<Arc<Buffer>>, usize, usize)
    {
        let iter = vec![source.0.clone() as Arc<_>, source.1.clone() as Arc<_>].into_iter();
        (iter, source.0.len(), source.1.len())
    }
}

/// Implements the `Vertex` trait on a struct.
// TODO: add example
#[macro_export]
macro_rules! impl_vertex {
    ($out:ident $(, $member:ident)*) => (
        unsafe impl $crate::pipeline::vertex::Vertex for $out {
            #[inline(always)]
            fn member(name: &str) -> Option<$crate::pipeline::vertex::VertexMemberInfo> {
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
