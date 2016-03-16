//! # Vertex sources definition
//! 
//! When you create a graphics pipeline object, you need to pass an object which indicates the
//! layout of the vertex buffer(s) that will serve as input for the vertex shader. This is done
//! by passing an implementation of the `Definition` trait.
//! 
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
//! ```no_run
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
use buffer::AbstractBuffer;
use format::Format;
use memory::MemorySource;
use vk;

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum InputRate {
    Vertex = vk::VERTEX_INPUT_RATE_VERTEX,
    Instance = vk::VERTEX_INPUT_RATE_INSTANCE,
}

/// Describes an individual `Vertex`. In other words a collection of attributes that can be read
/// from a vertex shader.
pub unsafe trait Vertex {
    /// Returns the characteristics of a vertex attribute.
    fn attrib(name: &str) -> Option<AttributeInfo>;
}

/// Information about a single attribute within a vertex.
pub struct AttributeInfo {
    /// Number of bytes between the start of a vertex and the location of attribute.
    pub offset: usize,
    /// Data type of the attribute.
    pub format: Format,
}

/// Trait for types that contain a definition of the vertex input used by a graphics pipeline.
pub unsafe trait Definition {
    /// Iterator used to enumerate the list of buffers.
    type InfoIter: ExactSizeIterator<Item = (usize, InputRate)>;

    /// Returns information about an attribute, and the index of the buffer (within the iterator
    /// returned by `buffers`) in which the attribute is found.
    fn attrib(&self, name: &str) -> Option<(usize, AttributeInfo)>;

    /// Produces an iterator that returns the stride (in bytes) and input rate of each buffer.
    fn buffers(&self) -> Self::InfoIter;
}

/// Extension for `Definition` trait. The `L` parameter is an acceptable vertex source for this
/// vertex definition.
pub unsafe trait Source<L>: Definition {
    /// Iterator used by `decode`.
    type Iter: ExactSizeIterator<Item = Arc<AbstractBuffer>>;

    /// Checks and returns the list of buffers, number of vertices and number of instances.
    fn decode(&self, L) -> (Self::Iter, usize, usize);
}

/// Implementation of `Definition` for a single vertex buffer.
pub struct SingleBufferDefinition<T>(pub PhantomData<T>);

impl<T> SingleBufferDefinition<T> {
    #[inline]
    pub fn new() -> SingleBufferDefinition<T> { SingleBufferDefinition(PhantomData) }
}

unsafe impl<T> Definition for SingleBufferDefinition<T> where T: Vertex {
    type InfoIter = OptionIntoIter<(usize, InputRate)>;

    #[inline]
    fn attrib(&self, name: &str) -> Option<(usize, AttributeInfo)> {
        <T as Vertex>::attrib(name).map(|info| (0, info))
    }

    #[inline]
    fn buffers(&self) -> Self::InfoIter {
        Some((mem::size_of::<T>(), InputRate::Vertex)).into_iter()
    }
}

unsafe impl<'a, V, M> Source<&'a Arc<Buffer<[V], M>>> for SingleBufferDefinition<V>
    where V: Vertex + 'static, M: MemorySource + 'static
{
    type Iter = OptionIntoIter<Arc<AbstractBuffer>>;

    #[inline]
    fn decode(&self, source: &'a Arc<Buffer<[V], M>>)
              -> (OptionIntoIter<Arc<AbstractBuffer>>, usize, usize)
    {
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

unsafe impl<T, U> Definition for TwoBuffersDefinition<T, U> where T: Vertex, U: Vertex {
    type InfoIter = VecIntoIter<(usize, InputRate)>;

    #[inline]
    fn attrib(&self, name: &str) -> Option<(usize, AttributeInfo)> {
        if let Some(a) = <T as Vertex>::attrib(name) {
            Some((0, a))
        } else if let Some(a) = <U as Vertex>::attrib(name) {
            Some((1, a))
        } else {
            None
        }
    }

    #[inline]
    fn buffers(&self) -> Self::InfoIter {
        vec![
            (mem::size_of::<T>(), InputRate::Vertex),
            (mem::size_of::<U>(), InputRate::Vertex)
        ].into_iter()
    }
}

unsafe impl<'a, T, U, Mt, Mu> Source<(&'a Arc<Buffer<[T], Mt>>, &'a Arc<Buffer<[U], Mu>>)> for TwoBuffersDefinition<T, U>
    where T: Vertex + 'static, Mt: MemorySource + 'static,
          U: Vertex + 'static, Mu: MemorySource + 'static
{
    type Iter = VecIntoIter<Arc<AbstractBuffer>>;

    #[inline]
    fn decode(&self, source: (&'a Arc<Buffer<[T], Mt>>, &'a Arc<Buffer<[U], Mu>>))
              -> (VecIntoIter<Arc<AbstractBuffer>>, usize, usize)
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

unsafe impl<T, U> Definition for OneVertexOneInstanceDefinition<T, U> where T: Vertex, U: Vertex {
    type InfoIter = VecIntoIter<(usize, InputRate)>;

    #[inline]
    fn attrib(&self, name: &str) -> Option<(usize, AttributeInfo)> {
        if let Some(a) = <T as Vertex>::attrib(name) {
            Some((0, a))
        } else if let Some(a) = <U as Vertex>::attrib(name) {
            Some((1, a))
        } else {
            None
        }
    }

    #[inline]
    fn buffers(&self) -> Self::InfoIter {
        vec![
            (mem::size_of::<T>(), InputRate::Vertex),
            (mem::size_of::<U>(), InputRate::Instance)
        ].into_iter()
    }
}

unsafe impl<'a, T, U, Mt, Mu> Source<(&'a Arc<Buffer<[T], Mt>>, &'a Arc<Buffer<[U], Mu>>)> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex + 'static, Mt: MemorySource + 'static,
          U: Vertex + 'static, Mu: MemorySource + 'static
{
    type Iter = VecIntoIter<Arc<AbstractBuffer>>;

    #[inline]
    fn decode(&self, source: (&'a Arc<Buffer<[T], Mt>>, &'a Arc<Buffer<[U], Mu>>))
              -> (VecIntoIter<Arc<AbstractBuffer>>, usize, usize)
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
            fn attrib(name: &str) -> Option<$crate::pipeline::vertex::AttributeInfo> {
                $(
                    if name == stringify!($member) {
                        return Some($crate::pipeline::vertex::AttributeInfo {
                            offset: unsafe {
                                let dummy = 0usize as *const $out;
                                let member = (&(&*dummy).$member) as *const _;
                                member as usize
                            },

                            format: unsafe {
                                #[inline] fn f<T: $crate::pipeline::vertex::Data>(_: &T) -> $crate::format::Format { T::format() }
                                let dummy = 0usize as *const $out;
                                f(&(&*dummy).$member)
                            },
                        });
                    }
                )*

                None
            }
        }
    )
}

/// Trait for data types that can be used as vertex attributes. Used by the `impl_vertex!` macro.
pub unsafe trait Data {
    /// Returns the format of the attribute.
    fn format() -> Format;
}

unsafe impl Data for f32 {
    #[inline]
    fn format() -> Format {
        Format::R32Sfloat
    }
}

unsafe impl Data for [f32; 1] {
    #[inline]
    fn format() -> Format {
        Format::R32Sfloat
    }
}

unsafe impl Data for [f32; 2] {
    #[inline]
    fn format() -> Format {
        Format::R32G32Sfloat
    }
}

unsafe impl Data for [f32; 3] {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32Sfloat
    }
}

unsafe impl Data for [f32; 4] {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32A32Sfloat
    }
}

unsafe impl Data for (f32,) {
    #[inline]
    fn format() -> Format {
        Format::R32Sfloat
    }
}

unsafe impl Data for (f32, f32) {
    #[inline]
    fn format() -> Format {
        Format::R32G32Sfloat
    }
}

unsafe impl Data for (f32, f32, f32) {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32Sfloat
    }
}

unsafe impl Data for (f32, f32, f32, f32) {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32A32Sfloat
    }
}

unsafe impl Data for u32 {
    #[inline]
    fn format() -> Format {
        Format::R32Uint
    }
}

unsafe impl Data for [u32; 1] {
    #[inline]
    fn format() -> Format {
        Format::R32Uint
    }
}

unsafe impl Data for [u32; 2] {
    #[inline]
    fn format() -> Format {
        Format::R32G32Uint
    }
}

unsafe impl Data for [u32; 3] {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32Uint
    }
}

unsafe impl Data for [u32; 4] {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32A32Uint
    }
}

unsafe impl Data for (u32,) {
    #[inline]
    fn format() -> Format {
        Format::R32Uint
    }
}

unsafe impl Data for (u32, u32) {
    #[inline]
    fn format() -> Format {
        Format::R32G32Uint
    }
}

unsafe impl Data for (u32, u32, u32) {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32Uint
    }
}

unsafe impl Data for (u32, u32, u32, u32) {
    #[inline]
    fn format() -> Format {
        Format::R32G32B32A32Uint
    }
}
