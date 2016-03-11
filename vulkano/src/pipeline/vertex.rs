use std::marker::PhantomData;
use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use buffer::Buffer;
use buffer::AbstractBuffer;
use format::Format;
use memory::MemorySourceChunk;
use vk;

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum InputRate {
    Vertex = vk::VERTEX_INPUT_RATE_VERTEX,
    Instance = vk::VERTEX_INPUT_RATE_INSTANCE,
}

/// Describes an individual `Vertex`. More precisely, a collection of attributes that can be read
/// from a vertex shader.
pub unsafe trait Vertex {
    /// Returns the characteristics of a vertex attribute.
    fn attrib(name: &str) -> Option<AttributeInfo>;
}

pub struct AttributeInfo {
    pub offset: usize,
    pub format: Format,
}

/// Trait for types that contain the definition of the vertex input.
pub unsafe trait Definition {
    type InfoIter: ExactSizeIterator<Item = (usize, InputRate)>;

    /// Returns information about an attribute, and the index of the buffer in which the attribute
    /// is found.
    fn attrib(&self, name: &str) -> Option<(usize, AttributeInfo)>;

    /// Produces an iterator that returns the stride (in bytes) and input rate of each buffer.
    fn buffers(&self) -> Self::InfoIter;
}

pub unsafe trait Source<L>: Definition {
    type Iter: ExactSizeIterator<Item = Arc<AbstractBuffer>>;

    /// Returns the list of buffers, number of vertices and number of instances.
    fn decode(&self, L) -> (Self::Iter, usize, usize);
}

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
    where V: Vertex + 'static, M: MemorySourceChunk + 'static
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

// TODO: shouldn't be just `Two` be `Multi`
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
    where T: Vertex + 'static, Mt: MemorySourceChunk + 'static,
          U: Vertex + 'static, Mu: MemorySourceChunk + 'static
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

/*pub struct OneVertexOneInstanceDefinition<V, I>;

unsafe impl<V, I> Definition for OneVertexOneInstanceDefinition<V, I>
    where V: Vertex, I: Vertex
{
    type InfoIter = OptionIntoIter<(u32, InputRate)>;

    #[inline]
    fn attrib(&self, name: &str) -> Option<(u32, AttributeInfo)> {
        Vertex::attrib(name).map(|info| (0, info))
    }

    #[inline]
    fn buffers(&self) -> Self::InfoIter {
        Some((mem::size_of::<T>(), InputRate::Vertex)).into_iter()
    }
}

unsafe impl<'a, V, S> Source<S> for OneVertexOneInstanceDefinition<V>
    where V: Vertex, S: Into<BufferSlice<'a, [V]>>
{
    type Iter = OptionIntoIter<Arc<AbstractBuffer>>;

    #[inline]
    fn iter(&self, source: S) -> OptionIntoIter<Arc<AbstractBuffer>> {
        let source = source.into();
        assert!(source.offset() == 0);      // TODO: not supported otherwise
        Some(source.buffer().clone() as Arc<_>).into_iter()
    }

    #[inline]
    fn num_vertices(&self, source: S) -> usize {
        source.into().len()
    }
}*/

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

pub unsafe trait Data {
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
