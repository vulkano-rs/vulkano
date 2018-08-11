// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use buffer::BufferAccess;
use buffer::TypedBufferAccess;
use pipeline::shader::ShaderInterfaceDef;
use pipeline::vertex::AttributeInfo;
use pipeline::vertex::IncompatibleVertexDefinitionError;
use pipeline::vertex::InputRate;
use pipeline::vertex::Vertex;
use pipeline::vertex::VertexDefinition;
use pipeline::vertex::VertexSource;

/// Unstable.
// TODO: bad way to do things
pub struct OneVertexOneInstanceDefinition<T, U>(pub PhantomData<(T, U)>);

impl<T, U> OneVertexOneInstanceDefinition<T, U> {
    #[inline]
    pub fn new() -> OneVertexOneInstanceDefinition<T, U> {
        OneVertexOneInstanceDefinition(PhantomData)
    }
}

unsafe impl<T, U, I> VertexDefinition<I> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex,
          U: Vertex,
          I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(
        &self, interface: &I)
        -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        let attrib = {
            let mut attribs = Vec::with_capacity(interface.elements().len());
            for e in interface.elements() {
                let name = e.name.as_ref().unwrap();

                let (infos, buf_offset) = if let Some(infos) = <T as Vertex>::member(name) {
                    (infos, 0)
                } else if let Some(infos) = <U as Vertex>::member(name) {
                    (infos, 1)
                } else {
                    return Err(IncompatibleVertexDefinitionError::MissingAttribute {
                                   attribute: name.clone().into_owned(),
                               });
                };

                if !infos.ty.matches(infos.array_size,
                                     e.format,
                                     e.location.end - e.location.start)
                {
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
        }.into_iter(); // TODO: meh

        let buffers = vec![
            (0, mem::size_of::<T>(), InputRate::Vertex),
            (1, mem::size_of::<U>(), InputRate::Instance),
        ].into_iter();

        Ok((buffers, attrib))
    }
}

unsafe impl<T, U> VertexSource<Vec<Arc<BufferAccess + Send + Sync>>>
    for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex,
          U: Vertex
{
    #[inline]
    fn decode(&self, mut source: Vec<Arc<BufferAccess + Send + Sync>>)
              -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
        // FIXME: safety
        assert_eq!(source.len(), 2);
        let len = source[0].size() / mem::size_of::<T>();
        let inst = source[1].size() / mem::size_of::<U>();
        let s0 = source.remove(0);
        let s1 = source.remove(0);
        (vec![Box::new(s0) as Box<_>, Box::new(s1) as Box<_>], len, inst)
    }
}

unsafe impl<'a, T, U, Bt, Bu> VertexSource<(Bt, Bu)> for OneVertexOneInstanceDefinition<T, U>
    where T: Vertex,
          Bt: TypedBufferAccess<Content = [T]> + Send + Sync + 'static,
          U: Vertex,
          Bu: TypedBufferAccess<Content = [U]> + Send + Sync + 'static
{
    #[inline]
    fn decode(&self, source: (Bt, Bu)) -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
        let s1l = source.0.len();
        let s2l = source.1.len();
        (vec![Box::new(source.0) as Box<_>, Box::new(source.1) as Box<_>], s1l, s2l)
    }
}
