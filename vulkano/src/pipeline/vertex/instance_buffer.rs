// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::mem;
use std::option::IntoIter as OptionIntoIter;
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

/// Same as `SingleBufferDefinition` but advances by instance.
pub struct SingleInstanceBufferDefinition<T>(pub PhantomData<T>);

impl<T> SingleInstanceBufferDefinition<T> {
    #[inline]
    pub fn new() -> SingleInstanceBufferDefinition<T> {
        SingleInstanceBufferDefinition(PhantomData)
    }
}

unsafe impl<T, I> VertexDefinition<I> for SingleInstanceBufferDefinition<T>
where
    T: Vertex,
    I: ShaderInterfaceDef,
{
    type BuffersIter = OptionIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

    fn definition(
        &self,
        interface: &I,
    ) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        let attrib = {
            let mut attribs = Vec::with_capacity(interface.elements().len());
            for e in interface.elements() {
                let name = e.name.as_ref().unwrap();

                let infos = match <T as Vertex>::member(name) {
                    Some(m) => m,
                    None => {
                        return Err(IncompatibleVertexDefinitionError::MissingAttribute {
                            attribute: name.clone().into_owned(),
                        })
                    }
                };

                if !infos.ty.matches(
                    infos.array_size,
                    e.format,
                    e.location.end - e.location.start,
                ) {
                    return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                        attribute: name.clone().into_owned(),
                        shader: (e.format, (e.location.end - e.location.start) as usize),
                        definition: (infos.ty, infos.array_size),
                    });
                }

                let mut offset = infos.offset;
                for loc in e.location.clone() {
                    attribs.push((
                        loc,
                        0,
                        AttributeInfo {
                            offset: offset,
                            format: e.format,
                        },
                    ));
                    offset += e.format.size().unwrap();
                }
            }
            attribs
        }
        .into_iter(); // TODO: meh

        let buffers = Some((0, mem::size_of::<T>(), InputRate::Instance)).into_iter();
        Ok((buffers, attrib))
    }
}

unsafe impl<V> VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>
    for SingleInstanceBufferDefinition<V>
where
    V: Vertex,
{
    #[inline]
    fn decode(
        &self,
        mut source: Vec<Arc<dyn BufferAccess + Send + Sync>>,
    ) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        // FIXME: safety
        assert_eq!(source.len(), 1);
        let len = source[0].size() / mem::size_of::<V>();
        (vec![Box::new(source.remove(0))], 1, len)
    }
}

unsafe impl<'a, B, V> VertexSource<B> for SingleInstanceBufferDefinition<V>
where
    B: TypedBufferAccess<Content = [V]> + Send + Sync + 'static,
    V: Vertex,
{
    #[inline]
    fn decode(&self, source: B) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        let len = source.len();
        (vec![Box::new(source) as Box<_>], 1, len)
    }
}
