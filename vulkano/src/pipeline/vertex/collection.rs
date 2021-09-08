// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;

/// A collection of vertex buffers.
pub unsafe trait VertexBuffersCollection {
    /// Converts `self` into a list of buffers.
    // TODO: better than a Vec
    fn into_vec(self) -> Vec<Box<dyn BufferAccess>>;
}

unsafe impl VertexBuffersCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<Box<dyn BufferAccess>> {
        vec![]
    }
}

unsafe impl<T> VertexBuffersCollection for T
where
    T: BufferAccess + 'static,
{
    #[inline]
    fn into_vec(self) -> Vec<Box<dyn BufferAccess>> {
        vec![Box::new(self) as Box<_>]
    }
}

unsafe impl<T> VertexBuffersCollection for Vec<T>
where
    T: BufferAccess + 'static,
{
    #[inline]
    fn into_vec(self) -> Vec<Box<dyn BufferAccess>> {
        self.into_iter()
            .map(|source| Box::new(source) as Box<_>)
            .collect()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)+) => (
        unsafe impl<$first$(, $others)+> VertexBuffersCollection for ($first, $($others),+)
            where $first: BufferAccess + 'static
                  $(, $others: BufferAccess + 'static)*
        {
            #[inline]
            fn into_vec(self) -> Vec<Box<dyn BufferAccess>> {
                #![allow(non_snake_case)]

                let ($first, $($others,)*) = self;

                let mut list = Vec::new();
                list.push(Box::new($first) as Box<_>);
                $(
                    list.push(Box::new($others) as Box<_>);
                )+
                list
            }
        }

        impl_collection!($($others),+);
    );

    ($i:ident) => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
