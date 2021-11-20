// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use std::sync::Arc;

/// A collection of vertex buffers.
pub unsafe trait VertexBuffersCollection {
    /// Converts `self` into a list of buffers.
    // TODO: better than a Vec
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>>;
}

unsafe impl VertexBuffersCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
        vec![]
    }
}

unsafe impl<T> VertexBuffersCollection for Arc<T>
where
    T: BufferAccess + 'static,
{
    #[inline]
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
        vec![self as Arc<_>]
    }
}

unsafe impl<T> VertexBuffersCollection for Vec<Arc<T>>
where
    T: BufferAccess + 'static,
{
    #[inline]
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
        self.into_iter().map(|source| source as Arc<_>).collect()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)+) => (
        unsafe impl<$first$(, $others)+> VertexBuffersCollection for (Arc<$first>, $(Arc<$others>),+)
            where $first: BufferAccess + 'static
                  $(, $others: BufferAccess + 'static)*
        {
            #[inline]
            fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
                #![allow(non_snake_case)]

                let ($first, $($others,)*) = self;

                let mut list = Vec::new();
                list.push($first as Arc<_>);
                $(
                    list.push($others as Arc<_>);
                )+
                list
            }
        }

        impl_collection!($($others),+);
    );

    ($i:ident) => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
