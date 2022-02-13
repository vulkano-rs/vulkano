// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::buffer::BufferAccessObject;
use std::sync::Arc;

/// A collection of vertex buffers.
pub trait VertexBuffersCollection {
    /// Converts `self` into a list of buffers.
    // TODO: better than a Vec
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>>;
}

impl VertexBuffersCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
        Vec::new()
    }
}

impl<T: BufferAccessObject> VertexBuffersCollection for T {
    #[inline]
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
        vec![self.as_buffer_access_object()]
    }
}

impl<T: BufferAccessObject> VertexBuffersCollection for Vec<T> {
    #[inline]
    fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
        self.into_iter()
            .map(|src| src.as_buffer_access_object())
            .collect()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)+) => (
        impl<$first$(, $others)+> VertexBuffersCollection for ($first, $($others),+)
            where $first: BufferAccessObject
                  $(, $others: BufferAccessObject)*
        {
            #[inline]
            fn into_vec(self) -> Vec<Arc<dyn BufferAccess>> {
                #![allow(non_snake_case)]

                let ($first, $($others,)*) = self;
                let mut list = Vec::new();
                list.push($first.as_buffer_access_object());

                $(
                    list.push($others.as_buffer_access_object());
                )+

                list
            }
        }

        impl_collection!($($others),+);
    );

    ($i:ident) => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
