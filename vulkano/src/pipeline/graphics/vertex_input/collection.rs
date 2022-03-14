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

impl<T: BufferAccessObject, const N: usize> VertexBuffersCollection for [T; N] {
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

#[cfg(test)]
mod tests {
    use super::VertexBuffersCollection;
    use crate::buffer::BufferAccess;
    use crate::buffer::BufferAccessObject;
    use crate::buffer::BufferInner;
    use crate::device::Device;
    use crate::device::DeviceOwned;
    use crate::device::Queue;
    use crate::sync::AccessError;
    use crate::DeviceSize;
    use std::sync::Arc;

    struct DummyBufferA {}
    struct DummyBufferB {}

    unsafe impl BufferAccess for DummyBufferA {
        fn inner(&self) -> BufferInner<'_> {
            unimplemented!()
        }

        fn size(&self) -> DeviceSize {
            unimplemented!()
        }

        fn conflict_key(&self) -> (u64, u64) {
            unimplemented!()
        }

        fn try_gpu_lock(&self, write: bool, queue: &Queue) -> Result<(), AccessError> {
            unimplemented!()
        }

        unsafe fn increase_gpu_lock(&self, write: bool) {
            unimplemented!()
        }

        unsafe fn unlock(&self, write: bool) {
            unimplemented!()
        }
    }

    unsafe impl DeviceOwned for DummyBufferA {
        fn device(&self) -> &Arc<Device> {
            unimplemented!()
        }
    }

    impl BufferAccessObject for Arc<DummyBufferA> {
        fn as_buffer_access_object(self: &Arc<DummyBufferA>) -> Arc<dyn BufferAccess> {
            self.clone()
        }
    }

    unsafe impl BufferAccess for DummyBufferB {
        fn inner(&self) -> BufferInner<'_> {
            unimplemented!()
        }

        fn size(&self) -> DeviceSize {
            unimplemented!()
        }

        fn conflict_key(&self) -> (u64, u64) {
            unimplemented!()
        }

        fn try_gpu_lock(&self, write: bool, queue: &Queue) -> Result<(), AccessError> {
            unimplemented!()
        }

        unsafe fn increase_gpu_lock(&self, write: bool) {
            unimplemented!()
        }

        unsafe fn unlock(&self, write: bool) {
            unimplemented!()
        }
    }

    unsafe impl DeviceOwned for DummyBufferB {
        fn device(&self) -> &Arc<Device> {
            unimplemented!()
        }
    }

    impl BufferAccessObject for Arc<DummyBufferB> {
        fn as_buffer_access_object(self: &Arc<DummyBufferB>) -> Arc<dyn BufferAccess> {
            self.clone()
        }
    }

    fn takes_collection<C: VertexBuffersCollection>(_: C) {}

    #[test]
    fn vertex_buffer_collection() {
        let concrete_a = Arc::new(DummyBufferA {});
        let concrete_b = Arc::new(DummyBufferB {});
        let concrete_aa = Arc::new(DummyBufferA {});
        let dynamic_a = concrete_a.clone() as Arc<dyn BufferAccess>;
        let dynamic_b = concrete_b.clone() as Arc<dyn BufferAccess>;

        // Concrete/Dynamic alone are valid
        takes_collection(concrete_a.clone());
        takes_collection(dynamic_a.clone());

        // Tuples of any variation are valid
        takes_collection((concrete_a.clone(), concrete_b.clone()));
        takes_collection((concrete_a.clone(), dynamic_b.clone()));
        takes_collection((dynamic_a.clone(), dynamic_b.clone()));

        // Vec need all the same type
        takes_collection(vec![concrete_a.clone(), concrete_aa.clone()]);
        takes_collection(vec![dynamic_a.clone(), dynamic_b.clone()]);
        // But casting the first or starting off with a dynamic will allow all variations
        takes_collection(vec![
            concrete_a.clone() as Arc<dyn BufferAccess>,
            concrete_b.clone(),
        ]);
        takes_collection(vec![dynamic_a.clone(), concrete_b.clone()]);

        // Arrays are similar to Vecs
        takes_collection([concrete_a.clone(), concrete_aa.clone()]);
        takes_collection([dynamic_a.clone(), dynamic_b.clone()]);
        takes_collection([
            concrete_a.clone() as Arc<dyn BufferAccess>,
            concrete_b.clone(),
        ]);
        takes_collection([dynamic_a.clone(), concrete_b.clone()]);
    }
}
