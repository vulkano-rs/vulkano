use crate::buffer::Subbuffer;
use std::mem;

/// A collection of vertex buffers.
pub trait VertexBuffersCollection {
    /// Converts `self` into a list of buffers.
    // TODO: better than a Vec
    fn into_vec(self) -> Vec<Subbuffer<[u8]>>;
}

impl VertexBuffersCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<Subbuffer<[u8]>> {
        Vec::new()
    }
}

impl<T: ?Sized> VertexBuffersCollection for Subbuffer<T> {
    fn into_vec(self) -> Vec<Subbuffer<[u8]>> {
        vec![self.into_bytes()]
    }
}

impl<T: ?Sized> VertexBuffersCollection for Vec<Subbuffer<T>> {
    fn into_vec(self) -> Vec<Subbuffer<[u8]>> {
        assert_eq!(
            mem::size_of::<Subbuffer<T>>(),
            mem::size_of::<Subbuffer<[u8]>>()
        );
        assert_eq!(
            mem::align_of::<Subbuffer<T>>(),
            mem::align_of::<Subbuffer<[u8]>>()
        );

        // SAFETY: All `Subbuffer`s share the same layout.
        unsafe { mem::transmute::<Vec<Subbuffer<T>>, Vec<Subbuffer<[u8]>>>(self) }
    }
}

impl<T: ?Sized, const N: usize> VertexBuffersCollection for [Subbuffer<T>; N] {
    fn into_vec(self) -> Vec<Subbuffer<[u8]>> {
        self.into_iter().map(Subbuffer::into_bytes).collect()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)*) => (
        impl<$first: ?Sized $(, $others: ?Sized)*> VertexBuffersCollection
            for (Subbuffer<$first>, $(Subbuffer<$others>),*)
        {
            #[inline]
            #[allow(non_snake_case)]
            fn into_vec(self) -> Vec<Subbuffer<[u8]>> {
                let ($first, $($others,)*) = self;
                vec![$first.into_bytes() $(, $others.into_bytes())*]
            }
        }

        impl_collection!($($others),*);
    );
    () => {}
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
