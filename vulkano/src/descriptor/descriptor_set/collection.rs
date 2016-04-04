// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::iter::Empty as EmptyIter;
use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;
//use std::vec::IntoIter as VecIntoIter;

use descriptor::descriptor_set::DescriptorSet;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    /// An iterator that produces the list of descriptor set objects contained in this collection.
    type Iter: ExactSizeIterator<Item = Arc<DescriptorSet>>;

    /// Returns the list of descriptor set objects of this collection.
    fn list(&self) -> Self::Iter;
}

unsafe impl DescriptorSetsCollection for () {
    type Iter = EmptyIter<Arc<DescriptorSet>>;

    #[inline]
    fn list(&self) -> Self::Iter {
        iter::empty()
    }
}

unsafe impl<'a, T> DescriptorSetsCollection for Arc<T>
    where T: DescriptorSet
{
    type Iter = OptionIntoIter<Arc<DescriptorSet>>;

    #[inline]
    fn list(&self) -> Self::Iter {
        Some(self.clone() as Arc<_>).into_iter()
    }
}

unsafe impl<'a, T> DescriptorSetsCollection for &'a Arc<T>
    where T: DescriptorSet
{
    type Iter = OptionIntoIter<Arc<DescriptorSet>>;

    #[inline]
    fn list(&self) -> Self::Iter {
        Some((*self).clone() as Arc<_>).into_iter()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)*) => (
        /*unsafe impl<'a, $first$(, $others)*> DescriptorSetsCollection for
                                                        (&'a Arc<$first>, $(&'a Arc<$others>),*)
        {
            type Iter = VecIntoIter<Arc<DescriptorSet>>;

            #[inline]
            fn list(&self) -> Self::Iter {
                let list = vec![
                    $first,
                    $($others),*
                ];

                list.into_iter()
            }
        }*/

        impl_collection!($($others),*);
    );

    () => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
