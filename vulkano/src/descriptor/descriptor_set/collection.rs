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
use std::vec::IntoIter as VecIntoIter;

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    /// An iterator that produces the list of descriptor set objects contained in this collection.
    type ListIter: ExactSizeIterator<Item = Arc<DescriptorSet>>;

    /// An iterator that produces the description of the list of sets.
    type SetsIter: ExactSizeIterator<Item = Self::DescIter>;

    /// An iterator that produces the description of a set.
    type DescIter: ExactSizeIterator<Item = DescriptorDesc>;

    /// Returns the list of descriptor set objects of this collection.
    fn list(&self) -> Self::ListIter;

    /// Produces a description of the sets, as if it was a layout.
    fn description(&self) -> Self::SetsIter;
}

unsafe impl DescriptorSetsCollection for () {
    type ListIter = EmptyIter<Arc<DescriptorSet>>;
    type SetsIter = EmptyIter<EmptyIter<DescriptorDesc>>;
    type DescIter = EmptyIter<DescriptorDesc>;

    #[inline]
    fn list(&self) -> Self::ListIter {
        iter::empty()
    }

    #[inline]
    fn description(&self) -> Self::SetsIter {
        iter::empty()
    }
}

unsafe impl<'a, T> DescriptorSetsCollection for Arc<T>
    where T: DescriptorSet + DescriptorSetDesc
{
    type ListIter = OptionIntoIter<Arc<DescriptorSet>>;
    type SetsIter = OptionIntoIter<Self::DescIter>;
    type DescIter = <T as DescriptorSetDesc>::Iter;

    #[inline]
    fn list(&self) -> Self::ListIter {
        Some(self.clone() as Arc<_>).into_iter()
    }

    #[inline]
    fn description(&self) -> Self::SetsIter {
        Some(self.desc()).into_iter()
    }
}

unsafe impl<'a, T> DescriptorSetsCollection for &'a Arc<T>
    where T: DescriptorSet + DescriptorSetDesc
{
    type ListIter = OptionIntoIter<Arc<DescriptorSet>>;
    type SetsIter = OptionIntoIter<Self::DescIter>;
    type DescIter = <T as DescriptorSetDesc>::Iter;

    #[inline]
    fn list(&self) -> Self::ListIter {
        Some((*self).clone() as Arc<_>).into_iter()
    }

    #[inline]
    fn description(&self) -> Self::SetsIter {
        Some(self.desc()).into_iter()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)*) => (
        unsafe impl<'a, $first$(, $others)*> DescriptorSetsCollection for
                                                        (&'a Arc<$first>, $(&'a Arc<$others>),*)
            where $first: DescriptorSet + DescriptorSetDesc + 'static
                  $(, $others: DescriptorSet + DescriptorSetDesc + 'static)*
        {
            type ListIter = VecIntoIter<Arc<DescriptorSet>>;
            type SetsIter = VecIntoIter<Self::DescIter>;
            type DescIter = VecIntoIter<DescriptorDesc>;

            #[inline]
            fn list(&self) -> Self::ListIter {
                #![allow(non_snake_case)]
                let ($first, $($others),*) = *self;

                let list = vec![
                    $first.clone() as Arc<_>,
                    $($others.clone() as Arc<_>),*
                ];

                list.into_iter()
            }

            #[inline]
            fn description(&self) -> Self::SetsIter {
                #![allow(non_snake_case)]
                let ($first, $($others),*) = *self;

                let mut list = Vec::new();
                list.push($first.desc().collect::<Vec<_>>().into_iter());
                $(list.push($others.desc().collect::<Vec<_>>().into_iter());)*
                list.into_iter()
            }
        }

        impl_collection!($($others),*);
    );

    () => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
