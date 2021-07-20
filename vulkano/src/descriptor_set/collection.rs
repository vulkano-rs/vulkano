// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor_set::DescriptorSetWithOffsets;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets>;
}

unsafe impl DescriptorSetsCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
        vec![]
    }
}

unsafe impl<T> DescriptorSetsCollection for T
where
    T: Into<DescriptorSetWithOffsets>,
{
    #[inline]
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
        vec![self.into()]
    }
}

unsafe impl<T> DescriptorSetsCollection for Vec<T>
where
    T: Into<DescriptorSetWithOffsets>,
{
    #[inline]
    fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
        self.into_iter().map(|x| x.into()).collect()
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)+) => (
        unsafe impl<$first$(, $others)+> DescriptorSetsCollection for ($first, $($others),+)
            where $first: Into<DescriptorSetWithOffsets>
                  $(, $others: Into<DescriptorSetWithOffsets>)*
        {
            #[inline]
            fn into_vec(self) -> Vec<DescriptorSetWithOffsets> {
                #![allow(non_snake_case)]

                let ($first, $($others,)*) = self;

                let mut list = Vec::new();
                list.push($first.into());
                $(
                    list.push($others.into());
                )+
                list
            }
        }

        impl_collection!($($others),+);
    );

    ($i:ident) => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
