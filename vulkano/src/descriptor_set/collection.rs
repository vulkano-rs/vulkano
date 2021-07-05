// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor_set::layout::DescriptorDesc;
use crate::descriptor_set::DescriptorSet;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    fn into_vec(self) -> Vec<Box<dyn DescriptorSet + Send + Sync>>;

    fn set(&self, num: usize) -> Option<&(dyn DescriptorSet + Send + Sync)>;

    /// Returns the number of descriptors in the set. Includes possibly empty descriptors.
    ///
    /// Returns `None` if the set is out of range.
    // TODO: remove ; user should just use `into_vec` instead
    fn num_bindings_in_set(&self, set: usize) -> Option<usize>;

    /// Returns the descriptor for the given binding of the given set.
    ///
    /// Returns `None` if out of range.
    // TODO: remove ; user should just use `into_vec` instead
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc>;
}

unsafe impl DescriptorSetsCollection for () {
    #[inline]
    fn into_vec(self) -> Vec<Box<dyn DescriptorSet + Send + Sync>> {
        vec![]
    }

    #[inline]
    fn set(&self, num: usize) -> Option<&(dyn DescriptorSet + Send + Sync)> {
        None
    }

    #[inline]
    fn num_bindings_in_set(&self, _: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn descriptor(&self, _: usize, _: usize) -> Option<DescriptorDesc> {
        None
    }
}

unsafe impl<T> DescriptorSetsCollection for T
where
    T: DescriptorSet + Send + Sync + 'static,
{
    #[inline]
    fn into_vec(self) -> Vec<Box<dyn DescriptorSet + Send + Sync>> {
        vec![Box::new(self) as Box<_>]
    }

    #[inline]
    fn set(&self, num: usize) -> Option<&(dyn DescriptorSet + Send + Sync)> {
        match num {
            0 => Some(self),
            _ => None,
        }
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0 => Some(self.layout().num_bindings()),
            _ => None,
        }
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        match set {
            0 => self.layout().descriptor(binding),
            _ => None,
        }
    }
}

unsafe impl<T> DescriptorSetsCollection for Vec<T>
where
    T: DescriptorSet + Send + Sync + 'static,
{
    #[inline]
    fn into_vec(self) -> Vec<Box<dyn DescriptorSet + Send + Sync>> {
        let mut v = Vec::new();
        for o in self {
            v.push(Box::new(o) as Box<_>);
        }
        return v;
    }

    #[inline]
    fn set(&self, num: usize) -> Option<&(dyn DescriptorSet + Send + Sync)> {
        self.get(num).map(|x| x as _)
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.get(set).map(|x| x.layout().num_bindings())
    }
    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.get(set).and_then(|x| x.layout().descriptor(binding))
    }
}

macro_rules! impl_collection {
    ($first:ident $(, $others:ident)+) => (
        unsafe impl<$first$(, $others)+> DescriptorSetsCollection for ($first, $($others),+)
            where $first: DescriptorSet + Send + Sync + 'static
                  $(, $others: DescriptorSet + Send + Sync + 'static)*
        {
            #[inline]
            fn into_vec(self) -> Vec<Box<dyn DescriptorSet + Send + Sync>> {
                #![allow(non_snake_case)]

                let ($first, $($others,)*) = self;

                let mut list = Vec::new();
                list.push(Box::new($first) as Box<_>);
                $(
                    list.push(Box::new($others) as Box<_>);
                )+
                list
            }

            #[inline]
            fn set(&self, mut num: usize) -> Option<&(dyn DescriptorSet + Send + Sync)> {
                #![allow(non_snake_case)]
                #![allow(unused_mut)]       // For the `num` parameter.

                if num == 0 {
                    return Some(&self.0);
                }

                let &(_, $(ref $others,)*) = self;

                $(
                    num -= 1;
                    if num == 0 {
                        return Some($others);
                    }
                )*

                None
            }

            #[inline]
            fn num_bindings_in_set(&self, mut set: usize) -> Option<usize> {
                #![allow(non_snake_case)]
                #![allow(unused_mut)]       // For the `set` parameter.

                if set == 0 {
                    return Some(self.0.layout().num_bindings());
                }

                let &(_, $(ref $others,)*) = self;

                $(
                    set -= 1;
                    if set == 0 {
                        return Some($others.layout().num_bindings());
                    }
                )*

                None
            }

            #[inline]
            fn descriptor(&self, mut set: usize, binding: usize) -> Option<DescriptorDesc> {
                #![allow(non_snake_case)]
                #![allow(unused_mut)]       // For the `set` parameter.

                if set == 0 {
                    return self.0.layout().descriptor(binding);
                }

                let &(_, $(ref $others,)*) = self;

                $(
                    set -= 1;
                    if set == 0 {
                        return $others.layout().descriptor(binding);
                    }
                )*

                None
            }
        }

        impl_collection!($($others),+);
    );

    ($i:ident) => ();
}

impl_collection!(Z, Y, X, W, V, U, T, S, R, Q, P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
