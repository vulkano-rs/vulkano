// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

mod bound;
mod builder;
pub mod persistent;
pub mod pool;

pub use self::persistent::PersistentDescriptorSet;
pub use self::pool::DescriptorSetPool;

use crate::descriptor_set::layout::DescriptorImageDescDimensions;
use crate::format::Format;
use crate::OomError;
use std::error;
use std::fmt;

/// Error related to descriptor sets.
#[derive(Debug, Clone)]
pub enum DescriptorSetError {
    /// Builder is already within an array.
    AlreadyInArray,

    /// Builder is not in an array.
    NotInArray,

    /// Array doesn't contain the correct amount of descriptors
    ArrayLengthMismatch {
        /// Expected length
        expected: u32,
        /// Obtained length
        obtained: u32,
    },

    /// Builder doesn't expect anymore descriptors
    TooManyDescriptors,

    /// Runtime arrays must be the last binding in a set.
    RuntimeArrayMustBeLast,

    /// Not all descriptors have been added.
    DescriptorsMissing {
        /// Expected bindings
        expected: usize,
        /// Obtained bindings
        obtained: usize,
    },

    /// The buffer is missing the correct usage.
    MissingBufferUsage(MissingBufferUsage),

    /// The image is missing the correct usage.
    MissingImageUsage(MissingImageUsage),

    /// Expected one type of resource but got another.
    WrongDescriptorType,

    /// Resource belongs to another device.
    ResourceWrongDevice,

    /// The format of an image view doesn't match what was expected.
    ImageViewFormatMismatch {
        /// Expected format.
        expected: Format,
        /// Format of the image view that was passed.
        obtained: Format,
    },

    /// The type of an image view doesn't match what was expected.
    ImageViewTypeMismatch {
        /// Expected type.
        expected: DescriptorImageDescDimensions,
        /// Type of the image view that was passed.
        obtained: DescriptorImageDescDimensions,
    },

    /// Expected a multisampled image, but got a single-sampled image.
    ExpectedMultisampled,

    /// Expected a single-sampled image, but got a multisampled image.
    UnexpectedMultisampled,

    /// The number of array layers of an image doesn't match what was expected.
    ArrayLayersMismatch {
        /// Number of expected array layers for the image.
        expected: u32,
        /// Number of array layers of the image that was added.
        obtained: u32,
    },

    /// The image view has a component swizzle that is different from identity.
    NotIdentitySwizzled,

    /// The image view isn't compatible with the sampler.
    IncompatibleImageViewSampler,

    /// The builder has previously return an error and is an unknown state.
    BuilderPoisoned,

    /// Out of memory
    OomError(OomError),

    /// Operation can not be performed on an empty descriptor.
    DescriptorIsEmpty,
}

impl From<OomError> for DescriptorSetError {
    fn from(error: OomError) -> Self {
        Self::OomError(error)
    }
}

impl error::Error for DescriptorSetError {}

impl fmt::Display for DescriptorSetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::AlreadyInArray => "builder is already within an array",
                Self::NotInArray => "builder is not in an array",
                Self::ArrayLengthMismatch { .. } =>
                    "array doesn't contain the correct amount of descriptors",
                Self::TooManyDescriptors => "builder doesn't expect anymore descriptors",
                Self::RuntimeArrayMustBeLast => "runtime arrays must be the last binding in a set",
                Self::DescriptorsMissing { .. } => "not all descriptors have been added",
                Self::MissingBufferUsage(_) => "the buffer is missing the correct usage",
                Self::MissingImageUsage(_) => "the image is missing the correct usage",
                Self::WrongDescriptorType => "expected one type of resource but got another",
                Self::ResourceWrongDevice => "resource belongs to another device",
                Self::ImageViewFormatMismatch { .. } =>
                    "the format of an image view doesn't match what was expected",
                Self::ImageViewTypeMismatch { .. } =>
                    "the type of an image view doesn't match what was expected",
                Self::ExpectedMultisampled =>
                    "expected a multisampled image, but got a single-sampled image",
                Self::UnexpectedMultisampled =>
                    "expected a single-sampled image, but got a multisampled image",
                Self::ArrayLayersMismatch { .. } =>
                    "the number of array layers of an image doesn't match what was expected",
                Self::NotIdentitySwizzled =>
                    "the image view has a component swizzle that is different from identity",
                Self::IncompatibleImageViewSampler =>
                    "the image view isn't compatible with the sampler",
                Self::BuilderPoisoned =>
                    "the builder has previously return an error and is an unknown state",
                Self::OomError(_) => "out of memory",
                Self::DescriptorIsEmpty => "operation can not be performed on an empty descriptor",
            }
        )
    }
}

// Part of the DescriptorSetError for the case
// of missing usage on a buffer.
#[derive(Debug, Clone)]
pub enum MissingBufferUsage {
    StorageBuffer,
    UniformBuffer,
    StorageTexelBuffer,
    UniformTexelBuffer,
}

// Part of the DescriptorSetError for the case
// of missing usage on an image.
#[derive(Debug, Clone)]
pub enum MissingImageUsage {
    InputAttachment,
    Sampled,
    Storage,
}
