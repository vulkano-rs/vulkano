mod bound;
mod builder;
pub mod persistent;

use crate::descriptor_set::layout::DescriptorImageDescDimensions;
use crate::descriptor_set::persistent::{MissingBufferUsage, MissingImageUsage};
use crate::format::Format;
use crate::OomError;
use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum RuntimeDescriptorSetError {
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
}

impl From<OomError> for RuntimeDescriptorSetError {
    fn from(error: OomError) -> Self {
        Self::OomError(error)
    }
}

impl error::Error for RuntimeDescriptorSetError {}

impl fmt::Display for RuntimeDescriptorSetError {
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
            }
        )
    }
}
