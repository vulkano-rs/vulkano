mod bound;
pub mod persistent;
pub mod builder;

use crate::descriptor_set::layout::DescriptorImageDescDimensions;
use crate::descriptor_set::persistent::{MissingBufferUsage, MissingImageUsage};
use crate::format::Format;

pub enum RuntimeDescriptorSetError {
    /// Builder is already within an array.
    AlreadyInArray,

    /// Builder is not in an array.
    NotInArray,

    /// Array doesn't contain the correct amount of descriptors
    ArrayLengthMismatch {
        /// Expected length
        expected: usize,
        /// Obtained length
        obtained: usize,
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
}
