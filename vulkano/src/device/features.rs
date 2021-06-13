// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

macro_rules! features {
    {
        $($member:ident => {
            doc: $doc:expr,
			ffi_name: $ffi_field:ident,
            ffi_members: [$($ffi_struct:ident $(.$ffi_struct_field:ident)?),+],
            requires_features: [$($requires_feature:ident),*],
            conflicts_features: [$($conflicts_feature:ident),*],
            required_by_extensions: [$($required_by_extension:ident),*],
        },)*
    } => {
        /// Represents all the features that are available on a physical device or enabled on
        /// a logical device.
        ///
        /// Note that the `robust_buffer_access` is guaranteed to be supported by all Vulkan
        /// implementations.
        ///
        /// # Example
        ///
        /// ```
        /// use vulkano::device::Features;
        /// # let physical_device: vulkano::instance::PhysicalDevice = return;
        /// let minimal_features = Features {
        ///     geometry_shader: true,
        ///     .. Features::none()
        /// };
        ///
        /// let optimal_features = vulkano::device::Features {
        ///     geometry_shader: true,
        ///     tessellation_shader: true,
        ///     .. Features::none()
        /// };
        ///
        /// if !physical_device.supported_features().superset_of(&minimal_features) {
        ///     panic!("The physical device is not good enough for this application.");
        /// }
        ///
        /// assert!(optimal_features.superset_of(&minimal_features));
        /// let features_to_request = optimal_features.intersection(physical_device.supported_features());
        /// ```
        ///
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
        #[allow(missing_docs)]
        pub struct Features {
            $(
                #[doc = $doc]
                pub $member: bool,
            )*
        }

        impl Features {
            /// Checks enabled features against the device version, device extensions and each other.
            pub(super) fn check_requirements(
                &self,
                supported: &Features,
                api_version:
                crate::Version,
                extensions: &crate::device::DeviceExtensions,
            ) -> Result<(), crate::device::features::FeatureRestrictionError> {
                $(
                    if self.$member {
                        if !supported.$member {
                            return Err(crate::device::features::FeatureRestrictionError {
                                feature: stringify!($member),
                                restriction: crate::device::features::FeatureRestriction::NotSupported,
                            });
                        }

                        $(
                            if !self.$requires_feature {
                                return Err(crate::device::features::FeatureRestrictionError {
                                    feature: stringify!($member),
                                    restriction: crate::device::features::FeatureRestriction::RequiresFeature(stringify!($requires_feature)),
                                });
                            }
                        )*

                        $(
                            if self.$conflicts_feature {
                                return Err(crate::device::features::FeatureRestrictionError {
                                    feature: stringify!($member),
                                    restriction: crate::device::features::FeatureRestriction::ConflictsFeature(stringify!($conflicts_feature)),
                                });
                            }
                        )*
                    } else {
                        $(
                            if extensions.$required_by_extension {
                                return Err(crate::device::features::FeatureRestrictionError {
                                    feature: stringify!($member),
                                    restriction: crate::device::features::FeatureRestriction::RequiredByExtension(stringify!($required_by_extension)),
                                });
                            }
                        )*
                    }
                )*
                Ok(())
            }

            /// Builds a `Features` object with all values to false.
            pub fn none() -> Features {
                Features {
                    $($member: false,)*
                }
            }

            /// Builds a `Features` object with all values to true.
            ///
            /// > **Note**: This function is used for testing purposes, and is probably useless in
            /// > a real code.
            pub fn all() -> Features {
                Features {
                    $($member: true,)*
                }
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each feature of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub fn superset_of(&self, other: &Features) -> bool {
                $((self.$member == true || other.$member == false))&&+
            }

            /// Builds a `Features` that is the intersection of `self` and another `Features`
            /// object.
            ///
            /// The result's field will be true if it is also true in both `self` and `other`.
            pub fn intersection(&self, other: &Features) -> Features {
                Features {
                    $($member: self.$member && other.$member,)*
                }
            }

            /// Builds a `Features` that is the difference of another `Features` object from `self`.
            ///
            /// The result's field will be true if it is true in `self` but not `other`.
            pub fn difference(&self, other: &Features) -> Features {
                Features {
                    $($member: self.$member && !other.$member,)*
                }
            }
        }

        impl FeaturesFfi {
            pub(crate) fn write(&mut self, features: &Features) {
                $(
                    std::array::IntoIter::new([
                        $(self.$ffi_struct.as_mut().map(|s| &mut s$(.$ffi_struct_field)?.$ffi_field)),+
                    ]).flatten().next().map(|f| *f = features.$member as ash::vk::Bool32);
                )*
            }
        }

        impl From<&FeaturesFfi> for Features {
            fn from(features_ffi: &FeaturesFfi) -> Self {
                Features {
                    $(
                        $member: std::array::IntoIter::new([
                            $(features_ffi.$ffi_struct.map(|s| s$(.$ffi_struct_field)?.$ffi_field)),+
                        ]).flatten().next().unwrap_or(0) != 0,
                    )*
                }
            }
        }
    };
}

pub use crate::autogen::Features;
pub(crate) use features;

/// An error that can happen when enabling a feature on a device.
#[derive(Clone, Copy, Debug)]
pub struct FeatureRestrictionError {
    /// The feature in question.
    pub feature: &'static str,
    /// The restriction that was not met.
    pub restriction: FeatureRestriction,
}

impl error::Error for FeatureRestrictionError {}

impl fmt::Display for FeatureRestrictionError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "a restriction for the feature {} was not met: {}",
            self.feature, self.restriction,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FeatureRestriction {
    /// Not supported by the physical device.
    NotSupported,
    /// Requires a feature to be enabled.
    RequiresFeature(&'static str),
    /// Requires a feature to be disabled.
    ConflictsFeature(&'static str),
    /// An extension requires this feature to be enabled.
    RequiredByExtension(&'static str),
}

impl fmt::Display for FeatureRestriction {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            FeatureRestriction::NotSupported => {
                write!(fmt, "not supported by the physical device")
            }
            FeatureRestriction::RequiresFeature(feat) => {
                write!(fmt, "requires feature {} to be enabled", feat)
            }
            FeatureRestriction::ConflictsFeature(feat) => {
                write!(fmt, "requires feature {} to be disabled", feat)
            }
            FeatureRestriction::RequiredByExtension(ext) => {
                write!(fmt, "required to be enabled by extension {}", ext)
            }
        }
    }
}

macro_rules! features_ffi {
    {
        $api_version:ident,
        $device_extensions:ident,
        $instance_extensions:ident,
        $($member:ident => {
            ty: $ty:ident,
            provided_by: [$($provided_by:expr),+],
            conflicts: [$($conflicts:ident),*],
        },)+
    } => {
        #[derive(Default)]
        pub(crate) struct FeaturesFfi {
            features_vulkan10: Option<ash::vk::PhysicalDeviceFeatures2KHR>,

            $(
                $member: Option<ash::vk::$ty>,
            )+
        }

        impl FeaturesFfi {
            pub(crate) fn make_chain(
                &mut self,
                $api_version: crate::Version,
                $device_extensions: &DeviceExtensions,
                $instance_extensions: &InstanceExtensions,
            ) {
                self.features_vulkan10 = Some(Default::default());
                let head = self.features_vulkan10.as_mut().unwrap();

                $(
                    if std::array::IntoIter::new([$($provided_by),+]).any(|x| x) &&
                        std::array::IntoIter::new([$(self.$conflicts.is_none()),*]).all(|x| x) {
                        self.$member = Some(Default::default());
                        let member = self.$member.as_mut().unwrap();
                        member.p_next = head.p_next;
                        head.p_next = member as *mut _ as _;
                    }
                )+
            }

            pub(crate) fn head_as_ref(&self) -> &ash::vk::PhysicalDeviceFeatures2KHR {
                self.features_vulkan10.as_ref().unwrap()
            }

            pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceFeatures2KHR {
                self.features_vulkan10.as_mut().unwrap()
            }
        }
    };
}

pub(crate) use {crate::autogen::FeaturesFfi, features_ffi};
