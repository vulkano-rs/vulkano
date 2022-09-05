// Copyright (c) 2022 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

macro_rules! vulkan_bitflags {
    {
		$(#[doc = $ty_doc:literal])*
        $ty:ident = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                extensions: [$($extension:ident),+ $(,)?],
            })?
            ,
        )+
    } => {
		$(#[doc = $ty_doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $ty {
            $(
                $(#[doc = $flag_doc])*
                pub $flag_name: bool,
            )+
        }

        impl $ty {
            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn empty() -> Self {
                Self {
                    $(
                        $flag_name: false,
                    )+
                }
            }

            #[deprecated(since = "0.31.0", note = "Use `empty` instead.")]
            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn none() -> Self {
                Self::empty()
            }

            #[doc = concat!("Returns a `", stringify!($ty), "` with all of the flags set.")]
            #[inline]
            pub const fn all() -> Self {
                Self {
                    $(
                        $flag_name: true,
                    )+
                }
            }

            /// Returns whether no flags are set in `self`.
            #[inline]
            pub const fn is_empty(&self) -> bool {
                !(
                    $(
                        self.$flag_name
                    )||+
                )
            }

            /// Returns whether any flags are set in both `self` and `other`.
            #[inline]
            pub const fn intersects(&self, other: &Self) -> bool {
                $(
                    (self.$flag_name && other.$flag_name)
                )||+
            }

            /// Returns whether all flags in `other` are set in `self`.
            #[inline]
            pub const fn contains(&self, other: &Self) -> bool {
                $(
                    (self.$flag_name || !other.$flag_name)
                )&&+
            }

            /// Returns the union of `self` and `other`.
            #[inline]
            pub const fn union(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name || other.$flag_name),
                    )+
                }
            }

            /// Returns the intersection of `self` and `other`.
            #[inline]
            pub const fn intersection(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name && other.$flag_name),
                    )+
                }
            }

            /// Returns `self` without the flags set in `other`.
            #[inline]
            pub const fn difference(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name && !other.$flag_name),
                    )+
                }
            }

            /// Returns the flags set in `self` or `other`, but not both.
            #[inline]
            pub const fn symmetric_difference(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name ^ other.$flag_name),
                    )+
                }
            }

            /// Returns the flags not in `self`.
            #[inline]
            pub const fn complement(&self) -> Self {
                Self {
                    $(
                        $flag_name: !self.$flag_name,
                    )+
                }
            }

            #[allow(dead_code)]
            pub(crate) fn validate(
                self,
                #[allow(unused_variables)] device: &crate::device::Device
            ) -> Result<(), crate::macros::ExtensionNotEnabled> {
                $(
                    $(
                        if self.$flag_name && !(
                            $(
                                device.api_version() >= crate::Version::$api_version ||
                            )?
                            $(
                                device.enabled_extensions().$extension
                            )||+
                        ) {
                            return Err(crate::macros::ExtensionNotEnabled {
                                extension: stringify!($($extension)?),
                                reason: concat!(stringify!($ty), "::", stringify!($flag_name), " was used"),
                            });
                        }
                    )?
                )+

                Ok(())
            }
        }

		impl From<$ty> for ash::vk::$ty_ffi {
			#[inline]
			fn from(val: $ty) -> Self {
				let mut result = ash::vk::$ty_ffi::empty();
				$(
					if val.$flag_name { result |= ash::vk::$ty_ffi::$flag_name_ffi }
				)+
				result
			}
		}

        impl From<ash::vk::$ty_ffi> for $ty {
			#[inline]
			fn from(val: ash::vk::$ty_ffi) -> Self {
                Self {
                    $(
                        $flag_name: val.intersects(ash::vk::$ty_ffi::$flag_name_ffi),
                    )+
                }
			}
		}

        impl Default for $ty {
            #[inline]
            fn default() -> Self {
                Self {
                    $(
                        $flag_name: false,
                    )+
                }
            }
        }

        impl std::ops::BitAnd for $ty {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.intersection(&rhs)
            }
        }

        impl std::ops::BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(&rhs);
            }
        }

        impl std::ops::BitOr for $ty {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(&rhs)
            }
        }

        impl std::ops::BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(&rhs);
            }
        }

        impl std::ops::BitXor for $ty {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.symmetric_difference(&rhs)
            }
        }

        impl std::ops::BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(&rhs);
            }
        }

        impl std::ops::Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.difference(&rhs)
            }
        }

        impl std::ops::SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(&rhs);
            }
        }

        impl std::ops::Not for $ty {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                self.complement()
            }
        }
    };

    {
		$(#[doc = $ty_doc:literal])*
        #[non_exhaustive]
        $ty:ident = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                extensions: [$($extension:ident),+ $(,)?],
            })?
            ,
        )+
    } => {
		$(#[doc = $ty_doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $ty {
            $(
                $(#[doc = $flag_doc])*
                pub $flag_name: bool,
            )+
            pub _ne: crate::NonExhaustive,
        }

        impl $ty {
            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn empty() -> Self {
                Self {
                    $(
                        $flag_name: false,
                    )+
                    _ne: crate::NonExhaustive(()),
                }
            }

            #[deprecated(since = "0.31.0", note = "Use `empty` instead.")]
            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn none() -> Self {
                Self::empty()
            }

            /// Returns whether no flags are set in `self`.
            #[inline]
            pub const fn is_empty(&self) -> bool {
                !(
                    $(
                        self.$flag_name
                    )||+
                )
            }

            /// Returns whether any flags are set in both `self` and `other`.
            #[inline]
            pub const fn intersects(&self, other: &Self) -> bool {
                $(
                    (self.$flag_name && other.$flag_name)
                )||+
            }

            /// Returns whether all flags in `other` are set in `self`.
            #[inline]
            pub const fn contains(&self, other: &Self) -> bool {
                $(
                    (self.$flag_name || !other.$flag_name)
                )&&+
            }

            /// Returns the union of `self` and `other`.
            #[inline]
            pub const fn union(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name || other.$flag_name),
                    )+
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns the intersection of `self` and `other`.
            #[inline]
            pub const fn intersection(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name && other.$flag_name),
                    )+
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns `self` without the flags set in `other`.
            #[inline]
            pub const fn difference(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name ^ other.$flag_name),
                    )+
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns the flags set in `self` or `other`, but not both.
            #[inline]
            pub const fn symmetric_difference(&self, other: &Self) -> Self {
                Self {
                    $(
                        $flag_name: (self.$flag_name ^ other.$flag_name),
                    )+
                    _ne: crate::NonExhaustive(()),
                }
            }

            #[allow(dead_code)]
            pub(crate) fn validate(
                self,
                #[allow(unused_variables)] device: &crate::device::Device
            ) -> Result<(), crate::macros::ExtensionNotEnabled> {
                $(
                    $(
                        if self.$flag_name && !(
                            $(
                                device.api_version() >= crate::Version::$api_version ||
                            )?
                            $(
                                device.enabled_extensions().$extension
                            )||+
                        ) {
                            return Err(crate::macros::ExtensionNotEnabled {
                                extension: stringify!($($extension)?),
                                reason: concat!(stringify!($ty), "::", stringify!($flag_name), " was used"),
                            });
                        }
                    )?
                )+

                Ok(())
            }
        }

		impl From<$ty> for ash::vk::$ty_ffi {
			#[inline]
			fn from(val: $ty) -> Self {
				let mut result = ash::vk::$ty_ffi::empty();
				$(
					if val.$flag_name { result |= ash::vk::$ty_ffi::$flag_name_ffi }
				)+
				result
			}
		}

        impl From<ash::vk::$ty_ffi> for $ty {
			#[inline]
			fn from(val: ash::vk::$ty_ffi) -> Self {
                Self {
                    $(
                        $flag_name: val.intersects(ash::vk::$ty_ffi::$flag_name_ffi),
                    )+
                    _ne: crate::NonExhaustive(()),
                }
			}
		}

        impl Default for $ty {
            #[inline]
            fn default() -> Self {
                Self {
                    $(
                        $flag_name: false,
                    )+
                    _ne: crate::NonExhaustive(()),
                }
            }
        }

        impl std::ops::BitAnd for $ty {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.intersection(&rhs)
            }
        }

        impl std::ops::BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(&rhs);
            }
        }

        impl std::ops::BitOr for $ty {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(&rhs)
            }
        }

        impl std::ops::BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(&rhs);
            }
        }

        impl std::ops::BitXor for $ty {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.symmetric_difference(&rhs)
            }
        }

        impl std::ops::BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(&rhs);
            }
        }

        impl std::ops::Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.difference(&rhs)
            }
        }

        impl std::ops::SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(&rhs);
            }
        }
    };
}

macro_rules! vulkan_enum {
    {
		$(#[doc = $ty_doc:literal])*
        $ty:ident = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                extensions: [$($extension:ident),+ $(,)?],
            })?
            ,
        )+
    } => {
		$(#[doc = $ty_doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr($repr)]
        pub enum $ty {
            $(
                $(#[doc = $flag_doc])*
                $flag_name = ash::vk::$ty_ffi::$flag_name_ffi.as_raw(),
            )+
        }

		impl From<$ty> for ash::vk::$ty_ffi {
			#[inline]
			fn from(val: $ty) -> Self {
                ash::vk::$ty_ffi::from_raw(val as $repr)
			}
		}

        impl TryFrom<ash::vk::$ty_ffi> for $ty {
            type Error = ();

			#[inline]
			fn try_from(val: ash::vk::$ty_ffi) -> Result<Self, Self::Error> {
                Ok(match val {
                    $(
                        ash::vk::$ty_ffi::$flag_name_ffi => Self::$flag_name,
                    )+
                    _ => return Err(()),
                })
			}
		}
    };

    {
		$(#[doc = $ty_doc:literal])*
        #[non_exhaustive]
        $ty:ident = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                extensions: [$($extension:ident),+ $(,)?],
            })?
            ,
        )+
    } => {
		$(#[doc = $ty_doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        #[repr($repr)]
        pub enum $ty {
            $(
                $(#[doc = $flag_doc])*
                $flag_name = ash::vk::$ty_ffi::$flag_name_ffi.as_raw(),
            )+
        }

        impl $ty {
            #[allow(dead_code)]
            pub(crate) fn validate(
                self,
                #[allow(unused_variables)] device: &crate::device::Device
            ) -> Result<(), crate::macros::ExtensionNotEnabled> {
                match self {
                    $(
                        $(
                            Self::$flag_name => {
                                if !(
                                    $(
                                        device.api_version() >= crate::Version::$api_version ||
                                    )?
                                    $(
                                        device.enabled_extensions().$extension
                                    )||+
                                ) {
                                    return Err(crate::macros::ExtensionNotEnabled {
                                        extension: stringify!($($extension)?),
                                        reason: concat!(stringify!($ty), "::", stringify!($flag_name), " was used"),
                                    });
                                }
                            },
                        )?
                    )+
                    _ => (),
                }

                Ok(())
            }
        }

		impl From<$ty> for ash::vk::$ty_ffi {
			#[inline]
			fn from(val: $ty) -> Self {
                ash::vk::$ty_ffi::from_raw(val as $repr)
			}
		}

        impl TryFrom<ash::vk::$ty_ffi> for $ty {
            type Error = ();

			#[inline]
			fn try_from(val: ash::vk::$ty_ffi) -> Result<Self, Self::Error> {
                Ok(match val {
                    $(
                        ash::vk::$ty_ffi::$flag_name_ffi => Self::$flag_name,
                    )+
                    _ => return Err(()),
                })
			}
		}
    };
}

pub(crate) use {vulkan_bitflags, vulkan_enum};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExtensionNotEnabled {
    pub(crate) extension: &'static str,
    pub(crate) reason: &'static str,
}
