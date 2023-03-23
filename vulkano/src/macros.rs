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
        $ty:ident
        $( impl { $($impls:item)* } )?
        = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident,
        )+
    } => {
        $(#[doc = $ty_doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $ty($repr);

        impl $ty {
            $(
                $(#[doc = $flag_doc])*
                pub const $flag_name: Self = Self(ash::vk::$ty_ffi::$flag_name_ffi.as_raw());
            )*

            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn empty() -> Self {
                Self(0)
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
                Self(Self::all_raw())
            }

            const fn all_raw() -> $repr {
                0
                $(
                    | ash::vk::$ty_ffi::$flag_name_ffi.as_raw()
                )*
            }

            /// Returns whether no flags are set in `self`.
            #[inline]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// Returns whether any flags are set in both `self` and `other`.
            #[inline]
            pub const fn intersects(self, #[allow(unused_variables)] other: Self) -> bool {
                self.0 & other.0 != 0
            }

            /// Returns whether all flags in `other` are set in `self`.
            #[inline]
            pub const fn contains(self, #[allow(unused_variables)] other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Returns the union of `self` and `other`.
            #[inline]
            pub const fn union(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// Returns the intersection of `self` and `other`.
            #[inline]
            pub const fn intersection(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 & other.0)
            }

            /// Returns `self` without the flags set in `other`.
            #[inline]
            pub const fn difference(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            /// Returns the flags that are set in `self` or `other`, but not in both.
            #[inline]
            pub const fn symmetric_difference(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 ^ other.0)
            }

            /// Returns the flags not in `self`.
            #[inline]
            pub const fn complement(self) -> Self {
                Self(!self.0 & Self::all_raw())
            }

            $( $($impls)* )?
        }

        impl Default for $ty {
            #[inline]
            fn default() -> Self {
                Self::empty()
            }
        }

        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                #[allow(unused_mut)]
                let mut written = false;

                $(
                    if self.intersects(Self::$flag_name) {
                        if written {
                            write!(f, " | ")?;
                        }

                        write!(f, stringify!($flag_name))?;
                        written = true;
                    }
                )*

                if !written {
                    write!(f, "empty()")?;
                }

                Ok(())
            }
        }

        impl From<$ty> for ash::vk::$ty_ffi {
            #[inline]
            fn from(val: $ty) -> Self {
                ash::vk::$ty_ffi::from_raw(val.0)
            }
        }

        impl From<ash::vk::$ty_ffi> for $ty {
            #[inline]
            fn from(val: ash::vk::$ty_ffi) -> Self {
                Self(val.as_raw() & Self::all_raw())
            }
        }

        impl std::ops::BitAnd for $ty {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.intersection(rhs)
            }
        }

        impl std::ops::BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(rhs);
            }
        }

        impl std::ops::BitOr for $ty {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl std::ops::BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(rhs);
            }
        }

        impl std::ops::BitXor for $ty {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.symmetric_difference(rhs)
            }
        }

        impl std::ops::BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(rhs);
            }
        }

        impl std::ops::Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.difference(rhs)
            }
        }

        impl std::ops::SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(rhs);
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
        #[non_exhaustive]

        $(#[doc = $ty_doc:literal])*
        $ty:ident
        $( impl { $($impls:item)* } )?
        = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                $(features: [$($feature:ident),+ $(,)?],)?
                $(device_extensions: [$($device_extension:ident),+ $(,)?],)?
                $(instance_extensions: [$($instance_extension:ident),+ $(,)?],)?
            })?
            ,
        )*
    } => {
        $(#[doc = $ty_doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $ty($repr);

        impl $ty {
            $(
                $(#[doc = $flag_doc])*
                pub const $flag_name: Self = Self(ash::vk::$ty_ffi::$flag_name_ffi.as_raw());
            )*

            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn empty() -> Self {
                Self(0)
            }

            #[deprecated(since = "0.31.0", note = "Use `empty` instead.")]
            #[doc = concat!("Returns a `", stringify!($ty), "` with none of the flags set.")]
            #[inline]
            pub const fn none() -> Self {
                Self::empty()
            }

            const fn all_raw() -> $repr {
                0
                $(
                    | ash::vk::$ty_ffi::$flag_name_ffi.as_raw()
                )*
            }

            /// Returns the number of flags set in self.
            #[inline]
            pub const fn count(self) -> u32 {
                self.0.count_ones()
            }

            /// Returns whether no flags are set in `self`.
            #[inline]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// Returns whether any flags are set in both `self` and `other`.
            #[inline]
            pub const fn intersects(self, #[allow(unused_variables)] other: Self) -> bool {
                self.0 & other.0 != 0
            }

            /// Returns whether all flags in `other` are set in `self`.
            #[inline]
            pub const fn contains(self, #[allow(unused_variables)] other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// Returns the union of `self` and `other`.
            #[inline]
            pub const fn union(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// Returns the intersection of `self` and `other`.
            #[inline]
            pub const fn intersection(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 & other.0)
            }

            /// Returns `self` without the flags set in `other`.
            #[inline]
            pub const fn difference(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            /// Returns the flags that are set in `self` or `other`, but not in both.
            #[inline]
            pub const fn symmetric_difference(self, #[allow(unused_variables)] other: Self) -> Self {
                Self(self.0 ^ other.0)
            }

            #[allow(dead_code)]
            pub(crate) fn validate_device(
                self,
                #[allow(unused_variables)] device: &crate::device::Device,
            ) -> Result<(), crate::RequirementNotMet> {
                $(
                    $(
                        if self.intersects(Self::$flag_name) && ![
                            $(
                                device.api_version() >= crate::Version::$api_version,
                            )?
                            $($(
                                device.enabled_features().$feature,
                            )+)?
                            $($(
                                device.enabled_extensions().$device_extension,
                            )+)?
                            $($(
                                device.instance().enabled_extensions().$instance_extension,
                            )+)?
                        ].into_iter().any(|x| x) {
                            return Err(crate::RequirementNotMet {
                                required_for: concat!("`", stringify!($ty), "::", stringify!($flag_name), "`"),
                                requires_one_of: crate::RequiresOneOf {
                                    $(api_version: Some(crate::Version::$api_version),)?
                                    $(features: &[$(stringify!($feature)),+],)?
                                    $(device_extensions: &[$(stringify!($device_extension)),+],)?
                                    $(instance_extensions: &[$(stringify!($instance_extension)),+],)?
                                    ..Default::default()
                                },
                            });
                        }
                    )?
                )*

                Ok(())
            }

            #[allow(dead_code)]
            pub(crate) fn validate_physical_device(
                self,
                #[allow(unused_variables)] physical_device: &crate::device::physical::PhysicalDevice,
            ) -> Result<(), crate::RequirementNotMet> {
                $(
                    $(
                        if self.intersects(Self::$flag_name) && ![
                            $(
                                physical_device.api_version() >= crate::Version::$api_version,
                            )?
                            $($(
                                physical_device.supported_features().$feature,
                            )+)?
                            $($(
                                physical_device.supported_extensions().$device_extension,
                            )+)?
                            $($(
                                physical_device.instance().enabled_extensions().$instance_extension,
                            )+)?
                        ].into_iter().any(|x| x) {
                            return Err(crate::RequirementNotMet {
                                required_for: concat!("`", stringify!($ty), "::", stringify!($flag_name), "`"),
                                requires_one_of: crate::RequiresOneOf {
                                    $(api_version: Some(crate::Version::$api_version),)?
                                    $(features: &[$(stringify!($feature)),+],)?
                                    $(device_extensions: &[$(stringify!($device_extension)),+],)?
                                    $(instance_extensions: &[$(stringify!($instance_extension)),+],)?
                                    ..Default::default()
                                },
                            });
                        }
                    )?
                )*

                Ok(())
            }

            #[allow(dead_code)]
            pub(crate) fn validate_instance(
                self,
                #[allow(unused_variables)] instance: &crate::instance::Instance,
            ) -> Result<(), crate::RequirementNotMet> {
                $(
                    $(
                        if self.intersects(Self::$flag_name) && ![
                            $(
                                instance.api_version() >= crate::Version::$api_version,
                            )?
                            $($(
                                instance.enabled_extensions().$instance_extension,
                            )+)?
                        ].into_iter().any(|x| x) {
                            return Err(crate::RequirementNotMet {
                                required_for: concat!("`", stringify!($ty), "::", stringify!($flag_name), "`"),
                                requires_one_of: crate::RequiresOneOf {
                                    $(api_version: Some(crate::Version::$api_version),)?
                                    $(instance_extensions: &[$(stringify!($instance_extension)),+],)?
                                    ..Default::default()
                                },
                            });
                        }
                    )?
                )*

                Ok(())
            }

            $( $($impls)* )?
        }

        impl Default for $ty {
            #[inline]
            fn default() -> Self {
                Self::empty()
            }
        }

        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                #[allow(unused_mut)]
                let mut written = false;

                $(
                    if self.intersects(Self::$flag_name) {
                        if written {
                            write!(f, " | ")?;
                        }

                        write!(f, stringify!($flag_name))?;
                        written = true;
                    }
                )*

                if !written {
                    write!(f, "empty()")?;
                }

                Ok(())
            }
        }

        impl From<$ty> for ash::vk::$ty_ffi {
            #[inline]
            fn from(val: $ty) -> Self {
                ash::vk::$ty_ffi::from_raw(val.0)
            }
        }

        impl From<ash::vk::$ty_ffi> for $ty {
            #[inline]
            fn from(val: ash::vk::$ty_ffi) -> Self {
                Self(val.as_raw() & Self::all_raw())
            }
        }

        impl std::ops::BitAnd for $ty {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.intersection(rhs)
            }
        }

        impl std::ops::BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(rhs);
            }
        }

        impl std::ops::BitOr for $ty {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl std::ops::BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(rhs);
            }
        }

        impl std::ops::BitXor for $ty {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.symmetric_difference(rhs)
            }
        }

        impl std::ops::BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(rhs);
            }
        }

        impl std::ops::Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.difference(rhs)
            }
        }

        impl std::ops::SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(rhs);
            }
        }
    };
}

macro_rules! vulkan_enum {
    {
        $(#[doc = $ty_doc:literal])*
        $ty:ident
        $( impl { $($impls:item)* } )?
        = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident,
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

        $(
            impl $ty {
                $($impls)*
            }
        )?

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
        #[non_exhaustive]

        $(#[doc = $ty_doc:literal])*
        $ty:ident
        $( impl { $($impls:item)* } )?
        = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                $(features: [$($feature:ident),+ $(,)?],)?
                $(device_extensions: [$($device_extension:ident),+ $(,)?],)?
                $(instance_extensions: [$($instance_extension:ident),+ $(,)?],)?
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
            pub(crate) fn validate_device(
                self,
                #[allow(unused_variables)] device: &crate::device::Device,
            ) -> Result<(), crate::RequirementNotMet> {
                match self {
                    $(
                        $(
                            Self::$flag_name => {
                                if ![
                                    $(
                                        device.api_version() >= crate::Version::$api_version,
                                    )?
                                    $($(
                                        device.enabled_features().$feature,
                                    )+)?
                                    $($(
                                        device.enabled_extensions().$device_extension,
                                    )+)?
                                    $($(
                                        device.instance().enabled_extensions().$instance_extension,
                                    )+)?
                                 ].into_iter().any(|x| x) {
                                    return Err(crate::RequirementNotMet {
                                        required_for: concat!("`", stringify!($ty), "::", stringify!($flag_name), "`"),
                                        requires_one_of: crate::RequiresOneOf {
                                            $(api_version: Some(crate::Version::$api_version),)?
                                            $(features: &[$(stringify!($feature)),+],)?
                                            $(device_extensions: &[$(stringify!($device_extension)),+],)?
                                            $(instance_extensions: &[$(stringify!($instance_extension)),+],)?
                                            ..Default::default()
                                        },
                                    });
                                }
                            },
                        )?
                    )+
                    _ => (),
                }

                Ok(())
            }

            #[allow(dead_code)]
            pub(crate) fn validate_physical_device(
                self,
                #[allow(unused_variables)] physical_device: &crate::device::physical::PhysicalDevice,
            ) -> Result<(), crate::RequirementNotMet> {
                match self {
                    $(
                        $(
                            Self::$flag_name => {
                                if ![
                                    $(
                                        physical_device.api_version() >= crate::Version::$api_version,
                                    )?
                                    $($(
                                        physical_device.supported_features().$feature,
                                    )+)?
                                    $($(
                                        physical_device.supported_extensions().$device_extension,
                                    )+)?
                                    $($(
                                        physical_device.instance().enabled_extensions().$instance_extension,
                                    )+)?
                                 ].into_iter().any(|x| x) {
                                    return Err(crate::RequirementNotMet {
                                        required_for: concat!("`", stringify!($ty), "::", stringify!($flag_name), "`"),
                                        requires_one_of: crate::RequiresOneOf {
                                            $(api_version: Some(crate::Version::$api_version),)?
                                            $(features: &[$(stringify!($feature)),+],)?
                                            $(device_extensions: &[$(stringify!($device_extension)),+],)?
                                            $(instance_extensions: &[$(stringify!($instance_extension)),+],)?
                                            ..Default::default()
                                        },
                                    });
                                }
                            },
                        )?
                    )+
                    _ => (),
                }

                Ok(())
            }

            #[allow(dead_code)]
            pub(crate) fn validate_instance(
                self,
                #[allow(unused_variables)] instance: &crate::instance::Instance,
            ) -> Result<(), crate::RequirementNotMet> {
                match self {
                    $(
                        $(
                            Self::$flag_name => {
                                if ![
                                    $(
                                        instance.api_version() >= crate::Version::$api_version,
                                    )?
                                    $($(
                                        instance.enabled_extensions().$instance_extension,
                                    )+)?
                                 ].into_iter().any(|x| x) {
                                    return Err(crate::RequirementNotMet {
                                        required_for: concat!("`", stringify!($ty), "::", stringify!($flag_name), "`"),
                                        requires_one_of: crate::RequiresOneOf {
                                            $(api_version: Some(crate::Version::$api_version),)?
                                            $(instance_extensions: &[$(stringify!($instance_extension)),+],)?
                                            ..Default::default()
                                        },
                                    });
                                }
                            },
                        )?
                    )+
                    _ => (),
                }

            Ok(())
        }

        $(
            $($impls)*
        )?
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

macro_rules! vulkan_bitflags_enum {
    {
        #[non_exhaustive]

        $(#[doc = $ty_bitflags_doc:literal])*
        $ty_bitflags:ident
        $( impl { $($impls_bitflags:item)* } )?
        ,

        $(#[doc = $ty_enum_doc:literal])*
        $ty_enum:ident
        $( impl { $($impls_enum:item)* } )?
        ,

        = $ty_ffi:ident($repr:ty);

        $(
            $(#[doc = $flag_doc:literal])*
            $flag_name_bitflags:ident, $flag_name_enum:ident = $flag_name_ffi:ident
            $({
                $(api_version: $api_version:ident,)?
                $(features: [$($feature:ident),+ $(,)?],)?
                $(device_extensions: [$($device_extension:ident),+ $(,)?],)?
                $(instance_extensions: [$($instance_extension:ident),+ $(,)?],)?
            })?
            ,
        )*
    } => {
        crate::macros::vulkan_bitflags! {
            #[non_exhaustive]

            $(#[doc = $ty_bitflags_doc])*
            $ty_bitflags
            impl {
                /// Returns whether `self` contains the flag corresponding to `val`.
                #[inline]
                pub fn contains_enum(self, val: $ty_enum) -> bool {
                    self.intersects(val.into())
                }

                $( $($impls_bitflags)* )?
            }
            = $ty_ffi($repr);

            $(
                $(#[doc = $flag_doc])*
                $flag_name_bitflags = $flag_name_ffi
                $({
                    $(api_version: $api_version,)?
                    $(features: [$($feature),+],)?
                    $(device_extensions: [$($device_extension),+],)?
                    $(instance_extensions: [$($instance_extension),+],)?
                })?
                ,
            )*
        }

        crate::macros::vulkan_enum! {
            #[non_exhaustive]

            $(#[doc = $ty_enum_doc])*
            $ty_enum
            $( impl { $($impls_enum)* } )?
            = $ty_ffi($repr);

            $(
                $(#[doc = $flag_doc])*
                $flag_name_enum = $flag_name_ffi
                $({
                    $(api_version: $api_version,)?
                    $(features: [$($feature),+],)?
                    $(device_extensions: [$($device_extension),+],)?
                    $(instance_extensions: [$($instance_extension),+],)?
                })?
                ,
            )*
        }

        impl From<$ty_enum> for $ty_bitflags {
            #[inline]
            fn from(val: $ty_enum) -> Self {
                Self(val as $repr)
            }
        }

        impl FromIterator<$ty_enum> for $ty_bitflags {
            #[inline]
            fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = $ty_enum> {
                iter.into_iter().map(|item| Self::from(item)).fold(Self::empty(), |r, i| r.union(i))
            }
        }

        impl IntoIterator for $ty_bitflags {
            type Item = $ty_enum;
            type IntoIter = std::iter::Flatten<
                std::array::IntoIter<
                    Option<Self::Item>,
                    { $ty_bitflags::all_raw().count_ones() as usize },
                >
            >;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                [
                    $(
                        self.intersects(Self::$flag_name_bitflags)
                            .then_some($ty_enum::$flag_name_enum),
                    )*
                ].into_iter().flatten()
            }
        }
    }
}

macro_rules! impl_id_counter {
    ($type:ident $(< $($param:ident $(: $bound:ident $(+ $bounds:ident)* )?),+ >)?) => {
        $crate::macros::impl_id_counter!(
            @inner $type $(< $($param),+ >)?, $( $($param $(: $bound $(+ $bounds)* )?),+)?
        );
    };
    ($type:ident $(< $($param:ident $(: $bound:ident $(+ $bounds:ident)* )? + ?Sized),+ >)?) => {
        $crate::macros::impl_id_counter!(
            @inner $type $(< $($param),+ >)?, $( $($param $(: $bound $(+ $bounds)* )? + ?Sized),+)?
        );
    };
    (@inner $type:ident $(< $($param:ident),+ >)?, $($bounds:tt)*) => {
        impl< $($bounds)* > $type $(< $($param),+ >)? {
            fn next_id() -> std::num::NonZeroU64 {
                use std::{
                    num::NonZeroU64,
                    sync::atomic::{AtomicU64, Ordering},
                };

                static COUNTER: AtomicU64 = AtomicU64::new(1);

                NonZeroU64::new(COUNTER.fetch_add(1, Ordering::Relaxed)).unwrap_or_else(|| {
                    println!("an ID counter has overflown ...somehow");
                    std::process::abort();
                })
            }
        }

        impl< $($bounds)* > PartialEq for $type $(< $($param),+ >)? {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }

        impl< $($bounds)* > Eq for $type $(< $($param),+ >)? {}

        impl< $($bounds)* > std::hash::Hash for $type $(< $($param),+ >)? {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.id.hash(state);
            }
        }
    };
}

// TODO: Replace with the `?` operator once its constness is stabilized.
macro_rules! try_opt {
    ($e:expr) => {
        if let Some(val) = $e {
            val
        } else {
            return None;
        }
    };
}

pub(crate) use {impl_id_counter, try_opt, vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum};
