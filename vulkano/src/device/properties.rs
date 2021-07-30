use crate::device::physical::{
    ConformanceVersion, DriverId, PhysicalDeviceType, PointClippingBehavior, ShaderCoreProperties,
    ShaderFloatControlsIndependence, SubgroupFeatures,
};
use crate::image::{SampleCount, SampleCounts};
use crate::pipeline::shader::ShaderStages;
use crate::render_pass::ResolveModes;
use crate::Version;
use std::convert::TryInto;
use std::ffi::CStr;

/// this is a macro that outputs either T or Option<T> depending on the 2nd argument
macro_rules! property_type {
    ($ty:ty, true) => {
        $ty
    };
    ($ty:ty, false) => {
        Option<$ty>
    };
}

pub(crate) use property_type;

/// this is a macro that executes the correct from_vulkan call depending on whether or not the type is Option<T>
macro_rules! property_from_vulkan {
    ($ty:ty, [$($ffi_struct:ident $(.$ffi_struct_field:ident)*),+], $ffi_field:ident, true, $properties:ident) => {
        std::array::IntoIter::new([
            $($properties.$ffi_struct$(.$ffi_struct_field)*.$ffi_field),+
        ]).next().and_then(|x| <$ty>::from_vulkan(x)).expect(concat!("expected good ", stringify!($ffi_field)))
    };
    ($ty:ty, [$($ffi_struct:ident $(.$ffi_struct_field:ident)*),+], $ffi_field:ident, false, $properties:ident) => {
        std::array::IntoIter::new([
            $($properties.$ffi_struct.map(|s| s$(.$ffi_struct_field)*.$ffi_field)),+
        ]).flatten().next().and_then(|x| <$ty>::from_vulkan(x))
    };
}

pub(crate) use property_from_vulkan;

macro_rules! properties {
    {
        $($member:ident => {
            doc: $doc:expr,
            ty: $ty:ty,
            ffi_name: $ffi_field:ident,
            ffi_members: [$($ffi_struct:ident $(.$ffi_struct_field:ident)*),+],
            required: $required:tt,
        },)*
    } => {
        /// Represents all the properties of a physical device.
        ///
        /// Depending on the highest version of Vulkan supported by the physical device, and the
        /// available extensions, not every property may be available. For that reason, properties
        /// are wrapped in an `Option`.
        #[derive(Clone, Debug, Default)]
        #[allow(missing_docs)]
        pub struct Properties {
            $(
                #[doc = $doc]
                pub $member: $crate::device::properties::property_type!($ty, $required),
            )*
        }

        impl From<&PropertiesFfi> for Properties {
            fn from(properties_ffi: &PropertiesFfi) -> Self {
                use crate::device::properties::FromVulkan;

                Properties {
                    $(
                        $member: crate::device::properties::property_from_vulkan!($ty, [ $($ffi_struct$(.$ffi_struct_field)*),+ ], $ffi_field, $required, properties_ffi),
                    )*
                }
            }
        }
    };
}

pub use crate::autogen::Properties;
pub(crate) use properties;

macro_rules! properties_ffi {
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
        pub(crate) struct PropertiesFfi {
            properties_vulkan10: ash::vk::PhysicalDeviceProperties2KHR,

            $(
                $member: Option<ash::vk::$ty>,
            )+
        }

        impl PropertiesFfi {
            pub(crate) fn make_chain(
				&mut self,
				$api_version: crate::Version,
				$device_extensions: &crate::device::DeviceExtensions,
				$instance_extensions: &crate::instance::InstanceExtensions,
			) {
                self.properties_vulkan10 = Default::default();
                let head = &mut self.properties_vulkan10;

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

            pub(crate) fn head_as_ref(&self) -> &ash::vk::PhysicalDeviceProperties2KHR {
                &self.properties_vulkan10
            }

            pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceProperties2KHR {
                &mut self.properties_vulkan10
            }
        }
    };
}

pub(crate) use {crate::autogen::PropertiesFfi, properties_ffi};

// A bit of a hack...
pub(crate) trait FromVulkan<F>
where
    Self: Sized,
{
    fn from_vulkan(val: F) -> Option<Self>;
}

impl FromVulkan<u8> for u8 {
    #[inline]
    fn from_vulkan(val: u8) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[u8; N]> for [u8; N] {
    #[inline]
    fn from_vulkan(val: [u8; N]) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<u32> for u32 {
    #[inline]
    fn from_vulkan(val: u32) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[u32; N]> for [u32; N] {
    #[inline]
    fn from_vulkan(val: [u32; N]) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<u64> for u64 {
    #[inline]
    fn from_vulkan(val: u64) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<usize> for usize {
    #[inline]
    fn from_vulkan(val: usize) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<i32> for i32 {
    #[inline]
    fn from_vulkan(val: i32) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<f32> for f32 {
    #[inline]
    fn from_vulkan(val: f32) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[f32; N]> for [f32; N] {
    #[inline]
    fn from_vulkan(val: [f32; N]) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[std::os::raw::c_char; N]> for String {
    #[inline]
    fn from_vulkan(val: [i8; N]) -> Option<Self> {
        Some(unsafe { CStr::from_ptr(val.as_ptr()).to_string_lossy().into_owned() })
    }
}

impl FromVulkan<u32> for Version {
    #[inline]
    fn from_vulkan(val: u32) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<ash::vk::Bool32> for bool {
    #[inline]
    fn from_vulkan(val: ash::vk::Bool32) -> Option<Self> {
        Some(val != 0)
    }
}

impl FromVulkan<ash::vk::ConformanceVersion> for ConformanceVersion {
    #[inline]
    fn from_vulkan(val: ash::vk::ConformanceVersion) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<ash::vk::DriverId> for DriverId {
    #[inline]
    fn from_vulkan(val: ash::vk::DriverId) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<ash::vk::Extent2D> for [u32; 2] {
    #[inline]
    fn from_vulkan(val: ash::vk::Extent2D) -> Option<Self> {
        Some([val.width, val.height])
    }
}

impl FromVulkan<ash::vk::PhysicalDeviceType> for PhysicalDeviceType {
    #[inline]
    fn from_vulkan(val: ash::vk::PhysicalDeviceType) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<ash::vk::PointClippingBehavior> for PointClippingBehavior {
    #[inline]
    fn from_vulkan(val: ash::vk::PointClippingBehavior) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<ash::vk::ResolveModeFlags> for ResolveModes {
    #[inline]
    fn from_vulkan(val: ash::vk::ResolveModeFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<ash::vk::SampleCountFlags> for SampleCounts {
    #[inline]
    fn from_vulkan(val: ash::vk::SampleCountFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<ash::vk::SampleCountFlags> for SampleCount {
    #[inline]
    fn from_vulkan(val: ash::vk::SampleCountFlags) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<ash::vk::ShaderCorePropertiesFlagsAMD> for ShaderCoreProperties {
    #[inline]
    fn from_vulkan(val: ash::vk::ShaderCorePropertiesFlagsAMD) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<ash::vk::ShaderFloatControlsIndependence> for ShaderFloatControlsIndependence {
    #[inline]
    fn from_vulkan(val: ash::vk::ShaderFloatControlsIndependence) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<ash::vk::ShaderStageFlags> for ShaderStages {
    #[inline]
    fn from_vulkan(val: ash::vk::ShaderStageFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<ash::vk::SubgroupFeatureFlags> for SubgroupFeatures {
    #[inline]
    fn from_vulkan(val: ash::vk::SubgroupFeatureFlags) -> Option<Self> {
        Some(val.into())
    }
}
