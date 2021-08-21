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

pub use crate::autogen::Properties;
pub(crate) use crate::autogen::PropertiesFfi;

// A bit of a hack...
// TODO: integrate into autogen?
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

impl FromVulkan<i64> for i64 {
    #[inline]
    fn from_vulkan(val: i64) -> Option<Self> {
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
    fn from_vulkan(val: [std::os::raw::c_char; N]) -> Option<Self> {
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
