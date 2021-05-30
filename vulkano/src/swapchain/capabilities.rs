// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::Format;
use crate::image::ImageUsage;
use std::iter::FromIterator;

/// The capabilities of a surface when used by a physical device.
///
/// You have to match these capabilities when you create a swapchain.
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Minimum number of images that must be present in the swapchain.
    pub min_image_count: u32,

    /// Maximum number of images that must be present in the swapchain, or `None` if there is no
    /// maximum value. Note that "no maximum" doesn't mean that you can set a very high value, as
    /// you may still get out of memory errors.
    pub max_image_count: Option<u32>,

    /// The current dimensions of the surface. `None` means that the surface's dimensions will
    /// depend on the dimensions of the swapchain that you are going to create.
    pub current_extent: Option<[u32; 2]>,

    /// Minimum width and height of a swapchain that uses this surface.
    pub min_image_extent: [u32; 2],

    /// Maximum width and height of a swapchain that uses this surface.
    pub max_image_extent: [u32; 2],

    /// Maximum number of image layers if you create an image array. The minimum is 1.
    pub max_image_array_layers: u32,

    /// List of transforms supported for the swapchain.
    pub supported_transforms: SupportedSurfaceTransforms,

    /// Current transform used by the surface.
    pub current_transform: SurfaceTransform,

    /// List of composite alpha modes supports for the swapchain.
    pub supported_composite_alpha: SupportedCompositeAlpha,

    /// List of image usages that are supported for images of the swapchain. Only
    /// the `color_attachment` usage is guaranteed to be supported.
    pub supported_usage_flags: ImageUsage,

    /// List of formats supported for the swapchain.
    pub supported_formats: Vec<(Format, ColorSpace)>, // TODO: https://github.com/KhronosGroup/Vulkan-Docs/issues/207

    /// List of present modes that are supported. `Fifo` is always guaranteed to be supported.
    pub present_modes: SupportedPresentModes,
}

/// The way presenting a swapchain is accomplished.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PresentMode {
    /// Immediately shows the image to the user. May result in visible tearing.
    Immediate = ash::vk::PresentModeKHR::IMMEDIATE.as_raw(),

    /// The action of presenting an image puts it in wait. When the next vertical blanking period
    /// happens, the waiting image is effectively shown to the user. If an image is presented while
    /// another one is waiting, it is replaced.
    Mailbox = ash::vk::PresentModeKHR::MAILBOX.as_raw(),

    /// The action of presenting an image adds it to a queue of images. At each vertical blanking
    /// period, the queue is popped and an image is presented.
    ///
    /// Guaranteed to be always supported.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of 1.
    Fifo = ash::vk::PresentModeKHR::FIFO.as_raw(),

    /// Same as `Fifo`, except that if the queue was empty during the previous vertical blanking
    /// period then it is equivalent to `Immediate`.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of -1.
    Relaxed = ash::vk::PresentModeKHR::FIFO_RELAXED.as_raw(),
    // TODO: These can't be enabled yet because they have to be used with shared present surfaces
    // which vulkano doesnt support yet.
    //SharedDemand = ash::vk::PresentModeKHR::SHARED_DEMAND_REFRESH,
    //SharedContinuous = ash::vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH,
}

impl From<PresentMode> for ash::vk::PresentModeKHR {
    #[inline]
    fn from(val: PresentMode) -> Self {
        Self::from_raw(val as i32)
    }
}

/// List of `PresentMode`s that are supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SupportedPresentModes {
    pub immediate: bool,
    pub mailbox: bool,
    pub fifo: bool,
    pub relaxed: bool,
    pub shared_demand: bool,
    pub shared_continuous: bool,
}

impl FromIterator<ash::vk::PresentModeKHR> for SupportedPresentModes {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = ash::vk::PresentModeKHR>,
    {
        let mut result = SupportedPresentModes::none();
        for e in iter {
            match e {
                ash::vk::PresentModeKHR::IMMEDIATE => result.immediate = true,
                ash::vk::PresentModeKHR::MAILBOX => result.mailbox = true,
                ash::vk::PresentModeKHR::FIFO => result.fifo = true,
                ash::vk::PresentModeKHR::FIFO_RELAXED => result.relaxed = true,
                ash::vk::PresentModeKHR::SHARED_DEMAND_REFRESH => result.shared_demand = true,
                ash::vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH => {
                    result.shared_continuous = true
                }
                _ => {}
            }
        }
        result
    }
}

impl SupportedPresentModes {
    /// Builds a `SupportedPresentModes` with all fields set to false.
    #[inline]
    pub fn none() -> SupportedPresentModes {
        SupportedPresentModes {
            immediate: false,
            mailbox: false,
            fifo: false,
            relaxed: false,
            shared_demand: false,
            shared_continuous: false,
        }
    }

    /// Returns true if the given present mode is in this list of supported modes.
    #[inline]
    pub fn supports(&self, mode: PresentMode) -> bool {
        match mode {
            PresentMode::Immediate => self.immediate,
            PresentMode::Mailbox => self.mailbox,
            PresentMode::Fifo => self.fifo,
            PresentMode::Relaxed => self.relaxed,
        }
    }

    /// Returns an iterator to the list of supported present modes.
    #[inline]
    pub fn iter(&self) -> SupportedPresentModesIter {
        SupportedPresentModesIter(self.clone())
    }
}

/// Enumeration of the `PresentMode`s that are supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SupportedPresentModesIter(SupportedPresentModes);

impl Iterator for SupportedPresentModesIter {
    type Item = PresentMode;

    #[inline]
    fn next(&mut self) -> Option<PresentMode> {
        if self.0.immediate {
            self.0.immediate = false;
            return Some(PresentMode::Immediate);
        }
        if self.0.mailbox {
            self.0.mailbox = false;
            return Some(PresentMode::Mailbox);
        }
        if self.0.fifo {
            self.0.fifo = false;
            return Some(PresentMode::Fifo);
        }
        if self.0.relaxed {
            self.0.relaxed = false;
            return Some(PresentMode::Relaxed);
        }
        None
    }
}

/// A transformation to apply to the image before showing it on the screen.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SurfaceTransform {
    /// Don't transform the image.
    Identity = ash::vk::SurfaceTransformFlagsKHR::IDENTITY.as_raw(),
    /// Rotate 90 degrees.
    Rotate90 = ash::vk::SurfaceTransformFlagsKHR::ROTATE_90.as_raw(),
    /// Rotate 180 degrees.
    Rotate180 = ash::vk::SurfaceTransformFlagsKHR::ROTATE_180.as_raw(),
    /// Rotate 270 degrees.
    Rotate270 = ash::vk::SurfaceTransformFlagsKHR::ROTATE_270.as_raw(),
    /// Mirror the image horizontally.
    HorizontalMirror = ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR.as_raw(),
    /// Mirror the image horizontally and rotate 90 degrees.
    HorizontalMirrorRotate90 =
        ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_90.as_raw(),
    /// Mirror the image horizontally and rotate 180 degrees.
    HorizontalMirrorRotate180 =
        ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_180.as_raw(),
    /// Mirror the image horizontally and rotate 270 degrees.
    HorizontalMirrorRotate270 =
        ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_270.as_raw(),
    /// Let the operating system or driver implementation choose.
    Inherit = ash::vk::SurfaceTransformFlagsKHR::INHERIT.as_raw(),
}

impl From<SurfaceTransform> for ash::vk::SurfaceTransformFlagsKHR {
    #[inline]
    fn from(val: SurfaceTransform) -> Self {
        Self::from_raw(val as u32)
    }
}

/// How the alpha values of the pixels of the window are treated.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CompositeAlpha {
    /// The alpha channel of the image is ignored. All the pixels are considered as if they have a
    /// value of 1.0.
    Opaque = ash::vk::CompositeAlphaFlagsKHR::OPAQUE.as_raw(),

    /// The alpha channel of the image is respected. The color channels are expected to have
    /// already been multiplied by the alpha value.
    PreMultiplied = ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED.as_raw(),

    /// The alpha channel of the image is respected. The color channels will be multiplied by the
    /// alpha value by the compositor before being added to what is behind.
    PostMultiplied = ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED.as_raw(),

    /// Let the operating system or driver implementation choose.
    Inherit = ash::vk::CompositeAlphaFlagsKHR::INHERIT.as_raw(),
}

impl From<CompositeAlpha> for ash::vk::CompositeAlphaFlagsKHR {
    #[inline]
    fn from(val: CompositeAlpha) -> Self {
        Self::from_raw(val as u32)
    }
}

/// List of supported composite alpha modes.
///
/// See the docs of `CompositeAlpha`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct SupportedCompositeAlpha {
    pub opaque: bool,
    pub pre_multiplied: bool,
    pub post_multiplied: bool,
    pub inherit: bool,
}

impl From<ash::vk::CompositeAlphaFlagsKHR> for SupportedCompositeAlpha {
    #[inline]
    fn from(val: ash::vk::CompositeAlphaFlagsKHR) -> SupportedCompositeAlpha {
        let mut result = SupportedCompositeAlpha::none();
        if !(val & ash::vk::CompositeAlphaFlagsKHR::OPAQUE).is_empty() {
            result.opaque = true;
        }
        if !(val & ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED).is_empty() {
            result.pre_multiplied = true;
        }
        if !(val & ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED).is_empty() {
            result.post_multiplied = true;
        }
        if !(val & ash::vk::CompositeAlphaFlagsKHR::INHERIT).is_empty() {
            result.inherit = true;
        }
        result
    }
}

impl SupportedCompositeAlpha {
    /// Builds a `SupportedCompositeAlpha` with all fields set to false.
    #[inline]
    pub fn none() -> SupportedCompositeAlpha {
        SupportedCompositeAlpha {
            opaque: false,
            pre_multiplied: false,
            post_multiplied: false,
            inherit: false,
        }
    }

    /// Returns true if the given `CompositeAlpha` is in this list.
    #[inline]
    pub fn supports(&self, value: CompositeAlpha) -> bool {
        match value {
            CompositeAlpha::Opaque => self.opaque,
            CompositeAlpha::PreMultiplied => self.pre_multiplied,
            CompositeAlpha::PostMultiplied => self.post_multiplied,
            CompositeAlpha::Inherit => self.inherit,
        }
    }

    /// Returns an iterator to the list of supported composite alpha.
    #[inline]
    pub fn iter(&self) -> SupportedCompositeAlphaIter {
        SupportedCompositeAlphaIter(self.clone())
    }
}

/// Enumeration of the `CompositeAlpha` that are supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SupportedCompositeAlphaIter(SupportedCompositeAlpha);

impl Iterator for SupportedCompositeAlphaIter {
    type Item = CompositeAlpha;

    #[inline]
    fn next(&mut self) -> Option<CompositeAlpha> {
        if self.0.opaque {
            self.0.opaque = false;
            return Some(CompositeAlpha::Opaque);
        }
        if self.0.pre_multiplied {
            self.0.pre_multiplied = false;
            return Some(CompositeAlpha::PreMultiplied);
        }
        if self.0.post_multiplied {
            self.0.post_multiplied = false;
            return Some(CompositeAlpha::PostMultiplied);
        }
        if self.0.inherit {
            self.0.inherit = false;
            return Some(CompositeAlpha::Inherit);
        }
        None
    }
}

/// List of supported composite alpha modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SupportedSurfaceTransforms {
    pub identity: bool,
    pub rotate90: bool,
    pub rotate180: bool,
    pub rotate270: bool,
    pub horizontal_mirror: bool,
    pub horizontal_mirror_rotate90: bool,
    pub horizontal_mirror_rotate180: bool,
    pub horizontal_mirror_rotate270: bool,
    pub inherit: bool,
}

impl From<ash::vk::SurfaceTransformFlagsKHR> for SupportedSurfaceTransforms {
    fn from(val: ash::vk::SurfaceTransformFlagsKHR) -> Self {
        macro_rules! v {
            ($val:expr, $out:ident, $e:expr, $f:ident) => {
                if !($val & $e).is_empty() {
                    $out.$f = true;
                }
            };
        }

        let mut result = SupportedSurfaceTransforms::none();
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::IDENTITY,
            identity
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::ROTATE_90,
            rotate90
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::ROTATE_180,
            rotate180
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::ROTATE_270,
            rotate270
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR,
            horizontal_mirror
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_90,
            horizontal_mirror_rotate90
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_180,
            horizontal_mirror_rotate180
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_270,
            horizontal_mirror_rotate270
        );
        v!(
            val,
            result,
            ash::vk::SurfaceTransformFlagsKHR::INHERIT,
            inherit
        );
        result
    }
}

impl SupportedSurfaceTransforms {
    /// Builds a `SupportedSurfaceTransforms` with all fields set to false.
    #[inline]
    pub fn none() -> SupportedSurfaceTransforms {
        SupportedSurfaceTransforms {
            identity: false,
            rotate90: false,
            rotate180: false,
            rotate270: false,
            horizontal_mirror: false,
            horizontal_mirror_rotate90: false,
            horizontal_mirror_rotate180: false,
            horizontal_mirror_rotate270: false,
            inherit: false,
        }
    }

    /// Returns true if the given `SurfaceTransform` is in this list.
    #[inline]
    pub fn supports(&self, value: SurfaceTransform) -> bool {
        match value {
            SurfaceTransform::Identity => self.identity,
            SurfaceTransform::Rotate90 => self.rotate90,
            SurfaceTransform::Rotate180 => self.rotate180,
            SurfaceTransform::Rotate270 => self.rotate270,
            SurfaceTransform::HorizontalMirror => self.horizontal_mirror,
            SurfaceTransform::HorizontalMirrorRotate90 => self.horizontal_mirror_rotate90,
            SurfaceTransform::HorizontalMirrorRotate180 => self.horizontal_mirror_rotate180,
            SurfaceTransform::HorizontalMirrorRotate270 => self.horizontal_mirror_rotate270,
            SurfaceTransform::Inherit => self.inherit,
        }
    }

    /// Returns an iterator to the list of supported composite alpha.
    #[inline]
    pub fn iter(&self) -> SupportedSurfaceTransformsIter {
        SupportedSurfaceTransformsIter(self.clone())
    }
}

/// Enumeration of the `SurfaceTransform` that are supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SupportedSurfaceTransformsIter(SupportedSurfaceTransforms);

impl Iterator for SupportedSurfaceTransformsIter {
    type Item = SurfaceTransform;

    #[inline]
    fn next(&mut self) -> Option<SurfaceTransform> {
        if self.0.identity {
            self.0.identity = false;
            return Some(SurfaceTransform::Identity);
        }
        if self.0.rotate90 {
            self.0.rotate90 = false;
            return Some(SurfaceTransform::Rotate90);
        }
        if self.0.rotate180 {
            self.0.rotate180 = false;
            return Some(SurfaceTransform::Rotate180);
        }
        if self.0.rotate270 {
            self.0.rotate270 = false;
            return Some(SurfaceTransform::Rotate270);
        }
        if self.0.horizontal_mirror {
            self.0.horizontal_mirror = false;
            return Some(SurfaceTransform::HorizontalMirror);
        }
        if self.0.horizontal_mirror_rotate90 {
            self.0.horizontal_mirror_rotate90 = false;
            return Some(SurfaceTransform::HorizontalMirrorRotate90);
        }
        if self.0.horizontal_mirror_rotate180 {
            self.0.horizontal_mirror_rotate180 = false;
            return Some(SurfaceTransform::HorizontalMirrorRotate180);
        }
        if self.0.horizontal_mirror_rotate270 {
            self.0.horizontal_mirror_rotate270 = false;
            return Some(SurfaceTransform::HorizontalMirrorRotate270);
        }
        if self.0.inherit {
            self.0.inherit = false;
            return Some(SurfaceTransform::Inherit);
        }
        None
    }
}

impl Default for SurfaceTransform {
    #[inline]
    fn default() -> SurfaceTransform {
        SurfaceTransform::Identity
    }
}

/// How the presentation engine should interpret the data.
///
/// # A quick lesson about color spaces
///
/// ## What is a color space?
///
/// Each pixel of a monitor is made of three components: one red, one green, and one blue. In the
/// past, computers would simply send to the monitor the intensity of each of the three components.
///
/// This proved to be problematic, because depending on the brand of the monitor the colors would
/// not exactly be the same. For example on some monitors, a value of `[1.0, 0.0, 0.0]` would be a
/// bit more orange than on others.
///
/// In order to standardize this, there exist what are called *color spaces*: sRGB, AdobeRGB,
/// DCI-P3, scRGB, etc. When you manipulate RGB values in a specific color space, these values have
/// a precise absolute meaning in terms of color, that is the same across all systems and monitors.
///
/// > **Note**: Color spaces are orthogonal to concept of RGB. *RGB* only indicates what is the
/// > representation of the data, but not how it is interpreted. You can think of this a bit like
/// > text encoding. An *RGB* value is a like a byte, in other words it is the medium by which
/// > values are communicated, and a *color space* is like a text encoding (eg. UTF-8), in other
/// > words it is the way the value should be interpreted.
///
/// The most commonly used color space today is sRGB. Most monitors today use this color space,
/// and most images files are encoded in this color space.
///
/// ## Pixel formats and linear vs non-linear
///
/// In Vulkan all images have a specific format in which the data is stored. The data of an image
/// consists of pixels in RGB but contains no information about the color space (or lack thereof)
/// of these pixels. You are free to store them in whatever color space you want.
///
/// But one big practical problem with color spaces is that they are sometimes not linear, and in
/// particular the popular sRGB color space is not linear. In a non-linear color space, a value of
/// `[0.6, 0.6, 0.6]` for example is **not** twice as bright as a value of `[0.3, 0.3, 0.3]`. This
/// is problematic, because operations such as taking the average of two colors or calculating the
/// lighting of a texture with a dot product are mathematically incorrect and will produce
/// incorrect colors.
///
/// > **Note**: If the texture format has an alpha component, it is not affected by the color space
/// > and always behaves linearly.
///
/// In order to solve this Vulkan also provides image formats with the `Srgb` suffix, which are
/// expected to contain RGB data in the sRGB color space. When you sample an image with such a
/// format from a shader, the implementation will automatically turn the pixel values into a linear
/// color space that is suitable for linear operations (such as additions or multiplications).
/// When you write to a framebuffer attachment with such a format, the implementation will
/// automatically perform the opposite conversion. These conversions are most of the time performed
/// by the hardware and incur no additional cost.
///
/// ## Color space of the swapchain
///
/// The color space that you specify when you create a swapchain is how the implementation will
/// interpret the raw data inside of the image.
///
/// > **Note**: The implementation can choose to send the data in the swapchain image directly to
/// > the monitor, but it can also choose to write it in an intermediary buffer that is then read
/// > by the operating system or windowing system. Therefore the color space that the
/// > implementation supports is not necessarily the same as the one supported by the monitor.
///
/// It is *your* job to ensure that the data in the swapchain image is in the color space
/// that is specified here, otherwise colors will be incorrect.
/// The implementation will never perform any additional automatic conversion after the colors have
/// been written to the swapchain image.
///
/// # How do I handle this correctly?
///
/// The easiest way to handle color spaces in a cross-platform program is:
///
/// - Always request the `SrgbNonLinear` color space when creating the swapchain.
/// - Make sure that all your image files use the sRGB color space, and load them in images whose
///   format has the `Srgb` suffix. Only use non-sRGB image formats for intermediary computations
///   or to store non-color data.
/// - Swapchain images should have a format with the `Srgb` suffix.
///
/// > **Note**: It is unclear whether the `SrgbNonLinear` color space is always supported by the
/// > the implementation or not. See https://github.com/KhronosGroup/Vulkan-Docs/issues/442.
///
/// > **Note**: Lots of developers are confused by color spaces. You can sometimes find articles
/// > talking about gamma correction and suggestion to put your colors to the power 2.2 for
/// > example. These are all hacks and you should use the sRGB pixel formats instead.
///
/// If you follow these three rules, then everything should render the same way on all platforms.
///
/// Additionally you can try detect whether the implementation supports any additional color space
/// and perform a manual conversion to that color space from inside your shader.
///
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ColorSpace {
    SrgbNonLinear = ash::vk::ColorSpaceKHR::SRGB_NONLINEAR.as_raw(),
    DisplayP3NonLinear = ash::vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT.as_raw(),
    ExtendedSrgbLinear = ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT.as_raw(),
    DciP3Linear = ash::vk::ColorSpaceKHR::DCI_P3_LINEAR_EXT.as_raw(),
    DciP3NonLinear = ash::vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT.as_raw(),
    Bt709Linear = ash::vk::ColorSpaceKHR::BT709_LINEAR_EXT.as_raw(),
    Bt709NonLinear = ash::vk::ColorSpaceKHR::BT709_NONLINEAR_EXT.as_raw(),
    Bt2020Linear = ash::vk::ColorSpaceKHR::BT2020_LINEAR_EXT.as_raw(),
    Hdr10St2084 = ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT.as_raw(),
    DolbyVision = ash::vk::ColorSpaceKHR::DOLBYVISION_EXT.as_raw(),
    Hdr10Hlg = ash::vk::ColorSpaceKHR::HDR10_HLG_EXT.as_raw(),
    AdobeRgbLinear = ash::vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT.as_raw(),
    AdobeRgbNonLinear = ash::vk::ColorSpaceKHR::ADOBERGB_NONLINEAR_EXT.as_raw(),
    PassThrough = ash::vk::ColorSpaceKHR::PASS_THROUGH_EXT.as_raw(),
}

impl From<ColorSpace> for ash::vk::ColorSpaceKHR {
    #[inline]
    fn from(val: ColorSpace) -> Self {
        Self::from_raw(val as i32)
    }
}

impl From<ash::vk::ColorSpaceKHR> for ColorSpace {
    #[inline]
    fn from(val: ash::vk::ColorSpaceKHR) -> Self {
        match val {
            ash::vk::ColorSpaceKHR::SRGB_NONLINEAR => ColorSpace::SrgbNonLinear,
            ash::vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => ColorSpace::DisplayP3NonLinear,
            ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => ColorSpace::ExtendedSrgbLinear,
            ash::vk::ColorSpaceKHR::DCI_P3_LINEAR_EXT => ColorSpace::DciP3Linear,
            ash::vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT => ColorSpace::DciP3NonLinear,
            ash::vk::ColorSpaceKHR::BT709_LINEAR_EXT => ColorSpace::Bt709Linear,
            ash::vk::ColorSpaceKHR::BT709_NONLINEAR_EXT => ColorSpace::Bt709NonLinear,
            ash::vk::ColorSpaceKHR::BT2020_LINEAR_EXT => ColorSpace::Bt2020Linear,
            ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT => ColorSpace::Hdr10St2084,
            ash::vk::ColorSpaceKHR::DOLBYVISION_EXT => ColorSpace::DolbyVision,
            ash::vk::ColorSpaceKHR::HDR10_HLG_EXT => ColorSpace::Hdr10Hlg,
            ash::vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT => ColorSpace::AdobeRgbLinear,
            ash::vk::ColorSpaceKHR::ADOBERGB_NONLINEAR_EXT => ColorSpace::AdobeRgbNonLinear,
            ash::vk::ColorSpaceKHR::PASS_THROUGH_EXT => ColorSpace::PassThrough,
            _ => panic!("Wrong value for color space enum"),
        }
    }
}
