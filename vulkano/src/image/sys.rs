use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use device::Device;
use format::Format;
use format::FormatTy;
use image::MipmapsCount;
use memory::DeviceMemory;
use memory::MemoryRequirements;
use sync::Sharing;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// A storage for pixels or arbitrary data.
pub struct UnsafeImage {
    image: vk::Image,
    device: Arc<Device>,
    usage: vk::ImageUsageFlagBits,
    format: Format,

    dimensions: [f32; 3],
    array_layers: u32,
    samples: u32,
    mipmaps: u32,

    // `vkDestroyImage` is called only if `needs_destruction` is true.
    needs_destruction: bool,
}

impl UnsafeImage {
    /// Creates a new image and allocates memory for it.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the dimensions is 0.
    /// - Panicks if the number of mipmaps is 0.
    /// - Panicks if the number of samples is 0.
    ///
    pub unsafe fn new<'a, Mi, I>(device: &Arc<Device>, usage: &Usage, format: Format,
                                 dimensions: Dimensions, num_samples: u32, mipmaps: Mi,
                                 sharing: Sharing<I>, linear_tiling: bool,
                                 preinitialized_layout: bool)
                                 -> Result<(UnsafeImage, MemoryRequirements), OomError>
        where Mi: Into<MipmapsCount>, I: Iterator<Item = u32>
    {
        // Preprocessing parameters.
        let sharing = sharing.into();
        let usage = usage.to_usage_bits();
        assert!(num_samples >= 1);

        // Compute the maximum number of mipmaps.
        // TODO: only compte if necessary?
        let max_mipmaps = {
            let smallest_dim: u32 = match dimensions {
                Dimensions::Dim1d { width } | Dimensions::Dim1dArray { width, .. } => width,
                Dimensions::Dim2d { width, height } | Dimensions::Dim2dArray { width, height, .. } => {
                    if width < height { width } else { height }
                },
                Dimensions::Dim3d { width, height, depth } => {
                    if width < height {
                        if depth < width { depth } else { width }
                    } else {
                        if depth < height { depth } else { height }
                    }
                },
            };

            assert!(smallest_dim >= 1);
            32 - smallest_dim.leading_zeros()
        };

        // Compute the number of mipmaps.
        let mipmaps = match mipmaps.into() {
            MipmapsCount::Specific(num) => {
                assert!(num >= 1);
                assert!(num <= max_mipmaps);
                num
            },
            MipmapsCount::Max => max_mipmaps,
            MipmapsCount::One => 1,
        };

        let vk = device.pointers();

        // TODO: check for limits
        let (ty, extent, array_layers, dims) = match dimensions {
            Dimensions::Dim1d { width } => {
                let extent = vk::Extent3D { width: width, height: 1, depth: 1 };
                let dims = [width as f32, 1.0, 1.0];
                (vk::IMAGE_TYPE_1D, extent, 1, dims)
            },
            Dimensions::Dim1dArray { width, array_layers } => {
                let extent = vk::Extent3D { width: width, height: 1, depth: 1 };
                let dims = [width as f32, 1.0, 1.0];
                (vk::IMAGE_TYPE_1D, extent, array_layers, dims)
            },
            Dimensions::Dim2d { width, height } => {
                let extent = vk::Extent3D { width: width, height: height, depth: 1 };
                let dims = [width as f32, height as f32, 1.0];
                (vk::IMAGE_TYPE_2D, extent, 1, dims)
            },
            Dimensions::Dim2dArray { width, height, array_layers } => {
                let extent = vk::Extent3D { width: width, height: height, depth: 1 };
                let dims = [width as f32, height as f32, 1.0];
                (vk::IMAGE_TYPE_2D, extent, array_layers, dims)
            },
            Dimensions::Dim3d { width, height, depth } => {
                let extent = vk::Extent3D { width: width, height: height, depth: depth };
                let dims = [width as f32, height as f32, depth as f32];
                (vk::IMAGE_TYPE_3D, extent, 1, dims)
            },
        };

        let image = {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                Sharing::Concurrent(ref ids) => unimplemented!() /*(vk::SHARING_MODE_CONCURRENT, ids.len() as u32,
                                                     ids.as_ptr())*/,
            };

            let infos = vk::ImageCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,                               // TODO:
                imageType: ty,
                format: format as u32,
                extent: extent,
                mipLevels: mipmaps,
                arrayLayers: array_layers,
                samples: num_samples,
                tiling: if linear_tiling {
                    vk::IMAGE_TILING_LINEAR     // FIXME: check whether it's supported
                } else {
                    vk::IMAGE_TILING_OPTIMAL
                },
                usage: usage,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_count,
                pQueueFamilyIndices: sh_indices,
                initialLayout: if preinitialized_layout {
                    vk::IMAGE_LAYOUT_PREINITIALIZED
                } else {
                    vk::IMAGE_LAYOUT_UNDEFINED
                },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateImage(device.internal_object(), &infos,
                                             ptr::null(), &mut output)));
            output
        };

        let mem_reqs: vk::MemoryRequirements = {
            let mut output = mem::uninitialized();
            vk.GetImageMemoryRequirements(device.internal_object(), image, &mut output);
            debug_assert!(output.memoryTypeBits != 0);
            output
        };

        let image = UnsafeImage {
            device: device.clone(),
            image: image,
            usage: usage,
            format: format,
            dimensions: dims,
            array_layers: array_layers,
            samples: num_samples,
            mipmaps: mipmaps,
            needs_destruction: true,
        };

        Ok((image, mem_reqs.into()))
    }

    /// Creates an image from a raw handle. The image won't be destroyed.
    ///
    /// This function is for example used at the swapchain's initialization.
    pub unsafe fn from_raw(device: &Arc<Device>, handle: u64, usage: u32, format: Format,
                           dimensions: Dimensions, array_layers: u32, samples: u32, mipmaps: u32)
                           -> UnsafeImage
    {
        // TODO: DRY
        let dims = match dimensions {
            Dimensions::Dim1d { width } => {
                [width as f32, 1.0, 1.0]
            },
            Dimensions::Dim1dArray { width, array_layers } => {
                [width as f32, 1.0, 1.0]
            },
            Dimensions::Dim2d { width, height } => {
                [width as f32, height as f32, 1.0]
            },
            Dimensions::Dim2dArray { width, height, array_layers } => {
                [width as f32, height as f32, 1.0]
            },
            Dimensions::Dim3d { width, height, depth } => {
                [width as f32, height as f32, depth as f32]
            },
        };

        UnsafeImage {
            device: device.clone(),
            image: handle,
            usage: usage,
            format: format,
            dimensions: dims,
            array_layers: array_layers,
            samples: samples,
            mipmaps: mipmaps,
            needs_destruction: false,       // TODO: pass as parameter
        }
    }

    pub unsafe fn bind_memory(&self, memory: &DeviceMemory, range: Range<usize>)
                              -> Result<(), OomError>
    {
        let vk = self.device.pointers();
        try!(check_errors(vk.BindImageMemory(self.device.internal_object(), self.image,
                                             memory.internal_object(),
                                             range.start as vk::DeviceSize)));
        Ok(())
    }
}

unsafe impl VulkanObject for UnsafeImage {
    type Object = vk::Image;

    #[inline]
    fn internal_object(&self) -> vk::Image {
        self.image
    }
}

impl Drop for UnsafeImage {
    #[inline]
    fn drop(&mut self) {
        if !self.needs_destruction {
            return;
        }

        unsafe {
            let vk = self.device.pointers();
            vk.DestroyImage(self.device.internal_object(), self.image, ptr::null());
        }
    }
}

pub struct UnsafeImageView {
    view: vk::ImageView,
    device: Arc<Device>,
    identity_swizzle: bool,
}

impl UnsafeImageView {
    /// Creates a new view from an image.
    ///
    /// Note that you must create the view with identity swizzling if you want to use this view
    /// as a framebuffer attachment.
    pub unsafe fn new(image: &UnsafeImage) -> Result<UnsafeImageView, OomError> {
        let vk = image.device.pointers();

        let aspect_mask = match image.format.ty() {
            FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                vk::IMAGE_ASPECT_COLOR_BIT
            },
            FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
            FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
            FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
        };

        let view = {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                image: image.internal_object(),
                viewType: vk::IMAGE_VIEW_TYPE_2D,     // FIXME:
                format: image.format as u32,
                components: vk::ComponentMapping { r: 0, g: 0, b: 0, a: 0 },     // FIXME:
                subresourceRange: vk::ImageSubresourceRange {
                    aspectMask: aspect_mask,
                    baseMipLevel: 0,            // TODO:
                    levelCount: image.mipmaps,          // TODO:
                    baseArrayLayer: 0,          // TODO:
                    layerCount: 1,          // TODO:
                },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateImageView(image.device.internal_object(), &infos,
                                                 ptr::null(), &mut output)));
            output
        };

        Ok(UnsafeImageView {
            view: view,
            device: image.device.clone(),
            identity_swizzle: true,     // FIXME:
        })
    }
}

impl Drop for UnsafeImageView {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyImageView(self.device.internal_object(), self.view, ptr::null());
        }
    }
}

pub enum Dimensions {
    Dim1d { width: u32 },
    Dim1dArray { width: u32, array_layers: u32 },
    Dim2d { width: u32, height: u32 },
    Dim2dArray { width: u32, height: u32, array_layers: u32 },
    Dim3d { width: u32, height: u32, depth: u32 }
}

/// Describes how an image is going to be used. This is **not** an optimization.
///
/// If you try to use an image in a way that you didn't declare, a panic will happen.
// TODO: enforce the fact that `transient_attachment` can't be set at the same time as other bits
#[derive(Debug, Copy, Clone)]
pub struct Usage {
    pub transfer_source: bool,
    pub transfer_dest: bool,
    pub sampled: bool,
    pub storage: bool,
    pub color_attachment: bool,
    pub depth_stencil_attachment: bool,
    pub transient_attachment: bool,
    pub input_attachment: bool,
}

impl Usage {
    /// Builds a `Usage` with all values set to true. Can be used for quick prototyping.
    #[inline]
    pub fn all() -> Usage {
        Usage {
            transfer_source: true,
            transfer_dest: true,
            sampled: true,
            storage: true,
            color_attachment: true,
            depth_stencil_attachment: true,
            transient_attachment: true,
            input_attachment: true,
        }
    }

    /// Builds a `Usage` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::image::Usage as ImageUsage;
    ///
    /// let _usage = ImageUsage {
    ///     transfer_dest: true,
    ///     sampled: true,
    ///     .. ImageUsage::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> Usage {
        Usage {
            transfer_source: false,
            transfer_dest: false,
            sampled: false,
            storage: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn to_usage_bits(&self) -> vk::ImageUsageFlagBits {
        let mut result = 0;
        if self.transfer_source { result |= vk::IMAGE_USAGE_TRANSFER_SRC_BIT; }
        if self.transfer_dest { result |= vk::IMAGE_USAGE_TRANSFER_DST_BIT; }
        if self.sampled { result |= vk::IMAGE_USAGE_SAMPLED_BIT; }
        if self.storage { result |= vk::IMAGE_USAGE_STORAGE_BIT; }
        if self.color_attachment { result |= vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
        if self.depth_stencil_attachment { result |= vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; }
        if self.transient_attachment { result |= vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT; }
        if self.input_attachment { result |= vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT; }
        result
    }

    #[inline]
    #[doc(hidden)]
    pub fn from_bits(val: u32) -> Usage {
        Usage {
            transfer_source: (val & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0,
            transfer_dest: (val & vk::IMAGE_USAGE_TRANSFER_DST_BIT) != 0,
            sampled: (val & vk::IMAGE_USAGE_SAMPLED_BIT) != 0,
            storage: (val & vk::IMAGE_USAGE_STORAGE_BIT) != 0,
            color_attachment: (val & vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
            depth_stencil_attachment: (val & vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0,
            transient_attachment: (val & vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0,
            input_attachment: (val & vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT) != 0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Layout {
    Undefined = vk::IMAGE_LAYOUT_UNDEFINED,
    General = vk::IMAGE_LAYOUT_GENERAL,
    ColorAttachmentOptimal = vk::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    DepthStencilAttachmentOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    DepthStencilReadOnlyOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    ShaderReadOnlyOptimal = vk::IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    TransferSrcOptimal = vk::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    TransferDstOptimal = vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    Preinitialized = vk::IMAGE_LAYOUT_PREINITIALIZED,
    PresentSrc = vk::IMAGE_LAYOUT_PRESENT_SRC_KHR,
}
