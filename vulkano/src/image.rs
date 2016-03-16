//! Images storage (1D, 2D, 3D, arrays, etc.).
//! 
//! # Strong typing
//! 
//! Images in vulkano are strong-typed. Their signature is `Image<Ty, F, M>`.
//! 
//! The `Ty` parameter describes the type of image: 1D, 2D, 3D, 1D array, 2D array. All these come
//! in two variants: with or without multisampling. The actual type of `Ty` must be one of the
//! marker structs of this module that start with the `Ty` prefix.
//! 
//! The `F` parameter describes the format of each pixel of the image. It must be one of the marker
//! structs of the `formats` module.
//! 
//! The `M` parameter describes where the image's memory was allocated from. It is similar to
//! buffers.
//!
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use command_buffer::CommandBufferPool;
use device::Device;
use device::Queue;
use format::FormatDesc;
use format::FormatTy;
use memory::ChunkProperties;
use memory::ChunkRange;
use memory::MemorySource;
use memory::MemorySourceChunk;
use sync::Fence;
use sync::Resource;
use sync::Semaphore;
use sync::SharingMode;

use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub unsafe trait AbstractImage: Resource + ::VulkanObjectU64 {
    /// All images in vulkano must have a *default layout*. Whenever this image is used in a
    /// command buffer, it is switched from this default layout to something else (if necessary),
    /// then back again to the default.
    fn default_layout(&self) -> Layout;
    
    /// Instructs the resource that it is going to be used by the GPU soon in the future. The
    /// function should block if the memory is currently being accessed by the CPU.
    ///
    /// `write` indicates whether the GPU will write to the memory. If `false`, then it will only
    /// be written.
    ///
    /// `queue` is the queue where the command buffer that accesses the memory will be submitted.
    /// If the `gpu_access` function submits something to that queue, it will thus be submitted
    /// beforehand. This behavior can be used for example to submit sparse binding commands.
    ///
    /// `fence` is a fence that will be signaled when this GPU access will stop. It should be
    /// waited upon whenever the user wants to read this memory from the CPU. If `requires_fence`
    /// returned false, then this value will be `None`.
    ///
    /// `semaphore` is a semaphore that will be signaled when this GPU access will stop. This value
    /// is intended to be returned later, in a follow-up call to `gpu_access`. If
    /// `requires_semaphore` returned false, then this value will be `None`.
    ///
    /// The function can return a semaphore which will be waited up by the GPU before the
    /// work starts.
    unsafe fn gpu_access(&self, write: bool, queue: &Arc<Queue>, fence: Option<Arc<Fence>>,
                         semaphore: Option<Arc<Semaphore>>) -> Option<Arc<Semaphore>>;
}

pub unsafe trait AbstractImageView: Resource + ::VulkanObjectU64 {
    fn default_layout(&self) -> Layout;

    unsafe fn gpu_access(&self, write: bool, queue: &Arc<Queue>, fence: Option<Arc<Fence>>,
                         semaphore: Option<Arc<Semaphore>>) -> Option<Arc<Semaphore>>;

    /// True if the image can be used as a source for transfers.
    fn usage_transfer_src(&self) -> bool;
    /// True if the image can be used as a destination for transfers.
    fn usage_transfer_dest(&self) -> bool;
    /// True if the image can be sampled from a shader.
    fn usage_sampled(&self) -> bool;
    /// True if the image can be used for image loads/stores in shaders.
    fn usage_storage(&self) -> bool;
    /// True if the image can be used as a color attachment in a framebuffer.
    fn usage_color_attachment(&self) -> bool;
    /// True if the image can be used as a depth and/or stencil attachment in a framebuffer.
    fn usage_depth_stencil_attachment(&self) -> bool;
    /// True if the image can be used as a transient attachment in a framebuffer.
    fn usage_transient_attachment(&self) -> bool;
    /// True if the image can be used as an input attachment in a framebuffer.
    fn usage_input_attachment(&self) -> bool;
}

// IMPORTANT: we transmute some Arc<AbstractTypedImageView> into Arc<AbstractImageView>, so
//            the signature of AbstractTypedImageView should never require another trait
pub unsafe trait AbstractTypedImageView<Ty, F>: AbstractImageView
    where Ty: ImageTypeMarker, F: FormatDesc
{
}

pub unsafe trait AbstractTypedImageViewsTuple {
    type Iter: Iterator<Item = Arc<AbstractImageView>>;
    fn iter(self) -> Self::Iter;
}

macro_rules! impl_abstract_typed_image_views_tuple {
    ({$f_ty:ident, $f_m:ident} $({$o_ty:ident, $o_m:ident})*) => (
        unsafe impl<$f_ty, $f_m $(, $o_ty, $o_m)*> AbstractTypedImageViewsTuple for
            (Arc<AbstractTypedImageView<$f_ty, $f_m>>, $(Arc<AbstractTypedImageView<$o_ty, $o_m>>),*)
        {
            type Iter = VecIntoIter<Arc<AbstractImageView>>;

            #[inline]
            #[allow(non_snake_case)]
            fn iter(self) -> VecIntoIter<Arc<AbstractImageView>> {
                // TODO: Holy shit that transmute is ultra dangerous, but it's the only
                //       solution to turn that Arc<AbstractTypedImageView> into an
                //       Arc<AbstractImageView>.
                //       See https://github.com/rust-lang/rust/issues/5665.
                unsafe {
                    let ($f_ty, $($o_ty,)*) = self;
                    vec![
                        mem::transmute($f_ty) $(, mem::transmute($o_ty))*
                    ].into_iter()
                }
            }
        }

        impl_abstract_typed_image_views_tuple!($({$o_ty, $o_m})*);
    );

    () => ();
}

impl_abstract_typed_image_views_tuple!({Aty, Am} {Bty, Bm} {Cty, Cm} {Dty, Dm} {Ety, Em} {Fty, Fm}
                                       {Gty, Gm} {Hty, Hm} {Ity, Im} {Jty, Jm} {Kty, Km} {Lty, Lm}
                                       {Mty, Mm} {Nty, Nm} {Oty, Om} {Pty, Pm} {Qty, Qm} {Rty, Rm}
                                       {Sty, Sm} {Tty, Tm} {Uty, Um} {Vty, Vm} {Wty, Wm} {Xty, Xm});


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

pub unsafe trait TypeMarker {}

pub unsafe trait ImageTypeMarker: TypeMarker {
    type Dimensions: Copy + Clone;
    type NumSamples: Copy + Clone;

    fn extent(Self::Dimensions) -> [u32; 3];

    fn array_layers(Self::Dimensions) -> u32;

    /// Must return `1` for non-multisampled types.
    fn num_samples(Self::NumSamples) -> u32;

    fn ty() -> ImageType;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageType {
    Type1d = vk::IMAGE_TYPE_1D,
    Type2d = vk::IMAGE_TYPE_2D,
    Type3d = vk::IMAGE_TYPE_3D,
}

pub unsafe trait ImageViewTypeMarker: TypeMarker {}
pub unsafe trait CanCreateView<Dest>: ImageTypeMarker where Dest: ImageViewTypeMarker {}
unsafe impl<T> CanCreateView<T> for T where T: ImageTypeMarker + ImageViewTypeMarker {}
pub unsafe trait MultisampleType: TypeMarker {}

/// Specifies how many mipmaps must be allocated.
///
/// Note that at least one mipmap must be allocated, to store the main level of the image.
#[derive(Debug, Copy, Clone)]
pub enum MipmapsCount {
    /// Allocate the given number of mipmaps.
    Specific(u32),

    /// Allocates the number of mipmaps required to store all the mipmaps of the image where each
    /// mipmap is half the dimensions of the previous level.
    Max,

    /// Allocate one mipmap (ie. just the main level).
    One,
}

impl From<u32> for MipmapsCount {
    #[inline]
    fn from(num: u32) -> MipmapsCount {
        MipmapsCount::Specific(num)
    }
}

/// A storage for pixels or arbitrary data.
pub struct Image<Ty, F, M> where Ty: ImageTypeMarker, M: MemorySource {
    device: Arc<Device>,
    image: vk::Image,
    memory: M::Chunk,
    usage: vk::ImageUsageFlagBits,
    format: F,
    dimensions: Ty::Dimensions,

    // Number of samples per pixel.
    samples: Ty::NumSamples,

    // Number of mipmaps in the image.
    mipmaps: u32,

    sharing: SharingMode,

    // `vkDestroyImage` is called only if `needs_destruction` is true.
    needs_destruction: bool,

    // Contains the layout of the image.
    //
    // Can only ever get two values. First, "undefined" or "preinitialized". Then after the
    // transition, contains the default layout of the image.
    layout: Layout,
}

impl<Ty, F, M> Image<Ty, F, M>
    where M: MemorySource, Ty: ImageTypeMarker, F: FormatDesc
{
    /// Creates a new image and allocates memory for it.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the dimensions is 0.
    /// - Panicks if the number of mipmaps is 0.
    /// - Panicks if the number of samples is 0.
    ///
    pub fn new<Mi, Sh>(device: &Arc<Device>, usage: &Usage, memory: M, sharing: Sh,
                       format: F, dimensions: Ty::Dimensions, num_samples: Ty::NumSamples, mipmaps: Mi)
                       -> Result<ImagePrototype<Ty, F, M>, OomError>
        where Mi: Into<MipmapsCount>, Sh: Into<SharingMode>
    {
        let vk = device.pointers();

        let usage = usage.to_usage_bits();

        assert!(!memory.is_sparse());       // not implemented

        let samples = Ty::num_samples(num_samples);
        assert!(samples >= 1);

        // Compute the maximum number of mipmaps.
        let max_mipmaps = {
            let dims = Ty::extent(dimensions);
            let dim: u32 = match Ty::ty() {
                ImageType::Type1d => dims[0],
                ImageType::Type2d => [dims[0], dims[1]].iter().cloned().min().unwrap(),
                ImageType::Type3d => [dims[0], dims[1], dims[2]].iter().cloned().min().unwrap(),
            };
            assert!(dim >= 1);
            32 - dim.leading_zeros()
        };

        // compute the number of mipmaps
        let mipmaps = match mipmaps.into() {
            MipmapsCount::Specific(num) => {
                assert!(num >= 1);
                assert!(num <= max_mipmaps);
                num
            },
            MipmapsCount::Max => max_mipmaps,
            MipmapsCount::One => 1,
        };

        let sharing = sharing.into();

        let image = unsafe {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                SharingMode::Exclusive(id) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (vk::SHARING_MODE_CONCURRENT, ids.len() as u32,
                                                     ids.as_ptr()),
            };

            let infos = vk::ImageCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,                               // TODO:
                imageType: Ty::ty() as u32,
                format: format.format() as u32,
                extent: {
                    let dims = Ty::extent(dimensions);
                    assert!(dims[0] >= 1); assert!(dims[1] >= 1); assert!(dims[2] >= 1);
                    vk::Extent3D { width: dims[0], height: dims[1], depth: dims[2] }
                },
                mipLevels: mipmaps,
                arrayLayers: Ty::array_layers(dimensions),
                samples: samples,
                tiling: vk::IMAGE_TILING_OPTIMAL,           // TODO:
                usage: usage,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_count,
                pQueueFamilyIndices: sh_indices,
                initialLayout: vk::IMAGE_LAYOUT_UNDEFINED,      // TODO:
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateImage(device.internal_object(), &infos,
                                             ptr::null(), &mut output)));
            output
        };

        let mem_reqs: vk::MemoryRequirements = unsafe {
            let mut output = mem::uninitialized();
            vk.GetImageMemoryRequirements(device.internal_object(), image, &mut output);
            output
        };

        let memory = memory.allocate(device, mem_reqs.size as usize, mem_reqs.alignment as usize,
                                     mem_reqs.memoryTypeBits)
                           .expect("failed to allocate");     // TODO: use try!() instead

        unsafe {
            match memory.properties() {
                ChunkProperties::Regular { memory, offset, .. } => {
                    try!(check_errors(vk.BindImageMemory(device.internal_object(), image,
                                                         memory.internal_object(),
                                                         offset as vk::DeviceSize)));
                },
                _ => unimplemented!()
            }
        }

        Ok(ImagePrototype {
            image: Image {
                device: device.clone(),
                image: image,
                memory: memory,
                usage: usage,
                format: format,
                dimensions: dimensions.clone(),
                samples: num_samples,
                mipmaps: mipmaps,
                sharing: sharing,
                needs_destruction: true,
                layout: Layout::Undefined,        // TODO: can be "predefined"
            },
        })
    }

    /// Creates an image from a raw handle. The image won't be destroyed.
    ///
    /// This function is for example used at the swapchain's initialization.
    pub unsafe fn from_raw_unowned(device: &Arc<Device>, handle: u64, memory: M::Chunk,
                                   sharing: SharingMode, usage: u32, format: F,
                                   dimensions: Ty::Dimensions, samples: Ty::NumSamples,
                                   mipmaps: u32)
                                   -> ImagePrototype<Ty, F, M>
    {
        ImagePrototype{
            image: Image {
                device: device.clone(),
                image: handle,
                memory: memory,
                usage: usage,
                format: format,
                dimensions: dimensions.clone(),
                samples: samples,
                mipmaps: mipmaps,
                sharing: sharing,
                needs_destruction: false,
                layout: Layout::Undefined,
            },
        }
    }
}

impl<Ty, F, M> Image<Ty, F, M>
    where Ty: ImageTypeMarker, F: FormatDesc, M: MemorySource
{
    /// Returns the dimensions of this image.
    #[inline]
    pub fn dimensions(&self) -> Ty::Dimensions {
        self.dimensions
    }

    /// Returns the number of array layers of this image.
    #[inline]
    pub fn array_layers(&self) -> u32 {
        Ty::array_layers(self.dimensions)
    }

    /// Returns the number of mipmap levels of this image.
    #[inline]
    pub fn mipmap_levels(&self) -> u32 {
        self.mipmaps
    }

    /// Returns the number of samples of each pixel of this image.
    ///
    /// Returns `1` if the image is not multisampled.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        Ty::num_samples(self.samples)
    }

    /// Returns the format of the texture.
    #[inline]
    pub fn format(&self) -> &F {
        &self.format
    }
}

unsafe impl<Ty, F, M> VulkanObject for Image<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
    type Object = vk::Image;

    #[inline]
    fn internal_object(&self) -> vk::Image {
        self.image
    }
}

unsafe impl<Ty, F, M> Resource for Image<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
    #[inline]
    fn requires_fence(&self) -> bool {
        self.memory.requires_fence()
    }

    #[inline]
    fn requires_semaphore(&self) -> bool {
        self.memory.requires_semaphore()
    }

    #[inline]
    fn sharing_mode(&self) -> &SharingMode {
        &self.sharing
    }
}

unsafe impl<Ty, F, M> AbstractImage for Image<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
    #[inline]
    fn default_layout(&self) -> Layout {
        self.layout
    }

    #[inline]
    unsafe fn gpu_access(&self, write: bool, queue: &Arc<Queue>, fence: Option<Arc<Fence>>,
                         semaphore: Option<Arc<Semaphore>>) -> Option<Arc<Semaphore>>
    {
        // FIXME: if the image is in its initial transition phase, we need to a semaphore
        self.memory.gpu_access(write, ChunkRange::All, queue, fence, semaphore)
    }
}

impl<Ty, F, M> Drop for Image<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
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

/// Prototype of an image.
///
/// Needs to be transitionned to a proper layout in order to be turned into a regular `Image`.
pub struct ImagePrototype<Ty, F, M> where Ty: ImageTypeMarker, M: MemorySource {
    image: Image<Ty, F, M>,
}

impl<Ty, F, M> ImagePrototype<Ty, F, M>
    where M: MemorySource, Ty: ImageTypeMarker, F: FormatDesc
{
    /// Returns the dimensions of this image.
    #[inline]
    pub fn dimensions(&self) -> Ty::Dimensions {
        self.image.dimensions()
    }

    /// Returns the number of array layers of this image.
    #[inline]
    pub fn array_layers(&self) -> u32 {
        self.image.array_layers()
    }

    /// Returns the number of mipmap levels of this image.
    #[inline]
    pub fn mipmap_levels(&self) -> u32 {
        self.image.mipmap_levels()
    }

    /// Returns the number of samples of each pixel of this image.
    ///
    /// Returns `1` if the image is not multisampled.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        self.image.num_samples()
    }

    /// Transitions the image prototype into a real image by submitting a one-shot command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if `layout` is `Undefined` or `Preinitialized`.
    // FIXME: PresentSrc is only allowed for swapchain images
    pub fn transition(self, layout: Layout, pool: &CommandBufferPool, submit_queue: &Arc<Queue>)
                      -> Result<Arc<Image<Ty, F, M>>, OomError>     // FIXME: error type
    {
        // FIXME: check pool and submit queue correspondance

        assert!(layout != Layout::Undefined);
        assert!(layout != Layout::Preinitialized);

        let mut image = self.image;
        let old_layout = image.layout;
        image.layout = layout;
        let image = Arc::new(image);

        let device = image.device.clone();
        let vk = device.pointers();

        unsafe {
            let cmd = {
                let pool = pool.internal_object_guard();

                let infos = vk::CommandBufferAllocateInfo {
                    sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    pNext: ptr::null(),
                    commandPool: *pool,
                    level: vk::COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount: 1,
                };

                let mut output = mem::uninitialized();
                try!(check_errors(vk.AllocateCommandBuffers(device.internal_object(), &infos,
                                                            &mut output)));
                output
            };

            {
                let infos = vk::CommandBufferBeginInfo {
                    sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    pNext: ptr::null(),
                    flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                    pInheritanceInfo: ptr::null(),
                };

                try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));
            }

            {
                let barrier = vk::ImageMemoryBarrier {
                    sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    pNext: ptr::null(),
                    srcAccessMask: 0,      // FIXME:
                    dstAccessMask: 0x0001ffff,      // FIXME:
                    oldLayout: old_layout as u32,
                    newLayout: layout as u32,
                    srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                    image: image.image,
                    subresourceRange: vk::ImageSubresourceRange {
                        aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,     // FIXME:
                        baseMipLevel: 0,
                        levelCount: 1,  // TODO: vk::REMAINING_MIP_LEVELS,
                        baseArrayLayer: 0,
                        layerCount: 1,  // TODO: vk::REMAINING_ARRAY_LAYERS,
                    }
                };

                vk.CmdPipelineBarrier(cmd, 0, vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0,
                                      ptr::null(), 0, ptr::null(), 1, &barrier);
            }

            try!(check_errors(vk.EndCommandBuffer(cmd)));

            {
                let infos = vk::SubmitInfo {
                    sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
                    pNext: ptr::null(),
                    waitSemaphoreCount: 0,
                    pWaitSemaphores: ptr::null(),
                    pWaitDstStageMask: ptr::null(),
                    commandBufferCount: 1,
                    pCommandBuffers: &cmd,
                    signalSemaphoreCount: 0,            // TODO:
                    pSignalSemaphores: ptr::null(),         // TODO:
                };

                try!(check_errors(vk.QueueSubmit(*submit_queue.internal_object_guard(), 1, &infos, 0)));
            }
        }

        Ok(image)
    }
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

/// A representation of an image.
///
/// Accessing an image from within a shader can only be done through an `ImageView`. An `ImageView`
/// represents a region of an image. You can also do things like creating a 2D view of a 3D
/// image, swizzle the channels, or change the format of the texture (with some restrictions).
pub struct ImageView<Ty, F, M> where Ty: ImageTypeMarker, M: MemorySource {
    image: Arc<Image<Ty, F, M>>,
    view: vk::ImageView,
    /// The view was created with identity swizzling.
    identity_swizzle: bool,
}

impl<Ty, F, M> ImageView<Ty, F, M> where Ty: ImageTypeMarker, M: MemorySource {
    /// Creates a new view from an image.
    ///
    /// Note that you must create the view with identity swizzling if you want to use this view
    /// as a framebuffer attachment.
    pub fn new<'a, I>(image: I) -> Result<Arc<ImageView<Ty, F, M>>, OomError>
        where I: Into<ImageSubresourceRange<'a, Ty, F, M>>, F: FormatDesc + 'a, Ty: 'a, M: 'a
    {
        let image = image.into();
        let vk = image.image.device.pointers();

        let aspect_mask = match image.image.format().format().ty() {
            FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                vk::IMAGE_ASPECT_COLOR_BIT
            },
            FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
            FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
            FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
        };

        let view = unsafe {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                image: image.image.image,
                viewType: vk::IMAGE_VIEW_TYPE_2D,     // FIXME:
                format: image.image.format.format() as u32,
                components: vk::ComponentMapping { r: 0, g: 0, b: 0, a: 0 },     // FIXME:
                subresourceRange: vk::ImageSubresourceRange {
                    aspectMask: aspect_mask,
                    baseMipLevel: image.base_mip_level,
                    levelCount: image.level_count,
                    baseArrayLayer: image.base_array_layer,
                    layerCount: image.layer_count,
                },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateImageView(image.image.device.internal_object(), &infos,
                                                 ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(ImageView {
            image: image.image.clone(),
            view: view,
            identity_swizzle: true,     // FIXME:
        }))
    }

    /// Returns the image from which this view is taken from.
    #[inline]
    pub fn image(&self) -> &Arc<Image<Ty, F, M>> {
        &self.image
    }

    /// Returns true if the swizzling of this image view is identity.
    #[inline]
    pub fn is_identity_swizzled(&self) -> bool {
        self.identity_swizzle
    }
}

impl<Ty, F, M> ImageView<Ty, F, M> where Ty: ImageTypeMarker, M: MemorySource {
    // TODO: hack, remove
    #[doc(hidden)]
    pub fn id(&self) -> u64 { self.view }
}

unsafe impl<Ty, F, M> VulkanObject for ImageView<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
    type Object = vk::ImageView;

    #[inline]
    fn internal_object(&self) -> vk::ImageView {
        self.view
    }
}

unsafe impl<Ty, F, M> Resource for ImageView<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
    #[inline]
    fn requires_fence(&self) -> bool {
        self.image.requires_fence()
    }

    #[inline]
    fn requires_semaphore(&self) -> bool {
        self.image.requires_semaphore()
    }

    #[inline]
    fn sharing_mode(&self) -> &SharingMode {
        self.image.sharing_mode()
    }
}

unsafe impl<Ty, F, M> AbstractImageView for ImageView<Ty, F, M>
    where Ty: ImageTypeMarker, M: MemorySource
{
    #[inline]
    fn default_layout(&self) -> Layout {
        self.image.default_layout()
    }

    #[inline]
    unsafe fn gpu_access(&self, write: bool, queue: &Arc<Queue>, fence: Option<Arc<Fence>>,
                         semaphore: Option<Arc<Semaphore>>) -> Option<Arc<Semaphore>>
    {
        self.image.gpu_access(write, queue, fence, semaphore)
    }

    #[inline]
    fn usage_transfer_src(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0
    }

    #[inline]
    fn usage_transfer_dest(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_TRANSFER_DST_BIT) != 0
    }

    #[inline]
    fn usage_sampled(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_SAMPLED_BIT) != 0
    }

    #[inline]
    fn usage_storage(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_STORAGE_BIT) != 0
    }

    #[inline]
    fn usage_color_attachment(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0
    }

    #[inline]
    fn usage_depth_stencil_attachment(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0
    }

    #[inline]
    fn usage_transient_attachment(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0
    }

    #[inline]
    fn usage_input_attachment(&self) -> bool {
        (self.image.usage & vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT) != 0
    }
}

unsafe impl<Ty, F, M> AbstractTypedImageView<Ty, F> for ImageView<Ty, F, M>
    where Ty: ImageTypeMarker, F: FormatDesc, M: MemorySource
{
}

impl<Ty, F, M> Drop for ImageView<Ty, F, M> where Ty: ImageTypeMarker, M: MemorySource {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.image.device.pointers();
            vk.DestroyImageView(self.image.device.internal_object(), self.view, ptr::null());
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Swizzle {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComponentSwizzle {
    Identity,
    Zero,
    One,
    Red,
    Green,
    Blue,
    Alpha,
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}

pub struct Type1d;

unsafe impl TypeMarker for Type1d {
}

unsafe impl ImageViewTypeMarker for Type1d {
}

unsafe impl ImageTypeMarker for Type1d {
    type Dimensions = u32;
    type NumSamples = ();

    #[inline]
    fn extent(dims: u32) -> [u32; 3] {
        [dims, 1, 1]
    }

    #[inline]
    fn array_layers(_: u32) -> u32 {
        1
    }

    #[inline]
    fn num_samples(_: ()) -> u32 {
        1
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type1d
    }
}

pub struct Type1dMultisample;

unsafe impl TypeMarker for Type1dMultisample {
}

unsafe impl ImageViewTypeMarker for Type1dMultisample {
}

unsafe impl ImageTypeMarker for Type1dMultisample {
    type Dimensions = u32;
    type NumSamples = u32;

    #[inline]
    fn extent(dims: u32) -> [u32; 3] {
        [dims, 1, 1]
    }

    #[inline]
    fn array_layers(_: u32) -> u32 {
        1
    }

    #[inline]
    fn num_samples(num: u32) -> u32 {
        num
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type1d
    }
}

unsafe impl MultisampleType for Type1dMultisample {
}

pub struct Type2d;

unsafe impl TypeMarker for Type2d {
}

unsafe impl ImageViewTypeMarker for Type2d {
}

unsafe impl ImageTypeMarker for Type2d {
    type Dimensions = [u32; 2];
    type NumSamples = ();

    #[inline]
    fn extent(dims: [u32; 2]) -> [u32; 3] {
        [dims[0], dims[1], 1]
    }

    #[inline]
    fn array_layers(_: [u32; 2]) -> u32 {
        1
    }

    #[inline]
    fn num_samples(_: ()) -> u32 {
        1
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type2d
    }
}

pub struct Type2dMultisample;

unsafe impl TypeMarker for Type2dMultisample {
}

unsafe impl ImageViewTypeMarker for Type2dMultisample {
}

unsafe impl ImageTypeMarker for Type2dMultisample {
    type Dimensions = [u32; 2];
    type NumSamples = u32;

    #[inline]
    fn extent(dims: [u32; 2]) -> [u32; 3] {
        [dims[0], dims[1], 1]
    }

    #[inline]
    fn array_layers(_: [u32; 2]) -> u32 {
        1
    }

    #[inline]
    fn num_samples(num: u32) -> u32 {
        num
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type2d
    }
}

unsafe impl MultisampleType for Type2dMultisample {
}

pub struct Type3d;

unsafe impl TypeMarker for Type3d {
}

unsafe impl ImageViewTypeMarker for Type3d {
}

unsafe impl ImageTypeMarker for Type3d {
    type Dimensions = [u32; 3];
    type NumSamples = ();

    #[inline]
    fn extent(dims: [u32; 3]) -> [u32; 3] {
        dims
    }

    #[inline]
    fn array_layers(_: [u32; 3]) -> u32 {
        1
    }

    #[inline]
    fn num_samples(_: ()) -> u32 {
        1
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type3d
    }
}

pub struct Type3dMultisample;

unsafe impl TypeMarker for Type3dMultisample {
}

unsafe impl ImageViewTypeMarker for Type3dMultisample {
}

unsafe impl ImageTypeMarker for Type3dMultisample {
    type Dimensions = [u32; 3];
    type NumSamples = u32;

    #[inline]
    fn extent(dims: [u32; 3]) -> [u32; 3] {
        dims
    }

    #[inline]
    fn array_layers(_: [u32; 3]) -> u32 {
        1
    }

    #[inline]
    fn num_samples(num: u32) -> u32 {
        num
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type3d
    }
}

unsafe impl MultisampleType for Type3dMultisample {
}

pub struct TypeCube;

unsafe impl TypeMarker for TypeCube {
}

unsafe impl ImageViewTypeMarker for TypeCube {
}

pub struct TypeCubeMultisample;

unsafe impl TypeMarker for TypeCubeMultisample {
}

unsafe impl ImageViewTypeMarker for TypeCubeMultisample {
}

unsafe impl MultisampleType for TypeCubeMultisample {
}

pub struct Type1dArray;

unsafe impl TypeMarker for Type1dArray {
}

unsafe impl ImageViewTypeMarker for Type1dArray {
}

pub struct Type1dArrayMultisample;

unsafe impl TypeMarker for Type1dArrayMultisample {
}

unsafe impl ImageViewTypeMarker for Type1dArrayMultisample {
}

unsafe impl MultisampleType for Type1dArrayMultisample {
}

pub struct Type2dArray;

unsafe impl TypeMarker for Type2dArray {
}

unsafe impl ImageViewTypeMarker for Type2dArray {
}

unsafe impl ImageTypeMarker for Type2dArray {
    type Dimensions = ([u32; 2], u32);
    type NumSamples = ();

    #[inline]
    fn extent(dims: ([u32; 2], u32)) -> [u32; 3] {
        [dims.0[0], dims.0[1], 1]
    }

    #[inline]
    fn array_layers(dims: ([u32; 2], u32)) -> u32 {
        dims.1
    }

    #[inline]
    fn num_samples(_: ()) -> u32 {
        1
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type2d
    }
}

pub struct Type2dArrayMultisample;

unsafe impl TypeMarker for Type2dArrayMultisample {
}

unsafe impl ImageViewTypeMarker for Type2dArrayMultisample {
}

unsafe impl ImageTypeMarker for Type2dArrayMultisample {
    type Dimensions = ([u32; 2], u32);
    type NumSamples = u32;

    #[inline]
    fn extent(dims: ([u32; 2], u32)) -> [u32; 3] {
        [dims.0[0], dims.0[1], 1]
    }

    #[inline]
    fn array_layers(dims: ([u32; 2], u32)) -> u32 {
        dims.1
    }

    #[inline]
    fn num_samples(num: u32) -> u32 {
        num
    }

    #[inline]
    fn ty() -> ImageType {
        ImageType::Type2d
    }
}

unsafe impl MultisampleType for Type2dArrayMultisample {
}

pub struct TypeCubeArray;

unsafe impl TypeMarker for TypeCubeArray {
}

unsafe impl ImageViewTypeMarker for TypeCubeArray {
}

pub struct TypeCubeArrayMultisample;

unsafe impl TypeMarker for TypeCubeArrayMultisample {
}

unsafe impl ImageViewTypeMarker for TypeCubeArrayMultisample {
}

unsafe impl MultisampleType for TypeCubeArrayMultisample {
}

/// A subpart of an image.
///
/// Each mipmap level of each array layer is a *subresource*. This structure corresponds to a
/// range of subresources.
///
/// This object doesn't correspond to any Vulkan object. It exists for the programmer's
/// convenience.
pub struct ImageSubresourceRange<'a, Ty: 'a, F: 'a, M: 'a>
    where Ty: ImageTypeMarker, F: FormatDesc, M: MemorySource
{
    image: &'a Arc<Image<Ty, F, M>>,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
}

impl<'a, Ty: 'a, F: 'a, M: 'a> From<&'a Arc<Image<Ty, F, M>>> for ImageSubresourceRange<'a, Ty, F, M>
    where Ty: ImageTypeMarker, F: FormatDesc, M: MemorySource
{
    #[inline]
    fn from(image: &'a Arc<Image<Ty, F, M>>) -> ImageSubresourceRange<'a, Ty, F, M> {
        ImageSubresourceRange {
            image: image,
            base_mip_level: 0,
            level_count: image.mipmap_levels(),
            base_array_layer: 0,
            layer_count: image.array_layers(),
        }
    }
}
