// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::os::raw::c_void;

macro_rules! ptrs {
    ($struct_name:ident, { $($name:ident => ($($param_n:ident: $param_ty:ty),*) -> $ret:ty,)+ }) => (
        pub struct $struct_name {
            $(
                $name: extern "system" fn($($param_ty),*) -> $ret,
            )+
        }

        impl fmt::Debug for $struct_name {
            #[inline]
            fn fmt(&self, fmt: &mut fmt::Formatter) -> ::std::result::Result<(), fmt::Error> {
                write!(fmt, "<Vulkan functions>")       // TODO:
            }
        }

        unsafe impl Send for $struct_name {}
        unsafe impl Sync for $struct_name {}

        impl $struct_name {
            pub fn load<F>(mut f: F) -> $struct_name
                where F: FnMut(&CStr) -> *const c_void
            {
                $struct_name {
                    $(
                        $name: unsafe {
                            extern "system" fn $name($(_: $param_ty),*) { panic!("function pointer `{}` not loaded", stringify!($name)) }
                            let name = CString::new(concat!("vk", stringify!($name)).to_owned()).unwrap();
                            let val = f(&name);
                            if val.is_null() { mem::transmute($name as *const ()) } else { mem::transmute(val) }
                        },
                    )+
                }
            }

            $(
                #[inline]
                pub unsafe fn $name(&self $(, $param_n: $param_ty)*) -> $ret {
                    let ptr = self.$name;
                    ptr($($param_n),*)
                }
            )+
        }
    )
}

ptrs!(Static, {
    GetInstanceProcAddr => (instance: Instance, pName: *const c_char) -> PFN_vkVoidFunction,
});

ptrs!(EntryPoints, {
    CreateInstance => (pCreateInfo: *const InstanceCreateInfo, pAllocator: *const AllocationCallbacks, pInstance: *mut Instance) -> Result,
    EnumerateInstanceExtensionProperties => (pLayerName: *const c_char, pPropertyCount: *mut u32, pProperties: *mut ExtensionProperties) -> Result,
    EnumerateInstanceLayerProperties => (pPropertyCount: *mut u32, pProperties: *mut LayerProperties) -> Result,
});

ptrs!(InstancePointers, {
    DestroyInstance => (instance: Instance, pAllocator: *const AllocationCallbacks) -> (),
    GetDeviceProcAddr => (device: Device, pName: *const c_char) -> PFN_vkVoidFunction,
    EnumeratePhysicalDevices => (instance: Instance, pPhysicalDeviceCount: *mut u32, pPhysicalDevices: *mut PhysicalDevice) -> Result,
    EnumerateDeviceExtensionProperties => (physicalDevice: PhysicalDevice, pLayerName: *const c_char, pPropertyCount: *mut u32, pProperties: *mut ExtensionProperties) -> Result,
    EnumerateDeviceLayerProperties => (physicalDevice: PhysicalDevice, pPropertyCount: *mut u32, pProperties: *mut LayerProperties) -> Result,
    CreateDevice => (physicalDevice: PhysicalDevice, pCreateInfo: *const DeviceCreateInfo, pAllocator: *const AllocationCallbacks, pDevice: *mut Device) -> Result,
    GetPhysicalDeviceFeatures => (physicalDevice: PhysicalDevice, pFeatures: *mut PhysicalDeviceFeatures) -> (),
    GetPhysicalDeviceFormatProperties => (physicalDevice: PhysicalDevice, format: Format, pFormatProperties: *mut FormatProperties) -> (),
    GetPhysicalDeviceImageFormatProperties => (physicalDevice: PhysicalDevice, format: Format, ty: ImageType, tiling: ImageTiling, usage: ImageUsageFlags, flags: ImageCreateFlags, pImageFormatProperties: *mut ImageFormatProperties) -> Result,
    GetPhysicalDeviceProperties => (physicalDevice: PhysicalDevice, pProperties: *mut PhysicalDeviceProperties) -> (),
    GetPhysicalDeviceQueueFamilyProperties => (physicalDevice: PhysicalDevice, pQueueFamilyPropertyCount: *mut u32, pQueueFamilyProperties: *mut QueueFamilyProperties) -> (),
    GetPhysicalDeviceMemoryProperties => (physicalDevice: PhysicalDevice, pMemoryProperties: *mut PhysicalDeviceMemoryProperties) -> (),
    GetPhysicalDeviceSparseImageFormatProperties => (physicalDevice: PhysicalDevice, format: Format, ty: ImageType, samples: SampleCountFlagBits, usage: ImageUsageFlags, tiling: ImageTiling, pPropertyCount: *mut u32, pProperties: *mut SparseImageFormatProperties) -> (),
    DestroySurfaceKHR => (instance: Instance, surface: SurfaceKHR, pAllocator: *const AllocationCallbacks) -> (),
    CreateXlibSurfaceKHR => (instance: Instance, pCreateInfo: *const XlibSurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    GetPhysicalDeviceXlibPresentationSupportKHR => (physicalDevice: PhysicalDevice, queueFamilyIndex: u32, dpy: *mut c_void, visualID: u32/* FIXME: VisualID */) -> Bool32,
    CreateXcbSurfaceKHR => (instance: Instance, pCreateInfo: *const XcbSurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    GetPhysicalDeviceXcbPresentationSupportKHR => (physicalDevice: PhysicalDevice, queueFamilyIndex: u32, connection: *mut c_void, visual_id: u32 /* FIXME: xcb_visualid */) -> Bool32,
    CreateWaylandSurfaceKHR => (instance: Instance, pCreateInfo: *const WaylandSurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    GetPhysicalDeviceWaylandPresentationSupportKHR => (physicalDevice: PhysicalDevice, queueFamilyIndex: u32, display: *mut c_void) -> Bool32,
    CreateMirSurfaceKHR => (instance: Instance, pCreateInfo: *const MirSurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    GetPhysicalDeviceMirPresentationSupportKHR => (physicalDevice: PhysicalDevice, queueFamilyIndex: u32, connection: *mut c_void) -> Bool32,
    CreateAndroidSurfaceKHR => (instance: Instance, pCreateInfo: *const AndroidSurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    CreateWin32SurfaceKHR => (instance: Instance, pCreateInfo: *const Win32SurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    GetPhysicalDeviceWin32PresentationSupportKHR => (physicalDevice: PhysicalDevice, queueFamilyIndex: u32) -> Bool32,
    GetPhysicalDeviceDisplayPropertiesKHR => (physicalDevice: PhysicalDevice, pPropertyCount: *mut u32, pProperties: *mut DisplayPropertiesKHR) -> Result,
    GetPhysicalDeviceDisplayPlanePropertiesKHR => (physicalDevice: PhysicalDevice, pPropertyCount: *mut u32, pProperties: *mut DisplayPlanePropertiesKHR) -> Result,
    GetDisplayPlaneSupportedDisplaysKHR => (physicalDevice: PhysicalDevice, planeIndex: u32, pDisplayCount: *mut u32, pDisplays: *mut DisplayKHR) -> Result,
    GetDisplayModePropertiesKHR => (physicalDevice: PhysicalDevice, display: DisplayKHR, pPropertyCount: *mut u32, pProperties: *mut DisplayModePropertiesKHR) -> Result,
    CreateDisplayModeKHR => (physicalDevice: PhysicalDevice, display: DisplayKHR, pCreateInfo: *const DisplayModeCreateInfoKHR, pAllocator: *const AllocationCallbacks, pMode: *mut DisplayModeKHR) -> Result,
    GetDisplayPlaneCapabilitiesKHR => (physicalDevice: PhysicalDevice, mode: DisplayModeKHR, planeIndex: u32, pCapabilities: *mut DisplayPlaneCapabilitiesKHR) -> Result,
    CreateDisplayPlaneSurfaceKHR => (instance: Instance, pCreateInfo: *const DisplaySurfaceCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSurface: *mut SurfaceKHR) -> Result,
    GetPhysicalDeviceSurfaceSupportKHR => (physicalDevice: PhysicalDevice, queueFamilyIndex: u32, surface: SurfaceKHR, pSupported: *mut Bool32) -> Result,
    GetPhysicalDeviceSurfaceCapabilitiesKHR => (physicalDevice: PhysicalDevice, surface: SurfaceKHR, pSurfaceCapabilities: *mut SurfaceCapabilitiesKHR) -> Result,
    GetPhysicalDeviceSurfaceFormatsKHR => (physicalDevice: PhysicalDevice, surface: SurfaceKHR, pSurfaceFormatCount: *mut u32, pSurfaceFormats: *mut SurfaceFormatKHR) -> Result,
    GetPhysicalDeviceSurfacePresentModesKHR => (physicalDevice: PhysicalDevice, surface: SurfaceKHR, pPresentModeCount: *mut u32, pPresentModes: *mut PresentModeKHR) -> Result,
    CreateDebugReportCallbackEXT => (instance: Instance, pCreateInfo: *const DebugReportCallbackCreateInfoEXT, pAllocator: *const AllocationCallbacks, pCallback: *mut DebugReportCallbackEXT) -> Result,
    DestroyDebugReportCallbackEXT => (instance: Instance, callback: DebugReportCallbackEXT, pAllocator: *const AllocationCallbacks) -> (),
    DebugReportMessageEXT => (instance: Instance, flags: DebugReportFlagsEXT, objectType: DebugReportObjectTypeEXT, object: u64, location: usize, messageCode: i32, pLayerPrefix: *const c_char, pMessage: *const c_char) -> (),
});

ptrs!(DevicePointers, {
    DestroyDevice => (device: Device, pAllocator: *const AllocationCallbacks) -> (),
    GetDeviceQueue => (device: Device, queueFamilyIndex: u32, queueIndex: u32, pQueue: *mut Queue) -> (),
    QueueSubmit => (queue: Queue, submitCount: u32, pSubmits: *const SubmitInfo, fence: Fence) -> Result,
    QueueWaitIdle => (queue: Queue) -> Result,
    DeviceWaitIdle => (device: Device) -> Result,
    AllocateMemory => (device: Device, pAllocateInfo: *const MemoryAllocateInfo, pAllocator: *const AllocationCallbacks, pMemory: *mut DeviceMemory) -> Result,
    FreeMemory => (device: Device, memory: DeviceMemory, pAllocator: *const AllocationCallbacks) -> (),
    MapMemory => (device: Device, memory: DeviceMemory, offset: DeviceSize, size: DeviceSize, flags: MemoryMapFlags, ppData: *mut *mut c_void) -> Result,
    UnmapMemory => (device: Device, memory: DeviceMemory) -> (),
    FlushMappedMemoryRanges => (device: Device, memoryRangeCount: u32, pMemoryRanges: *const MappedMemoryRange) -> Result,
    InvalidateMappedMemoryRanges => (device: Device, memoryRangeCount: u32, pMemoryRanges: *const MappedMemoryRange) -> Result,
    GetDeviceMemoryCommitment => (device: Device, memory: DeviceMemory, pCommittedMemoryInBytes: *mut DeviceSize) -> (),
    BindBufferMemory => (device: Device, buffer: Buffer, memory: DeviceMemory, memoryOffset: DeviceSize) -> Result,
    BindImageMemory => (device: Device, image: Image, memory: DeviceMemory, memoryOffset: DeviceSize) -> Result,
    GetBufferMemoryRequirements => (device: Device, buffer: Buffer, pMemoryRequirements: *mut MemoryRequirements) -> (),
    GetImageMemoryRequirements => (device: Device, image: Image, pMemoryRequirements: *mut MemoryRequirements) -> (),
    GetImageSparseMemoryRequirements => (device: Device, image: Image, pSparseMemoryRequirementCount: *mut u32, pSparseMemoryRequirements: *mut SparseImageMemoryRequirements) -> (),
    QueueBindSparse => (queue: Queue, bindInfoCount: u32, pBindInfo: *const BindSparseInfo, fence: Fence) -> Result,
    CreateFence => (device: Device, pCreateInfo: *const FenceCreateInfo, pAllocator: *const AllocationCallbacks, pFence: *mut Fence) -> Result,
    DestroyFence => (device: Device, fence: Fence, pAllocator: *const AllocationCallbacks) -> (),
    ResetFences => (device: Device, fenceCount: u32, pFences: *const Fence) -> Result,
    GetFenceStatus => (device: Device, fence: Fence) -> Result,
    WaitForFences => (device: Device, fenceCount: u32, pFences: *const Fence, waitAll: Bool32, timeout: u64) -> Result,
    CreateSemaphore => (device: Device, pCreateInfo: *const SemaphoreCreateInfo, pAllocator: *const AllocationCallbacks, pSemaphore: *mut Semaphore) -> Result,
    DestroySemaphore => (device: Device, semaphore: Semaphore, pAllocator: *const AllocationCallbacks) -> (),
    CreateEvent => (device: Device, pCreateInfo: *const EventCreateInfo, pAllocator: *const AllocationCallbacks, pEvent: *mut Event) -> Result,
    DestroyEvent => (device: Device, event: Event, pAllocator: *const AllocationCallbacks) -> (),
    GetEventStatus => (device: Device, event: Event) -> Result,
    SetEvent => (device: Device, event: Event) -> Result,
    ResetEvent => (device: Device, event: Event) -> Result,
    CreateQueryPool => (device: Device, pCreateInfo: *const QueryPoolCreateInfo, pAllocator: *const AllocationCallbacks, pQueryPool: *mut QueryPool) -> Result,
    DestroyQueryPool => (device: Device, queryPool: QueryPool, pAllocator: *const AllocationCallbacks) -> (),
    GetQueryPoolResults => (device: Device, queryPool: QueryPool, firstQuery: u32, queryCount: u32, dataSize: usize, pData: *mut c_void, stride: DeviceSize, flags: QueryResultFlags) -> Result,
    CreateBuffer => (device: Device, pCreateInfo: *const BufferCreateInfo, pAllocator: *const AllocationCallbacks, pBuffer: *mut Buffer) -> Result,
    DestroyBuffer => (device: Device, buffer: Buffer, pAllocator: *const AllocationCallbacks) -> (),
    CreateBufferView => (device: Device, pCreateInfo: *const BufferViewCreateInfo, pAllocator: *const AllocationCallbacks, pView: *mut BufferView) -> Result,
    DestroyBufferView => (device: Device, bufferView: BufferView, pAllocator: *const AllocationCallbacks) -> (),
    CreateImage => (device: Device, pCreateInfo: *const ImageCreateInfo, pAllocator: *const AllocationCallbacks, pImage: *mut Image) -> Result,
    DestroyImage => (device: Device, image: Image, pAllocator: *const AllocationCallbacks) -> (),
    GetImageSubresourceLayout => (device: Device, image: Image, pSubresource: *const ImageSubresource, pLayout: *mut SubresourceLayout) -> (),
    CreateImageView => (device: Device, pCreateInfo: *const ImageViewCreateInfo, pAllocator: *const AllocationCallbacks, pView: *mut ImageView) -> Result,
    DestroyImageView => (device: Device, imageView: ImageView, pAllocator: *const AllocationCallbacks) -> (),
    CreateShaderModule => (device: Device, pCreateInfo: *const ShaderModuleCreateInfo, pAllocator: *const AllocationCallbacks, pShaderModule: *mut ShaderModule) -> Result,
    DestroyShaderModule => (device: Device, shaderModule: ShaderModule, pAllocator: *const AllocationCallbacks) -> (),
    CreatePipelineCache => (device: Device, pCreateInfo: *const PipelineCacheCreateInfo, pAllocator: *const AllocationCallbacks, pPipelineCache: *mut PipelineCache) -> Result,
    DestroyPipelineCache => (device: Device, pipelineCache: PipelineCache, pAllocator: *const AllocationCallbacks) -> (),
    GetPipelineCacheData => (device: Device, pipelineCache: PipelineCache, pDataSize: *mut usize, pData: *mut c_void) -> Result,
    MergePipelineCaches => (device: Device, dstCache: PipelineCache, srcCacheCount: u32, pSrcCaches: *const PipelineCache) -> Result,
    CreateGraphicsPipelines => (device: Device, pipelineCache: PipelineCache, createInfoCount: u32, pCreateInfos: *const GraphicsPipelineCreateInfo, pAllocator: *const AllocationCallbacks, pPipelines: *mut Pipeline) -> Result,
    CreateComputePipelines => (device: Device, pipelineCache: PipelineCache, createInfoCount: u32, pCreateInfos: *const ComputePipelineCreateInfo, pAllocator: *const AllocationCallbacks, pPipelines: *mut Pipeline) -> Result,
    DestroyPipeline => (device: Device, pipeline: Pipeline, pAllocator: *const AllocationCallbacks) -> (),
    CreatePipelineLayout => (device: Device, pCreateInfo: *const PipelineLayoutCreateInfo, pAllocator: *const AllocationCallbacks, pPipelineLayout: *mut PipelineLayout) -> Result,
    DestroyPipelineLayout => (device: Device, pipelineLayout: PipelineLayout, pAllocator: *const AllocationCallbacks) -> (),
    CreateSampler => (device: Device, pCreateInfo: *const SamplerCreateInfo, pAllocator: *const AllocationCallbacks, pSampler: *mut Sampler) -> Result,
    DestroySampler => (device: Device, sampler: Sampler, pAllocator: *const AllocationCallbacks) -> (),
    CreateDescriptorSetLayout => (device: Device, pCreateInfo: *const DescriptorSetLayoutCreateInfo, pAllocator: *const AllocationCallbacks, pSetLayout: *mut DescriptorSetLayout) -> Result,
    DestroyDescriptorSetLayout => (device: Device, descriptorSetLayout: DescriptorSetLayout, pAllocator: *const AllocationCallbacks) -> (),
    CreateDescriptorPool => (device: Device, pCreateInfo: *const DescriptorPoolCreateInfo, pAllocator: *const AllocationCallbacks, pDescriptorPool: *mut DescriptorPool) -> Result,
    DestroyDescriptorPool => (device: Device, descriptorPool: DescriptorPool, pAllocator: *const AllocationCallbacks) -> (),
    ResetDescriptorPool => (device: Device, descriptorPool: DescriptorPool, flags: DescriptorPoolResetFlags) -> Result,
    AllocateDescriptorSets => (device: Device, pAllocateInfo: *const DescriptorSetAllocateInfo, pDescriptorSets: *mut DescriptorSet) -> Result,
    FreeDescriptorSets => (device: Device, descriptorPool: DescriptorPool, descriptorSetCount: u32, pDescriptorSets: *const DescriptorSet) -> Result,
    UpdateDescriptorSets => (device: Device, descriptorWriteCount: u32, pDescriptorWrites: *const WriteDescriptorSet, descriptorCopyCount: u32, pDescriptorCopies: *const CopyDescriptorSet) -> (),
    CreateFramebuffer => (device: Device, pCreateInfo: *const FramebufferCreateInfo, pAllocator: *const AllocationCallbacks, pFramebuffer: *mut Framebuffer) -> Result,
    DestroyFramebuffer => (device: Device, framebuffer: Framebuffer, pAllocator: *const AllocationCallbacks) -> (),
    CreateRenderPass => (device: Device, pCreateInfo: *const RenderPassCreateInfo, pAllocator: *const AllocationCallbacks, pRenderPass: *mut RenderPass) -> Result,
    DestroyRenderPass => (device: Device, renderPass: RenderPass, pAllocator: *const AllocationCallbacks) -> (),
    GetRenderAreaGranularity => (device: Device, renderPass: RenderPass, pGranularity: *mut Extent2D) -> (),
    CreateCommandPool => (device: Device, pCreateInfo: *const CommandPoolCreateInfo, pAllocator: *const AllocationCallbacks, pCommandPool: *mut CommandPool) -> Result,
    DestroyCommandPool => (device: Device, commandPool: CommandPool, pAllocator: *const AllocationCallbacks) -> (),
    ResetCommandPool => (device: Device, commandPool: CommandPool, flags: CommandPoolResetFlags) -> Result,
    AllocateCommandBuffers => (device: Device, pAllocateInfo: *const CommandBufferAllocateInfo, pCommandBuffers: *mut CommandBuffer) -> Result,
    FreeCommandBuffers => (device: Device, commandPool: CommandPool, commandBufferCount: u32, pCommandBuffers: *const CommandBuffer) -> (),
    BeginCommandBuffer => (commandBuffer: CommandBuffer, pBeginInfo: *const CommandBufferBeginInfo) -> Result,
    EndCommandBuffer => (commandBuffer: CommandBuffer) -> Result,
    ResetCommandBuffer => (commandBuffer: CommandBuffer, flags: CommandBufferResetFlags) -> Result,
    CmdBindPipeline => (commandBuffer: CommandBuffer, pipelineBindPoint: PipelineBindPoint, pipeline: Pipeline) -> (),
    CmdSetViewport => (commandBuffer: CommandBuffer, firstViewport: u32, viewportCount: u32, pViewports: *const Viewport) -> (),
    CmdSetScissor => (commandBuffer: CommandBuffer, firstScissor: u32, scissorCount: u32, pScissors: *const Rect2D) -> (),
    CmdSetLineWidth => (commandBuffer: CommandBuffer, lineWidth: f32) -> (),
    CmdSetDepthBias => (commandBuffer: CommandBuffer, depthBiasConstantFactor: f32, depthBiasClamp: f32, depthBiasSlopeFactor: f32) -> (),
    CmdSetBlendConstants => (commandBuffer: CommandBuffer, blendConstants: [f32; 4]) -> (),
    CmdSetDepthBounds => (commandBuffer: CommandBuffer, minDepthBounds: f32, maxDepthBounds: f32) -> (),
    CmdSetStencilCompareMask => (commandBuffer: CommandBuffer, faceMask: StencilFaceFlags, compareMask: u32) -> (),
    CmdSetStencilWriteMask => (commandBuffer: CommandBuffer, faceMask: StencilFaceFlags, writeMask: u32) -> (),
    CmdSetStencilReference => (commandBuffer: CommandBuffer, faceMask: StencilFaceFlags, reference: u32) -> (),
    CmdBindDescriptorSets => (commandBuffer: CommandBuffer, pipelineBindPoint: PipelineBindPoint, layout: PipelineLayout, firstSet: u32, descriptorSetCount: u32, pDescriptorSets: *const DescriptorSet, dynamicOffsetCount: u32, pDynamicOffsets: *const u32) -> (),
    CmdBindIndexBuffer => (commandBuffer: CommandBuffer, buffer: Buffer, offset: DeviceSize, indexType: IndexType) -> (),
    CmdBindVertexBuffers => (commandBuffer: CommandBuffer, firstBinding: u32, bindingCount: u32, pBuffers: *const Buffer, pOffsets: *const DeviceSize) -> (),
    CmdDraw => (commandBuffer: CommandBuffer, vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32) -> (),
    CmdDrawIndexed => (commandBuffer: CommandBuffer, indexCount: u32, instanceCount: u32, firstIndex: u32, vertexOffset: i32, firstInstance: u32) -> (),
    CmdDrawIndirect => (commandBuffer: CommandBuffer, buffer: Buffer, offset: DeviceSize, drawCount: u32, stride: u32) -> (),
    CmdDrawIndexedIndirect => (commandBuffer: CommandBuffer, buffer: Buffer, offset: DeviceSize, drawCount: u32, stride: u32) -> (),
    CmdDispatch => (commandBuffer: CommandBuffer, x: u32, y: u32, z: u32) -> (),
    CmdDispatchIndirect => (commandBuffer: CommandBuffer, buffer: Buffer, offset: DeviceSize) -> (),
    CmdCopyBuffer => (commandBuffer: CommandBuffer, srcBuffer: Buffer, dstBuffer: Buffer, regionCount: u32, pRegions: *const BufferCopy) -> (),
    CmdCopyImage => (commandBuffer: CommandBuffer, srcImage: Image, srcImageLayout: ImageLayout, dstImage: Image, dstImageLayout: ImageLayout, regionCount: u32, pRegions: *const ImageCopy) -> (),
    CmdBlitImage => (commandBuffer: CommandBuffer, srcImage: Image, srcImageLayout: ImageLayout, dstImage: Image, dstImageLayout: ImageLayout, regionCount: u32, pRegions: *const ImageBlit, filter: Filter) -> (),
    CmdCopyBufferToImage => (commandBuffer: CommandBuffer, srcBuffer: Buffer, dstImage: Image, dstImageLayout: ImageLayout, regionCount: u32, pRegions: *const BufferImageCopy) -> (),
    CmdCopyImageToBuffer => (commandBuffer: CommandBuffer, srcImage: Image, srcImageLayout: ImageLayout, dstBuffer: Buffer, regionCount: u32, pRegions: *const BufferImageCopy) -> (),
    CmdUpdateBuffer => (commandBuffer: CommandBuffer, dstBuffer: Buffer, dstOffset: DeviceSize, dataSize: DeviceSize, pData: *const u32) -> (),
    CmdFillBuffer => (commandBuffer: CommandBuffer, dstBuffer: Buffer, dstOffset: DeviceSize, size: DeviceSize, data: u32) -> (),
    CmdClearColorImage => (commandBuffer: CommandBuffer, image: Image, imageLayout: ImageLayout, pColor: *const ClearColorValue, rangeCount: u32, pRanges: *const ImageSubresourceRange) -> (),
    CmdClearDepthStencilImage => (commandBuffer: CommandBuffer, image: Image, imageLayout: ImageLayout, pDepthStencil: *const ClearDepthStencilValue, rangeCount: u32, pRanges: *const ImageSubresourceRange) -> (),
    CmdClearAttachments => (commandBuffer: CommandBuffer, attachmentCount: u32, pAttachments: *const ClearAttachment, rectCount: u32, pRects: *const ClearRect) -> (),
    CmdResolveImage => (commandBuffer: CommandBuffer, srcImage: Image, srcImageLayout: ImageLayout, dstImage: Image, dstImageLayout: ImageLayout, regionCount: u32, pRegions: *const ImageResolve) -> (),
    CmdSetEvent => (commandBuffer: CommandBuffer, event: Event, stageMask: PipelineStageFlags) -> (),
    CmdResetEvent => (commandBuffer: CommandBuffer, event: Event, stageMask: PipelineStageFlags) -> (),
    CmdWaitEvents => (commandBuffer: CommandBuffer, eventCount: u32, pEvents: *const Event, srcStageMask: PipelineStageFlags, dstStageMask: PipelineStageFlags, memoryBarrierCount: u32, pMemoryBarriers: *const MemoryBarrier, bufferMemoryBarrierCount: u32, pBufferMemoryBarriers: *const BufferMemoryBarrier, imageMemoryBarrierCount: u32, pImageMemoryBarriers: *const ImageMemoryBarrier) -> (),
    CmdPipelineBarrier => (commandBuffer: CommandBuffer, srcStageMask: PipelineStageFlags, dstStageMask: PipelineStageFlags, dependencyFlags: DependencyFlags, memoryBarrierCount: u32, pMemoryBarriers: *const MemoryBarrier, bufferMemoryBarrierCount: u32, pBufferMemoryBarriers: *const BufferMemoryBarrier, imageMemoryBarrierCount: u32, pImageMemoryBarriers: *const ImageMemoryBarrier) -> (),
    CmdBeginQuery => (commandBuffer: CommandBuffer, queryPool: QueryPool, query: u32, flags: QueryControlFlags) -> (),
    CmdEndQuery => (commandBuffer: CommandBuffer, queryPool: QueryPool, query: u32) -> (),
    CmdResetQueryPool => (commandBuffer: CommandBuffer, queryPool: QueryPool, firstQuery: u32, queryCount: u32) -> (),
    CmdWriteTimestamp => (commandBuffer: CommandBuffer, pipelineStage: PipelineStageFlagBits, queryPool: QueryPool, query: u32) -> (),
    CmdCopyQueryPoolResults => (commandBuffer: CommandBuffer, queryPool: QueryPool, firstQuery: u32, queryCount: u32, dstBuffer: Buffer, dstOffset: DeviceSize, stride: DeviceSize, flags: QueryResultFlags) -> (),
    CmdPushConstants => (commandBuffer: CommandBuffer, layout: PipelineLayout, stageFlags: ShaderStageFlags, offset: u32, size: u32, pValues: *const c_void) -> (),
    CmdBeginRenderPass => (commandBuffer: CommandBuffer, pRenderPassBegin: *const RenderPassBeginInfo, contents: SubpassContents) -> (),
    CmdNextSubpass => (commandBuffer: CommandBuffer, contents: SubpassContents) -> (),
    CmdEndRenderPass => (commandBuffer: CommandBuffer) -> (),
    CmdExecuteCommands => (commandBuffer: CommandBuffer, commandBufferCount: u32, pCommandBuffers: *const CommandBuffer) -> (),
    CreateSwapchainKHR => (device: Device, pCreateInfo: *const SwapchainCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSwapchain: *mut SwapchainKHR) -> Result,
    DestroySwapchainKHR => (device: Device, swapchain: SwapchainKHR, pAllocator: *const AllocationCallbacks) -> (),
    GetSwapchainImagesKHR => (device: Device, swapchain: SwapchainKHR, pSwapchainImageCount: *mut u32, pSwapchainImages: *mut Image) -> Result,
    AcquireNextImageKHR => (device: Device, swapchain: SwapchainKHR, timeout: u64, semaphore: Semaphore, fence: Fence, pImageIndex: *mut u32) -> Result,
    QueuePresentKHR => (queue: Queue, pPresentInfo: *const PresentInfoKHR) -> Result,
    CreateSharedSwapchainsKHR => (device: Device, swapchainCount: u32, pCreateInfos: *const SwapchainCreateInfoKHR, pAllocator: *const AllocationCallbacks, pSwapchains: *mut SwapchainKHR) -> Result,
});
