use std::mem;

pub type PFN_vkAllocationFunction = extern fn(*mut c_void, usize, usize, SystemAllocationScope) -> *mut c_void;
pub type PFN_vkReallocationFunction = extern fn(*mut c_void, *mut c_void, usize, usize, SystemAllocationScope) -> *mut c_void;
pub type PFN_vkFreeFunction = extern fn(*mut c_void, *mut c_void);
pub type PFN_vkInternalAllocationNotification = extern fn(*mut c_void, usize, InternalAllocationType, SystemAllocationScope) -> *mut c_void;
pub type PFN_vkInternalFreeNotification = extern fn(*mut c_void, usize, InternalAllocationType, SystemAllocationScope) -> *mut c_void;

pub type PFN_vkVoidFunction = extern fn() -> ();

#[repr(C)]
pub struct ApplicationInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub pApplicationName: *const c_char,
    pub applicationVersion: u32,
    pub pEngineName: *const c_char,
    pub engineVersion: u32,
    pub apiVersion: u32,
}

#[repr(C)]
pub struct InstanceCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: InstanceCreateFlags,
    pub pApplicationInfo: *const ApplicationInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
}

#[repr(C)]
pub struct AllocationCallbacks {
    pub pUserData: *mut c_void,
    pub pfnAllocation: PFN_vkAllocationFunction,
    pub pfnReallocation: PFN_vkReallocationFunction,
    pub pfnFree: PFN_vkFreeFunction,
    pub pfnInternalAllocation: PFN_vkInternalAllocationNotification,
    pub pfnInternalFree: PFN_vkInternalFreeNotification,
}

#[repr(C)]
pub struct PhysicalDeviceFeatures {
    pub robustBufferAccess: Bool32,
    pub fullDrawIndexUint32: Bool32,
    pub imageCubeArray: Bool32,
    pub independentBlend: Bool32,
    pub geometryShader: Bool32,
    pub tessellationShader: Bool32,
    pub sampleRateShading: Bool32,
    pub dualSrcBlend: Bool32,
    pub logicOp: Bool32,
    pub multiDrawIndirect: Bool32,
    pub drawIndirectFirstInstance: Bool32,
    pub depthClamp: Bool32,
    pub depthBiasClamp: Bool32,
    pub fillModeNonSolid: Bool32,
    pub depthBounds: Bool32,
    pub wideLines: Bool32,
    pub largePoints: Bool32,
    pub alphaToOne: Bool32,
    pub multiViewport: Bool32,
    pub samplerAnisotropy: Bool32,
    pub textureCompressionETC2: Bool32,
    pub textureCompressionASTC_LDR: Bool32,
    pub textureCompressionBC: Bool32,
    pub occlusionQueryPrecise: Bool32,
    pub pipelineStatisticsQuery: Bool32,
    pub vertexPipelineStoresAndAtomics: Bool32,
    pub fragmentStoresAndAtomics: Bool32,
    pub shaderTessellationAndGeometryPointSize: Bool32,
    pub shaderImageGatherExtended: Bool32,
    pub shaderStorageImageExtendedFormats: Bool32,
    pub shaderStorageImageMultisample: Bool32,
    pub shaderStorageImageReadWithoutFormat: Bool32,
    pub shaderStorageImageWriteWithoutFormat: Bool32,
    pub shaderUniformBufferArrayDynamicIndexing: Bool32,
    pub shaderSampledImageArrayDynamicIndexing: Bool32,
    pub shaderStorageBufferArrayDynamicIndexing: Bool32,
    pub shaderStorageImageArrayDynamicIndexing: Bool32,
    pub shaderClipDistance: Bool32,
    pub shaderCullDistance: Bool32,
    pub shaderf3264: Bool32,
    pub shaderInt64: Bool32,
    pub shaderInt16: Bool32,
    pub shaderResourceResidency: Bool32,
    pub shaderResourceMinLod: Bool32,
    pub sparseBinding: Bool32,
    pub sparseResidencyBuffer: Bool32,
    pub sparseResidencyImage2D: Bool32,
    pub sparseResidencyImage3D: Bool32,
    pub sparseResidency2Samples: Bool32,
    pub sparseResidency4Samples: Bool32,
    pub sparseResidency8Samples: Bool32,
    pub sparseResidency16Samples: Bool32,
    pub sparseResidencyAliased: Bool32,
    pub variableMultisampleRate: Bool32,
    pub inheritedQueries: Bool32,
}

#[repr(C)]
pub struct FormatProperties {
    pub linearTilingFeatures: FormatFeatureFlags,
    pub optimalTilingFeatures: FormatFeatureFlags,
    pub bufferFeatures: FormatFeatureFlags,
}

#[repr(C)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
pub struct ImageFormatProperties {
    pub maxExtent: Extent3D,
    pub maxMipLevels: u32,
    pub maxArrayLayers: u32,
    pub sampleCounts: SampleCountFlags,
    pub maxResourceSize: DeviceSize,
}

#[repr(C)]
pub struct PhysicalDeviceLimits {
    pub maxImageDimension1D: u32,
    pub maxImageDimension2D: u32,
    pub maxImageDimension3D: u32,
    pub maxImageDimensionCube: u32,
    pub maxImageArrayLayers: u32,
    pub maxTexelBufferElements: u32,
    pub maxUniformBufferRange: u32,
    pub maxStorageBufferRange: u32,
    pub maxPushConstantsSize: u32,
    pub maxMemoryAllocationCount: u32,
    pub maxSamplerAllocationCount: u32,
    pub bufferImageGranularity: DeviceSize,
    pub sparseAddressSpaceSize: DeviceSize,
    pub maxBoundDescriptorSets: u32,
    pub maxPerStageDescriptorSamplers: u32,
    pub maxPerStageDescriptorUniformBuffers: u32,
    pub maxPerStageDescriptorStorageBuffers: u32,
    pub maxPerStageDescriptorSampledImages: u32,
    pub maxPerStageDescriptorStorageImages: u32,
    pub maxPerStageDescriptorInputAttachments: u32,
    pub maxPerStageResources: u32,
    pub maxDescriptorSetSamplers: u32,
    pub maxDescriptorSetUniformBuffers: u32,
    pub maxDescriptorSetUniformBuffersDynamic: u32,
    pub maxDescriptorSetStorageBuffers: u32,
    pub maxDescriptorSetStorageBuffersDynamic: u32,
    pub maxDescriptorSetSampledImages: u32,
    pub maxDescriptorSetStorageImages: u32,
    pub maxDescriptorSetInputAttachments: u32,
    pub maxVertexInputAttributes: u32,
    pub maxVertexInputBindings: u32,
    pub maxVertexInputAttributeOffset: u32,
    pub maxVertexInputBindingStride: u32,
    pub maxVertexOutputComponents: u32,
    pub maxTessellationGenerationLevel: u32,
    pub maxTessellationPatchSize: u32,
    pub maxTessellationControlPerVertexInputComponents: u32,
    pub maxTessellationControlPerVertexOutputComponents: u32,
    pub maxTessellationControlPerPatchOutputComponents: u32,
    pub maxTessellationControlTotalOutputComponents: u32,
    pub maxTessellationEvaluationInputComponents: u32,
    pub maxTessellationEvaluationOutputComponents: u32,
    pub maxGeometryShaderInvocations: u32,
    pub maxGeometryInputComponents: u32,
    pub maxGeometryOutputComponents: u32,
    pub maxGeometryOutputVertices: u32,
    pub maxGeometryTotalOutputComponents: u32,
    pub maxFragmentInputComponents: u32,
    pub maxFragmentOutputAttachments: u32,
    pub maxFragmentDualSrcAttachments: u32,
    pub maxFragmentCombinedOutputResources: u32,
    pub maxComputeSharedMemorySize: u32,
    pub maxComputeWorkGroupCount: [u32; 3],
    pub maxComputeWorkGroupInvocations: u32,
    pub maxComputeWorkGroupSize: [u32; 3],
    pub subPixelPrecisionBits: u32,
    pub subTexelPrecisionBits: u32,
    pub mipmapPrecisionBits: u32,
    pub maxDrawIndexedIndexValue: u32,
    pub maxDrawIndirectCount: u32,
    pub maxSamplerLodBias: f32,
    pub maxSamplerAnisotropy: f32,
    pub maxViewports: u32,
    pub maxViewportDimensions: [u32; 2],
    pub viewportBoundsRange: [f32; 2],
    pub viewportSubPixelBits: u32,
    pub minMemoryMapAlignment: usize,
    pub minTexelBufferOffsetAlignment: DeviceSize,
    pub minUniformBufferOffsetAlignment: DeviceSize,
    pub minStorageBufferOffsetAlignment: DeviceSize,
    pub minTexelOffset: i32,
    pub maxTexelOffset: u32,
    pub minTexelGatherOffset: i32,
    pub maxTexelGatherOffset: u32,
    pub minInterpolationOffset: f32,
    pub maxInterpolationOffset: f32,
    pub subPixelInterpolationOffsetBits: u32,
    pub maxFramebufferWidth: u32,
    pub maxFramebufferHeight: u32,
    pub maxFramebufferLayers: u32,
    pub framebufferColorSampleCounts: SampleCountFlags,
    pub framebufferDepthSampleCounts: SampleCountFlags,
    pub framebufferStencilSampleCounts: SampleCountFlags,
    pub framebufferNoAttachmentsSampleCounts: SampleCountFlags,
    pub maxColorAttachments: u32,
    pub sampledImageColorSampleCounts: SampleCountFlags,
    pub sampledImageIntegerSampleCounts: SampleCountFlags,
    pub sampledImageDepthSampleCounts: SampleCountFlags,
    pub sampledImageStencilSampleCounts: SampleCountFlags,
    pub storageImageSampleCounts: SampleCountFlags,
    pub maxSampleMaskWords: u32,
    pub timestampComputeAndGraphics: Bool32,
    pub timestampPeriod: f32,
    pub maxClipDistances: u32,
    pub maxCullDistances: u32,
    pub maxCombinedClipAndCullDistances: u32,
    pub discreteQueuePriorities: u32,
    pub pointSizeRange: [f32; 2],
    pub lineWidthRange: [f32; 2],
    pub pointSizeGranularity: f32,
    pub lineWidthGranularity: f32,
    pub strictLines: Bool32,
    pub standardSampleLocations: Bool32,
    pub optimalBufferCopyOffsetAlignment: DeviceSize,
    pub optimalBufferCopyRowPitchAlignment: DeviceSize,
    pub nonCoherentAtomSize: DeviceSize,
}

#[repr(C)]
pub struct PhysicalDeviceSparseProperties {
    pub residencyStandard2DBlockShape: Bool32,
    pub residencyStandard2DMultisampleBlockShape: Bool32,
    pub residencyStandard3DBlockShape: Bool32,
    pub residencyAlignedMipSize: Bool32,
    pub residencyNonResidentStrict: Bool32,
}

#[repr(C)]
pub struct PhysicalDeviceProperties {
    pub apiVersion: u32,
    pub driverVersion: u32,
    pub vendorID: u32,
    pub deviceID: u32,
    pub deviceType: PhysicalDeviceType,
    pub deviceName: [c_char; MAX_PHYSICAL_DEVICE_NAME_SIZE as usize],
    pub pipelineCacheUUID: [u8; UUID_SIZE as usize],
    pub limits: PhysicalDeviceLimits,
    pub sparseProperties: PhysicalDeviceSparseProperties,
}

#[repr(C)]
pub struct QueueFamilyProperties {
    pub queueFlags: QueueFlags,
    pub queueCount: u32,
    pub timestampValidBits: u32,
    pub minImageTransferGranularity: Extent3D,
}

#[repr(C)]
pub struct MemoryType {
    pub propertyFlags: MemoryPropertyFlags,
    pub heapIndex: u32,
}

#[repr(C)]
pub struct MemoryHeap {
    pub size: DeviceSize,
    pub flags: MemoryHeapFlags,
}

#[repr(C)]
pub struct PhysicalDeviceMemoryProperties {
    pub memoryTypeCount: u32,
    pub memoryTypes: [MemoryType; MAX_MEMORY_TYPES as usize],
    pub memoryHeapCount: u32,
    pub memoryHeaps: [MemoryHeap; MAX_MEMORY_HEAPS as usize],
}

#[repr(C)]
pub struct DeviceQueueCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: DeviceQueueCreateFlags,
    pub queueFamilyIndex: u32,
    pub queueCount: u32,
    pub pQueuePriorities: *const f32,
}

#[repr(C)]
pub struct DeviceCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: DeviceCreateFlags,
    pub queueCreateInfoCount: u32,
    pub pQueueCreateInfos: *const DeviceQueueCreateInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
    pub pEnabledFeatures: *const PhysicalDeviceFeatures,
}

#[repr(C)]
pub struct ExtensionProperties {
    pub extensionName: [c_char; MAX_EXTENSION_NAME_SIZE as usize],
    pub specVersion: u32,
}

#[repr(C)]
pub struct LayerProperties {
    pub layerName: [c_char; MAX_EXTENSION_NAME_SIZE as usize],
    pub specVersion: u32,
    pub implementationVersion: u32,
    pub description: [c_char; MAX_DESCRIPTION_SIZE as usize],
}

#[repr(C)]
pub struct SubmitInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const Semaphore,
    pub pWaitDstStageMask: *const PipelineStageFlags,
    pub commandBufferCount: u32,
    pub pCommandBuffers: *const CommandBuffer,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const Semaphore,
}

#[repr(C)]
pub struct MemoryAllocateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub allocationSize: DeviceSize,
    pub memoryTypeIndex: u32,
}

#[repr(C)]
pub struct MappedMemoryRange {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub memory: DeviceMemory,
    pub offset: DeviceSize,
    pub size: DeviceSize,
}

#[repr(C)]
pub struct MemoryRequirements {
    pub size: DeviceSize,
    pub alignment: DeviceSize,
    pub memoryTypeBits: u32,
}

#[repr(C)]
pub struct SparseImageFormatProperties {
    pub aspectMask: ImageAspectFlags,
    pub imageGranularity: Extent3D,
    pub flags: SparseImageFormatFlags,
}

#[repr(C)]
pub struct SparseImageMemoryRequirements {
    pub formatProperties: SparseImageFormatProperties,
    pub imageMipTailFirstLod: u32,
    pub imageMipTailSize: DeviceSize,
    pub imageMipTailOffset: DeviceSize,
    pub imageMipTailStride: DeviceSize,
}

#[repr(C)]
pub struct SparseMemoryBind {
    pub resourceOffset: DeviceSize,
    pub size: DeviceSize,
    pub memory: DeviceMemory,
    pub memoryOffset: DeviceSize,
    pub flags: SparseMemoryBindFlags,
}

#[repr(C)]
pub struct SparseBufferMemoryBindInfo {
    pub buffer: Buffer,
    pub bindCount: u32,
    pub pBinds: *const SparseMemoryBind,
}

#[repr(C)]
pub struct SparseImageOpaqueMemoryBindInfo {
    pub image: Image,
    pub bindCount: u32,
    pub pBinds: *const SparseMemoryBind,
}

#[repr(C)]
pub struct ImageSubresource {
    pub aspectMask: ImageAspectFlags,
    pub mipLevel: u32,
    pub arrayLayer: u32,
}

#[repr(C)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[repr(C)]
pub struct SparseImageMemoryBind {
    pub subresource: ImageSubresource,
    pub offset: Offset3D,
    pub extent: Extent3D,
    pub memory: DeviceMemory,
    pub memoryOffset: DeviceSize,
    pub flags: SparseMemoryBindFlags,
}

#[repr(C)]
pub struct SparseImageMemoryBindInfo {
    pub image: Image,
    pub bindCount: u32,
    pub pBinds: *const SparseImageMemoryBind,
}

#[repr(C)]
pub struct BindSparseInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const Semaphore,
    pub bufferBindCount: u32,
    pub pBufferBinds: *const SparseBufferMemoryBindInfo,
    pub imageOpaqueBindCount: u32,
    pub pImageOpaqueBinds: *const SparseImageOpaqueMemoryBindInfo,
    pub imageBindCount: u32,
    pub pImageBinds: *const SparseImageMemoryBindInfo,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const Semaphore,
}

#[repr(C)]
pub struct FenceCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: FenceCreateFlags,
}

#[repr(C)]
pub struct SemaphoreCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: SemaphoreCreateFlags,
}

#[repr(C)]
pub struct EventCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: EventCreateFlags,
}

#[repr(C)]
pub struct QueryPoolCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: QueryPoolCreateFlags,
    pub queryType: QueryType,
    pub queryCount: u32,
    pub pipelineStatistics: QueryPipelineStatisticFlags,
}

#[repr(C)]
pub struct BufferCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: BufferCreateFlags,
    pub size: DeviceSize,
    pub usage: BufferUsageFlags,
    pub sharingMode: SharingMode,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
}

#[repr(C)]
pub struct BufferViewCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: BufferViewCreateFlags,
    pub buffer: Buffer,
    pub format: Format,
    pub offset: DeviceSize,
    pub range: DeviceSize,
}

#[repr(C)]
pub struct ImageCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: ImageCreateFlags,
    pub imageType: ImageType,
    pub format: Format,
    pub extent: Extent3D,
    pub mipLevels: u32,
    pub arrayLayers: u32,
    pub samples: SampleCountFlagBits,
    pub tiling: ImageTiling,
    pub usage: ImageUsageFlags,
    pub sharingMode: SharingMode,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
    pub initialLayout: ImageLayout,
}

#[repr(C)]
pub struct SubresourceLayout {
    pub offset: DeviceSize,
    pub size: DeviceSize,
    pub rowPitch: DeviceSize,
    pub arrayPitch: DeviceSize,
    pub depthPitch: DeviceSize,
}

#[repr(C)]
pub struct ComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[repr(C)]
pub struct ImageSubresourceRange {
    pub aspectMask: ImageAspectFlags,
    pub baseMipLevel: u32,
    pub levelCount: u32,
    pub baseArrayLayer: u32,
    pub layerCount: u32,
}

#[repr(C)]
pub struct ImageViewCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: ImageViewCreateFlags,
    pub image: Image,
    pub viewType: ImageViewType,
    pub format: Format,
    pub components: ComponentMapping,
    pub subresourceRange: ImageSubresourceRange,
}

#[repr(C)]
pub struct ShaderModuleCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: ShaderModuleCreateFlags,
    pub codeSize: usize,
    pub pCode: *const u32,
}

#[repr(C)]
pub struct PipelineCacheCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineCacheCreateFlags,
    pub initialDataSize: usize,
    pub pInitialData: *const c_void,
}

#[repr(C)]
pub struct SpecializationMapEntry {
    pub constantID: u32,
    pub offset: u32,
    pub size: usize,
}

#[repr(C)]
pub struct SpecializationInfo {
    pub mapEntryCount: u32,
    pub pMapEntries: *const SpecializationMapEntry,
    pub dataSize: usize,
    pub pData: *const c_void,
}

#[repr(C)]
pub struct PipelineShaderStageCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineShaderStageCreateFlags,
    pub stage: ShaderStageFlagBits,
    pub module: ShaderModule,
    pub pName: *const c_char,
    pub pSpecializationInfo: *const SpecializationInfo,
}

#[repr(C)]
pub struct VertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub inputRate: VertexInputRate,
}

#[repr(C)]
pub struct VertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

#[repr(C)]
pub struct PipelineVertexInputStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineVertexInputStateCreateFlags,
    pub vertexBindingDescriptionCount: u32,
    pub pVertexBindingDescriptions: *const VertexInputBindingDescription,
    pub vertexAttributeDescriptionCount: u32,
    pub pVertexAttributeDescriptions: *const VertexInputAttributeDescription,
}

#[repr(C)]
pub struct PipelineInputAssemblyStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineInputAssemblyStateCreateFlags,
    pub topology: PrimitiveTopology,
    pub primitiveRestartEnable: Bool32,
}

#[repr(C)]
pub struct PipelineTessellationStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineTessellationStateCreateFlags,
    pub patchControlPoints: u32,
}

#[repr(C)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub minDepth: f32,
    pub maxDepth: f32,
}

#[repr(C)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

#[repr(C)]
pub struct PipelineViewportStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineViewportStateCreateFlags,
    pub viewportCount: u32,
    pub pViewports: *const Viewport,
    pub scissorCount: u32,
    pub pScissors: *const Rect2D,
}

#[repr(C)]
pub struct PipelineRasterizationStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineRasterizationStateCreateFlags,
    pub depthClampEnable: Bool32,
    pub rasterizerDiscardEnable: Bool32,
    pub polygonMode: PolygonMode,
    pub cullMode: CullModeFlags,
    pub frontFace: FrontFace,
    pub depthBiasEnable: Bool32,
    pub depthBiasConstantFactor: f32,
    pub depthBiasClamp: f32,
    pub depthBiasSlopeFactor: f32,
    pub lineWidth: f32,
}

#[repr(C)]
pub struct PipelineMultisampleStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineMultisampleStateCreateFlags,
    pub rasterizationSamples: SampleCountFlagBits,
    pub sampleShadingEnable: Bool32,
    pub minSampleShading: f32,
    pub pSampleMask: *const SampleMask,
    pub alphaToCoverageEnable: Bool32,
    pub alphaToOneEnable: Bool32,
}

#[repr(C)]
pub struct StencilOpState {
    pub failOp: StencilOp,
    pub passOp: StencilOp,
    pub depthFailOp: StencilOp,
    pub compareOp: CompareOp,
    pub compareMask: u32,
    pub writeMask: u32,
    pub reference: u32,
}

#[repr(C)]
pub struct PipelineDepthStencilStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineDepthStencilStateCreateFlags,
    pub depthTestEnable: Bool32,
    pub depthWriteEnable: Bool32,
    pub depthCompareOp: CompareOp,
    pub depthBoundsTestEnable: Bool32,
    pub stencilTestEnable: Bool32,
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub minDepthBounds: f32,
    pub maxDepthBounds: f32,
}

#[repr(C)]
pub struct PipelineColorBlendAttachmentState {
    pub blendEnable: Bool32,
    pub srcColorBlendFactor: BlendFactor,
    pub dstColorBlendFactor: BlendFactor,
    pub colorBlendOp: BlendOp,
    pub srcAlphaBlendFactor: BlendFactor,
    pub dstAlphaBlendFactor: BlendFactor,
    pub alphaBlendOp: BlendOp,
    pub colorWriteMask: ColorComponentFlags,
}

#[repr(C)]
pub struct PipelineColorBlendStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineColorBlendStateCreateFlags,
    pub logicOpEnable: Bool32,
    pub logicOp: LogicOp,
    pub attachmentCount: u32,
    pub pAttachments: *const PipelineColorBlendAttachmentState,
    pub blendConstants: [f32; 4],
}

#[repr(C)]
pub struct PipelineDynamicStateCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineDynamicStateCreateFlags,
    pub dynamicStateCount: u32,
    pub pDynamicStates: *const DynamicState,
}

#[repr(C)]
pub struct GraphicsPipelineCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineCreateFlags,
    pub stageCount: u32,
    pub pStages: *const PipelineShaderStageCreateInfo,
    pub pVertexInputState: *const PipelineVertexInputStateCreateInfo,
    pub pInputAssemblyState: *const PipelineInputAssemblyStateCreateInfo,
    pub pTessellationState: *const PipelineTessellationStateCreateInfo,
    pub pViewportState: *const PipelineViewportStateCreateInfo,
    pub pRasterizationState: *const PipelineRasterizationStateCreateInfo,
    pub pMultisampleState: *const PipelineMultisampleStateCreateInfo,
    pub pDepthStencilState: *const PipelineDepthStencilStateCreateInfo,
    pub pColorBlendState: *const PipelineColorBlendStateCreateInfo,
    pub pDynamicState: *const PipelineDynamicStateCreateInfo,
    pub layout: PipelineLayout,
    pub renderPass: RenderPass,
    pub subpass: u32,
    pub basePipelineHandle: Pipeline,
    pub basePipelineIndex: i32,
}

#[repr(C)]
pub struct ComputePipelineCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineCreateFlags,
    pub stage: PipelineShaderStageCreateInfo,
    pub layout: PipelineLayout,
    pub basePipelineHandle: Pipeline,
    pub basePipelineIndex: i32,
}

#[repr(C)]
pub struct PushConstantRange {
    pub stageFlags: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct PipelineLayoutCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: PipelineLayoutCreateFlags,
    pub setLayoutCount: u32,
    pub pSetLayouts: *const DescriptorSetLayout,
    pub pushConstantRangeCount: u32,
    pub pPushConstantRanges: *const PushConstantRange,
}

#[repr(C)]
pub struct SamplerCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: SamplerCreateFlags,
    pub magFilter: Filter,
    pub minFilter: Filter,
    pub mipmapMode: SamplerMipmapMode,
    pub addressModeU: SamplerAddressMode,
    pub addressModeV: SamplerAddressMode,
    pub addressModeW: SamplerAddressMode,
    pub mipLodBias: f32,
    pub anisotropyEnable: Bool32,
    pub maxAnisotropy: f32,
    pub compareEnable: Bool32,
    pub compareOp: CompareOp,
    pub minLod: f32,
    pub maxLod: f32,
    pub borderColor: BorderColor,
    pub unnormalizedCoordinates: Bool32,
}

#[repr(C)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptorType: DescriptorType,
    pub descriptorCount: u32,
    pub stageFlags: ShaderStageFlags,
    pub pImmutableSamplers: *const Sampler,
}

#[repr(C)]
pub struct DescriptorSetLayoutCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: DescriptorSetLayoutCreateFlags,
    pub bindingCount: u32,
    pub pBindings: *const DescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct DescriptorPoolSize {
    pub ty: DescriptorType,
    pub descriptorCount: u32,
}

#[repr(C)]
pub struct DescriptorPoolCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: DescriptorPoolCreateFlags,
    pub maxSets: u32,
    pub poolSizeCount: u32,
    pub pPoolSizes: *const DescriptorPoolSize,
}

#[repr(C)]
pub struct DescriptorSetAllocateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub descriptorPool: DescriptorPool,
    pub descriptorSetCount: u32,
    pub pSetLayouts: *const DescriptorSetLayout,
}

#[repr(C)]
pub struct DescriptorImageInfo {
    pub sampler: Sampler,
    pub imageView: ImageView,
    pub imageLayout: ImageLayout,
}

#[repr(C)]
pub struct DescriptorBufferInfo {
    pub buffer: Buffer,
    pub offset: DeviceSize,
    pub range: DeviceSize,
}

#[repr(C)]
pub struct WriteDescriptorSet {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub dstSet: DescriptorSet,
    pub dstBinding: u32,
    pub dstArrayElement: u32,
    pub descriptorCount: u32,
    pub descriptorType: DescriptorType,
    pub pImageInfo: *const DescriptorImageInfo,
    pub pBufferInfo: *const DescriptorBufferInfo,
    pub pTexelBufferView: *const BufferView,
}

#[repr(C)]
pub struct CopyDescriptorSet {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub srcSet: DescriptorSet,
    pub srcBinding: u32,
    pub srcArrayElement: u32,
    pub dstSet: DescriptorSet,
    pub dstBinding: u32,
    pub dstArrayElement: u32,
    pub descriptorCount: u32,
}

#[repr(C)]
pub struct FramebufferCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: FramebufferCreateFlags,
    pub renderPass: RenderPass,
    pub attachmentCount: u32,
    pub pAttachments: *const ImageView,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

#[repr(C)]
pub struct AttachmentDescription {
    pub flags: AttachmentDescriptionFlags,
    pub format: Format,
    pub samples: SampleCountFlagBits,
    pub loadOp: AttachmentLoadOp,
    pub storeOp: AttachmentStoreOp,
    pub stencilLoadOp: AttachmentLoadOp,
    pub stencilStoreOp: AttachmentStoreOp,
    pub initialLayout: ImageLayout,
    pub finalLayout: ImageLayout,
}

#[repr(C)]
pub struct AttachmentReference {
    pub attachment: u32,
    pub layout: ImageLayout,
}

#[repr(C)]
pub struct SubpassDescription {
    pub flags: SubpassDescriptionFlags,
    pub pipelineBindPoint: PipelineBindPoint,
    pub inputAttachmentCount: u32,
    pub pInputAttachments: *const AttachmentReference,
    pub colorAttachmentCount: u32,
    pub pColorAttachments: *const AttachmentReference,
    pub pResolveAttachments: *const AttachmentReference,
    pub pDepthStencilAttachment: *const AttachmentReference,
    pub preserveAttachmentCount: u32,
    pub pPreserveAttachments: *const u32,
}

#[repr(C)]
pub struct SubpassDependency {
    pub srcSubpass: u32,
    pub dstSubpass: u32,
    pub srcStageMask: PipelineStageFlags,
    pub dstStageMask: PipelineStageFlags,
    pub srcAccessMask: AccessFlags,
    pub dstAccessMask: AccessFlags,
    pub dependencyFlags: DependencyFlags,
}

#[repr(C)]
pub struct RenderPassCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: RenderPassCreateFlags,
    pub attachmentCount: u32,
    pub pAttachments: *const AttachmentDescription,
    pub subpassCount: u32,
    pub pSubpasses: *const SubpassDescription,
    pub dependencyCount: u32,
    pub pDependencies: *const SubpassDependency,
}

#[repr(C)]
pub struct CommandPoolCreateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: CommandPoolCreateFlags,
    pub queueFamilyIndex: u32,
}

#[repr(C)]
pub struct CommandBufferAllocateInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub commandPool: CommandPool,
    pub level: CommandBufferLevel,
    pub commandBufferCount: u32,
}

#[repr(C)]
pub struct CommandBufferInheritanceInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub renderPass: RenderPass,
    pub subpass: u32,
    pub framebuffer: Framebuffer,
    pub occlusionQueryEnable: Bool32,
    pub queryFlags: QueryControlFlags,
    pub pipelineStatistics: QueryPipelineStatisticFlags,
}

#[repr(C)]
pub struct CommandBufferBeginInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: CommandBufferUsageFlags,
    pub pInheritanceInfo: *const CommandBufferInheritanceInfo,
}

#[repr(C)]
pub struct BufferCopy {
    pub srcOffset: DeviceSize,
    pub dstOffset: DeviceSize,
    pub size: DeviceSize,
}

#[repr(C)]
pub struct ImageSubresourceLayers {
    pub aspectMask: ImageAspectFlags,
    pub mipLevel: u32,
    pub baseArrayLayer: u32,
    pub layerCount: u32,
}

#[repr(C)]
pub struct ImageCopy {
    pub srcSubresource: ImageSubresourceLayers,
    pub srcOffset: Offset3D,
    pub dstSubresource: ImageSubresourceLayers,
    pub dstOffset: Offset3D,
    pub extent: Extent3D,
}

#[repr(C)]
pub struct ImageBlit {
    pub srcSubresource: ImageSubresourceLayers,
    pub srcOffsets: [Offset3D; 2],
    pub dstSubresource: ImageSubresourceLayers,
    pub dstOffsets: [Offset3D; 2],
}

#[repr(C)]
pub struct BufferImageCopy {
    pub bufferOffset: DeviceSize,
    pub bufferRowLength: u32,
    pub bufferImageHeight: u32,
    pub imageSubresource: ImageSubresourceLayers,
    pub imageOffset: Offset3D,
    pub imageExtent: Extent3D,
}

#[repr(C)]
pub struct ClearColorValue([u32; 4]);

impl ClearColorValue {
    #[inline] pub fn as_float32(&self) -> &[f32; 4] { unsafe { mem::transmute(&self.0) } }
    #[inline] pub fn as_int32(&self) -> &[i32; 4] { unsafe { mem::transmute(&self.0) } }
    #[inline] pub fn as_uint32(&self) -> &[u32; 4] { &self.0 }

    #[inline] pub fn float32(val: [f32; 4]) -> ClearColorValue { ClearColorValue(unsafe { mem::transmute(val) }) }
    #[inline] pub fn int32(val: [i32; 4]) -> ClearColorValue { ClearColorValue(unsafe { mem::transmute(val) }) }
    #[inline] pub fn uint32(val: [u32; 4]) -> ClearColorValue { ClearColorValue(val) }
}

#[repr(C)]
pub struct ClearDepthStencilValue {
    pub depth: f32,
    pub stencil: u32,
}

#[repr(C)]
pub struct ClearValue(ClearColorValue);

impl ClearValue {
    #[inline] pub fn as_color(&self) -> &ClearColorValue { &self.0 }
    #[inline] pub fn as_depth_stencil(&self) -> &ClearDepthStencilValue { unsafe { mem::transmute(&self.0) } }

    #[inline] pub fn color(val: ClearColorValue) -> ClearValue { ClearValue(val) }
    #[inline] pub fn depth_stencil(val: ClearDepthStencilValue) -> ClearValue { let val = (val, [0u32, 0u32]); ClearValue(unsafe { mem::transmute(val) }) }
}

#[repr(C)]
pub struct ClearAttachment {
    pub aspectMask: ImageAspectFlags,
    pub colorAttachment: u32,
    pub clearValue: ClearValue,
}

#[repr(C)]
pub struct ClearRect {
    pub rect: Rect2D,
    pub baseArrayLayer: u32,
    pub layerCount: u32,
}

#[repr(C)]
pub struct ImageResolve {
    pub srcSubresource: ImageSubresourceLayers,
    pub srcOffset: Offset3D,
    pub dstSubresource: ImageSubresourceLayers,
    pub dstOffset: Offset3D,
    pub extent: Extent3D,
}

#[repr(C)]
pub struct MemoryBarrier {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub srcAccessMask: AccessFlags,
    pub dstAccessMask: AccessFlags,
}

#[repr(C)]
pub struct BufferMemoryBarrier {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub srcAccessMask: AccessFlags,
    pub dstAccessMask: AccessFlags,
    pub srcQueueFamilyIndex: u32,
    pub dstQueueFamilyIndex: u32,
    pub buffer: Buffer,
    pub offset: DeviceSize,
    pub size: DeviceSize,
}

#[repr(C)]
pub struct ImageMemoryBarrier {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub srcAccessMask: AccessFlags,
    pub dstAccessMask: AccessFlags,
    pub oldLayout: ImageLayout,
    pub newLayout: ImageLayout,
    pub srcQueueFamilyIndex: u32,
    pub dstQueueFamilyIndex: u32,
    pub image: Image,
    pub subresourceRange: ImageSubresourceRange,
}

#[repr(C)]
pub struct RenderPassBeginInfo {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub renderPass: RenderPass,
    pub framebuffer: Framebuffer,
    pub renderArea: Rect2D,
    pub clearValueCount: u32,
    pub pClearValues: *const ClearValue,
}

#[repr(C)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
pub struct DrawIndexedIndirectCommand {
    pub indexCount: u32,
    pub instanceCount: u32,
    pub firstIndex: u32,
    pub vertexOffset: i32,
    pub firstInstance: u32,
}

#[repr(C)]
pub struct DrawIndirectCommand {
    pub vertexCount: u32,
    pub instanceCount: u32,
    pub firstVertex: u32,
    pub firstInstance: u32,
}

#[repr(C)]
pub struct SurfaceCapabilitiesKHR {
    pub minImageCount: u32,
    pub maxImageCount: u32,
    pub currentExtent: Extent2D,
    pub minImageExtent: Extent2D,
    pub maxImageExtent: Extent2D,
    pub maxImageArrayLayers: u32,
    pub supportedTransforms: SurfaceTransformFlagsKHR,
    pub currentTransform: SurfaceTransformFlagBitsKHR,
    pub supportedCompositeAlpha: CompositeAlphaFlagsKHR,
    pub supportedUsageFlags: ImageUsageFlags,
}

#[repr(C)]
pub struct SurfaceFormatKHR {
    pub format: Format,
    pub colorSpace: ColorSpaceKHR,
}

pub type SwapchainCreateFlagsKHR = Flags;

#[repr(C)]
pub struct SwapchainCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: SwapchainCreateFlagsKHR,
    pub surface: SurfaceKHR,
    pub minImageCount: u32,
    pub imageFormat: Format,
    pub imageColorSpace: ColorSpaceKHR,
    pub imageExtent: Extent2D,
    pub imageArrayLayers: u32,
    pub imageUsage: ImageUsageFlags,
    pub imageSharingMode: SharingMode,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
    pub preTransform: SurfaceTransformFlagBitsKHR,
    pub compositeAlpha: CompositeAlphaFlagBitsKHR,
    pub presentMode: PresentModeKHR,
    pub clipped: Bool32,
    pub oldSwapchain: SwapchainKHR,
}

#[repr(C)]
pub struct PresentInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const Semaphore,
    pub swapchainCount: u32,
    pub pSwapchains: *const SwapchainKHR,
    pub pImageIndices: *const u32,
    pub pResults: *mut Result,
}


#[repr(C)]
pub struct DisplayPropertiesKHR {
    pub display: DisplayKHR,
    pub displayName: *const c_char,
    pub physicalDimensions: Extent2D,
    pub physicalResolution: Extent2D,
    pub supportedTransforms: SurfaceTransformFlagsKHR,
    pub planeReorderPossible: Bool32,
    pub persistentContent: Bool32,
}

#[repr(C)]
pub struct DisplayModeParametersKHR {
    pub visibleRegion: Extent2D,
    pub refreshRate: u32,
}

#[repr(C)]
pub struct DisplayModePropertiesKHR {
    pub displayMode: DisplayModeKHR,
    pub parameters: DisplayModeParametersKHR,
}

#[repr(C)]
pub struct DisplayModeCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: DisplayModeCreateFlagsKHR,
    pub parameters: DisplayModeParametersKHR,
}

#[repr(C)]
pub struct DisplayPlaneCapabilitiesKHR {
    pub supportedAlpha: DisplayPlaneAlphaFlagsKHR,
    pub minSrcPosition: Offset2D,
    pub maxSrcPosition: Offset2D,
    pub minSrcExtent: Extent2D,
    pub maxSrcExtent: Extent2D,
    pub minDstPosition: Offset2D,
    pub maxDstPosition: Offset2D,
    pub minDstExtent: Extent2D,
    pub maxDstExtent: Extent2D,
}

#[repr(C)]
pub struct DisplayPlanePropertiesKHR {
    pub currentDisplay: DisplayKHR,
    pub currentStackIndex: u32,
}

#[repr(C)]
pub struct DisplaySurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: DisplaySurfaceCreateFlagsKHR,
    pub displayMode: DisplayModeKHR,
    pub planeIndex: u32,
    pub planeStackIndex: u32,
    pub transform: SurfaceTransformFlagBitsKHR,
    pub globalAlpha: f32,
    pub alphaMode: DisplayPlaneAlphaFlagBitsKHR,
    pub imageExtent: Extent2D,
}

#[repr(C)]
pub struct DisplayPresentInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub srcRect: Rect2D,
    pub dstRect: Rect2D,
    pub persistent: Bool32,
}


pub type XlibSurfaceCreateFlagsKHR = Flags;

#[repr(C)]
pub struct XlibSurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: XlibSurfaceCreateFlagsKHR,
    pub dpy: *mut c_void,
    pub window: *const c_void,
}

pub type XcbSurfaceCreateFlagsKHR = Flags;

#[repr(C)]
pub struct XcbSurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: XcbSurfaceCreateFlagsKHR,
    pub connection: *const c_void,
    pub window: *const c_void,
}


pub type WaylandSurfaceCreateFlagsKHR = Flags;

#[repr(C)]
pub struct WaylandSurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: WaylandSurfaceCreateFlagsKHR,
    pub display: *mut c_void,
    pub surface: *mut c_void,
}


pub type MirSurfaceCreateFlagsKHR = Flags;

#[repr(C)]
pub struct MirSurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: MirSurfaceCreateFlagsKHR,
    pub connection: *mut c_void,
    pub mirSurface: *mut c_void,
}

pub type AndroidSurfaceCreateFlagsKHR = Flags;

#[repr(C)]
pub struct AndroidSurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: AndroidSurfaceCreateFlagsKHR,
    pub window: *mut c_void,
}


pub type Win32SurfaceCreateFlagsKHR = Flags;

#[repr(C)]
pub struct Win32SurfaceCreateInfoKHR {
    pub sType: StructureType,
    pub pNext: *const c_void,
    pub flags: Win32SurfaceCreateFlagsKHR,
    pub hinstance: *mut c_void,
    pub hwnd: *mut c_void,
}

