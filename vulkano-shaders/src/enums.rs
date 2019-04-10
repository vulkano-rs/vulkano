// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::parse::ParseError;

macro_rules! enumeration {
    ($(typedef enum $unused:ident { $($elem:ident = $value:expr,)+ } $name:ident;)+) => (
        $(
            #[derive(Debug, Clone, PartialEq)]
            pub enum $name {
                $($elem),+
            }

            impl $name {
                pub fn from_num(num: u32) -> Result<$name, ParseError> {
                    match num {
                        $(
                            $value => Ok($name::$elem),
                        )+
                        _ => Err(ParseError::UnknownConstant(stringify!($name), num)),
                    }
                }
            }
        )+
    )
}

// The code below is a copy-paste from `spirv-2.h`, with the `Spv` prefixes removed.

enumeration! {
    typedef enum SourceLanguage_ {
        SourceLanguageUnknown = 0,
        SourceLanguageESSL = 1,
        SourceLanguageGLSL = 2,
        SourceLanguageOpenCL_C = 3,
        SourceLanguageOpenCL_CPP = 4,
    } SourceLanguage;

    typedef enum ExecutionModel_ {
        ExecutionModelVertex = 0,
        ExecutionModelTessellationControl = 1,
        ExecutionModelTessellationEvaluation = 2,
        ExecutionModelGeometry = 3,
        ExecutionModelFragment = 4,
        ExecutionModelGLCompute = 5,
        ExecutionModelKernel = 6,
    } ExecutionModel;

    typedef enum AddressingModel_ {
        AddressingModelLogical = 0,
        AddressingModelPhysical32 = 1,
        AddressingModelPhysical64 = 2,
    } AddressingModel;

    typedef enum MemoryModel_ {
        MemoryModelSimple = 0,
        MemoryModelGLSL450 = 1,
        MemoryModelOpenCL = 2,
    } MemoryModel;

    typedef enum ExecutionMode_ {
        ExecutionModeInvocations = 0,
        ExecutionModeSpacingEqual = 1,
        ExecutionModeSpacingFractionalEven = 2,
        ExecutionModeSpacingFractionalOdd = 3,
        ExecutionModeVertexOrderCw = 4,
        ExecutionModeVertexOrderCcw = 5,
        ExecutionModePixelCenterInteger = 6,
        ExecutionModeOriginUpperLeft = 7,
        ExecutionModeOriginLowerLeft = 8,
        ExecutionModeEarlyFragmentTests = 9,
        ExecutionModePointMode = 10,
        ExecutionModeXfb = 11,
        ExecutionModeDepthReplacing = 12,
        ExecutionModeDepthGreater = 14,
        ExecutionModeDepthLess = 15,
        ExecutionModeDepthUnchanged = 16,
        ExecutionModeLocalSize = 17,
        ExecutionModeLocalSizeHint = 18,
        ExecutionModeInputPoints = 19,
        ExecutionModeInputLines = 20,
        ExecutionModeInputLinesAdjacency = 21,
        ExecutionModeTriangles = 22,
        ExecutionModeInputTrianglesAdjacency = 23,
        ExecutionModeQuads = 24,
        ExecutionModeIsolines = 25,
        ExecutionModeOutputVertices = 26,
        ExecutionModeOutputPoints = 27,
        ExecutionModeOutputLineStrip = 28,
        ExecutionModeOutputTriangleStrip = 29,
        ExecutionModeVecTypeHint = 30,
        ExecutionModeContractionOff = 31,
    } ExecutionMode;

    typedef enum StorageClass_ {
        StorageClassUniformConstant = 0,
        StorageClassInput = 1,
        StorageClassUniform = 2,
        StorageClassOutput = 3,
        StorageClassWorkgroup = 4,
        StorageClassCrossWorkgroup = 5,
        StorageClassPrivate = 6,
        StorageClassFunction = 7,
        StorageClassGeneric = 8,
        StorageClassPushConstant = 9,
        StorageClassAtomicCounter = 10,
        StorageClassImage = 11,
        StorageClassStorageBuffer = 12,
    } StorageClass;

    typedef enum Dim_ {
        Dim1D = 0,
        Dim2D = 1,
        Dim3D = 2,
        DimCube = 3,
        DimRect = 4,
        DimBuffer = 5,
        DimSubpassData = 6,
    } Dim;

    typedef enum SamplerAddressingMode_ {
        SamplerAddressingModeNone = 0,
        SamplerAddressingModeClampToEdge = 1,
        SamplerAddressingModeClamp = 2,
        SamplerAddressingModeRepeat = 3,
        SamplerAddressingModeRepeatMirrored = 4,
    } SamplerAddressingMode;

    typedef enum SamplerFilterMode_ {
        SamplerFilterModeNearest = 0,
        SamplerFilterModeLinear = 1,
    } SamplerFilterMode;

    typedef enum ImageFormat_ {
        ImageFormatUnknown = 0,
        ImageFormatRgba32f = 1,
        ImageFormatRgba16f = 2,
        ImageFormatR32f = 3,
        ImageFormatRgba8 = 4,
        ImageFormatRgba8Snorm = 5,
        ImageFormatRg32f = 6,
        ImageFormatRg16f = 7,
        ImageFormatR11fG11fB10f = 8,
        ImageFormatR16f = 9,
        ImageFormatRgba16 = 10,
        ImageFormatRgb10A2 = 11,
        ImageFormatRg16 = 12,
        ImageFormatRg8 = 13,
        ImageFormatR16 = 14,
        ImageFormatR8 = 15,
        ImageFormatRgba16Snorm = 16,
        ImageFormatRg16Snorm = 17,
        ImageFormatRg8Snorm = 18,
        ImageFormatR16Snorm = 19,
        ImageFormatR8Snorm = 20,
        ImageFormatRgba32i = 21,
        ImageFormatRgba16i = 22,
        ImageFormatRgba8i = 23,
        ImageFormatR32i = 24,
        ImageFormatRg32i = 25,
        ImageFormatRg16i = 26,
        ImageFormatRg8i = 27,
        ImageFormatR16i = 28,
        ImageFormatR8i = 29,
        ImageFormatRgba32ui = 30,
        ImageFormatRgba16ui = 31,
        ImageFormatRgba8ui = 32,
        ImageFormatR32ui = 33,
        ImageFormatRgb10a2ui = 34,
        ImageFormatRg32ui = 35,
        ImageFormatRg16ui = 36,
        ImageFormatRg8ui = 37,
        ImageFormatR16ui = 38,
        ImageFormatR8ui = 39,
    } ImageFormat;

    typedef enum ImageChannelOrder_ {
        ImageChannelOrderR = 0,
        ImageChannelOrderA = 1,
        ImageChannelOrderRG = 2,
        ImageChannelOrderRA = 3,
        ImageChannelOrderRGB = 4,
        ImageChannelOrderRGBA = 5,
        ImageChannelOrderBGRA = 6,
        ImageChannelOrderARGB = 7,
        ImageChannelOrderIntensity = 8,
        ImageChannelOrderLuminance = 9,
        ImageChannelOrderRx = 10,
        ImageChannelOrderRGx = 11,
        ImageChannelOrderRGBx = 12,
        ImageChannelOrderDepth = 13,
        ImageChannelOrderDepthStencil = 14,
        ImageChannelOrdersRGB = 15,
        ImageChannelOrdersRGBx = 16,
        ImageChannelOrdersRGBA = 17,
        ImageChannelOrdersBGRA = 18,
    } ImageChannelOrder;

    typedef enum ImageChannelDataType_ {
        ImageChannelDataTypeSnormInt8 = 0,
        ImageChannelDataTypeSnormInt16 = 1,
        ImageChannelDataTypeUnormInt8 = 2,
        ImageChannelDataTypeUnormInt16 = 3,
        ImageChannelDataTypeUnormShort565 = 4,
        ImageChannelDataTypeUnormShort555 = 5,
        ImageChannelDataTypeUnormInt101010 = 6,
        ImageChannelDataTypeSignedInt8 = 7,
        ImageChannelDataTypeSignedInt16 = 8,
        ImageChannelDataTypeSignedInt32 = 9,
        ImageChannelDataTypeUnsignedInt8 = 10,
        ImageChannelDataTypeUnsignedInt16 = 11,
        ImageChannelDataTypeUnsignedInt32 = 12,
        ImageChannelDataTypeHalfFloat = 13,
        ImageChannelDataTypeFloat = 14,
        ImageChannelDataTypeUnormInt24 = 15,
        ImageChannelDataTypeUnormInt101010_2 = 16,
    } ImageChannelDataType;

    typedef enum ImageOperandsShift_ {
        ImageOperandsBiasShift = 0,
        ImageOperandsLodShift = 1,
        ImageOperandsGradShift = 2,
        ImageOperandsConstOffsetShift = 3,
        ImageOperandsOffsetShift = 4,
        ImageOperandsConstOffsetsShift = 5,
        ImageOperandsSampleShift = 6,
        ImageOperandsMinLodShift = 7,
    } ImageOperandsShift;

    typedef enum ImageOperandsMask_ {
        ImageOperandsMaskNone = 0,
        ImageOperandsBiasMask = 0x00000001,
        ImageOperandsLodMask = 0x00000002,
        ImageOperandsGradMask = 0x00000004,
        ImageOperandsConstOffsetMask = 0x00000008,
        ImageOperandsOffsetMask = 0x00000010,
        ImageOperandsConstOffsetsMask = 0x00000020,
        ImageOperandsSampleMask = 0x00000040,
        ImageOperandsMinLodMask = 0x00000080,
    } ImageOperandsMask;

    typedef enum FPFastMathModeShift_ {
        FPFastMathModeNotNaNShift = 0,
        FPFastMathModeNotInfShift = 1,
        FPFastMathModeNSZShift = 2,
        FPFastMathModeAllowRecipShift = 3,
        FPFastMathModeFastShift = 4,
    } FPFastMathModeShift;

    typedef enum FPFastMathModeMask_ {
        FPFastMathModeMaskNone = 0,
        FPFastMathModeNotNaNMask = 0x00000001,
        FPFastMathModeNotInfMask = 0x00000002,
        FPFastMathModeNSZMask = 0x00000004,
        FPFastMathModeAllowRecipMask = 0x00000008,
        FPFastMathModeFastMask = 0x00000010,
    } FPFastMathModeMask;

    typedef enum FPRoundingMode_ {
        FPRoundingModeRTE = 0,
        FPRoundingModeRTZ = 1,
        FPRoundingModeRTP = 2,
        FPRoundingModeRTN = 3,
    } FPRoundingMode;

    typedef enum LinkageType_ {
        LinkageTypeExport = 0,
        LinkageTypeImport = 1,
    } LinkageType;

    typedef enum AccessQualifier_ {
        AccessQualifierReadOnly = 0,
        AccessQualifierWriteOnly = 1,
        AccessQualifierReadWrite = 2,
    } AccessQualifier;

    typedef enum FunctionParameterAttribute_ {
        FunctionParameterAttributeZext = 0,
        FunctionParameterAttributeSext = 1,
        FunctionParameterAttributeByVal = 2,
        FunctionParameterAttributeSret = 3,
        FunctionParameterAttributeNoAlias = 4,
        FunctionParameterAttributeNoCapture = 5,
        FunctionParameterAttributeNoWrite = 6,
        FunctionParameterAttributeNoReadWrite = 7,
    } FunctionParameterAttribute;

    typedef enum Decoration_ {
        DecorationRelaxedPrecision = 0,
        DecorationSpecId = 1,
        DecorationBlock = 2,
        DecorationBufferBlock = 3,
        DecorationRowMajor = 4,
        DecorationColMajor = 5,
        DecorationArrayStride = 6,
        DecorationMatrixStride = 7,
        DecorationGLSLShared = 8,
        DecorationGLSLPacked = 9,
        DecorationCPacked = 10,
        DecorationBuiltIn = 11,
        DecorationNoPerspective = 13,
        DecorationFlat = 14,
        DecorationPatch = 15,
        DecorationCentroid = 16,
        DecorationSample = 17,
        DecorationInvariant = 18,
        DecorationRestrict = 19,
        DecorationAliased = 20,
        DecorationVolatile = 21,
        DecorationConstant = 22,
        DecorationCoherent = 23,
        DecorationNonWritable = 24,
        DecorationNonReadable = 25,
        DecorationUniform = 26,
        DecorationSaturatedConversion = 28,
        DecorationStream = 29,
        DecorationLocation = 30,
        DecorationComponent = 31,
        DecorationIndex = 32,
        DecorationBinding = 33,
        DecorationDescriptorSet = 34,
        DecorationOffset = 35,
        DecorationXfbBuffer = 36,
        DecorationXfbStride = 37,
        DecorationFuncParamAttr = 38,
        DecorationFPRoundingMode = 39,
        DecorationFPFastMathMode = 40,
        DecorationLinkageAttributes = 41,
        DecorationNoContraction = 42,
        DecorationInputAttachmentIndex = 43,
        DecorationAlignment = 44,
    } Decoration;

    typedef enum BuiltIn_ {
        BuiltInPosition = 0,
        BuiltInPointSize = 1,
        BuiltInClipDistance = 3,
        BuiltInCullDistance = 4,
        BuiltInVertexId = 5,
        BuiltInInstanceId = 6,
        BuiltInPrimitiveId = 7,
        BuiltInInvocationId = 8,
        BuiltInLayer = 9,
        BuiltInViewportIndex = 10,
        BuiltInTessLevelOuter = 11,
        BuiltInTessLevelInner = 12,
        BuiltInTessCoord = 13,
        BuiltInPatchVertices = 14,
        BuiltInFragCoord = 15,
        BuiltInPointCoord = 16,
        BuiltInFrontFacing = 17,
        BuiltInSampleId = 18,
        BuiltInSamplePosition = 19,
        BuiltInSampleMask = 20,
        BuiltInFragDepth = 22,
        BuiltInHelperInvocation = 23,
        BuiltInNumWorkgroups = 24,
        BuiltInWorkgroupSize = 25,
        BuiltInWorkgroupId = 26,
        BuiltInLocalInvocationId = 27,
        BuiltInGlobalInvocationId = 28,
        BuiltInLocalInvocationIndex = 29,
        BuiltInWorkDim = 30,
        BuiltInGlobalSize = 31,
        BuiltInEnqueuedWorkgroupSize = 32,
        BuiltInGlobalOffset = 33,
        BuiltInGlobalLinearId = 34,
        BuiltInSubgroupSize = 36,
        BuiltInSubgroupMaxSize = 37,
        BuiltInNumSubgroups = 38,
        BuiltInNumEnqueuedSubgroups = 39,
        BuiltInSubgroupId = 40,
        BuiltInSubgroupLocalInvocationId = 41,
        BuiltInVertexIndex = 42,
        BuiltInInstanceIndex = 43,
    } BuiltIn;

    typedef enum SelectionControlShift_ {
        SelectionControlFlattenShift = 0,
        SelectionControlDontFlattenShift = 1,
    } SelectionControlShift;

    typedef enum SelectionControlMask_ {
        SelectionControlMaskNone = 0,
        SelectionControlFlattenMask = 0x00000001,
        SelectionControlDontFlattenMask = 0x00000002,
    } SelectionControlMask;

    typedef enum LoopControlShift_ {
        LoopControlUnrollShift = 0,
        LoopControlDontUnrollShift = 1,
    } LoopControlShift;

    typedef enum LoopControlMask_ {
        LoopControlMaskNone = 0,
        LoopControlUnrollMask = 0x00000001,
        LoopControlDontUnrollMask = 0x00000002,
    } LoopControlMask;

    typedef enum FunctionControlShift_ {
        FunctionControlInlineShift = 0,
        FunctionControlDontInlineShift = 1,
        FunctionControlPureShift = 2,
        FunctionControlConstShift = 3,
    } FunctionControlShift;

    typedef enum FunctionControlMask_ {
        FunctionControlMaskNone = 0,
        FunctionControlInlineMask = 0x00000001,
        FunctionControlDontInlineMask = 0x00000002,
        FunctionControlPureMask = 0x00000004,
        FunctionControlConstMask = 0x00000008,
    } FunctionControlMask;

    typedef enum MemorySemanticsShift_ {
        MemorySemanticsAcquireShift = 1,
        MemorySemanticsReleaseShift = 2,
        MemorySemanticsAcquireReleaseShift = 3,
        MemorySemanticsSequentiallyConsistentShift = 4,
        MemorySemanticsUniformMemoryShift = 6,
        MemorySemanticsSubgroupMemoryShift = 7,
        MemorySemanticsWorkgroupMemoryShift = 8,
        MemorySemanticsCrossWorkgroupMemoryShift = 9,
        MemorySemanticsAtomicCounterMemoryShift = 10,
        MemorySemanticsImageMemoryShift = 11,
    } MemorySemanticsShift;

    typedef enum MemorySemanticsMask_ {
        MemorySemanticsMaskNone = 0,
        MemorySemanticsAcquireMask = 0x00000002,
        MemorySemanticsReleaseMask = 0x00000004,
        MemorySemanticsAcquireReleaseMask = 0x00000008,
        MemorySemanticsSequentiallyConsistentMask = 0x00000010,
        MemorySemanticsUniformMemoryMask = 0x00000040,
        MemorySemanticsSubgroupMemoryMask = 0x00000080,
        MemorySemanticsWorkgroupMemoryMask = 0x00000100,
        MemorySemanticsCrossWorkgroupMemoryMask = 0x00000200,
        MemorySemanticsAtomicCounterMemoryMask = 0x00000400,
        MemorySemanticsImageMemoryMask = 0x00000800,
    } MemorySemanticsMask;

    typedef enum MemoryAccessShift_ {
        MemoryAccessVolatileShift = 0,
        MemoryAccessAlignedShift = 1,
        MemoryAccessNontemporalShift = 2,
    } MemoryAccessShift;

    typedef enum MemoryAccessMask_ {
        MemoryAccessMaskNone = 0,
        MemoryAccessVolatileMask = 0x00000001,
        MemoryAccessAlignedMask = 0x00000002,
        MemoryAccessNontemporalMask = 0x00000004,
    } MemoryAccessMask;

    typedef enum Scope_ {
        ScopeCrossDevice = 0,
        ScopeDevice = 1,
        ScopeWorkgroup = 2,
        ScopeSubgroup = 3,
        ScopeInvocation = 4,
    } Scope;

    typedef enum GroupOperation_ {
        GroupOperationReduce = 0,
        GroupOperationInclusiveScan = 1,
        GroupOperationExclusiveScan = 2,
    } GroupOperation;

    typedef enum KernelEnqueueFlags_ {
        KernelEnqueueFlagsNoWait = 0,
        KernelEnqueueFlagsWaitKernel = 1,
        KernelEnqueueFlagsWaitWorkGroup = 2,
    } KernelEnqueueFlags;

    typedef enum KernelProfilingInfoShift_ {
        KernelProfilingInfoCmdExecTimeShift = 0,
    } KernelProfilingInfoShift;

    typedef enum KernelProfilingInfoMask_ {
        KernelProfilingInfoMaskNone = 0,
        KernelProfilingInfoCmdExecTimeMask = 0x00000001,
    } KernelProfilingInfoMask;

    typedef enum Capability_ {
        CapabilityMatrix = 0,
        CapabilityShader = 1,
        CapabilityGeometry = 2,
        CapabilityTessellation = 3,
        CapabilityAddresses = 4,
        CapabilityLinkage = 5,
        CapabilityKernel = 6,
        CapabilityVector16 = 7,
        CapabilityFloat16Buffer = 8,
        CapabilityFloat16 = 9,
        CapabilityFloat64 = 10,
        CapabilityInt64 = 11,
        CapabilityInt64Atomics = 12,
        CapabilityImageBasic = 13,
        CapabilityImageReadWrite = 14,
        CapabilityImageMipmap = 15,
        CapabilityPipes = 17,
        CapabilityGroups = 18,
        CapabilityDeviceEnqueue = 19,
        CapabilityLiteralSampler = 20,
        CapabilityAtomicStorage = 21,
        CapabilityInt16 = 22,
        CapabilityTessellationPointSize = 23,
        CapabilityGeometryPointSize = 24,
        CapabilityImageGatherExtended = 25,
        CapabilityStorageImageMultisample = 27,
        CapabilityUniformBufferArrayDynamicIndexing = 28,
        CapabilitySampledImageArrayDynamicIndexing = 29,
        CapabilityStorageBufferArrayDynamicIndexing = 30,
        CapabilityStorageImageArrayDynamicIndexing = 31,
        CapabilityClipDistance = 32,
        CapabilityCullDistance = 33,
        CapabilityImageCubeArray = 34,
        CapabilitySampleRateShading = 35,
        CapabilityImageRect = 36,
        CapabilitySampledRect = 37,
        CapabilityGenericPointer = 38,
        CapabilityInt8 = 39,
        CapabilityInputAttachment = 40,
        CapabilitySparseResidency = 41,
        CapabilityMinLod = 42,
        CapabilitySampled1D = 43,
        CapabilityImage1D = 44,
        CapabilitySampledCubeArray = 45,
        CapabilitySampledBuffer = 46,
        CapabilityImageBuffer = 47,
        CapabilityImageMSArray = 48,
        CapabilityStorageImageExtendedFormats = 49,
        CapabilityImageQuery = 50,
        CapabilityDerivativeControl = 51,
        CapabilityInterpolationFunction = 52,
        CapabilityTransformFeedback = 53,
        CapabilityGeometryStreams = 54,
        CapabilityStorageImageReadWithoutFormat = 55,
        CapabilityStorageImageWriteWithoutFormat = 56,
        CapabilityMultiViewport = 57,
        CapabilityStorageUniformBufferBlock16 = 4433,
        CapabilityStorageUniform16 = 4434,
        CapabilityStoragePushConstant16 = 4435,
        CapabilityStorageInputOutput16 = 4436,
    } Capability;
}
