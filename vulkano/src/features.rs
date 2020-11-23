// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomPinned;
use std::pin::Pin;
use std::{mem, ptr};

use vk;

macro_rules! features_init {
  (core { $name:ident => $vk:ident }, $out:expr, $val:expr) => {
     $out.$name = $val;
  };
  (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $out:expr, $val:expr) => {
    $($out.$name = $val;)+
  };
}
macro_rules! features_superset_of {
  (core { $name:ident => $vk:ident }, $self:expr, $other:expr) => {
     ($self.$name == true || $other.$name == false)
  };
  (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $self:expr, $other:expr) => {
    $(($self.$name == true || $other.$name == false))&&+
  };
}
macro_rules! features_intersection {
  (core { $name:ident => $vk:ident }, $out:expr, $self:expr, $other:expr) => {
     $out.$name = $self.$name && $other.$name;
  };
  (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $out:expr, $self:expr, $other:expr) => {
     $($out.$name = $self.$name && $other.$name;)+
  };
}
macro_rules! features_difference {
  (core { $name:ident => $vk:ident }, $out:expr, $self:expr, $other:expr) => {
     $out.$name = $self.$name && !$other.$name;
  };
  (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $out:expr, $self:expr, $other:expr) => {
     $($out.$name = $self.$name && !$other.$name;)+
  };
}

macro_rules! from_feature_v1 {
    (core { $name:ident => $vk:ident }, $out:expr, $features:expr) => {
        $out.$name = $features.$vk != vk::FALSE;
    };
    (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $out:expr, $features:expr) => {
        // nothing.
    };
}
macro_rules! into_feature_v1 {
    (core { $name:ident => $vk:ident }, $out:expr, $self:expr) => {
        $out.$vk = if $self.$name { vk::TRUE } else { vk::FALSE };
    };
    (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $out:expr, $self:expr) => {
        // nothing.
    };
}

macro_rules! from_ext_features_match {
  (core { $name:ident => $vk:ident }, $output:expr, $base_ref:expr, $base_ptr:expr) => {
    // nothing.
  };
  (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $output:expr, $base_ref:expr, $base_ptr:expr) => {
    if $base_ref.sType == $stype {
      let ptr = $base_ptr as *const $ty;
      let r = ptr.as_ref().unwrap();
      $($output.$name = r.$vk != vk::FALSE;)+
    }
  };
}
macro_rules! into_ext_features_match {
  (core { $name:ident => $vk:ident }, $output:expr, $base_ref:expr, $base_ptr:expr) => {
    // nothing.
  };
  (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)+
    ],
  }, $this:expr, $base_ref:expr, $base_ptr:expr) => {
    if $base_ref.sType == $stype {
      let ptr = $base_ptr as *mut $ty;
      let r = ptr.as_mut().unwrap();
      $(r.$vk = if $this.$name { vk::TRUE } else { vk::FALSE };)+
    }
  };
}
macro_rules! features_ffi_init_pinned {
    (core { $name:ident => $vk:ident }, $this:expr, $prev:expr) => {
        // nothing.
    };
    (extension {
    ty: $ty:ty,
    ffi_name: $ffi_name:ident,
    sType: $stype:expr,
    fields: [
      $($name:ident => $vk:ident,)*
    ],
  }, $this:expr, $prev:expr) => {{
        $this.$ffi_name.sType = $stype;
        let next = &mut $this.$ffi_name as *mut _ as *mut Base;
        (&mut *$prev).pNext = next;
        $prev = next;
    }};
}

#[allow(non_snake_case)]
#[repr(C)]
pub(crate) struct Base {
    sType: vk::StructureType,
    pNext: *mut Base,
}

// Can't define this structure with macros :(
/// Represents all the features that are available on a physical device or enabled on
/// a logical device.
///
/// Note that the `robust_buffer_access` is guaranteed to be supported by all Vulkan
/// implementations.
///
/// # Example
///
/// ```
/// use vulkano::device::Features;
/// # let physical_device: vulkano::instance::PhysicalDevice = return;
/// let minimal_features = Features {
///     geometry_shader: true,
///     .. Features::none()
/// };
///
/// let optimal_features = vulkano::device::Features {
///     geometry_shader: true,
///     tessellation_shader: true,
///     .. Features::none()
/// };
///
/// if !physical_device.supported_features().superset_of(&minimal_features) {
///     panic!("The physical device is not good enough for this application.");
/// }
///
/// assert!(optimal_features.superset_of(&minimal_features));
/// let features_to_request = optimal_features.intersection(physical_device.supported_features());
/// ```
///
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs)]
pub struct Features {
    pub robust_buffer_access: bool,
    pub full_draw_index_uint32: bool,
    pub image_cube_array: bool,
    pub independent_blend: bool,
    pub geometry_shader: bool,
    pub tessellation_shader: bool,
    pub sample_rate_shading: bool,
    pub dual_src_blend: bool,
    pub logic_op: bool,
    pub multi_draw_indirect: bool,
    pub draw_indirect_first_instance: bool,
    pub depth_clamp: bool,
    pub depth_bias_clamp: bool,
    pub fill_mode_non_solid: bool,
    pub depth_bounds: bool,
    pub wide_lines: bool,
    pub large_points: bool,
    pub alpha_to_one: bool,
    pub multi_viewport: bool,
    pub sampler_anisotropy: bool,
    pub texture_compression_etc2: bool,
    pub texture_compression_astc_ldr: bool,
    pub texture_compression_bc: bool,
    pub occlusion_query_precise: bool,
    pub pipeline_statistics_query: bool,
    pub vertex_pipeline_stores_and_atomics: bool,
    pub fragment_stores_and_atomics: bool,
    pub shader_tessellation_and_geometry_point_size: bool,
    pub shader_image_gather_extended: bool,
    pub shader_storage_image_extended_formats: bool,
    pub shader_storage_image_multisample: bool,
    pub shader_storage_image_read_without_format: bool,
    pub shader_storage_image_write_without_format: bool,
    pub shader_uniform_buffer_array_dynamic_indexing: bool,
    pub shader_sampled_image_array_dynamic_indexing: bool,
    pub shader_storage_buffer_array_dynamic_indexing: bool,
    pub shader_storage_image_array_dynamic_indexing: bool,
    pub shader_clip_distance: bool,
    pub shader_cull_distance: bool,
    pub shader_float64: bool,
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_resource_residency: bool,
    pub shader_resource_min_lod: bool,
    pub sparse_binding: bool,
    pub sparse_residency_buffer: bool,
    pub sparse_residency_image2d: bool,
    pub sparse_residency_image3d: bool,
    pub sparse_residency2_samples: bool,
    pub sparse_residency4_samples: bool,
    pub sparse_residency8_samples: bool,
    pub sparse_residency16_samples: bool,
    pub sparse_residency_aliased: bool,
    pub variable_multisample_rate: bool,
    pub inherited_queries: bool,

    pub buffer_device_address: bool,
    pub buffer_device_address_capture_replay: bool,
    pub buffer_device_address_multi_device: bool,

    pub variable_pointers_storage_buffer: bool,
    pub variable_pointers: bool,

    pub shader_buffer_int64_atomics: bool,
    pub shader_shared_int64_atomics: bool,

    pub storage_buffer_8bit: bool,
    pub storage_uniform_8bit: bool,
    pub storage_push_constant_8bit: bool,

    pub storage_buffer_16bit: bool,
    pub storage_uniform_16bit: bool,
    pub storage_push_constant_16bit: bool,
    pub storage_input_output_16bit: bool,

    pub shader_float16: bool,
    pub shader_int8: bool,
}

pub(crate) struct FeaturesFfi {
    _pinned: PhantomPinned,
    pub(crate) main: vk::PhysicalDeviceFeatures2KHR,
    phys_dev_buf_addr: vk::PhysicalDeviceBufferAddressFeaturesEXT,
    variable_pointers: vk::PhysicalDeviceVariablePointersFeatures,
    shader_atomic_i64: vk::PhysicalDeviceShaderAtomicInt64Features,
    i8_storage: vk::PhysicalDevice8BitStorageFeatures,
    i16_storage: vk::PhysicalDevice16BitStorageFeatures,
    f16_i8: vk::PhysicalDeviceShaderFloat16Int8Features,
}

macro_rules! features {
    ($($kind:ident $args:tt,)+) => (
        impl Features {
            /// Builds a `Features` object with all values to false.
            pub fn none() -> Features {
                let mut out = Features::default();
                $(features_init!($kind $args, out, false);)+
                out
            }

            /// Builds a `Features` object with all values to true.
            ///
            /// > **Note**: This function is used for testing purposes, and is probably useless in
            /// > a real code.
            pub fn all() -> Features {
                let mut out = Features::default();
                $(features_init!($kind $args, out, true);)+
                out
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each feature of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub fn superset_of(&self, other: &Features) -> bool {
                $(features_superset_of!($kind $args, self, other))&&+
            }

            /// Builds a `Features` that is the intersection of `self` and another `Features`
            /// object.
            ///
            /// The result's field will be true if it is also true in both `self` and `other`.
            pub fn intersection(&self, other: &Features) -> Features {
                let mut out = Self::none();
                $(features_intersection!($kind $args, out, self, other);)+
                out
            }

            /// Builds a `Features` that is the difference of another `Features` object from `self`.
            ///
            /// The result's field will be true if it is true in `self` but not `other`.
            pub fn difference(&self, other: &Features) -> Features {
                let mut out = Self::none();
                $(features_difference!($kind $args, out, self, other);)+
                out
            }

            pub(crate) fn from_vulkan_features(features: vk::PhysicalDeviceFeatures) -> Features {
                let mut out = Self::none();
                $(from_feature_v1!($kind $args, out, features);)+
                out
            }

            pub(crate) fn into_vulkan_features(self) -> vk::PhysicalDeviceFeatures {
                let mut out: vk::PhysicalDeviceFeatures = unsafe { mem::zeroed() };
                $(into_feature_v1!($kind $args, out, self);)+
                out
            }

            #[inline(always)]
            pub(crate) fn into_vulkan_features_v2(&self) -> Pin<Box<FeaturesFfi>> {
                let mut features = FeaturesFfi::new();
                unsafe {
                    let mut next = FeaturesFfi::mut_base_ptr(&mut features);
                    while let Some(next_mut) = next.as_mut() {
                        match next_mut.sType {
                            vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR => {
                                let r = (next as *mut vk::PhysicalDeviceFeatures2KHR).as_mut().unwrap();
                                $(into_feature_v1!($kind $args, r.features, self);)+
                            },
                            _ => {
                                $(into_ext_features_match!($kind $args, self, next_mut, next);)+
                            },
                        }

                        next = next_mut.pNext as *mut Base;
                    }
                }
                features
            }

            #[inline(always)]
            pub(crate) fn from_vulkan_features_v2(features: &vk::PhysicalDeviceFeatures2KHR) -> Self {
                let mut output = Self::none();

                unsafe {
                    let mut next: *const Base = features as *const vk::PhysicalDeviceFeatures2KHR as _;
                    while let Some(next_ref) = next.as_ref() {
                        match next_ref.sType {
                            vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR => {
                                let r = (next as *const vk::PhysicalDeviceFeatures2KHR).as_ref().unwrap();
                                $(from_feature_v1!($kind $args, output, r.features);)+
                            },
                            _ => {
                                $(from_ext_features_match!($kind $args, output, next_ref, next);)+
                            },
                        }

                        next = next_ref.pNext as *const Base;
                    }
                }

                output
            }
        }

        impl FeaturesFfi {
          #[inline(always)]
          pub(crate) fn new() -> Pin<Box<Self>> {
            #![allow(unused_assignments)]

            let this = FeaturesFfi {
              _pinned: PhantomPinned,
              main: vk::PhysicalDeviceFeatures2KHR {
                 sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
                 pNext: ptr::null(),
                 features: unsafe { mem::zeroed() },
              },
              .. unsafe { mem::zeroed() }
            };

            let mut this = Box::pin(this);
            unsafe {
              let this = this.as_mut();
              let this = this.get_unchecked_mut();
              let mut base = &mut this.main as *mut _ as *mut Base;
              $(features_ffi_init_pinned!($kind $args, this, base);)*
            }
            this
          }
          #[inline(always)]
          pub(crate) fn base_ptr(&self) -> *const Base {
            &self.main as *const _ as *const Base
          }
          #[inline(always)]
          pub(crate) fn mut_base_ptr(this: &mut Pin<Box<Self>>) -> *mut Base {
            unsafe {
              let this = this.as_mut();
              let this = this.get_unchecked_mut();
              &mut this.main as *mut _ as *mut Base
            }
          }
        }
    )
}

features! {
    core { robust_buffer_access => robustBufferAccess },
    core { full_draw_index_uint32 => fullDrawIndexUint32 },
    core { image_cube_array => imageCubeArray },
    core { independent_blend => independentBlend },
    core { geometry_shader => geometryShader },
    core { tessellation_shader => tessellationShader },
    core { sample_rate_shading => sampleRateShading },
    core { dual_src_blend => dualSrcBlend },
    core { logic_op => logicOp },
    core { multi_draw_indirect => multiDrawIndirect },
    core { draw_indirect_first_instance => drawIndirectFirstInstance },
    core { depth_clamp => depthClamp },
    core { depth_bias_clamp => depthBiasClamp },
    core { fill_mode_non_solid => fillModeNonSolid },
    core { depth_bounds => depthBounds },
    core { wide_lines => wideLines },
    core { large_points => largePoints },
    core { alpha_to_one => alphaToOne },
    core { multi_viewport => multiViewport },
    core { sampler_anisotropy => samplerAnisotropy },
    core { texture_compression_etc2 => textureCompressionETC2 },
    core { texture_compression_astc_ldr => textureCompressionASTC_LDR },
    core { texture_compression_bc => textureCompressionBC },
    core { occlusion_query_precise => occlusionQueryPrecise },
    core { pipeline_statistics_query => pipelineStatisticsQuery },
    core { vertex_pipeline_stores_and_atomics => vertexPipelineStoresAndAtomics },
    core { fragment_stores_and_atomics => fragmentStoresAndAtomics },
    core { shader_tessellation_and_geometry_point_size => shaderTessellationAndGeometryPointSize },
    core { shader_image_gather_extended => shaderImageGatherExtended },
    core { shader_storage_image_extended_formats => shaderStorageImageExtendedFormats },
    core { shader_storage_image_multisample => shaderStorageImageMultisample },
    core { shader_storage_image_read_without_format => shaderStorageImageReadWithoutFormat },
    core { shader_storage_image_write_without_format => shaderStorageImageWriteWithoutFormat },
    core { shader_uniform_buffer_array_dynamic_indexing => shaderUniformBufferArrayDynamicIndexing },
    core { shader_sampled_image_array_dynamic_indexing => shaderSampledImageArrayDynamicIndexing },
    core { shader_storage_buffer_array_dynamic_indexing => shaderStorageBufferArrayDynamicIndexing },
    core { shader_storage_image_array_dynamic_indexing => shaderStorageImageArrayDynamicIndexing },
    core { shader_clip_distance => shaderClipDistance },
    core { shader_cull_distance => shaderCullDistance },
    core { shader_float64 => shaderFloat64 },
    core { shader_int64 => shaderInt64 },
    core { shader_int16 => shaderInt16 },
    core { shader_resource_residency => shaderResourceResidency },
    core { shader_resource_min_lod => shaderResourceMinLod },
    core { sparse_binding => sparseBinding },
    core { sparse_residency_buffer => sparseResidencyBuffer },
    core { sparse_residency_image2d => sparseResidencyImage2D },
    core { sparse_residency_image3d => sparseResidencyImage3D },
    core { sparse_residency2_samples => sparseResidency2Samples },
    core { sparse_residency4_samples => sparseResidency4Samples },
    core { sparse_residency8_samples => sparseResidency8Samples },
    core { sparse_residency16_samples => sparseResidency16Samples },
    core { sparse_residency_aliased => sparseResidencyAliased },
    core { variable_multisample_rate => variableMultisampleRate },
    core { inherited_queries => inheritedQueries },

    extension {
      ty: vk::PhysicalDeviceBufferAddressFeaturesEXT,
      ffi_name: phys_dev_buf_addr,
      sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
      fields: [
        buffer_device_address => bufferDeviceAddress,
        buffer_device_address_capture_replay => bufferDeviceAddressCaptureReplay,
        buffer_device_address_multi_device => bufferDeviceAddressMultiDevice,
      ],
    },
    extension {
      ty: vk::PhysicalDeviceVariablePointersFeatures,
      ffi_name: variable_pointers,
      sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES,
      fields: [
        variable_pointers_storage_buffer => variablePointersStorageBuffer,
        variable_pointers => variablePointers,
      ],
    },
    extension {
      ty: vk::PhysicalDeviceShaderAtomicInt64Features,
      ffi_name: shader_atomic_i64,
      sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
      fields: [
        shader_buffer_int64_atomics => shaderBufferInt64Atomics,
        shader_shared_int64_atomics => shaderSharedInt64Atomics,
      ],
    },
    extension {
      ty: vk::PhysicalDevice8BitStorageFeatures,
      ffi_name: i8_storage,
      sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
      fields: [
        storage_buffer_8bit => storageBuffer8BitAccess,
        storage_uniform_8bit => uniformAndStorageBuffer8BitAccess,
        storage_push_constant_8bit => storagePushConstant8,
      ],
    },
    extension {
      ty: vk::PhysicalDevice16BitStorageFeatures,
      ffi_name: i16_storage,
      sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
      fields: [
        storage_buffer_16bit => storageBuffer16BitAccess,
        storage_uniform_16bit => uniformAndStorageBuffer16BitAccess,
        storage_push_constant_16bit => storagePushConstant16,
        storage_input_output_16bit => storageInputOutput16,
      ],
    },
    extension {
      ty: vk::PhysicalDeviceShaderFloat16Int8Features,
      ffi_name: f16_i8,
      sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
      fields: [
        shader_float16 => shaderFloat16,
        shader_int8 => shaderInt8,
      ],
    },
}
