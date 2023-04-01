// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{extensions::RequiresOneOf, write_file, IndexMap, VkRegistryData};
use heck::ToSnakeCase;
use once_cell::sync::Lazy;
use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote};
use regex::Regex;
use vk_parse::{
    Enum, EnumSpec, Extension, ExtensionChild, Feature, Format, FormatChild, InterfaceItem,
};

pub fn write(vk_data: &VkRegistryData) {
    write_file(
        "formats.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        formats_output(&formats_members(
            &vk_data.formats,
            &vk_data.features,
            &vk_data.extensions,
        )),
    );
}

#[derive(Clone, Debug)]
struct FormatMember {
    name: Ident,
    ffi_name: Ident,
    requires: Vec<RequiresOneOf>,

    aspect_color: bool,
    aspect_depth: bool,
    aspect_stencil: bool,
    aspect_plane0: bool,
    aspect_plane1: bool,
    aspect_plane2: bool,

    block_extent: [u32; 3],
    block_size: Option<u64>,
    compatibility: Ident,
    components: [u8; 4],
    compression: Option<Ident>,
    planes: Vec<Ident>,
    texels_per_block: u8,
    type_color: Option<Ident>,
    type_depth: Option<Ident>,
    type_stencil: Option<Ident>,
    ycbcr_chroma_sampling: Option<Ident>,

    type_std_array: Option<TokenStream>,
    type_cgmath: Option<TokenStream>,
    type_nalgebra: Option<TokenStream>,
}

fn formats_output(members: &[FormatMember]) -> TokenStream {
    let enum_items = members.iter().map(|FormatMember { name, ffi_name, .. }| {
        quote! { #name = ash::vk::Format::#ffi_name.as_raw(), }
    });
    let aspects_items = members.iter().map(
        |FormatMember {
             name,
             aspect_color,
             aspect_depth,
             aspect_stencil,
             aspect_plane0,
             aspect_plane1,
             aspect_plane2,
             ..
         }| {
            let aspect_items = [
                aspect_color.then(|| quote! { crate::image::ImageAspects::COLOR }),
                aspect_depth.then(|| quote! { crate::image::ImageAspects::DEPTH }),
                aspect_stencil.then(|| quote! { crate::image::ImageAspects::STENCIL }),
                aspect_plane0.then(|| quote! { crate::image::ImageAspects::PLANE_0 }),
                aspect_plane1.then(|| quote! { crate::image::ImageAspects::PLANE_1 }),
                aspect_plane2.then(|| quote! { crate::image::ImageAspects::PLANE_2 }),
            ]
            .into_iter()
            .flatten();

            quote! {
                Self::#name => #(#aspect_items)|*,
            }
        },
    );
    let block_extent_items = members.iter().filter_map(
        |FormatMember {
             name,
             block_extent: [x, y, z],
             ..
         }| {
            if *x == 1 && *y == 1 && *z == 1 {
                None
            } else {
                let x = Literal::u32_unsuffixed(*x);
                let y = Literal::u32_unsuffixed(*y);
                let z = Literal::u32_unsuffixed(*z);
                Some(quote! { Self::#name => [#x, #y, #z], })
            }
        },
    );
    let block_size_items = members.iter().filter_map(
        |FormatMember {
             name, block_size, ..
         }| {
            block_size.as_ref().map(|size| {
                let size = Literal::u64_unsuffixed(*size);
                quote! { Self::#name => Some(#size), }
            })
        },
    );
    let compatibility_items = members.iter().map(
        |FormatMember {
             name,
             compatibility,
             ..
         }| {
            quote! {
                Self::#name => &FormatCompatibilityInner::#compatibility,
            }
        },
    );
    let components_items = members.iter().map(
        |FormatMember {
             name, components, ..
         }| {
            let components = components.iter().map(|c| Literal::u8_unsuffixed(*c));
            quote! {
                Self::#name => [#(#components),*],
            }
        },
    );
    let compression_items = members.iter().filter_map(
        |FormatMember {
             name, compression, ..
         }| {
            compression
                .as_ref()
                .map(|x| quote! { Self::#name => Some(CompressionType::#x), })
        },
    );
    let planes_items = members
        .iter()
        .filter_map(|FormatMember { name, planes, .. }| {
            if planes.is_empty() {
                None
            } else {
                Some(quote! { Self::#name => &[#(Self::#planes),*], })
            }
        });
    let texels_per_block_items = members.iter().filter_map(
        |FormatMember {
             name,
             texels_per_block,
             ..
         }| {
            (*texels_per_block != 1).then(|| {
                let texels_per_block = Literal::u8_unsuffixed(*texels_per_block);
                quote! { Self::#name => #texels_per_block, }
            })
        },
    );
    let type_color_items = members.iter().filter_map(
        |FormatMember {
             name, type_color, ..
         }| {
            type_color
                .as_ref()
                .map(|ty| quote! { Self::#name => Some(NumericType::#ty), })
        },
    );
    let type_depth_items = members.iter().filter_map(
        |FormatMember {
             name, type_depth, ..
         }| {
            type_depth
                .as_ref()
                .map(|ty| quote! { Self::#name => Some(NumericType::#ty), })
        },
    );
    let type_stencil_items = members.iter().filter_map(
        |FormatMember {
             name, type_stencil, ..
         }| {
            type_stencil
                .as_ref()
                .map(|ty| quote! { Self::#name => Some(NumericType::#ty), })
        },
    );
    let ycbcr_chroma_sampling_items = members.iter().filter_map(
        |FormatMember {
             name,
             ycbcr_chroma_sampling,
             ..
         }| {
            ycbcr_chroma_sampling
                .as_ref()
                .map(|ty| quote! { Self::#name => Some(ChromaSampling::#ty), })
        },
    );
    let try_from_items = members.iter().map(|FormatMember { name, ffi_name, .. }| {
        quote! { ash::vk::Format::#ffi_name => Ok(Self::#name), }
    });

    let type_for_format_items = members.iter().filter_map(
        |FormatMember {
             name,
             type_std_array,
             ..
         }| {
            type_std_array.as_ref().map(|ty| {
                quote! { (#name) => { #ty }; }
            })
        },
    );
    let type_for_format_cgmath_items = members.iter().filter_map(
        |FormatMember {
             name,
             type_std_array,
             type_cgmath,
             ..
         }| {
            (type_cgmath.as_ref().or(type_std_array.as_ref())).map(|ty| {
                quote! { (cgmath, #name) => { #ty }; }
            })
        },
    );
    let type_for_format_nalgebra_items = members.iter().filter_map(
        |FormatMember {
             name,
             type_std_array,
             type_nalgebra,
             ..
         }| {
            (type_nalgebra.as_ref().or(type_std_array.as_ref())).map(|ty| {
                quote! { (nalgebra, #name) => { #ty }; }
            })
        },
    );

    let validate_device_items = members.iter().map(|FormatMember { name, requires, .. }| {
        let requires_items = requires.iter().map(
            |RequiresOneOf {
                 api_version,
                 device_extensions,
                 instance_extensions,
             }| {
                let condition_items = (api_version.iter().map(|(major, minor)| {
                    let version = format_ident!("V{}_{}", major, minor);
                    quote! { device.api_version() >= crate::Version::#version }
                }))
                .chain(device_extensions.iter().map(|ext| {
                    quote! { device.enabled_extensions().#ext }
                }))
                .chain(instance_extensions.iter().map(|ext| {
                    quote! { device.instance().enabled_extensions().#ext }
                }));
                let required_for = format!("`Format::{}`", name);
                let requires_one_of_items = (api_version.iter().map(|(major, minor)| {
                    let version = format_ident!("V{}_{}", major, minor);
                    quote! { api_version: Some(crate::Version::#version), }
                }))
                .chain((!device_extensions.is_empty()).then(|| {
                    let items = device_extensions.iter().map(|ext| ext.to_string());
                    quote! { device_extensions: &[#(#items),*], }
                }))
                .chain((!instance_extensions.is_empty()).then(|| {
                    let items = instance_extensions.iter().map(|ext| ext.to_string());
                    quote! { instance_extensions: &[#(#items),*], }
                }));

                quote! {
                    if !(#(#condition_items)||*) {
                        return Err(crate::RequirementNotMet {
                            required_for: #required_for,
                            requires_one_of: crate::RequiresOneOf {
                                #(#requires_one_of_items)*
                                ..Default::default()
                            },
                        });
                    }
                }
            },
        );

        quote! {
            Self::#name => {
                #(#requires_items)*
            }
        }
    });
    let validate_physical_device_items =
        members.iter().map(|FormatMember { name, requires, .. }| {
            let requires_items = requires.iter().map(
                |RequiresOneOf {
                     api_version,
                     device_extensions,
                     instance_extensions,
                 }| {
                    let condition_items = (api_version.iter().map(|(major, minor)| {
                        let version = format_ident!("V{}_{}", major, minor);
                        quote! { physical_device.api_version() >= crate::Version::#version }
                    }))
                    .chain(device_extensions.iter().map(|ext| {
                        quote! { physical_device.supported_extensions().#ext }
                    }))
                    .chain(instance_extensions.iter().map(|ext| {
                        quote! { physical_device.instance().enabled_extensions().#ext }
                    }));
                    let required_for = format!("`Format::{}`", name);
                    let requires_one_of_items = (api_version.iter().map(|(major, minor)| {
                        let version = format_ident!("V{}_{}", major, minor);
                        quote! { api_version: Some(crate::Version::#version), }
                    }))
                    .chain((!device_extensions.is_empty()).then(|| {
                        let items = device_extensions.iter().map(|ext| ext.to_string());
                        quote! { device_extensions: &[#(#items),*], }
                    }))
                    .chain((!instance_extensions.is_empty()).then(|| {
                        let items = instance_extensions.iter().map(|ext| ext.to_string());
                        quote! { instance_extensions: &[#(#items),*], }
                    }));

                    quote! {
                        if !(#(#condition_items)||*) {
                            return Err(crate::RequirementNotMet {
                                required_for: #required_for,
                                requires_one_of: crate::RequiresOneOf {
                                    #(#requires_one_of_items)*
                                    ..Default::default()
                                },
                            });
                        }
                    }
                },
            );

            quote! {
                Self::#name => {
                    #(#requires_items)*
                }
            }
        });

    quote! {
        /// An enumeration of all the possible formats.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(i32)]
        #[allow(non_camel_case_types)]
        #[non_exhaustive]
        pub enum Format {
            #(#enum_items)*
        }

        impl Format {
            /// Returns the aspects that images of this format have.
            pub fn aspects(self) -> ImageAspects {
                match self {
                    #(#aspects_items)*
                }
            }

            /// Returns the extent in texels (horizontally and vertically) of a single texel
            /// block of this format. A texel block is a rectangle of pixels that is represented by
            /// a single element of this format. It is also the minimum granularity of the extent of
            /// an image; images must always have an extent that's a multiple of the block extent.
            ///
            /// For normal formats, the block extent is [1, 1, 1], meaning that each element of the
            /// format represents one texel. Block-compressed formats encode multiple texels into
            /// a single element. The 422 and 420 YCbCr formats have a block extent of [2, 1, 1] and
            /// [2, 2, 1] respectively, as the red and blue components are shared across multiple
            /// texels.
            pub fn block_extent(self) -> [u32; 3] {
                match self {
                    #(#block_extent_items)*
                    _ => [1, 1, 1],
                }
            }

            /// Returns the size in bytes of a single texel block of this format. Returns `None`
            /// if the texel block size is not well-defined for this format.
            ///
            /// For regular formats, this is the size of a single texel, but for more specialized
            /// formats this may be the size of multiple texels.
            ///
            /// Depth/stencil formats are considered to have an opaque memory representation, and do
            /// not have a well-defined size. Multi-planar formats store the color components
            /// disjointly in memory, and therefore do not have a well-defined size for all
            /// components as a whole. The individual planes do have a well-defined size.
            pub fn block_size(self) -> Option<DeviceSize> {
                match self {
                    #(#block_size_items)*
                    _ => None,
                }
            }

            /// Returns the an opaque object representing the compatibility class of the format.
            /// This can be used to determine whether two formats are compatible for the purposes
            /// of certain Vulkan operations, such as image copying.
            pub fn compatibility(self) -> FormatCompatibility {
                FormatCompatibility(match self {
                    #(#compatibility_items)*
                })
            }

            /// Returns the number of bits per texel block that each component (R, G, B, A) is
            /// represented with. Components that are not present in the format have 0 bits.
            ///
            /// For depth/stencil formats, the depth component is the first, stencil the second. For
            /// multi-planar formats, this is the number of bits across all planes.
            ///
            /// For block-compressed formats, the number of bits in individual components is not
            /// well-defined, and the return value is merely binary: 1 indicates a component
            /// that is present in the format, 0 indicates one that is absent.
            pub fn components(self) -> [u8; 4] {
                match self {
                    #(#components_items)*
                }
            }

            /// Returns the block compression scheme used for this format, if any. Returns `None` if
            /// the format does not use compression.
            pub fn compression(self) -> Option<CompressionType> {
                match self {
                    #(#compression_items)*
                    _ => None,
                }
            }

            /// For multi-planar formats, returns a slice of length 2 or 3, containing the
            /// equivalent regular format of each plane.
            ///
            /// For non-planar formats, returns the empty slice.
            pub fn planes(self) -> &'static [Self] {
                match self {
                    #(#planes_items)*
                    _ => &[],
                }
            }

            /// Returns the number of texels for a single texel block. For most formats, this is
            /// the product of the `block_extent` elements, but for some it differs.
            pub fn texels_per_block(self) -> u8 {
                match self {
                    #(#texels_per_block_items)*
                    _ => 1,
                }
            }

            /// Returns the numeric data type of the color aspect of this format. Returns `None`
            /// for depth/stencil formats.
            pub fn type_color(self) -> Option<NumericType> {
                match self {
                    #(#type_color_items)*
                    _ => None,
                }
            }

            /// Returns the numeric data type of the depth aspect of this format. Returns `None`
            /// color and stencil-only formats.
            pub fn type_depth(self) -> Option<NumericType> {
                match self {
                    #(#type_depth_items)*
                    _ => None,
                }
            }

            /// Returns the numeric data type of the stencil aspect of this format. Returns `None`
            /// for color and depth-only formats.
            pub fn type_stencil(self) -> Option<NumericType> {
                match self {
                    #(#type_stencil_items)*
                    _ => None,
                }
            }

            /// For YCbCr (YUV) formats, returns the way in which the chroma components are
            /// represented. Returns `None` for non-YCbCr formats.
            ///
            /// If an image view is created for one of the formats for which this function returns
            /// `Some`, with the `color` aspect selected, then the view and any samplers that sample
            /// it must be created with an attached sampler YCbCr conversion object.
            pub fn ycbcr_chroma_sampling(self) -> Option<ChromaSampling> {
                match self {
                    #(#ycbcr_chroma_sampling_items)*
                    _ => None,
                }
            }

            #[allow(dead_code)]
            pub(crate) fn validate_device(
                self,
                #[allow(unused_variables)] device: &crate::device::Device,
            ) -> Result<(), crate::RequirementNotMet> {
                match self {
                    #(#validate_device_items)*
                }

                Ok(())
            }

            #[allow(dead_code)]
            pub(crate) fn validate_physical_device(
                self,
                #[allow(unused_variables)] physical_device: &crate::device::physical::PhysicalDevice,
            ) -> Result<(), crate::RequirementNotMet> {
                match self {
                    #(#validate_physical_device_items)*
                }

                Ok(())
            }
        }

        impl TryFrom<ash::vk::Format> for Format {
            type Error = ();

            fn try_from(val: ash::vk::Format) -> Result<Format, ()> {
                match val {
                    #(#try_from_items)*
                    _ => Err(()),
                }
            }
        }

        /// Converts a format enum identifier to a standard Rust type that is suitable for
        /// representing the format in a buffer or image.
        ///
        /// This macro returns one possible suitable representation, but there are usually other
        /// possibilities for a given format. A compile error occurs for formats that have no
        /// well-defined size (the `size` method returns `None`).
        ///
        /// - For regular unpacked formats with one component, this returns a single floating point,
        ///   signed or unsigned integer with the appropriate number of bits. For formats with
        ///   multiple components, an array is returned.
        /// - For packed formats, this returns an unsigned integer with the size of the packed
        ///   element. For multi-packed formats (such as `2PACK16`), an array is returned.
        /// - For compressed formats, this returns `[u8; N]` where N is the size of a block.
        ///
        /// Note: for 16-bit floating point values, you need to import the [`half::f16`] type.
        ///
        /// # Examples
        ///
        /// For arrays:
        ///
        /// ```
        /// # use vulkano::type_for_format;
        /// let pixel: type_for_format!(R32G32B32A32_SFLOAT);
        /// pixel = [1.0f32, 0.0, 0.0, 1.0];
        /// ```
        ///
        /// For [`cgmath`]:
        ///
        /// ```
        /// # use vulkano::type_for_format;
        /// let pixel: type_for_format!(cgmath, R32G32B32A32_SFLOAT);
        /// pixel = cgmath::Vector4::new(1.0f32, 0.0, 0.0, 1.0);
        /// ```
        ///
        /// For [`nalgebra`]:
        ///
        /// ```
        /// # use vulkano::type_for_format;
        /// let pixel: type_for_format!(nalgebra, R32G32B32A32_SFLOAT);
        /// pixel = nalgebra::vector![1.0f32, 0.0, 0.0, 1.0];
        /// ```
        ///
        /// [`cgmath`]: https://crates.io/crates/cgmath
        /// [`nalgebra`]: https://crates.io/crates/nalgebra
        #[macro_export]
        macro_rules! type_for_format {
            #(#type_for_format_items)*
            #(#type_for_format_cgmath_items)*
            #(#type_for_format_nalgebra_items)*
        }
    }
}

fn formats_members(
    formats: &[&Format],
    features: &IndexMap<&str, &Feature>,
    extensions: &IndexMap<&str, &Extension>,
) -> Vec<FormatMember> {
    static BLOCK_EXTENT_REGEX: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(\d+),(\d+),(\d+)$").unwrap());

    formats
        .iter()
        .map(|format| {
            let vulkan_name = format.name.strip_prefix("VK_FORMAT_").unwrap();
            let ffi_name = format_ident!("{}", vulkan_name.to_ascii_uppercase());

            let mut parts = vulkan_name.split('_').collect::<Vec<_>>();

            if ["EXT", "IMG"].contains(parts.last().unwrap()) {
                parts.pop();
            }

            let name = format_ident!("{}", parts.join("_"));

            let mut member = FormatMember {
                name,
                ffi_name,
                requires: Vec::new(),

                aspect_color: false,
                aspect_depth: false,
                aspect_stencil: false,
                aspect_plane0: false,
                aspect_plane1: false,
                aspect_plane2: false,

                block_extent: [1, 1, 1],
                block_size: None,
                compatibility: format_ident!(
                    "Class_{}",
                    format.class.replace('-', "").replace(' ', "_")
                ),
                components: [0u8; 4],
                compression: format
                    .compressed
                    .as_ref()
                    .map(|c| format_ident!("{}", c.replace(' ', "_"))),
                planes: vec![],
                texels_per_block: format.texelsPerBlock,
                type_color: None,
                type_depth: None,
                type_stencil: None,
                ycbcr_chroma_sampling: None,

                type_std_array: None,
                type_cgmath: None,
                type_nalgebra: None,
            };

            for child in &format.children {
                match child {
                    FormatChild::Component {
                        name,
                        bits,
                        numericFormat,
                        ..
                    } => {
                        let bits = if bits == "compressed" {
                            1u8
                        } else {
                            bits.parse().unwrap()
                        };
                        let ty = format_ident!("{}", numericFormat);

                        match name.as_str() {
                            "R" => {
                                member.aspect_color = true;
                                member.components[0] += bits;
                                member.type_color = Some(ty);
                            }
                            "G" => {
                                member.aspect_color = true;
                                member.components[1] += bits;
                                member.type_color = Some(ty);
                            }
                            "B" => {
                                member.aspect_color = true;
                                member.components[2] += bits;
                                member.type_color = Some(ty);
                            }
                            "A" => {
                                member.aspect_color = true;
                                member.components[3] += bits;
                                member.type_color = Some(ty);
                            }
                            "D" => {
                                member.aspect_depth = true;
                                member.components[0] += bits;
                                member.type_depth = Some(ty);
                            }
                            "S" => {
                                member.aspect_stencil = true;
                                member.components[1] += bits;
                                member.type_stencil = Some(ty);
                            }
                            _ => {
                                panic!("Unknown component type {} on format {}", name, format.name)
                            }
                        }
                    }
                    FormatChild::Plane {
                        index, compatible, ..
                    } => {
                        match *index {
                            0 => member.aspect_plane0 = true,
                            1 => member.aspect_plane1 = true,
                            2 => member.aspect_plane2 = true,
                            _ => (),
                        }

                        assert_eq!(*index as usize, member.planes.len());
                        member.planes.push(format_ident!(
                            "{}",
                            compatible.strip_prefix("VK_FORMAT_").unwrap()
                        ));
                    }
                    //FormatChild::SpirvImageFormat { name, .. } => (),
                    _ => (),
                }
            }

            if let Some(block_extent) = format.blockExtent.as_ref() {
                let captures = BLOCK_EXTENT_REGEX.captures(block_extent).unwrap();
                member.block_extent = [
                    captures.get(1).unwrap().as_str().parse().unwrap(),
                    captures.get(2).unwrap().as_str().parse().unwrap(),
                    captures.get(3).unwrap().as_str().parse().unwrap(),
                ];
            } else {
                match format.chroma.as_deref() {
                    Some("420") => member.block_extent = [2, 2, 1],
                    Some("422") => member.block_extent = [2, 1, 1],
                    _ => (),
                }
            };

            // Depth-stencil and multi-planar formats don't have well-defined block sizes.
            if let (Some(numeric_type), true) = (&member.type_color, member.planes.is_empty()) {
                member.block_size = Some(format.blockSize as u64);

                if format.compressed.is_some() {
                    member.type_std_array = Some({
                        let block_size = Literal::usize_unsuffixed(format.blockSize as usize);
                        quote! { [u8; #block_size] }
                    });
                } else if let Some(pack_bits) = format.packed {
                    let pack_elements = format.blockSize * 8 / pack_bits;
                    let element_type = format_ident!("u{}", pack_bits);

                    member.type_std_array = Some(if pack_elements > 1 {
                        let elements = Literal::usize_unsuffixed(pack_elements as usize);
                        quote! { [#element_type; #elements] }
                    } else {
                        quote! { #element_type }
                    });
                } else {
                    let prefix = match numeric_type.to_string().as_str() {
                        "SFLOAT" => "f",
                        "SINT" | "SNORM" | "SSCALED" => "i",
                        "UINT" | "UNORM" | "USCALED" | "SRGB" => "u",
                        _ => unreachable!(),
                    };
                    let bits = member.components[0];
                    let component_type = format_ident!("{}{}", prefix, bits);

                    let component_count = if member.components[1] == 2 * bits {
                        // 422 format with repeated G component
                        4
                    } else {
                        // Normal format
                        member
                            .components
                            .into_iter()
                            .filter(|&c| {
                                if c != 0 {
                                    debug_assert!(c == bits);
                                    true
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    if component_count > 1 {
                        let elements = Literal::usize_unsuffixed(component_count);
                        member.type_std_array = Some(quote! { [#component_type; #elements] });

                        // cgmath only has 1, 2, 3 and 4-component vector types.
                        // Fall back to arrays for anything else.
                        if matches!(component_count, 1 | 2 | 3 | 4) {
                            let ty = format_ident!("{}", format!("Vector{}", component_count));
                            member.type_cgmath = Some(quote! { cgmath::#ty<#component_type> });
                        }

                        member.type_nalgebra = Some(quote! {
                            nalgebra::base::SVector<#component_type, #component_count>
                        });
                    } else {
                        member.type_std_array = Some(quote! { #component_type });
                    }
                }
            }

            if let Some(chroma) = format.chroma.as_ref() {
                member.ycbcr_chroma_sampling = Some(format_ident!("Mode{}", chroma));
            }

            debug_assert!(
                !member.components.iter().all(|x| *x == 0),
                "format {} has 0 components",
                vulkan_name
            );

            for &feature in features.values() {
                for child in &feature.children {
                    if let ExtensionChild::Require { items, .. } = child {
                        for item in items {
                            match item {
                                InterfaceItem::Enum(Enum {
                                    name,
                                    spec: EnumSpec::Offset { extends, .. },
                                    ..
                                }) if name == &format.name && extends == "VkFormat" => {
                                    if let Some(version) = feature.name.strip_prefix("VK_VERSION_")
                                    {
                                        let (major, minor) = version.split_once('_').unwrap();
                                        member.requires.push(RequiresOneOf {
                                            api_version: Some((
                                                major.to_string(),
                                                minor.to_string(),
                                            )),
                                            ..Default::default()
                                        });
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                }
            }

            for &extension in extensions.values() {
                for child in &extension.children {
                    if let ExtensionChild::Require { items, .. } = child {
                        for item in items {
                            if let InterfaceItem::Enum(en) = item {
                                if matches!(
                                    en,
                                    Enum {
                                        name,
                                        spec: EnumSpec::Offset { extends, .. },
                                        ..
                                    } if name == &format.name && extends == "VkFormat")
                                    || matches!(
                                        en,
                                        Enum {
                                            spec: EnumSpec::Alias { alias, extends, .. },
                                            ..
                                        } if alias == &format.name && extends.as_deref() == Some("VkFormat"))
                                {
                                    let extension_name =
                                        extension.name.strip_prefix("VK_").unwrap().to_snake_case();

                                    if member.requires.is_empty() {
                                        member.requires.push(Default::default());
                                    };

                                    let requires = member.requires.first_mut().unwrap();

                                    match extension.ext_type.as_deref() {
                                        Some("device") => {
                                            requires
                                                .device_extensions
                                                .push(format_ident!("{}", extension_name));
                                        }
                                        Some("instance") => {
                                            requires
                                                .instance_extensions
                                                .push(format_ident!("{}", extension_name));
                                        }
                                        _ => (),
                                    }
                                }
                            }
                        }
                    }
                }
            }

            member
        })
        .collect()
}
