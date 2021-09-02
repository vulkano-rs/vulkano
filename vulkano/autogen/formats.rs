// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use lazy_static::lazy_static;
use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote};
use regex::Regex;

pub fn write(formats: &[&str]) -> TokenStream {
    write_formats(&make_formats(formats))
}

#[derive(Clone, Debug)]
struct FormatMember {
    name: Ident,
    ffi_name: Ident,

    aspect_color: bool,
    aspect_depth: bool,
    aspect_stencil: bool,
    aspect_plane0: bool,
    aspect_plane1: bool,
    aspect_plane2: bool,

    block_dimensions: [u32; 2],
    compatibility: TokenStream,
    components: [u8; 4],
    compression: Option<Ident>,
    planes: Vec<Ident>,
    size: Option<u64>,
    type_color: Option<Ident>,
    type_depth: Option<Ident>,
    type_stencil: Option<Ident>,
}

fn write_formats(members: &[FormatMember]) -> TokenStream {
    let enum_items = members.iter().map(|FormatMember { name, ffi_name, .. }| {
        quote! { #name = ash::vk::Format::#ffi_name.as_raw(), }
    });
    let aspects_color_items = members.iter().filter_map(
        |FormatMember {
             name, aspect_color, ..
         }| {
            if !aspect_color {
                // Negated to reduce the length of the list
                Some(name)
            } else {
                None
            }
        },
    );
    let aspects_depth_items = members.iter().filter_map(
        |FormatMember {
             name, aspect_depth, ..
         }| {
            if *aspect_depth {
                Some(name)
            } else {
                None
            }
        },
    );
    let aspects_stencil_items = members.iter().filter_map(
        |FormatMember {
             name,
             aspect_stencil,
             ..
         }| {
            if *aspect_stencil {
                Some(name)
            } else {
                None
            }
        },
    );
    let aspects_plane0_items = members.iter().filter_map(
        |FormatMember {
             name,
             aspect_plane0,
             ..
         }| {
            if *aspect_plane0 {
                Some(name)
            } else {
                None
            }
        },
    );
    let aspects_plane1_items = members.iter().filter_map(
        |FormatMember {
             name,
             aspect_plane1,
             ..
         }| {
            if *aspect_plane1 {
                Some(name)
            } else {
                None
            }
        },
    );
    let aspects_plane2_items = members.iter().filter_map(
        |FormatMember {
             name,
             aspect_plane2,
             ..
         }| {
            if *aspect_plane2 {
                Some(name)
            } else {
                None
            }
        },
    );
    let block_dimensions_items = members.iter().filter_map(
        |FormatMember {
             name,
             block_dimensions: [x, y],
             ..
         }| {
            if *x == 1 && *y == 1 {
                None
            } else {
                let x = Literal::u32_unsuffixed(*x);
                let y = Literal::u32_unsuffixed(*y);
                Some(quote! { Self::#name => [#x, #y], })
            }
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
    let size_items = members
        .iter()
        .filter_map(|FormatMember { name, size, .. }| {
            size.as_ref().map(|size| {
                let size = Literal::u64_unsuffixed(*size);
                quote! { Self::#name => Some(#size), }
            })
        });
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
    let try_from_items = members.iter().map(|FormatMember { name, ffi_name, .. }| {
        quote! { ash::vk::Format::#ffi_name => Ok(Self::#name), }
    });

    quote! {
        /// An enumeration of all the possible formats.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(i32)]
        #[allow(non_camel_case_types)]
        pub enum Format {
            #(#enum_items)*
        }

        impl Format {
            /// Returns the aspects that images of this format have.
            pub fn aspects(&self) -> crate::image::ImageAspects {
                crate::image::ImageAspects {
                    color: !matches!(self, #(Format::#aspects_color_items)|* ),
                    depth: matches!(self, #(Format::#aspects_depth_items)|* ),
                    stencil: matches!(self, #(Format::#aspects_stencil_items)|* ),
                    plane0: matches!(self, #(Format::#aspects_plane0_items)|* ),
                    plane1: matches!(self, #(Format::#aspects_plane1_items)|* ),
                    plane2: matches!(self, #(Format::#aspects_plane2_items)|* ),
                    ..crate::image::ImageAspects::none()
                }
            }

            /// Returns the dimensions in texels (horizontally and vertically) of a single texel
            /// block of this format. A texel block is a rectangle of pixels that is represented by
            /// a single element of this format. It is also the minimum granularity of the size of
            /// an image; images must always have a size that's a multiple of the block size.
            ///
            /// For normal formats, the block size is [1, 1], meaning that each element of the
            /// format represents one texel. Block-compressed formats encode multiple texels into
            /// a single element. The 422 and 420 YCbCr formats have a block size of [2, 1] and
            /// [2, 2] respectively, as the red and blue components are shared across multiple
            /// texels.
            pub fn block_dimensions(&self) -> [u32; 2] {
                match self {
                    #(#block_dimensions_items)*
                    _ => [1, 1],
                }
            }

            /// Returns the an opaque object representing the compatibility class of the format.
            /// This can be used to determine whether two formats are compatible for the purposes
            /// of certain Vulkan operations, such as image copying.
            pub fn compatibility(&self) -> crate::format::FormatCompatibility {
                use crate::format::{CompressionType, FormatCompatibilityInner};
                crate::format::FormatCompatibility(match self {
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
            /// well-defined, and the return value will is merely binary: 1 indicates a component
            /// that is present in the format, 0 indicates one that is absent.
            pub fn components(&self) -> [u8; 4] {
                match self {
                    #(#components_items)*
                }
            }

            /// Returns the block compression scheme used for this format, if any. Returns `None` if
            /// the format does not use compression.
            pub fn compression(&self) -> Option<crate::format::CompressionType> {
                use crate::format::CompressionType;
                match self {
                    #(#compression_items)*
                    _ => None,
                }
            }

            /// For multi-planar formats, returns a slice of length 2 or 3, containing the
            /// equivalent regular format of each plane.
            ///
            /// For non-planar formats, returns the empty slice.
            pub fn planes(&self) -> &'static [Self] {
                match self {
                    #(#planes_items)*
                    _ => &[],
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
            pub fn size(&self) -> Option<crate::DeviceSize> {
                match self {
                    #(#size_items)*
                    _ => None,
                }
            }

            /// Returns the numeric data type of the color aspect of this format. Returns `None`
            /// for depth/stencil formats.
            pub fn type_color(&self) -> Option<crate::format::NumericType> {
                use crate::format::NumericType;
                match self {
                    #(#type_color_items)*
                    _ => None,
                }
            }

            /// Returns the numeric data type of the depth aspect of this format. Returns `None`
            /// color and stencil-only formats.
            pub fn type_depth(&self) -> Option<crate::format::NumericType> {
                use crate::format::NumericType;
                match self {
                    #(#type_depth_items)*
                    _ => None,
                }
            }

            /// Returns the numeric data type of the stencil aspect of this format. Returns `None`
            /// for color and depth-only formats.
            pub fn type_stencil(&self) -> Option<crate::format::NumericType> {
                use crate::format::NumericType;
                match self {
                    #(#type_stencil_items)*
                    _ => None,
                }
            }
        }

        impl std::convert::TryFrom<ash::vk::Format> for Format {
            type Error = ();

            fn try_from(val: ash::vk::Format) -> Result<Format, ()> {
                match val {
                    #(#try_from_items)*
                    _ => Err(()),
                }
            }
        }
    }
}

lazy_static! {
    static ref DEPTH_REGEX: Regex = Regex::new(r"^D\d+$").unwrap();
    static ref STENCIL_REGEX: Regex = Regex::new(r"^S\d+$").unwrap();
    static ref PACK_REGEX: Regex = Regex::new(r"^(\d*)PACK(\d+)$").unwrap();
    static ref COMPONENTS_REGEX: Regex = Regex::new(r"([ABDEGRSX])(\d+)").unwrap();
    static ref RGB_COMPONENTS_REGEX: Regex = Regex::new(r"([BGR])(\d+)").unwrap();
}

fn make_formats(formats: &[&str]) -> Vec<FormatMember> {
    let mut members = formats
        .iter()
        .map(|vulkan_name| {
            let vulkan_name = vulkan_name.strip_prefix("VK_FORMAT_").unwrap();
            let ffi_name = format_ident!("{}", vulkan_name.to_ascii_uppercase());

            let mut parts = vulkan_name.split('_').collect::<Vec<_>>();

            if ["EXT", "IMG"].contains(parts.last().unwrap()) {
                parts.pop();
            }

            let name = format_ident!("{}", parts.join("_"));

            let mut member = FormatMember {
                name: name.clone(),
                ffi_name: ffi_name.clone(),

                aspect_color: false,
                aspect_depth: false,
                aspect_stencil: false,
                aspect_plane0: false,
                aspect_plane1: false,
                aspect_plane2: false,

                block_dimensions: [1, 1],
                compatibility: Default::default(),
                components: [0u8; 4],
                compression: None,
                planes: vec![],
                size: None,
                type_color: None,
                type_depth: None,
                type_stencil: None,
            };

            let depth_match = parts.iter().position(|part| DEPTH_REGEX.is_match(part));
            let stencil_match = parts.iter().position(|part| STENCIL_REGEX.is_match(part));

            if depth_match.is_some() || stencil_match.is_some() {
                // Depth+stencil formats

                member.compatibility = quote! {
                    DepthStencil { ty: Self::#name as u8 }
                };

                if let Some(pos) = depth_match {
                    let (_, components) = components(vulkan_name, &parts[pos..pos + 1]);
                    member.aspect_depth = true;
                    member.components[0] = components[0];
                    member.type_depth = Some(format_ident!("{}", parts[pos + 1]));
                }

                if let Some(pos) = stencil_match {
                    let (_, components) = components(vulkan_name, &parts[pos..pos + 1]);
                    member.aspect_stencil = true;
                    member.components[1] = components[1];
                    member.type_stencil = Some(format_ident!("{}", parts[pos + 1]));
                }
            } else if *parts.last().unwrap() == "BLOCK" {
                // Block-compressed formats

                parts.pop();
                let numeric_type = format_ident!("{}", parts.pop().unwrap());
                let ty = parts[0];
                let mut subtype: u8 = 0;

                let (block_size, components) = if ty == "ASTC" {
                    let dim = parts[1].split_once('x').unwrap();
                    member.block_dimensions = [dim.0.parse().unwrap(), dim.1.parse().unwrap()];
                    subtype =
                        (member.block_dimensions[0] as u8) << 4 | member.block_dimensions[1] as u8;
                    (16, [1, 1, 1, 1])
                } else if ty == "BC1" && parts[1] == "RGB" {
                    member.block_dimensions = [4, 4];
                    subtype = 3;
                    (8, [1, 1, 1, 0])
                } else if ty == "BC1" && parts[1] == "RGBA" {
                    member.block_dimensions = [4, 4];
                    subtype = 4;
                    (8, [1, 1, 1, 1])
                } else if ty == "BC4" {
                    member.block_dimensions = [4, 4];
                    (8, [1, 0, 0, 0])
                } else if ty == "BC5" {
                    member.block_dimensions = [4, 4];
                    (16, [1, 1, 0, 0])
                } else if ty == "BC6H" {
                    member.block_dimensions = [4, 4];
                    (16, [1, 1, 1, 0])
                } else if ty == "BC2" || ty == "BC3" || ty == "BC7" {
                    member.block_dimensions = [4, 4];
                    (16, [1, 1, 1, 1])
                } else if ty == "EAC" && parts[1] == "R11" {
                    member.block_dimensions = [4, 4];
                    subtype = 1;
                    (8, [1, 0, 0, 0])
                } else if ty == "EAC" && parts[1] == "R11G11" {
                    member.block_dimensions = [4, 4];
                    subtype = 2;
                    (16, [1, 1, 0, 0])
                } else if ty == "ETC2" && parts[1] == "R8G8B8" {
                    member.block_dimensions = [4, 4];
                    subtype = 3;
                    (8, [1, 1, 1, 0])
                } else if ty == "ETC2" && parts[1] == "R8G8B8A1" {
                    member.block_dimensions = [4, 4];
                    subtype = 31;
                    (8, [1, 1, 1, 1])
                } else if ty == "ETC2" && parts[1] == "R8G8B8A8" {
                    member.block_dimensions = [4, 4];
                    subtype = 4;
                    (16, [1, 1, 1, 1])
                } else if ty.starts_with("PVRTC") {
                    if parts[1] == "2BPP" {
                        member.block_dimensions = [8, 4];
                        subtype = 2;
                    } else if parts[1] == "4BPP" {
                        member.block_dimensions = [4, 4];
                        subtype = 4;
                    }

                    (8, [1, 1, 1, 1])
                } else {
                    panic!("Unrecognised block compression format: {}", vulkan_name);
                };

                let compression = format_ident!("{}", ty);
                member.aspect_color = true;
                member.compatibility = quote! {
                    Compressed {
                        compression: CompressionType::#compression,
                        subtype: #subtype,
                    }
                };
                member.components = components;
                member.compression = Some(compression);
                member.size = Some(block_size);
                member.type_color = Some(numeric_type);
            } else {
                // Other formats

                let many_pack = PACK_REGEX
                    .captures(*parts.last().unwrap())
                    .map(|captures| {
                        parts.pop();
                        let first = captures.get(1).unwrap().as_str();
                        first == "3" || first == "4"
                    })
                    .unwrap_or(false);

                let numeric_type = parts.pop().unwrap();
                member.aspect_color = true;
                member.type_color = Some(format_ident!("{}", numeric_type));

                if ["420", "422", "444"].contains(parts.last().unwrap()) {
                    let ty = parts.pop().unwrap();

                    if ty == "420" {
                        member.block_dimensions = [2, 2];
                    } else if ty == "422" {
                        member.block_dimensions = [2, 1];
                    }
                }

                let size_factor =
                    member.block_dimensions[0] as u64 * member.block_dimensions[1] as u64;

                if let Some(planes) = parts.last().unwrap().strip_suffix("PLANE") {
                    // Multi-planar formats

                    let planes: usize = planes.parse().unwrap();
                    parts.pop();
                    let captures = COMPONENTS_REGEX.captures(parts[0]).unwrap();
                    let bits: u8 = captures.get(2).unwrap().as_str().parse().unwrap();

                    {
                        member.aspect_plane0 = true;
                        member
                            .planes
                            .push(plane_format(&parts[0], numeric_type, bits));
                    }

                    {
                        member.aspect_plane1 = true;
                        member
                            .planes
                            .push(plane_format(&parts[1], numeric_type, bits));
                    }

                    if planes == 3 {
                        member.aspect_plane2 = true;
                        member
                            .planes
                            .push(plane_format(&parts[2], numeric_type, bits));
                    }

                    let compatibility = format_ident!("YCbCr{}Plane", planes);
                    let (_, mut components) = components(vulkan_name, &parts);
                    let plane0_index = component_index(captures.get(1).unwrap().as_str()).unwrap();
                    components[plane0_index] *= size_factor as u8;

                    let block_texels = size_factor as u8;

                    member.compatibility = quote! {
                        #compatibility {
                            bits: #bits,
                            block_texels: #block_texels,
                        }
                    };
                    member.components = components;
                } else {
                    // Non-planar formats

                    {
                        let (block_size, components) = components(vulkan_name, &parts);
                        member.components = components;
                        member.size = Some(block_size * size_factor);

                        if size_factor != 1 {
                            let captures = COMPONENTS_REGEX.captures(parts[0]).unwrap();
                            let bits: u8 = captures.get(2).unwrap().as_str().parse().unwrap();
                            let g_even = captures.get(1).unwrap().as_str() != "G";
                            member.compatibility = quote! {
                                YCbCr1Plane {
                                    bits: #bits,
                                    g_even: #g_even,
                                }
                            };
                        } else if many_pack {
                            let captures = COMPONENTS_REGEX.captures(parts[0]).unwrap();
                            let bits: u8 = captures.get(2).unwrap().as_str().parse().unwrap();
                            member.compatibility = quote! {
                                YCbCrRGBA {
                                    bits: #bits,
                                }
                            };
                        } else {
                            let size = block_size as u8;
                            member.compatibility = quote! {
                                Normal {
                                    size: #size,
                                }
                            };
                        }
                    }
                }
            }

            debug_assert!(
                !member.components.iter().all(|x| *x == 0),
                "format {} has 0 components",
                vulkan_name
            );

            debug_assert!(
                member.type_color.is_some()
                    || member.type_depth.is_some()
                    || member.type_stencil.is_some(),
                "format {} has no numeric type",
                vulkan_name,
            );

            member
        })
        .collect::<Vec<_>>();

    members.sort_by_key(|member| {
        if member.aspect_plane0 {
            if member.block_dimensions == [2, 2] {
                10
            } else if member.block_dimensions == [2, 1] {
                11
            } else {
                12
            }
        } else if member.compression.is_some() {
            1
        } else if member.block_dimensions != [1, 1] {
            5
        } else {
            0
        }
    });

    members
}

fn components(vulkan_name: &str, parts: &[&str]) -> (u64, [u8; 4]) {
    let mut total_bits = 0;
    let mut components = [0u8; 4];

    for &part in parts {
        for component in COMPONENTS_REGEX.captures_iter(part) {
            let bits: u64 = component.get(2).unwrap().as_str().parse().unwrap();
            total_bits += bits;
            let index = match component_index(component.get(1).unwrap().as_str()) {
                Some(x) => x,
                None => continue,
            };
            components[index] += bits as u8;
        }
    }

    if total_bits % 8 != 0 {
        panic!("total bits is not divisible by 0 for {}", vulkan_name);
    }

    (total_bits / 8, components)
}

fn component_index(letter: &str) -> Option<usize> {
    match letter {
        "R" | "D" => Some(0),
        "G" | "S" => Some(1),
        "B" => Some(2),
        "A" => Some(3),
        _ => None,
    }
}

fn plane_format(part: &str, numeric_type: &str, bits: u8) -> Ident {
    let mut format = part.to_owned();
    let mut num_components = 0;

    for (index, component) in RGB_COMPONENTS_REGEX.captures_iter(part).enumerate() {
        let capture = component.get(1).unwrap();
        let letter = match index {
            0 => "R",
            1 => "G",
            2 => "B",
            _ => unimplemented!(),
        };
        num_components += 1;

        unsafe {
            format[capture.range()]
                .as_bytes_mut()
                .copy_from_slice(letter.as_bytes());
        }
    }

    if bits % 8 == 0 {
        format_ident!("{}_{}", format, numeric_type)
    } else {
        assert!(bits > 8 && bits < 16);
        let prefix = format!("{}", num_components);
        format_ident!(
            "{}_{}_{}PACK16",
            format,
            numeric_type,
            if num_components == 1 { "" } else { &prefix }
        )
    }
}
