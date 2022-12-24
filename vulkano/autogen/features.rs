// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, IndexMap, VkRegistryData};
use ahash::HashMap;
use heck::ToSnakeCase;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use regex::Regex;
use std::{collections::hash_map::Entry, fmt::Write as _};
use vk_parse::{Extension, Type, TypeMember, TypeMemberMarkup, TypeSpec};

// This is not included in vk.xml, so it's added here manually
fn requires_features(name: &str) -> &'static [&'static str] {
    match name {
        "sparseImageInt64Atomics" => &["shaderImageInt64Atomics"],
        "sparseImageFloat32Atomics" => &["shaderImageFloat32Atomics"],
        "sparseImageFloat32AtomicAdd" => &["shaderImageFloat32AtomicAdd"],
        "sparseImageFloat32AtomicMinMax" => &["shaderImageFloat32AtomicMinMax"],
        _ => &[],
    }
}

fn conflicts_features(name: &str) -> &'static [&'static str] {
    match name {
        "shadingRateImage" => &[
            "pipelineFragmentShadingRate",
            "primitiveFragmentShadingRate",
            "attachmentFragmentShadingRate",
        ],
        "fragmentDensityMap" => &[
            "pipelineFragmentShadingRate",
            "primitiveFragmentShadingRate",
            "attachmentFragmentShadingRate",
        ],
        "pipelineFragmentShadingRate" => &["shadingRateImage", "fragmentDensityMap"],
        "primitiveFragmentShadingRate" => &["shadingRateImage", "fragmentDensityMap"],
        "attachmentFragmentShadingRate" => &["shadingRateImage", "fragmentDensityMap"],
        _ => &[],
    }
}

fn required_by_extensions(name: &str) -> &'static [(&'static str, &'static str)] {
    match name {
        "shaderDrawParameters" => &[("V1_2", "VK_KHR_shader_draw_parameters")],
        "drawIndirectCount" => &[("V1_2", "VK_KHR_draw_indirect_count")],
        "samplerMirrorClampToEdge" => &[("V1_2", "VK_KHR_sampler_mirror_clamp_to_edge")],
        "descriptorIndexing" => &[("V1_2", "VK_EXT_descriptor_indexing")],
        "samplerFilterMinmax" => &[("V1_2", "VK_EXT_sampler_filter_minmax")],
        "shaderOutputViewportIndex" => &[("V1_2", "VK_EXT_shader_viewport_index_layer")],
        "shaderOutputLayer" => &[("V1_2", "VK_EXT_shader_viewport_index_layer")],
        _ => &[],
    }
}

pub fn write(vk_data: &VkRegistryData) {
    let features_output = features_output(&features_members(&vk_data.types));
    let features_ffi_output =
        features_ffi_output(&features_ffi_members(&vk_data.types, &vk_data.extensions));
    write_file(
        "features.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        quote! {
            #features_output
            #features_ffi_output
        },
    );
}

#[derive(Clone, Debug)]
struct FeaturesMember {
    name: Ident,
    doc: String,
    raw: String,
    ffi_name: Ident,
    ffi_members: Vec<(Ident, TokenStream)>,
    requires_features: Vec<Ident>,
    conflicts_features: Vec<Ident>,
    required_by_extensions: Vec<(Ident, Ident)>,
    optional: bool,
}

fn features_output(members: &[FeaturesMember]) -> TokenStream {
    let struct_items = members.iter().map(|FeaturesMember { name, doc, .. }| {
        quote! {
            #[doc = #doc]
            pub #name: bool,
        }
    });

    let check_requirements_items = members.iter().map(
        |FeaturesMember {
             name,
             requires_features,
             conflicts_features,
             required_by_extensions,
             ..
         }| {
            let name_string = name.to_string();
            let requires_features_items = requires_features.iter().map(|feature| {
                let string = feature.to_string();
                quote! {
                    if !self.#feature {
                        return Err(crate::device::FeatureRestrictionError {
                            feature: #name_string,
                            restriction: crate::device::FeatureRestriction::RequiresFeature(#string),
                        });
                    }
                }
            });
            let conflicts_features_items = conflicts_features.iter().map(|feature| {
                let string = feature.to_string();
                quote! {
                    if self.#feature {
                        return Err(crate::device::FeatureRestrictionError {
                            feature: #name_string,
                            restriction: crate::device::FeatureRestriction::ConflictsFeature(#string),
                        });
                    }
                }
            });
            let required_by_extensions_items =
                required_by_extensions.iter().map(|(version, extension)| {
                    let string = extension.to_string();
                    quote! {
                        if extensions.#extension && api_version >= crate::Version::#version {
                            return Err(crate::device::FeatureRestrictionError {
                                feature: #name_string,
                                restriction: crate::device::FeatureRestriction::RequiredByExtension(#string),
                            });
                        }
                    }
                });
            quote! {
                if self.#name {
                    if !supported.#name {
                        return Err(crate::device::FeatureRestrictionError {
                            feature: #name_string,
                            restriction: crate::device::FeatureRestriction::NotSupported,
                        });
                    }

                    #(#requires_features_items)*
                    #(#conflicts_features_items)*
                } else {
                    #(#required_by_extensions_items)*
                }
            }
        },
    );

    let empty_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: false,
        }
    });

    let all_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: true,
        }
    });

    let intersects_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            (self.#name && other.#name)
        }
    });

    let contains_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            (self.#name || !other.#name)
        }
    });

    let union_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: self.#name || other.#name,
        }
    });

    let intersection_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: self.#name && other.#name,
        }
    });

    let difference_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: self.#name && !other.#name,
        }
    });

    let symmetric_difference_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: self.#name ^ other.#name,
        }
    });

    let debug_items = members.iter().map(|FeaturesMember { name, raw, .. }| {
        quote! {
            if self.#name {
                if !first { write!(f, ", ")? }
                else { first = false; }
                f.write_str(#raw)?;
            }
        }
    });

    let arr_items = members.iter().map(|FeaturesMember { name, raw, .. }| {
        quote! {
            (#raw, self.#name),
        }
    });
    let arr_len = members.len();

    let write_items = members.iter().map(
        |FeaturesMember {
             name,
             ffi_name,
             ffi_members,
             optional,
             ..
         }| {
            if *optional {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { self.#ffi_member.as_mut().map(|s| &mut s #ffi_member_field .#ffi_name) }
                });
                quote! {
                    if let Some(f) = [
                        #(#ffi_members),*
                    ].into_iter().flatten().next() {
                        *f = features.#name as ash::vk::Bool32;
                    }
                }
            } else {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { &mut self.#ffi_member #ffi_member_field .#ffi_name }
                });
                quote! {
                    if let Some(f) = [
                        #(#ffi_members),*
                    ].into_iter().next() {
                        *f = features.#name as ash::vk::Bool32;
                    }
                }
            }
        },
    );

    let from_items = members.iter().map(
        |FeaturesMember {
             name,
             ffi_name,
             ffi_members,
             optional,
             ..
         }| {
            if *optional {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { features_ffi.#ffi_member.map(|s| s #ffi_member_field .#ffi_name) }
                });
                quote! {
                    #name: [
                        #(#ffi_members),*
                    ].into_iter().flatten().next().unwrap_or(0) != 0,
                }
            } else {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { features_ffi.#ffi_member #ffi_member_field .#ffi_name }
                });
                quote! {
                    #name: [
                        #(#ffi_members),*
                    ].into_iter().next().unwrap_or(0) != 0,
                }
            }
        },
    );

    quote! {
        /// Represents all the features that are available on a physical device or enabled on
        /// a logical device.
        ///
        /// Note that the `robust_buffer_access` is guaranteed to be supported by all Vulkan
        /// implementations.
        ///
        /// # Examples
        ///
        /// ```
        /// use vulkano::device::Features;
        /// # let physical_device: vulkano::device::physical::PhysicalDevice = return;
        /// let minimal_features = Features {
        ///     geometry_shader: true,
        ///     ..Features::empty()
        /// };
        ///
        /// let optimal_features = vulkano::device::Features {
        ///     geometry_shader: true,
        ///     tessellation_shader: true,
        ///     ..Features::empty()
        /// };
        ///
        /// if !physical_device.supported_features().is_superset_of(&minimal_features) {
        ///     panic!("The physical device is not good enough for this application.");
        /// }
        ///
        /// assert!(optimal_features.is_superset_of(&minimal_features));
        /// let features_to_request = optimal_features.intersection(physical_device.supported_features());
        /// ```
        #[derive(Copy, Clone, PartialEq, Eq, Hash)]
        pub struct Features {
            #(#struct_items)*
            pub _ne: crate::NonExhaustive,
        }

        impl Default for Features {
            #[inline]
            fn default() -> Self {
                Self::empty()
            }
        }

        impl Features {
            /// Checks enabled features against the device version, device extensions and each
            /// other.
            pub(super) fn check_requirements(
                &self,
                supported: &Features,
                api_version: crate::Version,
                extensions: &crate::device::DeviceExtensions,
            ) -> Result<(), crate::device::FeatureRestrictionError> {
                #(#check_requirements_items)*
                Ok(())
            }

            /// Returns an `Features` object with none of the members set.
            #[inline]
            pub const fn empty() -> Self {
                Self {
                    #(#empty_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns an `Features` object with none of the members set.
            #[deprecated(since = "0.31.0", note = "Use `empty` instead.")]
            #[inline]
            pub const fn none() -> Self {
                Self::empty()
            }

            /// Returns a `Features` object with all of the members set.
            #[cfg(test)]
            pub(crate) const fn all() -> Features {
                Features {
                    #(#all_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns whether any members are set in both `self` and `other`.
            #[inline]
            pub const fn intersects(&self, other: &Self) -> bool {
                #(#intersects_items)||*
            }

            /// Returns whether all members in `other` are set in `self`.
            #[inline]
            pub const fn contains(&self, other: &Self) -> bool {
                #(#contains_items)&&*
            }

            /// Returns whether all members in `other` are set in `self`.
            #[deprecated(since = "0.31.0", note = "Use `contains` instead.")]
            #[inline]
            pub const fn is_superset_of(&self, other: &Self) -> bool {
                self.contains(other)
            }

            /// Returns the union of `self` and `other`.
            #[inline]
            pub const fn union(&self, other: &Self) -> Self {
                Self {
                    #(#union_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns the intersection of `self` and `other`.
            #[inline]
            pub const fn intersection(&self, other: &Self) -> Self {
                Self {
                    #(#intersection_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns `self` without the members set in `other`.
            #[inline]
            pub const fn difference(&self, other: &Self) -> Self {
                Self {
                    #(#difference_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns the members set in `self` or `other`, but not both.
            #[inline]
            pub const fn symmetric_difference(&self, other: &Self) -> Self {
                Self {
                    #(#symmetric_difference_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }
        }

        impl std::ops::BitAnd for Features {
            type Output = Features;

            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                self.intersection(&rhs)
            }
        }

        impl std::ops::BitAndAssign for Features {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(&rhs);
            }
        }

        impl std::ops::BitOr for Features {
            type Output = Features;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                self.union(&rhs)
            }
        }

        impl std::ops::BitOrAssign for Features {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(&rhs);
            }
        }

        impl std::ops::BitXor for Features {
            type Output = Features;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                self.symmetric_difference(&rhs)
            }
        }

        impl std::ops::BitXorAssign for Features {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(&rhs);
            }
        }

        impl std::ops::Sub for Features {
            type Output = Features;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                self.difference(&rhs)
            }
        }

        impl std::ops::SubAssign for Features {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(&rhs);
            }
        }

        impl std::fmt::Debug for Features {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut std::fmt:: Formatter<'_>) -> Result<(), std::fmt::Error> {
                write!(f, "[")?;

                let mut first = true;
                #(#debug_items)*

                write!(f, "]")
            }
        }

        impl FeaturesFfi {
            pub(crate) fn write(&mut self, features: &Features) {
                #(#write_items)*
            }
        }

        impl From<&FeaturesFfi> for Features {
            fn from(features_ffi: &FeaturesFfi) -> Self {
                Features {
                    #(#from_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }
        }

        impl IntoIterator for Features {
            type Item = (&'static str, bool);
            type IntoIter = std::array::IntoIter<Self::Item, #arr_len>;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                [#(#arr_items)*].into_iter()
            }
        }
    }
}

fn features_members(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<FeaturesMember> {
    let mut features = HashMap::default();
    std::iter::once(&types["VkPhysicalDeviceFeatures"])
        .chain(sorted_structs(types).into_iter())
        .filter(|(ty, _)| {
            ty.name.as_deref() == Some("VkPhysicalDeviceFeatures")
                || ty.structextends.as_deref()
                    == Some("VkPhysicalDeviceFeatures2,VkDeviceCreateInfo")
        })
        .for_each(|(ty, _)| {
            let vulkan_ty_name = ty.name.as_ref().unwrap();

            let (ty_name, optional) = if vulkan_ty_name == "VkPhysicalDeviceFeatures" {
                (
                    (format_ident!("features_vulkan10"), quote! { .features }),
                    false,
                )
            } else {
                (
                    (format_ident!("{}", ffi_member(vulkan_ty_name)), quote! {}),
                    true,
                )
            };

            members(ty).into_iter().for_each(|vulkan_name| {
                let name = vulkan_name.to_snake_case();
                match features.entry(name.clone()) {
                    Entry::Vacant(entry) => {
                        let requires_features = requires_features(vulkan_name);
                        let conflicts_features = conflicts_features(vulkan_name);
                        let required_by_extensions = required_by_extensions(vulkan_name);

                        let mut member = FeaturesMember {
                            name: format_ident!("{}", name),
                            doc: String::new(),
                            ffi_name: format_ident!("{}", name),
                            raw: vulkan_name.to_owned(),
                            ffi_members: vec![ty_name.clone()],
                            requires_features: requires_features
                                .iter()
                                .map(|&s| format_ident!("{}", s.to_snake_case()))
                                .collect(),
                            conflicts_features: conflicts_features
                                .iter()
                                .map(|&s| format_ident!("{}", s.to_snake_case()))
                                .collect(),
                            required_by_extensions: required_by_extensions
                                .iter()
                                .map(|(version, vk_name)| {
                                    let version = format_ident!("{}", version);
                                    let name = format_ident!(
                                        "{}",
                                        vk_name.strip_prefix("VK_").unwrap().to_snake_case()
                                    );
                                    (version, name)
                                })
                                .collect(),
                            optional,
                        };
                        make_doc(&mut member, vulkan_ty_name);
                        entry.insert(member);
                    }
                    Entry::Occupied(entry) => {
                        entry.into_mut().ffi_members.push(ty_name.clone());
                    }
                };
            });
        });

    let mut names: Vec<_> = features
        .values()
        .map(|feat| feat.name.to_string())
        .collect();
    names.sort_unstable();
    names
        .into_iter()
        .map(|name| features.remove(&name).unwrap())
        .collect()
}

fn make_doc(feat: &mut FeaturesMember, vulkan_ty_name: &str) {
    let writer = &mut feat.doc;
    write!(
        writer,
        "- [Vulkan documentation](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/{}.html#features-{})",
        vulkan_ty_name,
        feat.raw
    )
    .unwrap();

    if !feat.requires_features.is_empty() {
        let links: Vec<_> = feat
            .requires_features
            .iter()
            .map(|ext| format!("[`{}`](crate::device::Features::{0})", ext))
            .collect();
        write!(
            writer,
            "\n- Requires feature{}: {}",
            if feat.requires_features.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }

    if !feat.required_by_extensions.is_empty() {
        let links: Vec<_> = feat
            .required_by_extensions
            .iter()
            .map(|(_, ext)| format!("[`{}`](crate::device::DeviceExtensions::{0})", ext))
            .collect();
        write!(
            writer,
            "\n- Required by device extension{}: {}",
            if feat.required_by_extensions.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }

    if !feat.conflicts_features.is_empty() {
        let links: Vec<_> = feat
            .conflicts_features
            .iter()
            .map(|ext| format!("[`{}`](crate::device::Features::{0})", ext))
            .collect();
        write!(
            writer,
            "\n- Conflicts with feature{}: {}",
            if feat.conflicts_features.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }
}

#[derive(Clone, Debug)]
struct FeaturesFfiMember {
    name: Ident,
    ty: Ident,
    provided_by: Vec<TokenStream>,
    conflicts: Vec<Ident>,
}

fn features_ffi_output(members: &[FeaturesFfiMember]) -> TokenStream {
    let struct_items = members.iter().map(|FeaturesFfiMember { name, ty, .. }| {
        quote! { #name: Option<ash::vk::#ty>, }
    });

    let make_chain_items = members.iter().map(
        |FeaturesFfiMember {
             name,
             provided_by,
             conflicts,
             ..
         }| {
            quote! {
                if [#(#provided_by),*].into_iter().any(|x| x) &&
                    [#(self.#conflicts.is_none()),*].into_iter().all(|x| x) {
                    self.#name = Some(Default::default());
                    let member = self.#name.as_mut().unwrap();
                    member.p_next = head.p_next;
                    head.p_next = member as *mut _ as _;
                }
            }
        },
    );

    quote! {
        #[derive(Default)]
        pub(crate) struct FeaturesFfi {
            features_vulkan10: ash::vk::PhysicalDeviceFeatures2KHR,
            #(#struct_items)*
        }

        impl FeaturesFfi {
            pub(crate) fn make_chain(
                &mut self,
                api_version: crate::Version,
                device_extensions: &crate::device::DeviceExtensions,
                _instance_extensions: &crate::instance::InstanceExtensions,
            ) {
                self.features_vulkan10 = Default::default();
                let head = &mut self.features_vulkan10;
                #(#make_chain_items)*
            }

            pub(crate) fn head_as_ref(&self) -> &ash::vk::PhysicalDeviceFeatures2KHR {
                &self.features_vulkan10
            }

            pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceFeatures2KHR {
                &mut self.features_vulkan10
            }
        }
    }
}

fn features_ffi_members<'a>(
    types: &'a HashMap<&str, (&Type, Vec<&str>)>,
    extensions: &IndexMap<&'a str, &Extension>,
) -> Vec<FeaturesFfiMember> {
    let mut feature_included_in: HashMap<&str, Vec<&str>> = HashMap::default();
    sorted_structs(types)
        .into_iter()
        .map(|(ty, provided_by)| {
            let ty_name = ty.name.as_ref().unwrap();
            let provided_by = provided_by
                .iter()
                .map(|provided_by| {
                    if let Some(version) = provided_by.strip_prefix("VK_VERSION_") {
                        let version = format_ident!("V{}", version);
                        quote! { api_version >= crate::Version::#version }
                    } else {
                        let member = format_ident!(
                            "{}_extensions",
                            extensions[provided_by].ext_type.as_ref().unwrap().as_str()
                        );
                        let name = format_ident!(
                            "{}",
                            provided_by
                                .strip_prefix("VK_")
                                .unwrap()
                                .to_ascii_lowercase(),
                        );

                        quote! { #member.#name }
                    }
                })
                .collect();
            let mut conflicts = vec![];
            members(ty)
                .into_iter()
                .for_each(|member| match feature_included_in.entry(member) {
                    Entry::Vacant(entry) => {
                        entry.insert(vec![ty_name]);
                    }
                    Entry::Occupied(entry) => {
                        let conflicters = entry.into_mut();
                        conflicters.iter().for_each(|conflicter| {
                            let conflicter = ffi_member(conflicter);
                            if !conflicts.contains(&conflicter) {
                                conflicts.push(conflicter);
                            }
                        });
                        conflicters.push(ty_name);
                    }
                });

            FeaturesFfiMember {
                name: format_ident!("{}", ffi_member(ty_name)),
                ty: format_ident!("{}", ty_name.strip_prefix("Vk").unwrap()),
                provided_by,
                conflicts: conflicts
                    .into_iter()
                    .map(|s| format_ident!("{}", s))
                    .collect(),
            }
        })
        .collect()
}

fn sorted_structs<'a>(
    types: &'a HashMap<&str, (&'a Type, Vec<&'a str>)>,
) -> Vec<&'a (&'a Type, Vec<&'a str>)> {
    let mut structs: Vec<_> = types
        .values()
        .filter(|(ty, _)| {
            ty.structextends.as_deref() == Some("VkPhysicalDeviceFeatures2,VkDeviceCreateInfo")
        })
        .collect();
    let regex = Regex::new(r"^VkPhysicalDeviceVulkan\d+Features$").unwrap();
    structs.sort_unstable_by_key(|&(ty, provided_by)| {
        let name = ty.name.as_ref().unwrap();
        (
            !regex.is_match(name),
            if let Some(version) = provided_by
                .iter()
                .find_map(|s| s.strip_prefix("VK_VERSION_"))
            {
                let (major, minor) = version.split_once('_').unwrap();
                major.parse::<i32>().unwrap() << 22 | minor.parse::<i32>().unwrap() << 12
            } else if provided_by.iter().any(|s| s.starts_with("VK_KHR_")) {
                i32::MAX - 2
            } else if provided_by.iter().any(|s| s.starts_with("VK_EXT_")) {
                i32::MAX - 1
            } else {
                i32::MAX
            },
            name,
        )
    });

    structs
}

fn ffi_member(ty_name: &str) -> String {
    let ty_name = ty_name
        .strip_prefix("VkPhysicalDevice")
        .unwrap()
        .to_snake_case();
    let (base, suffix) = ty_name.rsplit_once("_features").unwrap();
    format!("features_{}{}", base, suffix)
}

fn members(ty: &Type) -> Vec<&str> {
    if let TypeSpec::Members(members) = &ty.spec {
        members
            .iter()
            .filter_map(|member| {
                if let TypeMember::Definition(def) = member {
                    let name = def.markup.iter().find_map(|markup| match markup {
                        TypeMemberMarkup::Name(name) => Some(name.as_str()),
                        _ => None,
                    });
                    if name != Some("sType") && name != Some("pNext") {
                        return name;
                    }
                }
                None
            })
            .collect()
    } else {
        vec![]
    }
}
