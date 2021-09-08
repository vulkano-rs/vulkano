// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, VkRegistryData};
use heck::SnakeCase;
use indexmap::IndexMap;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use regex::Regex;
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Write as _,
};
use vk_parse::{Extension, Type, TypeMember, TypeMemberMarkup, TypeSpec};

// This is not included in vk.xml, so it's added here manually
fn requires_features(name: &str) -> &'static [&'static str] {
    match name {
        "sparseImageInt64Atomics" => &["shaderImageInt64Atomics"],
        "sparseImageFloat32Atomics" => &["shaderImageFloat32Atomics"],
        "sparseImageFloat32AtomicAdd" => &["shaderImageFloat32AtomicAdd"],
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

fn required_by_extensions(name: &str) -> &'static [&'static str] {
    match name {
        "shaderDrawParameters" => &["VK_KHR_shader_draw_parameters"],
        "drawIndirectCount" => &["VK_KHR_draw_indirect_count"],
        "samplerMirrorClampToEdge" => &["VK_KHR_sampler_mirror_clamp_to_edge"],
        "descriptorIndexing" => &["VK_EXT_descriptor_indexing"],
        "samplerFilterMinmax" => &["VK_EXT_sampler_filter_minmax"],
        "shaderOutputViewportIndex" => &["VK_EXT_shader_viewport_index_layer"],
        "shaderOutputLayer" => &["VK_EXT_shader_viewport_index_layer"],
        _ => &[],
    }
}

pub fn write(vk_data: &VkRegistryData) {
    let features_output = features_output(&features_members(&vk_data.types));
    let features_ffi_output =
        features_ffi_output(&features_ffi_members(&vk_data.types, &vk_data.extensions));
    write_file(
        "features.rs",
        format!("vk.xml header version {}", vk_data.header_version),
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
    ffi_name: Ident,
    ffi_members: Vec<(Ident, TokenStream)>,
    requires_features: Vec<Ident>,
    conflicts_features: Vec<Ident>,
    required_by_extensions: Vec<Ident>,
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
                        return Err(FeatureRestrictionError {
                            feature: #name_string,
                            restriction: FeatureRestriction::RequiresFeature(#string),
                        });
                    }
                }
            });
            let conflicts_features_items = conflicts_features.iter().map(|feature| {
                let string = feature.to_string();
                quote! {
                    if self.#feature {
                        return Err(FeatureRestrictionError {
                            feature: #name_string,
                            restriction: FeatureRestriction::ConflictsFeature(#string),
                        });
                    }
                }
            });
            let required_by_extensions_items = required_by_extensions.iter().map(|extension| {
                let string = extension.to_string();
                quote! {
                    if extensions.#extension {
                        return Err(FeatureRestrictionError {
                            feature: #name_string,
                            restriction: FeatureRestriction::RequiredByExtension(#string),
                        });
                    }
                }
            });
            quote! {
                if self.#name {
                    if !supported.#name {
                        return Err(FeatureRestrictionError {
                            feature: #name_string,
                            restriction: FeatureRestriction::NotSupported,
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

    let none_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: false,
        }
    });

    let all_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            #name: true,
        }
    });

    let is_superset_of_items = members.iter().map(|FeaturesMember { name, .. }| {
        quote! {
            (self.#name || !other.#name)
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
                    std::array::IntoIter::new([
                        #(#ffi_members),*
                    ]).flatten().next().map(|f| *f = features.#name as ash::vk::Bool32);
                }
            } else {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { &mut self.#ffi_member #ffi_member_field .#ffi_name }
                });
                quote! {
                    std::array::IntoIter::new([
                        #(#ffi_members),*
                    ]).next().map(|f| *f = features.#name as ash::vk::Bool32);
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
                    #name: std::array::IntoIter::new([
                        #(#ffi_members),*
                    ]).flatten().next().unwrap_or(0) != 0,
                }
            } else {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { features_ffi.#ffi_member #ffi_member_field .#ffi_name }
                });
                quote! {
                    #name: std::array::IntoIter::new([
                        #(#ffi_members),*
                    ]).next().unwrap_or(0) != 0,
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
        /// # Example
        ///
        /// ```
        /// use vulkano::device::Features;
        /// # let physical_device: vulkano::device::physical::PhysicalDevice = return;
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
        /// if !physical_device.supported_features().is_superset_of(&minimal_features) {
        ///     panic!("The physical device is not good enough for this application.");
        /// }
        ///
        /// assert!(optimal_features.is_superset_of(&minimal_features));
        /// let features_to_request = optimal_features.intersection(physical_device.supported_features());
        /// ```
        ///
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        pub struct Features {
            #(#struct_items)*
        }

        impl Features {
            /// Checks enabled features against the device version, device extensions and each other.
            pub(super) fn check_requirements(
                &self,
                supported: &Features,
                api_version:
                Version,
                extensions: &DeviceExtensions,
            ) -> Result<(), FeatureRestrictionError> {
                #(#check_requirements_items)*
                Ok(())
            }

            /// Builds a `Features` object with all values to false.
            pub const fn none() -> Features {
                Features {
                    #(#none_items)*
                }
            }

            /// Builds a `Features` object with all values to true.
            ///
            /// > **Note**: This function is used for testing purposes, and is probably useless in
            /// > a real code.
            pub const fn all() -> Features {
                Features {
                    #(#all_items)*
                }
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each feature of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub const fn is_superset_of(&self, other: &Features) -> bool {
                #(#is_superset_of_items)&&*
            }

            /// Builds a `Features` that is the intersection of `self` and another `Features`
            /// object.
            ///
            /// The result's field will be true if it is also true in both `self` and `other`.
            pub const fn intersection(&self, other: &Features) -> Features {
                Features {
                    #(#intersection_items)*
                }
            }

            /// Builds a `Features` that is the difference of another `Features` object from `self`.
            ///
            /// The result's field will be true if it is true in `self` but not `other`.
            pub const fn difference(&self, other: &Features) -> Features {
                Features {
                    #(#difference_items)*
                }
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
                }
            }
        }

    }
}

fn features_members(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<FeaturesMember> {
    let mut features = HashMap::new();
    std::iter::once(&types["VkPhysicalDeviceFeatures"])
        .chain(sorted_structs(types).into_iter())
        .filter(|(ty, _)| {
            ty.name.as_ref().map(|s| s.as_str()) == Some("VkPhysicalDeviceFeatures")
                || ty.structextends.as_ref().map(|s| s.as_str())
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
                            ffi_members: vec![ty_name.clone()],
                            requires_features: requires_features
                                .into_iter()
                                .map(|&s| format_ident!("{}", s.to_snake_case()))
                                .collect(),
                            conflicts_features: conflicts_features
                                .into_iter()
                                .map(|&s| format_ident!("{}", s.to_snake_case()))
                                .collect(),
                            required_by_extensions: required_by_extensions
                                .iter()
                                .map(|vk_name| {
                                    format_ident!(
                                        "{}",
                                        vk_name.strip_prefix("VK_").unwrap().to_snake_case()
                                    )
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
    write!(writer, "- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/{}.html#features-{})", vulkan_ty_name, feat.name).unwrap();

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
            .map(|ext| format!("[`{}`](crate::device::DeviceExtensions::{0})", ext))
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
                if std::array::IntoIter::new([#(#provided_by),*]).any(|x| x) &&
                    std::array::IntoIter::new([#(self.#conflicts.is_none()),*]).all(|x| x) {
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
                api_version: Version,
                device_extensions: &DeviceExtensions,
                instance_extensions: &InstanceExtensions,
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
    let mut feature_included_in: HashMap<&str, Vec<&str>> = HashMap::new();
    sorted_structs(types)
        .into_iter()
        .map(|(ty, provided_by)| {
            let ty_name = ty.name.as_ref().unwrap();
            let provided_by = provided_by
                .iter()
                .map(|provided_by| {
                    if let Some(version) = provided_by.strip_prefix("VK_VERSION_") {
                        let version = format_ident!("V{}", version);
                        quote! { api_version >= Version::#version }
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
            ty.structextends.as_ref().map(|s| s.as_str())
                == Some("VkPhysicalDeviceFeatures2,VkDeviceCreateInfo")
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
            } else if provided_by
                .iter()
                .find(|s| s.starts_with("VK_KHR_"))
                .is_some()
            {
                i32::MAX - 2
            } else if provided_by
                .iter()
                .find(|s| s.starts_with("VK_EXT_"))
                .is_some()
            {
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
