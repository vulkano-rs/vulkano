// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, RegistryData};
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

pub fn write(data: &RegistryData) {
    let properties_output = properties_output(&properties_members(&data.types));
    let properties_ffi_output =
        properties_ffi_output(&properties_ffi_members(&data.types, &data.extensions));
    write_file(
        "properties.rs",
        format!("vk.xml header version {}", data.header_version),
        quote! {
            #properties_output
            #properties_ffi_output
        },
    );
}

#[derive(Clone, Debug)]
struct PropertiesMember {
    name: Ident,
    ty: TokenStream,
    doc: String,
    ffi_name: Ident,
    ffi_members: Vec<(Ident, TokenStream)>,
    optional: bool,
}

fn properties_output(members: &[PropertiesMember]) -> TokenStream {
    let struct_items = members.iter().map(
        |PropertiesMember {
             name,
             ty,
             doc,
             optional,
             ..
         }| {
            if *optional {
                quote! {
                    #[doc = #doc]
                    pub #name: Option<#ty>,
                }
            } else {
                quote! {
                    #[doc = #doc]
                    pub #name: #ty,
                }
            }
        },
    );

    let from_items = members.iter().map(
        |PropertiesMember {
             name,
             ty,
             ffi_name,
             ffi_members,
             optional,
             ..
         }| {
            if *optional {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { properties_ffi.#ffi_member.map(|s| s #ffi_member_field .#ffi_name) }
                });

                quote! {
                    #name: std::array::IntoIter::new([
                        #(#ffi_members),*
                    ]).flatten().next().and_then(|x| <#ty>::from_vulkan(x)),
                }
            } else {
                let ffi_members = ffi_members.iter().map(|(ffi_member, ffi_member_field)| {
                    quote! { properties_ffi.#ffi_member #ffi_member_field .#ffi_name }
                });

                quote! {
                    #name: std::array::IntoIter::new([
                        #(#ffi_members),*
                    ]).next().and_then(|x| <#ty>::from_vulkan(x)).unwrap(),
                }
            }
        },
    );

    quote! {
        /// Represents all the properties of a physical device.
        ///
        /// Depending on the highest version of Vulkan supported by the physical device, and the
        /// available extensions, not every property may be available. For that reason, some
        /// properties are wrapped in an `Option`.
        #[derive(Clone, Debug, Default)]
        pub struct Properties {
            #(#struct_items)*
        }

        impl From<&PropertiesFfi> for Properties {
            fn from(properties_ffi: &PropertiesFfi) -> Self {
                Properties {
                    #(#from_items)*
                }
            }
        }
    }
}

fn properties_members(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<PropertiesMember> {
    let mut properties = HashMap::new();
    std::array::IntoIter::new([
        &types["VkPhysicalDeviceProperties"],
        &types["VkPhysicalDeviceLimits"],
        &types["VkPhysicalDeviceSparseProperties"],
    ])
    .chain(sorted_structs(types).into_iter())
    .filter(|(ty, _)| {
        let name = ty.name.as_ref().map(|s| s.as_str());
        name == Some("VkPhysicalDeviceProperties")
            || name == Some("VkPhysicalDeviceLimits")
            || name == Some("VkPhysicalDeviceSparseProperties")
            || ty.structextends.as_ref().map(|s| s.as_str()) == Some("VkPhysicalDeviceProperties2")
    })
    .for_each(|(ty, _)| {
        let vulkan_ty_name = ty.name.as_ref().unwrap();

        let (ty_name, optional) = if vulkan_ty_name == "VkPhysicalDeviceProperties" {
            (
                (format_ident!("properties_vulkan10"), quote! { .properties }),
                false,
            )
        } else if vulkan_ty_name == "VkPhysicalDeviceLimits" {
            (
                (
                    format_ident!("properties_vulkan10"),
                    quote! { .properties.limits },
                ),
                false,
            )
        } else if vulkan_ty_name == "VkPhysicalDeviceSparseProperties" {
            (
                (
                    format_ident!("properties_vulkan10"),
                    quote! { .properties.sparse_properties },
                ),
                false,
            )
        } else {
            (
                (format_ident!("{}", ffi_member(vulkan_ty_name)), quote! {}),
                true,
            )
        };

        members(ty)
            .into_iter()
            .for_each(|Member { name, ty, len }| {
                if ty == "VkPhysicalDeviceLimits" || ty == "VkPhysicalDeviceSparseProperties" {
                    return;
                }

                let vulkano_member = name.to_snake_case();
                let vulkano_ty = match name {
                    "apiVersion" => quote! { Version },
                    _ => vulkano_type(ty, len),
                };
                match properties.entry(vulkano_member.clone()) {
                    Entry::Vacant(entry) => {
                        let mut member = PropertiesMember {
                            name: format_ident!("{}", vulkano_member),
                            ty: vulkano_ty,
                            doc: String::new(),
                            ffi_name: format_ident!("{}", vulkano_member),
                            ffi_members: vec![ty_name.clone()],
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

    let mut names: Vec<_> = properties
        .values()
        .map(|prop| prop.name.to_string())
        .collect();
    names.sort_unstable();
    names
        .into_iter()
        .map(|name| properties.remove(&name).unwrap())
        .collect()
}

fn make_doc(prop: &mut PropertiesMember, vulkan_ty_name: &str) {
    let writer = &mut prop.doc;
    write!(writer, "- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/{}.html#limits-{})", vulkan_ty_name, prop.name).unwrap();
}

#[derive(Clone, Debug)]
struct PropertiesFfiMember {
    name: Ident,
    ty: Ident,
    provided_by: Vec<TokenStream>,
    conflicts: Vec<Ident>,
}

fn properties_ffi_output(members: &[PropertiesFfiMember]) -> TokenStream {
    let struct_items = members.iter().map(|PropertiesFfiMember { name, ty, .. }| {
        quote! { #name: Option<ash::vk::#ty>, }
    });

    let make_chain_items = members.iter().map(
        |PropertiesFfiMember {
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
        pub(crate) struct PropertiesFfi {
            properties_vulkan10: ash::vk::PhysicalDeviceProperties2KHR,
            #(#struct_items)*
        }

        impl PropertiesFfi {
            pub(crate) fn make_chain(
                &mut self,
                api_version: Version,
                device_extensions: &DeviceExtensions,
                instance_extensions: &InstanceExtensions,
            ) {
                self.properties_vulkan10 = Default::default();
                let head = &mut self.properties_vulkan10;
                #(#make_chain_items)*
            }

            pub(crate) fn head_as_ref(&self) -> &ash::vk::PhysicalDeviceProperties2KHR {
                &self.properties_vulkan10
            }

            pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceProperties2KHR {
                &mut self.properties_vulkan10
            }
        }
    }
}

fn properties_ffi_members<'a>(
    types: &'a HashMap<&str, (&Type, Vec<&str>)>,
    extensions: &IndexMap<&'a str, &Extension>,
) -> Vec<PropertiesFfiMember> {
    let mut property_included_in: HashMap<&str, Vec<&str>> = HashMap::new();
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
            members(ty).into_iter().for_each(|Member { name, .. }| {
                match property_included_in.entry(name) {
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
                }
            });

            PropertiesFfiMember {
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
            ty.structextends.as_ref().map(|s| s.as_str()) == Some("VkPhysicalDeviceProperties2")
        })
        .collect();
    let regex = Regex::new(r"^VkPhysicalDeviceVulkan\d+Properties$").unwrap();
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
    let (base, suffix) = ty_name.rsplit_once("_properties").unwrap();
    format!("properties_{}{}", base, suffix)
}

struct Member<'a> {
    name: &'a str,
    ty: &'a str,
    len: Option<&'a str>,
}

fn members(ty: &Type) -> Vec<Member> {
    let regex = Regex::new(r"\[([A-Za-z0-9_]+)\]\s*$").unwrap();
    if let TypeSpec::Members(members) = &ty.spec {
        members
            .iter()
            .filter_map(|member| {
                if let TypeMember::Definition(def) = member {
                    let name = def.markup.iter().find_map(|markup| match markup {
                        TypeMemberMarkup::Name(name) => Some(name.as_str()),
                        _ => None,
                    });
                    let ty = def.markup.iter().find_map(|markup| match markup {
                        TypeMemberMarkup::Type(ty) => Some(ty.as_str()),
                        _ => None,
                    });
                    let len = def
                        .markup
                        .iter()
                        .find_map(|markup| match markup {
                            TypeMemberMarkup::Enum(len) => Some(len.as_str()),
                            _ => None,
                        })
                        .or_else(|| {
                            regex
                                .captures(&def.code)
                                .and_then(|cap| cap.get(1))
                                .map(|m| m.as_str())
                        });
                    if name != Some("sType") && name != Some("pNext") {
                        return name.map(|name| Member {
                            name,
                            ty: ty.unwrap(),
                            len,
                        });
                    }
                }
                None
            })
            .collect()
    } else {
        vec![]
    }
}

fn vulkano_type(ty: &str, len: Option<&str>) -> TokenStream {
    if let Some(len) = len {
        match ty {
            "char" => quote! { String },
            "uint8_t" if len == "VK_LUID_SIZE" => quote! { [u8; 8] },
            "uint8_t" if len == "VK_UUID_SIZE" => quote! { [u8; 16] },
            "uint32_t" if len == "2" => quote! { [u32; 2] },
            "uint32_t" if len == "3" => quote! { [u32; 3] },
            "float" if len == "2" => quote! { [f32; 2] },
            _ => unimplemented!("{}[{}]", ty, len),
        }
    } else {
        match ty {
            "float" => quote! { f32 },
            "int32_t" => quote! { i32 },
            "int64_t" => quote! { i64 },
            "size_t" => quote! { usize },
            "uint8_t" => quote! { u8 },
            "uint32_t" => quote! { u32 },
            "uint64_t" => quote! { u64 },
            "VkBool32" => quote! { bool },
            "VkConformanceVersion" => quote! { ConformanceVersion },
            "VkDeviceSize" => quote! { DeviceSize },
            "VkDriverId" => quote! { DriverId },
            "VkExtent2D" => quote! { [u32; 2] },
            "VkPhysicalDeviceType" => quote! { PhysicalDeviceType },
            "VkPointClippingBehavior" => quote! { PointClippingBehavior },
            "VkResolveModeFlags" => quote! { ResolveModes },
            "VkSampleCountFlags" => quote! { SampleCounts },
            "VkSampleCountFlagBits" => quote! { SampleCount },
            "VkShaderCorePropertiesFlagsAMD" => quote! { ShaderCoreProperties },
            "VkShaderFloatControlsIndependence" => quote! { ShaderFloatControlsIndependence },
            "VkShaderStageFlags" => quote! { ShaderStages },
            "VkSubgroupFeatureFlags" => quote! { SubgroupFeatures },
            _ => unimplemented!("{}", ty),
        }
    }
}
