// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    spirv_grammar::{SpirvGrammar, SpirvKindEnumerant},
    write_file, VkRegistryData,
};
use heck::SnakeCase;
use indexmap::{map::Entry, IndexMap};
use lazy_static::lazy_static;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use regex::Regex;
use vk_parse::SpirvExtOrCap;

pub fn write(vk_data: &VkRegistryData, grammar: &SpirvGrammar) {
    let grammar_enumerants = grammar
        .operand_kinds
        .iter()
        .find(|operand_kind| operand_kind.kind == "Capability")
        .unwrap()
        .enumerants
        .as_slice();
    let spirv_capabilities_output = spirv_reqs_output(
        &spirv_capabilities_members(&vk_data.spirv_capabilities, grammar_enumerants),
        false,
    );
    let spirv_extensions_output =
        spirv_reqs_output(&spirv_extensions_members(&vk_data.spirv_extensions), true);
    write_file(
        "spirv_reqs.rs",
        format!("vk.xml header version {}", vk_data.header_version),
        quote! {
            #spirv_capabilities_output
            #spirv_extensions_output
        },
    );
}

#[derive(Clone, Debug)]
struct SpirvReqsMember {
    name: String,
    enables: Vec<(Enable, String)>,
}

#[derive(Clone, Debug, PartialEq)]
enum Enable {
    Core((String, String)),
    Extension(Ident),
    Feature(Ident),
    Property((Ident, PropertyValue)),
}

#[derive(Clone, Debug, PartialEq)]
enum PropertyValue {
    Bool,
    BoolMember(Ident),
}

fn spirv_reqs_output(members: &[SpirvReqsMember], extension: bool) -> TokenStream {
    let items = members.iter().map(|SpirvReqsMember { name, enables }| {
        let arm = if extension {
            quote! { #name }
        } else {
            let name = format_ident!("{}", name);
            quote! { Capability::#name }
        };

        if enables.is_empty() {
            quote! {
                #arm => (),
            }
        } else {
            let enables_items = enables.iter().map(|(enable, _description)| match enable {
                Enable::Core((major, minor)) => {
                    let version = format_ident!("V{}_{}", major, minor);
                    quote! {
                        device.api_version() >= Version::#version
                    }
                }
                Enable::Extension(extension) => quote! {
                    device.enabled_extensions().#extension
                },
                Enable::Feature(feature) => quote! {
                    device.enabled_features().#feature
                },
                Enable::Property((name, value)) => {
                    let access = match value {
                        PropertyValue::Bool => quote! {},
                        PropertyValue::BoolMember(member) => quote! {
                            .map(|x| x.#member)
                        },
                    };

                    quote! {
                        device.physical_device().properties().#name #access .unwrap_or(false)
                    }
                }
            });

            let description_items = enables.iter().map(|(_enable, description)| description);

            quote! {
                #arm => {
                    if !(#(#enables_items)||*) {
                        return Err(ShaderSupportError::RequirementsNotMet(&[
                            #(#description_items),*
                        ]));
                    }
                },
            }
        }
    });

    if extension {
        quote! {
            fn check_spirv_extension(device: &Device, extension: &str) -> Result<(), ShaderSupportError> {
                match extension {
                    #(#items)*
                    _ => return Err(ShaderSupportError::NotSupportedByVulkan),
                }
                Ok(())
            }
        }
    } else {
        quote! {
            fn check_spirv_capability(device: &Device, capability: Capability) -> Result<(), ShaderSupportError> {
                match capability {
                    #(#items)*
                    _ => return Err(ShaderSupportError::NotSupportedByVulkan),
                }
                Ok(())
            }
        }
    }
}

fn spirv_capabilities_members(
    capabilities: &[&SpirvExtOrCap],
    grammar_enumerants: &[SpirvKindEnumerant],
) -> Vec<SpirvReqsMember> {
    let mut members: IndexMap<String, SpirvReqsMember> = IndexMap::new();

    for ext_or_cap in capabilities {
        let mut enables: Vec<_> = ext_or_cap.enables.iter().filter_map(make_enable).collect();
        enables.dedup();

        // Find the capability in the list of enumerants, then go backwards through the list to find
        // the first enumerant with the same value.
        let enumerant_pos = grammar_enumerants
            .iter()
            .position(|enumerant| enumerant.enumerant == ext_or_cap.name)
            .unwrap();
        let enumerant_value = &grammar_enumerants[enumerant_pos].value;

        let name = if let Some(enumerant) = grammar_enumerants[..enumerant_pos]
            .iter()
            .rev()
            .take_while(|enumerant| &enumerant.value == enumerant_value)
            .last()
        {
            // Another enumerant was found with the same value, so this one is an alias.
            &enumerant.enumerant
        } else {
            // No other enumerant was found, so this is its canonical name.
            &ext_or_cap.name
        };

        match members.entry(name.clone()) {
            Entry::Occupied(entry) => {
                entry.into_mut().enables.extend(enables);
            }
            Entry::Vacant(entry) => {
                entry.insert(SpirvReqsMember {
                    name: name.clone(),
                    enables,
                });
            }
        }
    }

    members.into_iter().map(|(_, v)| v).collect()
}

fn spirv_extensions_members(extensions: &[&SpirvExtOrCap]) -> Vec<SpirvReqsMember> {
    extensions
        .iter()
        .map(|ext_or_cap| {
            let enables: Vec<_> = ext_or_cap.enables.iter().filter_map(make_enable).collect();

            SpirvReqsMember {
                name: ext_or_cap.name.clone(),
                enables,
            }
        })
        .collect()
}

lazy_static! {
    static ref BIT: Regex = Regex::new(r"_BIT(?:_NV)?$").unwrap();
}

fn make_enable(enable: &vk_parse::Enable) -> Option<(Enable, String)> {
    if matches!(enable, vk_parse::Enable::Version(version) if version == "VK_API_VERSION_1_0") {
        return None;
    }

    Some(match enable {
        vk_parse::Enable::Version(version) => {
            let version = version.strip_prefix("VK_API_VERSION_").unwrap();
            let (major, minor) = version.split_once('_').unwrap();

            (
                Enable::Core((major.parse().unwrap(), minor.parse().unwrap())),
                format!("Vulkan API version {}.{}", major, minor),
            )
        }
        vk_parse::Enable::Extension(extension) => {
            let extension_name = extension.strip_prefix("VK_").unwrap().to_snake_case();

            (
                Enable::Extension(format_ident!("{}", extension_name)),
                format!("device extension `{}`", extension_name),
            )
        }
        vk_parse::Enable::Feature(feature) => {
            let feature_name = feature.feature.to_snake_case();

            (
                Enable::Feature(format_ident!("{}", feature_name)),
                format!("feature `{}`", feature_name),
            )
        }
        vk_parse::Enable::Property(property) => {
            let property_name = property.member.to_snake_case();

            let (value, description) = if property.value == "VK_TRUE" {
                (PropertyValue::Bool, format!("property `{}`", property_name))
            } else if let Some(member) = property.value.strip_prefix("VK_SUBGROUP_FEATURE_") {
                let member = BIT.replace(member, "").to_snake_case();
                (
                    PropertyValue::BoolMember(format_ident!("{}", member)),
                    format!("property `{}.{}`", property_name, member),
                )
            } else {
                unimplemented!()
            };

            (
                Enable::Property((format_ident!("{}", property_name), value)),
                description,
            )
        }
        _ => unimplemented!(),
    })
}
