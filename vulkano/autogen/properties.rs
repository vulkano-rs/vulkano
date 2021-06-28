// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use heck::SnakeCase;
use indexmap::IndexMap;
use regex::Regex;
use std::{
    collections::{hash_map::Entry, HashMap},
    io::Write,
};
use vk_parse::{Extension, Type, TypeMember, TypeMemberMarkup, TypeSpec};

pub fn write<W: Write>(
    writer: &mut W,
    types: &HashMap<&str, (&Type, Vec<&str>)>,
    extensions: &IndexMap<&str, &Extension>,
) {
    write!(writer, "crate::device::properties::properties! {{").unwrap();

    for feat in make_vulkano_properties(&types) {
        write!(writer, "\n\t{} => {{", feat.member).unwrap();
        write_doc(writer, &feat);
        write!(writer, "\n\t\tty: {},", feat.ty).unwrap();
        write!(writer, "\n\t\tffi_name: {},", feat.ffi_name).unwrap();
        write!(
            writer,
            "\n\t\tffi_members: [{}],",
            feat.ffi_members.join(", ")
        )
        .unwrap();
        write!(writer, "\n\t}},").unwrap();
    }

    write!(
        writer,
        "\n}}\n\ncrate::device::properties::properties_ffi! {{\n\tapi_version,\n\tdevice_extensions,\n\tinstance_extensions,"
    )
    .unwrap();

    for ffi in make_vulkano_properties_ffi(types, extensions) {
        write!(writer, "\n\t{} => {{", ffi.member).unwrap();
        write!(writer, "\n\t\tty: {},", ffi.ty).unwrap();
        write!(
            writer,
            "\n\t\tprovided_by: [{}],",
            ffi.provided_by.join(", ")
        )
        .unwrap();
        write!(writer, "\n\t\tconflicts: [{}],", ffi.conflicts.join(", ")).unwrap();
        write!(writer, "\n\t}},").unwrap();
    }

    write!(writer, "\n}}").unwrap();
}

#[derive(Clone, Debug)]
struct VulkanoProperty {
    member: String,
    ty: String,
    vulkan_doc: String,
    ffi_name: String,
    ffi_members: Vec<String>,
}

fn make_vulkano_properties(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<VulkanoProperty> {
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

        let ty_name = if vulkan_ty_name == "VkPhysicalDeviceProperties" {
            "properties_vulkan10.properties".to_owned()
        } else if vulkan_ty_name == "VkPhysicalDeviceLimits" {
            "properties_vulkan10.properties.limits".to_owned()
        } else if vulkan_ty_name == "VkPhysicalDeviceSparseProperties" {
            "properties_vulkan10.properties.sparse_properties".to_owned()
        } else {
            ffi_member(vulkan_ty_name)
        };

        members(ty)
            .into_iter()
            .for_each(|Member { name, ty, len }| {
                if ty == "VkPhysicalDeviceLimits" || ty == "VkPhysicalDeviceSparseProperties" {
                    return;
                }

                let vulkano_member = name.to_snake_case();
                let vulkano_ty = match name {
                    "apiVersion" => "crate::Version",
                    _ => vulkano_type(ty, len),
                };
                match properties.entry(vulkano_member.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(VulkanoProperty {
                            member: vulkano_member.clone(),
                            ty: vulkano_ty.to_owned(),
                            vulkan_doc: format!("{}.html#limits-{}", vulkan_ty_name, name),
                            ffi_name: vulkano_member,
                            ffi_members: vec![ty_name.to_owned()],
                        });
                    }
                    Entry::Occupied(entry) => {
                        entry.into_mut().ffi_members.push(ty_name.to_owned());
                    }
                };
            });
    });

    let mut names: Vec<_> = properties
        .values()
        .map(|feat| feat.member.clone())
        .collect();
    names.sort_unstable();
    names
        .into_iter()
        .map(|name| properties.remove(&name).unwrap())
        .collect()
}

fn write_doc<W>(writer: &mut W, feat: &VulkanoProperty)
where
    W: Write,
{
    write!(writer, "\n\t\tdoc: \"\n\t\t\t- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/{})", feat.vulkan_doc).unwrap();
    write!(writer, "\n\t\t\",").unwrap();
}

#[derive(Clone, Debug)]
struct VulkanoPropertyFfi {
    member: String,
    ty: String,
    provided_by: Vec<String>,
    conflicts: Vec<String>,
}

fn make_vulkano_properties_ffi<'a>(
    types: &'a HashMap<&str, (&Type, Vec<&str>)>,
    extensions: &IndexMap<&'a str, &Extension>,
) -> Vec<VulkanoPropertyFfi> {
    let mut property_included_in: HashMap<&str, Vec<&str>> = HashMap::new();
    sorted_structs(types)
        .into_iter()
        .map(|(ty, provided_by)| {
            let ty_name = ty.name.as_ref().unwrap();
            let provided_by = provided_by
                .iter()
                .map(|provided_by| {
                    if let Some(version) = provided_by.strip_prefix("VK_VERSION_") {
                        format!("api_version >= crate::Version::V{}", version)
                    } else {
                        format!(
                            "{}_extensions.{}",
                            extensions[provided_by].ext_type.as_ref().unwrap().as_str(),
                            provided_by
                                .strip_prefix("VK_")
                                .unwrap()
                                .to_ascii_lowercase()
                        )
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

            VulkanoPropertyFfi {
                member: ffi_member(ty_name),
                ty: ty_name.strip_prefix("Vk").unwrap().to_owned(),
                provided_by,
                conflicts,
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

fn vulkano_type(ty: &str, len: Option<&str>) -> &'static str {
    if let Some(len) = len {
        match ty {
            "char" => "String",
            "uint8_t" if len == "VK_LUID_SIZE" => "[u8; 8]",
            "uint8_t" if len == "VK_UUID_SIZE" => "[u8; 16]",
            "uint32_t" if len == "2" => "[u32; 2]",
            "uint32_t" if len == "3" => "[u32; 3]",
            "float" if len == "2" => "[f32; 2]",
            _ => unimplemented!("{}[{}]", ty, len),
        }
    } else {
        match ty {
            "float" => "f32",
            "int32_t" => "i32",
            "size_t" => "usize",
            "uint8_t" => "u8",
            "uint32_t" => "u32",
            "uint64_t" => "u64",
            "VkBool32" => "bool",
            "VkConformanceVersion" => "crate::device::physical::ConformanceVersion",
            "VkDeviceSize" => "u64",
            "VkDriverId" => "crate::device::physical::DriverId",
            "VkExtent2D" => "[u32; 2]",
            "VkPhysicalDeviceType" => "crate::device::physical::PhysicalDeviceType",
            "VkPointClippingBehavior" => "crate::device::physical::PointClippingBehavior",
            "VkResolveModeFlags" => "crate::render_pass::ResolveModes",
            "VkSampleCountFlags" => "crate::image::SampleCounts",
            "VkSampleCountFlagBits" => "crate::image::SampleCount",
            "VkShaderCorePropertiesFlagsAMD" => "crate::device::physical::ShaderCoreProperties",
            "VkShaderFloatControlsIndependence" => {
                "crate::device::physical::ShaderFloatControlsIndependence"
            }
            "VkShaderStageFlags" => "crate::descriptor::descriptor::ShaderStages",
            "VkSubgroupFeatureFlags" => "crate::device::physical::SubgroupFeatures",
            _ => unimplemented!("{}", ty),
        }
    }
}
