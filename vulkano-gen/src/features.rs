// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use heck::SnakeCase;
use regex::Regex;
use std::{
    collections::{hash_map::Entry, HashMap},
    io::Write,
};
use vk_parse::{Type, TypeMember, TypeMemberMarkup, TypeSpec};

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

pub fn write<W: Write>(writer: &mut W, types: &HashMap<&str, (&Type, Vec<&str>)>) {
    write!(writer, "features! {{").unwrap();

    for feat in make_vulkano_features(&types) {
        write!(writer, "\n\t{} => {{", feat.member).unwrap();
        write_doc(writer, &feat);
        write!(writer, "\n\t\tffi_name: {},", feat.ffi_name).unwrap();
        write!(
            writer,
            "\n\t\tffi_members: [{}],",
            feat.ffi_members.join(", ")
        )
        .unwrap();
        write!(
            writer,
            "\n\t\trequires_features: [{}],",
            feat.requires_features.join(", ")
        )
        .unwrap();
        write!(
            writer,
            "\n\t\tconflicts_features: [{}],",
            feat.conflicts_features.join(", ")
        )
        .unwrap();
        write!(
            writer,
            "\n\t\trequired_by_extensions: [{}],",
            feat.required_by_extensions.join(", ")
        )
        .unwrap();
        write!(writer, "\n\t}},").unwrap();
    }

    write!(
        writer,
        "\n}}\n\nfeatures_ffi! {{\n\tapi_version,\n\textensions,"
    )
    .unwrap();

    for ffi in make_vulkano_features_ffi(&types) {
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
struct VulkanoFeature {
    member: String,
    vulkan_doc: String,
    ffi_name: String,
    ffi_members: Vec<String>,
    requires_features: Vec<String>,
    conflicts_features: Vec<String>,
    required_by_extensions: Vec<String>,
}

fn make_vulkano_features(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<VulkanoFeature> {
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

            let ty_name = if vulkan_ty_name == "VkPhysicalDeviceFeatures" {
                "features_vulkan10.features".to_owned()
            } else {
                ffi_member(vulkan_ty_name)
            };

            members(ty).into_iter().for_each(|name| {
                let member = name.to_snake_case();
                match features.entry(member.clone()) {
                    Entry::Vacant(entry) => {
                        let requires_features = requires_features(name);
                        let conflicts_features = conflicts_features(name);
                        let required_by_extensions = required_by_extensions(name);

                        entry.insert(VulkanoFeature {
                            member: member.clone(),
                            vulkan_doc: format!("{}.html#features-{}", vulkan_ty_name, name),
                            ffi_name: member,
                            ffi_members: vec![ty_name.to_owned()],
                            requires_features: requires_features
                                .into_iter()
                                .map(|&s| s.to_snake_case())
                                .collect(),
                            conflicts_features: conflicts_features
                                .into_iter()
                                .map(|&s| s.to_snake_case())
                                .collect(),
                            required_by_extensions: required_by_extensions
                                .iter()
                                .map(|vk_name| vk_name.strip_prefix("VK_").unwrap().to_snake_case())
                                .collect(),
                        });
                    }
                    Entry::Occupied(entry) => {
                        entry.into_mut().ffi_members.push(ty_name.to_owned());
                    }
                };
            });
        });

    let mut names: Vec<_> = features.values().map(|feat| feat.member.clone()).collect();
    names.sort_unstable();
    names
        .into_iter()
        .map(|name| features.remove(&name).unwrap())
        .collect()
}

fn write_doc<W>(writer: &mut W, feat: &VulkanoFeature)
where
    W: Write,
{
    write!(writer, "\n\t\tdoc: \"\n\t\t\t- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/{})", feat.vulkan_doc).unwrap();

    if !feat.requires_features.is_empty() {
        let links: Vec<_> = feat
            .requires_features
            .iter()
            .map(|ext| format!("[`{}`](crate::device::Features::{0})", ext))
            .collect();
        write!(
            writer,
            "\n\t\t\t- Requires feature{}: {}",
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
            "\n\t\t\t- Required by device extension{}: {}",
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
            "\n\t\t\t- Conflicts with feature{}: {}",
            if feat.conflicts_features.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }

    write!(writer, "\n\t\t\",").unwrap();
}

#[derive(Clone, Debug)]
struct VulkanoFeatureFfi {
    member: String,
    ty: String,
    provided_by: Vec<String>,
    conflicts: Vec<String>,
}

fn make_vulkano_features_ffi(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<VulkanoFeatureFfi> {
    let mut feature_included_in: HashMap<&str, Vec<&str>> = HashMap::new();
    sorted_structs(types)
        .into_iter()
        .map(|(ty, provided_by)| {
            let name = ty.name.as_ref().unwrap();
            let provided_by = provided_by
                .iter()
                .map(|provided_by| {
                    if let Some(version) = provided_by.strip_prefix("VK_VERSION_") {
                        format!("api_version >= Version::V{}", version)
                    } else {
                        format!(
                            "extensions.{}",
                            provided_by
                                .strip_prefix("VK_")
                                .unwrap()
                                .to_ascii_lowercase()
                        )
                    }
                })
                .collect();
            let mut conflicts = vec![];
            members(ty)
                .into_iter()
                .for_each(|member| match feature_included_in.entry(member) {
                    Entry::Vacant(entry) => {
                        entry.insert(vec![name]);
                    }
                    Entry::Occupied(entry) => {
                        let conflicters = entry.into_mut();
                        conflicters.iter().for_each(|conflicter| {
                            let conflicter = ffi_member(conflicter);
                            if !conflicts.contains(&conflicter) {
                                conflicts.push(conflicter);
                            }
                        });
                        conflicters.push(name);
                    }
                });

            VulkanoFeatureFfi {
                member: ffi_member(name),
                ty: name.strip_prefix("Vk").unwrap().to_owned(),
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
