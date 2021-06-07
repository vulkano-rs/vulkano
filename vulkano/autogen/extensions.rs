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
use std::io::Write;
use vk_parse::Extension;

// This is not included in vk.xml, so it's added here manually
fn required_if_supported(name: &str) -> bool {
    match name {
        "VK_KHR_portability_subset" => true,
        _ => false,
    }
}

fn conflicts_extensions(name: &str) -> &'static [&'static str] {
    match name {
        "VK_KHR_buffer_device_address" => &["VK_EXT_buffer_device_address"],
        "VK_EXT_buffer_device_address" => &["VK_KHR_buffer_device_address"],
        _ => &[],
    }
}

pub fn write<W: Write>(writer: &mut W, extensions: &IndexMap<&str, &Extension>) {
    write_device_extensions(writer, make_vulkano_extensions("device", &extensions));
    write!(writer, "\n\n").unwrap();
    write_instance_extensions(writer, make_vulkano_extensions("instance", &extensions));
}

#[derive(Clone, Debug)]
struct VulkanoExtension {
    member: String,
    raw: String,
    requires_core: (u16, u16),
    requires_device_extensions: Vec<String>,
    requires_instance_extensions: Vec<String>,
    required_if_supported: bool,
    conflicts_device_extensions: Vec<String>,
    status: Option<ExtensionStatus>,
}

#[derive(Clone, Debug)]
enum Replacement {
    Core((u16, u16)),
    DeviceExtension(String),
    InstanceExtension(String),
}

#[derive(Clone, Debug)]
enum ExtensionStatus {
    Promoted(Replacement),
    Deprecated(Option<Replacement>),
}

fn make_vulkano_extensions(
    ty: &str,
    extensions: &IndexMap<&str, &Extension>,
) -> Vec<VulkanoExtension> {
    extensions
        .values()
        .filter(|ext| ext.ext_type.as_ref().unwrap() == ty)
        .map(|ext| {
            let raw = ext.name.to_owned();
            let member = raw.strip_prefix("VK_").unwrap().to_snake_case();
            let (major, minor) = ext
                .requires_core
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("1.0")
                .split_once('.')
                .unwrap();
            let requires_extensions: Vec<_> = ext
                .requires
                .as_ref()
                .map(|s| s.split(',').collect())
                .unwrap_or_default();
            let conflicts_extensions = conflicts_extensions(&ext.name);

            VulkanoExtension {
                member: member.clone(),
                raw,
                requires_core: (major.parse().unwrap(), minor.parse().unwrap()),
                requires_device_extensions: requires_extensions
                    .iter()
                    .filter(|&&vk_name| extensions[vk_name].ext_type.as_ref().unwrap() == "device")
                    .map(|vk_name| vk_name.strip_prefix("VK_").unwrap().to_snake_case())
                    .collect(),
                requires_instance_extensions: requires_extensions
                    .iter()
                    .filter(|&&vk_name| {
                        extensions[vk_name].ext_type.as_ref().unwrap() == "instance"
                    })
                    .map(|vk_name| vk_name.strip_prefix("VK_").unwrap().to_snake_case())
                    .collect(),
                required_if_supported: required_if_supported(ext.name.as_str()),
                conflicts_device_extensions: conflicts_extensions
                    .iter()
                    .filter(|&&vk_name| extensions[vk_name].ext_type.as_ref().unwrap() == "device")
                    .map(|vk_name| vk_name.strip_prefix("VK_").unwrap().to_snake_case())
                    .collect(),
                status: ext
                    .promotedto
                    .as_ref()
                    .map(|s| s.as_str())
                    .and_then(|pr| {
                        if let Some(version) = pr.strip_prefix("VK_VERSION_") {
                            let (major, minor) = version.split_once('_').unwrap();
                            Some(ExtensionStatus::Promoted(Replacement::Core((
                                major.parse().unwrap(),
                                minor.parse().unwrap(),
                            ))))
                        } else {
                            let member = pr.strip_prefix("VK_").unwrap().to_snake_case();
                            match extensions[pr].ext_type.as_ref().unwrap().as_str() {
                                "device" => Some(ExtensionStatus::Promoted(
                                    Replacement::DeviceExtension(member),
                                )),
                                "instance" => Some(ExtensionStatus::Promoted(
                                    Replacement::InstanceExtension(member),
                                )),
                                _ => unreachable!(),
                            }
                        }
                    })
                    .or_else(|| {
                        ext.deprecatedby
                            .as_ref()
                            .map(|s| s.as_str())
                            .and_then(|depr| {
                                if depr.is_empty() {
                                    Some(ExtensionStatus::Deprecated(None))
                                } else if let Some(version) = depr.strip_prefix("VK_VERSION_") {
                                    let (major, minor) = version.split_once('_').unwrap();
                                    Some(ExtensionStatus::Deprecated(Some(Replacement::Core((
                                        major.parse().unwrap(),
                                        minor.parse().unwrap(),
                                    )))))
                                } else {
                                    let member = depr.strip_prefix("VK_").unwrap().to_snake_case();
                                    match extensions[depr].ext_type.as_ref().unwrap().as_str() {
                                        "device" => Some(ExtensionStatus::Deprecated(Some(
                                            Replacement::DeviceExtension(member),
                                        ))),
                                        "instance" => Some(ExtensionStatus::Deprecated(Some(
                                            Replacement::InstanceExtension(member),
                                        ))),
                                        _ => unreachable!(),
                                    }
                                }
                            })
                    }),
            }
        })
        .collect()
}

fn write_device_extensions<W, I>(writer: &mut W, extensions: I)
where
    W: Write,
    I: IntoIterator<Item = VulkanoExtension>,
{
    write!(writer, "crate::device::extensions::device_extensions! {{").unwrap();
    for ext in extensions {
        write!(writer, "\n\t{} => {{", ext.member).unwrap();
        write_doc(writer, &ext);
        write!(writer, "\n\t\traw: b\"{}\",", ext.raw).unwrap();
        write!(
            writer,
            "\n\t\trequires_core: crate::Version::V{}_{},",
            ext.requires_core.0, ext.requires_core.1
        )
        .unwrap();
        write!(
            writer,
            "\n\t\trequires_device_extensions: [{}],",
            ext.requires_device_extensions.join(", ")
        )
        .unwrap();
        write!(
            writer,
            "\n\t\trequires_instance_extensions: [{}],",
            ext.requires_instance_extensions.join(", ")
        )
        .unwrap();
        write!(
            writer,
            "\n\t\trequired_if_supported: {},",
            ext.required_if_supported
        )
        .unwrap();
        write!(
            writer,
            "\n\t\tconflicts_device_extensions: [{}],",
            ext.conflicts_device_extensions.join(", ")
        )
        .unwrap();

        /*if let Some(promoted_to_core) = ext.promoted_to_core {
            write!(
                writer,
                "\n\t\tpromoted_to_core: Some(Version::V{}_{}),",
                promoted_to_core.0, promoted_to_core.1
            )
            .unwrap();
        } else {
            write!(writer, "\n\t\tpromoted_to_core: None,",).unwrap();
        }*/

        write!(writer, "\n\t}},").unwrap();
    }
    write!(writer, "\n}}").unwrap();
}

fn write_instance_extensions<W, I>(writer: &mut W, extensions: I)
where
    W: Write,
    I: IntoIterator<Item = VulkanoExtension>,
{
    write!(
        writer,
        "crate::instance::extensions::instance_extensions! {{"
    )
    .unwrap();
    for ext in extensions {
        write!(writer, "\n\t{} => {{", ext.member).unwrap();
        write_doc(writer, &ext);
        write!(writer, "\n\t\traw: b\"{}\",", ext.raw).unwrap();
        write!(
            writer,
            "\n\t\trequires_core: crate::Version::V{}_{},",
            ext.requires_core.0, ext.requires_core.1
        )
        .unwrap();
        write!(
            writer,
            "\n\t\trequires_extensions: [{}],",
            ext.requires_instance_extensions.join(", ")
        )
        .unwrap();

        /*if let Some(promoted_to_core) = ext.promoted_to_core {
            write!(
                writer,
                "\n\t\tpromoted_to_core: Some(crate::Version::V{}_{}),",
                promoted_to_core.0, promoted_to_core.1
            )
            .unwrap();
        } else {
            write!(writer, "\n\t\tpromoted_to_core: None,",).unwrap();
        }*/

        write!(writer, "\n\t}},").unwrap();
    }
    write!(writer, "\n}}").unwrap();
}

fn write_doc<W>(writer: &mut W, ext: &VulkanoExtension)
where
    W: Write,
{
    write!(writer, "\n\t\tdoc: \"\n\t\t\t- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/{}.html)", ext.raw).unwrap();

    if ext.requires_core != (1, 0) {
        write!(
            writer,
            "\n\t\t\t- Requires Vulkan {}.{}",
            ext.requires_core.0, ext.requires_core.1
        )
        .unwrap();
    }

    if !ext.requires_device_extensions.is_empty() {
        let links: Vec<_> = ext
            .requires_device_extensions
            .iter()
            .map(|ext| format!("[`{}`](crate::device::DeviceExtensions::{0})", ext))
            .collect();
        write!(
            writer,
            "\n\t\t\t- Requires device extension{}: {}",
            if ext.requires_device_extensions.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }

    if !ext.requires_instance_extensions.is_empty() {
        let links: Vec<_> = ext
            .requires_instance_extensions
            .iter()
            .map(|ext| format!("[`{}`](crate::instance::InstanceExtensions::{0})", ext))
            .collect();
        write!(
            writer,
            "\n\t\t\t- Requires instance extension{}: {}",
            if ext.requires_instance_extensions.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }

    if ext.required_if_supported {
        write!(
            writer,
            "\n\t\t\t- Must be enabled if it is supported by the physical device",
        )
        .unwrap();
    }

    if !ext.conflicts_device_extensions.is_empty() {
        let links: Vec<_> = ext
            .conflicts_device_extensions
            .iter()
            .map(|ext| format!("[`{}`](crate::device::DeviceExtensions::{0})", ext))
            .collect();
        write!(
            writer,
            "\n\t\t\t- Conflicts with device extension{}: {}",
            if ext.conflicts_device_extensions.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }

    if let Some(status) = ext.status.as_ref() {
        match status {
            ExtensionStatus::Promoted(replacement) => {
                write!(writer, "\n\t\t\t- Promoted to ",).unwrap();

                match replacement {
                    Replacement::Core(version) => {
                        write!(writer, "Vulkan {}.{}", version.0, version.1).unwrap();
                    }
                    Replacement::DeviceExtension(ext) => {
                        write!(writer, "[`{}`](crate::device::DeviceExtensions::{0})", ext)
                            .unwrap();
                    }
                    Replacement::InstanceExtension(ext) => {
                        write!(
                            writer,
                            "[`{}`](crate::instance::InstanceExtensions::{0})",
                            ext
                        )
                        .unwrap();
                    }
                }
            }
            ExtensionStatus::Deprecated(replacement) => {
                write!(writer, "\n\t\t\t- Deprecated ",).unwrap();

                match replacement {
                    Some(Replacement::Core(version)) => {
                        write!(writer, "by Vulkan {}.{}", version.0, version.1).unwrap();
                    }
                    Some(Replacement::DeviceExtension(ext)) => {
                        write!(
                            writer,
                            "by [`{}`](crate::device::DeviceExtensions::{0})",
                            ext
                        )
                        .unwrap();
                    }
                    Some(Replacement::InstanceExtension(ext)) => {
                        write!(
                            writer,
                            "by [`{}`](crate::instance::InstanceExtensions::{0})",
                            ext
                        )
                        .unwrap();
                    }
                    None => {
                        write!(writer, "without a replacement").unwrap();
                    }
                }
            }
        }
    }

    write!(writer, "\n\t\t\",").unwrap();
}
