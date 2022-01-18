// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, VkRegistryData};
use heck::ToSnakeCase;
use indexmap::IndexMap;
use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote};
use std::fmt::Write as _;
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

pub fn write(vk_data: &VkRegistryData) {
    write_device_extensions(vk_data);
    write_instance_extensions(vk_data);
}

#[derive(Clone, Debug)]
struct ExtensionsMember {
    name: Ident,
    doc: String,
    raw: String,
    required_if_supported: bool,
    requires: Vec<OneOfDependencies>,
    conflicts_device_extensions: Vec<Ident>,
    status: Option<ExtensionStatus>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct OneOfDependencies {
    api_version: Option<(String, String)>,
    device_extensions: Vec<Ident>,
    instance_extensions: Vec<Ident>,
}

#[derive(Clone, Debug)]
enum Replacement {
    Core((String, String)),
    DeviceExtension(Ident),
    InstanceExtension(Ident),
}

#[derive(Clone, Debug)]
enum ExtensionStatus {
    Promoted(Replacement),
    Deprecated(Option<Replacement>),
}

fn write_device_extensions(vk_data: &VkRegistryData) {
    write_file(
        "device_extensions.rs",
        format!("vk.xml header version {}", vk_data.header_version),
        device_extensions_output(&extensions_members("device", &vk_data.extensions)),
    );
}

fn write_instance_extensions(vk_data: &VkRegistryData) {
    write_file(
        "instance_extensions.rs",
        format!("vk.xml header version {}", vk_data.header_version),
        instance_extensions_output(&extensions_members("instance", &vk_data.extensions)),
    );
}

fn device_extensions_output(members: &[ExtensionsMember]) -> TokenStream {
    let common = extensions_common_output(format_ident!("DeviceExtensions"), members);

    let check_requirements_items = members.iter().map(|ExtensionsMember {
        name,
        requires,
        conflicts_device_extensions,
        required_if_supported,
        ..
    }| {
        let name_string = name.to_string();

        let requires_items = requires.iter().map(|require| {
            let require_items = require.api_version.iter().map(|version| {
                let version = format_ident!("V{}_{}", version.0, version.1);
                quote! { api_version >= Version::#version }
            }).chain(require.instance_extensions.iter().map(|ext| {
                quote! { instance_extensions.#ext }
            })).chain(require.device_extensions.iter().map(|ext| {
                quote! { device_extensions.#ext }
            }));

            let api_version_items = require.api_version.as_ref().map(|version| {
                let version = format_ident!("V{}_{}", version.0, version.1);
                quote! { Some(Version::#version) }
            }).unwrap_or_else(|| quote!{ None });
            let device_extensions_items = require.device_extensions.iter().map(|ext| ext.to_string());
            let instance_extensions_items = require.instance_extensions.iter().map(|ext| ext.to_string());

            quote! {
                if !(#(#require_items)||*) {
                    return Err(ExtensionRestrictionError {
                        extension: #name_string,
                        restriction: ExtensionRestriction::Requires(OneOfRequirements {
                            api_version: #api_version_items,
                            device_extensions: &[#(#device_extensions_items),*],
                            instance_extensions: &[#(#instance_extensions_items),*],
                        }),
                    })
                }
            }
        });
        let conflicts_device_extensions_items = conflicts_device_extensions.iter().map(|extension| {
            let string = extension.to_string();
            quote! {
                if self.#extension {
                    return Err(ExtensionRestrictionError {
                        extension: #name_string,
                        restriction: ExtensionRestriction::ConflictsDeviceExtension(#string),
                    });
                }
            }
        });
        let required_if_supported = if *required_if_supported {
            quote! {
                if supported.#name {
                    return Err(ExtensionRestrictionError {
                        extension: #name_string,
                        restriction: ExtensionRestriction::RequiredIfSupported,
                    });
                }
            }
        } else {
            quote! {}
        };

        quote! {
            if self.#name {
                if !supported.#name {
                    return Err(ExtensionRestrictionError {
                        extension: #name_string,
                        restriction: ExtensionRestriction::NotSupported,
                    });
                }

                #(#requires_items)*
                #(#conflicts_device_extensions_items)*
            } else {
                #required_if_supported
            }
        }
    });

    let required_if_supported_extensions_items = members.iter().map(
        |ExtensionsMember {
             name,
             required_if_supported,
             ..
         }| {
            quote! {
                #name: #required_if_supported,
            }
        },
    );

    quote! {
        #common

        impl DeviceExtensions {
            /// Checks enabled extensions against the device version, instance extensions and each other.
            pub(super) fn check_requirements(
                &self,
                supported: &DeviceExtensions,
                api_version: Version,
                instance_extensions: &InstanceExtensions,
            ) -> Result<(), ExtensionRestrictionError> {
                let device_extensions = self;
                #(#check_requirements_items)*
                Ok(())
            }

            pub(crate) fn required_if_supported_extensions() -> Self {
                Self {
                    #(#required_if_supported_extensions_items)*
                    _unbuildable: Unbuildable(())
                }
            }
        }
    }
}

fn instance_extensions_output(members: &[ExtensionsMember]) -> TokenStream {
    let common = extensions_common_output(format_ident!("InstanceExtensions"), members);

    let check_requirements_items =
        members
            .iter()
            .map(|ExtensionsMember { name, requires, .. }| {
                let name_string = name.to_string();

                let requires_items = requires.iter().map(|require| {
                    let require_items = require
                        .api_version
                        .iter()
                        .map(|version| {
                            let version = format_ident!("V{}_{}", version.0, version.1);
                            quote! { api_version >= Version::#version }
                        })
                        .chain(require.instance_extensions.iter().map(|ext| {
                            quote! { instance_extensions.#ext }
                        }))
                        .chain(require.device_extensions.iter().map(|ext| {
                            quote! { device_extensions.#ext }
                        }));

                    let api_version_items = require
                        .api_version
                        .as_ref()
                        .map(|version| {
                            let version = format_ident!("V{}_{}", version.0, version.1);
                            quote! { Some(Version::#version) }
                        })
                        .unwrap_or_else(|| quote! { None });
                    let device_extensions_items =
                        require.device_extensions.iter().map(|ext| ext.to_string());
                    let instance_extensions_items = require
                        .instance_extensions
                        .iter()
                        .map(|ext| ext.to_string());

                    quote! {
                        if !(#(#require_items)||*) {
                            return Err(ExtensionRestrictionError {
                                extension: #name_string,
                                restriction: ExtensionRestriction::Requires(OneOfRequirements {
                                    api_version: #api_version_items,
                                    device_extensions: &[#(#device_extensions_items),*],
                                    instance_extensions: &[#(#instance_extensions_items),*],
                                }),
                            })
                        }
                    }
                });

                quote! {
                    if self.#name {
                        if !supported.#name {
                            return Err(ExtensionRestrictionError {
                                extension: #name_string,
                                restriction: ExtensionRestriction::NotSupported,
                            });
                        }

                        #(#requires_items)*
                    }
                }
            });

    quote! {
        #common

        impl InstanceExtensions {
            /// Checks enabled extensions against the instance version and each other.
            pub(super) fn check_requirements(
                &self,
                supported: &InstanceExtensions,
                api_version: Version,
            ) -> Result<(), ExtensionRestrictionError> {
                let instance_extensions = self;
                #(#check_requirements_items)*
                Ok(())
            }
        }
    }
}

fn extensions_common_output(struct_name: Ident, members: &[ExtensionsMember]) -> TokenStream {
    let struct_items = members.iter().map(|ExtensionsMember { name, doc, .. }| {
        quote! {
            #[doc = #doc]
            pub #name: bool,
        }
    });

    let none_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            #name: false,
        }
    });

    let is_superset_of_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            (self.#name || !other.#name)
        }
    });

    let union_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            #name: self.#name || other.#name,
        }
    });

    let intersection_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            #name: self.#name && other.#name,
        }
    });

    let difference_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            #name: self.#name && !other.#name,
        }
    });

    let debug_items = members.iter().map(|ExtensionsMember { name, raw, .. }| {
        quote! {
            if self.#name {
                if !first { write!(f, ", ")? }
                else { first = false; }
                f.write_str(#raw)?;
            }
        }
    });

    let from_cstr_for_extensions_items =
        members.iter().map(|ExtensionsMember { name, raw, .. }| {
            let raw = Literal::byte_string(raw.as_bytes());
            quote! {
                #raw => { extensions.#name = true; }
            }
        });

    let from_extensions_for_vec_cstring_items =
        members.iter().map(|ExtensionsMember { name, raw, .. }| {
            quote! {
                if x.#name { data.push(CString::new(&#raw[..]).unwrap()); }
            }
        });

    quote! {
        /// List of extensions that are enabled or available.
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct #struct_name {
            #(#struct_items)*

            /// This field ensures that an instance of this `Extensions` struct
            /// can only be created through Vulkano functions and the update
            /// syntax. This way, extensions can be added to Vulkano without
            /// breaking existing code.
            pub _unbuildable: Unbuildable,
        }

        impl #struct_name {
            /// Returns an `Extensions` object with all members set to `false`.
            #[inline]
            pub const fn none() -> Self {
                Self {
                    #(#none_items)*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each extension of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub fn is_superset_of(&self, other: &Self) -> bool {
                #(#is_superset_of_items)&&*
            }

            /// Returns the union of this list and another list.
            #[inline]
            pub const fn union(&self, other: &Self) -> Self {
                Self {
                    #(#union_items)*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the intersection of this list and another list.
            #[inline]
            pub const fn intersection(&self, other: &Self) -> Self {
                Self {
                    #(#intersection_items)*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the difference of another list from this list.
            #[inline]
            pub const fn difference(&self, other: &Self) -> Self {
                Self {
                    #(#difference_items)*
                    _unbuildable: Unbuildable(())
                }
            }
        }

        impl std::fmt::Debug for #struct_name {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                write!(f, "[")?;

                let mut first = true;
                #(#debug_items)*

                write!(f, "]")
            }
        }

        impl<'a, I> From<I> for #struct_name where I: IntoIterator<Item = &'a CStr> {
            fn from(names: I) -> Self {
                let mut extensions = Self::none();
                for name in names {
                    match name.to_bytes() {
                        #(#from_cstr_for_extensions_items)*
                        _ => (),
                    }
                }
                extensions
            }
        }

        impl<'a> From<&'a #struct_name> for Vec<CString> {
            fn from(x: &'a #struct_name) -> Self {
                let mut data = Self::new();
                #(#from_extensions_for_vec_cstring_items)*
                data
            }
        }
    }
}

fn extensions_members(ty: &str, extensions: &IndexMap<&str, &Extension>) -> Vec<ExtensionsMember> {
    extensions
        .values()
        .filter(|ext| ext.ext_type.as_ref().unwrap() == ty)
        .map(|ext| {
            let raw = ext.name.to_owned();
            let name = raw.strip_prefix("VK_").unwrap().to_snake_case();

            let mut requires = Vec::new();

            if let Some(core) = ext.requires_core.as_ref() {
                let (major, minor) = core.split_once('.').unwrap();
                requires.push(OneOfDependencies {
                    api_version: Some((major.to_owned(), minor.to_owned())),
                    ..Default::default()
                });
            }

            if let Some(req) = ext.requires.as_ref() {
                requires.extend(req.split(',').map(|mut vk_name| {
                    let mut dependencies = OneOfDependencies::default();

                    loop {
                        if let Some(version) = vk_name.strip_prefix("VK_VERSION_") {
                            let (major, minor) = version.split_once('_').unwrap();
                            dependencies.api_version = Some((major.to_owned(), minor.to_owned()));
                            break;
                        } else {
                            let ident = format_ident!(
                                "{}",
                                vk_name.strip_prefix("VK_").unwrap().to_snake_case()
                            );
                            let extension = extensions[vk_name];

                            match extension.ext_type.as_ref().map(|s| s.as_str()) {
                                Some("device") => &mut dependencies.device_extensions,
                                Some("instance") => &mut dependencies.instance_extensions,
                                _ => unreachable!(),
                            }
                            .insert(0, ident);

                            if let Some(promotedto) = extension.promotedto.as_ref() {
                                vk_name = promotedto.as_str();
                            } else {
                                break;
                            }
                        }
                    }

                    dependencies
                }));
            }

            let conflicts_extensions = conflicts_extensions(&ext.name);

            let mut member = ExtensionsMember {
                name: format_ident!("{}", name),
                doc: String::new(),
                raw,
                required_if_supported: required_if_supported(ext.name.as_str()),
                requires,
                conflicts_device_extensions: conflicts_extensions
                    .iter()
                    .filter(|&&vk_name| extensions[vk_name].ext_type.as_ref().unwrap() == "device")
                    .map(|vk_name| {
                        format_ident!("{}", vk_name.strip_prefix("VK_").unwrap().to_snake_case())
                    })
                    .collect(),
                status: ext
                    .promotedto
                    .as_ref()
                    .map(|s| s.as_str())
                    .and_then(|pr| {
                        if let Some(version) = pr.strip_prefix("VK_VERSION_") {
                            let (major, minor) = version.split_once('_').unwrap();
                            Some(ExtensionStatus::Promoted(Replacement::Core((
                                major.to_owned(),
                                minor.to_owned(),
                            ))))
                        } else {
                            let member = pr.strip_prefix("VK_").unwrap().to_snake_case();
                            match extensions[pr].ext_type.as_ref().unwrap().as_str() {
                                "device" => Some(ExtensionStatus::Promoted(
                                    Replacement::DeviceExtension(format_ident!("{}", member)),
                                )),
                                "instance" => Some(ExtensionStatus::Promoted(
                                    Replacement::InstanceExtension(format_ident!("{}", member)),
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
                                            Replacement::DeviceExtension(format_ident!(
                                                "{}", member
                                            )),
                                        ))),
                                        "instance" => Some(ExtensionStatus::Deprecated(Some(
                                            Replacement::InstanceExtension(format_ident!(
                                                "{}", member
                                            )),
                                        ))),
                                        _ => unreachable!(),
                                    }
                                }
                            })
                    }),
            };
            make_doc(&mut member);
            member
        })
        .collect()
}

fn make_doc(ext: &mut ExtensionsMember) {
    let writer = &mut ext.doc;
    write!(writer, "- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/{}.html)", ext.raw).unwrap();

    if ext.required_if_supported {
        write!(
            writer,
            "\n- Must be enabled if it is supported by the physical device",
        )
        .unwrap();
    }

    if let Some(status) = ext.status.as_ref() {
        match status {
            ExtensionStatus::Promoted(replacement) => {
                write!(writer, "\n- Promoted to ",).unwrap();

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
                write!(writer, "\n- Deprecated ",).unwrap();

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

    if !ext.requires.is_empty() {
        write!(writer, "\n- Requires:").unwrap();
    }

    for require in &ext.requires {
        let mut line = Vec::new();

        if let Some((major, minor)) = require.api_version.as_ref() {
            line.push(format!("Vulkan API version {}.{}", major, minor));
        }

        line.extend(require.device_extensions.iter().map(|ext| {
            format!(
                "device extension [`{}`](crate::device::DeviceExtensions::{0})",
                ext
            )
        }));
        line.extend(require.instance_extensions.iter().map(|ext| {
            format!(
                "instance extension [`{}`](crate::instance::InstanceExtensions::{0})",
                ext
            )
        }));

        if line.len() == 1 {
            write!(writer, "\n  - {}", line[0]).unwrap();
        } else {
            write!(writer, "\n  - One of: {}", line.join(", ")).unwrap();
        }
    }

    if !ext.conflicts_device_extensions.is_empty() {
        let links: Vec<_> = ext
            .conflicts_device_extensions
            .iter()
            .map(|ext| format!("[`{}`](crate::device::DeviceExtensions::{0})", ext))
            .collect();
        write!(
            writer,
            "\n- Conflicts with device extension{}: {}",
            if ext.conflicts_device_extensions.len() > 1 {
                "s"
            } else {
                ""
            },
            links.join(", ")
        )
        .unwrap();
    }
}
