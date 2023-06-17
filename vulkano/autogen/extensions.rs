// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, IndexMap, VkRegistryData};
use heck::ToSnakeCase;
use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote};
use std::fmt::Write as _;
use vk_parse::Extension;

// This is not included in vk.xml, so it's added here manually
fn required_if_supported(name: &str) -> bool {
    #[allow(clippy::match_like_matches_macro)]
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
    requires: Vec<RequiresOneOf>,
    conflicts_device_extensions: Vec<Ident>,
    status: Option<ExtensionStatus>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequiresOneOf {
    pub api_version: Option<(String, String)>,
    pub device_extensions: Vec<Ident>,
    pub instance_extensions: Vec<Ident>,
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
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        device_extensions_output(&extensions_members("device", &vk_data.extensions)),
    );
}

fn write_instance_extensions(vk_data: &VkRegistryData) {
    write_file(
        "instance_extensions.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        instance_extensions_output(&extensions_members("instance", &vk_data.extensions)),
    );
}

fn device_extensions_output(members: &[ExtensionsMember]) -> TokenStream {
    let common = extensions_common_output(format_ident!("DeviceExtensions"), members);

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
                            quote! { api_version >= crate::Version::#version }
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
                            quote! { Some(crate::Version::#version) }
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
                            return Err(crate::ValidationError {
                                problem: format!("contains `{}`", #name_string).into(),
                                requires_one_of: crate::RequiresOneOf {
                                    api_version: #api_version_items,
                                    device_extensions: &[#(#device_extensions_items),*],
                                    instance_extensions: &[#(#instance_extensions_items),*],
                                    ..Default::default()
                                },
                                ..Default::default()
                            });
                        }
                    }
                });

                quote! {
                    if self.#name {
                        if !supported.#name {
                            return Err(crate::ValidationError {
                                problem: format!(
                                    "contains `{}`, but this extension is not supported \
                                    by the physical device",
                                    #name_string,
                                ).into(),
                                ..Default::default()
                            });
                        }

                        #(#requires_items)*
                    }
                }
            });

    quote! {
        #common

        impl DeviceExtensions {
            /// Checks enabled extensions against the device version, instance extensions and each other.
            pub(super) fn check_requirements(
                &self,
                supported: &DeviceExtensions,
                api_version: crate::Version,
                instance_extensions: &crate::instance::InstanceExtensions,
            ) -> Result<(), crate::ValidationError> {
                let device_extensions = self;
                #(#check_requirements_items)*
                Ok(())
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
                            quote! { api_version >= crate::Version::#version }
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
                            quote! { Some(crate::Version::#version) }
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
                            return Err(crate::ValidationError {
                                problem: format!("contains `{}`", #name_string).into(),
                                requires_one_of: crate::RequiresOneOf {
                                    api_version: #api_version_items,
                                    device_extensions: &[#(#device_extensions_items),*],
                                    instance_extensions: &[#(#instance_extensions_items),*],
                                    ..Default::default()
                                },
                                ..Default::default()
                            });
                        }
                    }
                });

                quote! {
                    if self.#name {
                        if !supported.#name {
                            return Err(crate::ValidationError {
                                problem: format!(
                                    "contains `{}`, but this extension is not supported \
                                    by the library",
                                    #name_string,
                                )
                                .into(),
                                ..Default::default()
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
                api_version: crate::Version,
            ) -> Result<(), crate::ValidationError> {
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

    let empty_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            #name: false,
        }
    });

    let intersects_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            (self.#name && other.#name)
        }
    });

    let contains_items = members.iter().map(|ExtensionsMember { name, .. }| {
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

    let symmetric_difference_items = members.iter().map(|ExtensionsMember { name, .. }| {
        quote! {
            #name: self.#name ^ other.#name,
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

    let arr_items = members.iter().map(|ExtensionsMember { name, raw, .. }| {
        quote! {
            (#raw, self.#name),
        }
    });
    let arr_len = members.len();

    let from_str_for_extensions_items =
        members.iter().map(|ExtensionsMember { name, raw, .. }| {
            let raw = Literal::string(raw);
            quote! {
                #raw => { extensions.#name = true; }
            }
        });

    let from_extensions_for_vec_cstring_items =
        members.iter().map(|ExtensionsMember { name, raw, .. }| {
            quote! {
                if x.#name { data.push(std::ffi::CString::new(#raw).unwrap()); }
            }
        });

    quote! {
        /// List of extensions that are enabled or available.
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct #struct_name {
            #(#struct_items)*

            pub _ne: crate::NonExhaustive,
        }

        impl Default for #struct_name {
            #[inline]
            fn default() -> Self {
                Self::empty()
            }
        }

        impl #struct_name {
            /// Returns an `Extensions` object with none of the members set.
            #[inline]
            pub const fn empty() -> Self {
                Self {
                    #(#empty_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }

            /// Returns an `Extensions` object with none of the members set.
            #[deprecated(since = "0.31.0", note = "Use `empty` instead.")]
            #[inline]
            pub const fn none() -> Self {
                Self::empty()
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

        impl std::ops::BitAnd for #struct_name {
            type Output = #struct_name;

            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                self.union(&rhs)
            }
        }

        impl std::ops::BitAndAssign for #struct_name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.union(&rhs);
            }
        }

        impl std::ops::BitOr for #struct_name {
            type Output = #struct_name;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                self.intersection(&rhs)
            }
        }

        impl std::ops::BitOrAssign for #struct_name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.intersection(&rhs);
            }
        }

        impl std::ops::BitXor for #struct_name {
            type Output = #struct_name;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                self.symmetric_difference(&rhs)
            }
        }

        impl std::ops::BitXorAssign for #struct_name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(&rhs);
            }
        }

        impl std::ops::Sub for #struct_name {
            type Output = #struct_name;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                self.difference(&rhs)
            }
        }

        impl std::ops::SubAssign for #struct_name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(&rhs);
            }
        }

        impl std::fmt::Debug for #struct_name {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                write!(f, "[")?;

                let mut first = true;
                #(#debug_items)*

                write!(f, "]")
            }
        }

        impl<'a> FromIterator<&'a str> for #struct_name {
            fn from_iter<I>(iter: I) -> Self
                where I: IntoIterator<Item = &'a str>
            {
                let mut extensions = Self::empty();
                for name in iter {
                    match name {
                        #(#from_str_for_extensions_items)*
                        _ => (),
                    }
                }
                extensions
            }
        }

        impl<'a> From<&'a #struct_name> for Vec<std::ffi::CString> {
            fn from(x: &'a #struct_name) -> Self {
                let mut data = Self::new();
                #(#from_extensions_for_vec_cstring_items)*
                data
            }
        }

        impl IntoIterator for #struct_name {
            type Item = (&'static str, bool);
            type IntoIter = std::array::IntoIter<Self::Item, #arr_len>;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                [#(#arr_items)*].into_iter()
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
                requires.push(RequiresOneOf {
                    api_version: Some((major.to_owned(), minor.to_owned())),
                    ..Default::default()
                });
            }

            if let Some(req) = ext.requires.as_ref() {
                requires.extend(req.split(',').map(|mut vk_name| {
                    let mut dependencies = RequiresOneOf::default();

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

                            match extension.ext_type.as_deref() {
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
                    .as_deref()
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
                        ext.deprecatedby.as_deref().and_then(|depr| {
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
                                        Replacement::DeviceExtension(format_ident!("{}", member)),
                                    ))),
                                    "instance" => Some(ExtensionStatus::Deprecated(Some(
                                        Replacement::InstanceExtension(format_ident!("{}", member)),
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
    write!(writer, "- [Vulkan documentation](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/{}.html)", ext.raw).unwrap();

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
