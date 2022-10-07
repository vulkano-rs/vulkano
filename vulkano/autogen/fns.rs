// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, IndexMap, VkRegistryData};
use heck::{ToSnakeCase, ToUpperCamelCase};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use vk_parse::{Extension, ExtensionChild, InterfaceItem};

pub fn write(vk_data: &VkRegistryData) {
    let entry_fns_output = fns_output(
        &[],
        "Entry",
        "Raw Vulkan global entry point-level functions.\n\nTo use these, you need to include the Ash crate, using the same version Vulkano uses.",
    );
    let instance_fns_output = fns_output(
        &instance_extension_fns_members(&vk_data.extensions),
        "Instance",
        "Raw Vulkan instance-level functions.\n\nTo use these, you need to include the Ash crate, using the same version Vulkano uses.",
    );
    let device_fns_output = fns_output(
        &device_extension_fns_members(&vk_data.extensions),
        "Device",
        "Raw Vulkan device-level functions.\n\nTo use these, you need to include the Ash crate, using the same version Vulkano uses.",
    );
    write_file(
        "fns.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        quote! {
            #entry_fns_output
            #instance_fns_output
            #device_fns_output
        },
    );
}

#[derive(Clone, Debug)]
struct FnsMember {
    name: Ident,
    fn_struct: Ident,
}

fn fns_output(extension_members: &[FnsMember], fns_level: &str, doc: &str) -> TokenStream {
    let struct_name = format_ident!("{}Functions", fns_level);
    let members = ["1_0", "1_1", "1_2", "1_3"]
        .into_iter()
        .map(|version| FnsMember {
            name: format_ident!("v{}", version),
            fn_struct: format_ident!("{}FnV{}", fns_level, version),
        })
        .chain(extension_members.iter().cloned())
        .collect::<Vec<_>>();

    let struct_items = members.iter().map(|FnsMember { name, fn_struct }| {
        quote! { pub #name: ash::vk::#fn_struct, }
    });

    let load_items = members.iter().map(|FnsMember { name, fn_struct }| {
        quote! { #name: ash::vk::#fn_struct::load(&mut load_fn), }
    });

    quote! {
        #[doc = #doc]
        #[allow(missing_docs)]
        pub struct #struct_name {
            #(#struct_items)*
            pub _ne: crate::NonExhaustive,
        }

        impl #struct_name {
            pub(crate) fn load<F>(mut load_fn: F) -> #struct_name
                where F: FnMut(&CStr) -> *const c_void
            {
                #struct_name {
                    #(#load_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }
        }

        impl std::fmt::Debug for #struct_name {
            #[inline]
            fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                Ok(())
            }
        }
    }
}

fn device_extension_fns_members(extensions: &IndexMap<&str, &Extension>) -> Vec<FnsMember> {
    extensions
        .values()
        // Include any device extensions that have functions.
        .filter(|ext| ext.ext_type.as_ref().unwrap() == "device")
        .filter(|ext| {
            ext.children.iter().any(|ch| {
                if let ExtensionChild::Require { items, .. } = ch {
                    items
                        .iter()
                        .any(|i| matches!(i, InterfaceItem::Command { .. }))
                } else {
                    false
                }
            })
        })
        .map(|ext| {
            let base = ext.name.strip_prefix("VK_").unwrap().to_snake_case();
            let name = format_ident!("{}", base);
            let fn_struct = format_ident!("{}Fn", base.to_upper_camel_case());
            FnsMember { name, fn_struct }
        })
        .collect()
}

fn instance_extension_fns_members(extensions: &IndexMap<&str, &Extension>) -> Vec<FnsMember> {
    extensions
        .values()
        .filter(|ext| {
            match ext.ext_type.as_deref().unwrap() {
                // Include any instance extensions that have functions.
                "instance" => ext.children.iter().any(|ch| {
                    if let ExtensionChild::Require { items, .. } = ch {
                        items
                            .iter()
                            .any(|i| matches!(i, InterfaceItem::Command { .. }))
                    } else {
                        false
                    }
                }),
                // Include device extensions that have functions containing "PhysicalDevice".
                // Note: this test might not be sufficient in the long run...
                "device" => ext.children.iter().any(|ch| {
                    if let ExtensionChild::Require { items, .. } = ch {
                        items
                            .iter()
                            .any(|i| matches!(i, InterfaceItem::Command { name, .. } if name.contains("PhysicalDevice")))
                    } else {
                        false
                    }
                }),
                _ => unreachable!(),
            }
        })
        .map(|ext| {
            let base = ext.name.strip_prefix("VK_").unwrap().to_snake_case();
            let name = format_ident!("{}", base);
            let fn_struct = format_ident!("{}Fn", base.to_upper_camel_case());
            FnsMember { name, fn_struct }
        })
        .collect()
}
