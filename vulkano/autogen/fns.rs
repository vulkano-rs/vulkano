// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{write_file, VkRegistryData};
use heck::{CamelCase, SnakeCase};
use indexmap::IndexMap;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use vk_parse::{Extension, ExtensionChild, InterfaceItem};

pub fn write(vk_data: &VkRegistryData) {
    let entry_fns_output = fns_output(&[], "Entry");
    let instance_fns_output = fns_output(
        &extension_fns_members("instance", &vk_data.extensions),
        "Instance",
    );
    let device_fns_output = fns_output(
        &extension_fns_members("device", &vk_data.extensions),
        "Device",
    );
    write_file(
        "fns.rs",
        format!("vk.xml header version {}", vk_data.header_version),
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

fn fns_output(extension_members: &[FnsMember], fns_level: &str) -> TokenStream {
    let struct_name = format_ident!("{}Functions", fns_level);
    let members = ["1_0", "1_1", "1_2"]
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
        pub struct #struct_name {
            #(#struct_items)*
        }

        impl #struct_name {
            pub fn load<F>(mut load_fn: F) -> #struct_name
                where F: FnMut(&CStr) -> *const c_void
            {
                #struct_name {
                    #(#load_items)*
                }
            }
        }
    }
}

fn extension_fns_members(ty: &str, extensions: &IndexMap<&str, &Extension>) -> Vec<FnsMember> {
    extensions
        .values()
        .filter(|ext| ext.ext_type.as_ref().unwrap() == ty)
        // Filter only extensions that have functions
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
            let fn_struct = format_ident!("{}Fn", base.to_camel_case());
            FnsMember { name, fn_struct }
        })
        .collect()
}
