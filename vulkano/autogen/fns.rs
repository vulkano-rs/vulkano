// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use heck::{CamelCase, SnakeCase};
use indexmap::IndexMap;
use std::io::Write;
use vk_parse::{Extension, ExtensionChild, InterfaceItem};

pub fn write<W: Write>(writer: &mut W, extensions: &IndexMap<&str, &Extension>) {
    write_fns(writer, std::iter::empty(), "Entry");
    write!(writer, "\n\n").unwrap();
    write_fns(
        writer,
        make_vulkano_extension_fns("instance", &extensions),
        "Instance",
    );
    write!(writer, "\n\n").unwrap();
    write_fns(
        writer,
        make_vulkano_extension_fns("device", &extensions),
        "Device",
    );
}

#[derive(Clone, Debug)]
struct VulkanoFns {
    member: String,
    fn_struct: String,
}

fn make_vulkano_extension_fns(
    ty: &str,
    extensions: &IndexMap<&str, &Extension>,
) -> Vec<VulkanoFns> {
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
            let member = ext.name.strip_prefix("VK_").unwrap().to_snake_case();
            let fn_struct = member.to_camel_case() + "Fn";
            VulkanoFns { member, fn_struct }
        })
        .collect()
}

fn write_fns<W, I>(writer: &mut W, extension_fns: I, ty: &str)
where
    W: Write,
    I: IntoIterator<Item = VulkanoFns>,
{
    write!(writer, "crate::fns::fns!({}Functions, {{", ty).unwrap();

    for version in std::array::IntoIter::new(["1_0", "1_1", "1_2"]) {
        write!(writer, "\n\tv{} => {}FnV{0},", version, ty).unwrap();
    }

    for ext in extension_fns {
        write!(writer, "\n\t{} => {},", ext.member, ext.fn_struct).unwrap();
    }
    write!(writer, "\n}});").unwrap();
}
