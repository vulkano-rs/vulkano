// Copyright (c) 2021 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use self::spirv_grammar::SpirvGrammar;
use ahash::HashMap;
use once_cell::sync::Lazy;
use regex::Regex;
use std::{
    env,
    fmt::Display,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    process::Command,
};
use vk_parse::{
    Enum, EnumSpec, Enums, EnumsChild, Extension, ExtensionChild, Feature, Format, InterfaceItem,
    Registry, RegistryChild, SpirvExtOrCap, Type, TypeSpec, TypesChild,
};

mod errors;
mod extensions;
mod features;
mod fns;
mod formats;
mod properties;
mod spirv_grammar;
mod spirv_parse;
mod spirv_reqs;
mod version;

pub type IndexMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

pub fn autogen() {
    let registry = get_vk_registry("vk.xml");
    let vk_data = VkRegistryData::new(&registry);
    let spirv_grammar = get_spirv_grammar("spirv.core.grammar.json");

    errors::write(&vk_data);
    extensions::write(&vk_data);
    features::write(&vk_data);
    formats::write(&vk_data);
    fns::write(&vk_data);
    properties::write(&vk_data);
    spirv_parse::write(&spirv_grammar);
    spirv_reqs::write(&vk_data, &spirv_grammar);
    version::write(&vk_data);
}

fn write_file(file: impl AsRef<Path>, source: impl AsRef<str>, content: impl Display) {
    let path = Path::new(&env::var_os("OUT_DIR").unwrap()).join(file.as_ref());
    let mut writer = BufWriter::new(File::create(&path).unwrap());

    write!(
        writer,
        "\
        // This file is auto-generated by vulkano autogen from {}.\n\
        // It should not be edited manually. Changes should be made by editing autogen.\n\
        \n\n{}",
        source.as_ref(),
        content,
    )
    .unwrap();

    std::mem::drop(writer); // Ensure that the file is fully written
    Command::new("rustfmt").arg(&path).status().ok();
}

fn get_vk_registry<P: AsRef<Path> + ?Sized>(path: &P) -> Registry {
    let (registry, errors) = vk_parse::parse_file(path.as_ref()).unwrap();

    if !errors.is_empty() {
        eprintln!("The following errors were found while parsing the file:");

        for error in errors {
            eprintln!("{:?}", error);
        }
    }

    registry
}

pub struct VkRegistryData<'r> {
    pub header_version: (u16, u16, u16),
    pub errors: Vec<&'r str>,
    pub extensions: IndexMap<&'r str, &'r Extension>,
    pub features: IndexMap<&'r str, &'r Feature>,
    pub formats: Vec<&'r Format>,
    pub spirv_capabilities: Vec<&'r SpirvExtOrCap>,
    pub spirv_extensions: Vec<&'r SpirvExtOrCap>,
    pub types: HashMap<&'r str, (&'r Type, Vec<&'r str>)>,
}

impl<'r> VkRegistryData<'r> {
    fn new(registry: &'r Registry) -> Self {
        let aliases = Self::get_aliases(registry);
        let extensions = Self::get_extensions(registry);
        let features = Self::get_features(registry);
        let formats = Self::get_formats(registry);
        let spirv_capabilities = Self::get_spirv_capabilities(registry);
        let spirv_extensions = Self::get_spirv_extensions(registry);
        let errors = Self::get_errors(registry, &features, &extensions);
        let types = Self::get_types(registry, &aliases, &features, &extensions);
        let header_version = Self::get_header_version(registry);

        VkRegistryData {
            header_version,
            errors,
            extensions,
            features,
            formats,
            spirv_capabilities,
            spirv_extensions,
            types,
        }
    }

    fn get_header_version(registry: &Registry) -> (u16, u16, u16) {
        static VK_HEADER_VERSION: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"#define\s+VK_HEADER_VERSION\s+(\d+)\s*$").unwrap());
        static VK_HEADER_VERSION_COMPLETE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"#define\s+VK_HEADER_VERSION_COMPLETE\s+VK_MAKE_API_VERSION\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*VK_HEADER_VERSION\s*\)").unwrap()
        });

        let mut major = None;
        let mut minor = None;
        let mut patch = None;

        for child in registry.0.iter() {
            if let RegistryChild::Types(types) = child {
                for ty in types.children.iter() {
                    if let TypesChild::Type(ty) = ty {
                        if let TypeSpec::Code(code) = &ty.spec {
                            if let Some(captures) = VK_HEADER_VERSION.captures(&code.code) {
                                patch = Some(captures.get(1).unwrap().as_str().parse().unwrap());
                            } else if let Some(captures) =
                                VK_HEADER_VERSION_COMPLETE.captures(&code.code)
                            {
                                major = Some(captures.get(2).unwrap().as_str().parse().unwrap());
                                minor = Some(captures.get(3).unwrap().as_str().parse().unwrap());
                            }
                        }
                    }
                }
            }
        }

        (major.unwrap(), minor.unwrap(), patch.unwrap())
    }

    fn get_aliases(registry: &Registry) -> HashMap<&str, &str> {
        registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::Types(types) = child {
                    return Some(types.children.iter().filter_map(|ty| {
                        if let TypesChild::Type(ty) = ty {
                            if let Some(alias) = ty.alias.as_deref() {
                                return Some((ty.name.as_ref().unwrap().as_str(), alias));
                            }
                        }
                        None
                    }));
                }
                None
            })
            .flatten()
            .collect()
    }

    fn get_errors<'a>(
        registry: &'a Registry,
        features: &IndexMap<&'a str, &'a Feature>,
        extensions: &IndexMap<&'a str, &'a Extension>,
    ) -> Vec<&'a str> {
        (registry
            .0
            .iter()
            .filter_map(|child| match child {
                RegistryChild::Enums(Enums {
                    name: Some(name),
                    children,
                    ..
                }) if name == "VkResult" => Some(children.iter().filter_map(|en| {
                    if let EnumsChild::Enum(en) = en {
                        if let EnumSpec::Value { value, .. } = &en.spec {
                            if value.starts_with('-') {
                                return Some(en.name.as_str());
                            }
                        }
                    }
                    None
                })),
                _ => None,
            })
            .flatten())
        .chain(
            (features.values().map(|feature| feature.children.iter()))
                .chain(
                    extensions
                        .values()
                        .map(|extension| extension.children.iter()),
                )
                .flatten()
                .filter_map(|child| {
                    if let ExtensionChild::Require { items, .. } = child {
                        return Some(items.iter().filter_map(|item| match item {
                            InterfaceItem::Enum(Enum {
                                name,
                                spec:
                                    EnumSpec::Offset {
                                        extends,
                                        dir: false,
                                        ..
                                    },
                                ..
                            }) if extends == "VkResult" => Some(name.as_str()),
                            _ => None,
                        }));
                    }
                    None
                })
                .flatten(),
        )
        .collect()
    }

    fn get_extensions(registry: &Registry) -> IndexMap<&str, &Extension> {
        let iter = registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::Extensions(ext) = child {
                    return Some(ext.children.iter().filter(|ext| {
                        if ext.supported.as_deref() == Some("vulkan") && ext.obsoletedby.is_none() {
                            return true;
                        }
                        false
                    }));
                }
                None
            })
            .flatten();

        let extensions: HashMap<&str, &Extension> =
            iter.clone().map(|ext| (ext.name.as_str(), ext)).collect();
        let mut names: Vec<_> = iter.map(|ext| ext.name.as_str()).collect();
        names.sort_unstable_by_key(|name| {
            if name.starts_with("VK_KHR_") {
                (0, name.to_owned())
            } else if name.starts_with("VK_EXT_") {
                (1, name.to_owned())
            } else {
                (2, name.to_owned())
            }
        });

        names.iter().map(|&name| (name, extensions[name])).collect()
    }

    fn get_features(registry: &Registry) -> IndexMap<&str, &Feature> {
        registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::Feature(feat) = child {
                    return Some((feat.name.as_str(), feat));
                }

                None
            })
            .collect()
    }

    fn get_formats(registry: &Registry) -> Vec<&Format> {
        registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::Formats(formats) = child {
                    return Some(formats.children.iter());
                }
                None
            })
            .flatten()
            .collect()
    }

    fn get_spirv_capabilities(registry: &Registry) -> Vec<&SpirvExtOrCap> {
        registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::SpirvCapabilities(capabilities) = child {
                    return Some(capabilities.children.iter());
                }
                None
            })
            .flatten()
            .collect()
    }

    fn get_spirv_extensions(registry: &Registry) -> Vec<&SpirvExtOrCap> {
        registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::SpirvExtensions(extensions) = child {
                    return Some(extensions.children.iter());
                }
                None
            })
            .flatten()
            .collect()
    }

    fn get_types<'a>(
        registry: &'a Registry,
        aliases: &HashMap<&'a str, &'a str>,
        features: &IndexMap<&'a str, &'a Feature>,
        extensions: &IndexMap<&'a str, &'a Extension>,
    ) -> HashMap<&'a str, (&'a Type, Vec<&'a str>)> {
        let mut types: HashMap<&str, (&Type, Vec<&str>)> = registry
            .0
            .iter()
            .filter_map(|child| {
                if let RegistryChild::Types(types) = child {
                    return Some(types.children.iter().filter_map(|ty| {
                        if let TypesChild::Type(ty) = ty {
                            if ty.alias.is_none() {
                                return ty.name.as_ref().map(|name| (name.as_str(), (ty, vec![])));
                            }
                        }
                        None
                    }));
                }
                None
            })
            .flatten()
            .collect();

        features
            .iter()
            .map(|(name, feature)| (name, &feature.children))
            .chain(extensions.iter().map(|(name, ext)| (name, &ext.children)))
            .for_each(|(provided_by, children)| {
                children
                    .iter()
                    .filter_map(|child| {
                        if let ExtensionChild::Require { items, .. } = child {
                            return Some(items.iter());
                        }
                        None
                    })
                    .flatten()
                    .filter_map(|item| {
                        if let InterfaceItem::Type { name, .. } = item {
                            return Some(name.as_str());
                        }
                        None
                    })
                    .for_each(|item_name| {
                        let item_name = aliases.get(item_name).unwrap_or(&item_name);
                        if let Some(ty) = types.get_mut(item_name) {
                            if !ty.1.contains(provided_by) {
                                ty.1.push(provided_by);
                            }
                        }
                    });
            });

        types
            .into_iter()
            .filter(|(_key, val)| !val.1.is_empty())
            .collect()
    }
}

pub fn get_spirv_grammar<P: AsRef<Path> + ?Sized>(path: &P) -> SpirvGrammar {
    let mut grammar = SpirvGrammar::new(path);

    // Remove duplicate opcodes and enum values, preferring "more official" suffixes
    grammar
        .instructions
        .sort_by_key(|instruction| (instruction.opcode, suffix_key(&instruction.opname)));
    grammar
        .instructions
        .dedup_by_key(|instruction| instruction.opcode);

    grammar
        .operand_kinds
        .iter_mut()
        .filter(|operand_kind| operand_kind.category == "BitEnum")
        .for_each(|operand_kind| {
            operand_kind.enumerants.sort_by_key(|enumerant| {
                let value = enumerant
                    .value
                    .as_str()
                    .unwrap()
                    .strip_prefix("0x")
                    .unwrap();
                (
                    u32::from_str_radix(value, 16).unwrap(),
                    suffix_key(&enumerant.enumerant),
                )
            });
        });

    grammar
        .operand_kinds
        .iter_mut()
        .filter(|operand_kind| operand_kind.category == "ValueEnum")
        .for_each(|operand_kind| {
            operand_kind.enumerants.sort_by_key(|enumerant| {
                (enumerant.value.as_u64(), suffix_key(&enumerant.enumerant))
            });
        });

    grammar
}

fn suffix_key(name: &str) -> u32 {
    static VENDOR_SUFFIXES: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?:AMD|GOOGLE|INTEL|NV)$").unwrap());

    #[allow(clippy::bool_to_int_with_if)]
    if VENDOR_SUFFIXES.is_match(name) {
        3
    } else if name.ends_with("EXT") {
        2
    } else if name.ends_with("KHR") {
        1
    } else {
        0
    }
}
