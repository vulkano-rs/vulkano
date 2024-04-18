use super::{write_file, IndexMap, VkRegistryData};
use ahash::{HashMap, HashSet};
use heck::ToSnakeCase;
use nom::{
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{self, digit1},
    combinator::{all_consuming, eof},
    sequence::{delimited, tuple},
    IResult,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use std::{collections::hash_map::Entry, fmt::Write as _};
use vk_parse::{Extension, Type, TypeMember, TypeMemberDefinition, TypeMemberMarkup, TypeSpec};

pub fn write(vk_data: &VkRegistryData<'_>) {
    let properties_output = properties_output(&properties_members(&vk_data.types));
    let properties_ffi_output =
        properties_ffi_output(&properties_ffi_members(&vk_data.types, &vk_data.extensions));
    write_file(
        "properties.rs",
        format!(
            "vk.xml header version {}.{}.{}",
            vk_data.header_version.0, vk_data.header_version.1, vk_data.header_version.2
        ),
        quote! {
            #properties_output
            #properties_ffi_output
        },
    );
}

#[derive(Clone, Debug)]
struct PropertiesMember {
    name: Ident,
    ty: TokenStream,
    doc: String,
    raw: String,
    ffi_name: Ident,
    ffi_members: Vec<FFIMember>,
    optional: bool,
}

#[derive(Debug, Clone)]
struct FFIMember {
    ident: Ident,
    tokens: TokenStream,
    len_field_name: Option<String>,
}

fn properties_output(members: &[PropertiesMember]) -> TokenStream {
    let struct_items = members.iter().map(
        |PropertiesMember {
             name,
             ty,
             doc,
             optional,
             ..
         }| {
            if *optional {
                quote! {
                    #[doc = #doc]
                    pub #name: Option<#ty>,
                }
            } else {
                quote! {
                    #[doc = #doc]
                    pub #name: #ty,
                }
            }
        },
    );

    let default_items = members.iter().map(|PropertiesMember { name, .. }| {
        quote! {
            #name: Default::default(),
        }
    });

    let from_items = members.iter().map(
        |PropertiesMember {
             name,
             ty,
             ffi_name,
             ffi_members,
             optional,
             ..
         }| {
            if *optional {
                let ffi_members = ffi_members.iter().map(|FFIMember { ident: ffi_member, tokens: ffi_member_field, len_field_name }| {
                    if let Some(len_field_name) = len_field_name {
                        let len_field_name = Ident::new(len_field_name.as_str(), Span::call_site());

                        quote! {
                            properties_ffi.#ffi_member.and_then(|s| {
                                let ptr = s #ffi_member_field .#ffi_name .cast_const();
                                if ptr == std::ptr::null() {
                                    return None;
                                };

                                Some(unsafe {
                                    std::slice::from_raw_parts(
                                        ptr,
                                        s #ffi_member_field .#len_field_name as _,
                                    )
                                })
                            })
                        }
                    } else {
                        quote! { properties_ffi.#ffi_member.map(|s| s #ffi_member_field .#ffi_name) }
                    }
                });

                quote! {
                    #name: [
                        #(#ffi_members),*
                    ].into_iter().flatten().next().and_then(<#ty>::from_vulkan),
                }
            } else {
                let ffi_members = ffi_members.iter().map(|FFIMember { ident: ffi_member, tokens: ffi_member_field, len_field_name }| {
                    assert_eq!(*len_field_name, None);
                    quote! { properties_ffi.#ffi_member #ffi_member_field .#ffi_name }
                });

                quote! {
                    #name: [
                        #(#ffi_members),*
                    ].into_iter().next().and_then(<#ty>::from_vulkan).unwrap(),
                }
            }
        },
    );

    quote! {
        /// Represents all the properties of a physical device.
        ///
        /// Depending on the highest version of Vulkan supported by the physical device, and the
        /// available extensions, not every property may be available. For that reason, some
        /// properties are wrapped in an `Option`.
        #[derive(Clone, Debug)]
        pub struct DeviceProperties {
            #(#struct_items)*
            pub _ne: crate::NonExhaustive,
        }

        impl Default for DeviceProperties {
            fn default() -> Self {
                DeviceProperties {
                    #(#default_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }
        }

        impl From<&DevicePropertiesFfi> for DeviceProperties {
            fn from(properties_ffi: &DevicePropertiesFfi) -> Self {
                DeviceProperties {
                    #(#from_items)*
                    _ne: crate::NonExhaustive(()),
                }
            }
        }
    }
}

fn properties_members(types: &HashMap<&str, (&Type, Vec<&str>)>) -> Vec<PropertiesMember> {
    let mut properties = HashMap::default();

    [
        &types["VkPhysicalDeviceProperties"],
        &types["VkPhysicalDeviceLimits"],
        &types["VkPhysicalDeviceSparseProperties"],
    ]
    .into_iter()
    .chain(sorted_structs(types))
    .filter(|(ty, _)| {
        let name = ty.name.as_deref();
        name == Some("VkPhysicalDeviceProperties")
            || name == Some("VkPhysicalDeviceLimits")
            || name == Some("VkPhysicalDeviceSparseProperties")
            || ty.structextends.as_deref() == Some("VkPhysicalDeviceProperties2")
    })
    .for_each(|(ty, _)| {
        let vulkan_ty_name = ty.name.as_ref().unwrap();

        let (ty_name, optional) = if vulkan_ty_name == "VkPhysicalDeviceProperties" {
            (
                (format_ident!("properties_vulkan10"), quote! { .properties }),
                false,
            )
        } else if vulkan_ty_name == "VkPhysicalDeviceLimits" {
            (
                (
                    format_ident!("properties_vulkan10"),
                    quote! { .properties.limits },
                ),
                false,
            )
        } else if vulkan_ty_name == "VkPhysicalDeviceSparseProperties" {
            (
                (
                    format_ident!("properties_vulkan10"),
                    quote! { .properties.sparse_properties },
                ),
                false,
            )
        } else {
            (
                (format_ident!("{}", ffi_member(vulkan_ty_name)), quote! {}),
                true,
            )
        };

        members(ty)
            .into_iter()
            .for_each(|Member { name, ty, len }| {
                if ty == "VkPhysicalDeviceLimits" || ty == "VkPhysicalDeviceSparseProperties" {
                    return;
                }

                let ffi_name = name.to_snake_case();
                let vulkano_ty = match name {
                    "apiVersion" => quote! { Version },
                    "bufferImageGranularity"
                    | "minStorageBufferOffsetAlignment"
                    | "minTexelBufferOffsetAlignment"
                    | "minUniformBufferOffsetAlignment"
                    | "nonCoherentAtomSize"
                    | "optimalBufferCopyOffsetAlignment"
                    | "optimalBufferCopyRowPitchAlignment"
                    | "robustStorageBufferAccessSizeAlignment"
                    | "robustUniformBufferAccessSizeAlignment"
                    | "storageTexelBufferOffsetAlignmentBytes"
                    | "uniformTexelBufferOffsetAlignmentBytes" => {
                        quote! { DeviceAlignment }
                    }
                    _ => vulkano_type(ty, len),
                };

                let len_field_name = len.and_then(|it| match it {
                    LenKind::Field(it) => Some(it.to_snake_case()),
                    _ => None,
                });

                let vulkano_member = if len_field_name.is_some() {
                    ffi_name
                        .strip_prefix("p_")
                        .map(|it| it.to_string())
                        .unwrap()
                } else {
                    ffi_name.clone()
                };

                let ffi_member = FFIMember {
                    ident: ty_name.0.clone(),
                    tokens: ty_name.1.clone(),
                    len_field_name,
                };

                match properties.entry(ffi_name.clone()) {
                    Entry::Vacant(entry) => {
                        let mut member = PropertiesMember {
                            name: format_ident!("{}", vulkano_member),
                            ty: vulkano_ty,
                            doc: String::new(),
                            raw: name.to_owned(),
                            ffi_name: format_ident!("{}", ffi_name),
                            ffi_members: vec![ffi_member],
                            optional,
                        };
                        make_doc(&mut member, vulkan_ty_name);
                        entry.insert(member);
                    }
                    Entry::Occupied(entry) => {
                        entry.into_mut().ffi_members.push(ffi_member);
                    }
                };
            });
    });

    let mut ffi_names: Vec<_> = properties
        .values()
        .map(|prop| prop.ffi_name.to_string())
        .collect();
    ffi_names.sort_unstable();

    let to_remove = properties
        .values()
        .flat_map(|value| {
            value
                .ffi_members
                .iter()
                .flat_map(|ffi_member| ffi_member.len_field_name.clone())
        })
        .collect::<HashSet<String>>();

    let ffi_names = ffi_names
        .iter()
        .filter(|it| !to_remove.contains(it.as_str()));

    ffi_names
        .into_iter()
        .map(|name| properties.remove(name).unwrap())
        .collect()
}

fn make_doc(prop: &mut PropertiesMember, vulkan_ty_name: &str) {
    let writer = &mut prop.doc;
    write!(
        writer,
        "- [Vulkan documentation](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/{}.html#limits-{})",
        vulkan_ty_name,
        prop.raw
    )
        .unwrap();
}

#[derive(Clone, Debug)]
struct PropertiesFfiMember {
    name: Ident,
    ty: Ident,
    provided_by: Vec<TokenStream>,
    conflicts: Vec<Ident>,
}

fn properties_ffi_output(members: &[PropertiesFfiMember]) -> TokenStream {
    let struct_items = members.iter().map(|PropertiesFfiMember { name, ty, .. }| {
        quote! { #name: Option<ash::vk::#ty<'static>>, }
    });

    let make_chain_items = members.iter().map(
        |PropertiesFfiMember {
             name,
             provided_by,
             conflicts,
             ..
         }| {
            quote! {
                if [#(#provided_by),*].into_iter().any(|x| x) &&
                    [#(self.#conflicts.is_none()),*].into_iter().all(|x| x) {
                    self.#name = Some(Default::default());
                    let member = self.#name.as_mut().unwrap();
                    member.p_next = head.p_next;
                    head.p_next = <*mut _>::cast(member);
                }
            }
        },
    );

    quote! {
        #[derive(Default)]
        pub(crate) struct DevicePropertiesFfi {
            properties_vulkan10: ash::vk::PhysicalDeviceProperties2KHR<'static>,
            #(#struct_items)*
        }

        impl DevicePropertiesFfi {
            pub(crate) fn make_chain(
                &mut self,
                api_version: Version,
                device_extensions: &DeviceExtensions,
                instance_extensions: &InstanceExtensions,
            ) {
                self.properties_vulkan10 = Default::default();
                let head = &mut self.properties_vulkan10;
                #(#make_chain_items)*
            }

            pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceProperties2KHR<'static> {
                &mut self.properties_vulkan10
            }
        }
    }
}

fn properties_ffi_members<'a>(
    types: &'a HashMap<&str, (&Type, Vec<&str>)>,
    extensions: &IndexMap<&'a str, &Extension>,
) -> Vec<PropertiesFfiMember> {
    let mut property_included_in: HashMap<&str, Vec<&str>> = HashMap::default();
    sorted_structs(types)
        .into_iter()
        .map(|(ty, provided_by)| {
            let ty_name = ty.name.as_ref().unwrap();
            let provided_by = provided_by
                .iter()
                .map(|provided_by| {
                    if let Some(version) = provided_by.strip_prefix("VK_VERSION_") {
                        let version = format_ident!("V{}", version);
                        quote! { api_version >= Version::#version }
                    } else {
                        let member = format_ident!(
                            "{}_extensions",
                            extensions[provided_by].ext_type.as_ref().unwrap().as_str()
                        );
                        let name = format_ident!(
                            "{}",
                            provided_by
                                .strip_prefix("VK_")
                                .unwrap()
                                .to_ascii_lowercase(),
                        );

                        quote! { #member.#name }
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

            PropertiesFfiMember {
                name: format_ident!("{}", ffi_member(ty_name)),
                ty: format_ident!("{}", ty_name.strip_prefix("Vk").unwrap()),
                provided_by,
                conflicts: conflicts
                    .into_iter()
                    .map(|s| format_ident!("{}", s))
                    .collect(),
            }
        })
        .collect()
}

fn sorted_structs<'a>(
    types: &'a HashMap<&str, (&'a Type, Vec<&'a str>)>,
) -> Vec<&'a (&'a Type, Vec<&'a str>)> {
    let mut structs: Vec<_> = types
        .values()
        .filter(|(ty, _)| ty.structextends.as_deref() == Some("VkPhysicalDeviceProperties2"))
        .collect();

    fn is_physical_device_properties(name: &str) -> bool {
        tuple((
            tag::<_, &str, ()>("VkPhysicalDeviceVulkan"),
            digit1,
            tag("Properties"),
            eof,
        ))(name)
        .is_ok()
    }

    structs.sort_unstable_by_key(|&(ty, provided_by)| {
        let name = ty.name.as_ref().unwrap();
        (
            !is_physical_device_properties(name),
            if let Some(version) = provided_by
                .iter()
                .find_map(|s| s.strip_prefix("VK_VERSION_"))
            {
                let (major, minor) = version.split_once('_').unwrap();
                major.parse::<i32>().unwrap() << 22 | minor.parse::<i32>().unwrap() << 12
            } else if provided_by.iter().any(|s| s.starts_with("VK_KHR_")) {
                i32::MAX - 2
            } else if provided_by.iter().any(|s| s.starts_with("VK_EXT_")) {
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
    len: Option<LenKind<'a>>,
}

#[derive(Copy, Clone)]
enum LenKind<'a> {
    /// Length information in a member with this name
    Field(&'a str),
    Raw(&'a str),
}

fn members(ty: &Type) -> Vec<Member<'_>> {
    fn array_len(input: &str) -> IResult<&str, &str> {
        let (input, _) = take_until("[")(input)?;
        all_consuming(delimited(
            complete::char('['),
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
            complete::char(']'),
        ))(input)
    }

    fn type_member_name(def: &TypeMemberDefinition) -> Option<&str> {
        def.markup.iter().find_map(|markup| match markup {
            TypeMemberMarkup::Name(name) => Some(name.as_str()),
            _ => None,
        })
    }

    fn member_by_name<'a>(
        members: &'a [TypeMember],
        name: &str,
    ) -> Option<&'a TypeMemberDefinition> {
        members.iter().find_map(|it| {
            let TypeMember::Definition(defs) = it else {
                return None;
            };

            let member_name = type_member_name(defs)?;

            if member_name != name {
                return None;
            };

            Some(defs)
        })
    }

    let TypeSpec::Members(members) = &ty.spec else {
        return vec![];
    };

    members
        .iter()
        .filter_map(|member| {
            let TypeMember::Definition(def) = member else {
                return None;
            };

            let name = type_member_name(def);
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
                .or_else(|| array_len(&def.code).map(|(_, len)| len).ok())
                .map(LenKind::Raw)
                .or_else(|| {
                    let len = def.len.as_ref()?;
                    let _member = member_by_name(members.as_slice(), len)?;

                    Some(LenKind::Field(len.as_str()))
                });

            if name == Some("sType") || name == Some("pNext") {
                return None;
            }

            name.map(|name| Member {
                name,
                ty: ty.unwrap(),
                len,
            })
        })
        .collect()
}

fn vulkano_type(ty: &str, len: Option<LenKind<'_>>) -> TokenStream {
    match len {
        Some(LenKind::Raw(len)) => match ty {
            "char" => quote! { String },
            "uint8_t" if len == "VK_LUID_SIZE" => quote! { [u8; 8] },
            "uint8_t" if len == "VK_UUID_SIZE" => quote! { [u8; 16] },
            "uint32_t" if len == "2" => quote! { [u32; 2] },
            "uint32_t" if len == "3" => quote! { [u32; 3] },
            "float" if len == "2" => quote! { [f32; 2] },
            _ => unimplemented!("{}[{}]", ty, len),
        },
        Some(LenKind::Field(_)) => {
            let inner = vulkano_type(ty, None);

            quote! { Vec<#inner> }
        }
        None => match ty {
            "float" => quote! { f32 },
            "int32_t" => quote! { i32 },
            "int64_t" => quote! { i64 },
            "size_t" => quote! { usize },
            "uint8_t" => quote! { u8 },
            "uint32_t" => quote! { u32 },
            "uint64_t" => quote! { u64 },
            "VkBool32" => quote! { bool },
            "VkConformanceVersion" => quote! { ConformanceVersion },
            "VkDeviceSize" => quote! { DeviceSize },
            "VkDriverId" => quote! { DriverId },
            "VkExtent2D" => quote! { [u32; 2] },
            "VkMemoryDecompressionMethodFlagsNV" => quote! { MemoryDecompressionMethods },
            "VkOpticalFlowGridSizeFlagsNV" => quote! { OpticalFlowGridSizes },
            "VkPhysicalDeviceType" => quote! { PhysicalDeviceType },
            "VkPipelineRobustnessBufferBehaviorEXT" => quote! { PipelineRobustnessBufferBehavior },
            "VkPipelineRobustnessImageBehaviorEXT" => quote! { PipelineRobustnessImageBehavior },
            "VkPointClippingBehavior" => quote! { PointClippingBehavior },
            "VkQueueFlags" => quote! { QueueFlags },
            "VkRayTracingInvocationReorderModeNV" => quote! { RayTracingInvocationReorderMode },
            "VkResolveModeFlags" => quote! { ResolveModes },
            "VkSampleCountFlags" => quote! { SampleCounts },
            "VkSampleCountFlagBits" => quote! { SampleCount },
            "VkShaderCorePropertiesFlagsAMD" => quote! { ShaderCoreProperties },
            "VkShaderFloatControlsIndependence" => quote! { ShaderFloatControlsIndependence },
            "VkShaderStageFlags" => quote! { ShaderStages },
            "VkSubgroupFeatureFlags" => quote! { SubgroupFeatures },
            "VkImageLayout" => quote! { ImageLayout },
            "VkImageUsageFlags" => quote! { ImageUsage },
            "VkBufferUsageFlags" => quote! { BufferUsage },
            "VkChromaLocation" => quote! { ChromaLocation },
            "VkLayeredDriverUnderlyingApiMSFT" => quote! { LayeredDriverUnderlyingApi },
            "VkPhysicalDeviceSchedulingControlsFlagsARM" => {
                quote! { PhysicalDeviceSchedulingControlsFlags }
            }
            _ => unimplemented!("{}", ty),
        },
    }
}
