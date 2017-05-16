extern crate glsl_to_spirv;
extern crate proc_macro;
extern crate syn;
extern crate vulkano_shaders;

use proc_macro::TokenStream;

#[proc_macro_derive(VulkanoShader, attributes(src, ty))]
pub fn derive(input: TokenStream) -> TokenStream {
    let syn_item = syn::parse_macro_input(&input.to_string()).unwrap();

    let src = syn_item.attrs.iter().filter_map(|attr| {
        match attr.value {
            syn::MetaItem::NameValue(ref i, syn::Lit::Str(ref val, _)) if i == "src" => {
                Some(val.clone())
            },
            _ => None
        }
    }).next().expect("Can't find `src` attribute ; put #[src = \"...\"] for example.");

    let ty_str = syn_item.attrs.iter().filter_map(|attr| {
        match attr.value {
            syn::MetaItem::NameValue(ref i, syn::Lit::Str(ref val, _)) if i == "ty" => {
                Some(val.clone())
            },
            _ => None
        }
    }).next().expect("Can't find `ty` attribute ; put #[ty = \"vertex\"] for example.");

    let ty = match &ty_str[..] {
        "vertex" => glsl_to_spirv::ShaderType::Vertex,
        "fragment" => glsl_to_spirv::ShaderType::Fragment,
        "geometry" => glsl_to_spirv::ShaderType::Geometry,
        "tess_ctrl" => glsl_to_spirv::ShaderType::TessellationControl,
        "tess_eval" => glsl_to_spirv::ShaderType::TessellationEvaluation,
        "compute" => glsl_to_spirv::ShaderType::Compute,
        _ => panic!("Unexpected shader type ; valid values: vertex, fragment, geometry, tess_ctrl, tess_eval, compute")
    };

    let spirv_data = match glsl_to_spirv::compile(&src, ty) {
        Ok(compiled) => compiled,
        Err(message) => panic!("{}\nfailed to compile shader", message),
    };

    vulkano_shaders::reflect("Shader", spirv_data).unwrap().parse().unwrap()
}
