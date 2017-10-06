extern crate glsl_to_spirv;
extern crate proc_macro;
extern crate syn;
extern crate vulkano_shaders;

use std::path::Path;
use std::fs::File;
use std::io::Read;

use proc_macro::TokenStream;

enum SourceKind {
    Src(String),
    Path(String),
}

#[proc_macro_derive(VulkanoShader, attributes(src, path, ty))]
pub fn derive(input: TokenStream) -> TokenStream {
    let syn_item = syn::parse_macro_input(&input.to_string()).unwrap();

    let source_code = {
        let mut iter = syn_item.attrs.iter().filter_map(|attr| {
            match attr.value {
                syn::MetaItem::NameValue(ref i, syn::Lit::Str(ref val, _)) if i == "src" => {
                    Some(SourceKind::Src(val.clone()))
                },

                syn::MetaItem::NameValue(ref i, syn::Lit::Str(ref val, _)) if i == "path" => {
                    Some(SourceKind::Path(val.clone()))
                },

                _ => None
            }
        });

        let source = iter.next().expect("No source attribute given ; put #[src = \"...\"] or #[path = \"...\"]");

        if iter.next().is_some() {
            panic!("Multiple src or path attributes given ; please provide only one");
        }

        match source {
            SourceKind::Src(source) => source,

            SourceKind::Path(path) => {
                let root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or(".".into());
                let full_path = Path::new(&root).join(&path);

                if full_path.is_file() {
                    let mut buf = String::new();
                    File::open(full_path)
                        .and_then(|mut file| file.read_to_string(&mut buf))
                        .expect(&format!("Error reading source from {:?}", path));
                    buf
                } else {
                    panic!("File {:?} was not found ; note that the path must be relative to your Cargo.toml", path);
                }
            }
        }
    };

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

    let spirv_data = match glsl_to_spirv::compile(&source_code, ty) {
        Ok(compiled) => compiled,
        Err(message) => panic!("{}\nfailed to compile shader", message),
    };

    vulkano_shaders::reflect("Shader", spirv_data).unwrap().parse().unwrap()
}
