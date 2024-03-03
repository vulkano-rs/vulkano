#[cfg(test)]
mod tests {
    use crate::{codegen::reflect, structs::TypeRegistry, MacroInput};
    use proc_macro2::Span;
    use syn::LitStr;

    fn spv_to_words(data: &[u8]) -> Vec<u32> {
        data.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn rust_gpu_reflect_vertex() {
        let insts = spv_to_words(include_bytes!("../tests/rust-gpu/test_shader-vertex.spv"));

        let mut type_registry = TypeRegistry::default();
        let (_shader_code, _structs) = reflect(
            &MacroInput::empty(),
            LitStr::new("rust-gpu vertex shader", Span::call_site()),
            String::new(),
            &insts,
            Vec::new(),
            &mut type_registry,
        )
        .expect("reflecting spv failed");
    }

    #[test]
    fn rust_gpu_reflect_fragment() {
        let insts = spv_to_words(include_bytes!("../tests/rust-gpu/test_shader-fragment.spv"));

        let mut type_registry = TypeRegistry::default();
        let (_shader_code, _structs) = reflect(
            &MacroInput::empty(),
            LitStr::new("rust-gpu vertex shader", Span::call_site()),
            String::new(),
            &insts,
            Vec::new(),
            &mut type_registry,
        )
        .expect("reflecting spv failed");
    }
}
