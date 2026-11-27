use self::{arena::Arena, structs::TypeRegistry};
use crate::{bail, CodegenContext};
use foldhash::HashMap;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::{
    borrow::Cow,
    collections::hash_map::Entry,
    env, fs,
    path::{Path, PathBuf},
    slice,
};
use syn::{
    braced, bracketed, parenthesized,
    parse::{Parse, ParseStream},
    parse_quote, Error, Ident, LitBool, LitStr, Path as SynPath, Token,
};

pub(crate) mod arena;
pub(crate) mod shaders;
pub(crate) mod structs;

pub(super) fn shader(ccx: &CodegenContext<'_>, input: TokenStream) -> Result<TokenStream> {
    let MacroInput { shaders, options } = syn::parse2(input)?;

    let global_arena = Arena::new();
    let mut state = MacroState::new(ccx, &options, &global_arena)?;

    match shaders {
        Shaders::Single(shader_fields) => {
            state.process_shader(None, shader_fields)?;
        }
        Shaders::Multiple(shaders) => {
            for (shader_ident, shader_fields) in shaders {
                let shader_name = global_arena.allocate_str_copy(&shader_ident.to_string());
                state.process_shader(Some(shader_name), shader_fields)?;
            }
        }
    }

    let result = state.finalize();

    if options.dump.value {
        println!("{}", result);
        bail!(options.dump, "`shader!` Rust codegen dumped");
    }

    Ok(result)
}

struct MacroState<'global_arena> {
    // These two have a longer lifetime than `'global_arena`, but since both lifetimes are
    // covariant and we don't need to recover the original lifetime, it doesn't make a difference.
    ccx: &'global_arena CodegenContext<'global_arena>,
    options: &'global_arena MacroOptions,
    root_path: PathBuf,
    relative_path_error_message: String,
    shaders_code: TokenStream,
    structs_code: TokenStream,
    type_registry: TypeRegistry<'global_arena>,
}

impl<'global_arena> MacroState<'global_arena> {
    fn new(
        ccx: &'global_arena CodegenContext<'global_arena>,
        options: &'global_arena MacroOptions,
        global_arena: &'global_arena Arena,
    ) -> Result<Self> {
        let (root_path, relative_path_error_message) = match options.root_path_env.as_ref() {
            None => root_path_from_call_site(),
            Some(root_path_env) => root_path_from_env_var(root_path_env),
        }?;

        Ok(MacroState {
            ccx,
            options,
            root_path,
            relative_path_error_message,
            shaders_code: TokenStream::new(),
            structs_code: TokenStream::new(),
            type_registry: TypeRegistry::new(global_arena),
        })
    }

    fn process_shader(
        &mut self,
        shader_name: Option<&'global_arena str>,
        shader_fields: ShaderFields,
    ) -> Result {
        let ShaderFields {
            shader_kind,
            source_kind,
            source_language,
            entry_point,
            compiler,
            macro_defines,
        } = shader_fields;

        let (lit, bytes, input_paths) = match source_kind.unwrap() {
            SourceKind::Compiled(compiled_source_kind) => {
                let source_path;
                let source_code;
                let working_dir;

                let lit = match compiled_source_kind {
                    CompiledSourceKind::Path(lit) => {
                        source_path = Some(self.root_path.join(lit.value()));
                        let path = source_path.as_deref().unwrap();

                        self.check_file_exists(&lit, path)?;

                        source_code = read_file_to_string(&lit, path)?;
                        working_dir = path.parent().unwrap();

                        lit
                    }
                    CompiledSourceKind::Inline(lit) => {
                        source_path = None;
                        source_code = lit.value();
                        working_dir = &self.root_path;

                        lit
                    }
                };

                let (bytes, mut input_paths) = shaders::compile_shader(
                    self.options,
                    &source_code,
                    working_dir,
                    shader_kind,
                    source_language,
                    entry_point.as_deref(),
                    compiler,
                    &macro_defines.unwrap_or_default(),
                )
                .map_err(|err| Error::new_spanned(&lit, err))?;

                if let Some(source_path) = source_path {
                    input_paths.push(source_path.into_os_string().into_string().unwrap());
                }

                (lit, bytes, input_paths)
            }
            SourceKind::Precompiled(lit) => {
                let path = self.root_path.join(lit.value());

                self.check_file_exists(&lit, &path)?;

                let bytes = read_file(&lit, &path)?;

                let input_paths = vec![path.into_os_string().into_string().unwrap()];

                (lit, bytes, input_paths)
            }
        };

        let words = spirv_bytes_to_words(&bytes)
            .map_err(|err| Error::new_spanned(&lit, format!("malformed SPIR-V: {err}")))?;

        let shaders_code = shaders::generate_shaders(self.ccx, shader_name, &words, input_paths)?;
        let structs_code = structs::generate_structs(
            self.ccx,
            self.options,
            lit,
            shader_name,
            &words,
            &mut self.type_registry,
        )?;

        self.shaders_code.extend(shaders_code);
        self.structs_code.extend(structs_code);

        Ok(())
    }

    fn check_file_exists(&self, lit: &LitStr, path: &Path) -> Result {
        if !path.is_file() {
            let msg = &self.relative_path_error_message;

            bail!(
                lit,
                "file `{path:?}` was not found; note that the path must be relative {msg}",
            );
        }

        Ok(())
    }

    fn finalize(self) -> TokenStream {
        let shaders_code = self.shaders_code;
        let structs_code = self.structs_code;

        quote! {
            #shaders_code
            #structs_code
        }
    }
}

fn root_path_from_call_site() -> Result<(PathBuf, String)> {
    let Some(file_path) = Span::call_site().unwrap().local_file() else {
        bail!(
            "failed to get the file invoking the macro from the compiler; this only happens when \
            using rust-analyzer with the file with this macro invocation open, and you can ignore \
            this error; if you can't live with it, you can use the `root_path_env: \
            \"CARGO_MANIFEST_DIR\"` option to search relative to your Cargo.toml instead",
        );
    };

    // This comes from cargo, so it better be a real directory path we have access to.
    //
    // TODO: Use a variable set by cargo for the working directory of rustc invocations once it's
    // stable such that this is truly bullet-proof.
    let working_dir = env::current_dir().expect("failed to get the current working directory");

    // The file path should be relative to the current working directory that rustc was invoked
    // with. The proc macros rustc invokes should have the same current working directory as that
    // rustc. So this should always give us the right full path.
    let full_path = working_dir
        .join(&file_path)
        .components()
        .collect::<PathBuf>();

    if !full_path.exists() {
        bail!(
            "the file invoking the macro, `{full_path:?}`, doesn't exist; you can use the \
            `root_path_env: \"CARGO_MANIFEST_DIR\"` option to search relative to your Cargo.toml \
            instead",
        );
    }

    // The file path comes from rustc, so it better be a real file path.
    let root = full_path.parent().expect(
        "the file invoking the macro we got from the compiler doesn't reside in a directory",
    );

    let error = format!("to the file `{file_path:?}` invoking the macro");

    Ok((root.into(), error))
}

fn root_path_from_env_var(root_path_env: &LitStr) -> Result<(PathBuf, String)> {
    let env = root_path_env.value();
    let root = env::var(&env).map_err(|e| {
        Error::new_spanned(
            root_path_env,
            format_args!(
                "failed to fetch the environment variable `{env}`: {e}; typical parameters are \
                `OUT_DIR` to gather results from your build script, `CARGO_MANIFEST_DIR` to \
                search relative to your Cargo.toml, or left default to search relative to the \
                file invoking the macro",
            ),
        )
    })?;
    let error = format!("to the path `{root}` specified by the env variable `{env:?}`");

    Ok((root.into(), error))
}

fn read_file_to_string(lit: &LitStr, path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|err| {
        Error::new_spanned(lit, format_args!("failed to read file `{path:?}`: {err}"))
    })
}

fn read_file(lit: &LitStr, path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|err| {
        Error::new_spanned(lit, format_args!("failed to read file `{path:?}`: {err}"))
    })
}

/// Converts SPIR-V bytes to words. If necessary, the byte order is swapped to match native-endian.
pub(crate) fn spirv_bytes_to_words(bytes: &[u8]) -> Result<Cow<'_, [u32]>, String> {
    let (words, []) = bytes.as_chunks::<{ size_of::<u32>() }>() else {
        return Err("the length of the bytes is not a multiple of 4".to_owned());
    };

    if words.is_empty() {
        return Err("invalid header".to_owned());
    }

    let magic = words[0];
    let is_little_endian = match magic {
        [0x03, 0x02, 0x23, 0x07] => true,
        [0x07, 0x23, 0x02, 0x03] => false,
        _ => return Err("invalid header".to_owned()),
    };
    let endiannesses_match = !(is_little_endian ^ cfg!(target_endian = "little"));

    let words = if endiannesses_match {
        // If the SPIR-V and current target endiannesses match (which they always will in practice)
        // and the slice already has the right size and alignment (which it also always has in
        // practice), then we can just cast the slice.
        if <*const _>::addr(bytes).is_multiple_of(align_of::<u32>()) {
            let ptr = bytes.as_ptr().cast::<u32>();
            let len = bytes.len() / size_of::<u32>();

            // SAFETY: We checked that the slice is sufficiently aligned. The length is definitely
            // in bounds as we divided it by the size of `u32`.
            let words = unsafe { slice::from_raw_parts(ptr, len) };

            return Ok(Cow::Borrowed(words));
        }

        words.iter().copied().map(u32::from_ne_bytes).collect()
    } else {
        words.iter().copied().map(u32_from_not_ne_bytes).collect()
    };

    Ok(Cow::Owned(words))
}

fn u32_from_not_ne_bytes(bytes: [u8; 4]) -> u32 {
    u32::from_ne_bytes(bytes).swap_bytes()
}

pub(super) struct MacroInput {
    shaders: Shaders,
    options: MacroOptions,
}

enum Shaders {
    Single(ShaderFields),
    Multiple(HashMap<Ident, ShaderFields>),
}

#[derive(Default)]
struct ShaderFields {
    shader_kind: Option<ShaderKind>,
    source_kind: Option<SourceKind>,
    source_language: Option<SourceLanguage>,
    entry_point: Option<String>,
    compiler: Option<Compiler>,
    macro_defines: Option<Vec<(String, String)>>,
}

enum SourceKind {
    Compiled(CompiledSourceKind),
    Precompiled(LitStr),
}

enum CompiledSourceKind {
    Path(LitStr),
    Inline(LitStr),
}

pub(crate) struct MacroOptions {
    pub(crate) global_source_language: SourceLanguage,
    pub(crate) global_compiler: Option<Compiler>,
    pub(crate) root_path_env: Option<LitStr>,
    pub(crate) include_directories: Vec<PathBuf>,
    pub(crate) global_macro_defines: Vec<(String, String)>,
    pub(crate) spirv_version: SpirvVersion,
    pub(crate) vulkan_version: VulkanVersion,
    pub(crate) generate_structs: bool,
    pub(crate) custom_derives: Vec<SynPath>,
    pub(crate) linalg_type: LinAlgType,
    pub(crate) dump: LitBool,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut parser = MacroInputParser::new();

        parser.parse(input)?;

        let shaders = parser.shaders.unwrap();

        let vulkan_version = parser.vulkan_version.unwrap_or_default();
        let options = MacroOptions {
            global_source_language: parser.global_source_language.unwrap_or_default(),
            global_compiler: parser.global_compiler,
            root_path_env: parser.root_path_env,
            include_directories: parser.include_directories.unwrap_or_default(),
            global_macro_defines: parser.global_macro_defines.unwrap_or_default(),
            vulkan_version,
            spirv_version: parser
                .spirv_version
                .unwrap_or(vulkan_version.to_spirv_version()),
            generate_structs: parser.generate_structs.unwrap_or(true),
            custom_derives: parser.custom_derives.unwrap_or_else(|| {
                vec![
                    parse_quote! { ::std::clone::Clone },
                    parse_quote! { ::std::marker::Copy },
                ]
            }),
            linalg_type: parser.linalg_type.unwrap_or_default(),
            dump: parser
                .dump
                .unwrap_or_else(|| LitBool::new(false, Span::call_site())),
        };

        Ok(MacroInput { shaders, options })
    }
}

impl MacroOptions {
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        MacroOptions {
            global_source_language: SourceLanguage::Glsl,
            global_compiler: None,
            root_path_env: None,
            include_directories: Vec::new(),
            global_macro_defines: Vec::new(),
            vulkan_version: VulkanVersion::V1_0,
            spirv_version: SpirvVersion::V1_0,
            generate_structs: true,
            custom_derives: Vec::new(),
            linalg_type: LinAlgType::default(),
            dump: LitBool::new(false, Span::call_site()),
        }
    }
}

struct MacroInputParser {
    manifest_dir: String,
    shaders: Option<Shaders>,
    global_source_language: Option<SourceLanguage>,
    global_compiler: Option<Compiler>,
    root_path_env: Option<LitStr>,
    include_directories: Option<Vec<PathBuf>>,
    global_macro_defines: Option<Vec<(String, String)>>,
    vulkan_version: Option<VulkanVersion>,
    spirv_version: Option<SpirvVersion>,
    generate_structs: Option<bool>,
    custom_derives: Option<Vec<SynPath>>,
    linalg_type: Option<LinAlgType>,
    dump: Option<LitBool>,
}

impl MacroInputParser {
    fn new() -> Self {
        MacroInputParser {
            manifest_dir: env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()),
            shaders: None,
            global_source_language: None,
            global_compiler: None,
            root_path_env: None,
            include_directories: None,
            global_macro_defines: None,
            vulkan_version: None,
            spirv_version: None,
            generate_structs: None,
            custom_derives: None,
            linalg_type: None,
            dump: None,
        }
    }

    fn parse(&mut self, input: ParseStream<'_>) -> Result {
        while !input.is_empty() {
            let field_ident = input.parse::<Ident>()?;
            input.parse::<Token![:]>()?;
            let field_name = field_ident.to_string();

            match field_name.as_str() {
                "ty" | "path" | "src" | "bytes" | "entry_point" => {
                    self.parse_shader_field(&field_ident, &field_name, input)?
                }
                "lang" => self.parse_lang(&field_ident, input)?,
                "compiler" => self.parse_compiler(&field_ident, input)?,
                "shaders" => self.parse_shader_entries(&field_ident, input)?,
                "root_path_env" => self.parse_root_path_env(&field_ident, input)?,
                "include" => self.parse_include(&field_ident, input)?,
                "define" => self.parse_define(&field_ident, input)?,
                "vulkan_version" => self.parse_vulkan_version(&field_ident, input)?,
                "spirv_version" => self.parse_spirv_version(&field_ident, input)?,
                "generate_structs" => self.parse_generate_structs(&field_ident, input)?,
                "custom_derives" => self.parse_custom_derives(&field_ident, input)?,
                "linalg_type" => self.parse_linalg_type(&field_ident, input)?,
                "dump" => self.parse_dump(&field_ident, input)?,
                _ => bail!(
                    field_ident,
                    "expected `ty`, `path`, `src`, `bytes`, `entry_point`, `lang`, `compiler`, \
                    `shaders`, `root_path_env`, `include`, `define`, `vulkan_version`, \
                    `spirv_version`, `generate_structs`, `custom_derives`, `linalg_type` or \
                    `dump` as a field, found `{field_name}`",
                ),
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        self.check_valid()
    }

    fn parse_shader_field(
        &mut self,
        field_ident: &Ident,
        field_name: &str,
        input: ParseStream<'_>,
    ) -> Result {
        if matches!(&self.shaders, Some(Shaders::Multiple(_))) {
            bail!(
                field_ident,
                "only one of `path`, `src`, `bytes` or `shaders` can be defined",
            );
        }

        if self.shaders.is_none() {
            self.shaders = Some(Shaders::Single(ShaderFields::default()));
        }

        let Some(Shaders::Single(shader_fields)) = &mut self.shaders else {
            unreachable!();
        };

        shader_fields.parse_shader_field(field_ident, field_name, input)
    }

    fn parse_lang(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        parse_lang(&mut self.global_source_language, None, field_ident, input)
    }

    fn parse_compiler(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        parse_compiler(&mut self.global_compiler, None, field_ident, input)
    }

    fn parse_shader_entries(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if !self.shaders.is_none() {
            bail!(
                field_ident,
                "only one of `path`, `src`, `bytes` or `shaders` can be defined",
            );
        }

        self.shaders = Some(Shaders::Multiple(HashMap::default()));

        let Some(Shaders::Multiple(shaders)) = &mut self.shaders else {
            unreachable!();
        };

        let in_braces;
        braced!(in_braces in input);

        while !in_braces.is_empty() {
            let shader_ident = in_braces.parse::<Ident>()?;

            let shader_fields = match shaders.entry(shader_ident) {
                Entry::Occupied(entry) => {
                    let shader_ident = entry.key();
                    let shader_name = shader_ident.to_string();

                    bail!(
                        shader_ident,
                        "shader entry `{shader_name}` is already defined",
                    );
                }
                Entry::Vacant(entry) => entry.insert(ShaderFields::default()),
            };

            in_braces.parse::<Token![:]>()?;

            let in_shader_definition;
            braced!(in_shader_definition in in_braces);

            while !in_shader_definition.is_empty() {
                let field_ident = in_shader_definition.parse::<Ident>()?;
                in_shader_definition.parse::<Token![:]>()?;
                let field_name = field_ident.to_string();

                match field_name.as_str() {
                    "ty" | "path" | "src" | "bytes" | "lang" | "entry_point" | "compiler"
                    | "define" => {
                        shader_fields.parse_shader_field(
                            &field_ident,
                            &field_name,
                            &in_shader_definition,
                        )?;
                    }
                    _ => bail!(
                        field_ident,
                        "expected `ty`, `path`, `src`, `bytes`, `lang`, `entry_point`, `compiler` \
                        or `define` as a field, found `{field_name}`",
                    ),
                }

                if !in_shader_definition.is_empty() {
                    in_shader_definition.parse::<Token![,]>()?;
                }
            }

            if !in_braces.is_empty() {
                in_braces.parse::<Token![,]>()?;
            }
        }

        if shaders.is_empty() {
            bail!(
                field_ident,
                "please specify at least one shader entry (e.g, `shaders: {{ `ty: \"vertex\", \
                path: \"entry_point.glsl\" }}`)",
            );
        }

        Ok(())
    }

    fn parse_root_path_env(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.root_path_env.is_some() {
            bail!(field_ident, "field `root_path_env` is already defined");
        }

        let lit = input.parse::<LitStr>()?;

        self.root_path_env = Some(lit);

        Ok(())
    }

    fn parse_include(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.include_directories.is_some() {
            bail!(field_ident, "field `include` is already defined");
        }

        self.include_directories = Some(Vec::new());

        let Some(include_directories) = &mut self.include_directories else {
            unreachable!();
        };

        let in_brackets;
        bracketed!(in_brackets in input);

        while !in_brackets.is_empty() {
            let path = in_brackets.parse::<LitStr>()?;

            include_directories.push([&self.manifest_dir, &path.value()].into_iter().collect());

            if !in_brackets.is_empty() {
                in_brackets.parse::<Token![,]>()?;
            }
        }

        Ok(())
    }

    fn parse_define(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        parse_define(&mut self.global_macro_defines, field_ident, input)
    }

    fn parse_vulkan_version(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.vulkan_version.is_some() {
            bail!(field_ident, "field `vulkan_version` is already defined");
        }

        let lit = input.parse::<LitStr>()?;

        self.vulkan_version = Some(match lit.value().as_str() {
            "1.0" => VulkanVersion::V1_0,
            "1.1" => VulkanVersion::V1_1,
            "1.2" => VulkanVersion::V1_2,
            "1.3" => VulkanVersion::V1_3,
            ver => bail!(lit, "expected `1.0`, `1.1`, `1.2` or `1.3`, found `{ver}`"),
        });

        Ok(())
    }

    fn parse_spirv_version(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.spirv_version.is_some() {
            bail!(field_ident, "field `spirv_version` is already defined");
        }

        let lit = input.parse::<LitStr>()?;

        self.spirv_version = Some(match lit.value().as_str() {
            "1.0" => SpirvVersion::V1_0,
            "1.1" => SpirvVersion::V1_1,
            "1.2" => SpirvVersion::V1_2,
            "1.3" => SpirvVersion::V1_3,
            "1.4" => SpirvVersion::V1_4,
            "1.5" => SpirvVersion::V1_5,
            "1.6" => SpirvVersion::V1_6,
            ver => bail!(
                lit,
                "expected `1.0`, `1.1`, `1.2`, `1.3`, `1.4`, `1.5` or `1.6`, found `{ver}`",
            ),
        });

        Ok(())
    }

    fn parse_generate_structs(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.generate_structs.is_some() {
            bail!(field_ident, "field `generate_structs` is already defined");
        }

        let lit = input.parse::<LitBool>()?;

        self.generate_structs = Some(lit.value);

        Ok(())
    }

    fn parse_custom_derives(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.custom_derives.is_some() {
            bail!(field_ident, "field `custom_derives` is already defined");
        }

        self.custom_derives = Some(Vec::new());

        let Some(custom_derives) = &mut self.custom_derives else {
            unreachable!();
        };

        let in_brackets;
        bracketed!(in_brackets in input);

        while !in_brackets.is_empty() {
            custom_derives.push(in_brackets.parse::<SynPath>()?);

            if !in_brackets.is_empty() {
                in_brackets.parse::<Token![,]>()?;
            }
        }

        Ok(())
    }

    fn parse_linalg_type(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.linalg_type.is_some() {
            bail!(field_ident, "field `linalg_type` is already defined");
        }

        let lit = input.parse::<LitStr>()?;

        self.linalg_type = Some(match lit.value().as_str() {
            "std" => LinAlgType::Std,
            "cgmath" => LinAlgType::CgMath,
            "nalgebra" => LinAlgType::Nalgebra,
            ty => bail!(lit, "expected `std`, `cgmath` or `nalgebra`, found `{ty}`"),
        });

        Ok(())
    }

    fn parse_dump(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.dump.is_some() {
            bail!(field_ident, "field `dump` is already defined");
        }

        let lit = input.parse::<LitBool>()?;

        self.dump = Some(lit);

        Ok(())
    }

    fn check_valid(&self) -> Result {
        match &self.shaders {
            Some(Shaders::Single(shader_fields)) => {
                shader_fields.check_valid(
                    Span::call_site(),
                    self.global_source_language,
                    self.global_compiler,
                )?;
            }
            Some(Shaders::Multiple(shaders)) if !shaders.is_empty() => {
                for (shader_ident, shader_fields) in shaders {
                    shader_fields.check_valid(
                        shader_ident.span(),
                        self.global_source_language,
                        self.global_compiler,
                    )?;
                }
            }
            Some(Shaders::Multiple(_)) | None => {
                bail!(
                    "please specify at least one shader (e.g., `ty: \"vertex\", path: \
                    \"entry_point.glsl\"`)",
                );
            }
        }

        Ok(())
    }
}

impl ShaderFields {
    fn parse_shader_field(
        &mut self,
        field_ident: &Ident,
        field_name: &str,
        input: ParseStream<'_>,
    ) -> Result {
        match field_name {
            "ty" => self.parse_ty(field_ident, input)?,
            "path" => self.parse_path(field_ident, input)?,
            "src" => self.parse_src(field_ident, input)?,
            "bytes" => self.parse_bytes(field_ident, input)?,
            "lang" => self.parse_lang(field_ident, input)?,
            "entry_point" => self.parse_entry_point(field_ident, input)?,
            "compiler" => self.parse_compiler(field_ident, input)?,
            "define" => self.parse_define(field_ident, input)?,
            _ => unreachable!(),
        }

        Ok(())
    }

    fn parse_ty(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.shader_kind.is_some() {
            bail!(field_ident, "field `ty` is already defined");
        }

        if matches!(self.source_kind, Some(SourceKind::Precompiled(_))) {
            bail!(
                field_ident,
                "fields `ty` and `bytes` cannot be defined in the same shader entry",
            );
        }

        let lit = input.parse::<LitStr>()?;

        self.shader_kind = Some(match lit.value().as_str() {
            "vertex" => ShaderKind::Vertex,
            "tess_ctrl" => ShaderKind::TessControl,
            "tess_eval" => ShaderKind::TessEvaluation,
            "geometry" => ShaderKind::Geometry,
            "task" => ShaderKind::Task,
            "mesh" => ShaderKind::Mesh,
            "fragment" => ShaderKind::Fragment,
            "compute" => ShaderKind::Compute,
            "raygen" => ShaderKind::RayGeneration,
            "anyhit" => ShaderKind::AnyHit,
            "closesthit" => ShaderKind::ClosestHit,
            "miss" => ShaderKind::Miss,
            "intersection" => ShaderKind::Intersection,
            "callable" => ShaderKind::Callable,
            kind => bail!(
                lit,
                "expected `vertex`, `tess_ctrl`, `tess_eval`, `geometry`, `task`, `mesh`, \
                `fragment`, `compute`, `raygen`, `anyhit`, `closesthit`, `miss`, `intersection` \
                or `callable`, found `{kind}`",
            ),
        });

        Ok(())
    }

    fn parse_path(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if let Some(source_kind) = &self.source_kind {
            let msg = match source_kind {
                SourceKind::Compiled(CompiledSourceKind::Path(_)) => {
                    "field `path` is already defined"
                }
                SourceKind::Compiled(CompiledSourceKind::Inline(_)) => {
                    "fields `path` and `src` cannot be defined in the same shader entry"
                }
                SourceKind::Precompiled(_) => {
                    "fields `path` and `bytes` cannot be defined in the same shader entry"
                }
            };

            bail!(field_ident, "{msg}");
        }

        let lit = input.parse::<LitStr>()?;

        self.source_kind = Some(SourceKind::Compiled(CompiledSourceKind::Path(lit)));

        Ok(())
    }

    fn parse_src(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if let Some(source_kind) = &self.source_kind {
            let msg = match source_kind {
                SourceKind::Compiled(CompiledSourceKind::Path(_)) => {
                    "fields `src` and `path` cannot be defined in the same shader entry"
                }
                SourceKind::Compiled(CompiledSourceKind::Inline(_)) => {
                    "field `src` is already defined"
                }
                SourceKind::Precompiled(_) => {
                    "fields `src` and `bytes` cannot be defined in the same shader entry"
                }
            };

            bail!(field_ident, "{msg}");
        }

        let lit = input.parse::<LitStr>()?;

        self.source_kind = Some(SourceKind::Compiled(CompiledSourceKind::Inline(lit)));

        Ok(())
    }

    fn parse_bytes(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if let Some(source_kind) = &self.source_kind {
            let msg = match source_kind {
                SourceKind::Compiled(CompiledSourceKind::Path(_)) => {
                    "fields `bytes` and `path` cannot be defined in the same shader entry"
                }
                SourceKind::Compiled(CompiledSourceKind::Inline(_)) => {
                    "fields `bytes` and `src` cannot be defined in the same shader entry"
                }
                SourceKind::Precompiled(_) => "field `bytes` is already defined",
            };

            bail!(field_ident, "{msg}");
        }

        if self.shader_kind.is_some() {
            bail!(
                field_ident,
                "fields `bytes` and `ty` cannot be defined in the same shader entry",
            );
        }

        if self.source_language.is_some() {
            bail!(
                field_ident,
                "fields `bytes` and `lang` cannot be defined in the same shader entry",
            );
        }

        if self.entry_point.is_some() {
            bail!(
                field_ident,
                "fields `bytes` and `entry_point` cannot be defined in the same shader entry",
            );
        }

        if self.compiler.is_some() {
            bail!(
                field_ident,
                "fields `bytes` and `compiler` cannot be defined in the same shader entry",
            );
        }

        let lit = input.parse::<LitStr>()?;

        self.source_kind = Some(SourceKind::Precompiled(lit));

        Ok(())
    }

    fn parse_lang(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        parse_lang(
            &mut self.source_language,
            self.source_kind.as_ref(),
            field_ident,
            input,
        )
    }

    fn parse_entry_point(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        if self.entry_point.is_some() {
            bail!(field_ident, "field `entry_point` is already defined");
        }

        if matches!(self.source_kind, Some(SourceKind::Precompiled(_))) {
            bail!(
                field_ident,
                "fields `entry_point` and `bytes` cannot be defined in the same shader entry",
            );
        }

        let lit = input.parse::<LitStr>()?;

        self.entry_point = Some(lit.value());

        Ok(())
    }

    fn parse_compiler(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        parse_compiler(
            &mut self.compiler,
            self.source_kind.as_ref(),
            field_ident,
            input,
        )
    }

    fn parse_define(&mut self, field_ident: &Ident, input: ParseStream<'_>) -> Result {
        parse_define(&mut self.macro_defines, field_ident, input)
    }

    fn check_valid(
        &self,
        span: Span,
        global_source_language: Option<SourceLanguage>,
        global_compiler: Option<Compiler>,
    ) -> Result {
        if self.source_kind.is_none() {
            return Err(Error::new(
                span,
                "please specify the shader source (e.g., `path: \"entry_point.glsl\"`)",
            ));
        }

        let source_language = self
            .source_language
            .or(global_source_language)
            .unwrap_or_default();
        let compiler = self
            .compiler
            .or(global_compiler)
            .unwrap_or(source_language.default_compiler());

        if self.entry_point.is_some() && compiler == Compiler::Shaderc {
            return Err(Error::new(
                span,
                "the `entry_point` option cannot be used when compiling with shaderc; please \
                remove it",
            ));
        }

        if !compiler.is_valid_for(source_language) {
            return Err(Error::new(
                span,
                format_args!(
                    "source language `{}` cannot be compiled with compiler `{}`",
                    source_language.as_str(),
                    compiler.as_str(),
                ),
            ));
        }

        Ok(())
    }
}

fn parse_lang(
    source_language: &mut Option<SourceLanguage>,
    source_kind: Option<&SourceKind>,
    field_ident: &Ident,
    input: ParseStream<'_>,
) -> Result {
    if source_language.is_some() {
        bail!(field_ident, "field `lang` is already defined");
    }

    if matches!(source_kind, Some(SourceKind::Precompiled(_))) {
        bail!(
            field_ident,
            "fields `lang` and `bytes` cannot be defined in the same shader entry",
        );
    }

    let lit = input.parse::<LitStr>()?;

    *source_language = Some(match lit.value().as_str() {
        "glsl" => SourceLanguage::Glsl,
        "hlsl" => SourceLanguage::Hlsl,
        "slang" => SourceLanguage::Slang,
        lang => bail!(lit, "expected `glsl`, `hlsl` or `slang`, found `{lang}`"),
    });

    Ok(())
}

fn parse_compiler(
    compiler: &mut Option<Compiler>,
    source_kind: Option<&SourceKind>,
    field_ident: &Ident,
    input: ParseStream<'_>,
) -> Result {
    if compiler.is_some() {
        bail!(field_ident, "field `compiler` is already defined");
    }

    if matches!(source_kind, Some(SourceKind::Precompiled(_))) {
        bail!(
            field_ident,
            "fields `compiler` and `bytes` cannot be defined in the same shader entry",
        );
    }

    let lit = input.parse::<LitStr>()?;

    *compiler = Some(match lit.value().as_str() {
        "shaderc" => Compiler::Shaderc,
        "slangc" => Compiler::Slangc,
        compiler => bail!(lit, "expected `shaderc` or `slangc`, found `{compiler}`"),
    });

    Ok(())
}

fn parse_define(
    macro_defines: &mut Option<Vec<(String, String)>>,
    field_ident: &Ident,
    input: ParseStream<'_>,
) -> Result {
    if macro_defines.is_some() {
        bail!(field_ident, "field `define` is already defined");
    }

    *macro_defines = Some(Vec::new());

    let Some(macro_defines) = macro_defines else {
        unreachable!();
    };

    let array_input;
    bracketed!(array_input in input);

    while !array_input.is_empty() {
        let tuple_input;
        parenthesized!(tuple_input in array_input);

        let name = tuple_input.parse::<LitStr>()?;
        tuple_input.parse::<Token![,]>()?;
        let value = tuple_input.parse::<LitStr>()?;
        macro_defines.push((name.value(), value.value()));

        if !array_input.is_empty() {
            array_input.parse::<Token![,]>()?;
        }
    }

    Ok(())
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum ShaderKind {
    Vertex,
    TessControl,
    TessEvaluation,
    Geometry,
    Task,
    Mesh,
    Fragment,
    Compute,
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,
}

impl ShaderKind {
    fn as_shaderc_stage(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vert",
            ShaderKind::TessControl => "tesc",
            ShaderKind::TessEvaluation => "tese",
            ShaderKind::Geometry => "geom",
            ShaderKind::Task => "task",
            ShaderKind::Mesh => "mesh",
            ShaderKind::Fragment => "frag",
            ShaderKind::Compute => "comp",
            ShaderKind::RayGeneration => "rgen",
            ShaderKind::AnyHit => "rahit",
            ShaderKind::ClosestHit => "rchit",
            ShaderKind::Miss => "rmiss",
            ShaderKind::Intersection => "rint",
            ShaderKind::Callable => "rcall",
        }
    }

    fn as_slangc_stage(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vertex",
            ShaderKind::TessControl => "hull",
            ShaderKind::TessEvaluation => "domain",
            ShaderKind::Geometry => "geometry",
            ShaderKind::Task => "amplification",
            ShaderKind::Mesh => "mesh",
            ShaderKind::Fragment => "fragment",
            ShaderKind::Compute => "compute",
            ShaderKind::RayGeneration => "raygeneration",
            ShaderKind::AnyHit => "anyhit",
            ShaderKind::ClosestHit => "closesthit",
            ShaderKind::Miss => "miss",
            ShaderKind::Intersection => "intersection",
            ShaderKind::Callable => "callable",
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) enum SourceLanguage {
    #[default]
    Glsl,
    Hlsl,
    Slang,
}

impl SourceLanguage {
    fn as_str(self) -> &'static str {
        match self {
            SourceLanguage::Glsl => "glsl",
            SourceLanguage::Hlsl => "hlsl",
            SourceLanguage::Slang => "slang",
        }
    }

    fn default_compiler(self) -> Compiler {
        match self {
            Self::Glsl => Compiler::Shaderc,
            Self::Hlsl | Self::Slang => Compiler::Slangc,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compiler {
    Shaderc,
    Slangc,
}

impl Compiler {
    fn as_str(self) -> &'static str {
        match self {
            Compiler::Shaderc => "shaderc",
            Compiler::Slangc => "slangc",
        }
    }

    fn as_command(self) -> &'static str {
        match self {
            Compiler::Shaderc => "glslc",
            Compiler::Slangc => "slangc",
        }
    }

    fn is_valid_for(self, source_language: SourceLanguage) -> bool {
        match self {
            Compiler::Shaderc => matches!(source_language, SourceLanguage::Glsl),
            Compiler::Slangc => {
                matches!(
                    source_language,
                    SourceLanguage::Glsl | SourceLanguage::Hlsl | SourceLanguage::Slang,
                )
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) enum VulkanVersion {
    #[default]
    V1_0,
    V1_1,
    V1_2,
    V1_3,
}

impl VulkanVersion {
    fn as_shaderc_target_env(self) -> &'static str {
        match self {
            VulkanVersion::V1_0 => "vulkan1.0",
            VulkanVersion::V1_1 => "vulkan1.1",
            VulkanVersion::V1_2 => "vulkan1.2",
            VulkanVersion::V1_3 => "vulkan1.3",
        }
    }

    fn to_spirv_version(self) -> SpirvVersion {
        match self {
            VulkanVersion::V1_0 => SpirvVersion::V1_0,
            VulkanVersion::V1_1 => SpirvVersion::V1_3,
            VulkanVersion::V1_2 => SpirvVersion::V1_5,
            VulkanVersion::V1_3 => SpirvVersion::V1_6,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum SpirvVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
    V1_4,
    V1_5,
    V1_6,
}

impl SpirvVersion {
    fn as_shaderc_target_spv(self) -> &'static str {
        match self {
            SpirvVersion::V1_0 => "spv1.0",
            SpirvVersion::V1_1 => "spv1.1",
            SpirvVersion::V1_2 => "spv1.2",
            SpirvVersion::V1_3 => "spv1.3",
            SpirvVersion::V1_4 => "spv1.4",
            SpirvVersion::V1_5 => "spv1.5",
            SpirvVersion::V1_6 => "spv1.6",
        }
    }

    fn as_slangc_profile(self) -> &'static str {
        match self {
            SpirvVersion::V1_0 => "spirv_1_0",
            SpirvVersion::V1_1 => "spirv_1_1",
            SpirvVersion::V1_2 => "spirv_1_2",
            SpirvVersion::V1_3 => "spirv_1_3",
            SpirvVersion::V1_4 => "spirv_1_4",
            SpirvVersion::V1_5 => "spirv_1_5",
            SpirvVersion::V1_6 => "spirv_1_6",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum LinAlgType {
    #[default]
    Std,
    CgMath,
    Nalgebra,
}

pub(crate) type Result<T = (), E = syn::Error> = ::std::result::Result<T, E>;
