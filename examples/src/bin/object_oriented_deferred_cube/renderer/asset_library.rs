use std::collections::HashMap;
use std::sync::Arc;

use anyhow::bail;

use image::EncodableLayout;

use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewType};
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerMipmapMode};
use vulkano::sync::GpuFuture;

use crate::asset::model::{Model, Vertex};
use crate::VulkanContext;

/// Stores handles to assets uploaded to the GPU.
pub struct AssetLibrary {
    /// The lookup table mapping to handles associated with the uploaded models.
    /// You'll probably want to use something like string-cache or arrayvec's `ArrayString`
    /// for the key, instead of owned `String`s, to improve performance.
    /// https://github.com/servo/string-cache
    /// https://github.com/bluss/arrayvec
    pub models: HashMap<String, ModelHandles>,
    /// The single array storing all textures,
    /// which is the recommended approach to texture management by Sascha Willems:
    /// https://www.reddit.com/r/vulkan/comments/7tk3wc/comment/dtggkkv
    /// The drawback is you can only use the same dimension/format/sampler for textures in the same array.
    /// If you need to support more variations, you could simply add more arrays,
    /// or even use an array of texture arrays (which isn't as ergonomic as it sounds,
    /// wouldn't actually recommend if you have control over what textures to use).
    pub texture_array: Arc<ImageView<ImmutableImage>>,
    /// The sampler used for the textures.
    pub sampler: Arc<Sampler>,
}

pub struct ModelHandles {
    pub vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,
    pub base_color: u32,
}

impl AssetLibrary {
    pub fn new<TWindow: Send + Sync + 'static>(
        vk: VulkanContext<TWindow>,
        models: HashMap<String, Model>,
        textures: HashMap<String, Vec<u8>>,
        image_size: u32,
    ) -> anyhow::Result<Arc<Self>> {
        // We'll use object to track all the futures (synchronization handles) generated
        // throughout the uploading process.
        // The `now` generates a dummy future for us to join.
        // The `Option` is for circumventing Rust's ownership system when mutating it.
        let mut future = Some(vulkano::sync::now(vk.device()).boxed());

        // First we build a lookup of texture ids to their array layer indices.
        let texture_lookup: HashMap<_, _> = textures
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.clone(), i as u32))
            .collect();
        // Then we concatenate all the textures' raw pixels into one giant array.
        // Don't forget to decode them first!
        // Also note most PNG encoders (like Photoshop or ImageMagick)
        // will optimize the PNG's pixel format if applicable,
        // resulting in unexpected errors when loading.
        // You could either enforce a format with no surprises,
        // converting them on-the-fly (like this example does),
        // or run `magick mogrify png32:filename.png` on them to convert to RGBA8.
        // Q: Can I use RGB without the A? I don't need transparency in textures.
        // A: You probably can, but it's
        //    a) only available on some Android devices and Intel integrated graphics
        //    https://vulkan.gpuinfo.org/listdevicescoverage.php?optimaltilingformat=R8G8B8_SRGB&featureflagbit=SAMPLED_IMAGE&platform=all
        //    b) perhaps very slow (I can't verify this myself, but considering how memory works).
        let texture_array_data: Vec<_> = textures
            .into_iter()
            .map(|(id, x)| {
                let image = image::load_from_memory(x.as_slice())?
                    // Convert 8-bit grayscale to RGBA8
                    .into_rgba8();
                if image.dimensions() != (image_size, image_size) {
                    bail!("Image {} isn't {}x{}", &id, image_size, image_size);
                }
                Ok(image.as_bytes().to_owned())
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|x| x)
            .collect();
        let (texture_array, texture_array_future) = ImmutableImage::from_iter(
            texture_array_data,
            ImageDimensions::Dim2d {
                width: image_size,
                height: image_size,
                array_layers: texture_lookup.len() as u32,
            },
            MipmapsCount::Log2,    // We'll want mipmaps for 3D.
            Format::R8G8B8A8_SRGB, // Mind the format!
            vk.main_queue(),
        )?;
        future = Some(future.take().unwrap().join(texture_array_future).boxed());
        let texture_array = ImageView::start(texture_array)
            // Vulkano auto-detects the ImageView's type by default,
            // which leads to strange bugs if the array only contains one element,
            // so it's better to explicitly specify the type on arrays.
            .with_type(ImageViewType::Dim2dArray)
            .build()?;

        // Builds a lookup for the uploaded models.
        let mut model_lut = HashMap::<String, ModelHandles>::new();
        for (id, model) in models {
            // Upload the models' vertex and index buffers.
            let (vertex_buffer, _vertex_buffer_future) = match ImmutableBuffer::from_iter(
                model.vertices,
                BufferUsage::vertex_buffer(),
                vk.main_queue(),
            ) {
                Ok(x) => x,
                Err(e) => {
                    bail!("Uploading of vertex buffer for {} failed: {}", &id, e);
                }
            };
            let (index_buffer, _index_buffer_future) = match ImmutableBuffer::from_iter(
                model.indices,
                BufferUsage::index_buffer(),
                vk.main_queue(),
            ) {
                Ok(x) => x,
                Err(e) => {
                    bail!("Uploading of index buffer for {} failed: {}", &id, e);
                }
            };

            // Convert the texture ids to texture array indices.
            let base_color = match texture_lookup.get(&model.base_color) {
                None => {
                    bail!(
                        "Model {} references a non-existent base color texture {}",
                        &id,
                        &model.base_color
                    )
                }
                Some(x) => x.to_owned(),
            };

            model_lut.insert(
                id,
                ModelHandles {
                    vertex_buffer,
                    index_buffer,
                    base_color,
                },
            );
        }

        // Create a simple linear sampler.
        let sampler = Sampler::start(vk.device())
            .filter(Filter::Linear)
            .mipmap_mode(SamplerMipmapMode::Linear)
            // Repeat is usually what you want for tileable textures.
            .address_mode(SamplerAddressMode::Repeat)
            // There are more options we haven't touched;
            // For example, anisotropic filtering belongs here.
            // See Sampler::start()'s source code for all the options.
            .build()?;

        // Don't forget we haven't waited for the GPU uploading to finish yet!
        // Instead of signaling a fence, you could just return the future
        // and chain it to the next operation to reduce the sync overhead, but
        // a) then you can't guarantee the creation of AssetLibrary is actually successful;
        // b) it's not very ergonomic to write.
        future
            .take()
            .unwrap()
            .then_signal_fence_and_flush()?
            // TODO You may want to specify a timeout here.
            .wait(None)?;

        Ok(Arc::new(Self {
            models: model_lut,
            texture_array,
            sampler,
        }))
    }

    /// Information needed for creating a new descriptor set needed to access the assets.
    /// Since descriptor sets are bound to specific pipelines, we can't create just one and use it everywhere.
    /// TODO for some reason, `WriteDescriptorSet` doesn't impl `Clone` yet, so this have to be dynamically generated.
    pub fn descriptor_writes(&self) -> Vec<WriteDescriptorSet> {
        vec![WriteDescriptorSet::image_view_sampler(
            0,
            self.texture_array.clone(),
            self.sampler.clone(),
        )]
    }
}
