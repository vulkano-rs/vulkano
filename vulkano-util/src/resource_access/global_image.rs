use super::ResourceAccess;
use std::{
    error::Error,
    fmt::{Display, Formatter},
};
use vulkano::{
    image::{
        sampler::ComponentMapping,
        view::{ImageView, ImageViewCreateInfo},
        AllocateImageError, Image, ImageCreateInfo, ImageLayout, ImageSubresourceRange,
    },
    memory::allocator::AllocationCreateInfo,
    Validated, VulkanError,
};
use vulkano_taskgraph::{
    descriptor_set::{SampledImageId, StorageImageId},
    graph::TaskGraph,
    resource::Resources,
    Id,
};

#[derive(Clone, Debug)]
pub enum GlobalImageCreateError {
    AllocateImageError(AllocateImageError),
    VulkanError(VulkanError),
}

impl Error for GlobalImageCreateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::AllocateImageError(err) => Some(err),
            Self::VulkanError(err) => Some(err),
        }
    }
}

impl Display for GlobalImageCreateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocateImageError(err) => err.fmt(f),
            Self::VulkanError(err) => err.fmt(f),
        }
    }
}

#[derive(Default)]
pub struct ModifyImageViewCreateInfo {
    pub component_mapping: Option<ComponentMapping>,
    pub subresource_range: Option<ImageSubresourceRange>,
}

#[derive(Default)]
pub struct GlobalImageCreateInfo {
    pub storage_layout: Option<ImageLayout>,
    pub sampled_layout: Option<ImageLayout>,
    pub image_view_create_info: ModifyImageViewCreateInfo,
}

impl GlobalImageCreateInfo {
    pub fn storage() -> Self {
        Self {
            storage_layout: Some(ImageLayout::General),
            ..Default::default()
        }
    }

    pub fn sampled() -> Self {
        Self {
            sampled_layout: Some(ImageLayout::General),
            ..Default::default()
        }
    }

    pub fn storage_sampled() -> Self {
        Self {
            storage_layout: Some(ImageLayout::General),
            sampled_layout: Some(ImageLayout::General),
            ..Default::default()
        }
    }

    pub fn modify_image_view_create_info(mut self, create_info: ModifyImageViewCreateInfo) -> Self {
        self.image_view_create_info = create_info;
        self
    }
}

#[derive(Clone, Copy)]
pub struct GlobalImageTracker {
    physical_id: Id<Image>,
    virtual_id: Option<Id<Image>>,
    storage_image_id: Option<StorageImageId>,
    sampled_image_id: Option<SampledImageId>,
}

impl GlobalImageTracker {
    pub fn new<W>(
        task_graph: Option<&mut TaskGraph<W>>,
        resources: &Resources,
        physical_id: Id<Image>,
        create_info: GlobalImageCreateInfo,
    ) -> Result<Self, Validated<VulkanError>> {
        let bcx = resources.bindless_context().unwrap();

        let virtual_id = task_graph.map(|task_graph| task_graph.add_image(&ImageCreateInfo::default()));

        if create_info.storage_layout.is_none() && create_info.sampled_layout.is_none() {
            return Ok(Self {
                physical_id,
                virtual_id,
                storage_image_id: None,
                sampled_image_id: None,
            });
        }

        let image = resources.image(physical_id).unwrap().image().clone();

        // this is really gross
        let mut image_view_create_info = ImageViewCreateInfo::from_image(&image);
        if let Some(component_mapping) = create_info.image_view_create_info.component_mapping {
            image_view_create_info.component_mapping = component_mapping;
        }
        if let Some(subresource_range) = create_info.image_view_create_info.subresource_range {
            image_view_create_info.subresource_range = subresource_range;
        }

        let image_view = ImageView::new(image, image_view_create_info)?;

        let storage_image_id = create_info.storage_layout.map(|storage_layout| bcx.global_set()
            .add_storage_image(image_view.clone(), storage_layout));

        let sampled_image_id = create_info.sampled_layout.map(|sampled_layout| bcx.global_set()
            .add_sampled_image(image_view, sampled_layout));

        Ok(Self {
            physical_id,
            virtual_id,
            storage_image_id,
            sampled_image_id,
        })
    }

    pub fn id(&self) -> Id<Image> {
        self.physical_id
    }

    pub fn v_id(&self) -> Id<Image> {
        self.virtual_id.unwrap()
    }

    pub fn storage(&self) -> StorageImageId {
        self.storage_image_id.unwrap()
    }

    pub fn sampled(&self) -> SampledImageId {
        self.sampled_image_id.unwrap()
    }

    pub fn update_virtual<W>(&mut self, task_graph: &mut TaskGraph<W>) {
        if let Some(virtual_id) = &mut self.virtual_id {
            *virtual_id = task_graph.add_image(&ImageCreateInfo::default());
        }
    }

    pub fn update_bindless(&mut self, resources: &Resources) -> Result<(), Validated<VulkanError>> {
        if self.storage_image_id.is_none() && self.sampled_image_id.is_none() {
            return Ok(());
        }

        let image_id = self.physical_id;

        let image_view =
            ImageView::new_default(resources.image(image_id).unwrap().image().clone())?;

        let bcx = resources.bindless_context().unwrap();

        if let Some(storage_image) = self.storage_image_id {
            let storage_layout = bcx
                .global_set()
                .storage_image(storage_image)
                .unwrap()
                .image_layout();

            unsafe { bcx.global_set().remove_storage_image(storage_image) };

            self.storage_image_id = Some(
                bcx.global_set()
                    .add_storage_image(image_view.clone(), storage_layout),
            );
        };

        if let Some(sampled_image) = self.sampled_image_id {
            let sampled_layout = bcx
                .global_set()
                .sampled_image(sampled_image)
                .unwrap()
                .image_layout();

            unsafe { bcx.global_set().remove_sampled_image(sampled_image) };

            self.sampled_image_id = Some(
                bcx.global_set()
                    .add_sampled_image(image_view, sampled_layout),
            );
        };

        Ok(())
    }

    pub fn new_extent(
        &mut self,
        access: &ResourceAccess,
        extent: [u32; 3],
    ) -> Result<(), Validated<VulkanError>> {
        let resources = access.resources();
        let image_state = resources.image(self.physical_id).unwrap();
        let old_image = image_state.image();

        unsafe { resources.remove_image(self.physical_id) }.unwrap();

        self.physical_id = resources
            .create_image(
                ImageCreateInfo {
                    format: old_image.format(),
                    usage: old_image.usage(),
                    array_layers: old_image.array_layers(),
                    extent,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

        access.update_global_image(self)
    }
}
