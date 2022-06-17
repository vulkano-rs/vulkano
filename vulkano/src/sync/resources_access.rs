use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use ash::vk::DeviceSize;
use rangemap::RangeMap;
use smallvec::{smallvec, SmallVec};

use crate::buffer::sys::UnsafeBuffer;
use crate::buffer::BufferAccess;
use crate::command_buffer::synced::Resource;
use crate::image::sys::UnsafeImage;
use crate::image::{ImageAccess, ImageLayout, ImageSubresourceRange};
use crate::sync::{
    AccessFlags, BufferMemoryBarrier, DependencyInfo, ImageMemoryBarrier, PipelineMemoryAccess,
    PipelineStages,
};

/// General barrier between two command batches
/// Similar to `BufferMemoryBarrier` but without buffer, range and queue transfer.
#[derive(Clone, Debug)]
struct MemoryBarrierDescription {
    source_stages: PipelineStages,
    source_access: AccessFlags,
    destination_stages: PipelineStages,
    destination_access: AccessFlags,
}

/// Image barrier between two command batches
/// Similar to `ImageMemoryBarrier` without image, range and queue transfer.
#[derive(Clone, Debug)]
struct ImageBarrierDescription {
    source_stages: PipelineStages,
    source_access: AccessFlags,
    destination_stages: PipelineStages,
    destination_access: AccessFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
}

/// Represents some change in memory access as a result of executing some commands (possibly with barriers)
///
/// **Example:**
/// `d1 = { in_access: { stages : VERTEX, access: READ }, out_access: {stages : FRAGMENT, access: WRITE}}`
/// It means that we will be reading memory on VERTEX stage, then we will have some other commands and possibly
/// barriers, after that we will be writing on the FRAGMENT stage at the end.
/// This delta then will help us to determine what barriers are needed **before** and **after** this batch in order
/// to guarantee resource synchronization (e.g. no write while reading, no read while writing, no double writing).
/// Let's define another delta:
/// `d2 = { in_access: {stages: COMPUTE, access: WRITE }, out_access: {stages: TRANSFER, access: WRITE}}`
/// in this case we can easily determine which barrier we need to put in between corresponding command batches.
/// If batch represented by `d2` should go first and then `d1` we know that barrier should have following parameters:
/// `barrier = {src_stage_mask = TRANSFER, src_access_mask = WRITE, dst_stage_mask = VERTEX, dst_access_mask = READ}`
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PipelineMemoryAccessDelta {
    /// What memory access is needed to start executing corresponding set of commands
    pub in_access: PipelineMemoryAccess,
    /// What memory access will this resource have after corresponding set of commands is executed
    pub out_access: PipelineMemoryAccess,

    /// Indicates if there were any barriers in between
    pub any_barrier: bool,
    /// Indicates if there were any actual access (any read or any write)
    pub any_access: bool,
}

fn combine_read_access(a: &PipelineMemoryAccess, b: &PipelineMemoryAccess) -> PipelineMemoryAccess {
    debug_assert!(!a.exclusive && !b.exclusive);
    PipelineMemoryAccess {
        stages: a.stages | b.stages,
        access: a.access | b.access,
        exclusive: false,
    }
}

impl PipelineMemoryAccessDelta {
    fn debug_validation(&self) {
        debug_assert!(self.any_barrier || self.in_access == self.out_access);
        debug_assert!(self.any_access || !self.any_barrier);
        if !self.any_access {
            debug_assert_eq!(self.in_access, PipelineMemoryAccess::default());
            debug_assert_eq!(self.out_access, PipelineMemoryAccess::default());
        }
    }

    /// This function combines two deltas into one that represents consequent execution of corresponding command batches.
    /// It also returns optional barrier that should be put in between to guarantee synchronization.
    ///
    /// There are 5 possible cases here:
    /// 1. `R-R` We start with reading and end with reading, no barriers were inserted in between
    /// 2. `R|R` We start with reading and end with reading but there was a barrier in between
    /// 3. `W|R` We start with writing and end with reading, there definitely was a barrier
    /// 4. `R|W` We start with reading and end with writing, there definitely was a barrier
    /// 5. `W|W` We start with writing and end with writing, there will be barrier if there were more than one write
    /// If it is not important which `W` or `R` is used we will use `X` instead.
    /// If it is not important there is a barrier or not we will use `?` instead of `|`.
    ///
    /// Note: in case `R-R` it is guaranteed that `in_access == out_access` - only reading without barriers
    fn then(
        &self,
        other: &PipelineMemoryAccessDelta,
    ) -> (PipelineMemoryAccessDelta, Option<MemoryBarrierDescription>) {
        self.debug_validation();
        other.debug_validation();

        // If there were no actual access in any of two - return another one
        if !self.any_access {
            return ((*other).clone(), None);
        } else if !other.any_access {
            return ((*self).clone(), None);
        }

        return if self.out_access.exclusive || other.in_access.exclusive {
            // First let's handle cases when we need additional barrier: `X?R + W?X` or `X?W + W?X` or `X?W + R?X`
            (
                Self {
                    in_access: self.in_access,
                    out_access: other.out_access,
                    any_barrier: true,
                    any_access: true,
                },
                Some(MemoryBarrierDescription {
                    source_stages: self.out_access.stages,
                    source_access: self.out_access.access,
                    destination_stages: other.in_access.stages,
                    destination_access: other.in_access.access,
                }),
            )
        } else {
            // Then let's handle cases that don't need additional barrier: `X?R + R?X`
            return (
                if !self.any_barrier && !other.any_barrier {
                    // `R-R + R-R`
                    Self {
                        in_access: combine_read_access(&self.out_access, &other.in_access),
                        out_access: combine_read_access(&self.out_access, &other.in_access),
                        any_barrier: false,
                        any_access: true,
                    }
                } else if !self.any_barrier {
                    // `R-R + R|X`
                    Self {
                        in_access: combine_read_access(&self.out_access, &other.in_access),
                        out_access: other.out_access,
                        any_barrier: true,
                        any_access: true,
                    }
                } else if !other.any_barrier {
                    // `X|R + R-R`
                    Self {
                        in_access: self.in_access,
                        out_access: combine_read_access(&self.out_access, &other.in_access),
                        any_barrier: true,
                        any_access: true,
                    }
                } else {
                    // `X|R + R|X`
                    Self {
                        in_access: self.in_access,
                        out_access: other.out_access,
                        any_barrier: true,
                        any_access: true,
                    }
                },
                None,
            );
        };
    }
}

/// Represents change in image access as a result of executing some commands (possibly with barriers)
///
/// This is an extension over PipelineMemoryAccessDelta that also handles image layout transitions
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PipelineImageAccessDelta {
    /// Corresponding memory access delta
    pub memory_access_delta: PipelineMemoryAccessDelta,
    /// Layout that image should have before executing corresponding batch of commands
    pub in_layout: ImageLayout,
    /// Layout that image will have after executing corresponding batch of commands
    pub out_layout: ImageLayout,
}

impl PipelineImageAccessDelta {
    fn then(
        &self,
        other: &PipelineImageAccessDelta,
    ) -> (PipelineImageAccessDelta, Option<ImageBarrierDescription>) {
        let (mut memory_delta, memory_barrier) =
            self.memory_access_delta.then(&other.memory_access_delta);

        let mut old_layout = self.out_layout;
        let mut new_layout = other.in_layout;

        if old_layout == ImageLayout::Undefined {
            old_layout = new_layout;
        }

        if new_layout == ImageLayout::Undefined {
            new_layout = old_layout;
        }

        let barrier = if old_layout != new_layout && memory_barrier.is_none() {
            Some(ImageBarrierDescription {
                source_stages: self.memory_access_delta.out_access.stages,
                source_access: self.memory_access_delta.out_access.access,
                destination_stages: other.memory_access_delta.in_access.stages,
                destination_access: other.memory_access_delta.in_access.access,
                old_layout,
                new_layout,
            })
        } else if let Some(memory_barrier) = memory_barrier.clone() {
            Some(ImageBarrierDescription {
                source_stages: memory_barrier.source_stages,
                source_access: memory_barrier.source_access,
                destination_stages: memory_barrier.destination_stages,
                destination_access: memory_barrier.destination_access,
                old_layout,
                new_layout,
            })
        } else {
            None
        };

        if let Some(barrier) = &barrier {
            // TODO in this PR: panic?
            debug_assert!(barrier.new_layout != ImageLayout::Undefined);
            debug_assert!(barrier.new_layout != ImageLayout::Preinitialized);
        }

        memory_delta.any_barrier |= barrier.is_some();

        (
            PipelineImageAccessDelta {
                memory_access_delta: memory_delta,
                in_layout: self.in_layout,
                out_layout: other.out_layout,
            },
            barrier,
        )
    }
}

/// Memory access delta but over entire range of a buffer
#[derive(Clone)]
pub(crate) struct BufferAccessDelta {
    pub range_map: RangeMap<DeviceSize, PipelineMemoryAccessDelta>,
}

impl BufferAccessDelta {
    pub fn then(
        &self,
        other: &BufferAccessDelta,
        buffer: &Arc<UnsafeBuffer>,
    ) -> (BufferAccessDelta, SmallVec<[BufferMemoryBarrier; 4]>) {
        let mut result_ranges = vec![];
        let mut barriers = smallvec![];

        for (range, delta_self) in self.range_map.range(&(0..buffer.size())) {
            for (sub_range, delta_other) in other.range_map.range(range) {
                let (delta, barrier) = delta_self.then(delta_other);
                if let Some(barrier) = barrier {
                    barriers.push(BufferMemoryBarrier {
                        source_stages: barrier.source_stages,
                        source_access: barrier.source_access,
                        destination_stages: barrier.destination_stages,
                        destination_access: barrier.destination_access,
                        range: (*sub_range).clone(),
                        ..BufferMemoryBarrier::buffer(buffer.clone())
                    });
                }
                result_ranges.push(((*sub_range).clone(), delta))
            }
        }

        (
            BufferAccessDelta {
                range_map: result_ranges.into_iter().collect(),
            },
            barriers,
        )
    }
}

/// Memory access delta but over entire range of an image
#[derive(Clone)]
pub(crate) struct ImageAccessDelta {
    pub range_map: RangeMap<DeviceSize, PipelineImageAccessDelta>,
}

impl ImageAccessDelta {
    pub fn then(
        &self,
        other: &ImageAccessDelta,
        image: &Arc<UnsafeImage>,
    ) -> (ImageAccessDelta, SmallVec<[ImageMemoryBarrier; 4]>) {
        let mut result_ranges = vec![];
        let mut barriers = smallvec![];

        for (range, delta_self) in self.range_map.range(&(0..image.range_size())) {
            for (sub_range, delta_other) in other.range_map.range(range) {
                let (delta, barrier) = delta_self.then(delta_other);

                if let Some(barrier) = barrier {
                    barriers.push(ImageMemoryBarrier {
                        source_stages: barrier.source_stages,
                        source_access: barrier.source_access,
                        destination_stages: barrier.destination_stages,
                        destination_access: barrier.destination_access,
                        old_layout: barrier.old_layout,
                        new_layout: barrier.new_layout,
                        subresource_range: image.range_to_subresources((*sub_range).clone()),
                        ..ImageMemoryBarrier::image(image.clone())
                    });
                }
                result_ranges.push(((*sub_range).clone(), delta))
            }
        }

        (
            ImageAccessDelta {
                range_map: result_ranges.into_iter().collect(),
            },
            barriers,
        )
    }
}

/// Access delta for some set of resources that represents change in access after executing
/// some batch of commands.
#[derive(Clone)]
pub(crate) struct ResourcesAccessDelta {
    pub buffer_access_deltas: HashMap<Arc<UnsafeBuffer>, BufferAccessDelta>,
    pub image_access_deltas: HashMap<Arc<UnsafeImage>, ImageAccessDelta>,
}

impl ResourcesAccessDelta {
    pub fn empty() -> Self {
        Self {
            buffer_access_deltas: HashMap::default(),
            image_access_deltas: HashMap::default(),
        }
    }

    pub fn then(
        &self,
        other: &ResourcesAccessDelta,
    ) -> (ResourcesAccessDelta, Option<DependencyInfo>) {
        let mut buffer_access_deltas = HashMap::default();
        let mut image_access_deltas = HashMap::default();
        let mut dependency_info = DependencyInfo::default();

        for (buffer, self_delta) in self.buffer_access_deltas.iter() {
            if let Some(other_delta) = other.buffer_access_deltas.get(buffer) {
                let (state, mut barriers) = self_delta.then(other_delta, buffer);
                dependency_info.buffer_memory_barriers.append(&mut barriers);
                buffer_access_deltas.insert(buffer.clone(), state);
            } else {
                buffer_access_deltas.insert(buffer.clone(), self_delta.clone());
            }
        }

        for buffer in other.buffer_access_deltas.keys() {
            if !self.buffer_access_deltas.contains_key(buffer) {
                buffer_access_deltas.insert(
                    buffer.clone(),
                    (*other.buffer_access_deltas.get(buffer).unwrap()).clone(),
                );
            }
        }

        for (image, self_delta) in self.image_access_deltas.iter() {
            if let Some(other_delta) = other.image_access_deltas.get(image) {
                let (state, mut barriers) = self_delta.then(other_delta, image);
                dependency_info.image_memory_barriers.append(&mut barriers);
                image_access_deltas.insert(image.clone(), state);
            } else {
                image_access_deltas.insert(image.clone(), self_delta.clone());
            }
        }

        for image in other.image_access_deltas.keys() {
            if !self.image_access_deltas.contains_key(image) {
                image_access_deltas.insert(
                    image.clone(),
                    (*other.image_access_deltas.get(image).unwrap()).clone(),
                );
            }
        }

        let dependency_info = if dependency_info.image_memory_barriers.is_empty()
            && dependency_info.buffer_memory_barriers.is_empty()
        {
            None
        } else {
            Some(dependency_info)
        };

        return (
            ResourcesAccessDelta {
                buffer_access_deltas,
                image_access_deltas,
            },
            dependency_info,
        );
    }

    pub fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess>,
        mut range: Range<DeviceSize>,
        memory: PipelineMemoryAccess,
    ) -> Result<(), ()> {
        let inner = buffer.inner();
        range.start += inner.offset;
        range.end += inner.offset;

        if self.buffer_access_deltas.contains_key(inner.buffer) {
            return Err(());
        }

        let mut buffer_access_delta = BufferAccessDelta {
            range_map: [(
                0..inner.buffer.size(),
                PipelineMemoryAccessDelta {
                    in_access: PipelineMemoryAccess::default(),
                    out_access: PipelineMemoryAccess::default(),
                    any_barrier: false,
                    any_access: false,
                },
            )]
            .into_iter()
            .collect(),
        };

        for (range, delta) in &mut buffer_access_delta.range_map.range_mut(&range) {
            delta.any_access = true;
            delta.in_access = memory.clone();
            delta.out_access = memory.clone();
        }

        self.buffer_access_deltas
            .insert(inner.buffer.clone(), buffer_access_delta);

        Ok(())
    }

    // We have two parameters memory_in and memory_out here so that we can insert fictional
    // deltas for initial and final image layout transition in primary command buffer
    // Normally you should use the same PipelineMemoryAccess
    pub fn add_image(
        &mut self,
        image: Arc<dyn ImageAccess>,
        mut subresource_range: ImageSubresourceRange,
        memory_in: PipelineMemoryAccess,
        memory_out: PipelineMemoryAccess,
        start_layout: ImageLayout,
        end_layout: ImageLayout,
    ) -> Result<(), ()> {
        let inner = image.inner();
        subresource_range.array_layers.start += inner.first_layer;
        subresource_range.array_layers.end += inner.first_layer;
        subresource_range.mip_levels.start += inner.first_mipmap_level;
        subresource_range.mip_levels.end += inner.first_mipmap_level;

        if self.image_access_deltas.contains_key(inner.image) {
            return Err(());
        }

        let mut image_access_delta = ImageAccessDelta {
            range_map: [(
                0..inner.image.range_size(),
                PipelineImageAccessDelta {
                    memory_access_delta: PipelineMemoryAccessDelta {
                        in_access: PipelineMemoryAccess::default(),
                        out_access: PipelineMemoryAccess::default(),
                        any_barrier: false,
                        any_access: false,
                    },
                    in_layout: ImageLayout::Undefined,
                    out_layout: ImageLayout::Undefined,
                },
            )]
            .into_iter()
            .collect(),
        };

        for range in inner.image.iter_ranges(subresource_range) {
            for (range, delta) in &mut image_access_delta.range_map.range_mut(&range) {
                delta.memory_access_delta.any_access = true;
                delta.memory_access_delta.in_access = memory_in.clone();
                delta.memory_access_delta.out_access = memory_out.clone();
                // TODO in this PR: maybe ad another flag? We do not have an actual barrier here
                delta.memory_access_delta.any_barrier = memory_out != memory_in;
                delta.in_layout = start_layout;
                delta.out_layout = end_layout;
            }
        }

        self.image_access_deltas
            .insert(inner.image.clone(), image_access_delta);

        Ok(())
    }

    pub fn add_resource(&mut self, resource: &Resource) {
        match (*resource).clone() {
            Resource::Buffer {
                buffer,
                range,
                memory,
            } => {
                self.add_buffer(buffer.clone(), range, memory).unwrap();
            }
            Resource::Image {
                image,
                subresource_range,
                memory,
                start_layout,
                end_layout,
            } => {
                self.add_image(
                    image,
                    subresource_range,
                    memory,
                    memory,
                    start_layout,
                    end_layout,
                )
                .unwrap();
            }
        }
    }
}
