//! The task graph data structure and associated types.

use crate::{
    resource::{AccessType, BufferRange, ImageLayoutType},
    Id, InvalidSlotError, QueueFamilyType, Task, BUFFER_TAG, IMAGE_TAG, SWAPCHAIN_TAG,
};
use concurrent_slotmap::{IterMut, IterUnprotected, SlotId, SlotMap};
use smallvec::SmallVec;
use std::{borrow::Cow, error::Error, fmt, hint, iter::FusedIterator, ops::Range, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo},
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange,
    },
    swapchain::{Swapchain, SwapchainCreateInfo},
    sync::{semaphore::Semaphore, AccessFlags, PipelineStages},
    DeviceSize,
};

const EXCLUSIVE_BIT: u32 = 1 << 6;
const VIRTUAL_BIT: u32 = 1 << 7;

/// The task graph is a [directed acyclic graph] consisting of [`Task`] nodes, with edges
/// representing happens-before relations.
///
/// [directed acyclic graph]: https://en.wikipedia.org/wiki/Directed_acyclic_graph
pub struct TaskGraph<W: ?Sized> {
    nodes: Nodes<W>,
    resources: Resources,
}

struct Nodes<W: ?Sized> {
    inner: SlotMap<Node<W>>,
}

struct Node<W: ?Sized> {
    name: Cow<'static, str>,
    inner: NodeInner<W>,
    in_edges: Vec<NodeIndex>,
    out_edges: Vec<NodeIndex>,
}

enum NodeInner<W: ?Sized> {
    Task(TaskNode<W>),
    // TODO:
    Semaphore,
}

type NodeIndex = u32;

struct Resources {
    inner: SlotMap<ResourceInfo>,
}

#[derive(Clone, Copy)]
enum ResourceInfo {
    Buffer(BufferInfo),
    Image(ImageInfo),
    Swapchain(SwapchainInfo),
}

#[derive(Clone, Copy)]
struct BufferInfo {
    size: DeviceSize,
}

#[derive(Clone, Copy)]
struct ImageInfo {
    flags: ImageCreateFlags,
    format: Format,
    array_layers: u32,
    mip_levels: u32,
}

#[derive(Clone, Copy)]
struct SwapchainInfo {
    image_array_layers: u32,
}

impl<W: ?Sized> TaskGraph<W> {
    /// Creates a new `TaskGraph`.
    ///
    /// `max_nodes` is the maximum number of nodes the graph can ever have. `max_resources` is the
    /// maximum number of resources the graph can ever have.
    #[must_use]
    pub fn new(max_nodes: u32, max_resources: u32) -> Self {
        TaskGraph {
            nodes: Nodes {
                inner: SlotMap::new(max_nodes),
            },
            resources: Resources {
                inner: SlotMap::new(max_resources),
            },
        }
    }

    /// Creates a new [`TaskNode`] from the given `task` and adds it to the graph. Returns a
    /// builder allowing you to add resource accesses to the task node.
    ///
    /// `queue_family_type` is the type of queue family the task can be executed on.
    #[must_use]
    pub fn create_task_node(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        queue_family_type: QueueFamilyType,
        task: impl Task<World = W>,
    ) -> TaskNodeBuilder<'_, W> {
        let id = self.nodes.add_node(
            name.into(),
            NodeInner::Task(TaskNode::new(queue_family_type, task)),
        );

        // SAFETY: We just inserted this task node.
        let task_node = unsafe { self.nodes.task_node_unchecked_mut(id.index()) };

        TaskNodeBuilder {
            id,
            task_node,
            resources: &self.resources,
        }
    }

    /// Removes the task node corresponding to `id` from the graph.
    pub fn remove_task_node(&mut self, id: NodeId) -> Result<TaskNode<W>> {
        self.task_node(id)?;

        let task = match self.nodes.remove_node(id).inner {
            NodeInner::Task(task) => task,
            // We checked that the node is a task above.
            _ => unreachable!(),
        };

        Ok(task)
    }

    /// Adds an edge starting at the node corresponding to `from` and ending at the node
    /// corresponding to `to`.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result {
        let [from_node, to_node] = self.nodes.node_many_mut([from, to])?;
        let out_edges = &mut from_node.out_edges;
        let in_edges = &mut to_node.in_edges;

        if !out_edges.contains(&to.index()) {
            out_edges.push(to.index());
            in_edges.push(from.index());

            Ok(())
        } else {
            Err(TaskGraphError::DuplicateEdge)
        }
    }

    /// Removes an edge starting at the node corresponding to `from` and ending at the node
    /// corresponding to `to`.
    pub fn remove_edge(&mut self, from: NodeId, to: NodeId) -> Result {
        let [from_node, to_node] = self.nodes.node_many_mut([from, to])?;
        let out_edges = &mut from_node.out_edges;
        let in_edges = &mut to_node.in_edges;

        if let Some(index) = out_edges.iter().position(|&i| i == to.index()) {
            out_edges.remove(index);
            let edge_index = in_edges.iter().position(|&i| i == from.index()).unwrap();
            in_edges.remove(edge_index);

            Ok(())
        } else {
            Err(TaskGraphError::InvalidEdge)
        }
    }

    /// Returns a reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node(&self, id: NodeId) -> Result<&TaskNode<W>> {
        self.nodes.task_node(id)
    }

    /// Returns a mutable reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node_mut(&mut self, id: NodeId) -> Result<&mut TaskNode<W>> {
        self.nodes.task_node_mut(id)
    }

    /// Returns an iterator over all [`TaskNode`]s.
    #[inline]
    pub fn task_nodes(&self) -> TaskNodes<'_, W> {
        TaskNodes {
            inner: self.nodes.nodes(),
        }
    }

    /// Returns an iterator over all [`TaskNode`]s that allows you to mutate them.
    #[inline]
    pub fn task_nodes_mut(&mut self) -> TaskNodesMut<'_, W> {
        TaskNodesMut {
            inner: self.nodes.nodes_mut(),
        }
    }

    /// Add a [virtual buffer resource] to the task graph.
    #[must_use]
    pub fn add_buffer(&mut self, create_info: &BufferCreateInfo) -> Id<Buffer> {
        self.resources.add_buffer(create_info)
    }

    /// Add a [virtual image resource] to the task graph.
    #[must_use]
    pub fn add_image(&mut self, create_info: &ImageCreateInfo) -> Id<Image> {
        self.resources.add_image(create_info)
    }

    /// Add a [virtual swapchain resource] to the task graph.
    #[must_use]
    pub fn add_swapchain(&mut self, create_info: &SwapchainCreateInfo) -> Id<Swapchain> {
        self.resources.add_swapchain(create_info)
    }
}

impl<W: ?Sized> Nodes<W> {
    fn add_node(&mut self, name: Cow<'static, str>, inner: NodeInner<W>) -> NodeId {
        let slot = self.inner.insert_mut(Node {
            name,
            inner,
            in_edges: Vec::new(),
            out_edges: Vec::new(),
        });

        NodeId { slot }
    }

    fn remove_node(&mut self, id: NodeId) -> Node<W> {
        let node = self.inner.remove_mut(id.slot).unwrap();

        // NOTE(Marc): We must not leave any broken edges because the rest of the code relies on
        // this being impossible.

        for &in_node_index in &node.in_edges {
            // SAFETY: By our own invariant, node indices in the edges vectors must be valid.
            let out_edges = &mut unsafe { self.node_unchecked_mut(in_node_index) }.out_edges;
            let edge_index = out_edges.iter().position(|&i| i == id.index()).unwrap();
            out_edges.remove(edge_index);
        }

        for &out_node_index in &node.out_edges {
            // SAFETY: By our own invariant, node indices in the edges vectors must be valid.
            let in_edges = &mut unsafe { self.node_unchecked_mut(out_node_index) }.in_edges;
            let edge_index = in_edges.iter().position(|&i| i == id.index()).unwrap();
            in_edges.remove(edge_index);
        }

        node
    }

    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn task_node(&self, id: NodeId) -> Result<&TaskNode<W>> {
        let node = self.node(id)?;

        if let NodeInner::Task(task_node) = &node.inner {
            Ok(task_node)
        } else {
            Err(TaskGraphError::InvalidNodeType)
        }
    }

    unsafe fn task_node_unchecked(&self, index: NodeIndex) -> &TaskNode<W> {
        // SAFETY: The caller must ensure that the `index` is valid.
        let node = unsafe { self.node_unchecked(index) };

        if let NodeInner::Task(task_node) = &node.inner {
            task_node
        } else {
            // SAFETY: The caller must ensure that the node is a `TaskNode`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn task_node_mut(&mut self, id: NodeId) -> Result<&mut TaskNode<W>> {
        let node = self.node_mut(id)?;

        if let NodeInner::Task(task_node) = &mut node.inner {
            Ok(task_node)
        } else {
            Err(TaskGraphError::InvalidNodeType)
        }
    }

    unsafe fn task_node_unchecked_mut(&mut self, index: NodeIndex) -> &mut TaskNode<W> {
        // SAFETY: The caller must ensure that the `index` is valid.
        let node = unsafe { self.node_unchecked_mut(index) };

        if let NodeInner::Task(task_node) = &mut node.inner {
            task_node
        } else {
            // SAFETY: The caller must ensure that the node is a `TaskNode`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn node(&self, id: NodeId) -> Result<&Node<W>> {
        // SAFETY: We never modify the map concurrently.
        unsafe { self.inner.get_unprotected(id.slot) }.ok_or(TaskGraphError::InvalidNode)
    }

    unsafe fn node_unchecked(&self, index: NodeIndex) -> &Node<W> {
        // SAFETY:
        // * The caller must ensure that the `index` is valid.
        // * We never modify the map concurrently.
        unsafe { self.inner.index_unchecked_unprotected(index) }
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node<W>> {
        self.inner
            .get_mut(id.slot)
            .ok_or(TaskGraphError::InvalidNode)
    }

    unsafe fn node_unchecked_mut(&mut self, index: NodeIndex) -> &mut Node<W> {
        // SAFETY: The caller must ensure that the `index` is valid.
        unsafe { self.inner.index_unchecked_mut(index) }
    }

    fn node_many_mut<const N: usize>(&mut self, ids: [NodeId; N]) -> Result<[&mut Node<W>; N]> {
        union Transmute<const N: usize> {
            src: [NodeId; N],
            dst: [SlotId; N],
        }

        // HACK: `transmute_unchecked` is not exposed even unstably at the moment, and the compiler
        // isn't currently smart enough to figure this out using `transmute`.
        //
        // SAFETY: `NodeId` is `#[repr(transparent)]` over `SlotId` and both arrays have the same
        // length.
        let ids = unsafe { Transmute { src: ids }.dst };

        self.inner
            .get_many_mut(ids)
            .ok_or(TaskGraphError::InvalidNode)
    }

    fn nodes(&self) -> IterUnprotected<'_, Node<W>> {
        // SAFETY: We never modify the map concurrently.
        unsafe { self.inner.iter_unprotected() }
    }

    fn nodes_mut(&mut self) -> IterMut<'_, Node<W>> {
        self.inner.iter_mut()
    }
}

impl Resources {
    fn add_buffer(&mut self, create_info: &BufferCreateInfo) -> Id<Buffer> {
        let resource_info = ResourceInfo::Buffer(BufferInfo {
            size: create_info.size,
        });
        let mut tag = BUFFER_TAG | VIRTUAL_BIT;

        if create_info.sharing.is_exclusive() {
            tag |= EXCLUSIVE_BIT;
        }

        let slot = self.inner.insert_with_tag_mut(resource_info, tag);

        Id::new(slot)
    }

    fn add_image(&mut self, create_info: &ImageCreateInfo) -> Id<Image> {
        let resource_info = ResourceInfo::Image(ImageInfo {
            flags: create_info.flags,
            format: create_info.format,
            array_layers: create_info.array_layers,
            mip_levels: create_info.mip_levels,
        });
        let mut tag = IMAGE_TAG | VIRTUAL_BIT;

        if create_info.sharing.is_exclusive() {
            tag |= EXCLUSIVE_BIT;
        }

        let slot = self.inner.insert_with_tag_mut(resource_info, tag);

        Id::new(slot)
    }

    fn add_swapchain(&mut self, create_info: &SwapchainCreateInfo) -> Id<Swapchain> {
        let resource_info = ResourceInfo::Swapchain(SwapchainInfo {
            image_array_layers: create_info.image_array_layers,
        });
        let tag = SWAPCHAIN_TAG | VIRTUAL_BIT;

        let slot = self.inner.insert_with_tag_mut(resource_info, tag);

        Id::new(slot)
    }

    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn buffer(&self, id: Id<Buffer>) -> Result<&BufferInfo, InvalidSlotError> {
        // SAFETY: We never modify the map concurrently.
        let resource_info =
            unsafe { self.inner.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))?;

        if let ResourceInfo::Buffer(buffer) = resource_info {
            Ok(buffer)
        } else {
            // SAFETY: The `get_unprotected` call above already successfully compared the tag, so
            // there is no need to check it again. We always ensure that buffer IDs get tagged with
            // the `BUFFER_TAG`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    unsafe fn buffer_unchecked(&self, id: Id<Buffer>) -> &BufferInfo {
        // SAFETY:
        // * The caller must ensure that the `id` is valid.
        // * We never modify the map concurrently.
        let resource_info = unsafe { self.inner.index_unchecked_unprotected(id.index()) };

        if let ResourceInfo::Buffer(buffer) = resource_info {
            buffer
        } else {
            // SAFETY: The caller must ensure that the `id` is valid.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn image(&self, id: Id<Image>) -> Result<&ImageInfo, InvalidSlotError> {
        // SAFETY: We never modify the map concurrently.
        let resource_info =
            unsafe { self.inner.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))?;

        if let ResourceInfo::Image(image) = resource_info {
            Ok(image)
        } else {
            // SAFETY: The `get_unprotected` call above already successfully compared the tag, so
            // there is no need to check it again. We always ensure that image IDs get tagged with
            // the `IMAGE_TAG`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    unsafe fn image_unchecked(&self, id: Id<Image>) -> &ImageInfo {
        // SAFETY:
        // * The caller must ensure that the `index` is valid.
        // * We never modify the map concurrently.
        let resource_info = unsafe { self.inner.index_unchecked_unprotected(id.index()) };

        if let ResourceInfo::Image(image) = resource_info {
            image
        } else {
            // SAFETY: The caller must ensure that the `id` is valid.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn swapchain(&self, id: Id<Swapchain>) -> Result<&SwapchainInfo, InvalidSlotError> {
        // SAFETY: We never modify the map concurrently.
        let resource_info =
            unsafe { self.inner.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))?;

        if let ResourceInfo::Swapchain(swapchain) = resource_info {
            Ok(swapchain)
        } else {
            // SAFETY: The `get_unprotected` call above already successfully compared the tag, so
            // there is no need to check it again. We always ensure that swapchain IDs get tagged
            // with the `SWAPCHAIN_TAG`.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    unsafe fn swapchain_unchecked(&self, id: Id<Swapchain>) -> &SwapchainInfo {
        // SAFETY:
        // * The caller must ensure that the `index` is valid.
        // * We never modify the map concurrently.
        let resource_info = unsafe { self.inner.index_unchecked_unprotected(id.index()) };

        if let ResourceInfo::Swapchain(swapchain) = resource_info {
            swapchain
        } else {
            // SAFETY: The caller must ensure that the `id` is valid.
            unsafe { hint::unreachable_unchecked() }
        }
    }

    fn iter(&self) -> IterUnprotected<'_, ResourceInfo> {
        // SAFETY: We never modify the map concurrently.
        unsafe { self.inner.iter_unprotected() }
    }
}

/// The ID type used to refer to a node within a [`TaskGraph`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodeId {
    slot: SlotId,
}

impl NodeId {
    fn index(self) -> NodeIndex {
        self.slot.index()
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeId")
            .field("index", &self.slot.index())
            .field("generation", &self.slot.generation())
            .finish()
    }
}

/// A node within a [`TaskGraph`] that represents a [`Task`] to be executed along with its resource
/// accesses.
pub struct TaskNode<W: ?Sized> {
    accesses: ResourceAccesses,
    queue_family_type: QueueFamilyType,
    queue_family_index: u32,
    dependency_level_index: u32,
    task: Box<dyn Task<World = W>>,
}

pub(crate) struct ResourceAccesses {
    inner: Vec<ResourceAccess>,
}

// TODO: Literally anything else
#[derive(Clone)]
enum ResourceAccess {
    Buffer(BufferAccess),
    Image(ImageAccess),
    Swapchain(SwapchainAccess),
}

#[derive(Clone)]
struct BufferAccess {
    id: Id<Buffer>,
    range: BufferRange,
    access_type: AccessType,
}

#[derive(Clone)]
struct ImageAccess {
    id: Id<Image>,
    subresource_range: ImageSubresourceRange,
    access_type: AccessType,
    layout_type: ImageLayoutType,
}

#[derive(Clone)]
struct SwapchainAccess {
    id: Id<Swapchain>,
    array_layers: Range<u32>,
    access_type: AccessType,
    layout_type: ImageLayoutType,
}

impl<W: ?Sized> TaskNode<W> {
    fn new(queue_family_type: QueueFamilyType, task: impl Task<World = W>) -> Self {
        TaskNode {
            accesses: ResourceAccesses { inner: Vec::new() },
            queue_family_type,
            queue_family_index: 0,
            dependency_level_index: 0,
            task: Box::new(task),
        }
    }

    /// Returns the queue family type the task node was created with.
    #[inline]
    #[must_use]
    pub fn queue_family_type(&self) -> QueueFamilyType {
        self.queue_family_type
    }

    /// Returns a reference to the task the task node was created with.
    #[inline]
    #[must_use]
    pub fn task(&self) -> &dyn Task<World = W> {
        &*self.task
    }

    /// Returns a mutable reference to the task the task node was created with.
    #[inline]
    #[must_use]
    pub fn task_mut(&mut self) -> &mut dyn Task<World = W> {
        &mut *self.task
    }

    /// Returns `true` if the task node has access of the given `access_type` to the buffer
    /// corresponding to `id` where the given `range` is contained within the access's range.
    #[inline]
    #[must_use]
    pub fn contains_buffer_access(
        &self,
        id: Id<Buffer>,
        range: BufferRange,
        access_type: AccessType,
    ) -> bool {
        self.accesses.contains_buffer_access(id, range, access_type)
    }

    /// Returns `true` if the task node has access of the given `access_type` and `layout_type` to
    /// the image corresponding to `id` where the given `subresource_range` is contained within
    /// the access's subresource range.
    #[inline]
    #[must_use]
    pub fn contains_image_access(
        &self,
        id: Id<Image>,
        subresource_range: ImageSubresourceRange,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> bool {
        self.accesses
            .contains_image_access(id, subresource_range, access_type, layout_type)
    }

    /// Returns `true` if the task node has access of the given `access_type` and `layout_type` to
    /// the swapchain corresponding to `id` where the given `array_layers` are contained within
    /// the access's array layers.
    #[inline]
    #[must_use]
    pub fn contains_swapchain_access(
        &self,
        id: Id<Swapchain>,
        array_layers: Range<u32>,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> bool {
        self.accesses
            .contains_swapchain_access(id, array_layers, access_type, layout_type)
    }
}

impl ResourceAccesses {
    pub(crate) fn contains_buffer_access(
        &self,
        id: Id<Buffer>,
        range: BufferRange,
        access_type: AccessType,
    ) -> bool {
        debug_assert!(!range.is_empty());

        self.inner.iter().any(|resource_access| {
            matches!(resource_access, ResourceAccess::Buffer(a) if a.id == id
                && a.access_type == access_type
                && a.range.start <= range.start
                && range.end <= a.range.end)
        })
    }

    pub(crate) fn contains_image_access(
        &self,
        id: Id<Image>,
        subresource_range: ImageSubresourceRange,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> bool {
        debug_assert!(!subresource_range.aspects.is_empty());
        debug_assert!(!subresource_range.mip_levels.is_empty());
        debug_assert!(!subresource_range.array_layers.is_empty());

        self.inner.iter().any(|resource_access| {
            matches!(resource_access, ResourceAccess::Image(a) if a.id == id
                && a.access_type == access_type
                && a.layout_type == layout_type
                && a.subresource_range.aspects.contains(subresource_range.aspects)
                && a.subresource_range.mip_levels.start <= subresource_range.mip_levels.start
                && subresource_range.mip_levels.end <= a.subresource_range.mip_levels.end
                && a.subresource_range.array_layers.start <= subresource_range.array_layers.start
                && subresource_range.array_layers.end <= a.subresource_range.array_layers.end)
        })
    }

    pub(crate) fn contains_swapchain_access(
        &self,
        id: Id<Swapchain>,
        array_layers: Range<u32>,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> bool {
        debug_assert!(!array_layers.is_empty());

        self.inner.iter().any(|resource_access| {
            matches!(resource_access, ResourceAccess::Swapchain(a) if a.id == id
                && a.access_type == access_type
                && a.layout_type == layout_type
                && a.array_layers.start <= array_layers.start
                && array_layers.end <= a.array_layers.end)
        })
    }
}

/// A builder used to add resource accesses to a [`TaskNode`].
pub struct TaskNodeBuilder<'a, W: ?Sized> {
    id: NodeId,
    task_node: &'a mut TaskNode<W>,
    resources: &'a Resources,
}

impl<W: ?Sized> TaskNodeBuilder<'_, W> {
    /// Adds a buffer access to this task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID.
    /// - Panics if `range` doesn't denote a valid range of the buffer.
    /// - Panics if `access_type` isn't a valid buffer access type.
    pub fn buffer_access(
        &mut self,
        id: Id<Buffer>,
        range: BufferRange,
        access_type: AccessType,
    ) -> &mut Self {
        let buffer = self.resources.buffer(id).expect("invalid buffer");

        assert!(range.end <= buffer.size);
        assert!(!range.is_empty());

        assert!(access_type.is_valid_buffer_access_type());

        // SAFETY: We checked the safety preconditions above.
        unsafe { self.buffer_access_unchecked(id, range, access_type) }
    }

    /// Adds a buffer access to this task node without doing any checks.
    ///
    /// # Safety
    ///
    /// - `id` must be a valid virtual resource ID.
    /// - `range` must denote a valid range of the buffer.
    /// - `access_type` must be a valid buffer access type.
    #[inline]
    pub unsafe fn buffer_access_unchecked(
        &mut self,
        id: Id<Buffer>,
        range: BufferRange,
        access_type: AccessType,
    ) -> &mut Self {
        self.task_node
            .accesses
            .inner
            .push(ResourceAccess::Buffer(BufferAccess {
                id,
                range,
                access_type,
            }));

        self
    }

    /// Adds an image access to this task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID.
    /// - Panics if `subresource_range` doesn't denote a valid subresource range of the image.
    /// - Panics if `access_type` isn't a valid image access type.
    pub fn image_access(
        &mut self,
        id: Id<Image>,
        mut subresource_range: ImageSubresourceRange,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> &mut Self {
        let image = self.resources.image(id).expect("invalid image");

        if image.flags.contains(ImageCreateFlags::DISJOINT) {
            subresource_range.aspects -= ImageAspects::COLOR;
            subresource_range.aspects |= match image.format.planes().len() {
                2 => ImageAspects::PLANE_0 | ImageAspects::PLANE_1,
                3 => ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2,
                _ => unreachable!(),
            };
        }

        assert!(image.format.aspects().contains(subresource_range.aspects));
        assert!(subresource_range.mip_levels.end <= image.mip_levels);
        assert!(subresource_range.array_layers.end <= image.array_layers);
        assert!(!subresource_range.aspects.is_empty());
        assert!(!subresource_range.mip_levels.is_empty());
        assert!(!subresource_range.array_layers.is_empty());

        assert!(access_type.is_valid_image_access_type());

        // SAFETY: We checked the safety preconditions above.
        unsafe { self.image_access_unchecked(id, subresource_range, access_type, layout_type) }
    }

    /// Adds an image access to this task node without doing any checks.
    ///
    /// # Safety
    ///
    /// - `id` must be a valid virtual resource ID.
    /// - `subresource_range` must denote a valid subresource range of the image. If the image
    ///   flags contain `ImageCreateFlags::DISJOINT`, then the color aspect is not considered
    ///   valid.
    /// - `access_type` must be a valid image access type.
    #[inline]
    pub unsafe fn image_access_unchecked(
        &mut self,
        id: Id<Image>,
        mut subresource_range: ImageSubresourceRange,
        access_type: AccessType,
        mut layout_type: ImageLayoutType,
    ) -> &mut Self {
        // Normalize the layout type so that comparisons of accesses are predictable.
        if access_type.image_layout() == ImageLayout::General {
            layout_type = ImageLayoutType::Optimal;
        }

        self.task_node
            .accesses
            .inner
            .push(ResourceAccess::Image(ImageAccess {
                id,
                subresource_range,
                access_type,
                layout_type,
            }));

        self
    }

    /// Adds a swapchain image access to this task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID.
    /// - Panics if `array_layers` doesn't denote a valid range of array layers of the swapchain.
    /// - Panics if `access_type` isn't a valid image access type.
    pub fn swapchain_access(
        &mut self,
        id: Id<Swapchain>,
        array_layers: Range<u32>,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> &mut Self {
        let swapchain = self.resources.swapchain(id).expect("invalid swapchain");

        assert!(array_layers.end <= swapchain.image_array_layers);
        assert!(!array_layers.is_empty());

        assert!(access_type.is_valid_image_access_type());

        // SAFETY: We checked the safety preconditions above.
        unsafe { self.swapchain_access_unchecked(id, array_layers, access_type, layout_type) }
    }

    /// Adds a swapchain image access to this task node without doing any checks.
    ///
    /// # Safety
    ///
    /// - `id` must be a valid virtual resource ID.
    /// - `array_layers` must denote a valid range of array layers of the swapchain.
    /// - `access_type` must be a valid image access type.
    #[inline]
    pub unsafe fn swapchain_access_unchecked(
        &mut self,
        id: Id<Swapchain>,
        array_layers: Range<u32>,
        access_type: AccessType,
        mut layout_type: ImageLayoutType,
    ) -> &mut Self {
        // Normalize the layout type so that comparisons of accesses are predictable.
        if access_type.image_layout() == ImageLayout::General {
            layout_type = ImageLayoutType::Optimal;
        }

        self.task_node
            .accesses
            .inner
            .push(ResourceAccess::Swapchain(SwapchainAccess {
                id,
                access_type,
                layout_type,
                array_layers,
            }));

        self
    }

    /// Finishes building the task node and returns the ID of the built node.
    #[inline]
    pub fn build(self) -> NodeId {
        self.id
    }
}

/// A [`TaskGraph`] that has been compiled into an executable form.
pub struct ExecutableTaskGraph<W: ?Sized> {
    graph: TaskGraph<W>,
    instructions: Vec<Instruction>,
    submissions: Vec<Submission>,
    buffer_barriers: Vec<BufferMemoryBarrier>,
    image_barriers: Vec<ImageMemoryBarrier>,
    semaphores: Vec<Semaphore>,
    swapchains: SmallVec<[Id<Swapchain>; 1]>,
    present_queue: Option<Arc<Queue>>,
}

// FIXME: Initial queue family ownership transfers
struct Submission {
    queue: Arc<Queue>,
    initial_buffer_barrier_range: Range<BarrierIndex>,
    initial_image_barrier_range: Range<BarrierIndex>,
    instruction_range: Range<InstructionIndex>,
}

type InstructionIndex = usize;

#[derive(Clone)]
enum Instruction {
    WaitAcquire {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    WaitSemaphore {
        semaphore_index: SemaphoreIndex,
        stage_mask: PipelineStages,
    },
    ExecuteTask {
        node_index: NodeIndex,
    },
    PipelineBarrier {
        buffer_barrier_range: Range<BarrierIndex>,
        image_barrier_range: Range<BarrierIndex>,
    },
    // TODO:
    // SetEvent {
    //     event_index: EventIndex,
    //     buffer_barriers: Range<BarrierIndex>,
    //     image_barriers: Range<BarrierIndex>,
    // },
    // WaitEvent {
    //     event_index: EventIndex,
    //     buffer_barriers: Range<BarrierIndex>,
    //     image_barriers: Range<BarrierIndex>,
    // },
    SignalSemaphore {
        semaphore_index: SemaphoreIndex,
        stage_mask: PipelineStages,
    },
    SignalPresent {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    FlushSubmit,
    Submit,
}

type SemaphoreIndex = usize;

type BarrierIndex = u32;

struct BufferMemoryBarrier {
    src_stage_mask: PipelineStages,
    src_access_mask: AccessFlags,
    dst_stage_mask: PipelineStages,
    dst_access_mask: AccessFlags,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    buffer: Id<Buffer>,
    range: BufferRange,
}

struct ImageMemoryBarrier {
    src_stage_mask: PipelineStages,
    src_access_mask: AccessFlags,
    dst_stage_mask: PipelineStages,
    dst_access_mask: AccessFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: ImageReference,
    subresource_range: ImageSubresourceRange,
}

// TODO: This really ought not to be necessary.
#[derive(Clone, Copy)]
enum ImageReference {
    Normal(Id<Image>),
    Swapchain(Id<Swapchain>),
}

impl<W: ?Sized> ExecutableTaskGraph<W> {
    /// Returns a reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node(&self, id: NodeId) -> Result<&TaskNode<W>> {
        self.graph.task_node(id)
    }

    /// Returns a mutable reference to the task node corresponding to `id`.
    #[inline]
    pub fn task_node_mut(&mut self, id: NodeId) -> Result<&mut TaskNode<W>> {
        self.graph.task_node_mut(id)
    }

    /// Returns an iterator over all [`TaskNode`]s.
    #[inline]
    pub fn task_nodes(&self) -> TaskNodes<'_, W> {
        self.graph.task_nodes()
    }

    /// Returns an iterator over all [`TaskNode`]s that allows you to mutate them.
    #[inline]
    pub fn task_nodes_mut(&mut self) -> TaskNodesMut<'_, W> {
        self.graph.task_nodes_mut()
    }
}

unsafe impl<W: ?Sized> DeviceOwned for ExecutableTaskGraph<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.submissions[0].queue.device()
    }
}

/// An iterator over all [`TaskNode`]s within a [`TaskGraph`].
///
/// This type is created by the [`task_nodes`] method on [`TaskGraph`].
///
/// [`task_nodes`]: TaskGraph::task_nodes
pub struct TaskNodes<'a, W: ?Sized> {
    inner: concurrent_slotmap::IterUnprotected<'a, Node<W>>,
}

impl<W: ?Sized> fmt::Debug for TaskNodes<'_, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskNodes").finish_non_exhaustive()
    }
}

impl<'a, W: ?Sized> Iterator for TaskNodes<'a, W> {
    type Item = &'a TaskNode<W>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next()?;

            if let NodeInner::Task(task_node) = &node.inner {
                break Some(task_node);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<W: ?Sized> DoubleEndedIterator for TaskNodes<'_, W> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next_back()?;

            if let NodeInner::Task(task_node) = &node.inner {
                break Some(task_node);
            }
        }
    }
}

impl<W: ?Sized> FusedIterator for TaskNodes<'_, W> {}

/// An iterator over all [`TaskNode`]s within a [`TaskGraph`] that allows you to mutate them.
///
/// This type is created by the [`task_nodes_mut`] method on [`TaskGraph`].
///
/// [`task_nodes_mut`]: TaskGraph::task_nodes_mut
pub struct TaskNodesMut<'a, W: ?Sized> {
    inner: concurrent_slotmap::IterMut<'a, Node<W>>,
}

impl<W: ?Sized> fmt::Debug for TaskNodesMut<'_, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskNodesMut").finish_non_exhaustive()
    }
}

impl<'a, W: ?Sized> Iterator for TaskNodesMut<'a, W> {
    type Item = &'a mut TaskNode<W>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next()?;

            if let NodeInner::Task(task_node) = &mut node.inner {
                break Some(task_node);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<W: ?Sized> DoubleEndedIterator for TaskNodesMut<'_, W> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (_, node) = self.inner.next_back()?;

            if let NodeInner::Task(task_node) = &mut node.inner {
                break Some(task_node);
            }
        }
    }
}

impl<W: ?Sized> FusedIterator for TaskNodesMut<'_, W> {}

type Result<T = (), E = TaskGraphError> = ::std::result::Result<T, E>;

/// Error that can happen when doing operations on a [`TaskGraph`].
#[derive(Debug, PartialEq, Eq)]
pub enum TaskGraphError {
    InvalidNode,
    InvalidNodeType,
    InvalidEdge,
    DuplicateEdge,
}

impl fmt::Display for TaskGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::InvalidNode => "a node with the given ID does not exist",
            Self::InvalidNodeType => {
                "the node with the given ID has a type that is incompatible with the operation"
            }
            Self::InvalidEdge => "an edge between the given nodes does not exist",
            Self::DuplicateEdge => "an edge between the given nodes already exists",
        };

        f.write_str(msg)
    }
}

impl Error for TaskGraphError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TaskContext, TaskResult};

    struct DummyTask;

    impl Task for DummyTask {
        type World = ();

        unsafe fn execute(&self, _tcx: &mut TaskContext<'_>, _world: &Self::World) -> TaskResult {
            Ok(())
        }
    }

    #[test]
    fn basic_usage1() {
        let mut graph = TaskGraph::new(10, 0);

        let x = graph
            .create_task_node("", QueueFamilyType::Graphics, DummyTask)
            .build();
        let y = graph
            .create_task_node("", QueueFamilyType::Graphics, DummyTask)
            .build();

        graph.add_edge(x, y).unwrap();
        assert!(graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(graph.nodes.node(y).unwrap().in_edges.contains(&x.index()));
        assert_eq!(graph.add_edge(x, y), Err(TaskGraphError::DuplicateEdge));

        graph.remove_edge(x, y).unwrap();
        assert!(!graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(!graph.nodes.node(y).unwrap().in_edges.contains(&x.index()));

        assert_eq!(graph.remove_edge(x, y), Err(TaskGraphError::InvalidEdge));

        graph.add_edge(y, x).unwrap();
        assert!(graph.nodes.node(y).unwrap().out_edges.contains(&x.index()));
        assert!(graph.nodes.node(x).unwrap().in_edges.contains(&y.index()));
        assert_eq!(graph.add_edge(y, x), Err(TaskGraphError::DuplicateEdge));

        graph.remove_edge(y, x).unwrap();
        assert!(!graph.nodes.node(y).unwrap().out_edges.contains(&x.index()));
        assert!(!graph.nodes.node(x).unwrap().in_edges.contains(&y.index()));

        assert_eq!(graph.remove_edge(y, x), Err(TaskGraphError::InvalidEdge));
    }

    #[test]
    fn basic_usage2() {
        let mut graph = TaskGraph::new(10, 0);

        let x = graph
            .create_task_node("", QueueFamilyType::Graphics, DummyTask)
            .build();
        let y = graph
            .create_task_node("", QueueFamilyType::Graphics, DummyTask)
            .build();
        let z = graph
            .create_task_node("", QueueFamilyType::Graphics, DummyTask)
            .build();

        assert!(graph.task_node(x).is_ok());
        assert!(graph.task_node(y).is_ok());
        assert!(graph.task_node(z).is_ok());
        assert!(graph.task_node_mut(x).is_ok());
        assert!(graph.task_node_mut(y).is_ok());
        assert!(graph.task_node_mut(z).is_ok());

        graph.add_edge(x, y).unwrap();
        graph.add_edge(y, z).unwrap();
        assert!(graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(graph.nodes.node(z).unwrap().in_edges.contains(&y.index()));

        graph.remove_task_node(y).unwrap();
        assert!(!graph.nodes.node(x).unwrap().out_edges.contains(&y.index()));
        assert!(!graph.nodes.node(z).unwrap().in_edges.contains(&y.index()));

        assert!(matches!(
            graph.remove_task_node(y),
            Err(TaskGraphError::InvalidNode),
        ));
    }

    #[test]
    fn self_referential_node() {
        let mut graph = TaskGraph::new(10, 0);

        let x = graph
            .create_task_node("", QueueFamilyType::Graphics, DummyTask)
            .build();

        assert_eq!(graph.add_edge(x, x), Err(TaskGraphError::InvalidNode));
    }
}
