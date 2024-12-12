//! The task graph data structure and associated types.

pub use self::{
    compile::{CompileError, CompileErrorKind, CompileInfo},
    execute::{ExecuteError, ResourceMap},
};
use crate::{
    resource::{self, AccessType, Flight, HostAccessType, ImageLayoutType},
    Id, InvalidSlotError, Object, ObjectType, QueueFamilyType, Task,
};
use ash::vk;
use concurrent_slotmap::{IterMut, IterUnprotected, SlotId, SlotMap};
use foldhash::HashMap;
use smallvec::SmallVec;
use std::{
    borrow::Cow, cell::RefCell, error::Error, fmt, hint, iter::FusedIterator, ops::Range, sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo},
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageCreateInfo, ImageLayout},
    swapchain::{Swapchain, SwapchainCreateInfo},
    sync::{semaphore::Semaphore, AccessFlags, PipelineStages},
};

mod compile;
mod execute;

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
    // TODO:
    #[allow(unused)]
    name: Cow<'static, str>,
    inner: NodeInner<W>,
    in_edges: Vec<NodeIndex>,
    out_edges: Vec<NodeIndex>,
}

enum NodeInner<W: ?Sized> {
    Task(TaskNode<W>),
    // TODO:
    #[allow(unused)]
    Semaphore,
}

type NodeIndex = u32;

pub(crate) struct Resources {
    inner: SlotMap<()>,
    physical_resources: Arc<resource::Resources>,
    physical_map: HashMap<Id, Id>,
    host_reads: Vec<Id<Buffer>>,
    host_writes: Vec<Id<Buffer>>,
}

impl<W: ?Sized> TaskGraph<W> {
    /// Creates a new `TaskGraph`.
    ///
    /// `max_nodes` is the maximum number of nodes the graph can ever have. `max_resources` is the
    /// maximum number of virtual resources the graph can ever have.
    #[must_use]
    pub fn new(
        physical_resources: &Arc<resource::Resources>,
        max_nodes: u32,
        max_resources: u32,
    ) -> Self {
        TaskGraph {
            nodes: Nodes {
                inner: SlotMap::new(max_nodes),
            },
            resources: Resources {
                inner: SlotMap::new(max_resources),
                physical_resources: physical_resources.clone(),
                physical_map: HashMap::default(),
                host_reads: Vec::new(),
                host_writes: Vec::new(),
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
            resources: &mut self.resources,
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

    /// Adds a host buffer access to this task graph.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    pub fn add_host_buffer_access(&mut self, id: Id<Buffer>, access_type: HostAccessType) {
        self.resources.add_host_buffer_access(id, access_type)
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
        let mut tag = Buffer::TAG | Id::VIRTUAL_BIT;

        if create_info.sharing.is_exclusive() {
            tag |= Id::EXCLUSIVE_BIT;
        }

        let slot = self.inner.insert_with_tag_mut((), tag);

        unsafe { Id::new(slot) }
    }

    fn add_image(&mut self, create_info: &ImageCreateInfo) -> Id<Image> {
        let mut tag = Image::TAG | Id::VIRTUAL_BIT;

        if create_info.sharing.is_exclusive() {
            tag |= Id::EXCLUSIVE_BIT;
        }

        let slot = self.inner.insert_with_tag_mut((), tag);

        unsafe { Id::new(slot) }
    }

    fn add_swapchain(&mut self, create_info: &SwapchainCreateInfo) -> Id<Swapchain> {
        let mut tag = Swapchain::TAG | Id::VIRTUAL_BIT;

        if create_info.image_sharing.is_exclusive() {
            tag |= Id::EXCLUSIVE_BIT;
        }

        let slot = self.inner.insert_with_tag_mut((), tag);

        unsafe { Id::new(slot) }
    }

    fn add_physical_buffer(
        &mut self,
        physical_id: Id<Buffer>,
    ) -> Result<Id<Buffer>, InvalidSlotError> {
        let physical_resources = self.physical_resources.clone();
        let buffer_state = physical_resources.buffer(physical_id)?;
        let buffer = buffer_state.buffer();
        let virtual_id = self.add_buffer(&BufferCreateInfo {
            sharing: buffer.sharing().clone(),
            ..Default::default()
        });
        self.physical_map
            .insert(physical_id.erase(), virtual_id.erase());

        Ok(virtual_id)
    }

    fn add_physical_image(
        &mut self,
        physical_id: Id<Image>,
    ) -> Result<Id<Image>, InvalidSlotError> {
        let physical_resources = self.physical_resources.clone();
        let image_state = physical_resources.image(physical_id)?;
        let image = image_state.image();
        let virtual_id = self.add_image(&ImageCreateInfo {
            sharing: image.sharing().clone(),
            ..Default::default()
        });
        self.physical_map
            .insert(physical_id.erase(), virtual_id.erase());

        Ok(virtual_id)
    }

    fn add_physical_swapchain(
        &mut self,
        id: Id<Swapchain>,
    ) -> Result<Id<Swapchain>, InvalidSlotError> {
        let physical_resources = self.physical_resources.clone();
        let swapchain_state = physical_resources.swapchain(id)?;
        let swapchain = swapchain_state.swapchain();
        let virtual_id = self.add_swapchain(&SwapchainCreateInfo {
            image_sharing: swapchain.image_sharing().clone(),
            ..Default::default()
        });
        self.physical_map.insert(id.erase(), virtual_id.erase());

        Ok(virtual_id)
    }

    fn add_host_buffer_access(&mut self, mut id: Id<Buffer>, access_type: HostAccessType) {
        if id.is_virtual() {
            self.get(id.erase()).expect("invalid buffer");
        } else if let Some(&virtual_id) = self.physical_map.get(&id.erase()) {
            id = unsafe { virtual_id.parametrize() };
        } else {
            id = self.add_physical_buffer(id).expect("invalid buffer");
        }

        let host_accesses = match access_type {
            HostAccessType::Read => &mut self.host_reads,
            HostAccessType::Write => &mut self.host_writes,
        };

        if !host_accesses.contains(&id) {
            host_accesses.push(id);
        }
    }

    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn get(&self, id: Id) -> Result<&(), InvalidSlotError> {
        // SAFETY: We never modify the map concurrently.
        unsafe { self.inner.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))
    }

    fn iter(&self) -> impl Iterator<Item = (Id, &())> {
        // SAFETY: We never modify the map concurrently.
        unsafe { self.inner.iter_unprotected() }.map(|(slot, v)| (unsafe { Id::new(slot) }, v))
    }

    pub(crate) fn contains_host_buffer_access(
        &self,
        mut id: Id<Buffer>,
        access_type: HostAccessType,
    ) -> bool {
        if !id.is_virtual() {
            if let Some(&virtual_id) = self.physical_map.get(&id.erase()) {
                id = unsafe { virtual_id.parametrize() };
            } else {
                return false;
            }
        }

        let host_accesses = match access_type {
            HostAccessType::Read => &self.host_reads,
            HostAccessType::Write => &self.host_writes,
        };

        host_accesses.contains(&id)
    }
}

impl<W: ?Sized> fmt::Debug for TaskGraph<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME:
        f.debug_struct("TaskGraph").finish_non_exhaustive()
    }
}

unsafe impl<W: ?Sized> DeviceOwned for TaskGraph<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.resources.physical_resources.device()
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
    inner: Vec<(Id, ResourceAccess)>,
}

#[derive(Clone, Copy, Default)]
struct ResourceAccess {
    stage_mask: PipelineStages,
    access_mask: AccessFlags,
    image_layout: ImageLayout,
    queue_family_index: u32,
}

impl<W: ?Sized> TaskNode<W> {
    fn new(queue_family_type: QueueFamilyType, task: impl Task<World = W>) -> Self {
        TaskNode {
            accesses: ResourceAccesses::new(),
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
}

impl ResourceAccesses {
    pub(crate) const fn new() -> Self {
        ResourceAccesses { inner: Vec::new() }
    }

    fn get_mut(
        &mut self,
        resources: &mut Resources,
        mut id: Id,
    ) -> Result<(Id, Option<&mut ResourceAccess>), InvalidSlotError> {
        if id.is_virtual() {
            resources.get(id)?;
        } else if let Some(&virtual_id) = resources.physical_map.get(&id) {
            id = virtual_id;
        } else {
            id = match id.object_type() {
                ObjectType::Buffer => resources
                    .add_physical_buffer(unsafe { id.parametrize() })?
                    .erase(),
                ObjectType::Image => resources
                    .add_physical_image(unsafe { id.parametrize() })?
                    .erase(),
                ObjectType::Swapchain => resources
                    .add_physical_swapchain(unsafe { id.parametrize() })?
                    .erase(),
                _ => unreachable!(),
            };
        }

        let access = self
            .iter_mut()
            .find_map(|(x, access)| (x == id).then_some(access));

        Ok((id, access))
    }

    fn iter(&self) -> impl Iterator<Item = (Id, &ResourceAccess)> {
        self.inner.iter().map(|(id, access)| (*id, access))
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = (Id, &mut ResourceAccess)> {
        self.inner.iter_mut().map(|(id, access)| (*id, access))
    }
}

/// A builder used to add resource accesses to a [`TaskNode`].
pub struct TaskNodeBuilder<'a, W: ?Sized> {
    id: NodeId,
    task_node: &'a mut TaskNode<W>,
    resources: &'a mut Resources,
}

impl<W: ?Sized> TaskNodeBuilder<'_, W> {
    /// Adds a buffer access to this task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_type` isn't a valid buffer access type.
    pub fn buffer_access(&mut self, id: Id<Buffer>, access_type: AccessType) -> &mut Self {
        let (id, access) = self.access_mut(id.erase()).expect("invalid buffer");

        assert!(access_type.is_valid_buffer_access_type());

        if let Some(access) = access {
            access.stage_mask |= access_type.stage_mask();
            access.access_mask |= access_type.access_mask();
        } else {
            self.task_node.accesses.inner.push((
                id.erase(),
                ResourceAccess {
                    stage_mask: access_type.stage_mask(),
                    access_mask: access_type.access_mask(),
                    image_layout: ImageLayout::Undefined,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            ));
        }

        self
    }

    /// Adds an image access to this task node.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid virtual resource ID nor a valid physical ID.
    /// - Panics if `access_type` isn't a valid image access type.
    /// - Panics if an access for `id` was already added and its image layout doesn't equal
    ///   `access_type.image_layout(layout_type)`.
    pub fn image_access(
        &mut self,
        id: Id<Image>,
        access_type: AccessType,
        layout_type: ImageLayoutType,
    ) -> &mut Self {
        let (id, access) = self.access_mut(id.erase()).expect("invalid image");

        assert!(access_type.is_valid_image_access_type());

        let image_layout = access_type.image_layout(layout_type);

        if let Some(access) = access {
            assert_eq!(access.image_layout, image_layout);

            access.stage_mask |= access_type.stage_mask();
            access.access_mask |= access_type.access_mask();
        } else {
            self.task_node.accesses.inner.push((
                id.erase(),
                ResourceAccess {
                    stage_mask: access_type.stage_mask(),
                    access_mask: access_type.access_mask(),
                    image_layout,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            ));
        }

        self
    }

    fn access_mut(
        &mut self,
        id: Id,
    ) -> Result<(Id, Option<&mut ResourceAccess>), InvalidSlotError> {
        self.task_node.accesses.get_mut(self.resources, id)
    }

    /// Finishes building the task node and returns the ID of the built node.
    #[inline]
    pub fn build(&mut self) -> NodeId {
        self.id
    }
}

/// A [`TaskGraph`] that has been compiled into an executable form.
pub struct ExecutableTaskGraph<W: ?Sized> {
    graph: TaskGraph<W>,
    flight_id: Id<Flight>,
    instructions: Vec<Instruction>,
    submissions: Vec<Submission>,
    buffer_barriers: Vec<BufferMemoryBarrier>,
    image_barriers: Vec<ImageMemoryBarrier>,
    semaphores: RefCell<Vec<Arc<Semaphore>>>,
    swapchains: SmallVec<[Id<Swapchain>; 1]>,
    present_queue: Option<Arc<Queue>>,
    last_accesses: Vec<ResourceAccess>,
}

// FIXME: Initial queue family ownership transfers
#[derive(Debug)]
struct Submission {
    queue: Arc<Queue>,
    initial_buffer_barrier_range: Range<BarrierIndex>,
    initial_image_barrier_range: Range<BarrierIndex>,
    instruction_range: Range<InstructionIndex>,
}

type InstructionIndex = usize;

#[derive(Clone, Debug)]
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
    SignalPrePresent {
        swapchain_id: Id<Swapchain>,
        stage_mask: PipelineStages,
    },
    WaitPrePresent {
        swapchain_id: Id<Swapchain>,
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

#[derive(Clone, Debug)]
struct BufferMemoryBarrier {
    src_stage_mask: PipelineStages,
    src_access_mask: AccessFlags,
    dst_stage_mask: PipelineStages,
    dst_access_mask: AccessFlags,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    buffer: Id<Buffer>,
}

#[derive(Clone, Debug)]
struct ImageMemoryBarrier {
    src_stage_mask: PipelineStages,
    src_access_mask: AccessFlags,
    dst_stage_mask: PipelineStages,
    dst_access_mask: AccessFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: Id,
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

    /// Returns the flight ID that the task graph was compiled with.
    #[inline]
    pub fn flight_id(&self) -> Id<Flight> {
        self.flight_id
    }
}

impl<W: ?Sized> fmt::Debug for ExecutableTaskGraph<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ExecutableTaskGraph");

        debug
            .field("graph", &self.graph)
            .field("flight_id", &self.flight_id)
            .field("instructions", &self.instructions)
            .field("submissions", &self.submissions)
            .field("buffer_barriers", &self.buffer_barriers)
            .field("image_barriers", &self.image_barriers)
            .field("semaphores", &self.semaphores)
            .field("swapchains", &self.swapchains)
            .field("present_queue", &self.present_queue)
            .finish_non_exhaustive()
    }
}

unsafe impl<W: ?Sized> DeviceOwned for ExecutableTaskGraph<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.graph.device()
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
    use crate::tests::test_queues;
    use std::marker::PhantomData;

    #[test]
    fn basic_usage1() {
        let (resources, _) = test_queues!();
        let mut graph = TaskGraph::<()>::new(&resources, 10, 0);

        let x = graph
            .create_task_node("X", QueueFamilyType::Graphics, PhantomData)
            .build();
        let y = graph
            .create_task_node("Y", QueueFamilyType::Graphics, PhantomData)
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
        let (resources, _) = test_queues!();
        let mut graph = TaskGraph::<()>::new(&resources, 10, 0);

        let x = graph
            .create_task_node("X", QueueFamilyType::Graphics, PhantomData)
            .build();
        let y = graph
            .create_task_node("Y", QueueFamilyType::Graphics, PhantomData)
            .build();
        let z = graph
            .create_task_node("Z", QueueFamilyType::Graphics, PhantomData)
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
        let (resources, _) = test_queues!();
        let mut graph = TaskGraph::<()>::new(&resources, 10, 0);

        let x = graph
            .create_task_node("X", QueueFamilyType::Graphics, PhantomData)
            .build();

        assert_eq!(graph.add_edge(x, x), Err(TaskGraphError::InvalidNode));
    }
}
