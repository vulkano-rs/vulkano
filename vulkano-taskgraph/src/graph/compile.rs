// FIXME: host read barriers

use self::linear_map::LinearMap;
use super::{
    BarrierIndex, ExecutableTaskGraph, Instruction, NodeIndex, NodeInner, ResourceAccess,
    SemaphoreIndex, Submission, TaskGraph,
};
use crate::{resource::Flight, Id, ObjectType, QueueFamilyType};
use ash::vk;
use smallvec::{smallvec, SmallVec};
use std::{cell::RefCell, cmp, error::Error, fmt, mem, ops::Range, sync::Arc};
use vulkano::{
    device::{Device, DeviceOwned, Queue, QueueFlags},
    image::{Image, ImageLayout},
    swapchain::Swapchain,
    sync::{semaphore::Semaphore, AccessFlags, PipelineStages},
    VulkanError,
};

impl<W: ?Sized> TaskGraph<W> {
    /// Compiles the task graph into an executable form.
    ///
    /// # Safety
    ///
    /// - There must be no conflicting device accesses in task nodes with no path between them.
    /// - There must be no accesses that are incompatible with the queue family type of the task
    ///   node.
    /// - There must be no accesses that are unsupported by the device.
    ///
    /// # Panics
    ///
    /// - Panics if `compile_info.queues` is empty.
    /// - Panics if the device of any queue in `compile_info.queues` or
    ///   `compile_info.present_queue` is not the same as that of `self`.
    /// - Panics if `compile_info.queues` contains duplicate queue families.
    /// - Panics if `compile_info.present_queue` is `None` and the task graph uses any swapchains.
    ///
    /// # Errors
    ///
    /// In order to get a successful compilation, the graph must satisfy the following conditions:
    /// - It must be [weakly connected]: every node must be able to reach every other node when
    ///   disregarding the direction of the edges.
    /// - It must have no [directed cycles]: if you were to walk starting from any node following
    ///   the direction of the edges, there must be no way to end up at the node you started at.
    ///
    /// [weakly connected]: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_vertices_and_graphs
    /// [directed cycles]: https://en.wikipedia.org/wiki/Cycle_(graph_theory)#Directed_circuit_and_directed_cycle
    pub unsafe fn compile(
        mut self,
        compile_info: &CompileInfo<'_>,
    ) -> Result<ExecutableTaskGraph<W>, CompileError<W>> {
        let &CompileInfo {
            queues,
            present_queue,
            flight_id,
            _ne: _,
        } = compile_info;

        assert_ne!(queues.len(), 0, "expected to be given at least one queue");

        let device = &self.device().clone();

        for queue in queues {
            assert_eq!(queue.device(), device);
            assert_eq!(
                queues
                    .iter()
                    .filter(|q| q.queue_family_index() == queue.queue_family_index())
                    .count(),
                1,
                "expected each queue in `compile_info.queues` to be from a unique queue family",
            );
        }

        if let Some(present_queue) = &present_queue {
            assert_eq!(present_queue.device(), device);
        }

        if !self.is_weakly_connected() {
            return Err(CompileError::new(self, CompileErrorKind::Unconnected));
        }

        let topological_order = match self.topological_sort() {
            Ok(topological_order) => topological_order,
            Err(kind) => return Err(CompileError::new(self, kind)),
        };
        unsafe { self.dependency_levels(&topological_order) };
        let queue_family_indices =
            match unsafe { self.queue_family_indices(device, queues, &topological_order) } {
                Ok(queue_family_indices) => queue_family_indices,
                Err(kind) => return Err(CompileError::new(self, kind)),
            };
        let mut queues_by_queue_family_index: SmallVec<[_; 8]> =
            smallvec![None; *queue_family_indices.iter().max().unwrap() as usize + 1];

        for &queue in queues {
            if let Some(x) =
                queues_by_queue_family_index.get_mut(queue.queue_family_index() as usize)
            {
                *x = Some(queue);
            }
        }

        let mut prev_accesses = vec![ResourceAccess::default(); self.resources.capacity() as usize];
        let mut barrier_stages = vec![BarrierStage::Stage0; self.resources.capacity() as usize];

        let (node_meta, semaphore_count, last_swapchain_accesses) = unsafe {
            self.node_metadata(&topological_order, &mut prev_accesses, &mut barrier_stages)
        };

        prev_accesses.fill(ResourceAccess::default());
        barrier_stages.fill(BarrierStage::Stage0);

        let mut state = CompileState::new(&mut prev_accesses, present_queue);
        let mut prev_submission_end = 0;

        while prev_submission_end < topological_order.len() {
            // First per-submission pass: compute the initial barriers for the submission.
            for (i, &node_index) in
                (prev_submission_end..).zip(&topological_order[prev_submission_end..])
            {
                let node = unsafe { self.nodes.node_unchecked(node_index) };
                let NodeInner::Task(task_node) = &node.inner else {
                    unreachable!();
                };

                for (id, access) in task_node.accesses.iter() {
                    let access = ResourceAccess {
                        queue_family_index: task_node.queue_family_index,
                        ..*access
                    };

                    let barrier_stage = &mut barrier_stages[id.index() as usize];

                    if *barrier_stage == BarrierStage::Stage0 {
                        if !id.is::<Swapchain>() {
                            if id.is::<Image>() {
                                state.transition_image(id, access);
                            } else if access.access_mask.contains_reads() {
                                state.memory_barrier(id, access);
                            } else {
                                state.execution_barrier(id, access);
                            }
                        }

                        *barrier_stage = BarrierStage::Stage1;
                    }
                }

                let should_submit = if let Some(&next_node_index) = topological_order.get(i + 1) {
                    let next_node = unsafe { self.nodes.node_unchecked(next_node_index) };
                    let NodeInner::Task(next_task_node) = &next_node.inner else {
                        unreachable!()
                    };

                    next_task_node.queue_family_index != task_node.queue_family_index
                } else {
                    true
                };

                if should_submit {
                    break;
                }
            }

            state.flush_initial_barriers();

            // Second per-submission pass: add instructions and barriers for the submission.
            for (i, &node_index) in
                (prev_submission_end..).zip(&topological_order[prev_submission_end..])
            {
                let node = unsafe { self.nodes.node_unchecked(node_index) };
                let NodeInner::Task(task_node) = &node.inner else {
                    unreachable!();
                };

                for &semaphore_index in &node_meta[node_index as usize].wait_semaphores {
                    state.wait_semaphore(semaphore_index);
                }

                for (id, access) in task_node.accesses.iter() {
                    let prev_access = state.prev_accesses[id.index() as usize];
                    let access = ResourceAccess {
                        queue_family_index: task_node.queue_family_index,
                        ..*access
                    };

                    let barrier_stage = &mut barrier_stages[id.index() as usize];

                    if *barrier_stage == BarrierStage::Stage1 {
                        if id.is::<Swapchain>() {
                            state.wait_acquire(unsafe { id.parametrize() }, access);
                        }

                        *barrier_stage = BarrierStage::Stage2;
                    } else if prev_access.queue_family_index != access.queue_family_index {
                        let prev_access = &mut state.prev_accesses[id.index() as usize];
                        prev_access.stage_mask = PipelineStages::empty();
                        prev_access.access_mask = AccessFlags::empty();

                        if id.is_exclusive() {
                            state.acquire_queue_family_ownership(id, access);
                        } else if prev_access.image_layout != access.image_layout {
                            state.transition_image(id, access);
                        } else {
                            state.prev_accesses[id.index() as usize] = access;
                        }
                    } else if prev_access.image_layout != access.image_layout {
                        state.transition_image(id, access);
                    } else if prev_access.access_mask.contains_writes()
                        && access.access_mask.contains_reads()
                    {
                        state.memory_barrier(id, access);
                    } else if access.access_mask.contains_writes() {
                        state.execution_barrier(id, access);
                    } else {
                        // TODO: Could there be use cases for read-after-read execution barriers?
                        let prev_access = &mut state.prev_accesses[id.index() as usize];
                        prev_access.stage_mask |= access.stage_mask;
                        prev_access.access_mask |= access.access_mask;
                    }
                }

                state.execute_task(node_index);

                for (id, _) in task_node.accesses.iter() {
                    if let Some((_, next_access)) = node_meta[node_index as usize]
                        .release_queue_family_ownership
                        .iter()
                        .find(|(x, _)| *x == id)
                    {
                        state.release_queue_family_ownership(id, *next_access);
                    }
                }

                for &semaphore_index in &node_meta[node_index as usize].signal_semaphores {
                    state.signal_semaphore(semaphore_index);
                }

                for (&swapchain_id, _) in last_swapchain_accesses
                    .iter()
                    .filter(|(_, &i)| i == node_index)
                {
                    state.signal_present(swapchain_id);
                }

                let should_submit = if let Some(&next_node_index) = topological_order.get(i + 1) {
                    let next_node = unsafe { self.nodes.node_unchecked(next_node_index) };
                    let NodeInner::Task(next_task_node) = &next_node.inner else {
                        unreachable!()
                    };

                    next_task_node.queue_family_index != task_node.queue_family_index
                } else {
                    true
                };

                if state.should_flush_submit || should_submit {
                    state.flush_submit();
                }

                if should_submit {
                    let queue = queues_by_queue_family_index[task_node.queue_family_index as usize]
                        .unwrap();
                    state.submit(queue);
                    prev_submission_end = i + 1;
                    break;
                }
            }
        }

        if !state
            .pre_present_queue_family_ownership_transfers
            .is_empty()
        {
            for swapchain_id in mem::take(&mut state.pre_present_queue_family_ownership_transfers) {
                state.pre_present_acquire_queue_family_ownership(swapchain_id);
            }

            state.flush_submit();
            state.submit(state.present_queue.unwrap());
        }

        let semaphores = match (0..semaphore_count)
            .map(|_| {
                // SAFETY: The parameters are valid.
                unsafe { Semaphore::new_unchecked(device.clone(), Default::default()) }
                    .map(Arc::new)
            })
            .collect::<Result<_, _>>()
        {
            Ok(semaphores) => semaphores,
            Err(err) => return Err(CompileError::new(self, CompileErrorKind::VulkanError(err))),
        };

        let swapchains = last_swapchain_accesses.iter().map(|(&id, _)| id).collect();

        Ok(ExecutableTaskGraph {
            graph: self,
            flight_id,
            instructions: state.instructions,
            submissions: state.submissions,
            buffer_barriers: state.buffer_barriers,
            image_barriers: state.image_barriers,
            semaphores: RefCell::new(semaphores),
            swapchains,
            present_queue: state.present_queue.cloned(),
            last_accesses: prev_accesses,
        })
    }

    /// Performs [depth-first search] on the equivalent undirected graph to determine if every node
    /// is visited, meaning the undirected graph is [connected]. If it is, then the directed graph
    /// is [weakly connected]. This property is required because otherwise it could happen that we
    /// end up with a submit that is in no way synchronized with the host.
    ///
    /// [depth-first search]: https://en.wikipedia.org/wiki/Depth-first_search
    /// [connected]: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_vertices_and_graphs
    /// [weakly connected]: https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_vertices_and_graphs
    fn is_weakly_connected(&self) -> bool {
        unsafe fn dfs<W: ?Sized>(
            graph: &TaskGraph<W>,
            node_index: NodeIndex,
            visited: &mut [bool],
            visited_count: &mut u32,
        ) {
            let is_visited = &mut visited[node_index as usize];

            if *is_visited {
                return;
            }

            *is_visited = true;
            *visited_count += 1;

            let node = unsafe { graph.nodes.node_unchecked(node_index) };

            for &node_index in node.in_edges.iter().chain(&node.out_edges) {
                unsafe { dfs(graph, node_index, visited, visited_count) };
            }
        }

        let mut visited = vec![false; self.nodes.capacity() as usize];
        let mut visited_count = 0;

        if let Some((id, _)) = self.nodes.nodes().next() {
            unsafe { dfs(self, id.index(), &mut visited, &mut visited_count) };
        }

        visited_count == self.nodes.len()
    }

    /// Performs [topological sort using depth-first search]. Returns a vector of node indices in
    /// topological order.
    ///
    /// [topological sort using depth-first search]: https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    fn topological_sort(&self) -> Result<Vec<NodeIndex>, CompileErrorKind> {
        type NodeState = u8;

        const VISITED_BIT: NodeState = 1 << 0;
        const ON_STACK_BIT: NodeState = 1 << 1;

        unsafe fn dfs<W: ?Sized>(
            graph: &TaskGraph<W>,
            node_index: NodeIndex,
            state: &mut [NodeState],
            output: &mut [NodeIndex],
            mut output_index: u32,
        ) -> Result<u32, CompileErrorKind> {
            let node_state = &mut state[node_index as usize];

            if *node_state == VISITED_BIT {
                return Ok(output_index);
            }

            if *node_state == ON_STACK_BIT {
                return Err(CompileErrorKind::Cycle);
            }

            *node_state = ON_STACK_BIT;

            let node = unsafe { graph.nodes.node_unchecked(node_index) };

            for &node_index in &node.out_edges {
                output_index = unsafe { dfs(graph, node_index, state, output, output_index) }?;
            }

            state[node_index as usize] = VISITED_BIT;
            output[output_index as usize] = node_index;

            Ok(output_index.wrapping_sub(1))
        }

        let mut state = vec![0; self.nodes.capacity() as usize];
        let mut output = vec![0; self.nodes.len() as usize];
        let mut output_index = self.nodes.len().wrapping_sub(1);

        for (id, _) in self.nodes.nodes() {
            output_index = unsafe { dfs(self, id.index(), &mut state, &mut output, output_index) }?;
        }

        debug_assert_eq!(output_index, u32::MAX);

        Ok(output)
    }

    /// Performs [longest path search] to assign the dependency level index to each task node.
    /// Tasks in the same dependency level don't depend on eachother and can therefore be run in
    /// parallel. Returns a vector of dependency levels in topological order indexed by the node's
    /// dependency level index.
    ///
    /// [longest path search]: https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs
    unsafe fn dependency_levels(&mut self, topological_order: &[NodeIndex]) -> Vec<Vec<NodeIndex>> {
        let mut distances = vec![0; self.nodes.capacity() as usize];
        let mut max_level = 0;

        for &node_index in topological_order {
            let node = unsafe { self.nodes.node_unchecked(node_index) };

            for &out_node_index in &node.out_edges {
                let new_distance = distances[node_index as usize] + 1;

                if distances[out_node_index as usize] < new_distance {
                    distances[out_node_index as usize] = new_distance;
                    max_level = cmp::max(max_level, new_distance);
                }
            }
        }

        let mut levels = vec![Vec::new(); max_level as usize + 1];

        for (id, node) in self.nodes.nodes_mut() {
            let NodeInner::Task(task_node) = &mut node.inner else {
                unreachable!();
            };

            let level_index = distances[id.index() as usize];
            levels[level_index as usize].push(id.index());
            task_node.dependency_level_index = level_index;
        }

        levels
    }

    /// Assigns a queue family index to each task node. Returns a vector of the used queue family
    /// indices in topological order.
    unsafe fn queue_family_indices(
        &mut self,
        device: &Device,
        queues: &[&Arc<Queue>],
        topological_order: &[NodeIndex],
    ) -> Result<SmallVec<[u32; 3]>, CompileErrorKind> {
        let queue_family_properties = device.physical_device().queue_family_properties();
        let graphics_queue_family_index = queues
            .iter()
            .find(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .map(|q| q.queue_family_index());
        let compute_queue_family_index = queues
            .iter()
            .filter(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .contains(QueueFlags::COMPUTE)
            })
            .min_by_key(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .count()
            })
            .map(|q| q.queue_family_index());
        let transfer_queue_family_index = queues
            .iter()
            .filter(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .contains(QueueFlags::TRANSFER)
            })
            .min_by_key(|q| {
                queue_family_properties[q.queue_family_index() as usize]
                    .queue_flags
                    .count()
            })
            .map(|q| q.queue_family_index())
            .or(compute_queue_family_index)
            .or(graphics_queue_family_index);

        let mut queue_family_indices = SmallVec::new();

        for &node_index in topological_order {
            let node = unsafe { self.nodes.node_unchecked_mut(node_index) };
            let NodeInner::Task(task_node) = &mut node.inner else {
                unreachable!();
            };

            let queue_family_index = match task_node.queue_family_type() {
                QueueFamilyType::Graphics => graphics_queue_family_index,
                QueueFamilyType::Compute => compute_queue_family_index,
                QueueFamilyType::Transfer => transfer_queue_family_index,
                QueueFamilyType::Specific { index } => queues
                    .iter()
                    .any(|q| q.queue_family_index() == index)
                    .then_some(index),
            }
            .ok_or(CompileErrorKind::InsufficientQueues)?;

            task_node.queue_family_index = queue_family_index;

            if !queue_family_indices.contains(&queue_family_index) {
                queue_family_indices.push(queue_family_index);
            }
        }

        Ok(queue_family_indices)
    }

    /// Does a preliminary pass over all nodes in the graph to collect information needed before
    /// the actual compilation pass. Returns a vector of metadata indexed by the node index, the
    /// current semaphore count, and a map from the swapchain ID to the last node that accessed the
    /// swapchain.
    // TODO: Cull redundant semaphores.
    unsafe fn node_metadata(
        &self,
        topological_order: &[NodeIndex],
        prev_accesses: &mut [ResourceAccess],
        barrier_stages: &mut [BarrierStage],
    ) -> (Vec<NodeMeta>, usize, LinearMap<Id<Swapchain>, NodeIndex, 1>) {
        let mut node_meta = vec![NodeMeta::default(); self.nodes.capacity() as usize];
        let mut prev_node_indices = vec![0; self.resources.capacity() as usize];
        let mut semaphore_count = 0;
        let mut last_swapchain_accesses = LinearMap::new();

        for &node_index in topological_order {
            let node = unsafe { self.nodes.node_unchecked(node_index) };
            let NodeInner::Task(task_node) = &node.inner else {
                unreachable!();
            };

            for &out_node_index in &node.out_edges {
                let out_node = unsafe { self.nodes.node_unchecked(out_node_index) };
                let NodeInner::Task(out_task_node) = &out_node.inner else {
                    unreachable!();
                };

                if task_node.queue_family_index != out_task_node.queue_family_index {
                    let semaphore_index = semaphore_count;
                    node_meta[node_index as usize]
                        .signal_semaphores
                        .push(semaphore_index);
                    node_meta[out_node_index as usize]
                        .wait_semaphores
                        .push(semaphore_index);
                    semaphore_count += 1;
                }
            }

            for (id, access) in task_node.accesses.iter() {
                let prev_access = &mut prev_accesses[id.index() as usize];
                let access = ResourceAccess {
                    queue_family_index: task_node.queue_family_index,
                    ..*access
                };
                let prev_node_index = &mut prev_node_indices[id.index() as usize];

                let barrier_stage = &mut barrier_stages[id.index() as usize];

                if *barrier_stage == BarrierStage::Stage0 {
                    *prev_access = access;
                    *prev_node_index = node_index;
                    *barrier_stage = BarrierStage::Stage1;
                } else {
                    if id.is_exclusive()
                        && prev_access.queue_family_index != access.queue_family_index
                    {
                        node_meta[*prev_node_index as usize]
                            .release_queue_family_ownership
                            .push((id, access));
                    }

                    if prev_access.queue_family_index != access.queue_family_index
                        || prev_access.image_layout != access.image_layout
                        || prev_access.access_mask.contains_writes()
                        || access.access_mask.contains_writes()
                    {
                        *prev_access = access;
                    } else {
                        prev_access.stage_mask |= access.stage_mask;
                        prev_access.access_mask |= access.access_mask;
                    }

                    *prev_node_index = node_index;
                }

                if id.is::<Swapchain>() {
                    *last_swapchain_accesses
                        .get_or_insert(unsafe { id.parametrize() }, node_index) = node_index;
                }
            }
        }

        (node_meta, semaphore_count, last_swapchain_accesses)
    }
}

#[derive(Clone, Default)]
struct NodeMeta {
    wait_semaphores: Vec<SemaphoreIndex>,
    signal_semaphores: Vec<SemaphoreIndex>,
    release_queue_family_ownership: Vec<(Id, ResourceAccess)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BarrierStage {
    Stage0,
    Stage1,
    Stage2,
}

struct CompileState<'a> {
    prev_accesses: &'a mut [ResourceAccess],
    instructions: Vec<Instruction>,
    submissions: Vec<Submission>,
    buffer_barriers: Vec<super::BufferMemoryBarrier>,
    image_barriers: Vec<super::ImageMemoryBarrier>,
    present_queue: Option<&'a Arc<Queue>>,
    initial_buffer_barrier_range: Range<BarrierIndex>,
    initial_image_barrier_range: Range<BarrierIndex>,
    has_flushed_submit: bool,
    should_flush_submit: bool,
    prev_buffer_barrier_index: usize,
    prev_image_barrier_index: usize,
    pre_present_queue_family_ownership_transfers: Vec<Id<Swapchain>>,
}

impl<'a> CompileState<'a> {
    fn new(prev_accesses: &'a mut [ResourceAccess], present_queue: Option<&'a Arc<Queue>>) -> Self {
        CompileState {
            prev_accesses,
            instructions: Vec::new(),
            submissions: Vec::new(),
            buffer_barriers: Vec::new(),
            image_barriers: Vec::new(),
            present_queue,
            initial_buffer_barrier_range: 0..0,
            initial_image_barrier_range: 0..0,
            has_flushed_submit: true,
            should_flush_submit: false,
            prev_buffer_barrier_index: 0,
            prev_image_barrier_index: 0,
            pre_present_queue_family_ownership_transfers: Vec::new(),
        }
    }

    fn release_queue_family_ownership(&mut self, id: Id, access: ResourceAccess) {
        debug_assert!(id.is_exclusive());

        let prev_access = &mut self.prev_accesses[id.index() as usize];
        let mut src = *prev_access;
        let dst = ResourceAccess {
            stage_mask: PipelineStages::empty(),
            access_mask: AccessFlags::empty(),
            ..access
        };

        if !prev_access.access_mask.contains_writes() {
            src.access_mask = AccessFlags::empty();
        }

        debug_assert_ne!(src.queue_family_index, dst.queue_family_index);

        self.memory_barrier_inner(id, src, dst);
    }

    fn acquire_queue_family_ownership(&mut self, id: Id, access: ResourceAccess) {
        debug_assert!(id.is_exclusive());

        let prev_access = &mut self.prev_accesses[id.index() as usize];
        let src = ResourceAccess {
            stage_mask: PipelineStages::empty(),
            access_mask: AccessFlags::empty(),
            ..*prev_access
        };
        let dst = access;

        debug_assert_ne!(src.queue_family_index, dst.queue_family_index);

        *prev_access = access;

        self.memory_barrier_inner(id, src, dst);
    }

    fn transition_image(&mut self, id: Id, access: ResourceAccess) {
        debug_assert_ne!(
            self.prev_accesses[id.index() as usize].image_layout,
            access.image_layout,
        );

        self.memory_barrier(id, access);
    }

    fn memory_barrier(&mut self, id: Id, access: ResourceAccess) {
        let prev_access = &mut self.prev_accesses[id.index() as usize];
        let src = ResourceAccess {
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..*prev_access
        };
        let dst = ResourceAccess {
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..access
        };

        *prev_access = access;

        self.memory_barrier_inner(id, src, dst);
    }

    fn execution_barrier(&mut self, id: Id, access: ResourceAccess) {
        let prev_access = &mut self.prev_accesses[id.index() as usize];
        let src = ResourceAccess {
            access_mask: AccessFlags::empty(),
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..*prev_access
        };
        let dst = ResourceAccess {
            access_mask: AccessFlags::empty(),
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            ..access
        };

        debug_assert_eq!(prev_access.image_layout, access.image_layout);

        *prev_access = access;

        self.memory_barrier_inner(id, src, dst);
    }

    fn memory_barrier_inner(&mut self, id: Id, src: ResourceAccess, dst: ResourceAccess) {
        match id.object_type() {
            ObjectType::Buffer => {
                self.buffer_barriers.push(super::BufferMemoryBarrier {
                    src_stage_mask: src.stage_mask,
                    src_access_mask: src.access_mask,
                    dst_stage_mask: dst.stage_mask,
                    dst_access_mask: dst.access_mask,
                    src_queue_family_index: src.queue_family_index,
                    dst_queue_family_index: dst.queue_family_index,
                    buffer: unsafe { id.parametrize() },
                });
            }
            ObjectType::Image | ObjectType::Swapchain => {
                self.image_barriers.push(super::ImageMemoryBarrier {
                    src_stage_mask: src.stage_mask,
                    src_access_mask: src.access_mask,
                    dst_stage_mask: dst.stage_mask,
                    dst_access_mask: dst.access_mask,
                    old_layout: src.image_layout,
                    new_layout: dst.image_layout,
                    src_queue_family_index: src.queue_family_index,
                    dst_queue_family_index: dst.queue_family_index,
                    image: id,
                });
            }
            _ => unreachable!(),
        }
    }

    fn flush_initial_barriers(&mut self) {
        self.initial_buffer_barrier_range = self.prev_buffer_barrier_index as BarrierIndex
            ..self.buffer_barriers.len() as BarrierIndex;
        self.initial_image_barrier_range = self.prev_image_barrier_index as BarrierIndex
            ..self.image_barriers.len() as BarrierIndex;
        self.prev_buffer_barrier_index = self.buffer_barriers.len();
        self.prev_image_barrier_index = self.image_barriers.len();
    }

    fn wait_acquire(&mut self, swapchain_id: Id<Swapchain>, access: ResourceAccess) {
        if !self.has_flushed_submit {
            self.flush_submit();
        }

        self.image_barriers.push(super::ImageMemoryBarrier {
            src_stage_mask: access.stage_mask,
            src_access_mask: AccessFlags::empty(),
            dst_stage_mask: access.stage_mask,
            dst_access_mask: access.access_mask,
            old_layout: ImageLayout::Undefined,
            new_layout: access.image_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: swapchain_id.erase(),
        });

        self.prev_accesses[swapchain_id.index() as usize] = access;

        self.instructions.push(Instruction::WaitAcquire {
            swapchain_id,
            stage_mask: access.stage_mask,
        });
    }

    fn wait_semaphore(&mut self, semaphore_index: SemaphoreIndex) {
        if !self.has_flushed_submit {
            self.flush_submit();
        }

        self.instructions.push(Instruction::WaitSemaphore {
            semaphore_index,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });
    }

    fn execute_task(&mut self, node_index: NodeIndex) {
        self.flush_barriers();

        self.instructions
            .push(Instruction::ExecuteTask { node_index });

        self.has_flushed_submit = false;
    }

    fn signal_semaphore(&mut self, semaphore_index: SemaphoreIndex) {
        self.instructions.push(Instruction::SignalSemaphore {
            semaphore_index,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });

        self.should_flush_submit = true;
    }

    fn signal_present(&mut self, swapchain_id: Id<Swapchain>) {
        let present_queue = self
            .present_queue
            .as_ref()
            .expect("expected to be given a present queue");

        let prev_access = self.prev_accesses[swapchain_id.index() as usize];

        if prev_access.queue_family_index == present_queue.queue_family_index()
            || !swapchain_id.is_exclusive()
        {
            self.memory_barrier(
                swapchain_id.erase(),
                ResourceAccess {
                    stage_mask: PipelineStages::empty(),
                    access_mask: AccessFlags::empty(),
                    image_layout: ImageLayout::PresentSrc,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            );

            self.instructions.push(Instruction::SignalPresent {
                swapchain_id,
                stage_mask: prev_access.stage_mask,
            });
        } else {
            self.pre_present_release_queue_family_ownership(swapchain_id);
        }

        self.should_flush_submit = true;
    }

    fn pre_present_release_queue_family_ownership(&mut self, swapchain_id: Id<Swapchain>) {
        let prev_access = self.prev_accesses[swapchain_id.index() as usize];

        self.release_queue_family_ownership(
            swapchain_id.erase(),
            ResourceAccess {
                stage_mask: PipelineStages::empty(),
                access_mask: AccessFlags::empty(),
                image_layout: ImageLayout::PresentSrc,
                queue_family_index: self.present_queue.as_ref().unwrap().queue_family_index(),
            },
        );

        self.instructions.push(Instruction::SignalPrePresent {
            swapchain_id,
            stage_mask: prev_access.stage_mask,
        });

        self.pre_present_queue_family_ownership_transfers
            .push(swapchain_id);
    }

    fn pre_present_acquire_queue_family_ownership(&mut self, swapchain_id: Id<Swapchain>) {
        if !self.has_flushed_submit {
            self.flush_submit();
        }

        self.instructions.push(Instruction::WaitPrePresent {
            swapchain_id,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });

        self.acquire_queue_family_ownership(
            swapchain_id.erase(),
            ResourceAccess {
                stage_mask: PipelineStages::empty(),
                access_mask: AccessFlags::empty(),
                image_layout: ImageLayout::PresentSrc,
                queue_family_index: self.present_queue.as_ref().unwrap().queue_family_index(),
            },
        );

        self.instructions.push(Instruction::SignalPresent {
            swapchain_id,
            stage_mask: PipelineStages::ALL_COMMANDS,
        });
    }

    fn flush_barriers(&mut self) {
        if self.prev_buffer_barrier_index != self.buffer_barriers.len()
            || self.prev_image_barrier_index != self.image_barriers.len()
        {
            self.instructions.push(Instruction::PipelineBarrier {
                buffer_barrier_range: self.prev_buffer_barrier_index as BarrierIndex
                    ..self.buffer_barriers.len() as BarrierIndex,
                image_barrier_range: self.prev_image_barrier_index as BarrierIndex
                    ..self.image_barriers.len() as BarrierIndex,
            });
            self.prev_buffer_barrier_index = self.buffer_barriers.len();
            self.prev_image_barrier_index = self.image_barriers.len();
        }
    }

    fn flush_submit(&mut self) {
        self.flush_barriers();
        self.instructions.push(Instruction::FlushSubmit);
        self.has_flushed_submit = true;
        self.should_flush_submit = false;
    }

    fn submit(&mut self, queue: &Arc<Queue>) {
        self.instructions.push(Instruction::Submit);

        let prev_instruction_range_end = self
            .submissions
            .last()
            .map(|s| s.instruction_range.end)
            .unwrap_or(0);
        self.submissions.push(Submission {
            queue: queue.clone(),
            initial_buffer_barrier_range: self.initial_buffer_barrier_range.clone(),
            initial_image_barrier_range: self.initial_image_barrier_range.clone(),
            instruction_range: prev_instruction_range_end..self.instructions.len(),
        });
    }
}

impl<W: ?Sized> ExecutableTaskGraph<W> {
    /// Decompiles the graph back into a modifiable form.
    #[inline]
    pub fn decompile(self) -> TaskGraph<W> {
        self.graph
    }
}

/// Parameters to [compile] a [`TaskGraph`].
///
/// [compile]: TaskGraph::compile
#[derive(Clone, Debug)]
pub struct CompileInfo<'a> {
    /// The queues to work with.
    ///
    /// You must supply at least one queue and all queues must be from unique queue families.
    ///
    /// The default value is empty, which must be overridden.
    pub queues: &'a [&'a Arc<Queue>],

    /// The queue to use for swapchain presentation, if any.
    ///
    /// You must supply this queue if the task graph uses any swapchains. It can be the same queue
    /// as one in the [`queues`] field, or a different one.
    ///
    /// The default value is `None`.
    ///
    /// [`queues`]: Self::queues
    pub present_queue: Option<&'a Arc<Queue>>,

    /// The flight which will be executed.
    ///
    /// The default value is `Id::INVALID`, which must be overridden.
    pub flight_id: Id<Flight>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for CompileInfo<'_> {
    #[inline]
    fn default() -> Self {
        CompileInfo {
            queues: &[],
            present_queue: None,
            flight_id: Id::INVALID,
            _ne: crate::NE,
        }
    }
}

/// Error that can happen when [compiling] a [`TaskGraph`].
///
/// [compiling]: TaskGraph::compile
pub struct CompileError<W: ?Sized> {
    pub graph: TaskGraph<W>,
    pub kind: CompileErrorKind,
}

/// The kind of [`CompileError`] that occurred.
#[derive(Debug)]
pub enum CompileErrorKind {
    Unconnected,
    Cycle,
    InsufficientQueues,
    VulkanError(VulkanError),
}

impl<W: ?Sized> CompileError<W> {
    fn new(graph: TaskGraph<W>, kind: CompileErrorKind) -> Self {
        CompileError { graph, kind }
    }
}

impl<W: ?Sized> fmt::Debug for CompileError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.kind, f)
    }
}

impl<W: ?Sized> fmt::Display for CompileError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            CompileErrorKind::Unconnected => f.write_str("the graph is not weakly connected"),
            CompileErrorKind::Cycle => f.write_str("the graph contains a directed cycle"),
            CompileErrorKind::InsufficientQueues => {
                f.write_str("the given queues are not sufficient for the requirements of a task")
            }
            CompileErrorKind::VulkanError(_) => f.write_str("a runtime error occurred"),
        }
    }
}

impl<W: ?Sized> Error for CompileError<W> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            CompileErrorKind::VulkanError(err) => Some(err),
            _ => None,
        }
    }
}

mod linear_map {
    use smallvec::{Array, SmallVec};

    pub struct LinearMap<K, V, const N: usize>
    where
        [(K, V); N]: Array<Item = (K, V)>,
    {
        inner: SmallVec<[(K, V); N]>,
    }

    impl<K, V, const N: usize> LinearMap<K, V, N>
    where
        [(K, V); N]: Array<Item = (K, V)>,
    {
        #[inline]
        pub fn new() -> Self {
            LinearMap {
                inner: SmallVec::new(),
            }
        }

        #[inline]
        pub fn get_or_insert(&mut self, key: K, value: V) -> &mut V
        where
            K: Eq,
        {
            let index = if let Some(index) = self.inner.iter().position(|(k, _)| k == &key) {
                index
            } else {
                let index = self.inner.len();
                self.inner.push((key, value));

                index
            };

            &mut unsafe { self.inner.get_unchecked_mut(index) }.1
        }

        #[inline]
        pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
            self.inner.iter().map(|(k, v)| (k, v))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        resource::{AccessType, ImageLayoutType},
        tests::test_queues,
    };
    use std::marker::PhantomData;
    use vulkano::{
        buffer::BufferCreateInfo, image::ImageCreateInfo, swapchain::SwapchainCreateInfo,
        sync::Sharing,
    };

    #[test]
    fn unconnected() {
        let (resources, queues) = test_queues!();
        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            ..Default::default()
        };

        {
            // ┌───┐
            // │ A │
            // └───┘
            // ┄┄┄┄┄
            // ┌───┐
            // │ B │
            // └───┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 0);
            graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            graph
                .create_task_node("B", QueueFamilyType::Compute, PhantomData)
                .build();

            assert!(matches!(
                unsafe { graph.compile(&compile_info) },
                Err(CompileError {
                    kind: CompileErrorKind::Unconnected,
                    ..
                }),
            ));
        }

        {
            // ┌───┐
            // │ A ├─┐
            // └───┘ │
            // ┌───┐ │
            // │ B ├┐│
            // └───┘││
            // ┄┄┄┄┄││┄┄┄┄┄┄
            //      ││ ┌───┐
            //      │└►│ C │
            //      │  └───┘
            //      │  ┌───┐
            //      └─►│ D │
            //         └───┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 0);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Compute, PhantomData)
                .build();
            let d = graph
                .create_task_node("D", QueueFamilyType::Compute, PhantomData)
                .build();
            graph.add_edge(a, c).unwrap();
            graph.add_edge(b, d).unwrap();

            assert!(matches!(
                unsafe { graph.compile(&compile_info) },
                Err(CompileError {
                    kind: CompileErrorKind::Unconnected,
                    ..
                }),
            ));
        }

        {
            // ┌───┐  ┌───┐  ┌───┐
            // │ A ├─►│ B ├─►│ C │
            // └───┘  └───┘  └───┘
            // ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
            // ┌───┐  ┌───┐  ┌───┐   ┌───┐
            // │ D ├┬►│ E ├┬►│ F ├──►│   │
            // └───┘│ └───┘│ └───┘┌─►│ G │
            //      │      └──────┘┌►│   │
            //      └──────────────┘ └───┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 0);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
                .build();
            graph.add_edge(a, b).unwrap();
            graph.add_edge(b, c).unwrap();
            let d = graph
                .create_task_node("D", QueueFamilyType::Compute, PhantomData)
                .build();
            let e = graph
                .create_task_node("E", QueueFamilyType::Compute, PhantomData)
                .build();
            let f = graph
                .create_task_node("F", QueueFamilyType::Compute, PhantomData)
                .build();
            let g = graph
                .create_task_node("G", QueueFamilyType::Compute, PhantomData)
                .build();
            graph.add_edge(d, e).unwrap();
            graph.add_edge(d, g).unwrap();
            graph.add_edge(e, f).unwrap();
            graph.add_edge(e, g).unwrap();
            graph.add_edge(f, g).unwrap();

            assert!(matches!(
                unsafe { graph.compile(&compile_info) },
                Err(CompileError {
                    kind: CompileErrorKind::Unconnected,
                    ..
                }),
            ));
        }
    }

    #[test]
    fn cycle() {
        let (resources, queues) = test_queues!();
        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            ..Default::default()
        };

        {
            //   ┌───┐  ┌───┐  ┌───┐
            // ┌►│ A ├─►│ B ├─►│ C ├┐
            // │ └───┘  └───┘  └───┘│
            // └────────────────────┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 0);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
                .build();
            graph.add_edge(a, b).unwrap();
            graph.add_edge(b, c).unwrap();
            graph.add_edge(c, a).unwrap();

            assert!(matches!(
                unsafe { graph.compile(&compile_info) },
                Err(CompileError {
                    kind: CompileErrorKind::Cycle,
                    ..
                }),
            ));
        }

        {
            //   ┌───┐  ┌───┐  ┌───┐
            // ┌►│ A ├┬►│ B ├─►│ C ├┐
            // │ └───┘│ └───┘  └───┘│
            // │┄┄┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄│┄┄┄┄┄┄┄
            // │      │ ┌───┐  ┌───┐│ ┌───┐
            // │      └►│ D ├─►│ E ├┴►│ F ├┐
            // │        └───┘  └───┘  └───┘│
            // └───────────────────────────┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 0);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
                .build();
            let d = graph
                .create_task_node("D", QueueFamilyType::Compute, PhantomData)
                .build();
            let e = graph
                .create_task_node("E", QueueFamilyType::Compute, PhantomData)
                .build();
            let f = graph
                .create_task_node("F", QueueFamilyType::Compute, PhantomData)
                .build();
            graph.add_edge(a, b).unwrap();
            graph.add_edge(a, d).unwrap();
            graph.add_edge(b, c).unwrap();
            graph.add_edge(c, f).unwrap();
            graph.add_edge(d, e).unwrap();
            graph.add_edge(e, f).unwrap();
            graph.add_edge(f, a).unwrap();

            assert!(matches!(
                unsafe { graph.compile(&compile_info) },
                Err(CompileError {
                    kind: CompileErrorKind::Cycle,
                    ..
                }),
            ));
        }

        {
            // ┌─────┐
            // │┌───┐└►┌───┐  ┌───┐
            // ││ A ├┬►│ B ├─►│ C ├┬──────┐
            // │└───┘│ └───┘┌►└───┘│      │
            // │┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄│┄
            // │     │ ┌───┐│ ┌───┐│ ┌───┐│
            // │     └►│ D ├┴►│ E │└►│ F ├│┐
            // │     ┌►└───┘  └───┘  └───┘││
            // │     └────────────────────┘│
            // └───────────────────────────┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 0);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Graphics, PhantomData)
                .build();
            let d = graph
                .create_task_node("D", QueueFamilyType::Compute, PhantomData)
                .build();
            let e = graph
                .create_task_node("E", QueueFamilyType::Compute, PhantomData)
                .build();
            let f = graph
                .create_task_node("F", QueueFamilyType::Compute, PhantomData)
                .build();
            graph.add_edge(a, b).unwrap();
            graph.add_edge(a, d).unwrap();
            graph.add_edge(b, c).unwrap();
            graph.add_edge(c, d).unwrap();
            graph.add_edge(c, f).unwrap();
            graph.add_edge(d, c).unwrap();
            graph.add_edge(d, e).unwrap();
            graph.add_edge(f, b).unwrap();

            assert!(matches!(
                unsafe { graph.compile(&compile_info) },
                Err(CompileError {
                    kind: CompileErrorKind::Cycle,
                    ..
                }),
            ));
        }
    }

    #[test]
    fn initial_pipeline_barrier() {
        let (resources, queues) = test_queues!();
        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            ..Default::default()
        };

        {
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let buffer = graph.add_buffer(&BufferCreateInfo::default());
            let image = graph.add_image(&ImageCreateInfo::default());
            let node = graph
                .create_task_node("", QueueFamilyType::Graphics, PhantomData)
                .buffer_access(buffer, AccessType::VertexShaderUniformRead)
                .image_access(
                    image,
                    AccessType::FragmentShaderSampledRead,
                    ImageLayoutType::Optimal,
                )
                .build();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                InitialPipelineBarrier {
                    buffer_barriers: [
                        {
                            dst_stage_mask: VERTEX_SHADER,
                            dst_access_mask: UNIFORM_READ,
                            buffer: buffer,
                        },
                    ],
                    image_barriers: [
                        {
                            dst_stage_mask: FRAGMENT_SHADER,
                            dst_access_mask: SHADER_SAMPLED_READ,
                            new_layout: ShaderReadOnlyOptimal,
                            image: image,
                        },
                    ],
                },
                ExecuteTask { node: node },
                FlushSubmit,
                Submit,
            );
        }
    }

    #[test]
    fn semaphore() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();
        let has_compute_only_queue = queues.iter().any(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS)
        });

        if !has_compute_only_queue {
            return;
        }

        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            ..Default::default()
        };

        {
            // ┌───┐
            // │ A ├─┐
            // └───┘ │
            // ┌───┐ │
            // │ B ├┐│
            // └───┘││
            // ┄┄┄┄┄││┄┄┄┄┄┄
            //      │└►┌───┐
            //      └─►│ C │
            //         └───┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Compute, PhantomData)
                .build();
            graph.add_edge(a, c).unwrap();
            graph.add_edge(b, c).unwrap();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                ExecuteTask { node: b },
                // TODO: This semaphore is redundant.
                SignalSemaphore {
                    semaphore_index: semaphore1,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                ExecuteTask { node: a },
                SignalSemaphore {
                    semaphore_index: semaphore2,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore1,
                    stage_mask: ALL_COMMANDS,
                },
                WaitSemaphore {
                    semaphore_index: semaphore2,
                    stage_mask: ALL_COMMANDS,
                },
                ExecuteTask { node: c },
                FlushSubmit,
                Submit,
            );
        }

        {
            // ┌───┐
            // │ A ├┐
            // └───┘│
            // ┄┄┄┄┄│┄┄┄┄┄┄
            //      │ ┌───┐
            //      ├►│ B │
            //      │ └───┘
            //      │ ┌───┐
            //      └►│ C │
            //        └───┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Compute, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Compute, PhantomData)
                .build();
            graph.add_edge(a, b).unwrap();
            graph.add_edge(a, c).unwrap();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                ExecuteTask { node: a },
                // TODO: This semaphore is redundant.
                SignalSemaphore {
                    semaphore_index: semaphore1,
                    stage_mask: ALL_COMMANDS,
                },
                SignalSemaphore {
                    semaphore_index: semaphore2,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore2,
                    stage_mask: ALL_COMMANDS,
                },
                ExecuteTask { node: c },
                FlushSubmit,
                WaitSemaphore {
                    semaphore_index: semaphore1,
                    stage_mask: ALL_COMMANDS,
                },
                ExecuteTask { node: b },
                FlushSubmit,
                Submit,
            );
        }

        {
            // ┌───┐                ┌───┐
            // │ A ├───────┬───────►│ E │
            // └───┘       │      ┌►└───┘
            // ┌───┐       │      │
            // │ B ├┐      │      │
            // └───┘│      │      │
            // ┄┄┄┄┄│┄┄┄┄┄┄│┄┄┄┄┄┄│┄┄
            //      │ ┌───┐└►┌───┐│
            //      └►│ C ├─►│ D ├┘
            //        └───┘  └───┘
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let a = graph
                .create_task_node("A", QueueFamilyType::Graphics, PhantomData)
                .build();
            let b = graph
                .create_task_node("B", QueueFamilyType::Graphics, PhantomData)
                .build();
            let c = graph
                .create_task_node("C", QueueFamilyType::Compute, PhantomData)
                .build();
            let d = graph
                .create_task_node("D", QueueFamilyType::Compute, PhantomData)
                .build();
            let e = graph
                .create_task_node("E", QueueFamilyType::Graphics, PhantomData)
                .build();
            graph.add_edge(a, d).unwrap();
            graph.add_edge(a, e).unwrap();
            graph.add_edge(b, c).unwrap();
            graph.add_edge(c, d).unwrap();
            graph.add_edge(d, e).unwrap();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            // TODO: This could be brought down to 3 submissions with task reordering.
            assert_matches_instructions!(
                graph,
                ExecuteTask { node: b },
                SignalSemaphore {
                    semaphore_index: semaphore1,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore1,
                    stage_mask: ALL_COMMANDS,
                },
                ExecuteTask { node: c },
                FlushSubmit,
                Submit,
                ExecuteTask { node: a },
                SignalSemaphore {
                    semaphore_index: semaphore2,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore2,
                    stage_mask: ALL_COMMANDS,
                },
                ExecuteTask { node: d },
                SignalSemaphore {
                    semaphore_index: semaphore3,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore3,
                    stage_mask: ALL_COMMANDS,
                },
                ExecuteTask { node: e },
                FlushSubmit,
                Submit,
            );
        }
    }

    #[test]
    fn queue_family_ownership_transfer() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();
        let has_compute_only_queue = queues.iter().any(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS)
        });

        if !has_compute_only_queue {
            return;
        }

        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            ..Default::default()
        };

        {
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let buffer1 = graph.add_buffer(&BufferCreateInfo::default());
            let buffer2 = graph.add_buffer(&BufferCreateInfo::default());
            let image1 = graph.add_image(&ImageCreateInfo::default());
            let image2 = graph.add_image(&ImageCreateInfo::default());
            let compute_node = graph
                .create_task_node("", QueueFamilyType::Compute, PhantomData)
                .buffer_access(buffer1, AccessType::ComputeShaderStorageWrite)
                .buffer_access(buffer2, AccessType::ComputeShaderStorageRead)
                .image_access(
                    image1,
                    AccessType::ComputeShaderStorageWrite,
                    ImageLayoutType::Optimal,
                )
                .image_access(
                    image2,
                    AccessType::ComputeShaderSampledRead,
                    ImageLayoutType::Optimal,
                )
                .build();
            let graphics_node = graph
                .create_task_node("", QueueFamilyType::Graphics, PhantomData)
                .buffer_access(buffer1, AccessType::IndexRead)
                .buffer_access(buffer2, AccessType::VertexShaderSampledRead)
                .image_access(
                    image1,
                    AccessType::VertexShaderSampledRead,
                    ImageLayoutType::General,
                )
                .image_access(
                    image2,
                    AccessType::FragmentShaderSampledRead,
                    ImageLayoutType::General,
                )
                .build();
            graph.add_edge(compute_node, graphics_node).unwrap();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                ExecuteTask {
                    node: compute_node,
                },
                SignalSemaphore {
                    semaphore_index: semaphore,
                    stage_mask: ALL_COMMANDS,
                },
                PipelineBarrier {
                    buffer_barriers: [
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: SHADER_STORAGE_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            buffer: buffer1,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: ,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            buffer: buffer2,
                        },
                    ],
                    image_barriers: [
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: SHADER_STORAGE_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: General,
                            new_layout: General,
                            image: image1,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: ,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: ShaderReadOnlyOptimal,
                            new_layout: General,
                            image: image2,
                        },
                    ],
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore,
                    stage_mask: ALL_COMMANDS,
                },
                PipelineBarrier {
                    buffer_barriers: [
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: INDEX_INPUT,
                            dst_access_mask: INDEX_READ,
                            buffer: buffer1,
                        },
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: VERTEX_SHADER,
                            dst_access_mask: SHADER_SAMPLED_READ,
                            buffer: buffer2,
                        },
                    ],
                    image_barriers: [
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: VERTEX_SHADER,
                            dst_access_mask: SHADER_SAMPLED_READ,
                            old_layout: General,
                            new_layout: General,
                            image: image1,
                        },
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: FRAGMENT_SHADER,
                            dst_access_mask: SHADER_SAMPLED_READ,
                            old_layout: ShaderReadOnlyOptimal,
                            new_layout: General,
                            image: image2,
                        },
                    ],
                },
                ExecuteTask {
                    node: graphics_node,
                },
                FlushSubmit,
                Submit,
            );
        }

        {
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let sharing = Sharing::Concurrent(
                compile_info
                    .queues
                    .iter()
                    .map(|q| q.queue_family_index())
                    .collect(),
            );
            let buffer1 = graph.add_buffer(&BufferCreateInfo {
                sharing: sharing.clone(),
                ..Default::default()
            });
            let buffer2 = graph.add_buffer(&BufferCreateInfo {
                sharing: sharing.clone(),
                ..Default::default()
            });
            let image1 = graph.add_image(&ImageCreateInfo {
                sharing: sharing.clone(),
                ..Default::default()
            });
            let image2 = graph.add_image(&ImageCreateInfo {
                sharing: sharing.clone(),
                ..Default::default()
            });
            let compute_node = graph
                .create_task_node("", QueueFamilyType::Compute, PhantomData)
                .buffer_access(buffer1, AccessType::ComputeShaderStorageWrite)
                .buffer_access(buffer2, AccessType::ComputeShaderStorageRead)
                .image_access(
                    image1,
                    AccessType::ComputeShaderStorageWrite,
                    ImageLayoutType::Optimal,
                )
                .image_access(
                    image2,
                    AccessType::ComputeShaderSampledRead,
                    ImageLayoutType::Optimal,
                )
                .build();
            let graphics_node = graph
                .create_task_node("", QueueFamilyType::Graphics, PhantomData)
                .buffer_access(buffer1, AccessType::IndexRead)
                .buffer_access(buffer2, AccessType::VertexShaderSampledRead)
                .image_access(
                    image1,
                    AccessType::VertexShaderSampledRead,
                    ImageLayoutType::General,
                )
                .image_access(
                    image2,
                    AccessType::FragmentShaderSampledRead,
                    ImageLayoutType::General,
                )
                .build();
            graph.add_edge(compute_node, graphics_node).unwrap();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                ExecuteTask {
                    node: compute_node,
                },
                SignalSemaphore {
                    semaphore_index: semaphore,
                    stage_mask: ALL_COMMANDS,
                },
                FlushSubmit,
                Submit,
                WaitSemaphore {
                    semaphore_index: semaphore,
                    stage_mask: ALL_COMMANDS,
                },
                PipelineBarrier {
                    buffer_barriers: [],
                    image_barriers: [
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: FRAGMENT_SHADER,
                            dst_access_mask: SHADER_SAMPLED_READ,
                            old_layout: ShaderReadOnlyOptimal,
                            new_layout: General,
                            image: image2,
                        },
                    ],
                },
                ExecuteTask {
                    node: graphics_node,
                },
                FlushSubmit,
                Submit,
            );
        }
    }

    #[test]
    fn swapchain() {
        let (resources, queues) = test_queues!();

        let queue_family_properties = resources
            .device()
            .physical_device()
            .queue_family_properties();

        let present_queue = queues.iter().find(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::GRAPHICS)
        });
        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            present_queue: Some(present_queue.unwrap()),
            ..Default::default()
        };

        {
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let swapchain1 = graph.add_swapchain(&SwapchainCreateInfo::default());
            let swapchain2 = graph.add_swapchain(&SwapchainCreateInfo::default());
            let node = graph
                .create_task_node("", QueueFamilyType::Graphics, PhantomData)
                .image_access(
                    swapchain1.current_image_id(),
                    AccessType::ColorAttachmentWrite,
                    ImageLayoutType::Optimal,
                )
                .image_access(
                    swapchain2.current_image_id(),
                    AccessType::ComputeShaderStorageWrite,
                    ImageLayoutType::Optimal,
                )
                .build();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                WaitAcquire {
                    swapchain_id: swapchain1,
                    stage_mask: COLOR_ATTACHMENT_OUTPUT,
                },
                WaitAcquire {
                    swapchain_id: swapchain2,
                    stage_mask: COMPUTE_SHADER,
                },
                PipelineBarrier {
                    buffer_barriers: [],
                    image_barriers: [
                        {
                            src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            src_access_mask: ,
                            dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            dst_access_mask: COLOR_ATTACHMENT_WRITE,
                            old_layout: Undefined,
                            new_layout: ColorAttachmentOptimal,
                            image: swapchain1,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: ,
                            dst_stage_mask: COMPUTE_SHADER,
                            dst_access_mask: SHADER_STORAGE_WRITE,
                            old_layout: Undefined,
                            new_layout: General,
                            image: swapchain2,
                        },
                    ],
                },
                ExecuteTask { node: node },
                SignalPresent {
                    swapchain_id: swapchain1,
                    stage_mask: COLOR_ATTACHMENT_OUTPUT,
                },
                SignalPresent {
                    swapchain_id: swapchain2,
                    stage_mask: COMPUTE_SHADER,
                },
                PipelineBarrier {
                    buffer_barriers: [],
                    image_barriers: [
                        {
                            src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            src_access_mask: COLOR_ATTACHMENT_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: ColorAttachmentOptimal,
                            new_layout: PresentSrc,
                            image: swapchain1,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: SHADER_STORAGE_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: General,
                            new_layout: PresentSrc,
                            image: swapchain2,
                        },
                    ],
                },
                FlushSubmit,
                Submit,
            );
        }

        let present_queue = queues.iter().find(|q| {
            let queue_flags = queue_family_properties[q.queue_family_index() as usize].queue_flags;

            queue_flags.contains(QueueFlags::COMPUTE) && !queue_flags.contains(QueueFlags::GRAPHICS)
        });

        if !present_queue.is_some() {
            return;
        }

        let compile_info = CompileInfo {
            queues: &queues.iter().collect::<Vec<_>>(),
            present_queue: Some(present_queue.unwrap()),
            ..Default::default()
        };

        {
            let mut graph = TaskGraph::<()>::new(&resources, 10, 10);
            let concurrent_sharing = Sharing::Concurrent(
                compile_info
                    .queues
                    .iter()
                    .map(|q| q.queue_family_index())
                    .collect(),
            );
            let swapchain1 = graph.add_swapchain(&SwapchainCreateInfo::default());
            let swapchain2 = graph.add_swapchain(&SwapchainCreateInfo {
                image_sharing: concurrent_sharing.clone(),
                ..Default::default()
            });
            let swapchain3 = graph.add_swapchain(&SwapchainCreateInfo::default());
            let swapchain4 = graph.add_swapchain(&SwapchainCreateInfo {
                image_sharing: concurrent_sharing.clone(),
                ..Default::default()
            });
            let node = graph
                .create_task_node("", QueueFamilyType::Graphics, PhantomData)
                .image_access(
                    swapchain1.current_image_id(),
                    AccessType::ColorAttachmentWrite,
                    ImageLayoutType::Optimal,
                )
                .image_access(
                    swapchain2.current_image_id(),
                    AccessType::ColorAttachmentWrite,
                    ImageLayoutType::Optimal,
                )
                .image_access(
                    swapchain3.current_image_id(),
                    AccessType::ComputeShaderStorageWrite,
                    ImageLayoutType::Optimal,
                )
                .image_access(
                    swapchain4.current_image_id(),
                    AccessType::ComputeShaderStorageWrite,
                    ImageLayoutType::Optimal,
                )
                .build();

            let graph = unsafe { graph.compile(&compile_info) }.unwrap();

            assert_matches_instructions!(
                graph,
                WaitAcquire {
                    swapchain_id: swapchain1,
                    stage_mask: COLOR_ATTACHMENT_OUTPUT,
                },
                WaitAcquire {
                    swapchain_id: swapchain2,
                    stage_mask: COLOR_ATTACHMENT_OUTPUT,
                },
                WaitAcquire {
                    swapchain_id: swapchain3,
                    stage_mask: COMPUTE_SHADER,
                },
                WaitAcquire {
                    swapchain_id: swapchain4,
                    stage_mask: COMPUTE_SHADER,
                },
                PipelineBarrier {
                    buffer_barriers: [],
                    image_barriers: [
                        {
                            src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            src_access_mask: ,
                            dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            dst_access_mask: COLOR_ATTACHMENT_WRITE,
                            old_layout: Undefined,
                            new_layout: ColorAttachmentOptimal,
                            image: swapchain1,
                        },
                        {
                            src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            src_access_mask: ,
                            dst_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            dst_access_mask: COLOR_ATTACHMENT_WRITE,
                            old_layout: Undefined,
                            new_layout: ColorAttachmentOptimal,
                            image: swapchain2,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: ,
                            dst_stage_mask: COMPUTE_SHADER,
                            dst_access_mask: SHADER_STORAGE_WRITE,
                            old_layout: Undefined,
                            new_layout: General,
                            image: swapchain3,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: ,
                            dst_stage_mask: COMPUTE_SHADER,
                            dst_access_mask: SHADER_STORAGE_WRITE,
                            old_layout: Undefined,
                            new_layout: General,
                            image: swapchain4,
                        },
                    ],
                },
                ExecuteTask { node: node },
                SignalPrePresent {
                    swapchain_id: swapchain1,
                    stage_mask: COLOR_ATTACHMENT_OUTPUT,
                },
                SignalPresent {
                    swapchain_id: swapchain2,
                    stage_mask: COLOR_ATTACHMENT_OUTPUT,
                },
                SignalPrePresent {
                    swapchain_id: swapchain3,
                    stage_mask: COMPUTE_SHADER,
                },
                SignalPresent {
                    swapchain_id: swapchain4,
                    stage_mask: COMPUTE_SHADER,
                },
                PipelineBarrier {
                    buffer_barriers: [],
                    image_barriers: [
                        {
                            src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            src_access_mask: COLOR_ATTACHMENT_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: ColorAttachmentOptimal,
                            new_layout: PresentSrc,
                            image: swapchain1,
                        },
                        {
                            src_stage_mask: COLOR_ATTACHMENT_OUTPUT,
                            src_access_mask: COLOR_ATTACHMENT_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: ColorAttachmentOptimal,
                            new_layout: PresentSrc,
                            image: swapchain2,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: SHADER_STORAGE_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: General,
                            new_layout: PresentSrc,
                            image: swapchain3,
                        },
                        {
                            src_stage_mask: COMPUTE_SHADER,
                            src_access_mask: SHADER_STORAGE_WRITE,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: General,
                            new_layout: PresentSrc,
                            image: swapchain4,
                        },
                    ],
                },
                FlushSubmit,
                Submit,
                WaitPrePresent {
                    swapchain_id: swapchain1,
                    stage_mask: ALL_COMMANDS,
                },
                SignalPresent {
                    swapchain_id: swapchain1,
                    stage_mask: ALL_COMMANDS,
                },
                WaitPrePresent {
                    swapchain_id: swapchain3,
                    stage_mask: ALL_COMMANDS,
                },
                SignalPresent {
                    swapchain_id: swapchain3,
                    stage_mask: ALL_COMMANDS,
                },
                PipelineBarrier {
                    buffer_barriers: [],
                    image_barriers: [
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: ColorAttachmentOptimal,
                            new_layout: PresentSrc,
                            image: swapchain1,
                        },
                        {
                            src_stage_mask: ,
                            src_access_mask: ,
                            dst_stage_mask: ,
                            dst_access_mask: ,
                            old_layout: General,
                            new_layout: PresentSrc,
                            image: swapchain3,
                        },
                    ],
                },
                FlushSubmit,
                Submit,
            );
        }
    }

    struct MatchingState {
        submission_index: usize,
        instruction_index: usize,
        semaphores: ahash::HashMap<&'static str, SemaphoreIndex>,
    }

    macro_rules! assert_matches_instructions {
        (
            $graph:ident,
            $($arg:tt)+
        ) => {
            let mut state = MatchingState {
                submission_index: 0,
                instruction_index: 0,
                semaphores: Default::default(),
            };
            assert_matches_instructions!(@ $graph, state, $($arg)+);
        };
        (
            @
            $graph:ident,
            $state:ident,
            InitialPipelineBarrier {
                buffer_barriers: [
                    $({
                        dst_stage_mask: $($buffer_dst_stage:ident)|*,
                        dst_access_mask: $($buffer_dst_access:ident)|*,
                        buffer: $buffer:ident,
                    },)*
                ],
                image_barriers: [
                    $({
                        dst_stage_mask: $($image_dst_stage:ident)|*,
                        dst_access_mask: $($image_dst_access:ident)|*,
                        new_layout: $image_new_layout:ident,
                        image: $image:ident,
                    },)*
                ],
            },
            $($arg:tt)*
        ) => {
            let submission = &$graph.submissions[$state.submission_index];
            let buffer_barrier_range = &submission.initial_buffer_barrier_range;
            let image_barrier_range = &submission.initial_image_barrier_range;

            let buffer_barrier_range =
                buffer_barrier_range.start as usize..buffer_barrier_range.end as usize;
            let buffer_barriers = &$graph.buffer_barriers[buffer_barrier_range];
            #[allow(unused_mut)]
            let mut buffer_barrier_count = 0;
            $(
                let barrier = buffer_barriers
                    .iter()
                    .find(|barrier| barrier.buffer == $buffer)
                    .unwrap();
                assert_eq!(barrier.src_stage_mask, PipelineStages::empty());
                assert_eq!(barrier.src_access_mask, AccessFlags::empty());
                assert_eq!(
                    barrier.dst_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$buffer_dst_stage)*,
                );
                assert_eq!(
                    barrier.dst_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$buffer_dst_access)*,
                );
                buffer_barrier_count += 1;
            )*
            assert_eq!(buffer_barriers.len(), buffer_barrier_count);

            let image_barrier_range =
                image_barrier_range.start as usize..image_barrier_range.end as usize;
            let image_barriers = &$graph.image_barriers[image_barrier_range];
            #[allow(unused_mut)]
            let mut image_barrier_count = 0;
            $(
                let barrier = image_barriers
                    .iter()
                    .find(|barrier| barrier.image == $image.erase())
                    .unwrap();
                assert_eq!(barrier.src_stage_mask, PipelineStages::empty());
                assert_eq!(barrier.src_access_mask, AccessFlags::empty());
                assert_eq!(
                    barrier.dst_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$image_dst_stage)*,
                );
                assert_eq!(
                    barrier.dst_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$image_dst_access)*,
                );
                assert_eq!(barrier.old_layout, ImageLayout::Undefined);
                assert_eq!(barrier.new_layout, ImageLayout::$image_new_layout);
                image_barrier_count += 1;
            )*
            assert_eq!(image_barriers.len(), image_barrier_count);

            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            WaitAcquire {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            WaitSemaphore {
                semaphore_index: $semaphore_index:ident,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::WaitSemaphore {
                    stage_mask,
                    ..
                } if stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            let Instruction::WaitSemaphore { semaphore_index, .. } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };

            assert_eq!(
                semaphore_index,
                $state.semaphores.get(stringify!($semaphore_index)).unwrap(),
            );

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            ExecuteTask {
                node: $node:expr $(,)?
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::ExecuteTask { node_index } if node_index == $node.index(),
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            PipelineBarrier {
                buffer_barriers: [
                    $({
                        src_stage_mask: $($buffer_src_stage:ident)|*,
                        src_access_mask: $($buffer_src_access:ident)|*,
                        dst_stage_mask: $($buffer_dst_stage:ident)|*,
                        dst_access_mask: $($buffer_dst_access:ident)|*,
                        buffer: $buffer:ident,
                    },)*
                ],
                image_barriers: [
                    $({
                        src_stage_mask: $($image_src_stage:ident)|*,
                        src_access_mask: $($image_src_access:ident)|*,
                        dst_stage_mask: $($image_dst_stage:ident)|*,
                        dst_access_mask: $($image_dst_access:ident)|*,
                        old_layout: $image_old_layout:ident,
                        new_layout: $image_new_layout:ident,
                        image: $image:ident,
                    },)*
                ],
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::PipelineBarrier { .. },
            ));
            let Instruction::PipelineBarrier { buffer_barrier_range, image_barrier_range } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };

            let buffer_barrier_range =
                buffer_barrier_range.start as usize..buffer_barrier_range.end as usize;
            let buffer_barriers = &$graph.buffer_barriers[buffer_barrier_range];
            #[allow(unused_mut)]
            let mut buffer_barrier_count = 0;
            $(
                let barrier = buffer_barriers
                    .iter()
                    .find(|barrier| barrier.buffer == $buffer)
                    .unwrap();
                assert_eq!(
                    barrier.src_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$buffer_src_stage)*,
                );
                assert_eq!(
                    barrier.src_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$buffer_src_access)*,
                );
                assert_eq!(
                    barrier.dst_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$buffer_dst_stage)*,
                );
                assert_eq!(
                    barrier.dst_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$buffer_dst_access)*,
                );
                buffer_barrier_count += 1;
            )*
            assert_eq!(buffer_barriers.len(), buffer_barrier_count);

            let image_barrier_range =
                image_barrier_range.start as usize..image_barrier_range.end as usize;
            let image_barriers = &$graph.image_barriers[image_barrier_range];
            #[allow(unused_mut)]
            let mut image_barrier_count = 0;
            $(
                let barrier = image_barriers
                    .iter()
                    .find(|barrier| barrier.image == $image.erase())
                    .unwrap();
                assert_eq!(
                    barrier.src_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$image_src_stage)*,
                );
                assert_eq!(
                    barrier.src_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$image_src_access)*,
                );
                assert_eq!(
                    barrier.dst_stage_mask,
                    PipelineStages::empty() $(| PipelineStages::$image_dst_stage)*,
                );
                assert_eq!(
                    barrier.dst_access_mask,
                    AccessFlags::empty() $(| AccessFlags::$image_dst_access)*,
                );
                assert_eq!(barrier.old_layout, ImageLayout::$image_old_layout);
                assert_eq!(barrier.new_layout, ImageLayout::$image_new_layout);
                image_barrier_count += 1;
            )*
            assert_eq!(image_barriers.len(), image_barrier_count);

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            SignalSemaphore {
                semaphore_index: $semaphore_index:ident,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::SignalSemaphore {
                    stage_mask,
                    ..
                } if stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            let Instruction::SignalSemaphore { semaphore_index, .. } =
                &$graph.instructions[$state.instruction_index]
            else {
                unreachable!();
            };

            assert!($state.semaphores.get(&stringify!($semaphore_index)).is_none());
            $state.semaphores.insert(stringify!($semaphore_index), *semaphore_index);

            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            SignalPrePresent {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            WaitPrePresent {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            SignalPresent {
                swapchain_id: $swapchain_id:expr,
                stage_mask: $($stage:ident)|*,
            },
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask,
                } if swapchain_id == $swapchain_id
                    && stage_mask == PipelineStages::empty() $(| PipelineStages::$stage)*,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            FlushSubmit,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::FlushSubmit,
            ));
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
            Submit,
            $($arg:tt)*
        ) => {
            assert!(matches!(
                $graph.instructions[$state.instruction_index],
                Instruction::Submit,
            ));
            $state.submission_index += 1;
            $state.instruction_index += 1;
            assert_matches_instructions!(@ $graph, $state, $($arg)*);
        };
        (
            @
            $graph:ident,
            $state:ident,
        ) => {
            assert_eq!($graph.submissions.len(), $state.submission_index);
            assert_eq!($graph.instructions.len(), $state.instruction_index);
        };
    }
    use assert_matches_instructions;
}
