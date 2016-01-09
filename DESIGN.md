# Design document

The Vulkan specs are not public, but [the Mantle specs are](http://www.amd.com/Documents/Mantle-Programming-Guide-and-API-Reference.pdf). Many design decisions can be made already based on what Mantle shows us.

## Objectives of this library

From highest to lowest priority:

 - Be memory safe.
 - Don't hide what it's doing.
 - Be convenient to use.

## Extensions

Mantle has an extensions system for things such as DMA, occlusion query or platform-specific bindings.
It is unclear how Vulkan handles this.

Mantle's bindings to Windows is pretty simple: you need to create a special kind of resource that is similar to images, and this special kind of resource can be drawn on a `HWND`. The image isn't tied to the `HWND` ; you need tell which `HWND` to use each time you swap buffers. Deciding how to bind this library to platform-specific interfaces requires a broader picture.

## Thread safety

Some Mantle functions are marked as not being thread-safe. However we most likely want to store them in `Arc`s.

If we used `&mut` for non-thread-safe operations, we would need to wrap the entire object in a `Mutex`. Instead the best solution is to use an internal mutex that is used only for thread-unsafe operations:

```rust
pub struct Device {
    mutex: Mutex<()>,
    ...
}

impl Device {
    pub fn thread_safe_operation(&self) {
        // mutex not used
        vkDoSomething();
    }

    pub fn not_thread_safe_operation(&self) {
        let _lock = self.mutex.lock();
        vkDoSomethingElse();
    }
}
```

## Shaders introspection

When you use a descriptor set so that it's used by a shader, it is important for safety that you make sure that it matches what the shader expects.

However the Vulkan API will likely not provide any way to introspect a shader. Therefore a small SPIR-V analyzer will need to be included in the library that parses the input data.

This situation could be improved after Rust plugins are made stable, so that the analysis is done at compile-time.

## Resources borrowing management

*In this sections, "resources" describes everything that is used by a command buffer: buffers, images, pipelines, dynamic state, etc.*

The rule is that objects must not be written by the CPU nor destroyed while they are still used by the GPU. The Vulkan library doesn't automatically handle this for you, but this library should enforce it.

### Ownership by command buffers

Command buffers need to somehow make sure that the resources they use are still valid as long as they exist. This could be handled with regular Rust lifetimes, but since Rust doesn't allow structs to borrow themselves my personal opinion is that a `CommandBuffer<'a>` struct would be too cumbersome. You would end up with something like this:

```rust
pub struct GameResources {
    ...
}

pub struct GameCommandBuffers<'a> {
    command_buffer1: vulkano::CommandBuffer<'a>,
    ...
}

fn main() {
    let resources = GameResources::new();
    let command_buffers = GameCommandBuffers::new(&resources);
}
```

In my opinion everything should simply use `Arc`s.

### When to destroy?

Resources should not be destroyed as long as they are still in use by the GPU.

The trick is that the only way that a resource could be in use by the GPU is through a command buffer. Command buffers too must be not destroyed as long as they are still in use. Command buffers already ensure that the resources they use stay alive, therefore the only tricky part is to know when to destroy command buffers.

If all command buffers that use a specific resource are destroyed, then we are sure that this resource is not in use by the GPU and we can delete it. The destructors of buffers, textures, etc. are trivial.

When it comes to command buffers, in my opinion it should look like this:

```rust
pub fn submit_command_buffer(cmd: Arc<CommandsBuffer>) -> FenceGuard {
    ...
}
```

The `FenceGuard` holds the `Arc` and ensures that the command buffer is alive. Destroying a `FenceGuard` blocks until the Vulkan fence is fulfilled.

*Leak-safety: `CommandsBuffer` should contain a flag indicating whether or not it is currently locked. Destroying the `FenceGuard` clears the flag. Destroying a locked command buffer panicks. This avoids problems if the user `mem::forget`s the guard.*.

### Mutability

*Only relevant for buffers and images.*

Since both a command buffer and the user have access to a borrow of the same resource, we have to ensure that a resource can't be read and written simultaneously (similarly to any other CPU object).

This is going to be done with a wrapper object, just like `RefCell` or `Mutex`.

Example:

```rust
pub struct Buffer {
    ...
}

impl Buffer {
    // always available
    pub fn modify(&mut self, ...) { ... }
}

/// Wraps around a resource that can be accessed either by the CPU or by the GPU.
pub struct GpuAccess<T> {
    // the whole synchronization scheme here needs some thought
    content: Mutex<T>,   // need to synchronize CPU access as well
    in_use: Mutex<Option<Fence>>,
}

impl GpuAccess<T> {
    // waits for the fence if necessary
    pub fn lock(&self) -> &mut T { ... }
    pub fn try_lock(&self) -> Result<&mut T, ...> { ... }
}

unsafe impl<T> SomeTrait<T> for GpuAccess<T> {
    unsafe fn force_access(&self) -> &mut T { ... }
    fn read_only_lock_until(&self, fence: Fence) { ... }
    fn exlusive_lock_until(&self, fence: Fence) { ... }
}
```

The command buffer writes the fence in the `GpuAccess` through a trait.

There could be multiple wrappers. One for single-threaded CPU accesses only. One that does read-write locks instead of exclusive locks. And so on.

### Fence or no fence

If you submit several command buffers in a row to the same queue, then you only need to create a fence at the last submission.

Creating a fence every time could become a big overhead. However you usually know too late whether you needed a fence or not at the previous submission. We can't predict the future, therefore the choice should be given to the user whether to create a fence or not.

Consequently, submitting a command buffer would look like this:

```rust
pub fn submit_command_buffer_fence(cmd: Arc<CommandsBuffer>) -> FenceGuard {
    ...
}

pub fn submit_command_buffer_nofence(cmd: Arc<CommandsBuffer>) -> NoFence {
    ...
}
```

`NoFence` would be a linear type. Rust doesn't have linear types, but this can be emulated by panicking in the destructor.

The only way to use a `NoFence` is to consume it by submitting another command buffer immediately after:

```rust
impl NoFence {
    pub fn submit_after_fence(self, cmd: Arc<CommandsBuffer>) -> FenceGuard { ... }
    pub fn submit_after_nofence(self, cmd: Arc<CommandsBuffer>) -> NoFence { ... }
}
```

The `GpuAccess` trait also needs new method to lock a resource until further notice to adjust for `NoFence`.

If possible this could be made more convenient by merging the two methods and returning a trait implementation instead:

```rust
pub fn submit_command_buffer<F>(cmd: Arc<CommandsBuffer>) -> F where F: GpuBlock { ... }

impl NoFence {
    pub fn submit_after<F>(self, cmd: Arc<CommandsBuffer>) -> F where F: GpuBlock { ... }
}
```

### Multiple queues

There is also the problem of multiple GPU queues.

The `GpuAccess` structs and similar should hold the ID of the queue that uses the resource, and the trait that gives access to its content should account for that.

It is unclear how Vulkan will handle that. Mantle uses a semaphore mechanism different from the fence mechanism, in which case this would add a `SemaphoreGuard` in addition to `FenceGuard` and `NoFence`. But it is likely that this changed in Vulkan.

## Resources state management

With Mantle, all buffers and textures are in a certain state. This is likely going to be the same in Vulkan.
To do certain operations, a resource must be in a given state. For example copying data in RAM to a buffer requires the buffer to be in the `GR_MEMORY_STATE_DATA_TRANSFER` state.

Changing the state of a resource requires using a command in a command buffer, and requires to know what the state of the resource at the time of the execution is. For example to read the same buffer from a shader, you must switch it from the `GR_MEMORY_STATE_DATA_TRANSFER` state to the `GR_MEMORY_STATE_GRAPHICS_SHADER_READ_WRITE` state. This is done by explicitely stating that the buffer currently is in the `GR_MEMORY_STATE_DATA_TRANSFER` state.

The problem is that we need to know what the state of a resource is at the time *when the command buffer is used*, and not at the time when the command buffer is created. Let's say that we build the command buffers A and B. A leaves a given resource in a certain state. B needs to know in which state that resource was left in. Requiring the user to state in which order the command buffers will be executed would be a burden.

Instead to solve this problem, resources will have a "default state".

 - At initialization, a resource is switched to this default state.
 - Command buffers are free to switch the state of a resource, but must restore the resource to its default state at the end.

## Pipelines and dynamic state

Before being able to draw, you have to create a "pipeline object" that holds all the state of the pipeline (list of shaders, blend function, etc.). It has been hinted that some parts are dynamic and some parts static, although it's a big unclear.

There are five type of objects in Mantle that are "dynamic states": the rasterizer, the viewport/scissor, the blender, the depth-stencil, and the multisampling states. One instance of each of these object types must be binded by command buffers before drawing.

## Descriptor sets

Descriptor sets are the Vulkan equivalent of OpenGL uniform buffers and texture binding points.

This poses a big challenge because the user manually chooses the layout in their shader, and manually chooses the bind points and the layout of their buffers, and then the two have to match. In glium for example, this is checked at runtime. But ideally it should be checked at compile-time.

What is sure is that this library will provide an unsafe API that maps exactly to the Vulkan API. The user has to manually make sure that the layouts do match, hence the unsafety.
The library may also provide a safe API thanks to plugins, although it's still very blurry.

## Context loss

Context losses will probably be similar to OpenGL, where commands can trigger errors. When a context loss occurs, the user has to destroy the context and reupload everything.

The design of glium is the following: only functions and methods that return actual data to the user can return a "context lost" error through a Result (for example reading from a buffer). Submitting a draw command, copying data between buffers, uploading data, etc. never trigger this error.

It is unclear whether context losses should result in panics or in regular errors. If more commands than in OpenGL can fail because of a context loss (for example mapping a buffer), then it may be too annoying to handle with regular results compared to OpenGL.

## Examples of what it should look like

These examples are likely to change, but they should give a good overview:

### Uploading a buffer

Uploading to CPU-accessible memory, ie. slow write:

```rust
#[derive(Copy, Clone, BufferLayout)]      // `#[derive_BufferLayout]` is not possible until plugins are stable though
struct Data {
    position: (f32, f32, f32),
    tex_coords: (f32, f32),
}

let data: &[Data] = &[Data { ... }, Data { ... }, ...];
let mut buffer = AccessBuffer::cpu_accessible_with_data(&device, &data);

// let's modify a value of the sake of the example
{
    let mut mapping = buffer.write();       // same API as RwLock
    mapping[1].position.0 = 1.0;
}
```

Using the DMA to copy:

```rust
let local_buffer = RwGpuAccess::new(AccessBuffer::pinned_memory_with_data(&device, &data));
let remote_buffer = RwGpuAccess::new(Buffer::empty(&device, std::mem::size_of_val(&data)));

let dma_queue = device.dma_queues().next().unwrap();
let command_buffer = CommandBufferBuilder::once(&device)            // `once` optimizes for buffers that trigger once
                        .copy_buffer(&local_buffer, &remote_buffer)
                        .build();
let fence = dma_queue.submit(&command_buffer);

assert!(local_buffer.try_write().is_none());        // buffer is in use, we can't access it because of safety
assert!(local_buffer.try_read().is_some());         // however we can read its content

// dropping the fence blocks until the fence is fulfilled
drop(fence);
```

### Hello triangle

**To do**
