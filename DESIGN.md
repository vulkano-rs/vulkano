This document contains the global design decisions made by the vulkano library. It can also be a
good start if you want to contribute to some internal parts of vulkano and don't know how it works.

This document assumes that you're already familiar with Vulkan and does not introduce the various
concepts. However it can still be a good read if you are not so familiar.

If you notice any mistake, feel free to open a PR. If you want to suggest something, feel free to
open a PR as well.

# The three kinds of objects

Vulkano provides wrappers around all objects of the Vulkan API. However these objects are split in
three categories, depending on their access pattern:

- Objects that are not created often and in very small numbers.
- Objects that are typically created at initialization and which are often accessed without mutation
  by performance-critical code.
- Objects that are created, destroyed or modified during performance-critical code, and that
  usually require a synchronization strategy to avoid race conditions.

The first category are objects that are not created often and created in very small numbers:
Instances, Devices, Surfaces, Swapchains. In a typical application each of these objects is only
created once and destroyed when the application exits. Vulkano's API provides a struct that
corresponds to each of these objects, and this struct is typically wrapped in an `Arc`.
Their `new` method in fact returns an `Arc<T>` instead of just a `T` in order to encourage users to
use `Arc`s. You use these objects by cloning them around like you would use objects in a
garbage-collected language such as Java.

The second category are objects like the GraphicsPipeline, ComputePipeline, PipelineLayout,
RenderPass and Framebuffer. They are usually created at initialization and don't perform any
operations themselves, but they describe to the Vulkan implementation operations that we are going
to perform and are thus frequently accessed in order to determine whether the operation that the
vulkano user requested is compliant to what was described. Just like the first category, each of
these objects has a struct that corresponds to them, but in order to make these checks as fast as
possible these structs have a template parameter that describes in a strongly-typed fashion the
operation on the CPU side. This makes it possible to move many checks to compile-time instead of
runtime. More information in another section of this document.

The third category are objects like CommandBuffers, CommandPools, DescriptorSets, DescriptorPools,
Buffers, Images, and memory pools (although not technically a Vulkan object). The way they are
implemented has a huge impact on the performance of the application. Contrary to the first two
categories, each of these objects is represented in vulkano by an unsafe trait (and not by a
struct) that can be freely implemented by the user if they wish. Vulkano provides unsafe structs
such as `UnsafeBuffer`, `UnsafeImage`, etc. which have zero overhead and do not perform any safety
checks, and are the tools used by the safe implementations of the traits. Vulkano also provides
some safe implementations for convenience such as `CpuAccessibleBuffer` or `AttachmentImage`.

# Runtime vs compile-time checks

The second category of objects described above are objects that describe to the Vulkan
implementation an operation that we are going to perform later. For example a `ComputePipeline`
object describes to the Vulkan implementation a compute operation and contains the shader's code
and the list of resources that we are going to bind and that are going to be accessed by the shader.

Since vulkano is a safe library, it needs to check whether the operation the user requests (eg.
executing a compute operation) matches the corresponding `ComputePipeline` (for example, check
that the list of resources passed by the user matches what the compute pipeline expects).
These checks can be expensive. For example when it comes to buffers, vulkano needs to check whether
the layout of the buffers passed by the user is the same as what is expected, by looping through all
the members and following several indirections. If you multiply this by several dozens or hundreds
of operations, it can become very expensive.

In order to reduce the stress caused by these checks, structs such as `ComputePipeline` have a
template parameter which describes the operation. Whenever vulkano performs a check, it queries
the templated object through a trait, and each safety check has its own trait. This means
that we can build strongly-typed objects at compile-time that describe a very precise operation and
whose method implementations are trivial. For example, we can create a `MyComputeOpDesc` type which
implements the `ResourcesListMatch<MyResourcesList>` trait (which was made up for the sake of the
example), and the user will only be able to pass a `MyResourcesList` object for the list of
resources. This moves the check to compile-time and totally eliminates any runtime check. The
compute pipeline is then expressed as `ComputePipeline<MyComputeOpDesc>`.

However this design has a drawback, which is that is can be difficult to explicitly express such a
type. A compute pipeline in the example above could be expressed as
`ComputePipeline<MyComputeOpDesc>`, but in practice these types (like `MyComputeOpDesc`) would be
built by builders and can become extremely long and annoying to put in a struct (just like for
example the type of `(10..).filter(|n| n*2).skip(3).take(5)` can be very long and annoying to put
in a struct). This is especially problematic as it concerns objects that are usually created at
initialization and stay alive for a long time, in other words the kind of objects that you would
put in a struct.

In order to solve this naming problem, all the traits that are used to describe operations must be
boxable so that we can turn `ComputePipeline<Very<Long<And<Complicated, Type>>>>` into
`ComputePipeline<Box<ComputePipelineDesc>>`. This means that we can't use associated types and
templates for any of the trait methods. Ideologically it is a bit annoying to have to restrict
ourselves in what we can do just because the user needs to be able to write out the precise type,
but it's the only pragmatic solution for now.

# Submissions

Any object that can be submitted to a GPU queue (for example a command buffer) implements
the `Submit` trait.

The `Submit` trait provides a function named `build` which returns a `Submission<Self>` object
(where `Self` is the type that implements the `Submit` trait). The `Submission` object must be kept
alive by the user for as long as the GPU hasn't finished executing the submission. Trying to
destroy a `Submission` will block until it is the case. Since the `Submission` holds the object
that was submitted, this object is also kept alive for as long as the GPU hasn't finished executing
it.

For the moment submitting an object always creates a fence, which is how the `Submission` knows
whether the GPU has finished executing it. Eventually this will need to be modified for the sake of
performance.

In order to make the `Submit` trait safer to implement, the method that actually needs to be
implemented is not `build` but `append_submission`. This method uses a API/lifetime trick to
guarantee that the GPU only executes command buffers that outlive the struct that implements
`Submit`.

SAFETY ISSUE HERE HOWEVER: the user can use mem::forget on the Submission and then drop the
objects referenced by it. There are two solutions to this: either store a bunch of Arc<Fence> in
every single object referenced by submissions (eg. pipeline objects), or force the user to use
either Arcs or give ownership of the object. The latter is preferred but not yet implemented.

# Pools

There are three kinds of pools in vulkano: memory pools, descriptor pools, and command pools. Only
the last two are technically Vulkan concepts, but using a memory pool is also a very common
pattern that you are strongly encouraged to embrace when you write a Vulkan application.

These three kinds of pools are each represented in vulkano by a trait. When you use the Vulkan API,
you are expected to create multiple command pools and multiple descriptor pools for maximum
performance. In vulkano however, it is the implementation of the pool trait that is responsible
for managing multiple actual pool objects. In other words a pool in vulkano is just a trait that
provides a method to allocate or free some resource, and the advanced functionality of Vulkan
pools (like resetting a command buffer, resetting a pool, or managing the descriptor pool's
capacity) is handled internally by the implementation of the trait. For example freeing a
command buffer can be implemented by resetting it and reusing it, instead of actually freeing it.

One of the goals of vulkano is to be easy to use by default. Therefore vulkano provides a default
implementation for each of these pools, and the `new` constructors of types that need a pool (ie.
buffers, images, descriptor sets, and command buffers) will use the default implementation. It is
possible for the user to use an alternative implementation of a pool by using an alternative
constructor, but the default implementations should be good for most usages. This is similar to
memory allocators in languages such as C++ and Rust, in the sense that some users want to be able
to use a custom allocator but most of the time it's not worth bothering with that.

# Command buffers

Command buffer objects belong to the last category of objects that were described above. They are
represented by an unsafe trait and can be implemented manually by the user if they wish.

However this poses a practical problem, which is that creating a command buffer in a safe way
is really complicated. There are tons of commands to implement, and each command has a ton of
safety requirements. If a user wants to create a custom command buffer type, it is just not an
option to ask them to reimplement these safety checks themselves.

The reason why users may want to create their own command buffer types is to implement
synchronization themselves. Vulkano's default implementation (which is `AutobarriersCommandBuffer`)
will automatically place pipeline barriers in order to handle cache flushes and image layout
transitions and avoid data races, but this automatic computation can be seen as expensive.

In order to make it possible to customize the synchronization story of command buffers, vulkano has
split the command buffer building process in two steps. First the user builds a list of commands
through an iterator-like API (and vulkano will check their validity), and then they are turned into
a command buffer through a trait. This means that the user can customize the synchronization
strategy (by customizing the second step) while still using the same command-building process
(the first step). Commands are not opinionated towards one strategy or another. The
command-building code is totally isolated from the synchronization strategy and only checks
whether the commands themselves are valid.

The fact that all the commands are added at once can be a little surprising for a user coming from
Vulkan. Vulkano's API looks very similar to Vulkan's API, but there is a major difference: in
Vulkan the cost of creating a command buffer is distributed between each function call, but in
vulkano it is done all at once. For example creating a command buffer with 6 commands with Vulkan
requires 8 function calls that take say 5µs each, while creating the same command buffer with
vulkano requires 8 function calls, but the first 7 are almost free and the last one takes 40µs.
After some thinking, it was considered to not be a problem.

Creating a list of commands with an iterator-like API has the problem that the type of the list of
commands changes every time you add a new command to the list
(just like for example `let iterator = iterator.skip(1)` changes the type of `iterator`). This is
a problem in situations where we don't know at compile-time the number of commands that we are
going to add. In order to solve this, it is required that the `CommandsList` trait be boxable,
so that the user can use a `Box<CommandsList>`. This is unfortunately not optimal as you will need
a memory allocation for each command that is added to the list. The situation here could still be
improved.

# The auto-barriers builder

As explained above, the default implementation of a command buffer provided by vulkano
automatically places pipeline barriers to avoid issues such as caches not being flushed, commands
being executed simultaneously when they shouldn't, or images having the wrong layout.

This is not an easy job, because Vulkan allows lots of weird access patterns that we want to make
available in vulkano. You can for example create a buffer object split into multiple sub-buffer
objects, or make some images and buffers share the same memory.

In order to make it possible to handle everything properly, the `Buffer` and `Image` traits need to
help us with the `conflicts` methods. Each buffer and image can be queried to know whether it
potentially uses the same memory as any other buffer or image. When two resources conflict, this
means that you can't write to one and read from the other one simultaneously or write to both
simultaneously.

But we don't want to check every single combination of buffer and image every time to check whether
they conflict. So in order to improve performance, buffers and images also need to provide a key
that identifies them. Two resources that can potentially conflict must always return the same key.
The regular `conflict` functions are still necessary to handle the situation where buffers or
images accidentally return the same key but don't actually conflict.

This conflict system is also used to make sure that the attachments of a framebuffer don't conflict
with each other or that the resources in a descriptor set don't conflict with each other (both
situations are forbidden).

# Image layouts

Tracking image layouts can be tedious. Vulkano uses a simple solution, which is that images must
always be in a specific layout at the beginning and the end of a command buffer. If a transition
is performed during a command buffer, the image must be transitioned back before the end of the
command buffer. The layout in question is queried with a method on the `Image` trait.

For example an `AttachmentImage` must always be in the `ColorAttachmentOptimal` layout for color
attachment, and the `DepthStencilAttachmentOptimal` layout for depth-stencil attachments. If any
command switches the image to another layout, then it will need to be switched back before the end
of the command buffer.

This system works very nicely in practice, and unnecessary layout transitions almost never happen.
The only situation where unnecessary transitions tend to happen in practice is for swapchain images
that are transitioned from `PresentSrc` to `ColorAttachmentOptimal` before the start of the
render pass, because the initial layout of the render pass attachment is `ColorAttachmentOptimal`
by default for color attachments. Vulkano should make it clear in the documentation of render
passes that the user is encouraged to specify when an attachment is expected to be in the
`PresentSrc` layout.

The only problematic area concerns the first usage of an image, where it must be transitioned from
the `Undefined` or `Preinitialized` layout. This is done by making the user pass a command buffer
builder in the constructor of images, and the constructor adds a transition command to it. The
image implementation is responsible for making sure that the transition command has been submitted
before any further command that uses the image.

# Inter-queue synchronization

When users submit two command buffers to two different queues, they expect the two command buffers
to execute in parallel. However this is forbidden if doing so could result in a data race,
like for example if one command buffer writes to an image and the other one reads from that same
image.
In this situation, the only possible technical solution is to make the execution of the second
command buffer block until the first command buffer has finished executing.
This case is similar to spawning two threads that each access the same resource protected by
a `RwLock` or a `Mutex`. One of the two threads will need to block until the first one is finished.

This raises the question: should vulkano implicitly block command buffers to avoid data races,
or should it force the user to explicitly add wait operations? By comparing a CPU-side
multithreaded program and a GPU-side multithreaded program, then the answer is to make it implicit,
as a CPU will also implicitly block when calling a function that happens to lock a `Mutex` or
a `RwLock`. In CPU code, these locking problems are always "fixed" by properly documenting the
behavior of the functions you call. Similarly, vulkano should precisely document its behavior.

More generally users are encouraged to avoid sharing resources between multiple queues unless these
resources are read-only, and in practice in a video game it is indeed rarely needed to share
resources between multiple queues. Just like for CPU-side multithreading, users are encouraged to
have a graph of the ways queues interact with each other.

However another problem arises. In order to make a command buffer wait for another, you need to
make the queue of the first command buffer submit a semaphore after execution, and the queue of
the second command buffer wait on that same semaphore before execution. Semaphores can only be used
once. This means that when you submit a command buffer to a queue, you must already know if any
other command buffers are going to wait on the one you are submitting, and if so how many. This is not
something that vulkano can automatically determine. The fact that there is therefore no optimal
algorithm for implicit synchronization would be a good point in favor of explicit synchronization.

The decision was taken to encourage users to explicitly handle synchronization between multiple
queues, but if they forget to do so then vulkano will automatically fall back to a dumb
worst-case-scenario but safe behavior. Whenever this dumb behavior is triggered, a debug message
is outputted by vulkano with the `vkDebugReportMessageEXT` function. This message can easily be
caught by the user by registering a callback, or with a debugger.

It is yet to be determined what exactly the user needs to handle. The user will at least need to
specify an optional list of semaphores to signal at each submission, but maybe not the list of
semaphores to wait upon if these can be determined automatically. This has yet to be seen.
