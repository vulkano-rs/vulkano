
/// Represents a prototype of a command buffer.
///
/// # Usage
///
/// ```norun
/// let commands_buffer =
///     CommandBufferBuilder::new(&device)
///         .copy_memory(..., ...)
///         .draw(...)
///         .build();
/// 
/// ```
///
/// # Implementation
///
/// Builds a command buffer and starts writing into it. Each function call directly adds a command
/// into the buffer.
///
/// Resources that are used are held in the `CommandBufferBuilder` and the `CommandBuffer`,
/// ensuring that they don't get destroyed as long as the command buffer is alive.
///
pub struct CommandBufferBuilder {
    device: Arc<Device>,
    cmd: Option<ffi::GR_CMD_BUFFER>,
    memory_refs: Vec<ffi::GR_MEMORY_REF>,
    buffers: Vec<Arc<Buffer>>,
    images: Vec<Arc<Image>>,
    pipelines: Vec<Arc<Pipeline>>,
}

impl CommandBufferBuilder {
    /// Builds a new prototype of a command buffer.
    pub fn new(device: &Arc<Device>) -> CommandBufferBuilder {
        let infos = ffi::GR_CMD_BUFFER_CREATE_INFO {
            queueType: ffi::GR_QUEUE_UNIVERSAL,
            flags: 0,
        };

        let cmd_buffer = unsafe {
            let mut cmd = mem::uninitialized();
            error::check_result(ffi::grCreateCommandBuffer(*device.get_id(),
                                                           &infos, &mut cmd)).unwrap();
            cmd
        };

        error::check_result(unsafe { ffi::grBeginCommandBuffer(cmd_buffer, 0) }).unwrap();

        CommandBufferBuilder {
            device: device.clone(),
            cmd: Some(cmd_buffer),
            memory_refs: Vec::new(),
        }
    }

    /// Builds the command buffer containing all the commands.
    pub fn build(mut self) -> Arc<CommandBuffer> {
        let cmd_buffer = self.cmd.take().unwrap();
        error::check_result(unsafe { ffi::grEndCommandBuffer(cmd_buffer) }).unwrap();

        Arc::new(CommandBuffer {
            device: self.device.clone(),
            cmd: cmd_buffer,
            memory_refs: mem::replace(&mut self.memory_refs, Vec::with_capacity(0)),
            buffers: mem::replace(&mut self.buffers, Vec::with_capacity(0)),
            images: mem::replace(&mut self.images, Vec::with_capacity(0)),
            pipelines: mem::replace(&mut self.pipelines, Vec::with_capacity(0)),
        })
    }
}

impl Drop for CommandBufferBuilder {
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd {
            error::check_result(unsafe { ffi::grEndCommandBuffer(cmd) }).unwrap();
            error::check_result(unsafe { ffi::grDestroyObject(cmd) }).unwrap();
        }
    }
}

/// Represents a command buffer.
///
/// Can be submitted to a queue.
pub struct CommandBuffer {
    device: Arc<Device>,
    cmd: ffi::GR_CMD_BUFFER,
    memory_refs: Vec<ffi::GR_MEMORY_REF>,
    buffers: Vec<Arc<Buffer>>,
    images: Vec<Arc<Image>>,
    pipelines: Vec<Arc<Pipeline>>,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        error::check_result(unsafe { ffi::grDestroyObject(cmd) }).unwrap();
    }
}
