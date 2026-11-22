use ash::vk;
use std::{
    cell::Cell,
    ffi::{c_char, CStr},
    mem,
    sync::{Arc, OnceLock},
    time::Duration,
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{future::FenceSignalFuture, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};

static SYSTEM_LOADER: OnceLock<Arc<VulkanLibrary>> = OnceLock::new();
static DEVICE_PROC_ADDR: OnceLock<vk::PFN_vkGetDeviceProcAddr> = OnceLock::new();

thread_local! {
    static FENCE_WAIT_ERROR: Cell<Option<vk::Result>> = const { Cell::new(None) };
}

unsafe extern "system" fn get_instance_proc_addr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: The caller provides the original loader's Vulkan handles and a valid function name.
    let function = unsafe {
        SYSTEM_LOADER
            .get()
            .unwrap()
            .get_instance_proc_addr(instance, name)
    }?;

    // SAFETY: Vulkan supplies a null-terminated function name.
    if unsafe { CStr::from_ptr(name) } == c"vkGetDeviceProcAddr" {
        DEVICE_PROC_ADDR.get_or_init(|| {
            // SAFETY: The loader returned the pointer for this exact function name.
            unsafe {
                mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetDeviceProcAddr>(function)
            }
        });

        // SAFETY: The Vulkan loader uses this erased type for function pointers.
        Some(unsafe {
            mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                get_device_proc_addr,
            )
        })
    } else {
        Some(function)
    }
}

unsafe extern "system" fn get_device_proc_addr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: Forwarding the caller's Vulkan device handle and function name unchanged.
    let function = unsafe { DEVICE_PROC_ADDR.get().unwrap()(device, name) }?;

    // SAFETY: Vulkan supplies a null-terminated function name.
    if unsafe { CStr::from_ptr(name) } == c"vkWaitForFences" {
        // SAFETY: The Vulkan loader uses this erased type for function pointers.
        Some(unsafe {
            mem::transmute::<vk::PFN_vkWaitForFences, unsafe extern "system" fn()>(wait_for_fences)
        })
    } else {
        Some(function)
    }
}

unsafe extern "system" fn wait_for_fences(
    device: vk::Device,
    fence_count: u32,
    fences: *const vk::Fence,
    wait_all: vk::Bool32,
    timeout: u64,
) -> vk::Result {
    let injected = FENCE_WAIT_ERROR.take();

    if injected == Some(vk::Result::TIMEOUT) {
        return vk::Result::TIMEOUT;
    }

    // SAFETY: The device belongs to the instance used to load the device dispatch function.
    let function =
        unsafe { DEVICE_PROC_ADDR.get().unwrap()(device, c"vkWaitForFences".as_ptr()) }.unwrap();
    // SAFETY: The original loader provided the pointer for this exact entry point.
    let wait =
        unsafe { mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkWaitForFences>(function) };
    // SAFETY: All arguments are forwarded under the original Vulkan contract.
    let result = unsafe { wait(device, fence_count, fences, wait_all, timeout) };

    // The healthy driver must actually finish the work before we simulate a terminal result.
    if result == vk::Result::SUCCESS {
        injected.unwrap_or(result)
    } else {
        result
    }
}

fn test_instance() -> Option<&'static Arc<Instance>> {
    // All devices using the cached device dispatch function must come from the same instance.
    static INSTANCE: OnceLock<Option<Arc<Instance>>> = OnceLock::new();

    INSTANCE
        .get_or_init(|| {
            // SAFETY: Use the platform's Vulkan loader as in the other GPU tests.
            SYSTEM_LOADER
                .set(unsafe { VulkanLibrary::new() }.ok()?)
                .unwrap();

            // SAFETY: The system loader lives for the process. The wrapper uses Vulkan's ABI and
            // forwards all calls except the configured wait result.
            let library = unsafe { VulkanLibrary::from_loader(get_instance_proc_addr) }.unwrap();
            Instance::new(&library, &Default::default()).ok()
        })
        .as_ref()
}

fn submit_copy() -> Option<(Subbuffer<u32>, FenceSignalFuture<impl GpuFuture>)> {
    let (physical_device, queue_family_index) = test_instance()?
        .enumerate_physical_devices()
        .ok()?
        .into_iter()
        .find_map(|physical_device| {
            let queue_family_index = physical_device
                .queue_family_properties()
                .iter()
                .position(|properties| properties.queue_flags.intersects(QueueFlags::GRAPHICS))?;
            Some((physical_device, queue_family_index as u32))
        })?;
    let (device, queues) = Device::new(
        &physical_device,
        &DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .ok()?;
    let [queue] = queues.try_into().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
    let buffer = |usage, value| {
        Buffer::from_data(
            &memory_allocator,
            &BufferCreateInfo {
                usage,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            value,
        )
        .unwrap()
    };
    let src = buffer(BufferUsage::TRANSFER_SRC, 42u32);
    let dst = buffer(BufferUsage::TRANSFER_DST, 0);
    let allocator = Arc::new(StandardCommandBufferAllocator::new(
        &device,
        &Default::default(),
    ));
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer(CopyBufferInfo::new(src, dst.clone()))
        .unwrap();

    let future = builder
        .build()
        .unwrap()
        .execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    Some((dst, future))
}

#[test]
fn wait_keeps_work_on_timeout() {
    let Some((buffer, future)) = submit_copy() else {
        return;
    };

    FENCE_WAIT_ERROR.set(Some(vk::Result::TIMEOUT));
    assert!(matches!(
        future.wait(Some(Duration::ZERO)),
        Err(Validated::Error(VulkanError::Timeout))
    ));
    assert!(
        buffer.write().is_err(),
        "the timed-out wait must retain the GPU lock"
    );
    future.wait(None).unwrap();
    assert!(
        buffer.write().is_ok(),
        "the successful wait must retire the GPU lock"
    );
}

#[test]
fn wait_retires_work_on_device_loss() {
    let Some((buffer, future)) = submit_copy() else {
        return;
    };

    assert!(buffer.write().is_err());
    FENCE_WAIT_ERROR.set(Some(vk::Result::ERROR_DEVICE_LOST));
    assert!(matches!(
        future.wait(None),
        Err(Validated::Error(VulkanError::DeviceLost))
    ));
    assert!(
        buffer.write().is_ok(),
        "device loss must retire the GPU lock"
    );
}

#[test]
fn drop_retires_work_on_device_loss() {
    let Some((buffer, future)) = submit_copy() else {
        return;
    };

    assert!(buffer.write().is_err());
    FENCE_WAIT_ERROR.set(Some(vk::Result::ERROR_DEVICE_LOST));
    drop(future);
    assert!(
        FENCE_WAIT_ERROR.get().is_none(),
        "drop must wait on the fence"
    );
    assert!(buffer.write().is_ok(), "drop must retire the GPU lock");
}
