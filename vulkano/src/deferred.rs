// Copyright (c) 2023 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Operations on the host that can be deferred.
//!
//! You typically pass a [`DeferredOperation`] object as part of a call to another function that
//! performs some potentially time-consuming work. The operation will then not be performed
//! immediately, but is put on hold. You must then call [`join`] repeatedly on one or more threads
//! to make the operation progress, until it is complete.
//!
//! [`join`]: DeferredOperation::join

use crate::{
    device::{Device, DeviceOwned},
    RequiresOneOf, VulkanError, VulkanObject,
};
use std::{mem::MaybeUninit, ptr, sync::Arc};

/// An operation on the host that has been deferred.
///
/// The object cannot be dropped while an operation is pending. If it is dropped
/// prematurely, the current thread will block to wait for the operation to finish.
#[derive(Debug)]
pub struct DeferredOperation {
    device: Arc<Device>,
    handle: ash::vk::DeferredOperationKHR,
}

impl DeferredOperation {
    /// Creates a new `DeferredOperation`.
    ///
    /// The [`khr_deferred_host_operations`] extension must be enabled on the device.
    ///
    /// [`khr_deferred_host_operations`]: crate::device::DeviceExtensions::khr_deferred_host_operations
    #[inline]
    pub fn new(device: Arc<Device>) -> Result<Arc<Self>, DeferredOperationCreateError> {
        Self::validate_new(&device)?;

        unsafe { Ok(Self::new_unchecked(device)?) }
    }

    fn validate_new(device: &Device) -> Result<(), DeferredOperationCreateError> {
        if !device.enabled_extensions().khr_deferred_host_operations {
            return Err(DeferredOperationCreateError::RequirementNotMet {
                required_for: "`DeferredOperation::new`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_deferred_host_operations"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(device: Arc<Device>) -> Result<Arc<Self>, VulkanError> {
        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_deferred_host_operations
                .create_deferred_operation_khr)(
                device.handle(), ptr::null(), output.as_mut_ptr()
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle))
    }

    /// Creates a new `DeferredOperation` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::DeferredOperationKHR,
    ) -> Arc<Self> {
        Arc::new(Self { device, handle })
    }

    /// Executes a portion of the operation on the current thread.
    pub fn join(&self) -> Result<DeferredOperationJoinStatus, VulkanError> {
        let result = unsafe {
            let fns = self.device.fns();
            (fns.khr_deferred_host_operations.deferred_operation_join_khr)(
                self.device.handle(),
                self.handle,
            )
        };

        match result {
            ash::vk::Result::SUCCESS => Ok(DeferredOperationJoinStatus::Complete),
            ash::vk::Result::THREAD_DONE_KHR => Ok(DeferredOperationJoinStatus::ThreadDone),
            ash::vk::Result::THREAD_IDLE_KHR => Ok(DeferredOperationJoinStatus::ThreadIdle),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Returns the result of the operation, or `None` if the operation is not yet complete.
    pub fn result(&self) -> Option<Result<(), VulkanError>> {
        let result = unsafe {
            let fns = self.device.fns();
            (fns.khr_deferred_host_operations
                .get_deferred_operation_result_khr)(self.device.handle(), self.handle)
        };

        match result {
            ash::vk::Result::NOT_READY => None,
            ash::vk::Result::SUCCESS => Some(Ok(())),
            err => Some(Err(VulkanError::from(err))),
        }
    }

    /// Waits for the operation to complete, then returns its result.
    pub fn wait(&self) -> Result<Result<(), VulkanError>, VulkanError> {
        // Based on example code on the extension's spec page.

        // Call `join` until we get `Complete` or `ThreadDone`.
        loop {
            match self.join()? {
                DeferredOperationJoinStatus::Complete => {
                    break;
                }
                DeferredOperationJoinStatus::ThreadDone => {
                    std::thread::yield_now();
                    break;
                }
                DeferredOperationJoinStatus::ThreadIdle => {}
            }
        }

        // Call `result` until we get `Some`.
        loop {
            if let Some(result) = self.result() {
                return Ok(result);
            }

            std::thread::yield_now();
        }
    }

    /// The maximum number of threads that could usefully execute the operation at this point in
    /// its execution, or zero if the operation is complete.
    ///
    /// Returns `None` if no exact number of threads can be calculated.
    pub fn max_concurrency(&self) -> Option<u32> {
        let result = unsafe {
            let fns = self.device.fns();
            (fns.khr_deferred_host_operations
                .get_deferred_operation_max_concurrency_khr)(
                self.device.handle(), self.handle
            )
        };

        (result != u32::MAX).then_some(result)
    }
}

impl Drop for DeferredOperation {
    #[inline]
    fn drop(&mut self) {
        let _ = self.wait(); // Ignore errors

        unsafe {
            let fns = self.device.fns();
            (fns.khr_deferred_host_operations
                .destroy_deferred_operation_khr)(
                self.device.handle(), self.handle, ptr::null()
            );
        }
    }
}

unsafe impl VulkanObject for DeferredOperation {
    type Handle = ash::vk::DeferredOperationKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DeferredOperation {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// Error that can happen when creating a `DeferredOperation`.
#[derive(Clone, Debug)]
pub enum DeferredOperationCreateError {
    VulkanError(VulkanError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl From<VulkanError> for DeferredOperationCreateError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

/// The status of the operation after [`join`] returns.
///
/// [`join`]: DeferredOperation::join
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeferredOperationJoinStatus {
    /// The operation completed.
    Complete,

    /// The operation did not complete yet,
    /// but there is no more work to be done on the current thread.
    ThreadDone,

    /// The operation did not complete yet,
    /// and there may be work to do on the current thread in the future.
    ThreadIdle,
}
