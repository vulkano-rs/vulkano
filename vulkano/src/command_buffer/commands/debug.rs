use crate::{
    command_buffer::{sys::RecordingCommandBuffer, AutoCommandBufferBuilder},
    device::{DeviceOwned, QueueFlags},
    instance::debug::DebugUtilsLabel,
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};

/// # Commands for debugging.
///
/// These commands all require the [`ext_debug_utils`] extension to be enabled on the instance.
///
/// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
impl<L> AutoCommandBufferBuilder<L> {
    /// Opens a command buffer debug label region.
    pub fn begin_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_debug_utils_label(&label_info)?;

        Ok(unsafe { self.begin_debug_utils_label_unchecked(label_info) })
    }

    fn validate_begin_debug_utils_label(
        &self,
        label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_begin_debug_utils_label(label_info)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_debug_utils_label_unchecked(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> &mut Self {
        self.add_command(
            "begin_debug_utils_label",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.begin_debug_utils_label_unchecked(&label_info) };
            },
        );

        self
    }

    /// Closes a command buffer debug label region.
    ///
    /// # Safety
    ///
    /// - When submitting the command buffer, there must be an outstanding command buffer label
    ///   region begun with `begin_debug_utils_label` in the queue, either within this command
    ///   buffer or a previously submitted one.
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_debug_utils_label()?;

        Ok(unsafe { self.end_debug_utils_label_unchecked() })
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), Box<ValidationError>> {
        self.inner.validate_end_debug_utils_label()?;

        // TODO:
        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912
        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01913

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) -> &mut Self {
        self.add_command(
            "end_debug_utils_label",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.end_debug_utils_label_unchecked() };
            },
        );

        self
    }

    /// Inserts a command buffer debug label.
    pub fn insert_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_insert_debug_utils_label(&label_info)?;

        Ok(unsafe { self.insert_debug_utils_label_unchecked(label_info) })
    }

    fn validate_insert_debug_utils_label(
        &self,
        label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_insert_debug_utils_label(label_info)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn insert_debug_utils_label_unchecked(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> &mut Self {
        self.add_command(
            "insert_debug_utils_label",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.insert_debug_utils_label_unchecked(&label_info) };
            },
        );

        self
    }
}

impl RecordingCommandBuffer {
    #[inline]
    pub unsafe fn begin_debug_utils_label(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_debug_utils_label(label_info)?;

        Ok(unsafe { self.begin_debug_utils_label_unchecked(label_info) })
    }

    fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdBeginDebugUtilsLabelEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_debug_utils_label_unchecked(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> &mut Self {
        let label_info_fields1_vk = label_info.to_vk_fields1();
        let label_info_vk = label_info.to_vk(&label_info_fields1_vk);

        let fns = self.device().fns();
        unsafe {
            (fns.ext_debug_utils.cmd_begin_debug_utils_label_ext)(
                self.handle(),
                &raw const label_info_vk,
            )
        };

        self
    }

    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_debug_utils_label()?;

        Ok(unsafe { self.end_debug_utils_label_unchecked() })
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) -> &mut Self {
        let fns = self.device().fns();
        unsafe { (fns.ext_debug_utils.cmd_end_debug_utils_label_ext)(self.handle()) };

        self
    }

    #[inline]
    pub unsafe fn insert_debug_utils_label(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_insert_debug_utils_label(label_info)?;

        Ok(unsafe { self.insert_debug_utils_label_unchecked(label_info) })
    }

    fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics or compute operations"
                    .into(),
                vuids: &["VUID-vkCmdInsertDebugUtilsLabelEXT-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn insert_debug_utils_label_unchecked(
        &mut self,
        label_info: &DebugUtilsLabel,
    ) -> &mut Self {
        let label_info_fields1_vk = label_info.to_vk_fields1();
        let label_info_vk = label_info.to_vk(&label_info_fields1_vk);

        let fns = self.device().fns();
        unsafe {
            (fns.ext_debug_utils.cmd_insert_debug_utils_label_ext)(
                self.handle(),
                &raw const label_info_vk,
            )
        };

        self
    }
}
