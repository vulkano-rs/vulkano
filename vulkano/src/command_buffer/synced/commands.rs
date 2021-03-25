// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::VulkanObject;
use buffer::BufferAccess;
use command_buffer::synced::base::Command;
use command_buffer::synced::base::FinalCommand;
use command_buffer::synced::base::KeyTy;
use command_buffer::synced::base::SyncCommandBufferBuilder;
use command_buffer::synced::base::SyncCommandBufferBuilderError;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::UnsafeCommandBufferBuilderBindVertexBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilderBufferImageCopy;
use command_buffer::sys::UnsafeCommandBufferBuilderColorImageClear;
use command_buffer::sys::UnsafeCommandBufferBuilderExecuteCommands;
use command_buffer::sys::UnsafeCommandBufferBuilderImageBlit;
use command_buffer::sys::UnsafeCommandBufferBuilderImageCopy;
use command_buffer::CommandBuffer;
use command_buffer::CommandBufferExecError;
use command_buffer::SubpassContents;
use descriptor::descriptor::DescriptorDescTy;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use format::ClearValue;
use framebuffer::FramebufferAbstract;
use image::ImageAccess;
use image::ImageLayout;
use pipeline::depth_stencil::DynamicStencilValue;
use pipeline::depth_stencil::StencilFaceFlags;
use pipeline::input_assembly::IndexType;
use pipeline::viewport::Scissor;
use pipeline::viewport::Viewport;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use sampler::Filter;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ffi::CStr;
use std::mem;
use std::ptr;
use std::sync::Arc;
use sync::AccessFlagBits;
use sync::Event;
use sync::PipelineMemoryAccess;
use sync::PipelineStages;
use SafeDeref;

impl SyncCommandBufferBuilder {
    /// Calls `vkBeginRenderPass` on the builder.
    // TODO: it shouldn't be possible to get an error if the framebuffer checked conflicts already
    // TODO: after begin_render_pass has been called, flushing should be forbidden and an error
    //       returned if conflict
    #[inline]
    pub unsafe fn begin_render_pass<F, I>(
        &mut self,
        framebuffer: F,
        subpass_contents: SubpassContents,
        clear_values: I,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        F: FramebufferAbstract + Send + Sync + 'static,
        I: Iterator<Item = ClearValue> + Send + Sync + 'static,
    {
        struct Cmd<F, I> {
            framebuffer: F,
            subpass_contents: SubpassContents,
            clear_values: Option<I>,
        }

        impl<F, I> Command for Cmd<F, I>
        where
            F: FramebufferAbstract + Send + Sync + 'static,
            I: Iterator<Item = ClearValue>,
        {
            fn name(&self) -> &'static str {
                "vkCmdBeginRenderPass"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.begin_render_pass(
                    &self.framebuffer,
                    self.subpass_contents,
                    self.clear_values.take().unwrap(),
                );
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<F>(F);
                impl<F> FinalCommand for Fin<F>
                where
                    F: FramebufferAbstract + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdBeginRenderPass"
                    }
                    fn image(&self, num: usize) -> &dyn ImageAccess {
                        self.0.attached_image_view(num).unwrap().image()
                    }
                    fn image_name(&self, num: usize) -> Cow<'static, str> {
                        format!("attachment {}", num).into()
                    }
                }
                Box::new(Fin(self.framebuffer))
            }

            fn image(&self, num: usize) -> &dyn ImageAccess {
                self.framebuffer.attached_image_view(num).unwrap().image()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                format!("attachment {}", num).into()
            }
        }

        let resources = (0..framebuffer.num_attachments())
            .map(|atch| {
                let desc = framebuffer.attachment_desc(atch).unwrap();
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                all_commands: true,
                                ..PipelineStages::none()
                            }, // TODO: wrong!
                            access: AccessFlagBits {
                                input_attachment_read: true,
                                color_attachment_read: true,
                                color_attachment_write: true,
                                depth_stencil_attachment_read: true,
                                depth_stencil_attachment_write: true,
                                ..AccessFlagBits::none()
                            }, // TODO: suboptimal
                            exclusive: true, // TODO: suboptimal ; note: remember to always pass true if desc.initial_layout != desc.final_layout
                        },
                        desc.initial_layout,
                        desc.final_layout,
                    )),
                )
            })
            .collect::<Vec<_>>();

        self.append_command(
            Cmd {
                framebuffer,
                subpass_contents,
                clear_values: Some(clear_values),
            },
            &resources,
        )?;

        self.prev_cmd_entered_render_pass();
        Ok(())
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer<B>(
        &mut self,
        buffer: B,
        index_ty: IndexType,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        struct Cmd<B> {
            buffer: B,
            index_ty: IndexType,
        }

        impl<B> Command for Cmd<B>
        where
            B: BufferAccess + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdBindIndexBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_index_buffer(&self.buffer, self.index_ty);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                where
                    B: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdBindIndexBuffer"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "index buffer".into()
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "index buffer".into()
            }
        }

        self.append_command(
            Cmd { buffer, index_ty },
            &[(
                KeyTy::Buffer,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            vertex_input: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlagBits {
                            index_read: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: false,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )?;

        Ok(())
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics<Gp>(&mut self, pipeline: Gp)
    where
        Gp: GraphicsPipelineAbstract + Send + Sync + 'static,
    {
        struct Cmd<Gp> {
            pipeline: Gp,
        }

        impl<Gp> Command for Cmd<Gp>
        where
            Gp: GraphicsPipelineAbstract + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdBindPipeline"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_pipeline_graphics(&self.pipeline);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<Gp>(Gp);
                impl<Gp> FinalCommand for Fin<Gp>
                where
                    Gp: Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdBindPipeline"
                    }
                }
                Box::new(Fin(self.pipeline))
            }
        }

        self.append_command(Cmd { pipeline }, &[]).unwrap();
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute<Cp>(&mut self, pipeline: Cp)
    where
        Cp: ComputePipelineAbstract + Send + Sync + 'static,
    {
        struct Cmd<Gp> {
            pipeline: Gp,
        }

        impl<Gp> Command for Cmd<Gp>
        where
            Gp: ComputePipelineAbstract + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdBindPipeline"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_pipeline_compute(&self.pipeline);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<Cp>(Cp);
                impl<Cp> FinalCommand for Fin<Cp>
                where
                    Cp: Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdBindPipeline"
                    }
                }
                Box::new(Fin(self.pipeline))
            }
        }

        self.append_command(Cmd { pipeline }, &[]).unwrap();
    }

    /// Starts the process of binding descriptor sets. Returns an intermediate struct which can be
    /// used to add the sets.
    #[inline]
    pub fn bind_descriptor_sets(&mut self) -> SyncCommandBufferBuilderBindDescriptorSets {
        SyncCommandBufferBuilderBindDescriptorSets {
            builder: self,
            inner: SmallVec::new(),
        }
    }

    /// Starts the process of binding vertex buffers. Returns an intermediate struct which can be
    /// used to add the buffers.
    #[inline]
    pub fn bind_vertex_buffers(&mut self) -> SyncCommandBufferBuilderBindVertexBuffer {
        SyncCommandBufferBuilderBindVertexBuffer {
            builder: self,
            inner: UnsafeCommandBufferBuilderBindVertexBuffer::new(),
            buffers: Vec::new(),
        }
    }

    /// Calls `vkCmdCopyImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image<S, D, R>(
        &mut self,
        source: S,
        source_layout: ImageLayout,
        destination: D,
        destination_layout: ImageLayout,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
        R: Iterator<Item = UnsafeCommandBufferBuilderImageCopy> + Send + Sync + 'static,
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            source_layout: ImageLayout,
            destination: Option<D>,
            destination_layout: ImageLayout,
            regions: Option<R>,
        }

        impl<S, D, R> Command for Cmd<S, D, R>
        where
            S: ImageAccess + Send + Sync + 'static,
            D: ImageAccess + Send + Sync + 'static,
            R: Iterator<Item = UnsafeCommandBufferBuilderImageCopy>,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_image(
                    self.source.as_ref().unwrap(),
                    self.source_layout,
                    self.destination.as_ref().unwrap(),
                    self.destination_layout,
                    self.regions.take().unwrap(),
                );
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                where
                    S: ImageAccess + Send + Sync + 'static,
                    D: ImageAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdCopyImage"
                    }
                    fn image(&self, num: usize) -> &dyn ImageAccess {
                        if num == 0 {
                            &self.0
                        } else if num == 1 {
                            &self.1
                        } else {
                            panic!()
                        }
                    }
                    fn image_name(&self, num: usize) -> Cow<'static, str> {
                        if num == 0 {
                            "source".into()
                        } else if num == 1 {
                            "destination".into()
                        } else {
                            panic!()
                        }
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(
                    self.source.take().unwrap(),
                    self.destination.take().unwrap(),
                ))
            }

            fn image(&self, num: usize) -> &dyn ImageAccess {
                if num == 0 {
                    self.source.as_ref().unwrap()
                } else if num == 1 {
                    self.destination.as_ref().unwrap()
                } else {
                    panic!()
                }
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                if num == 0 {
                    "source".into()
                } else if num == 1 {
                    "destination".into()
                } else {
                    panic!()
                }
            }
        }

        self.append_command(
            Cmd {
                source: Some(source),
                source_layout,
                destination: Some(destination),
                destination_layout,
                regions: Some(regions),
            },
            &[
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_read: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: false,
                        },
                        source_layout,
                        source_layout,
                    )),
                ),
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_write: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: true,
                        },
                        destination_layout,
                        destination_layout,
                    )),
                ),
            ],
        )?;

        Ok(())
    }

    /// Calls `vkCmdBlitImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn blit_image<S, D, R>(
        &mut self,
        source: S,
        source_layout: ImageLayout,
        destination: D,
        destination_layout: ImageLayout,
        regions: R,
        filter: Filter,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
        R: Iterator<Item = UnsafeCommandBufferBuilderImageBlit> + Send + Sync + 'static,
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            source_layout: ImageLayout,
            destination: Option<D>,
            destination_layout: ImageLayout,
            regions: Option<R>,
            filter: Filter,
        }

        impl<S, D, R> Command for Cmd<S, D, R>
        where
            S: ImageAccess + Send + Sync + 'static,
            D: ImageAccess + Send + Sync + 'static,
            R: Iterator<Item = UnsafeCommandBufferBuilderImageBlit>,
        {
            fn name(&self) -> &'static str {
                "vkCmdBlitImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.blit_image(
                    self.source.as_ref().unwrap(),
                    self.source_layout,
                    self.destination.as_ref().unwrap(),
                    self.destination_layout,
                    self.regions.take().unwrap(),
                    self.filter,
                );
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                where
                    S: ImageAccess + Send + Sync + 'static,
                    D: ImageAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdBlitImage"
                    }
                    fn image(&self, num: usize) -> &dyn ImageAccess {
                        if num == 0 {
                            &self.0
                        } else if num == 1 {
                            &self.1
                        } else {
                            panic!()
                        }
                    }
                    fn image_name(&self, num: usize) -> Cow<'static, str> {
                        if num == 0 {
                            "source".into()
                        } else if num == 1 {
                            "destination".into()
                        } else {
                            panic!()
                        }
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(
                    self.source.take().unwrap(),
                    self.destination.take().unwrap(),
                ))
            }

            fn image(&self, num: usize) -> &dyn ImageAccess {
                if num == 0 {
                    self.source.as_ref().unwrap()
                } else if num == 1 {
                    self.destination.as_ref().unwrap()
                } else {
                    panic!()
                }
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                if num == 0 {
                    "source".into()
                } else if num == 1 {
                    "destination".into()
                } else {
                    panic!()
                }
            }
        }

        self.append_command(
            Cmd {
                source: Some(source),
                source_layout,
                destination: Some(destination),
                destination_layout,
                regions: Some(regions),
                filter,
            },
            &[
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_read: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: false,
                        },
                        source_layout,
                        source_layout,
                    )),
                ),
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_write: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: true,
                        },
                        destination_layout,
                        destination_layout,
                    )),
                ),
            ],
        )?;

        Ok(())
    }

    /// Calls `vkCmdClearColorImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    pub unsafe fn clear_color_image<I, R>(
        &mut self,
        image: I,
        layout: ImageLayout,
        color: ClearValue,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        I: ImageAccess + Send + Sync + 'static,
        R: Iterator<Item = UnsafeCommandBufferBuilderColorImageClear> + Send + Sync + 'static,
    {
        struct Cmd<I, R> {
            image: Option<I>,
            layout: ImageLayout,
            color: ClearValue,
            regions: Option<R>,
        }

        impl<I, R> Command for Cmd<I, R>
        where
            I: ImageAccess + Send + Sync + 'static,
            R: Iterator<Item = UnsafeCommandBufferBuilderColorImageClear> + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdClearColorImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.clear_color_image(
                    self.image.as_ref().unwrap(),
                    self.layout,
                    self.color,
                    self.regions.take().unwrap(),
                );
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<I>(I);
                impl<I> FinalCommand for Fin<I>
                where
                    I: ImageAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdClearColorImage"
                    }
                    fn image(&self, num: usize) -> &dyn ImageAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn image_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "target".into()
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.image` without using an Option.
                Box::new(Fin(self.image.take().unwrap()))
            }

            fn image(&self, num: usize) -> &dyn ImageAccess {
                assert_eq!(num, 0);
                self.image.as_ref().unwrap()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "target".into()
            }
        }

        self.append_command(
            Cmd {
                image: Some(image),
                layout,
                color,
                regions: Some(regions),
            },
            &[(
                KeyTy::Image,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            transfer: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlagBits {
                            transfer_write: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: true,
                    },
                    layout,
                    layout,
                )),
            )],
        )?;

        Ok(())
    }

    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer<S, D, R>(
        &mut self,
        source: S,
        destination: D,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        S: BufferAccess + Send + Sync + 'static,
        D: BufferAccess + Send + Sync + 'static,
        R: Iterator<Item = (usize, usize, usize)> + Send + Sync + 'static,
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            destination: Option<D>,
            regions: Option<R>,
        }

        impl<S, D, R> Command for Cmd<S, D, R>
        where
            S: BufferAccess + Send + Sync + 'static,
            D: BufferAccess + Send + Sync + 'static,
            R: Iterator<Item = (usize, usize, usize)>,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_buffer(
                    self.source.as_ref().unwrap(),
                    self.destination.as_ref().unwrap(),
                    self.regions.take().unwrap(),
                );
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                where
                    S: BufferAccess + Send + Sync + 'static,
                    D: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdCopyBuffer"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        match num {
                            0 => &self.0,
                            1 => &self.1,
                            _ => panic!(),
                        }
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        match num {
                            0 => "source".into(),
                            1 => "destination".into(),
                            _ => panic!(),
                        }
                    }
                }
                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(
                    self.source.take().unwrap(),
                    self.destination.take().unwrap(),
                ))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                match num {
                    0 => self.source.as_ref().unwrap(),
                    1 => self.destination.as_ref().unwrap(),
                    _ => panic!(),
                }
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                match num {
                    0 => "source".into(),
                    1 => "destination".into(),
                    _ => panic!(),
                }
            }
        }

        self.append_command(
            Cmd {
                source: Some(source),
                destination: Some(destination),
                regions: Some(regions),
            },
            &[
                (
                    KeyTy::Buffer,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_read: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: false,
                        },
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
                (
                    KeyTy::Buffer,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_write: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: true,
                        },
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
            ],
        )?;

        Ok(())
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image<S, D, R>(
        &mut self,
        source: S,
        destination: D,
        destination_layout: ImageLayout,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        S: BufferAccess + Send + Sync + 'static,
        D: ImageAccess + Send + Sync + 'static,
        R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync + 'static,
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            destination: Option<D>,
            destination_layout: ImageLayout,
            regions: Option<R>,
        }

        impl<S, D, R> Command for Cmd<S, D, R>
        where
            S: BufferAccess + Send + Sync + 'static,
            D: ImageAccess + Send + Sync + 'static,
            R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyBufferToImage"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_buffer_to_image(
                    self.source.as_ref().unwrap(),
                    self.destination.as_ref().unwrap(),
                    self.destination_layout,
                    self.regions.take().unwrap(),
                );
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                where
                    S: BufferAccess + Send + Sync + 'static,
                    D: ImageAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdCopyBufferToImage"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "source".into()
                    }
                    fn image(&self, num: usize) -> &dyn ImageAccess {
                        assert_eq!(num, 0);
                        &self.1
                    }
                    fn image_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "destination".into()
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(
                    self.source.take().unwrap(),
                    self.destination.take().unwrap(),
                ))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                self.source.as_ref().unwrap()
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "source".into()
            }

            fn image(&self, num: usize) -> &dyn ImageAccess {
                assert_eq!(num, 0);
                self.destination.as_ref().unwrap()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "destination".into()
            }
        }

        self.append_command(
            Cmd {
                source: Some(source),
                destination: Some(destination),
                destination_layout,
                regions: Some(regions),
            },
            &[
                (
                    KeyTy::Buffer,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_read: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: false,
                        },
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_write: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: true,
                        },
                        destination_layout,
                        destination_layout,
                    )),
                ),
            ],
        )?;

        Ok(())
    }

    /// Calls `vkCmdCopyImageToBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_image_to_buffer<S, D, R>(
        &mut self,
        source: S,
        source_layout: ImageLayout,
        destination: D,
        regions: R,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        S: ImageAccess + Send + Sync + 'static,
        D: BufferAccess + Send + Sync + 'static,
        R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy> + Send + Sync + 'static,
    {
        struct Cmd<S, D, R> {
            source: Option<S>,
            source_layout: ImageLayout,
            destination: Option<D>,
            regions: Option<R>,
        }

        impl<S, D, R> Command for Cmd<S, D, R>
        where
            S: ImageAccess + Send + Sync + 'static,
            D: BufferAccess + Send + Sync + 'static,
            R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>,
        {
            fn name(&self) -> &'static str {
                "vkCmdCopyImageToBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.copy_image_to_buffer(
                    self.source.as_ref().unwrap(),
                    self.source_layout,
                    self.destination.as_ref().unwrap(),
                    self.regions.take().unwrap(),
                );
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<S, D>(S, D);
                impl<S, D> FinalCommand for Fin<S, D>
                where
                    S: ImageAccess + Send + Sync + 'static,
                    D: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdCopyImageToBuffer"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.1
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "destination".into()
                    }
                    fn image(&self, num: usize) -> &dyn ImageAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn image_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "source".into()
                    }
                }

                // Note: borrow checker somehow doesn't accept `self.source` and `self.destination`
                // without using an Option.
                Box::new(Fin(
                    self.source.take().unwrap(),
                    self.destination.take().unwrap(),
                ))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                self.destination.as_ref().unwrap()
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "destination".into()
            }

            fn image(&self, num: usize) -> &dyn ImageAccess {
                assert_eq!(num, 0);
                self.source.as_ref().unwrap()
            }

            fn image_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "source".into()
            }
        }

        self.append_command(
            Cmd {
                source: Some(source),
                destination: Some(destination),
                source_layout,
                regions: Some(regions),
            },
            &[
                (
                    KeyTy::Image,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_read: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: false,
                        },
                        source_layout,
                        source_layout,
                    )),
                ),
                (
                    KeyTy::Buffer,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                transfer: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                transfer_write: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: true,
                        },
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                ),
            ],
        )?;

        Ok(())
    }

    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn debug_marker_begin(&mut self, name: &'static CStr, color: [f32; 4]) {
        struct Cmd {
            name: &'static CStr,
            color: [f32; 4],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBeginDebugUtilsLabelEXT"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.debug_marker_begin(self.name, self.color);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdBeginDebugUtilsLabelEXT")
            }
        }

        self.append_command(Cmd { name, color }, &[]).unwrap();
    }

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// - The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    /// - There must be an outstanding `debug_marker_begin` command prior to the
    /// `debug_marker_end` on the queue.
    #[inline]
    pub unsafe fn debug_marker_end(&mut self) {
        struct Cmd {}

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdEndDebugUtilsLabelEXT"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.debug_marker_end();
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdEndDebugUtilsLabelEXT")
            }
        }

        self.append_command(Cmd {}, &[]).unwrap();
    }

    /// Calls `vkCmdInsertDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn debug_marker_insert(&mut self, name: &'static CStr, color: [f32; 4]) {
        struct Cmd {
            name: &'static CStr,
            color: [f32; 4],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdInsertDebugUtilsLabelEXT"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.debug_marker_insert(self.name, self.color);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdInsertDebugUtilsLabelEXT")
            }
        }

        self.append_command(Cmd { name, color }, &[]).unwrap();
    }

    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(&mut self, dimensions: [u32; 3]) {
        struct Cmd {
            dimensions: [u32; 3],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdDispatch"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.dispatch(self.dimensions);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdDispatch")
            }
        }

        self.append_command(Cmd { dimensions }, &[]).unwrap();
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect<B>(
        &mut self,
        buffer: B,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        struct Cmd<B> {
            buffer: B,
        }

        impl<B> Command for Cmd<B>
        where
            B: BufferAccess + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdDispatchIndirect"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.dispatch_indirect(&self.buffer);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                where
                    B: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdDispatchIndirect"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "indirect buffer".into()
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "indirect buffer".into()
            }
        }

        self.append_command(
            Cmd { buffer },
            &[(
                KeyTy::Buffer,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            draw_indirect: true,
                            ..PipelineStages::none()
                        }, // TODO: is draw_indirect correct?
                        access: AccessFlagBits {
                            indirect_command_read: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: false,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )?;

        Ok(())
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        struct Cmd {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdDraw"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw(
                    self.vertex_count,
                    self.instance_count,
                    self.first_vertex,
                    self.first_instance,
                );
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdDraw")
            }
        }

        self.append_command(
            Cmd {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        struct Cmd {
            index_count: u32,
            instance_count: u32,
            first_index: u32,
            vertex_offset: i32,
            first_instance: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdDrawIndexed"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw_indexed(
                    self.index_count,
                    self.instance_count,
                    self.first_index,
                    self.vertex_offset,
                    self.first_instance,
                );
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdDrawIndexed")
            }
        }

        self.append_command(
            Cmd {
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect<B>(
        &mut self,
        buffer: B,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        struct Cmd<B> {
            buffer: B,
            draw_count: u32,
            stride: u32,
        }

        impl<B> Command for Cmd<B>
        where
            B: BufferAccess + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdDrawIndirect"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw_indirect(&self.buffer, self.draw_count, self.stride);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                where
                    B: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdDrawIndirect"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "indirect buffer".into()
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "indirect buffer".into()
            }
        }

        self.append_command(
            Cmd {
                buffer,
                draw_count,
                stride,
            },
            &[(
                KeyTy::Buffer,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            draw_indirect: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlagBits {
                            indirect_command_read: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: false,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )?;

        Ok(())
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect<B>(
        &mut self,
        buffer: B,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        struct Cmd<B> {
            buffer: B,
            draw_count: u32,
            stride: u32,
        }

        impl<B> Command for Cmd<B>
        where
            B: BufferAccess + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdDrawIndexedIndirect"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.draw_indexed_indirect(&self.buffer, self.draw_count, self.stride);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                where
                    B: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdDrawIndexedIndirect"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        assert_eq!(num, 0);
                        "indirect buffer".into()
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                assert_eq!(num, 0);
                "indirect buffer".into()
            }
        }

        self.append_command(
            Cmd {
                buffer,
                draw_count,
                stride,
            },
            &[(
                KeyTy::Buffer,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            draw_indirect: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlagBits {
                            indirect_command_read: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: false,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )?;

        Ok(())
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        struct Cmd;

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdEndRenderPass"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.end_render_pass();
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdEndRenderPass")
            }
        }

        self.append_command(Cmd, &[]).unwrap();
        self.prev_cmd_left_render_pass();
    }

    /// Starts the process of executing secondary command buffers. Returns an intermediate struct
    /// which can be used to add the command buffers.
    #[inline]
    pub unsafe fn execute_commands(&mut self) -> SyncCommandBufferBuilderExecuteCommands {
        SyncCommandBufferBuilderExecuteCommands {
            builder: self,
            inner: Vec::new(),
        }
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer<B>(&mut self, buffer: B, data: u32)
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        struct Cmd<B> {
            buffer: B,
            data: u32,
        }

        impl<B> Command for Cmd<B>
        where
            B: BufferAccess + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdFillBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.fill_buffer(&self.buffer, self.data);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                where
                    B: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdFillBuffer"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, _: usize) -> Cow<'static, str> {
                        "destination".into()
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, _: usize) -> Cow<'static, str> {
                "destination".into()
            }
        }

        self.append_command(
            Cmd { buffer, data },
            &[(
                KeyTy::Buffer,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            transfer: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlagBits {
                            transfer_write: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: true,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )
        .unwrap();
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
        struct Cmd {
            subpass_contents: SubpassContents,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdNextSubpass"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.next_subpass(self.subpass_contents);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdNextSubpass")
            }
        }

        self.append_command(Cmd { subpass_contents }, &[]).unwrap();
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
    pub unsafe fn push_constants<Pl, D>(
        &mut self,
        pipeline_layout: Pl,
        stages: ShaderStages,
        offset: u32,
        size: u32,
        data: &D,
    ) where
        Pl: PipelineLayoutAbstract + Send + Sync + 'static,
        D: ?Sized + Send + Sync + 'static,
    {
        struct Cmd<Pl> {
            pipeline_layout: Pl,
            stages: ShaderStages,
            offset: u32,
            size: u32,
            data: Box<[u8]>,
        }

        impl<Pl> Command for Cmd<Pl>
        where
            Pl: PipelineLayoutAbstract + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdPushConstants"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.push_constants::<_, [u8]>(
                    &self.pipeline_layout,
                    self.stages,
                    self.offset,
                    self.size,
                    &self.data,
                );
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<Pl>(Pl);
                impl<Pl> FinalCommand for Fin<Pl>
                where
                    Pl: Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdPushConstants"
                    }
                }
                Box::new(Fin(self.pipeline_layout))
            }
        }

        debug_assert!(mem::size_of_val(data) >= size as usize);

        let mut out = Vec::with_capacity(size as usize);
        ptr::copy::<u8>(
            data as *const D as *const u8,
            out.as_mut_ptr(),
            size as usize,
        );
        out.set_len(size as usize);

        self.append_command(
            Cmd {
                pipeline_layout,
                stages,
                offset,
                size,
                data: out.into(),
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        struct Cmd {
            event: Arc<Event>,
            stages: PipelineStages,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdResetEvent"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.reset_event(&self.event, self.stages);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin(Arc<Event>);
                impl FinalCommand for Fin {
                    fn name(&self) -> &'static str {
                        "vkCmdResetEvent"
                    }
                }
                Box::new(Fin(self.event))
            }
        }

        self.append_command(Cmd { event, stages }, &[]).unwrap();
    }

    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        struct Cmd {
            constants: [f32; 4],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetBlendConstants"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_blend_constants(self.constants);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetBlendConstants")
            }
        }

        self.append_command(Cmd { constants }, &[]).unwrap();
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        struct Cmd {
            constant_factor: f32,
            clamp: f32,
            slope_factor: f32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetDepthBias"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bias(self.constant_factor, self.clamp, self.slope_factor);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetDepthBias")
            }
        }

        self.append_command(
            Cmd {
                constant_factor,
                clamp,
                slope_factor,
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        struct Cmd {
            min: f32,
            max: f32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetDepthBounds"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bounds(self.min, self.max);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetDepthBounds")
            }
        }

        self.append_command(Cmd { min, max }, &[]).unwrap();
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: Arc<Event>, stages: PipelineStages) {
        struct Cmd {
            event: Arc<Event>,
            stages: PipelineStages,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetEvent"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_event(&self.event, self.stages);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin(Arc<Event>);
                impl FinalCommand for Fin {
                    fn name(&self) -> &'static str {
                        "vkCmdSetEvent"
                    }
                }
                Box::new(Fin(self.event))
            }
        }

        self.append_command(Cmd { event, stages }, &[]).unwrap();
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        struct Cmd {
            line_width: f32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetLineWidth"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_line_width(self.line_width);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetLineWidth")
            }
        }

        self.append_command(Cmd { line_width }, &[]).unwrap();
    }

    /// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(&mut self, compare_mask: DynamicStencilValue) {
        struct Cmd {
            face_mask: StencilFaceFlags,
            compare_mask: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetStencilCompareMask"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_compare_mask(self.face_mask, self.compare_mask);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetStencilCompareMask")
            }
        }

        self.append_command(
            Cmd {
                face_mask: compare_mask.face,
                compare_mask: compare_mask.value,
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdSetStencilReference` on the builder.
    #[inline]
    pub unsafe fn set_stencil_reference(&mut self, reference: DynamicStencilValue) {
        struct Cmd {
            face_mask: StencilFaceFlags,
            reference: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetStencilReference"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_reference(self.face_mask, self.reference);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetStencilReference")
            }
        }

        self.append_command(
            Cmd {
                face_mask: reference.face,
                reference: reference.value,
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(&mut self, write_mask: DynamicStencilValue) {
        struct Cmd {
            face_mask: StencilFaceFlags,
            write_mask: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdSetStencilWriteMask"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_write_mask(self.face_mask, self.write_mask);
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetStencilWriteMask")
            }
        }

        self.append_command(
            Cmd {
                face_mask: write_mask.face,
                write_mask: write_mask.value,
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
    where
        I: Iterator<Item = Scissor> + Send + Sync + 'static,
    {
        struct Cmd<I> {
            first_scissor: u32,
            scissors: Option<I>,
        }

        impl<I> Command for Cmd<I>
        where
            I: Iterator<Item = Scissor>,
        {
            fn name(&self) -> &'static str {
                "vkCmdSetScissor"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_scissor(self.first_scissor, self.scissors.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetScissor")
            }
        }

        self.append_command(
            Cmd {
                first_scissor,
                scissors: Some(scissors),
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
    where
        I: Iterator<Item = Viewport> + Send + Sync + 'static,
    {
        struct Cmd<I> {
            first_viewport: u32,
            viewports: Option<I>,
        }

        impl<I> Command for Cmd<I>
        where
            I: Iterator<Item = Viewport>,
        {
            fn name(&self) -> &'static str {
                "vkCmdSetViewport"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_viewport(self.first_viewport, self.viewports.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                Box::new("vkCmdSetViewport")
            }
        }

        self.append_command(
            Cmd {
                first_viewport,
                viewports: Some(viewports),
            },
            &[],
        )
        .unwrap();
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<B, D, Dd>(&mut self, buffer: B, data: Dd)
    where
        B: BufferAccess + Send + Sync + 'static,
        D: ?Sized,
        Dd: SafeDeref<Target = D> + Send + Sync + 'static,
    {
        struct Cmd<B, Dd> {
            buffer: B,
            data: Dd,
        }

        impl<B, D, Dd> Command for Cmd<B, Dd>
        where
            B: BufferAccess + Send + Sync + 'static,
            D: ?Sized,
            Dd: SafeDeref<Target = D> + Send + Sync + 'static,
        {
            fn name(&self) -> &'static str {
                "vkCmdUpdateBuffer"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.update_buffer(&self.buffer, self.data.deref());
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin<B>(B);
                impl<B> FinalCommand for Fin<B>
                where
                    B: BufferAccess + Send + Sync + 'static,
                {
                    fn name(&self) -> &'static str {
                        "vkCmdUpdateBuffer"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        assert_eq!(num, 0);
                        &self.0
                    }
                    fn buffer_name(&self, _: usize) -> Cow<'static, str> {
                        "destination".into()
                    }
                }
                Box::new(Fin(self.buffer))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                assert_eq!(num, 0);
                &self.buffer
            }

            fn buffer_name(&self, _: usize) -> Cow<'static, str> {
                "destination".into()
            }
        }

        self.append_command(
            Cmd { buffer, data },
            &[(
                KeyTy::Buffer,
                Some((
                    PipelineMemoryAccess {
                        stages: PipelineStages {
                            transfer: true,
                            ..PipelineStages::none()
                        },
                        access: AccessFlagBits {
                            transfer_write: true,
                            ..AccessFlagBits::none()
                        },
                        exclusive: true,
                    },
                    ImageLayout::Undefined,
                    ImageLayout::Undefined,
                )),
            )],
        )
        .unwrap();
    }
}

pub struct SyncCommandBufferBuilderBindDescriptorSets<'b> {
    builder: &'b mut SyncCommandBufferBuilder,
    inner: SmallVec<[Box<dyn DescriptorSet + Send + Sync>; 12]>,
}

impl<'b> SyncCommandBufferBuilderBindDescriptorSets<'b> {
    /// Adds a descriptor set to the list.
    #[inline]
    pub fn add<S>(&mut self, set: S)
    where
        S: DescriptorSet + Send + Sync + 'static,
    {
        self.inner.push(Box::new(set));
    }

    #[inline]
    pub unsafe fn submit<Pl, I>(
        self,
        graphics: bool,
        pipeline_layout: Pl,
        first_binding: u32,
        dynamic_offsets: I,
    ) -> Result<(), SyncCommandBufferBuilderError>
    where
        Pl: PipelineLayoutAbstract + Send + Sync + 'static,
        I: Iterator<Item = u32> + Send + Sync + 'static,
    {
        if self.inner.is_empty() {
            return Ok(());
        }

        struct Cmd<Pl, I> {
            inner: SmallVec<[Box<dyn DescriptorSet + Send + Sync>; 12]>,
            graphics: bool,
            pipeline_layout: Pl,
            first_binding: u32,
            dynamic_offsets: Option<I>,
        }

        impl<Pl, I> Command for Cmd<Pl, I>
        where
            Pl: PipelineLayoutAbstract,
            I: Iterator<Item = u32>,
        {
            fn name(&self) -> &'static str {
                "vkCmdBindDescriptorSets"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_descriptor_sets(
                    self.graphics,
                    &self.pipeline_layout,
                    self.first_binding,
                    self.inner.iter().map(|s| s.inner()),
                    self.dynamic_offsets.take().unwrap(),
                );
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin(SmallVec<[Box<dyn DescriptorSet + Send + Sync>; 12]>);
                impl FinalCommand for Fin {
                    fn name(&self) -> &'static str {
                        "vkCmdBindDescriptorSets"
                    }
                    fn buffer(&self, mut num: usize) -> &dyn BufferAccess {
                        for set in self.0.iter() {
                            if let Some(buf) = set.buffer(num) {
                                return buf.0;
                            }
                            num -= set.num_buffers();
                        }
                        panic!()
                    }
                    fn buffer_name(&self, mut num: usize) -> Cow<'static, str> {
                        for (set_num, set) in self.0.iter().enumerate() {
                            if let Some(buf) = set.buffer(num) {
                                return format!(
                                    "Buffer bound to descriptor {} of set {}",
                                    buf.1, set_num
                                )
                                .into();
                            }
                            num -= set.num_buffers();
                        }
                        panic!()
                    }
                    fn image(&self, mut num: usize) -> &dyn ImageAccess {
                        for set in self.0.iter() {
                            if let Some(img) = set.image(num) {
                                return img.0.image();
                            }
                            num -= set.num_images();
                        }
                        panic!()
                    }
                    fn image_name(&self, mut num: usize) -> Cow<'static, str> {
                        for (set_num, set) in self.0.iter().enumerate() {
                            if let Some(img) = set.image(num) {
                                return format!(
                                    "Image bound to descriptor {} of set {}",
                                    img.1, set_num
                                )
                                .into();
                            }
                            num -= set.num_images();
                        }
                        panic!()
                    }
                }
                Box::new(Fin(self.inner))
            }

            fn buffer(&self, mut num: usize) -> &dyn BufferAccess {
                for set in self.inner.iter() {
                    if let Some(buf) = set.buffer(num) {
                        return buf.0;
                    }
                    num -= set.num_buffers();
                }
                panic!()
            }

            fn buffer_name(&self, mut num: usize) -> Cow<'static, str> {
                for (set_num, set) in self.inner.iter().enumerate() {
                    if let Some(buf) = set.buffer(num) {
                        return format!("Buffer bound to descriptor {} of set {}", buf.1, set_num)
                            .into();
                    }
                    num -= set.num_buffers();
                }
                panic!()
            }

            fn image(&self, mut num: usize) -> &dyn ImageAccess {
                for set in self.inner.iter() {
                    if let Some(img) = set.image(num) {
                        return img.0.image();
                    }
                    num -= set.num_images();
                }
                panic!()
            }

            fn image_name(&self, mut num: usize) -> Cow<'static, str> {
                for (set_num, set) in self.inner.iter().enumerate() {
                    if let Some(img) = set.image(num) {
                        return format!("Image bound to descriptor {} of set {}", img.1, set_num)
                            .into();
                    }
                    num -= set.num_images();
                }
                panic!()
            }
        }

        let resources = {
            let mut resources = Vec::new();
            for ds in self.inner.iter() {
                for buf_num in 0..ds.num_buffers() {
                    let desc = ds
                        .descriptor(ds.buffer(buf_num).unwrap().1 as usize)
                        .unwrap();
                    let exclusive = !desc.readonly;
                    let (stages, access) = desc.pipeline_stages_and_access();
                    resources.push((
                        KeyTy::Buffer,
                        Some((
                            PipelineMemoryAccess {
                                stages,
                                access,
                                exclusive,
                            },
                            ImageLayout::Undefined,
                            ImageLayout::Undefined,
                        )),
                    ));
                }
                for img_num in 0..ds.num_images() {
                    let (image_view, desc_num) = ds.image(img_num).unwrap();
                    let desc = ds.descriptor(desc_num as usize).unwrap();
                    let exclusive = !desc.readonly;
                    let (stages, access) = desc.pipeline_stages_and_access();
                    let mut ignore_me_hack = false;
                    let layouts = image_view
                        .image()
                        .descriptor_layouts()
                        .expect("descriptor_layouts must return Some when used in an image view");
                    let layout = match desc.ty {
                        DescriptorDescTy::CombinedImageSampler(_) => layouts.combined_image_sampler,
                        DescriptorDescTy::Image(ref img) => {
                            if img.sampled {
                                layouts.sampled_image
                            } else {
                                layouts.storage_image
                            }
                        }
                        DescriptorDescTy::InputAttachment { .. } => {
                            // FIXME: This is tricky. Since we read from the input attachment
                            // and this input attachment is being written in an earlier pass,
                            // vulkano will think that it needs to put a pipeline barrier and will
                            // return a `Conflict` error. For now as a work-around we simply ignore
                            // input attachments.
                            ignore_me_hack = true;
                            layouts.input_attachment
                        }
                        _ => panic!("Tried to bind an image to a non-image descriptor"),
                    };
                    resources.push((
                        KeyTy::Image,
                        if ignore_me_hack {
                            None
                        } else {
                            Some((
                                PipelineMemoryAccess {
                                    stages,
                                    access,
                                    exclusive,
                                },
                                layout,
                                layout,
                            ))
                        },
                    ));
                }
            }
            resources
        };

        self.builder.append_command(
            Cmd {
                inner: self.inner,
                graphics,
                pipeline_layout,
                first_binding,
                dynamic_offsets: Some(dynamic_offsets),
            },
            &resources,
        )?;

        Ok(())
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct SyncCommandBufferBuilderBindVertexBuffer<'a> {
    builder: &'a mut SyncCommandBufferBuilder,
    inner: UnsafeCommandBufferBuilderBindVertexBuffer,
    buffers: Vec<Box<dyn BufferAccess + Send + Sync>>,
}

impl<'a> SyncCommandBufferBuilderBindVertexBuffer<'a> {
    /// Adds a buffer to the list.
    #[inline]
    pub fn add<B>(&mut self, buffer: B)
    where
        B: BufferAccess + Send + Sync + 'static,
    {
        self.inner.add(&buffer);
        self.buffers.push(Box::new(buffer));
    }

    #[inline]
    pub unsafe fn submit(self, first_binding: u32) -> Result<(), SyncCommandBufferBuilderError> {
        struct Cmd {
            first_binding: u32,
            inner: Option<UnsafeCommandBufferBuilderBindVertexBuffer>,
            buffers: Vec<Box<dyn BufferAccess + Send + Sync>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdBindVertexBuffers"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                out.bind_vertex_buffers(self.first_binding, self.inner.take().unwrap());
            }

            fn into_final_command(self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin(Vec<Box<dyn BufferAccess + Send + Sync>>);
                impl FinalCommand for Fin {
                    fn name(&self) -> &'static str {
                        "vkCmdBindVertexBuffers"
                    }
                    fn buffer(&self, num: usize) -> &dyn BufferAccess {
                        &self.0[num]
                    }
                    fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                        format!("Buffer #{}", num).into()
                    }
                }
                Box::new(Fin(self.buffers))
            }

            fn buffer(&self, num: usize) -> &dyn BufferAccess {
                &self.buffers[num]
            }

            fn buffer_name(&self, num: usize) -> Cow<'static, str> {
                format!("Buffer #{}", num).into()
            }
        }

        let resources = self
            .buffers
            .iter()
            .map(|_| {
                (
                    KeyTy::Buffer,
                    Some((
                        PipelineMemoryAccess {
                            stages: PipelineStages {
                                vertex_input: true,
                                ..PipelineStages::none()
                            },
                            access: AccessFlagBits {
                                vertex_attribute_read: true,
                                ..AccessFlagBits::none()
                            },
                            exclusive: false,
                        },
                        ImageLayout::Undefined,
                        ImageLayout::Undefined,
                    )),
                )
            })
            .collect::<Vec<_>>();

        self.builder.append_command(
            Cmd {
                first_binding,
                inner: Some(self.inner),
                buffers: self.buffers,
            },
            &resources,
        )?;

        Ok(())
    }
}

/// Prototype for a `vkCmdExecuteCommands`.
pub struct SyncCommandBufferBuilderExecuteCommands<'a> {
    builder: &'a mut SyncCommandBufferBuilder,
    inner: Vec<Box<dyn CommandBuffer + Send + Sync>>,
}

impl<'a> SyncCommandBufferBuilderExecuteCommands<'a> {
    /// Adds a command buffer to the list.
    #[inline]
    pub fn add<C>(&mut self, command_buffer: C)
    where
        C: CommandBuffer + Send + Sync + 'static,
    {
        self.inner.push(Box::new(command_buffer));
    }

    #[inline]
    pub unsafe fn submit(self) -> Result<(), SyncCommandBufferBuilderError> {
        struct DropUnlock(Box<dyn CommandBuffer + Send + Sync>);
        impl std::ops::Deref for DropUnlock {
            type Target = Box<dyn CommandBuffer + Send + Sync>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        unsafe impl SafeDeref for DropUnlock {}

        impl Drop for DropUnlock {
            fn drop(&mut self) {
                unsafe {
                    self.unlock();
                }
            }
        }

        struct Cmd(Vec<DropUnlock>);

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "vkCmdExecuteCommands"
            }

            unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder) {
                let mut execute = UnsafeCommandBufferBuilderExecuteCommands::new();
                self.0
                    .iter()
                    .for_each(|cbuf| execute.add_raw(cbuf.inner().internal_object()));
                out.execute_commands(execute);
            }

            fn into_final_command(mut self: Box<Self>) -> Box<dyn FinalCommand + Send + Sync> {
                struct Fin(Vec<DropUnlock>);
                impl FinalCommand for Fin {
                    fn name(&self) -> &'static str {
                        "vkCmdExecuteCommands"
                    }

                    fn buffer(&self, mut num: usize) -> &dyn BufferAccess {
                        for cbuf in self.0.iter() {
                            if let Some(buf) = cbuf.buffer(num) {
                                return buf.0;
                            }
                            num -= cbuf.num_buffers();
                        }
                        panic!()
                    }

                    fn buffer_name(&self, mut num: usize) -> Cow<'static, str> {
                        for (cbuf_num, cbuf) in self.0.iter().enumerate() {
                            if let Some(buf) = cbuf.buffer(num) {
                                return format!(
                                    "Buffer bound to secondary command buffer {}",
                                    cbuf_num
                                )
                                .into();
                            }
                            num -= cbuf.num_buffers();
                        }
                        panic!()
                    }

                    fn image(&self, mut num: usize) -> &dyn ImageAccess {
                        for cbuf in self.0.iter() {
                            if let Some(img) = cbuf.image(num) {
                                return img.0;
                            }
                            num -= cbuf.num_images();
                        }
                        panic!()
                    }

                    fn image_name(&self, mut num: usize) -> Cow<'static, str> {
                        for (cbuf_num, cbuf) in self.0.iter().enumerate() {
                            if let Some(img) = cbuf.image(num) {
                                return format!(
                                    "Image bound to secondary command buffer {}",
                                    cbuf_num
                                )
                                .into();
                            }
                            num -= cbuf.num_images();
                        }
                        panic!()
                    }
                }
                Box::new(Fin(std::mem::take(&mut self.0)))
            }

            fn buffer(&self, mut num: usize) -> &dyn BufferAccess {
                for cbuf in self.0.iter() {
                    if let Some(buf) = cbuf.buffer(num) {
                        return buf.0;
                    }
                    num -= cbuf.num_buffers();
                }
                panic!()
            }

            fn buffer_name(&self, mut num: usize) -> Cow<'static, str> {
                for (cbuf_num, cbuf) in self.0.iter().enumerate() {
                    if let Some(buf) = cbuf.buffer(num) {
                        return format!("Buffer bound to secondary command buffer {}", cbuf_num)
                            .into();
                    }
                    num -= cbuf.num_buffers();
                }
                panic!()
            }

            fn image(&self, mut num: usize) -> &dyn ImageAccess {
                for cbuf in self.0.iter() {
                    if let Some(img) = cbuf.image(num) {
                        return img.0;
                    }
                    num -= cbuf.num_images();
                }
                panic!()
            }

            fn image_name(&self, mut num: usize) -> Cow<'static, str> {
                for (cbuf_num, cbuf) in self.0.iter().enumerate() {
                    if let Some(img) = cbuf.image(num) {
                        return format!("Image bound to secondary command buffer {}", cbuf_num)
                            .into();
                    }
                    num -= cbuf.num_images();
                }
                panic!()
            }
        }

        let resources = {
            let mut resources = Vec::new();
            for cbuf in self.inner.iter() {
                for buf_num in 0..cbuf.num_buffers() {
                    resources.push((
                        KeyTy::Buffer,
                        Some((
                            cbuf.buffer(buf_num).unwrap().1,
                            ImageLayout::Undefined,
                            ImageLayout::Undefined,
                        )),
                    ));
                }
                for img_num in 0..cbuf.num_images() {
                    let (_, memory, start_layout, end_layout) = cbuf.image(img_num).unwrap();
                    resources.push((KeyTy::Image, Some((memory, start_layout, end_layout))));
                }
            }
            resources
        };

        self.builder.append_command(
            Cmd(self
                .inner
                .into_iter()
                .map(|cbuf| {
                    cbuf.lock_record()?;
                    Ok(DropUnlock(cbuf))
                })
                .collect::<Result<Vec<_>, CommandBufferExecError>>()?),
            &resources,
        )?;

        Ok(())
    }
}
