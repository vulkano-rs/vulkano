// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to compile several shaders together using vulkano-shaders macro,
// such that the macro generates unique Shader types per each compiled shader, but generates common
// shareable set of Rust structs representing corresponding structs in the source glsl code.
//
// Normally, each vulkano-shaders macro invocation among other things generates a `ty` submodule
// containing all Rust types per each "struct" declaration of glsl code. Using this submodule
// the user can organize type-safe interoperability between Rust code and the shader interface
// input/output data tied to these structs. However, if the user compiles several shaders in
// independent Rust modules, each of these modules would contain independent `ty` submodule with
// each own set of Rust types. So, even if both shaders contain the same(or partially intersecting)
// glsl structs they will be duplicated in each generated `ty` submodule and treated by Rust as
// independent types. As such it would be tricky to organize interoperability between shader
// interfaces in Rust.
//
// To solve this problem the user can use "shared" generation mode of the macro. In this mode the
// user declares all shaders that possibly share common layout interfaces in a single macro
// invocation. The macro will check that there is no inconsistency between declared glsl structs
// with the same names, and it will put all generated Rust structs for all shaders in just a single
// `ty` submodule.

use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    instance::{Instance, VulkanLibrary},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

fn main() {
    let entry = VulkanLibrary::default();
    let instance = Instance::new(entry, Default::default()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_compute())
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();

    mod shaders {
        vulkano_shaders::shader! {
            // We declaring two simple compute shaders with push and specialization constants in
            // their layout interfaces.
            //
            // First one is just multiplying each value from the input array of ints to provided
            // value in push constants struct. And the second one in turn adds this value instead of
            // multiplying.
            //
            // However both shaders declare glsl struct `Parameters` for push constants in each
            // shader. Since each of the struct has exactly the same interface, they will be
            // treated by the macro as "shared".
            //
            // Also, note that glsl code duplications between shader sources is not necessary too.
            // In more complex system the user may want to declare independent glsl file with
            // such types, and include it in each shader entry-point files using "#include"
            // directive.
            shaders: {
                // Generate single unique `SpecializationConstants` struct for all shaders since
                // their specialization interfaces are the same. This option is turned off
                // by default and the macro by default producing unique
                // structs(`MultSpecializationConstants`, `AddSpecializationConstants`)
                shared_constants: true,
                mult: {
                    ty: "compute",
                    src: "
                        #version 450

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(constant_id = 0) const bool enabled = true;

                        layout(push_constant) uniform Parameters {
                          int value;
                        } pc;

                        layout(set = 0, binding = 0) buffer Data {
                            uint data[];
                        } data;

                        void main() {
                            if (!enabled) {
                                return;
                            }
                            uint idx = gl_GlobalInvocationID.x;
                            data.data[idx] *= pc.value;
                        }
                    "
                },
                add: {
                    ty: "compute",
                    src: "
                        #version 450

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(constant_id = 0) const bool enabled = true;

                        layout(push_constant) uniform Parameters {
                          int value;
                        } pc;

                        layout(set = 0, binding = 0) buffer Data {
                            uint data[];
                        } data;

                        void main() {
                            if (!enabled) {
                                return;
                            }
                            uint idx = gl_GlobalInvocationID.x;
                            data.data[idx] += pc.value;
                        }
                    "
                }
            }
        }

        // The macro will create the following things in this module:
        // - `ShaderMult` for the first shader loader/entry-point.
        // - `ShaderAdd` for the second shader loader/entry-point.
        // `SpecializationConstants` Rust struct for both shader's specialization constants.
        // `ty` submodule with `Parameters` Rust struct common for both shaders.
    }

    // We introducing generic function responsible for running any of the shaders above with
    // provided Push Constants parameter.
    // Note that shader's interface `parameter` here is shader-independent.
    fn run_shader(
        pipeline: Arc<ComputePipeline>,
        queue: Arc<Queue>,
        data_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
        parameters: shaders::ty::Parameters,
    ) {
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            queue.device().clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .push_constants(pipeline.layout().clone(), 0, parameters)
            .dispatch([1024, 1, 1])
            .unwrap();
        let command_buffer = builder.build().unwrap();

        let future = sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
    }

    // Preparing test data array `[0, 1, 2, 3....]`
    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };

    // Loading the first shader, and creating a Pipeline for the shader
    let mult_pipeline = ComputePipeline::new(
        device.clone(),
        shaders::load_mult(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap(),
        &shaders::SpecializationConstants { enabled: 1 },
        None,
        |_| {},
    )
    .unwrap();

    // Loading the second shader, and creating a Pipeline for the shader
    let add_pipeline = ComputePipeline::new(
        device.clone(),
        shaders::load_add(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap(),
        &shaders::SpecializationConstants { enabled: 1 },
        None,
        |_| {},
    )
    .unwrap();

    // Multiply each value by 2
    run_shader(
        mult_pipeline.clone(),
        queue.clone(),
        data_buffer.clone(),
        shaders::ty::Parameters { value: 2 },
    );

    // Then add 1 to each value
    run_shader(
        add_pipeline,
        queue.clone(),
        data_buffer.clone(),
        shaders::ty::Parameters { value: 1 },
    );

    // Then multiply each value by 3
    run_shader(
        mult_pipeline,
        queue.clone(),
        data_buffer.clone(),
        shaders::ty::Parameters { value: 3 },
    );

    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], (n * 2 + 1) * 3);
    }
    println!("Success");
}
