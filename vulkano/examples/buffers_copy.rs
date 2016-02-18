#[macro_use]
extern crate vulkano;

use std::sync::Arc;

fn main() {
    let app = vulkano::instance::ApplicationInfo { application_name: "test", application_version: 1, engine_name: "test", engine_version: 1 };
    let instance = vulkano::instance::Instance::new(Some(&app), None).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let queue = physical.queue_families().find(|q| q.supports_transfers())
                                                .expect("couldn't find a graphical queue family");


    let (device, queues) = vulkano::device::Device::new(&physical, physical.supported_features(),
                                                        [(queue, 0.5)].iter().cloned())
                                                                .expect("failed to create device");
    let queue = queues.into_iter().next().unwrap();



    let src: Arc<vulkano::buffer::Buffer<[u8; 16], _>> =
                                vulkano::buffer::Buffer::new(&device, &vulkano::buffer::Usage::all(),
                                                             vulkano::memory::HostVisible, &queue)
                                                                .expect("failed to create buffer");

    {
        let mut mapping = src.try_write().unwrap();
        for (v, o) in mapping.iter_mut().enumerate() { *o = v as u8; }
    }

    let dest: Arc<vulkano::buffer::Buffer<[u8; 16], _>> =
                                vulkano::buffer::Buffer::new(&device, &vulkano::buffer::Usage::all(),
                                                             vulkano::memory::HostVisible, &queue)
                                                                .expect("failed to create buffer");

    let cb_pool = vulkano::command_buffer::CommandBufferPool::new(&device, &queue.lock().unwrap().family())
                                                  .expect("failed to create command buffer pool");

    let command_buffer = vulkano::command_buffer::PrimaryCommandBufferBuilder::new(&cb_pool).unwrap()
        .copy_buffer(&src, &dest)
        .build().unwrap();

    {
        let mut queue = queue.lock().unwrap();
        command_buffer.submit(&mut queue).unwrap();
    }

    {
        let mut mapping = dest.read(1000000000);
        for (v, o) in mapping.iter_mut().enumerate() { assert_eq!(*o, v as u8); }
    }
}
