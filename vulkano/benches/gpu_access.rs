#[macro_use]
extern crate criterion;
extern crate vulkano;

use criterion::Criterion;
use vulkano::buffer::GpuAccess;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function_over_inputs("lock and unlock", |b, &&size| {
        let mut gpu_access = GpuAccess::new();

        b.iter(|| {
            for i in 1..size {
                let range = (i - 1) * 100..i * 100;
                gpu_access.try_lock(true, range.clone()).unwrap();
            }

            for i in 1..size {
                let range = (i - 1) * 100..i * 100;

                unsafe {
                    gpu_access.unlock(range);
                }
            }
        })
    }, &[100, 1000, 5000, 10000]);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
