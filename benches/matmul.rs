use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kernrt::ops::{matmul, matmul_tiled};
use kernrt::tensor::Tensor;
use std::hint::black_box;

/// Deterministic pseudo-random fill, so benchmarks are reproducible without
/// pulling in the `rand` crate. This is a 64-bit LCG (Knuth's constants).
fn deterministic(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
        data.push(x);
    }
    Tensor::from_vec(data, shape).unwrap()
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    for &size in &[64usize, 128, 256, 512] {
        let a = deterministic(&[size, size], 1);
        let b = deterministic(&[size, size], 2);
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| matmul(black_box(&a), black_box(&b)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("tiled", size), &size, |bench, _| {
            bench.iter(|| matmul_tiled(black_box(&a), black_box(&b)).unwrap());
        });
        #[cfg(target_arch = "aarch64")]
        {
            group.bench_with_input(BenchmarkId::new("neon", size), &size, |bench, _| {
                bench.iter(|| kernrt::ops::matmul_neon(black_box(&a), black_box(&b)).unwrap());
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
