//! Throughput of the inner kernel on a cache-resident problem.
//!
//! Sizing the operands to fit in L1/L2 removes memory from the measurement, so
//! what is left is the inner loop itself. That is the quantity register blocking
//! changes, and it is invisible in the end-to-end gemv numbers because those
//! are dominated by DRAM.
use criterion::{Criterion, criterion_group, criterion_main};
use kernrt::ops::matmul_packed;
use kernrt::tensor::Tensor;
use std::hint::black_box;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
    }
    v
}

fn probe(c: &mut Criterion) {
    let mut g = c.benchmark_group("microkernel");
    // 512^3 is 1 MB of A, 1 MB of B, 1 MB of C: L2 resident on M1.
    for &(m, k, n) in &[(512usize, 512usize, 512usize), (256, 256, 256)] {
        let a = Tensor::from_vec(det(m * k, 1), &[m, k]).unwrap();
        let b = Tensor::from_vec(det(k * n, 2), &[k, n]).unwrap();
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        g.bench_with_input(format!("{m}x{k}x{n}"), &(a, b), |bench, (a, b)| {
            bench.iter(|| black_box(matmul_packed(black_box(a), black_box(b)).unwrap()))
        });
        println!("{m}x{k}x{n}: {:.2} GFLOP", flops / 1e9);
    }
    g.finish();
}
criterion_group!(benches, probe);
criterion_main!(benches);
