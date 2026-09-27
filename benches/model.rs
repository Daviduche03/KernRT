use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kernrt::model::{Linear, Mlp};
use kernrt::ops::{Matrix, gemv, matmul_tiled, matvec, softmax};
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

fn mlp(input: usize, hidden: usize, classes: usize) -> Mlp {
    let fc1 = Linear::new(
        deterministic(&[input, hidden], 1),
        deterministic(&[hidden], 2),
    )
    .unwrap();
    let fc2 = Linear::new(
        deterministic(&[hidden, classes], 3),
        deterministic(&[classes], 4),
    )
    .unwrap();
    Mlp::new(fc1, fc2).unwrap()
}

fn bench_mlp(c: &mut Criterion) {
    let model = mlp(784, 128, 10);
    let mut group = c.benchmark_group("mlp_784_128_10");
    for &batch in &[1usize, 32, 256] {
        let x = deterministic(&[batch, 784], 5);
        group.bench_with_input(BenchmarkId::new("forward", batch), &x, |bench, x| {
            bench.iter(|| model.forward(black_box(x)).unwrap());
        });
        group.bench_with_input(
            BenchmarkId::new("forward_softmax", batch),
            &x,
            |bench, x| {
                bench.iter(|| softmax(&model.forward(black_box(x)).unwrap()));
            },
        );
    }
    group.finish();
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec");
    for &(rows, cols) in &[(784usize, 128usize), (128, 10)] {
        let mat = deterministic(&[rows, cols], 1);
        let x = deterministic(&[1, rows], 2);
        group.bench_with_input(
            BenchmarkId::new("matvec", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| bench.iter(|| matvec(black_box(&mat), black_box(x.data())).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("matmul_tiled", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| matmul_tiled(black_box(&x), black_box(&mat)).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv");
    // TinyStories + SmolLM2 projection shapes; the classifier dominates decode.
    for &(out_dim, in_dim) in &[
        (288usize, 288usize),
        (768, 288),
        (32000, 288),
        (576, 576),
        (192, 576),
        (1536, 576),
        (576, 1536),
        (49152, 576),
    ] {
        let w = deterministic(&[out_dim, in_dim], 1);
        let x = deterministic(&[in_dim], 2);
        let bits: Vec<u16> = w
            .data()
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let f32_matrix = Matrix::from_f32(w.clone());
        let bf16_matrix = Matrix::from_bf16_bits(bits, out_dim, in_dim);
        let q8_matrix = Matrix::from_f32_q8_0(w.clone()).unwrap();
        let q8x4_matrix = Matrix::from_f32_q8_0x4(w).unwrap();
        group.bench_with_input(
            BenchmarkId::new("f32", format!("{out_dim}x{in_dim}")),
            &(out_dim, in_dim),
            |bench, _| bench.iter(|| gemv(black_box(&f32_matrix), black_box(x.data())).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("bf16", format!("{out_dim}x{in_dim}")),
            &(out_dim, in_dim),
            |bench, _| bench.iter(|| gemv(black_box(&bf16_matrix), black_box(x.data())).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("q8_0", format!("{out_dim}x{in_dim}")),
            &(out_dim, in_dim),
            |bench, _| bench.iter(|| gemv(black_box(&q8_matrix), black_box(x.data())).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("q8_0x4", format!("{out_dim}x{in_dim}")),
            &(out_dim, in_dim),
            |bench, _| bench.iter(|| gemv(black_box(&q8x4_matrix), black_box(x.data())).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mlp, bench_matvec, bench_gemv);
criterion_main!(benches);
