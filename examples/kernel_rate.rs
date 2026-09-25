//! Print inner-kernel GFLOP/s for cache-resident sizes.
use kernrt::ops::matmul_packed;
use kernrt::tensor::Tensor;
use std::time::Instant;

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

fn main() {
    for &(m, k, n) in &[
        (128usize, 512usize, 512usize),
        (512, 512, 512),
        (256, 256, 256),
        (1024, 1024, 1024),
    ] {
        let a = Tensor::from_vec(det(m * k, 1), &[m, k]).unwrap();
        let b = Tensor::from_vec(det(k * n, 2), &[k, n]).unwrap();
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let _ = matmul_packed(&a, &b).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{m}x{k}x{n}: {:.2} ms  {:.1} GFLOP/s",
            best * 1e3,
            flops / best / 1e9
        );
    }
}
