//! Numeric operations over [`Tensor`](crate::tensor::Tensor).
//!
//! The kernels are split by concern rather than kept in one file:
//!
//! * [`dense`] — matrix multiplication and matrix-vector products, including
//!   the packed GEMM and the register-tiled micro-kernels.
//! * [`quant`] — [`Matrix`] and its storages (f32, bf16, Q8_0) and the
//!   quantized `gemv` that consumes them.
//! * [`dot`] — the dot-product primitives the `gemv` kernels are built from,
//!   one per storage width.
//! * [`elementwise`] — `relu`, `add_bias`, `softmax`.
//!
//! Everything is re-exported here, so `kernrt::ops::matmul` and friends read
//! the same as they did when this was one module. The tests live here because
//! they exercise several of the submodules together.

mod dense;
mod dot;
mod elementwise;
mod quant;

pub use dense::{
    gemm_nn_into, gemm_nt_into, matmul, matmul_neon, matmul_packed, matmul_packed_nt, matmul_tiled,
    matvec, transpose,
};
pub(crate) use elementwise::softmax_in_place;
pub use elementwise::{add_bias, relu, softmax};
pub use quant::{
    Matrix, Q8_0_BLOCK, Q8_0X4_ROWS, Q8Vec, gemm_q8_0x4_batched_into, gemv, gemv_into,
    gemv_q8_0_into,
};

#[allow(unused_imports)]
pub(crate) use dot::{
    bf16_to_f32, dot, dot_bf16, dot_bf16_scalar, dot_q8_0, dot_q8_0_scalar, dot_scalar,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, TensorError};

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape).unwrap()
    }

    /// Deterministic pseudo-random fill in `[-0.5, 0.5)`, so both kernels see
    /// non-trivial values with no `rand` dependency.
    fn random(shape: &[usize], seed: u64) -> Tensor {
        let count: usize = shape.iter().product();
        let mut state = seed;
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.push(((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
        }
        Tensor::from_vec(data, shape).unwrap()
    }

    #[test]
    fn matmul_2x2_known_result() {
        let a = t(vec![1., 2., 3., 4.], &[2, 2]);
        let b = t(vec![5., 6., 7., 8.], &[2, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[19., 22., 43., 50.]);
    }

    #[test]
    fn matmul_identity_is_left_unchanged() {
        let a = t(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let i = t(vec![1., 0., 0., 0., 1., 0., 0., 0., 1.], &[3, 3]);
        let c = matmul(&a, &i).unwrap();
        assert_eq!(c.data(), a.data());
    }

    #[test]
    fn matmul_non_square() {
        let a = t(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = t(vec![7., 8., 9., 10., 11., 12.], &[3, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.data(), &[58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let a = t(vec![1., 2., 3., 4.], &[2, 2]);
        let b = t(vec![1., 2., 3., 4., 5., 6.], &[3, 2]);
        assert!(matches!(
            matmul(&a, &b),
            Err(TensorError::ShapeMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn matmul_rejects_non_2d() {
        let a = Tensor::zeros(&[2, 2, 2]);
        let b = Tensor::zeros(&[2, 2]);
        assert!(matches!(
            matmul(&a, &b),
            Err(TensorError::RankMismatch { expected: 2, .. })
        ));
    }

    /// Compare a kernel's output against the reference with an error bound that
    /// scales with the accumulation length, since float error grows with `k`.
    fn assert_close(got: &Tensor, want: &Tensor, context: &str) {
        assert_eq!(got.shape(), want.shape(), "{context}: shape");
        for (g, w) in got.data().iter().zip(want.data()) {
            let tolerance = 1e-4 * (1.0 + w.abs());
            assert!((g - w).abs() <= tolerance, "{context}: {g} vs {w}");
        }
    }

    /// The blocked kernel must agree with the reference on shapes that exercise
    /// full tiles, the right edge, the bottom edge, both at once, and the
    /// degenerate cases where a dimension is zero.
    #[test]
    fn matmul_tiled_matches_reference() {
        let shapes = [
            (1, 1, 1),
            (3, 5, 7),
            (7, 9, 5),
            (8, 4, 8),
            (8, 8, 8),
            (16, 16, 16),
            (17, 19, 23),
            (33, 17, 9),
            (64, 64, 64),
            (0, 4, 0),
            (5, 0, 6),
            (0, 0, 0),
        ];
        for (m, k, n) in shapes {
            let a = random(&[m, k], 1);
            let b = random(&[k, n], 2);
            let reference = matmul(&a, &b).unwrap();
            let tiled = matmul_tiled(&a, &b).unwrap();
            assert_close(&tiled, &reference, &format!("{m}x{k}x{n}"));
        }
    }

    /// `matmul_packed_nt(a, w)` is `a * w^T`, checked against the plain
    /// reference with an explicit transpose.
    #[test]
    fn matmul_packed_nt_matches_reference() {
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (3, 5, 7),
            (8, 8, 8),
            (17, 19, 23),
            (33, 17, 9),
            (64, 64, 64),
            (65, 257, 513),
            (128, 576, 1536),
        ] {
            let a = random(&[m, k], 21);
            let w = random(&[n, k], 22);
            let reference = matmul(&a, &transpose(&w).unwrap()).unwrap();
            let got = matmul_packed_nt(&a, &w).unwrap();
            assert_eq!(got.shape(), reference.shape());
            assert_close(&got, &reference, &format!("{m}x{k}x{n}"));
        }
    }

    /// The slice-based core must agree with the `Tensor` wrapper it backs.
    ///
    /// `out` arrives deliberately dirty: the micro-kernel accumulates into it
    /// across `k` blocks, so a missing zeroing would show up as a large error
    /// here rather than a rounding-sized one.
    #[test]
    fn gemm_nt_into_matches_matmul_packed_nt() {
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (3, 5, 7),
            (8, 8, 8),
            (17, 19, 23),
            (33, 17, 9),
            (64, 64, 64),
            (65, 257, 513),
            (128, 576, 1536),
        ] {
            let a = random(&[m, k], 21);
            let w = random(&[n, k], 22);
            let want = matmul_packed_nt(&a, &w).unwrap();
            let mut out = vec![7.5f32; m * n];
            gemm_nt_into(a.data(), w.data(), &mut out, m, k, n);
            let got = Tensor::from_vec(out, &[m, n]).unwrap();
            assert_close(&got, &want, &format!("gemm_nt_into {m}x{k}x{n}"));
        }
    }

    /// The prefill path reuses one output buffer across layers, so a second
    /// call must not accumulate the first one's results.
    #[test]
    fn gemm_nt_into_overwrites_previous_results() {
        let a = random(&[16, 32], 3);
        let w = random(&[24, 32], 4);
        let want = matmul_packed_nt(&a, &w).unwrap();

        let mut out = vec![0.0f32; 16 * 24];
        for _ in 0..3 {
            gemm_nt_into(a.data(), w.data(), &mut out, 16, 32, 24);
        }
        let got = Tensor::from_vec(out, &[16, 24]).unwrap();
        assert_close(&got, &want, "repeated calls");
    }

    /// The plain NoTrans GEMM must agree with the reference, on the shapes the
    /// convolution path uses as well as the degenerate ones.
    #[test]
    fn gemm_nn_into_matches_matmul() {
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (3, 5, 7),
            (8, 8, 8),
            (17, 19, 23),
            (64, 64, 64),
            (65, 257, 513),
            (6400, 576, 64),
            (80, 27, 16),
        ] {
            let a = random(&[m, k], 31);
            let b = random(&[k, n], 32);
            let want = matmul(&a, &b).unwrap();
            let mut out = vec![7.5f32; m * n];
            gemm_nn_into(a.data(), b.data(), &mut out, n, m, k, n);
            let got = Tensor::from_vec(out, &[m, n]).unwrap();
            assert_close(&got, &want, &format!("gemm_nn_into {m}x{k}x{n}"));
        }
    }

    /// The rate the convolution path actually gets out of the NoTrans GEMM,
    /// on the shapes it issues rather than on a square test matrix: `a` is the
    /// weights `[filters, reduction]`, `b` is the im2col panel
    /// `[reduction, columns]`, and the result lands at `out_ld = output_h *
    /// output_w` so only one row band of it is live. Prints because the rate
    /// is the point; the floor is loose so a loaded machine does not turn a
    /// measurement into a failure.
    // A measurement, not a correctness check: it prints rates, and a loaded
    // machine turns its floor into a spurious failure. Run it on demand with
    // `cargo test --release gemm_nn_into_reaches -- --ignored --nocapture`.
    #[test]
    #[ignore = "rate measurement; needs an idle machine"]
    fn gemm_nn_into_reaches_a_usable_rate_on_conv_shapes() {
        // (filters, reduction, columns, output spatial) for one band of each
        // convolution class in yolo26n.
        for (m, k, n, ld) in [
            (16usize, 27usize, 19_200usize, 102_400usize),
            (32, 144, 12_800, 25_600),
            (64, 576, 6_400, 6_400),
            (128, 1_152, 1_600, 1_600),
            (16, 16, 102_400, 102_400),
            (256, 256, 6_400, 6_400),
            (128, 128, 1_600, 1_600),
        ] {
            let a = random(&[m, k], 41);
            let b = random(&[k, n], 42);
            let mut out = vec![0.0f32; m * ld];
            gemm_nn_into(a.data(), b.data(), &mut out, ld, m, k, n);
            let reps = (2_000_000_000usize / (m * k * n)).clamp(10, 500);
            let start = std::time::Instant::now();
            for _ in 0..reps {
                gemm_nn_into(a.data(), b.data(), &mut out, ld, m, k, n);
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            let gflops = 2.0 * (m * k * n) as f64 / secs / 1e9;
            println!("gemm {m:4}x{k:5}x{n:7} ld={ld:7}: {gflops:6.0} GFLOPS");
            // Loose: this is a measurement with a smoke-test floor, not a
            // target. The numbers are read from the output.
            assert!(gflops > 50.0, "{m}x{k}x{n} only reached {gflops:.0} GFLOPS");
        }
    }

    /// Like the `nt` variant, a dirty output buffer must not leak between calls.
    #[test]
    fn gemm_nn_into_overwrites_previous_results() {
        let a = random(&[16, 32], 5);
        let b = random(&[32, 24], 6);
        let want = matmul(&a, &b).unwrap();

        let mut out = vec![0.0f32; 16 * 24];
        for _ in 0..3 {
            gemm_nn_into(a.data(), b.data(), &mut out, 24, 16, 32, 24);
        }
        let got = Tensor::from_vec(out, &[16, 24]).unwrap();
        assert_close(&got, &want, "repeated calls");
    }

    /// The strided form writes one column range of a wider matrix, which is
    /// how the convolution tiles its output. It must match a single multiply
    /// over the whole matrix and leave the guard columns alone.
    #[test]
    fn gemm_nn_into_lands_in_the_right_column_range() {
        let (m, k, n) = (17usize, 23usize, 65usize);
        let a = random(&[m, k], 33);
        let b = random(&[k, n], 34);
        let want = matmul(&a, &b).unwrap();
        let want = want.data();

        let ld = n + 5;
        let guard = 9.25f32;
        let mut wide = vec![guard; m * ld];
        // The submatrix starts at column 1 of every row, so both the leading
        // guard column and the trailing ones have to come back untouched.
        gemm_nn_into(a.data(), b.data(), &mut wide[1..], ld, m, k, n);
        for i in 0..m {
            assert_eq!(wide[i * ld], guard, "leading guard column, row {i}");
            for j in n + 1..ld {
                assert_eq!(wide[i * ld + j], guard, "trailing guard {j}, row {i}");
            }
            for j in 0..n {
                let got = wide[i * ld + 1 + j];
                let expected = want[i * n + j];
                assert!(
                    (got - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                    "row {i} col {j}: got {got}, want {expected}"
                );
            }
        }
    }

    /// The packed GEMM must agree with the reference on the same shape grid,
    /// including the panel remainders (odd `m`, `k` and `n` all land in the
    /// pack-tail paths).
    #[test]
    fn matmul_packed_matches_reference() {
        let shapes = [
            (1, 1, 1),
            (3, 5, 7),
            (7, 9, 5),
            (8, 4, 8),
            (8, 8, 8),
            (16, 16, 16),
            (17, 19, 23),
            (33, 17, 9),
            (64, 64, 64),
            (65, 257, 513),
            (128, 576, 1536),
            (0, 4, 0),
            (5, 0, 6),
            (0, 0, 0),
        ];
        for (m, k, n) in shapes {
            let a = random(&[m, k], 11);
            let b = random(&[k, n], 12);
            let reference = matmul(&a, &b).unwrap();
            let packed = matmul_packed(&a, &b).unwrap();
            assert_close(&packed, &reference, &format!("{m}x{k}x{n}"));
        }
    }

    /// Same edge coverage for the NEON path. This is what exercises the
    /// `unsafe` loads/stores: if the tile preconditions were wrong for a
    /// degenerate shape, these cases would read or write out of bounds.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmul_neon_matches_reference() {
        let shapes = [
            (1, 1, 1),
            (3, 5, 7),
            (7, 9, 5),
            (8, 4, 8),
            (8, 8, 8),
            (16, 16, 16),
            (17, 19, 23),
            (33, 17, 9),
            (64, 64, 64),
            (0, 4, 0),
            (5, 0, 6),
            (0, 0, 0),
        ];
        for (m, k, n) in shapes {
            let a = random(&[m, k], 1);
            let b = random(&[k, n], 2);
            let reference = matmul(&a, &b).unwrap();
            let neon = matmul_neon(&a, &b).unwrap();
            assert_close(&neon, &reference, &format!("{m}x{k}x{n}"));
        }
    }

    #[test]
    fn matmul_tiled_rejects_bad_shapes() {
        assert!(matmul_tiled(&Tensor::zeros(&[2, 3]), &Tensor::zeros(&[2, 2])).is_err());
        assert!(matmul_tiled(&Tensor::zeros(&[2, 2, 2]), &Tensor::zeros(&[2, 2])).is_err());
    }

    #[test]
    fn relu_clamps_negatives_and_keeps_shape() {
        let x = t(vec![-2., -1., 0., 1., 2.], &[5]);
        assert_eq!(relu(&x).data(), &[0., 0., 0., 1., 2.]);

        let x = t(vec![-1., 2., -3., 4.], &[2, 2]);
        assert_eq!(relu(&x).shape(), &[2, 2]);
        assert_eq!(relu(&x).data(), &[0., 2., 0., 4.]);
    }

    #[test]
    fn add_bias_broadcasts_across_rows() {
        let x = t(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let bias = t(vec![10., 20., 30.], &[3]);
        let out = add_bias(&x, &bias).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.data(), &[11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn add_bias_rejects_bad_shapes() {
        let x = t(vec![1., 2., 3., 4.], &[2, 2]);
        // Bias longer/shorter than the row.
        assert!(add_bias(&x, &t(vec![1., 2., 3.], &[3])).is_err());
        // Bias that is not rank 1.
        assert!(add_bias(&x, &Tensor::zeros(&[2, 2])).is_err());
        // Input that is not rank 2.
        assert!(add_bias(&Tensor::zeros(&[4]), &Tensor::zeros(&[4])).is_err());
    }

    #[test]
    fn softmax_matches_hand_computed_values() {
        let got = softmax(&t(vec![1., 2., 3.], &[1, 3]));
        let want = [0.09003057, 0.24472847, 0.66524094];
        for (g, w) in got.data().iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[test]
    fn softmax_rows_sum_to_one_and_survive_large_inputs() {
        // Without the max subtraction, `exp(1000)` is `inf` and the row is NaN.
        let got = softmax(&t(vec![1000., 1000., -1000., 0.], &[2, 2]));
        assert_eq!(got.shape(), &[2, 2]);
        assert_eq!(got.data(), &[0.5, 0.5, 0.0, 1.0]);
        for row in got.data().chunks(2) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row sums to {sum}");
        }
    }

    #[test]
    fn matvec_matches_hand_computed() {
        let m = t(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let y = matvec(&m, &[1., 1.]).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        assert_eq!(y.data(), &[5., 7., 9.]);
    }

    /// The loaded block path and the scalar leftover path only divide cleanly
    /// for some `cols`; cover multiples of 16, remainders, and narrow matrices.
    #[test]
    fn matvec_matches_matmul_for_many_shapes() {
        let shapes = [
            (1usize, 1usize),
            (2, 3),
            (8, 16),
            (16, 16),
            (17, 10),
            (5, 32),
            (3, 1),
            (128, 10),
            (784, 128),
        ];
        for (rows, cols) in shapes {
            let m = random(&[rows, cols], 3);
            let x = random(&[1, rows], 4);
            let reference = matmul(&x, &m).unwrap();
            let got = matvec(&m, x.data()).unwrap();
            assert_close(&got, &reference, &format!("matvec {rows}x{cols}"));
        }
    }

    #[test]
    fn matvec_rejects_bad_shapes() {
        let m = t(vec![1., 2., 3., 4.], &[2, 2]);
        assert!(matvec(&m, &[1.0]).is_err());
        assert!(matvec(&Tensor::zeros(&[2, 2, 2]), &[1., 2.]).is_err());
    }

    #[test]
    fn transpose_swaps_axes() {
        let y = transpose(&t(vec![1., 2., 3., 4., 5., 6.], &[2, 3])).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.data(), &[1., 4., 2., 5., 3., 6.]);
        assert!(transpose(&Tensor::zeros(&[4])).is_err());
    }

    #[test]
    fn gemv_matches_hand_computed() {
        let w = Matrix::from_f32(t(vec![1., 2., 3., 4., 5., 6.], &[2, 3]));
        let y = gemv(&w, &[1., 1., 1.]).unwrap();
        assert_eq!(y.shape(), &[1, 2]);
        assert_eq!(y.data(), &[6., 15.]);
    }

    /// `gemv(w[out, in], x[in])` must equal `matmul(w, x[in, 1])`, up to the
    /// different summation order (`gemv` uses four accumulators).
    #[test]
    fn gemv_matches_matmul_for_many_shapes() {
        for &(out_dim, in_dim) in &[(1usize, 1usize), (2, 3), (5, 4), (288, 288), (768, 288)] {
            let w = random(&[out_dim, in_dim], 3);
            let x = random(&[in_dim, 1], 4);
            let reference = matmul(&w, &x).unwrap();
            let got = gemv(&Matrix::from_f32(w), x.data()).unwrap();
            assert_eq!(got.data().len(), reference.data().len());
            for (g, r) in got.data().iter().zip(reference.data()) {
                let tolerance = 1e-4 * (1.0 + r.abs());
                assert!(
                    (g - r).abs() <= tolerance,
                    "gemv {out_dim}x{in_dim}: {g} vs {r}"
                );
            }
        }
    }

    #[test]
    fn gemv_rejects_bad_shapes() {
        let w = Matrix::from_f32(Tensor::zeros(&[2, 3]));
        assert!(gemv(&w, &[1.0, 2.0]).is_err());
    }

    /// A bf16 matrix must produce the same result as the equivalent f32 one for
    /// values bf16 represents exactly.
    #[test]
    fn bf16_gemv_matches_f32() {
        let (out_dim, in_dim) = (5usize, 32usize);
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i % 17) as f32 / 4.0 - 2.0)
            .collect();
        let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let x = random(&[in_dim], 9);

        let f32_result = gemv(
            &Matrix::from_f32(t(values.clone(), &[out_dim, in_dim])),
            x.data(),
        )
        .unwrap();
        let bf16_result = gemv(&Matrix::from_bf16_bits(bits, out_dim, in_dim), x.data()).unwrap();
        for (a, b) in f32_result.data().iter().zip(bf16_result.data()) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }

    #[test]
    fn bf16_matrix_reads_rows() {
        let bits: Vec<u16> = [1.0f32, -2.0, 0.5, 4.0]
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let matrix = Matrix::from_bf16_bits(bits, 2, 2);
        assert_eq!(matrix.out_dim(), 2);
        assert_eq!(matrix.in_dim(), 2);
        assert_eq!(matrix.row(1), vec![0.5, 4.0]);
    }

    #[test]
    fn q8_quantization_error_is_half_a_step() {
        for seed in 0..8 {
            let x = random(&[96], 100 + seed);
            let q = Q8Vec::quantize(x.data());
            for (block, &scale) in q.scales.iter().enumerate() {
                for lane in 0..Q8_0_BLOCK {
                    let original = x.data()[block * Q8_0_BLOCK + lane];
                    let restored = f32::from(q.quants[block * Q8_0_BLOCK + lane]) * scale;
                    assert!(
                        (restored - original).abs() <= scale / 2.0 + 1e-6,
                        "seed {seed} block {block} lane {lane}: {restored} vs {original}"
                    );
                }
            }
        }
    }

    /// Values that are exact multiples of their block's scale must survive the
    /// round trip unchanged.
    #[test]
    fn q8_is_exact_on_step_multiples() {
        // A block scale of 1/128 is exactly representable, and so is every
        // multiple of it, so both the divide and the multiply back are lossless.
        // Each block leads with ±127 steps, which fixes its scale.
        let step = 1.0f32 / 128.0;
        let mut values = Vec::with_capacity(64);
        for _ in 0..2 {
            for lane in 0..Q8_0_BLOCK {
                let k = if lane == 0 { 127.0 } else { lane as f32 - 16.0 };
                values.push(k * step);
            }
        }
        let q = Q8Vec::quantize(&values);
        for (block, &scale) in q.scales.iter().enumerate() {
            assert_eq!(scale, step);
            for lane in 0..Q8_0_BLOCK {
                let restored = f32::from(q.quants[block * Q8_0_BLOCK + lane]) * scale;
                assert_eq!(restored, values[block * Q8_0_BLOCK + lane]);
            }
        }
    }

    /// The SDOT kernel does the same exact integer arithmetic as the scalar one;
    /// only the order in which the block results are summed in f32 differs.
    #[test]
    fn q8_dot_neon_matches_scalar() {
        let a = random(&[512], 200);
        let b = random(&[512], 201);
        let qa = Q8Vec::quantize(a.data());
        let qb = Q8Vec::quantize(b.data());
        let fast = dot_q8_0(&qa.quants, &qa.scales, &qb.quants, &qb.scales);
        let scalar = dot_q8_0_scalar(&qa.quants, &qa.scales, &qb.quants, &qb.scales);
        assert!(
            (fast - scalar).abs() <= 1e-5 * (1.0 + scalar.abs()),
            "{fast} vs {scalar}"
        );
    }

    #[test]
    fn q8_dot_tracks_f32_dot() {
        let n = 512;
        // Positive, correlated operands keep the dot product far from zero so a
        // relative error bound is meaningful.
        let a: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 + 1.0) * 0.2).collect();
        let want = dot(&a, &b);
        let qa = Q8Vec::quantize(&a);
        let qb = Q8Vec::quantize(&b);
        let got = dot_q8_0(&qa.quants, &qa.scales, &qb.quants, &qb.scales);
        assert!((got - want).abs() <= 0.03 * want.abs(), "{got} vs {want}");
    }

    #[test]
    fn q8_gemv_tracks_f32_gemv() {
        let (out_dim, in_dim) = (16usize, 128usize);
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 11) as f32 + 1.0) * 0.05)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 7) as f32 + 1.0) * 0.1).collect();
        let reference = gemv(&Matrix::from_f32(t(w.clone(), &[out_dim, in_dim])), &x).unwrap();
        let quantized = gemv(
            &Matrix::from_f32_q8_0(t(w, &[out_dim, in_dim])).unwrap(),
            &x,
        )
        .unwrap();
        for (q, r) in quantized.data().iter().zip(reference.data()) {
            assert!((q - r).abs() <= 0.05 * r.abs(), "{q} vs {r}");
        }
    }

    #[test]
    fn q8_requires_block_aligned_rows() {
        assert!(Matrix::from_f32_q8_0(t(vec![0.0; 10], &[2, 5])).is_none());
        assert!(Matrix::from_f32_q8_0(t(vec![0.0; 64], &[2, 32])).is_some());
    }

    #[test]
    fn q8_matrix_reads_rows() {
        let matrix = Matrix::from_f32_q8_0(t(vec![0.0; 128], &[2, 64])).unwrap();
        assert_eq!(matrix.out_dim(), 2);
        assert_eq!(matrix.in_dim(), 64);
        assert_eq!(matrix.row(0).len(), 64);
        assert!(matrix.row(1).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8x4_repacked_rows_match_plain_q8_rows() {
        for &(out_dim, in_dim) in &[(4usize, 32usize), (8, 64), (16, 128), (12, 96)] {
            let w = random(&[out_dim, in_dim], 711);
            let x = random(&[in_dim], 712);
            let plain = gemv(&Matrix::from_f32_q8_0(w.clone()).unwrap(), x.data()).unwrap();
            let packed = gemv(&Matrix::from_f32_q8_0x4(w).unwrap(), x.data()).unwrap();
            for (row, (a, b)) in packed.data().iter().zip(plain.data()).enumerate() {
                assert!(
                    (a - b).abs() <= 5e-4 * (1.0 + b.abs()),
                    "row {row}: q8x4 {a} vs row-wise {b}"
                );
            }
        }
    }

    #[test]
    fn q8x4_rejects_unaligned_shapes() {
        assert!(Matrix::from_f32_q8_0x4(t(vec![0.0; 6 * 32], &[6, 32])).is_none());
        assert!(Matrix::from_f32_q8_0x4(t(vec![0.0; 8 * 30], &[8, 30])).is_none());
        assert!(Matrix::from_f32_q8_0x4(t(vec![0.0; 8 * 32], &[8, 32])).is_some());
    }

    #[test]
    fn q8x4_decodes_each_row_to_the_plain_quantization() {
        let (out_dim, in_dim) = (8usize, 64usize);
        let data: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 17 % 251) as f32 - 125.0) / 137.0)
            .collect();
        let packed = Matrix::from_f32_q8_0x4(t(data.clone(), &[out_dim, in_dim])).unwrap();
        let plain = Matrix::from_f32_q8_0(t(data, &[out_dim, in_dim])).unwrap();
        for row in 0..out_dim {
            let a = packed.row(row);
            let b = plain.row(row);
            for (lane, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (x - y).abs() <= 5e-4 * (1.0 + y.abs()),
                    "row {row} lane {lane}: {x} vs {y}"
                );
            }
        }
    }

    /// The NEON kernel and the scalar one must agree, including on lengths that
    /// leave a vector-loop remainder.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_neon_matches_scalar() {
        for &n in &[1usize, 3, 4, 7, 15, 16, 17, 288, 767, 1024] {
            let a = random(&[n], 5);
            let b = random(&[n], 6);
            let got = dot(a.data(), b.data());
            let want = dot_scalar(a.data(), b.data());
            let tolerance = 1e-4 * (1.0 + want.abs());
            assert!((got - want).abs() <= tolerance, "n={n}: {got} vs {want}");
        }
    }
}
