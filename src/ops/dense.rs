use crate::tensor::{Tensor, TensorError};

/// Register-tile size of the blocked micro-kernel: `MR` output rows by `NR`
/// output columns are held in CPU registers as accumulators for the whole
/// reduction, then written to memory once.
///
/// 8x8 = 64 `f32` accumulators = 16 of the 32 NEON registers, leaving room for
/// the loaded `rhs` vectors and address temporaries. Both numbers came from
/// measurement rather than taste:
///
/// * `NR: 4 -> 8` was worth roughly 1.7x — each `a` broadcast is reused for 8
///   columns instead of 4, so loads per flop nearly halve.
/// * `MR: 8 -> 12` did not help: multiply-add throughput, not loads, is already
///   the limit — and small matrices got slower, consistent with the accumulators
///   plus temporaries beginning to spill out of the register file.
const MR: usize = 8;
const NR: usize = 8;

/// Validate that `a` and `b` are rank-2 and inner-compatible, returning
/// `(m, k, n)` for `[m, k] x [k, n] -> [m, n]`.
fn matmul_dims(a: &Tensor, b: &Tensor) -> Result<(usize, usize, usize), TensorError> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() != 2 || bsh.len() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            got: ash.len().max(bsh.len()),
        });
    }
    let (m, k) = (ash[0], ash[1]);
    let (k_b, n) = (bsh[0], bsh[1]);
    if k != k_b {
        return Err(TensorError::ShapeMismatch {
            op: "matmul",
            a: ash.to_vec(),
            b: bsh.to_vec(),
        });
    }
    Ok((m, k, n))
}

/// Scalar `i-k-j` kernel: accumulates `lhs * rhs` into the first `rows` rows of
/// `out`, starting at output column `j_start`. `lhs` and `out` are assumed to be
/// pre-offset to the first row of interest, so a caller can run this over a
/// sub-range of rows by slicing.
///
/// Loop order is `i-k-j` rather than the textbook `i-j-k`. Both compute the same
/// result, but `i-k-j` walks the inner `j` loop over two contiguous rows
/// (`out[i, :]` and `rhs[p, :]`), which the cache and auto-vectorizer exploit.
/// `i-j-k` strides down a column of `rhs` on every step, thrashing the cache.
fn scalar_kernel(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    n: usize,
    k: usize,
    j_start: usize,
) {
    for i in 0..rows {
        let out_row = &mut out[i * n..i * n + n];
        for p in 0..k {
            let a_ip = lhs[i * k + p];
            let b_row = &rhs[p * n..p * n + n];
            for j in j_start..n {
                out_row[j] += a_ip * b_row[j];
            }
        }
    }
}

/// Matrix multiply of two rank-2 tensors: `[m, k] x [k, n] -> [m, n]`.
///
/// This is the scalar *reference* implementation. It is simultaneously the
/// correctness oracle (optimized kernels are checked against it) and the
/// performance baseline that optimizations are measured against.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k, n) = matmul_dims(a, b)?;
    let mut out = vec![0.0f32; m * n];
    scalar_kernel(a.data(), b.data(), &mut out, m, n, k, 0);
    Tensor::from_vec(out, &[m, n])
}

/// Register-blocked matrix multiply: same result as [`matmul`], fewer memory
/// operations.
///
/// The output is partitioned into `MR x NR` tiles. For one tile, the `k`
/// reduction keeps `MR * NR` running totals in local variables (registers). Each
/// `rhs` value loaded is reused `MR` times, each `lhs` value `NR` times, and the
/// output tile is read/written once instead of once per element per step.
///
/// Edge tiles that do not fill `MR x NR` are handled by the scalar kernel.
pub fn matmul_tiled(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k, n) = matmul_dims(a, b)?;
    let lhs = a.data();
    let rhs = b.data();
    let mut out = vec![0.0f32; m * n];

    // Largest row/column boundaries covered entirely by full tiles.
    let i_full = m - m % MR;
    let j_full = n - n % NR;

    for i0 in (0..i_full).step_by(MR) {
        for j0 in (0..j_full).step_by(NR) {
            micro_kernel(lhs, rhs, &mut out, n, k, i0, j0);
        }
    }

    // Right edge: full-tile rows, leftover columns.
    if j_full < n {
        scalar_kernel(lhs, rhs, &mut out, i_full, n, k, j_full);
    }
    // Bottom edge: leftover rows, all columns. Rows are contiguous in both
    // `lhs` and `out`, so we can offset by slicing instead of copying.
    if i_full < m {
        scalar_kernel(
            &lhs[i_full * k..],
            rhs,
            &mut out[i_full * n..],
            m - i_full,
            n,
            k,
            0,
        );
    }

    Tensor::from_vec(out, &[m, n])
}

/// Panel widths for the packed GEMM. `MC x KC` of `A` and `KC x NC` of `B` are
/// packed into contiguous buffers so the innermost loop always walks memory in
/// order; the sizes are chosen so a packed panel pair sits inside L1/L2.
const GEMM_MC: usize = 128;
const GEMM_KC: usize = 768;
const GEMM_NC: usize = 768;

/// Cache-blocked, packed GEMM: `[m, k] x [k, n] -> [m, n]`.
///
/// [`matmul_tiled`] tiles the output but reads both operands straight from the
/// originals. That works while a row fits in cache and collapses when it does
/// not: at `n = 49152` each `rhs` row is 192 KB, so the `k` loop strides far
/// outside L2 every step and throughput fell from 65 to 17 GFLOP/s in the
/// benchmark. Prefill produces exactly those tall shapes.
///
/// This version fixes it the way BLAS does — pack and block:
///
/// 1. `A` is copied into `MC x KC` panels, `B` into `KC x NC` panels. Packing
///    makes each panel contiguous, so every inner-loop access is sequential and
///    a cache line fetched is a cache line used.
/// 2. The panels are walked so the `MR x NR` micro-kernel sees a `k` range of
///    at most `KC` and a `B` panel of at most `KC x NR` — a few kilobytes.
///
/// The result is identical to [`matmul_tiled`] up to floating-point summation
/// order, because both accumulate over `k` in a different grouping.
pub fn matmul_packed(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k, n) = matmul_dims(a, b)?;
    let lhs = a.data();
    let rhs = b.data();
    let mut out = vec![0.0f32; m * n];

    // Edge columns and rows keep the scalar path; prefill shapes are usually
    // multiples of the tile sizes, and correctness does not depend on it.
    let i_full = m - m % MR;
    let j_full = n - n % NR;

    if i_full == 0 || j_full == 0 {
        scalar_kernel(lhs, rhs, &mut out, m, n, k, 0);
        return Tensor::from_vec(out, &[m, n]);
    }

    let mut a_panel = vec![0.0f32; GEMM_MC * GEMM_KC];
    let mut b_panel = vec![0.0f32; GEMM_KC * GEMM_NC];

    for i0 in (0..i_full).step_by(GEMM_MC) {
        let mc = GEMM_MC.min(i_full - i0);
        for j0 in (0..j_full).step_by(GEMM_NC) {
            let nc = GEMM_NC.min(j_full - j0);
            for p0 in (0..k).step_by(GEMM_KC) {
                let kc = GEMM_KC.min(k - p0);
                // Pack `A[i0..i0+mc, p0..p0+kc]` row-major.
                for i in 0..mc {
                    let src = &lhs[(i0 + i) * k + p0..(i0 + i) * k + p0 + kc];
                    a_panel[i * kc..i * kc + kc].copy_from_slice(src);
                }
                // Pack `B[p0..p0+kc, j0..j0+nc]` row-major; its rows are the
                // strides that used to blow the cache, and here they become
                // contiguous.
                for p in 0..kc {
                    let src = &rhs[(p0 + p) * n + j0..(p0 + p) * n + j0 + nc];
                    b_panel[p * nc..p * nc + nc].copy_from_slice(src);
                }
                packed_micro(&a_panel, &b_panel, &mut out, n, mc, nc, kc, i0, j0, p0);
            }
        }
    }

    if j_full < n {
        scalar_kernel(lhs, rhs, &mut out, i_full, n, k, j_full);
    }
    if i_full < m {
        scalar_kernel(
            &lhs[i_full * k..],
            rhs,
            &mut out[i_full * n..],
            m - i_full,
            n,
            k,
            0,
        );
    }
    Tensor::from_vec(out, &[m, n])
}

/// Packed GEMM against a transposed right operand: `A[m, k] * W[n, k]^T`, giving
/// `[m, n]`.
///
/// This is the form a linear layer actually has: weights everywhere (llama2.c,
/// safetensors, GGUF) are stored `[out, in]`, so a batch of `m` activations
/// needs `X * W^T`. Materializing `W^T` would cost a full copy of the weights
/// (hundreds of megabytes), and it is not needed: only the packing step cares
/// about the operand layout, and it can gather a column of `W` instead of a row.
/// The transposed gather is exactly as cache-friendly once packed, because the
/// pack loop reads one `W` row per output column and writes it contiguously.
///
/// The result matches `matmul_packed(a, transpose(w))` up to summation order.
pub fn matmul_packed_nt(a: &Tensor, w: &Tensor) -> Result<Tensor, TensorError> {
    let ash = a.shape();
    let wsh = w.shape();
    if ash.len() != 2 || wsh.len() != 2 || ash[1] != wsh[1] {
        return Err(TensorError::ShapeMismatch {
            op: "matmul_packed_nt",
            a: ash.to_vec(),
            b: wsh.to_vec(),
        });
    }
    let (m, k) = (ash[0], ash[1]);
    let n = wsh[0];
    let mut out = vec![0.0f32; m * n];
    gemm_nt_into(a.data(), w.data(), &mut out, m, k, n);
    Tensor::from_vec(out, &[m, n])
}

/// Slice-based core of [`matmul_packed_nt`]: `lhs[m, k] * weights[n, k]^T`
/// written into `out[m, n]`.
///
/// Exists so a caller holding flat scratch buffers — the prefill path, which
/// keeps `[n_tok, width]` rows for a whole prompt — can run the same kernel
/// without moving those buffers in and out of a `Tensor` at every layer.
///
/// `out` is zeroed first, because the micro-kernel accumulates into it across
/// `k` blocks while the scalar edge cases assign; both need a known start.
///
/// # Panics
///
/// Panics on a debug build if the slice lengths disagree with the shapes.
pub fn gemm_nt_into(lhs: &[f32], weights: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(lhs.len(), m * k, "lhs must be m x k");
    debug_assert_eq!(weights.len(), n * k, "weights must be n x k");
    debug_assert_eq!(out.len(), m * n, "out must be m x n");
    out.fill(0.0);

    let i_full = m - m % MR;
    let j_full = n - n % NR;

    if i_full == 0 || j_full == 0 {
        // Scalar fallback against the transposed view, built row by row.
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc = lhs[i * k + p].mul_add(weights[j * k + p], acc);
                }
                out[i * n + j] = acc;
            }
        }
        return;
    }

    let mut a_panel = vec![0.0f32; GEMM_MC * GEMM_KC];
    let mut b_panel = vec![0.0f32; GEMM_KC * GEMM_NC];

    for i0 in (0..i_full).step_by(GEMM_MC) {
        let mc = GEMM_MC.min(i_full - i0);
        for j0 in (0..j_full).step_by(GEMM_NC) {
            let nc = GEMM_NC.min(j_full - j0);
            for p0 in (0..k).step_by(GEMM_KC) {
                let kc = GEMM_KC.min(k - p0);
                for i in 0..mc {
                    let src = &lhs[(i0 + i) * k + p0..(i0 + i) * k + p0 + kc];
                    a_panel[i * kc..i * kc + kc].copy_from_slice(src);
                }
                // Each output column is one row of `W`, so `b_panel[p * nc + j]`
                // must hold `W[j0 + j, p0 + p]`: a transpose during the pack.
                // Looping `p` outer keeps the *writes* sequential, which is the
                // half worth optimizing — the reads then come one row of `W` at
                // a time and stay cache-resident. Writing `j` outer scattered
                // across the whole panel and cost about half the throughput.
                for p in 0..kc {
                    let row = p0 + p;
                    for j in 0..nc {
                        b_panel[p * nc + j] = weights[(j0 + j) * k + row];
                    }
                }
                packed_micro(&a_panel, &b_panel, out, n, mc, nc, kc, i0, j0, p0);
            }
        }
    }

    if j_full < n || i_full < m {
        for i in i_full..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc = lhs[i * k + p].mul_add(weights[j * k + p], acc);
                }
                out[i * n + j] = acc;
            }
        }
        for i in 0..i_full {
            for j in j_full..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc = lhs[i * k + p].mul_add(weights[j * k + p], acc);
                }
                out[i * n + j] = acc;
            }
        }
    }
}

/// Micro-kernel over a panel pair where the `B` panel is stored so that panel
/// column `j` is contiguous in `k` (i.e. transposed relative to [`packed_micro`]).
///
/// Retained as the reference for that layout: `matmul_packed_nt` used to pack
/// into it, and it is what a column-major panel looks like unpacked. The live
/// path packs `k`-major instead, which measured about twice as fast, so nothing
/// calls this now.
#[allow(dead_code, clippy::too_many_arguments)]
fn packed_micro_transposed(
    a_panel: &[f32],
    b_panel: &[f32],
    out: &mut [f32],
    n: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    i0: usize,
    j0: usize,
) {
    let i_tiles = mc - mc % MR;
    let j_tiles = nc - nc % NR;
    for ii in (0..i_tiles).step_by(MR) {
        let a_rows: [&[f32]; MR] =
            std::array::from_fn(|r| &a_panel[(ii + r) * kc..(ii + r) * kc + kc]);
        for jj in (0..j_tiles).step_by(NR) {
            // `acc[i][j]` accumulates output row `i`, column `j`. The `B` panel
            // is stored column-major in `k`, so panel column `jj + j` holds
            // `W[jj + j, p0..p0+kc]` contiguously.
            let mut acc = [[0.0f32; NR]; MR];
            for p in 0..kc {
                for (i, row) in acc.iter_mut().enumerate() {
                    let a_ip = a_rows[i][p];
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell = a_ip.mul_add(b_panel[(jj + j) * kc + p], *cell);
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                let c = &mut out[(i0 + ii + i) * n + j0 + jj..(i0 + ii + i) * n + j0 + jj + NR];
                for (j, cell) in row.iter().enumerate() {
                    c[j] += cell;
                }
            }
        }
    }
    // Remainders, straight scalar over the packed panels.
    for ii in i_tiles..mc {
        let a_row = &a_panel[ii * kc..ii * kc + kc];
        for jj in 0..nc {
            let mut acc = 0.0f32;
            let b_col = &b_panel[jj * kc..jj * kc + kc];
            for p in 0..kc {
                acc = a_row[p].mul_add(b_col[p], acc);
            }
            out[(i0 + ii) * n + j0 + jj] += acc;
        }
    }
    for ii in 0..i_tiles {
        let a_row = &a_panel[ii * kc..ii * kc + kc];
        for jj in j_tiles..nc {
            let mut acc = 0.0f32;
            let b_col = &b_panel[jj * kc..jj * kc + kc];
            for p in 0..kc {
                acc = a_row[p].mul_add(b_col[p], acc);
            }
            out[(i0 + ii) * n + j0 + jj] += acc;
        }
    }
}

/// Run the `MR x NR` micro-kernel over one packed panel pair.
////// `a_panel` is `mc x kc` packed row-major and `b_panel` is `kc x nc` packed
/// row-major, so the `k` loop inside a tile reads two short contiguous runs
/// instead of striding across the full matrices.
#[allow(clippy::too_many_arguments)]
fn packed_micro(
    a_panel: &[f32],
    b_panel: &[f32],
    out: &mut [f32],
    n: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    i0: usize,
    j0: usize,
    _p0: usize,
) {
    let i_tiles = mc - mc % MR;
    let j_tiles = nc - nc % NR;
    for ii in (0..i_tiles).step_by(MR) {
        let a_rows: [&[f32]; MR] =
            std::array::from_fn(|r| &a_panel[(ii + r) * kc..(ii + r) * kc + kc]);
        for jj in (0..j_tiles).step_by(NR) {
            let mut acc = [[0.0f32; NR]; MR];
            for p in 0..kc {
                let b_row = &b_panel[p * nc + jj..p * nc + jj + NR];
                for (r, acc_row) in acc.iter_mut().enumerate() {
                    let a_rp = a_rows[r][p];
                    for (q, acc) in acc_row.iter_mut().enumerate() {
                        *acc = a_rp.mul_add(b_row[q], *acc);
                    }
                }
            }
            for (r, acc_row) in acc.iter().enumerate() {
                let c = &mut out[(i0 + ii + r) * n + j0 + jj..(i0 + ii + r) * n + j0 + jj + NR];
                for (q, acc) in acc_row.iter().enumerate() {
                    c[q] += acc;
                }
            }
        }
        // Column remainder within the panel.
        if j_tiles < nc {
            for (r, a_row) in a_rows.iter().enumerate() {
                let c_base = (i0 + ii + r) * n + j0;
                for q in j_tiles..nc {
                    let mut acc = 0.0f32;
                    for p in 0..kc {
                        acc = a_row[p].mul_add(b_panel[p * nc + q], acc);
                    }
                    out[c_base + q] += acc;
                }
            }
        }
    }
    // Row remainder within the panel, via the scalar kernel on the packed data.
    if i_tiles < mc {
        for i in i_tiles..mc {
            let a_row = &a_panel[i * kc..i * kc + kc];
            let c = &mut out[(i0 + i) * n + j0..(i0 + i) * n + j0 + nc];
            c.fill(0.0);
            for p in 0..kc {
                let ap = a_row[p];
                let brow = &b_panel[p * nc..p * nc + nc];
                for (q, cv) in c.iter_mut().enumerate() {
                    *cv += ap * brow[q];
                }
            }
        }
    }
}

/// Explicit NEON micro-kernel, compiled only for AArch64 targets.///
/// Identical algorithm to [`micro_kernel`] — same `MR x NR` tile, same hoisted
/// row slices, same fused multiply-add — but the arithmetic is written directly
/// as NEON intrinsics instead of asking LLVM's auto-vectorizer to reconstruct it
/// from a scalar-looking loop. Exposed so the two can be benchmarked against
/// each other.
#[cfg(target_arch = "aarch64")]
fn neon_micro_kernel(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    n: usize,
    k: usize,
    i0: usize,
    j0: usize,
) {
    use core::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32,
    };

    // A NEON register holds exactly 4 `f32`, so an `NR`-wide tile needs
    // `NR / 4` vectors per row. Enforced here rather than assumed, because a
    // wrong width would silently compute only part of every tile.
    const NR_VECTORS: usize = NR / 4;
    const _: () = assert!(NR % 4 == 0);

    // The preconditions the `unsafe` blocks below rely on, checked here so a
    // caller mistake surfaces as a panic in tests instead of as an
    // out-of-bounds access. They are derivable from slice lengths, so no extra
    // parameters are needed: `out.len() == m * n` bounds the stored rows,
    // `rhs.len() == k * n` together with `j0 + NR <= n` bounds the vector loads.
    debug_assert!(rhs.len() == k * n, "rhs must match the k x n shape");
    debug_assert!(j0 + NR <= n, "tile extends past the last output column");
    debug_assert!(
        (i0 + MR) * n <= out.len(),
        "tile extends past the last output row"
    );

    let a_rows: [&[f32]; MR] = std::array::from_fn(|ii| &lhs[(i0 + ii) * k..(i0 + ii) * k + k]);

    // SAFETY: NEON is architecturally mandatory on AArch64, which is exactly the
    // target this function is compiled for. `vdupq_n_f32` touches no memory, so
    // that is its entire obligation.
    let mut acc: [[float32x4_t; NR_VECTORS]; MR] = unsafe { [[vdupq_n_f32(0.0); NR_VECTORS]; MR] };

    for p in 0..k {
        // SAFETY: same NEON guarantee as above, plus — because `p < k`,
        // `j0 + NR <= n` and `rhs.len() == k * n` (all debug_asserted at the top)
        // — every 16-byte load at `rhs[p * n + j0 + c * 4]` ends at or before
        // `rhs[k * n]`, inside the slice. AArch64 permits unaligned loads from
        // ordinary memory, so `Vec<f32>`'s 4-byte alignment is enough.
        unsafe {
            // The tile's NR b-values, loaded once and reused across all MR rows.
            let b: [float32x4_t; NR_VECTORS] =
                core::array::from_fn(|c| vld1q_f32(rhs.as_ptr().add(p * n + j0 + c * 4)));
            for (acc_row, a_row) in acc.iter_mut().zip(a_rows) {
                let a = vdupq_n_f32(a_row[p]);
                for (acc_vec, b_vec) in acc_row.iter_mut().zip(&b) {
                    // `acc + b * a`, one fused instruction.
                    *acc_vec = vfmaq_f32(*acc_vec, *b_vec, a);
                }
            }
        }
    }

    for (ii, acc_row) in acc.iter().enumerate() {
        // SAFETY: `(i0 + MR) * n <= out.len()` (debug_asserted at the top) means
        // rows `i0 .. i0 + MR` exist in `out`, and `j0 + NR <= n` means each
        // 16-byte window fits inside its row — so every window lies within `out`.
        // `out` starts zeroed and each region is written by exactly one code
        // path (tile loop vs. edge loop), so the add is well-defined either way.
        for (c, acc_vec) in acc_row.iter().enumerate() {
            unsafe {
                let dst = out.as_mut_ptr().add((i0 + ii) * n + j0 + c * 4);
                let existing = vld1q_f32(dst);
                vst1q_f32(dst, vaddq_f32(existing, *acc_vec));
            }
        }
    }
}

/// Explicit-NEON matrix multiply. Same shape handling and edge fallback as
/// [`matmul_tiled`]; only the micro-kernel for full tiles differs.
///
/// Measured verdict at the current 8x8 tile: **tied** with [`matmul_tiled`]'s
/// auto-vectorized loop. Three clean back-to-back runs at 512 landed both at
/// 4.0-4.3 ms with the winner alternating inside the noise (4.19 vs 4.02,
/// 4.05 vs 4.28, 4.08 vs 4.13). The intrinsic loop is structurally smaller —
/// zero bounds checks per `k`-step where the emitted assembly shows three
/// compare+branch pairs — and that was worth 8-17% at the earlier 4-column
/// tile, where each check was amortized over half as much arithmetic. At the
/// wider tile the difference drops below measurement resolution. Kept as the
/// AArch64 reference point for future kernel work, not as a winner today.
#[cfg(target_arch = "aarch64")]
pub fn matmul_neon(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k, n) = matmul_dims(a, b)?;
    let lhs = a.data();
    let rhs = b.data();
    let mut out = vec![0.0f32; m * n];

    let i_full = m - m % MR;
    let j_full = n - n % NR;

    for i0 in (0..i_full).step_by(MR) {
        for j0 in (0..j_full).step_by(NR) {
            neon_micro_kernel(lhs, rhs, &mut out, n, k, i0, j0);
        }
    }

    if j_full < n {
        scalar_kernel(lhs, rhs, &mut out, i_full, n, k, j_full);
    }
    if i_full < m {
        scalar_kernel(
            &lhs[i_full * k..],
            rhs,
            &mut out[i_full * n..],
            m - i_full,
            n,
            k,
            0,
        );
    }

    Tensor::from_vec(out, &[m, n])
}

/// Compute one `MR x NR` output tile, adding it into `out`.
///
/// The caller guarantees `i0 + MR <= m` and `j0 + NR <= n`, so every access
/// below is in bounds without checks. `acc` lives on the stack but the compiler
/// promotes these fixed-size, constant-indexed loops into registers.
fn micro_kernel(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    n: usize,
    k: usize,
    i0: usize,
    j0: usize,
) {
    // Hoist each of the MR `A` rows into its own slice *once*. Inside the
    // reduction we index `a_row[p]` with `p < k`, and `a_row.len() == k` is
    // statically implied by how the slice was built, so the compiler proves the
    // access in-bounds and elides the per-element bounds check. Indexing
    // `lhs[(i0 + ii) * k + p]` directly leaves it unable to prove that, and the
    // resulting checks dominated the kernel.
    let a_rows: [&[f32]; MR] = std::array::from_fn(|ii| &lhs[(i0 + ii) * k..(i0 + ii) * k + k]);

    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..k {
        let b_row = &rhs[p * n + j0..p * n + j0 + NR];
        for (ii, acc_row) in acc.iter_mut().enumerate() {
            let a_ip = a_rows[ii][p];
            for (jj, acc) in acc_row.iter_mut().enumerate() {
                *acc = a_ip.mul_add(b_row[jj], *acc);
            }
        }
    }
    for (ii, acc_row) in acc.iter().enumerate() {
        let c_row = &mut out[(i0 + ii) * n + j0..(i0 + ii) * n + j0 + NR];
        for (jj, acc) in acc_row.iter().enumerate() {
            c_row[jj] += acc;
        }
    }
}

/// Multiply a matrix by a single vector: `mat[m, n] * x[m] -> [1, n]`.
///
/// This is the batch-1 case of a linear layer, where the tiled matmul cannot
/// help: with a single output row `m < MR`, every tile is an edge tile and the
/// whole product falls back to the scalar kernel. Here the output row is short
/// enough to keep in CPU registers for the entire reduction, so `mat` is
/// streamed exactly once and the accumulator is never re-read from memory.
///
/// Returns a rank-2 `[1, n]` so a caller composing layers does not have to
/// special-case the shape of a single-sample batch.
pub fn matvec(mat: &Tensor, x: &[f32]) -> Result<Tensor, TensorError> {
    if mat.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            got: mat.rank(),
        });
    }
    let rows = mat.shape()[0];
    let cols = mat.shape()[1];
    if x.len() != rows {
        return Err(TensorError::ShapeMismatch {
            op: "matvec",
            a: mat.shape().to_vec(),
            b: vec![x.len()],
        });
    }

    let mat = mat.data();
    let mut out = vec![0.0f32; cols];

    // Accumulate the output in register blocks, largest first, so the common
    // width (128 here) is covered by wide blocks and any leftover is still
    // vectorized down to four columns. 64 columns is sixteen NEON vectors of
    // independent accumulators — enough dependency chains to keep the
    // multiply-add units busy, and enough contiguous weights per row to stream
    // them efficiently.
    const BLOCK: usize = 64;
    let mut j0 = 0;
    while j0 + BLOCK <= cols {
        matvec_block::<BLOCK>(mat, x, &mut out, cols, j0);
        j0 += BLOCK;
    }
    while j0 + 16 <= cols {
        matvec_block::<16>(mat, x, &mut out, cols, j0);
        j0 += 16;
    }
    while j0 + 4 <= cols {
        matvec_block::<4>(mat, x, &mut out, cols, j0);
        j0 += 4;
    }
    // Fewer than four columns left: scalar.
    while j0 < cols {
        let mut acc = 0.0f32;
        for (p, &x_p) in x.iter().enumerate() {
            acc = x_p.mul_add(mat[p * cols + j0], acc);
        }
        out[j0] = acc;
        j0 += 1;
    }

    Tensor::from_vec(out, &[1, cols])
}

/// One register-resident column block of [`matvec`]: accumulate `N` output
/// columns across the whole reduction, then store once.
///
/// A separate function so the block width stays a compile-time constant — the
/// accumulator array only lives in registers if its size is known — while
/// [`matvec`] can still cover an arbitrary number of columns by trying smaller
/// blocks.
fn matvec_block<const N: usize>(mat: &[f32], x: &[f32], out: &mut [f32], cols: usize, j0: usize) {
    let mut acc = [0.0f32; N];
    for (p, &x_p) in x.iter().enumerate() {
        let base = p * cols + j0;
        let row = &mat[base..base + N];
        for (a, &w) in acc.iter_mut().zip(row) {
            *a = x_p.mul_add(w, *a);
        }
    }
    out[j0..j0 + N].copy_from_slice(&acc);
}

/// Transpose a rank-2 tensor: `[r, c] -> [c, r]`.
pub fn transpose(x: &Tensor) -> Result<Tensor, TensorError> {
    if x.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            got: x.rank(),
        });
    }
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut out = vec![0.0f32; rows * cols];
    if cols == 0 {
        return Tensor::from_vec(out, &[cols, rows]);
    }
    for (i, row) in x.data().chunks(cols).enumerate() {
        for (j, &value) in row.iter().enumerate() {
            out[j * rows + i] = value;
        }
    }
    Tensor::from_vec(out, &[cols, rows])
}
