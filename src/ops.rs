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
    let lhs = a.data();
    let weights = w.data();
    let mut out = vec![0.0f32; m * n];

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
                packed_micro(&a_panel, &b_panel, &mut out, n, mc, nc, kc, i0, j0, p0);
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
    Tensor::from_vec(out, &[m, n])
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

/// Matrix-vector product with a row-major `[out, in]` weight matrix:
/// `y[o] = sum_i w[o, i] * x[i]`, returned as `[1, out]`.
///
/// This is the layout every framework uses for a linear layer (PyTorch,
/// llama2.c, safetensors), and it is the shape a decode step needs: each output
/// element reads one contiguous row of `w`, so the matrix is streamed exactly
/// once. Unlike [`matvec`], no column blocking is involved, which matters when
/// `out` is large — the vocabulary projection here has 32,000 outputs.
pub fn gemv(w: &Matrix, x: &[f32]) -> Result<Tensor, TensorError> {
    let mut y = vec![0.0f32; w.out_dim()];
    gemv_into(w, x, &mut y)?;
    Tensor::from_vec(y, &[1, w.out_dim()])
}

/// `gemv` into a caller-owned buffer, so a decode loop can reuse one allocation
/// per output instead of one per call. There are 211 projections per token, and
/// at one heap allocation each the allocator was costing more than the kernels.
pub fn gemv_into(w: &Matrix, x: &[f32], y: &mut [f32]) -> Result<(), TensorError> {
    let out_dim = w.out_dim();
    let in_dim = w.in_dim();
    if x.len() != in_dim || y.len() != out_dim {
        return Err(TensorError::ShapeMismatch {
            op: "gemv",
            a: vec![out_dim, in_dim],
            b: vec![x.len(), y.len()],
        });
    }

    match w {
        Matrix::F32(tensor) => {
            for (row, y_o) in tensor.data().chunks_exact(in_dim).zip(y.iter_mut()) {
                *y_o = dot(row, x);
            }
        }
        Matrix::Bf16 { data, .. } => {
            for (row, y_o) in data.chunks_exact(in_dim).zip(y.iter_mut()) {
                *y_o = dot_bf16(row, x);
            }
        }
        Matrix::Q8_0 { scales, quants, .. } => {
            // The input is quantized once and reused by every row; only the
            // weights differ per row.
            let qx = Q8Vec::quantize(x);
            let blocks = in_dim / Q8_0_BLOCK;
            for (i, y_o) in y.iter_mut().enumerate() {
                let row = &quants[i * in_dim..(i + 1) * in_dim];
                let row_scales = &scales[i * blocks..(i + 1) * blocks];
                *y_o = dot_q8_0(row, row_scales, &qx.quants, &qx.scales);
            }
        }
    }
    Ok(())
}

/// `gemv` with an already-quantized activation, so several projections that
/// share an input (q/k/v, or gate/up) quantize it once between them.
pub fn gemv_q8_0_into(w: &Matrix, qx: &Q8Vec, y: &mut [f32]) -> Result<(), TensorError> {
    let Matrix::Q8_0 { scales, quants, .. } = w else {
        return Err(TensorError::ShapeMismatch {
            op: "gemv_q8_0_into (weights are not Q8_0)",
            a: vec![w.out_dim(), w.in_dim()],
            b: vec![y.len()],
        });
    };
    let out_dim = w.out_dim();
    let in_dim = w.in_dim();
    if qx.quants.len() < in_dim || y.len() != out_dim {
        return Err(TensorError::ShapeMismatch {
            op: "gemv_q8_0_into",
            a: vec![out_dim, in_dim],
            b: vec![qx.quants.len(), y.len()],
        });
    }
    let blocks = in_dim / Q8_0_BLOCK;
    for (i, y_o) in y.iter_mut().enumerate() {
        let row = &quants[i * in_dim..(i + 1) * in_dim];
        let row_scales = &scales[i * blocks..(i + 1) * blocks];
        *y_o = dot_q8_0(row, row_scales, &qx.quants, &qx.scales);
    }
    Ok(())
}

/// A dense weight matrix in row-major `[out, in]` layout.
///
/// Model files store these at low precision, and decode streams every weight
/// once per token, so keeping the file's own storage avoids doubling the bytes
/// moved. bf16 is widened to f32 *inside* the kernel (operands low precision,
/// accumulator f32), which is the usual mixed-precision arrangement. `Q8_0`
/// goes further and stores int8 with one f32 scale per 32-element block, a
/// quarter of the f32 bytes.
pub enum Matrix {
    F32(Tensor),
    Bf16 {
        data: Vec<u16>,
        out_dim: usize,
        in_dim: usize,
    },
    /// Quantized 8-bit, ggml's "Q8_0" block layout: 32 int8 values sharing one
    /// f32 scale. `quants` is `out_dim * in_dim` long and `scales` holds one
    /// entry per block, so row `r` owns `scales[r * in_dim / 32 ..][..in_dim / 32]`.
    Q8_0 {
        scales: Vec<f32>,
        quants: Vec<i8>,
        out_dim: usize,
        in_dim: usize,
    },
}

impl Matrix {
    /// Wrap an f32 tensor, which must be rank-2.
    pub fn from_f32(tensor: Tensor) -> Self {
        debug_assert_eq!(tensor.rank(), 2, "weight matrices are rank-2");
        Matrix::F32(tensor)
    }

    /// Wrap raw bf16 bit patterns, already in `[out, in]` order.
    pub fn from_bf16_bits(data: Vec<u16>, out_dim: usize, in_dim: usize) -> Self {
        debug_assert_eq!(data.len(), out_dim * in_dim);
        Matrix::Bf16 {
            data,
            out_dim,
            in_dim,
        }
    }

    /// Quantize an f32 tensor to the Q8_0 block layout.
    ///
    /// Returns `None` when `in_dim` is not a multiple of the block size, since
    /// blocks must not straddle row boundaries for the per-row scale indexing in
    /// [`gemv`] to line up.
    pub fn from_f32_q8_0(tensor: Tensor) -> Option<Self> {
        debug_assert_eq!(tensor.rank(), 2, "weight matrices are rank-2");
        let (out_dim, in_dim) = (tensor.shape()[0], tensor.shape()[1]);
        if in_dim % Q8_0_BLOCK != 0 {
            return None;
        }
        let q = Q8Vec::quantize(tensor.data());
        Some(Matrix::Q8_0 {
            scales: q.scales,
            quants: q.quants,
            out_dim,
            in_dim,
        })
    }

    /// Name of the in-memory storage, for diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Matrix::F32(_) => "f32",
            Matrix::Bf16 { .. } => "bf16",
            Matrix::Q8_0 { .. } => "q8_0",
        }
    }

    /// Number of output rows.
    pub fn out_dim(&self) -> usize {
        match self {
            Matrix::F32(t) => t.shape()[0],
            Matrix::Bf16 { out_dim, .. } | Matrix::Q8_0 { out_dim, .. } => *out_dim,
        }
    }

    /// Number of input columns.
    pub fn in_dim(&self) -> usize {
        match self {
            Matrix::F32(t) => t.shape()[1],
            Matrix::Bf16 { in_dim, .. } | Matrix::Q8_0 { in_dim, .. } => *in_dim,
        }
    }

    /// The whole matrix widened to f32, row-major `[out, in]`.
    ///
    /// Only for the batched path, where there is no low-precision GEMM yet: it
    /// costs a full copy of the weights, so it should be called once per model,
    /// not once per layer per token.
    pub fn to_f32_tensor(&self) -> Tensor {
        let (out_dim, in_dim) = (self.out_dim(), self.in_dim());
        let data = match self {
            Matrix::F32(t) => t.data().to_vec(),
            Matrix::Bf16 { data, .. } => data.iter().map(|&b| bf16_to_f32(b)).collect(),
            Matrix::Q8_0 { .. } => {
                let mut all = Vec::with_capacity(out_dim * in_dim);
                for r in 0..out_dim {
                    all.extend(self.row(r));
                }
                all
            }
        };
        Tensor::from_vec(data, &[out_dim, in_dim]).expect("shape matches the data")
    }

    /// Read row `index` as f32 — used for the token-embedding lookup.
    pub fn row(&self, index: usize) -> Vec<f32> {
        let (start, end) = (index * self.in_dim(), (index + 1) * self.in_dim());
        match self {
            Matrix::F32(t) => t.data()[start..end].to_vec(),
            Matrix::Bf16 { data, .. } => data[start..end].iter().map(|&b| bf16_to_f32(b)).collect(),
            Matrix::Q8_0 {
                scales,
                quants,
                in_dim,
                ..
            } => {
                let blocks = in_dim / Q8_0_BLOCK;
                let (qs, ss) = (
                    &quants[start..end],
                    &scales[index * blocks..(index + 1) * blocks],
                );
                let mut out = Vec::with_capacity(*in_dim);
                for (block, &scale) in ss.iter().enumerate() {
                    let lo = block * Q8_0_BLOCK;
                    for &q in &qs[lo..lo + Q8_0_BLOCK] {
                        out.push(f32::from(q) * scale);
                    }
                }
                out
            }
        }
    }
}

/// Elements per quantization block in the Q8_0 layout.
pub const Q8_0_BLOCK: usize = 32;

/// A vector quantized to Q8_0: one f32 scale per 32-element block plus the
/// int8 values. `quants` is always a whole number of blocks (the tail of a
/// partial block is zero-padded), which keeps the kernels branch-free.
pub struct Q8Vec {
    pub scales: Vec<f32>,
    pub quants: Vec<i8>,
}

impl Q8Vec {
    /// Quantize with the standard ggml rounding: the block scale is the largest
    /// magnitude divided by the largest int8, so the biggest value maps exactly
    /// to ±127 and everything else rounds to nearest.
    pub fn quantize(x: &[f32]) -> Self {
        let blocks = x.len().div_ceil(Q8_0_BLOCK);
        let mut scales = Vec::with_capacity(blocks);
        let mut quants = vec![0i8; blocks * Q8_0_BLOCK];
        for (block, chunk) in x.chunks(Q8_0_BLOCK).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            scales.push(d);
            for (i, &v) in chunk.iter().enumerate() {
                quants[block * Q8_0_BLOCK + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Q8Vec { scales, quants }
    }

    /// Quantize into an existing `Q8Vec`, reusing its allocations. The decode
    /// loop calls this for every projection, so growing the buffers once and
    /// refilling them avoids hundreds of allocations per token.
    pub fn quantize_into(&mut self, x: &[f32]) {
        let blocks = x.len().div_ceil(Q8_0_BLOCK);
        self.scales.clear();
        self.scales.reserve(blocks);
        self.quants.clear();
        self.quants.resize(blocks * Q8_0_BLOCK, 0);
        for (block, chunk) in x.chunks(Q8_0_BLOCK).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            self.scales.push(d);
            for (i, &v) in chunk.iter().enumerate() {
                self.quants[block * Q8_0_BLOCK + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}

/// Dot product of one Q8_0 weight row with a Q8_0 activation vector. The int8
/// products are summed exactly in integer arithmetic per block, so the only
/// rounding is the quantization itself (and one f32 multiply per block).
///
/// Weight scales arrive as f16 (the ggml/Q8_0 storage width) and are widened as
/// they are consumed; the activation scales are already f32.
#[inline]
pub(crate) fn dot_q8_0(a: &[i8], a_scales: &[f32], b: &[i8], b_scales: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: `dotprod` implies the SDOT instruction this uses.
            unsafe { dot_q8_0_sdot(a, a_scales, b, b_scales) }
        } else {
            dot_q8_0_scalar(a, a_scales, b, b_scales)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_q8_0_scalar(a, a_scales, b, b_scales)
    }
}

/// Portable Q8_0 dot product. Integer accumulation is exact, so this agrees
/// with the SDOT kernel up to the f32 summation order.
pub(crate) fn dot_q8_0_scalar(a: &[i8], a_scales: &[f32], b: &[i8], b_scales: &[f32]) -> f32 {
    let mut total = [0.0f32; 4];
    for (block, (&sa, &sb)) in a_scales.iter().zip(b_scales).enumerate() {
        let lo = block * Q8_0_BLOCK;
        let mut acc = 0i32;
        for (&x, &y) in a[lo..lo + Q8_0_BLOCK].iter().zip(&b[lo..lo + Q8_0_BLOCK]) {
            acc += i32::from(x) * i32::from(y);
        }
        total[block & 3] += (sa * sb) * acc as f32;
    }
    total[0] + total[1] + total[2] + total[3]
}

/// One `SDOT`: `acc[i] += a[4i..4i+4] . b[4i..4i+4]`, four int8 products per
/// int32 lane. Written as inline assembly because the `vdotq_s32` intrinsic is
/// still unstable in Rust 1.87 (`stdarch_neon_dotprod`).
///
/// SAFETY: the CPU must implement `dotprod`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot(
    acc: core::arch::aarch64::int32x4_t,
    a: core::arch::aarch64::int8x16_t,
    b: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    let mut acc = acc;
    // SAFETY: `dotprod` is the caller's contract; the operands are 128-bit
    // vectors resident in registers.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
    }
    acc
}

/// NEON Q8_0 dot product using SDOT, which sums four int8 products into each of
/// four int32 lanes — sixteen multiply-adds per instruction. That is what makes
/// int8 worth the bookkeeping: the kernel issues far fewer instructions than the
/// f32 one *and* reads a quarter of the bytes.
///
/// The block scale is applied while the partial sums are still in vector lanes,
/// and the horizontal reduction is deferred to the end of the row. Reducing per
/// block instead costs more than the multiply-adds it computes: the block is
/// only 32 elements, so a `vaddvq` plus a serial float chain per block ends up
/// dominating.
///
/// SAFETY: the caller must guarantee `dotprod` is available, and that `a` and
/// `b` each hold at least `32 * a_scales.len()` values.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q8_0_sdot(a: &[i8], a_scales: &[f32], b: &[i8], b_scales: &[f32]) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_n_f32, vld1q_s8,
    };

    // Everything below walks raw pointers. Indexing the slices each iteration
    // made LLVM emit a bounds check and re-derive the offsets per block, and the
    // resulting branches cost more than the multiply-adds: the kernel ran at
    // about half this machine's streaming bandwidth with them.
    let mut wp = a.as_ptr();
    let mut xp = b.as_ptr();
    let mut sp = a_scales.as_ptr();
    let mut sq = b_scales.as_ptr();
    let mut blocks = a_scales.len();

    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };

    // Four blocks (128 weight bytes) per trip, each into its own accumulator.
    // Fewer accumulators left both the FMA chain and the load units idle.
    while blocks >= 4 {
        // SAFETY: the caller guarantees `32 * blocks` values in each slice and
        // `blocks` counts the remaining blocks, so these reads are in range.
        let (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7) = unsafe {
            (
                vld1q_s8(wp),
                vld1q_s8(wp.add(16)),
                vld1q_s8(wp.add(32)),
                vld1q_s8(wp.add(48)),
                vld1q_s8(wp.add(64)),
                vld1q_s8(wp.add(80)),
                vld1q_s8(wp.add(96)),
                vld1q_s8(wp.add(112)),
                vld1q_s8(xp),
                vld1q_s8(xp.add(16)),
                vld1q_s8(xp.add(32)),
                vld1q_s8(xp.add(48)),
                vld1q_s8(xp.add(64)),
                vld1q_s8(xp.add(80)),
                vld1q_s8(xp.add(96)),
                vld1q_s8(xp.add(112)),
            )
        };
        // SAFETY: `dotprod` availability is the caller's contract. The two
        // halves of a block are independent SDOTs combined by a vector add,
        // rather than chaining the second through the first's accumulator,
        // which would serialize two multi-cycle operations per block.
        let d0 = unsafe { sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1) };
        let d1 = unsafe { sdot(sdot(vdupq_n_s32(0), w2, x2), w3, x3) };
        let d2 = unsafe { sdot(sdot(vdupq_n_s32(0), w4, x4), w5, x5) };
        let d3 = unsafe { sdot(sdot(vdupq_n_s32(0), w6, x6), w7, x7) };
        // SAFETY: scales are read straight from the caller's slices.
        let (s0, s1, s2, s3) = unsafe {
            (
                *sp * *sq,
                *sp.add(1) * *sq.add(1),
                *sp.add(2) * *sq.add(2),
                *sp.add(3) * *sq.add(3),
            )
        };
        // SAFETY: plain NEON arithmetic on values already in registers.
        unsafe {
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), s0);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(d1), s1);
            acc2 = vfmaq_n_f32(acc2, vcvtq_f32_s32(d2), s2);
            acc3 = vfmaq_n_f32(acc3, vcvtq_f32_s32(d3), s3);
            wp = wp.add(128);
            xp = xp.add(128);
            sp = sp.add(4);
            sq = sq.add(4);
        }
        blocks -= 4;
    }

    let mut acc = unsafe { vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)) };
    while blocks > 0 {
        // SAFETY: as above; at least one whole block remains.
        let (w0, w1, x0, x1) = unsafe {
            (
                vld1q_s8(wp),
                vld1q_s8(wp.add(16)),
                vld1q_s8(xp),
                vld1q_s8(xp.add(16)),
            )
        };
        let d = unsafe { sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1) };
        // SAFETY: as above.
        let s = unsafe { *sp * *sq };
        acc = unsafe { vfmaq_n_f32(acc, vcvtq_f32_s32(d), s) };
        // SAFETY: as above.
        unsafe {
            wp = wp.add(32);
            xp = xp.add(32);
            sp = sp.add(1);
            sq = sq.add(1);
        }
        blocks -= 1;
    }

    // SAFETY: as above.
    unsafe { vaddvq_f32(acc) }
}

/// bfloat16 bit pattern to f32: the value is the top half, so shift it up.
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Dot product with a bf16 weight row: half the bytes to load, widened in
/// registers before the multiply-add.
pub(crate) fn dot_bf16(a: &[u16], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        dot_bf16_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_bf16_scalar(a, b)
    }
}

/// Portable bf16 dot product with four independent chains.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn dot_bf16_scalar(a: &[u16], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= a.len() {
        acc[0] = bf16_to_f32(a[i]).mul_add(b[i], acc[0]);
        acc[1] = bf16_to_f32(a[i + 1]).mul_add(b[i + 1], acc[1]);
        acc[2] = bf16_to_f32(a[i + 2]).mul_add(b[i + 2], acc[2]);
        acc[3] = bf16_to_f32(a[i + 3]).mul_add(b[i + 3], acc[3]);
        i += 4;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < a.len() {
        sum = bf16_to_f32(a[i]).mul_add(b[i], sum);
        i += 1;
    }
    sum
}

/// NEON bf16 dot product.
///
/// The weights are stored as u16, so each load fetches 16 bytes = 8 bf16 values
/// and `vshll` widens them left by 16 — bf16 to f32 in a single instruction.
/// Four widen + four fma per 16 elements would be 12 instructions (the f32
/// kernel needs 8), and that extra instruction cost, not memory, is what limits
/// bf16. Loading at 128 bits and splitting with `vget_low`/`vget_high`, which
/// are pure register renames, drops it to 10; eight independent accumulators
/// hide the load and widen latencies behind each other.
///
/// SAFETY: NEON is architecturally mandatory on AArch64, the only target that
/// compiles this. Every load sits inside `i + 16 <= len` or the `i < len` tail.
#[cfg(target_arch = "aarch64")]
fn dot_bf16_neon(a: &[u16], b: &[f32]) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vget_high_u16, vget_low_u16, vld1q_f32,
        vld1q_u16, vreinterpretq_f32_u32, vshll_n_u16,
    };

    // SAFETY: see the function comment.
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        // A bf16 value is the top 16 bits of an f32, so widening a u16 lane left
        // by 16 bit positions *is* the conversion to f32.
        let widen = |lanes: core::arch::aarch64::uint16x4_t| {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(lanes))
        };
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 32 <= len {
            // One 16-byte load covers eight bf16 values, so four loads cover
            // this iteration's 32 elements exactly; the low/high splits are
            // free register renames, not instructions.
            let x0 = vld1q_u16(ap.add(i));
            let x1 = vld1q_u16(ap.add(i + 8));
            let x2 = vld1q_u16(ap.add(i + 16));
            let x3 = vld1q_u16(ap.add(i + 24));
            acc0 = vfmaq_f32(acc0, widen(vget_low_u16(x0)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, widen(vget_high_u16(x0)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, widen(vget_low_u16(x1)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, widen(vget_high_u16(x1)), vld1q_f32(bp.add(i + 12)));
            acc4 = vfmaq_f32(acc4, widen(vget_low_u16(x2)), vld1q_f32(bp.add(i + 16)));
            acc5 = vfmaq_f32(acc5, widen(vget_high_u16(x2)), vld1q_f32(bp.add(i + 20)));
            acc6 = vfmaq_f32(acc6, widen(vget_low_u16(x3)), vld1q_f32(bp.add(i + 24)));
            acc7 = vfmaq_f32(acc7, widen(vget_high_u16(x3)), vld1q_f32(bp.add(i + 28)));
            i += 32;
        }
        let acc45 = vaddq_f32(acc4, acc5);
        let acc67 = vaddq_f32(acc6, acc7);
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        sum += vaddvq_f32(vaddq_f32(acc45, acc67));
        while i < len {
            sum = bf16_to_f32(a[i]).mul_add(b[i], sum);
            i += 1;
        }
        sum
    }
}

/// Dot product. On AArch64 this dispatches to the NEON kernel; elsewhere to the
/// portable scalar one.
///
/// Auto-vectorization does not help here: a dot product is a *reduction*, and
/// LLVM may not reassociate floating-point additions, so it cannot split one
/// accumulator into several vector lanes on its own. (That is why the matmul
/// vectorized but this did not.) Four independent accumulators are written out
/// by hand to expose the parallelism.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        dot_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_scalar(a, b)
    }
}

/// Portable dot product with four independent chains.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= a.len() {
        acc[0] = a[i].mul_add(b[i], acc[0]);
        acc[1] = a[i + 1].mul_add(b[i + 1], acc[1]);
        acc[2] = a[i + 2].mul_add(b[i + 2], acc[2]);
        acc[3] = a[i + 3].mul_add(b[i + 3], acc[3]);
        i += 4;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < a.len() {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }
    sum
}

/// NEON dot product: four vector accumulators, so sixteen elements retire per
/// iteration with four independent dependency chains.
///
/// Four was chosen by measurement: eight accumulators was no faster, which
/// says the reduction is limited by memory streaming rather than by multiply-add
/// throughput or load latency.
///
/// SAFETY: NEON is architecturally mandatory on AArch64, the only target that
/// compiles this. Every load sits inside `i + 16 <= len` or the `i < len` tail,
/// and AArch64 permits unaligned vector loads from ordinary memory.
#[cfg(target_arch = "aarch64")]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    // SAFETY: see the function comment.
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= len {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
            i += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < len {
            sum = ap.add(i).read().mul_add(bp.add(i).read(), sum);
            i += 1;
        }
        sum
    }
}

/// Elementwise rectified linear unit: `max(0, x)`, preserving the shape.
///
/// Unlike the shape-combining ops this cannot fail, so it returns `Tensor`
/// rather than `Result`. It builds from `zeros` and fills in place: the same
/// allocation pattern serves every elementwise op, and the count of elements is
/// unchanged by construction, so no validation is needed.
pub fn relu(x: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(x.shape());
    for (out_v, &x_v) in out.data_mut().iter_mut().zip(x.data()) {
        *out_v = x_v.max(0.0);
    }
    out
}

/// Add a per-column bias to a rank-2 tensor: `x[m, n] + bias[n] -> [m, n]`.
///
/// This is the `+ b` of a dense layer. Only the trailing dimension is
/// broadcast, which is all a bias vector needs; general broadcasting is a much
/// larger feature with no caller here yet.
pub fn add_bias(x: &Tensor, bias: &Tensor) -> Result<Tensor, TensorError> {
    if x.rank() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            got: x.rank(),
        });
    }
    if bias.rank() != 1 {
        return Err(TensorError::RankMismatch {
            expected: 1,
            got: bias.rank(),
        });
    }
    let n = x.shape()[1];
    if bias.shape()[0] != n {
        return Err(TensorError::ShapeMismatch {
            op: "add_bias",
            a: x.shape().to_vec(),
            b: bias.shape().to_vec(),
        });
    }

    let mut out = Tensor::zeros(x.shape());
    if n == 0 {
        return Ok(out);
    }
    let bias = bias.data();
    for (out_row, x_row) in out.data_mut().chunks_mut(n).zip(x.data().chunks(n)) {
        for ((out_v, &x_v), &b_v) in out_row.iter_mut().zip(x_row).zip(bias) {
            *out_v = x_v + b_v;
        }
    }
    Ok(out)
}

/// Softmax over the last dimension.
///
/// Each row is shifted by its own maximum before exponentiating. That is not
/// cosmetic: `exp(1000)` overflows `f32` to infinity and the ratio becomes
/// `NaN`, while softmax is invariant to a constant shift, so `exp(x - max)`
/// computes the same function without ever overflowing for finite inputs.
/// A zero-length last dimension is returned unchanged (there is nothing to
/// normalize).
pub fn softmax(x: &Tensor) -> Tensor {
    let last = x.shape().last().copied().unwrap_or(0);
    let mut out = x.clone();
    if last == 0 {
        return out;
    }
    for row in out.data_mut().chunks_mut(last) {
        softmax_in_place(row);
    }
    out
}

/// In-place numerically-stable softmax over a slice. Shared by the
/// transformer's attention and the sampler.
pub(crate) fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
