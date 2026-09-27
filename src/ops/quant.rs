//! Weight storage and the quantized `gemv`.

use super::dot::{bf16_to_f32, dot, dot_bf16, dot_q8_0};
use crate::tensor::{Tensor, TensorError};

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
        Matrix::Q8_0x4(packed) => {
            let qx = Q8Vec::quantize(x);
            gemv_q8_0x4(packed, &qx, y);
        }
    }
    Ok(())
}

/// `gemv` with an already-quantized activation, so several projections that
/// share an input (q/k/v, or gate/up) quantize it once between them.
pub fn gemv_q8_0_into(w: &Matrix, qx: &Q8Vec, y: &mut [f32]) -> Result<(), TensorError> {
    let out_dim = w.out_dim();
    let in_dim = w.in_dim();
    if qx.quants.len() < in_dim || y.len() != out_dim {
        return Err(TensorError::ShapeMismatch {
            op: "gemv_q8_0_into",
            a: vec![out_dim, in_dim],
            b: vec![qx.quants.len(), y.len()],
        });
    }
    match w {
        Matrix::Q8_0 { scales, quants, .. } => {
            let blocks = in_dim / Q8_0_BLOCK;
            for (i, y_o) in y.iter_mut().enumerate() {
                let row = &quants[i * in_dim..(i + 1) * in_dim];
                let row_scales = &scales[i * blocks..(i + 1) * blocks];
                *y_o = dot_q8_0(row, row_scales, &qx.quants, &qx.scales);
            }
        }
        Matrix::Q8_0x4(packed) => {
            gemv_q8_0x4(packed, qx, y);
        }
        _ => {
            return Err(TensorError::ShapeMismatch {
                op: "gemv_q8_0_into (weights are not Q8_0)",
                a: vec![out_dim, in_dim],
                b: vec![y.len()],
            });
        }
    }
    Ok(())
}

/// Four-row-tiled Q8_0 weights with SDOT lane-broadcast GEMV. Each block
/// interleaves four rows in 4-byte chunks, so one `sdot` updates four row sums
/// from one 4-byte activation lane.
#[derive(Clone)]
pub struct Q8_0x4 {
    /// ggml-compatible blocks, each `[4 x f16 scales][4 x 32 int8 quants]`.
    blocks: Vec<Q8Block4>,
    out_dim: usize,
    in_dim: usize,
}

/// Four rows of one Q8_0 block, matching ggml's `block_q8_0x4` byte layout.
#[repr(C)]
#[derive(Clone)]
struct Q8Block4 {
    scales: [u16; 4],
    quants: [i8; 128],
}

const _: () = assert!(std::mem::size_of::<Q8Block4>() == 136);

impl Q8_0x4 {
    /// Quantize and repack `[out, in]` weights into groups of four rows.
    /// Returns `None` unless the row width is a multiple of 32 and output rows
    /// are a multiple of four.
    pub fn from_f32(tensor: Tensor) -> Option<Self> {
        debug_assert_eq!(tensor.rank(), 2);
        let (out_dim, in_dim) = (tensor.shape()[0], tensor.shape()[1]);
        if in_dim % Q8_0_BLOCK != 0 || out_dim % 4 != 0 {
            return None;
        }
        let groups = out_dim / 4;
        let blocks = in_dim / Q8_0_BLOCK;
        let mut packed = Vec::with_capacity(groups * blocks);
        for group in 0..groups {
            for block in 0..blocks {
                let col0 = block * Q8_0_BLOCK;
                let mut packed_block = Q8Block4 {
                    scales: [0; 4],
                    quants: [0; 128],
                };
                for row_in_group in 0..4 {
                    let row = group * 4 + row_in_group;
                    let src = &tensor.data()[row * in_dim + col0..row * in_dim + col0 + 32];
                    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = amax / 127.0;
                    packed_block.scales[row_in_group] = f32_to_f16_bits(scale);
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    for segment in 0..8 {
                        for lane in 0..4 {
                            let dst = segment * 16 + row_in_group * 4 + lane;
                            let value = src[segment * 4 + lane];
                            packed_block.quants[dst] =
                                (value * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
                packed.push(packed_block);
            }
        }
        Some(Self {
            blocks: packed,
            out_dim,
            in_dim,
        })
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Reference-row decoder for tests and the non-dotprod fallback.
    fn row(&self, row: usize) -> Vec<f32> {
        let blocks = self.in_dim / Q8_0_BLOCK;
        let group = row / 4;
        let row_in_group = row % 4;
        let mut out = Vec::with_capacity(self.in_dim);
        for block in 0..blocks {
            let packed = &self.blocks[group * blocks + block];
            let scale = f16_scale_to_f32(packed.scales[row_in_group]);
            for segment in 0..8 {
                for lane in 0..4 {
                    let q = packed.quants[segment * 16 + row_in_group * 4 + lane];
                    out.push(f32::from(q) * scale);
                }
            }
        }
        out
    }
}

/// GEMV for the four-row-interleaved Q8_0 layout.
fn gemv_q8_0x4(packed: &Q8_0x4, qx: &Q8Vec, y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod")
        && std::arch::is_aarch64_feature_detected!("fp16")
    {
        // SAFETY: feature detection plus constructor-validated group/block sizes.
        unsafe { gemv_q8_0x4_neon(packed, qx, y) };
        return;
    }
    for (row, value) in y.iter_mut().enumerate() {
        let row_q = Q8Vec::quantize(&packed.row(row));
        *value = dot_q8_0(&row_q.quants, &row_q.scales, &qx.quants, &qx.scales);
    }
}

/// Batched Q8_0x4 matrix product with pre-quantized activations.
///
/// `quants` and `scales_f16` hold `rows` consecutive Q8_0 vectors. Unlike
/// calling GEMV once per row, this walks one weight block and applies it to
/// every activation row while the 136-byte block is hot in L1. The operation
/// order for an individual output is still identical to [`gemv_q8_0x4`]: one
/// integer dot and one f32 FMA per quantization block.
pub fn gemm_q8_0x4_batched_into(
    packed: &Q8_0x4,
    quants: &[i8],
    scales_f16: &[u16],
    rows: usize,
    y: &mut [f32],
) {
    let blocks = packed.in_dim / Q8_0_BLOCK;
    debug_assert_eq!(quants.len(), rows * packed.in_dim);
    debug_assert_eq!(scales_f16.len(), rows * blocks);
    debug_assert_eq!(y.len(), rows * packed.out_dim);

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod")
        && std::arch::is_aarch64_feature_detected!("fp16")
    {
        // SAFETY: feature detection plus the shape assertions above.
        unsafe { gemm_q8_0x4_batched_neon(packed, quants, scales_f16, rows, y) };
        return;
    }

    // Portable reference. This keeps the same blockwise accumulation as the
    // NEON path and is also useful when running the test suite on x86.
    for row in 0..rows {
        for group in 0..packed.out_dim / 4 {
            let out =
                &mut y[row * packed.out_dim + group * 4..row * packed.out_dim + group * 4 + 4];
            out.fill(0.0);
            for block in 0..blocks {
                let weights = &packed.blocks[group * blocks + block];
                let xs = f16_scale_to_f32(scales_f16[row * blocks + block]);
                for (lane, slot) in out.iter_mut().enumerate() {
                    let ws = f16_scale_to_f32(weights.scales[lane]) * xs;
                    let mut sum = 0i32;
                    for segment in 0..8 {
                        for byte in 0..4 {
                            sum += i32::from(weights.quants[segment * 16 + lane * 4 + byte])
                                * i32::from(
                                    quants[row * packed.in_dim + block * 32 + segment * 4 + byte],
                                );
                        }
                    }
                    *slot = (sum as f32).mul_add(ws, *slot);
                }
            }
        }
    }
}

/// SDOT-lane-broadcast GEMV, four rows per activation block load.
///
/// SAFETY: dotprod is available; dimensions and buffers were validated by the
/// Q8_0x4 constructor and gemv shape checks.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn q8_0x4_block_dot(
    packed: &Q8_0x4,
    qx: &Q8Vec,
    group: usize,
    block: usize,
) -> core::arch::aarch64::int32x4_t {
    use core::arch::aarch64::{vdupq_n_s32, vld1q_s8};

    let blocks = packed.in_dim / Q8_0_BLOCK;
    let block_data = &packed.blocks[group * blocks + block];
    let xbase = block * 32;
    let xp = qx.quants.as_ptr();
    let wp = block_data.quants.as_ptr();
    // Load activation and all eight row-interleaved weight vectors before
    // starting the dependent SDOT chain.
    let (x0, x1) = unsafe { (vld1q_s8(xp.add(xbase)), vld1q_s8(xp.add(xbase + 16))) };
    let (w0, w1, w2, w3, w4, w5, w6, w7) = unsafe {
        (
            vld1q_s8(wp),
            vld1q_s8(wp.add(16)),
            vld1q_s8(wp.add(32)),
            vld1q_s8(wp.add(48)),
            vld1q_s8(wp.add(64)),
            vld1q_s8(wp.add(80)),
            vld1q_s8(wp.add(96)),
            vld1q_s8(wp.add(112)),
        )
    };
    // SAFETY: dotprod availability is checked by the caller; each lane group
    // in the packed weights corresponds to one output row.
    let mut sums = unsafe { vdupq_n_s32(0) };
    sums = unsafe { super::dot::sdot_lane::<0>(sums, w0, x0) };
    sums = unsafe { super::dot::sdot_lane::<1>(sums, w1, x0) };
    sums = unsafe { super::dot::sdot_lane::<2>(sums, w2, x0) };
    sums = unsafe { super::dot::sdot_lane::<3>(sums, w3, x0) };
    sums = unsafe { super::dot::sdot_lane::<0>(sums, w4, x1) };
    sums = unsafe { super::dot::sdot_lane::<1>(sums, w5, x1) };
    sums = unsafe { super::dot::sdot_lane::<2>(sums, w6, x1) };
    unsafe { super::dot::sdot_lane::<3>(sums, w7, x1) }
}

/// Same four-output integer dot as [`q8_0x4_block_dot`], but with an activation
/// pointer supplied by the batched kernel rather than a `Q8Vec` wrapper.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn q8_0x4_block_dot_raw(
    block_data: &Q8Block4,
    x: *const i8,
) -> core::arch::aarch64::int32x4_t {
    use core::arch::aarch64::{vdupq_n_s32, vld1q_s8};
    let (x0, x1) = unsafe { (vld1q_s8(x), vld1q_s8(x.add(16))) };
    let wp = block_data.quants.as_ptr();
    let (w0, w1, w2, w3, w4, w5, w6, w7) = unsafe {
        (
            vld1q_s8(wp),
            vld1q_s8(wp.add(16)),
            vld1q_s8(wp.add(32)),
            vld1q_s8(wp.add(48)),
            vld1q_s8(wp.add(64)),
            vld1q_s8(wp.add(80)),
            vld1q_s8(wp.add(96)),
            vld1q_s8(wp.add(112)),
        )
    };
    let mut sums = unsafe { vdupq_n_s32(0) };
    sums = unsafe { super::dot::sdot_lane::<0>(sums, w0, x0) };
    sums = unsafe { super::dot::sdot_lane::<1>(sums, w1, x0) };
    sums = unsafe { super::dot::sdot_lane::<2>(sums, w2, x0) };
    sums = unsafe { super::dot::sdot_lane::<3>(sums, w3, x0) };
    sums = unsafe { super::dot::sdot_lane::<0>(sums, w4, x1) };
    sums = unsafe { super::dot::sdot_lane::<1>(sums, w5, x1) };
    sums = unsafe { super::dot::sdot_lane::<2>(sums, w6, x1) };
    unsafe { super::dot::sdot_lane::<3>(sums, w7, x1) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn gemv_q8_0x4_neon(packed: &Q8_0x4, qx: &Q8Vec, y: &mut [f32]) {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vmulq_f32, vst1q_f32};
    let blocks = packed.in_dim / Q8_0_BLOCK;
    for group in 0..packed.out_dim / 4 {
        let mut total = unsafe { vdupq_n_f32(0.0) };
        for block in 0..blocks {
            let d = unsafe { q8_0x4_block_dot(packed, qx, group, block) };
            let ws = unsafe { widen_scale4(packed.blocks[group * blocks + block].scales.as_ptr()) };
            let xs = unsafe { widen_scale_dup(qx.scales_f16[block]) };
            total = unsafe { vfmaq_f32(total, vcvtq_f32_s32(d), vmulq_f32(ws, xs)) };
        }
        // Store the four lanes directly into the output row; the old temporary
        // array added a second copy to every output group in decode.
        unsafe { vst1q_f32(y.as_mut_ptr().add(group * 4), total) };
    }
}

/// Register-blocked Q8_0x4 GEMM. Four prompt rows stay in independent NEON
/// accumulators while walking a weight group's blocks. This hides SDOT latency
/// and eliminates a partial-output load/store after every quantization block.
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_q8_0x4_batched_neon(
    packed: &Q8_0x4,
    quants: &[i8],
    scales_f16: &[u16],
    rows: usize,
    y: &mut [f32],
) {
    use core::arch::aarch64::{vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vmulq_f32, vst1q_f32};
    let blocks = packed.in_dim / Q8_0_BLOCK;
    for group in 0..packed.out_dim / 4 {
        let mut row = 0;
        while row + 4 <= rows {
            let mut total0 = unsafe { vdupq_n_f32(0.0) };
            let mut total1 = unsafe { vdupq_n_f32(0.0) };
            let mut total2 = unsafe { vdupq_n_f32(0.0) };
            let mut total3 = unsafe { vdupq_n_f32(0.0) };
            for block in 0..blocks {
                let weight = &packed.blocks[group * blocks + block];
                let ws = unsafe { widen_scale4(weight.scales.as_ptr()) };
                let qbase = quants
                    .as_ptr()
                    .wrapping_add(row * packed.in_dim + block * 32);
                let d0 = unsafe { q8_0x4_block_dot_raw(weight, qbase) };
                let d1 = unsafe { q8_0x4_block_dot_raw(weight, qbase.add(packed.in_dim)) };
                let d2 = unsafe { q8_0x4_block_dot_raw(weight, qbase.add(2 * packed.in_dim)) };
                let d3 = unsafe { q8_0x4_block_dot_raw(weight, qbase.add(3 * packed.in_dim)) };
                let s = row * blocks + block;
                total0 = unsafe {
                    vfmaq_f32(
                        total0,
                        vcvtq_f32_s32(d0),
                        vmulq_f32(ws, widen_scale_dup(scales_f16[s])),
                    )
                };
                total1 = unsafe {
                    vfmaq_f32(
                        total1,
                        vcvtq_f32_s32(d1),
                        vmulq_f32(ws, widen_scale_dup(scales_f16[s + blocks])),
                    )
                };
                total2 = unsafe {
                    vfmaq_f32(
                        total2,
                        vcvtq_f32_s32(d2),
                        vmulq_f32(ws, widen_scale_dup(scales_f16[s + 2 * blocks])),
                    )
                };
                total3 = unsafe {
                    vfmaq_f32(
                        total3,
                        vcvtq_f32_s32(d3),
                        vmulq_f32(ws, widen_scale_dup(scales_f16[s + 3 * blocks])),
                    )
                };
            }
            unsafe {
                vst1q_f32(y.as_mut_ptr().add(row * packed.out_dim + group * 4), total0);
                vst1q_f32(
                    y.as_mut_ptr().add((row + 1) * packed.out_dim + group * 4),
                    total1,
                );
                vst1q_f32(
                    y.as_mut_ptr().add((row + 2) * packed.out_dim + group * 4),
                    total2,
                );
                vst1q_f32(
                    y.as_mut_ptr().add((row + 3) * packed.out_dim + group * 4),
                    total3,
                );
            }
            row += 4;
        }
        while row < rows {
            let mut total = unsafe { vdupq_n_f32(0.0) };
            for block in 0..blocks {
                let weight = &packed.blocks[group * blocks + block];
                let d = unsafe {
                    q8_0x4_block_dot_raw(
                        weight,
                        quants.as_ptr().add(row * packed.in_dim + block * 32),
                    )
                };
                let scale = unsafe {
                    vmulq_f32(
                        widen_scale4(weight.scales.as_ptr()),
                        widen_scale_dup(scales_f16[row * blocks + block]),
                    )
                };
                total = unsafe { vfmaq_f32(total, vcvtq_f32_s32(d), scale) };
            }
            unsafe { vst1q_f32(y.as_mut_ptr().add(row * packed.out_dim + group * 4), total) };
            row += 1;
        }
    }
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
    /// Q8_0 weights interleaved four rows at a time for lane-broadcast SDOT.
    Q8_0x4(Q8_0x4),
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

    /// Quantize and interleave four rows at a time for the SDOT-lane GEMV.
    pub fn from_f32_q8_0x4(tensor: Tensor) -> Option<Self> {
        Q8_0x4::from_f32(tensor).map(Matrix::Q8_0x4)
    }

    /// Name of the in-memory storage, for diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Matrix::F32(_) => "f32",
            Matrix::Bf16 { .. } => "bf16",
            Matrix::Q8_0 { .. } => "q8_0",
            Matrix::Q8_0x4(_) => "q8_0x4",
        }
    }

    /// Number of output rows.
    pub fn out_dim(&self) -> usize {
        match self {
            Matrix::F32(t) => t.shape()[0],
            Matrix::Bf16 { out_dim, .. } | Matrix::Q8_0 { out_dim, .. } => *out_dim,
            Matrix::Q8_0x4(q) => q.out_dim,
        }
    }

    /// Number of input columns.
    pub fn in_dim(&self) -> usize {
        match self {
            Matrix::F32(t) => t.shape()[1],
            Matrix::Bf16 { in_dim, .. } | Matrix::Q8_0 { in_dim, .. } => *in_dim,
            Matrix::Q8_0x4(q) => q.in_dim,
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
            Matrix::Q8_0x4(q) => (0..out_dim).flat_map(|r| q.row(r)).collect(),
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
            Matrix::Q8_0x4(q) => q.row(index),
        }
    }
}

/// Elements per quantization block in the Q8_0 layout.
pub const Q8_0_BLOCK: usize = 32;

/// Number of output rows interleaved in [`Q8_0x4`].
pub const Q8_0X4_ROWS: usize = 4;

/// Convert a finite f32 scale to IEEE binary16, round-to-nearest-even.
/// Q8_0 scales are non-negative and finite, but the converter handles zeros,
/// subnormals, and overflow too so it can be tested independently.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if (bits >> 23) & 0xff == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let significand = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let base = significand >> shift;
        let rem = significand & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let rounded = base + u32::from(rem > halfway || (rem == halfway && base & 1 != 0));
        return sign | rounded as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let base = mant >> 13;
    let rem = mant & 0x1fff;
    let rounded = base + u32::from(rem > 0x1000 || (rem == 0x1000 && base & 1 != 0));
    let (exp, mant) = if rounded == 0x400 {
        (exp + 1, 0)
    } else {
        (exp, rounded)
    };
    if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | mant as u16
    }
}

/// Fast conversion for finite normal Q8_0 scales. This is used for row
/// decoding and the portable fallback; the AArch64 kernel uses `fcvtl` on four
/// scales at once.
#[inline(always)]
fn f16_scale_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Widen four packed Q8_0 half scales with one `FCVTL` instruction.
///
/// SAFETY: the CPU must support FP16 SIMD, and `bits` must point to four
/// readable, normal finite half values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen_scale4(bits: *const u16) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::{vdupq_n_f32, vld1_u16};
    let half = unsafe { vld1_u16(bits) };
    let mut wide = unsafe { vdupq_n_f32(0.0) };
    // SAFETY: FP16 SIMD is checked by the dispatcher and both operands are
    // 128-bit NEON register values (the source occupies the low 64 bits).
    unsafe {
        core::arch::asm!(
            "fcvtl {wide:v}.4s, {half:v}.4h",
            wide = inout(vreg) wide,
            half = in(vreg) half,
            options(pure, nomem, nostack)
        );
    }
    wide
}

/// Widen one half scale and broadcast it to four f32 lanes with `dup` +
/// `fcvtl`, matching ggml's Q8_0 activation-scale path.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen_scale_dup(bits: u16) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::{vdup_n_u16, vdupq_n_f32};
    let half = unsafe { vdup_n_u16(bits) };
    let mut wide = unsafe { vdupq_n_f32(0.0) };
    // SAFETY: FP16 SIMD is checked by the dispatcher; operands are 4x16-bit
    // and 4x32-bit NEON vectors respectively.
    unsafe {
        core::arch::asm!(
            "fcvtl {wide:v}.4s, {half:v}.4h",
            wide = inout(vreg) wide,
            half = in(vreg) half,
            options(pure, nomem, nostack)
        );
    }
    wide
}

/// A vector quantized to Q8_0: one f32 scale per 32-element block plus the
/// int8 values. `quants` is always a whole number of blocks (the tail of a
/// partial block is zero-padded), which keeps the kernels branch-free.
pub struct Q8Vec {
    pub scales: Vec<f32>,
    /// ggml Q8_0 stores activation block scales as half. The row-wise kernel
    /// uses `scales`; the four-row kernel loads these compact bits and widens
    /// them in a vector instruction.
    pub scales_f16: Vec<u16>,
    pub quants: Vec<i8>,
}

impl Q8Vec {
    /// Quantize with the standard ggml rounding: the block scale is the largest
    /// magnitude divided by the largest int8, so the biggest value maps exactly
    /// to ±127 and everything else rounds to nearest.
    pub fn quantize(x: &[f32]) -> Self {
        let blocks = x.len().div_ceil(Q8_0_BLOCK);
        let mut scales = Vec::with_capacity(blocks);
        let mut scales_f16 = Vec::with_capacity(blocks);
        let mut quants = vec![0i8; blocks * Q8_0_BLOCK];
        for (block, chunk) in x.chunks(Q8_0_BLOCK).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            scales.push(d);
            scales_f16.push(f32_to_f16_bits(d));
            for (i, &v) in chunk.iter().enumerate() {
                quants[block * Q8_0_BLOCK + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Q8Vec {
            scales,
            scales_f16,
            quants,
        }
    }

    /// Quantize into an existing `Q8Vec`, reusing its allocations. The decode
    /// loop calls this for every projection, so growing the buffers once and
    /// refilling them avoids hundreds of allocations per token.
    pub fn quantize_into(&mut self, x: &[f32]) {
        let blocks = x.len().div_ceil(Q8_0_BLOCK);
        self.scales.clear();
        self.scales.reserve(blocks);
        self.scales_f16.clear();
        self.scales_f16.reserve(blocks);
        self.quants.clear();
        self.quants.resize(blocks * Q8_0_BLOCK, 0);
        for (block, chunk) in x.chunks(Q8_0_BLOCK).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            self.scales.push(d);
            self.scales_f16.push(f32_to_f16_bits(d));
            for (i, &v) in chunk.iter().enumerate() {
                self.quants[block * Q8_0_BLOCK + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}
