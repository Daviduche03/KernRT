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
