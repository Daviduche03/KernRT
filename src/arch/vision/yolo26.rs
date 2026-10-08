//! YOLO26 detection serving over the fixed-shape ONNX export.
//!
//! Ultralytics' NMS-free YOLO26 export has one input (`[1,3,H,W]`, f32 in
//! `0..1`) and one output (`[1,300,6]`: `x1,y1,x2,y2,score,class`). Keeping the
//! adapter at that contract makes the Rust side independent of PyTorch's
//! checkpoint serialization while still using the same exported graph as the
//! Torch and Ultralytics benchmark baselines.
//!
//! ## Environment flags
//!
//! - `KERNRT_NATIVE_CONV` — on by default, `=0` restores tract's convolution
//!   path. Replaces the convolutions tract can express with an im2col gather
//!   followed by KernRT's Accelerate GEMM, which avoids the strided output
//!   store tract's `OptMatMul` makes on every tile.
//! - `KERNRT_FAST_IM2COL` — on by default, `=0` restores the previous im2col
//!   shape (one full panel per layer, gathered even when the gather only
//!   copies its input). Exists so a before/after timing run can share one
//!   binary.
//! - `KERNRT_NATIVE_DEPTHWISE` — on by default, `=0` restores tract's
//!   `DepthWiseConv`. tract's f32 depthwise walks a zone visitor with a
//!   scalar inner loop; the native kernel does one shifted axpy per tap.
//! - `KERNRT_NATIVE_SILU` — on by default, `=0` restores tract's separate
//!   `Sigmoid` + `Mul` pair. The fused op writes `x * sigmoid(x)` straight
//!   into a fresh buffer, which skips the copy tract makes of the activation
//!   before running its in-place sigmoid kernel and the second full pass the
//!   unicast multiply makes afterwards.
//! - `KERNRT_NATIVE_MAXPOOL` — on by default, `=0` restores tract's
//!   `MaxPool`. tract's pool walks a zone visitor for each of its 25 taps; the
//!   native kernel splits the window into a horizontal pass and a vertical
//!   pass, ten element visits per output instead of twenty-five.
//! - `KERNRT_NATIVE_REDUCE` — on by default, `=0` restores tract's
//!   `Reduce<Max>`. The detection head reduces the class scores over the
//!   innermost axis, so the native kernel is a NEON `vmax` over contiguous
//!   rows where tract's `ndarray` reducer walks the same values one at a time.
//! - `KERNRT_NATIVE_UPSAMPLE` — on by default, `=0` disables the fused 2x
//!   nearest-neighbour upsample.
//! - `KERNRT_VERBOSE` — print node counts, the convolution breakdown, and
//!   the per-layer im2col/GEMM work.
//! - `KERNRT_TIMERS` — accumulate per-kernel nanoseconds and print the
//!   gather's GB/s and the GEMM's GFLOP/s every 50 inferences.

use anyhow::ensure;
use std::path::Path;
use std::sync::atomic::Ordering;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::{EvalOp, StaticName};
use tract_onnx::tract_hir::internal::DimLike;

use super::bench::{
    BIAS_BYTES, BIAS_NS, GEMM_FLOPS, GEMM_NS, IM2COL_BYTES, IM2COL_NS, MAXPOOL_NS, REDUCE_NS,
    RUN_NS, SILU_BYTES, SILU_NS, report_convs, report_head_ops, report_nodes, report_stragglers,
    timer_record_inference, timer_start, timer_stop,
};
use crate::ops::gemm_nn_into;

/// A single end-to-end YOLO26 prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

/// Loaded YOLO26 ONNX graph.
pub struct Yolo26 {
    model: TypedRunnableModel<TypedModel>,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone)]
struct KernUpsample2x {
    output_shape: Vec<usize>,
}

impl Op for KernUpsample2x {
    fn name(&self) -> StaticName {
        "KernRTUpsample2x".into()
    }

    tract_onnx::tract_core::op_as_typed_op!();
}

impl EvalOp for KernUpsample2x {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_array_view::<f32>()?;
        ensure!(input.ndim() == 4);
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let in_h = input.shape()[2];
        let in_w = input.shape()[3];
        let out_h = self.output_shape[2];
        let out_w = self.output_shape[3];
        let input = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("KernRTUpsample2x requires contiguous input"))?;
        let mut output = vec![0.0f32; batch * channels * out_h * out_w];
        for n in 0..batch {
            for c in 0..channels {
                for iy in 0..in_h {
                    let src = (n * channels + c) * in_h * in_w + iy * in_w;
                    let dst = (n * channels + c) * out_h * out_w + (iy * 2) * out_w;
                    let (top, bottom) = output[dst..dst + 2 * out_w].split_at_mut(out_w);
                    for ix in 0..in_w {
                        let value = input[src + ix];
                        let x = ix * 2;
                        top[x] = value;
                        top[x + 1] = value;
                    }
                    bottom.copy_from_slice(top);
                }
            }
        }
        Ok(tvec!(
            tract_ndarray::Array4::from_shape_vec((batch, channels, out_h, out_w), output)?
                .into_tensor()
                .into_tvalue()
        ))
    }
}

impl TypedOp for KernUpsample2x {
    tract_onnx::tract_core::as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(f32::fact(self.output_shape.clone())))
    }
}

/// `x * sigmoid(x)` evaluated in a single out-of-place pass.
///
/// The graph is a `Sigmoid` feeding a `Mul` with the sigmoid's own input.
/// [`rewrite_silu`] neutralises the `Sigmoid` to an identity, so both of this
/// node's inputs end up holding `x` and [`KernSiLU::eval`] only reads
/// `inputs[0]`. That removes the copy tract's element-wise eval makes of the
/// activation and the extra read/write pair the unicast multiply does.
#[derive(Debug, Clone)]
struct KernSiLU;

impl Op for KernSiLU {
    fn name(&self) -> StaticName {
        "KernRTSiLU".into()
    }

    tract_onnx::tract_core::op_as_typed_op!();
}

impl EvalOp for KernSiLU {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_array_view::<f32>()?;
        let shape = input.shape().to_vec();
        let input = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("KernRTSiLU requires contiguous input"))?;
        // Writing straight into the output tensor skips both a zero-filled
        // scratch vector and the copy `Tensor::from(Array)` makes from one.
        // SAFETY: `silu` stores every element of its output argument, so the
        // uninitialised memory is fully overwritten before the tensor can be
        // observed, and nothing else reads it in between.
        let mut tensor = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &shape)? };
        let output = tensor.as_slice_mut::<f32>()?;
        ensure!(output.len() == input.len());
        let started = timer_start();
        silu(input, output);
        timer_stop(started, &SILU_NS);
        if started.is_some() {
            // One pass reads every element and writes every element.
            SILU_BYTES.fetch_add(
                (2 * input.len() * size_of::<f32>()) as u64,
                Ordering::Relaxed,
            );
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for KernSiLU {
    tract_onnx::tract_core::as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        // The input may have been constant-folded; this node evaluates.
        fact.konst = None;
        Ok(tvec!(fact))
    }
}

/// `sigmoid` written as `1 / (1 + exp(-x))`, used for the scalar tail and as
/// the oracle for the vector kernel.
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(not(target_arch = "aarch64"))]
fn silu(input: &[f32], output: &mut [f32]) {
    for (out, &x) in output.iter_mut().zip(input) {
        *out = x * sigmoid_scalar(x);
    }
}

/// Degree-6 Taylor series for `exp(r)` on `|r| <= ln2 / 2`.
///
/// The seventh-order term `r^7/7!` is dropped: at the largest `|r|` the
/// reduction can produce (`ln2/2`) it is `1.2e-7`, which is inside the
/// `1e-7 + 1e-6*|w|` bar `silu_is_accurate_away_from_the_exponent_boundary`
/// holds the kernel to, and it saves one `fmadd` per four elements in the
/// innermost loop of the run. Degree 5 leaves `r^6/6! = 2.4e-6` at the same
/// `|r|`, which is past that bar, so 6 is as far as the budget allows.
///
/// Tract's own `arm64simd_sigmoid_f32_4n` does the same job in place; this is
/// the out-of-place shape the fused node needs. `n = round(x * log2(e))` can
/// land on the wrong side of a `.5` boundary when `x * log2(e)` carries its
/// usual rounding error, which puts `r` about one `ln2` away from where the
/// series was fitted: the two errors then cancel to leave roughly `1e-4`
/// relative error on the ~1-in-10^4 elements that hit it, and never a NaN or
/// a negative result. Correcting for it would cost more than the fusion
/// saves, and the perturbation is far below anything the detection output
/// can show.
#[cfg(target_arch = "aarch64")]
fn exp_f32x4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // `LN2_HI` is `0x3f31_8000`: with only trailing mantissa zeros,
    // `n * LN2_HI` stays exact for every `n` the reduction can produce, so
    // the two-step subtraction below leaves a faithful `r`.
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;

    unsafe {
        // Saturate the argument first: past here `sigmoid` is already 1 or 0
        // to f32 precision, and it keeps `x - n * ln2` from overflowing into
        // NaN when someone hands the kernel `f32::MAX`.
        let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(1000.0)), vdupq_n_f32(-1000.0));
        let n = vcvtnq_s32_f32(vmulq_n_f32(x, LOG2_E));
        let n_f = vcvtq_f32_s32(n);
        let r = vmlsq_n_f32(vmlsq_n_f32(x, n_f, LN2_HI), n_f, LN2_LO);
        let mut p = vdupq_n_f32(1.0 / 720.0);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), r, p);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), r, p);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), r, p);
        p = vfmaq_f32(vdupq_n_f32(0.5), r, p);
        p = vfmaq_f32(vdupq_n_f32(1.0), r, p);
        let poly = vfmaq_f32(vdupq_n_f32(1.0), r, p);
        let mut exp_bits = vaddq_s32(n, vdupq_n_s32(127));
        exp_bits = vminq_s32(vmaxq_s32(exp_bits, vdupq_n_s32(1)), vdupq_n_s32(254));
        let e = vmulq_f32(poly, vreinterpretq_f32_s32(vshlq_n_s32(exp_bits, 23)));
        // `n > 127` means `exp` left f32's range. Handing back the clamped
        // `2^127` there would leave `sigmoid` a hair above zero where
        // `f32::exp` overflows and it is exactly zero, and `x * sigmoid(x)`
        // has no room to absorb the difference for a large negative `x`.
        let over = vcgtq_s32(n, vdupq_n_s32(127));
        vbslq_f32(over, vdupq_n_f32(f32::INFINITY), e)
    }
}

/// The vector body of [`silu`], shared with [`silu_in_place`] and
/// [`silu_bias_in_place`] so the three can never drift apart.
///
/// `bias`, when it is not zero, is folded into the same pass:
/// `(x + bias) * sigmoid(x + bias)` needs `x + bias` twice, and adding it as a
/// separate scalar sweep over the activation costs a whole extra read and
/// write of every value the convolution just produced — for `yolo26n` that is
/// 16 M elements touched twice where once would do.
///
/// `input` and `output` may name the same buffer: each lane is read before it
/// is written and nothing else looks at a lane in between, which is exactly
/// what an elementwise function needs. That is why the arguments are raw
/// pointers rather than two slices — the borrow checker will not let one
/// alias the other.
///
/// SAFETY: `full` must be a multiple of four and both buffers must hold at
/// least `full` floats.
#[cfg(target_arch = "aarch64")]
unsafe fn silu_vectors(input: *const f32, output: *mut f32, full: usize, bias: f32) {
    use std::arch::aarch64::*;
    const LANES: usize = 4;
    debug_assert_eq!(full % LANES, 0);
    if full == 0 {
        return;
    }
    // Rust 2024 does not inherit `unsafe` into an `unsafe fn`'s body.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let bias_vec = vdupq_n_f32(bias);
        let biased = bias != 0.0;
        let mut i = 0;
        while i < full {
            // AArch64 permits unaligned accesses from ordinary memory.
            let x = vld1q_f32(input.add(i));
            // Measured against a `vrecpeq` + two `vrecpsq` corrections, the
            // divide is faster: the estimate needs a compare and a select to
            // survive the saturated `exp`, and LLVM already lowers this
            // particular `1.0 / x` well. That swap cost 4 ms of the run.
            let t = if biased { vaddq_f32(x, bias_vec) } else { x };
            let sig = vdivq_f32(one, vaddq_f32(one, exp_f32x4(vnegq_f32(t))));
            vst1q_f32(output.add(i), vmulq_f32(t, sig));
            i += LANES;
        }
    }
}

/// `out[i] = x[i] * sigmoid(x[i])`, reading `input` once and writing `output`
/// once.
#[cfg(target_arch = "aarch64")]
fn silu(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let full = input.len() - input.len() % 4;
    // SAFETY: `full` is `input.len()` rounded down to a whole vector, so both
    // bounds stay inside the slices, and they are distinct borrows of
    // different buffers.
    unsafe { silu_vectors(input.as_ptr(), output.as_mut_ptr(), full, 0.0) };
    for (out, &x) in output[full..].iter_mut().zip(&input[full..]) {
        *out = x * sigmoid_scalar(x);
    }
}

/// [`silu`] over a buffer it owns, which is what a convolution fused with its
/// activation needs: the GEMM has already written these lanes, so the result
/// belongs where they are rather than in a fresh tensor that would have to be
/// written to DRAM and read back again.
fn silu_in_place(values: &mut [f32]) {
    silu_bias_in_place(values, 0.0);
}

/// [`silu_in_place`] with the channel's bias folded into the same sweep, which
/// is what [`KernConv::activate_tile`] needs: the GEMM has already written
/// these lanes, and a separate `+= bias` pass over them would read and rewrite
/// every one a second time before the activation ever ran.
fn silu_bias_in_place(values: &mut [f32], bias: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        let full = values.len() - values.len() % 4;
        let input = values.as_ptr();
        let output = values.as_mut_ptr();
        // SAFETY: as above, and `input`/`output` naming one buffer is the
        // documented contract of `silu_vectors`.
        unsafe { silu_vectors(input, output, full, bias) };
        for value in &mut values[full..] {
            let t = *value + bias;
            *value = t * sigmoid_scalar(t);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        let t = *value + bias;
        *value = t * sigmoid_scalar(t);
    }
}

/// A group-1 NCHW convolution: one im2col pass, then a plain GEMM.
///
/// The panel is written as `[k, rows]`, which is the order OIHW weights already
/// use, so both GEMM operands arrive NoTrans. That has two consequences worth
/// the extra code: the large matrices are packed with contiguous reads, and the
/// result lands in the output's NCHW layout directly.
///
/// The pass is tiled over output rows. A whole-`hw` panel for the first layer
/// is 44 MB, which streams straight through the 12 MB L2 on its way between
/// the fill and the GEMM; a row tile sized by [`tile_rows_for`] stays resident,
/// and the output column range it covers is a strided submatrix the GEMM can
/// write into without a scatter. When the gather would be a byte-for-byte copy
/// of its input — every 1x1 layer — it is skipped entirely
/// ([`KernConv::gather_is_a_copy`]). `KERNRT_FAST_IM2COL` turns both off.
///
/// tract's convolution decomposes into a materialized `Pad`, a panel
/// regenerated per output tile, and a packed store whose write stride is
/// `out_h * out_w`. That strided store was roughly 45% of the profile.
#[derive(Debug, Clone)]
struct KernConv {
    weights: Vec<f32>,
    bias: Vec<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    input_channels: usize,
    output_channels: usize,
    output_h: usize,
    output_w: usize,
    /// Output rows per im2col+GEMM call, from [`tile_rows_for`].
    tile_rows: usize,
    /// Whether the fast im2col strategy is on, read by [`fast_im2col`] once at
    /// load rather than once per layer per inference.
    fast_im2col: bool,
    /// Fold the following SiLU into this node: bias and activation over each
    /// tile as soon as the GEMM writes it, instead of a bias pass over the
    /// whole activation at the end. Set only by [`fuse_conv_silu`]. The
    /// activation then reads output written microseconds earlier, while it is
    /// still dirty in L2, instead of a tensor every intervening node has
    /// already touched — that read is what costs 10 GiB/s when it comes from
    /// DRAM.
    apply_silu: bool,
}

/// Whether to use the fast im2col strategy (row tiles sized for L2, and no
/// gather at all when the gather would only copy its input).
///
/// `KERNRT_FAST_IM2COL=0` restores the single-panel-per-layer gather the kernel
/// had before, so a before/after timing run can share one binary.
fn fast_im2col() -> bool {
    std::env::var_os("KERNRT_FAST_IM2COL").is_none_or(|value| value != "0")
}

/// Output rows per tile, sized so one panel tile fits in L2.
///
/// The budget is deliberately well under the 12 MB L2: the tile shares it with
/// the input band being gathered from, the weights, and the output being
/// written.
fn tile_rows_for(fast: bool, reduction: usize, output_w: usize, output_h: usize) -> usize {
    const TILE_BYTES: usize = 2 << 20;
    let output_h = output_h.max(1);
    if !fast {
        return output_h;
    }
    let per_row = reduction * output_w * size_of::<f32>();
    (TILE_BYTES / per_row.max(1)).clamp(1, output_h)
}

/// The parts of a tract `Conv` both native kernels share: the constants, the
/// geometry, and the output shape.
///
/// Grouping is deliberately not resolved here. A plain convolution and a
/// depthwise one differ only in how they read `input_channels`, so each
/// kernel checks `Conv::group` itself and then takes the same extraction.
#[derive(Debug)]
struct ConvParams {
    weights: Vec<f32>,
    bias: Vec<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    input_channels: usize,
    output_channels: usize,
    output_h: usize,
    output_w: usize,
}

impl ConvParams {
    /// The group-1 im2col+GEMM kernel.
    fn into_conv(self, fast: bool) -> KernConv {
        let reduction = self.input_channels * self.kernel_h * self.kernel_w;
        let tile_rows = tile_rows_for(fast, reduction, self.output_w, self.output_h);
        KernConv {
            weights: self.weights,
            bias: self.bias,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            pad_h: self.pad_h,
            pad_w: self.pad_w,
            dilation_h: self.dilation_h,
            dilation_w: self.dilation_w,
            input_channels: self.input_channels,
            output_channels: self.output_channels,
            output_h: self.output_h,
            output_w: self.output_w,
            tile_rows,
            fast_im2col: fast,
            apply_silu: false,
        }
    }

    /// The depthwise kernel, or `None` when the channel counts say this is a
    /// grouped convolution that mixes channels rather than a depthwise one.
    fn into_depthwise(self) -> Option<KernDepthWise> {
        if self.input_channels != self.output_channels {
            return None;
        }
        Some(KernDepthWise {
            weights: self.weights,
            bias: self.bias,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            stride_h: self.stride_h,
            stride_w: self.stride_w,
            pad_h: self.pad_h,
            pad_w: self.pad_w,
            dilation_h: self.dilation_h,
            dilation_w: self.dilation_w,
            channels: self.input_channels,
            output_h: self.output_h,
            output_w: self.output_w,
        })
    }
}

/// Read a typed `Conv` into [`ConvParams`], or return `None` for anything the
/// native kernels do not model: quantized parameters, a non-OIHW kernel, a
/// layout other than NCHW, a kernel rank other than 2, a non-f32 input, a
/// weight or bias that is still behind a Source node, or asymmetric padding
/// (which would need two different offsets per axis).
///
/// The caller must run `prop_consts` first — the weights are usually still
/// behind a folding candidate straight out of `into_typed`.
fn conv_params(
    model: &TypedModel,
    node: &TypedNode,
    conv: &tract_onnx::tract_core::ops::cnn::conv::Conv,
) -> TractResult<Option<ConvParams>> {
    use tract_onnx::tract_core::ops::cnn::conv::KernelFormat;
    use tract_onnx::tract_core::ops::nn::DataFormat;

    if conv.q_params.is_some()
        || conv.kernel_fmt != KernelFormat::OIHW
        || conv.pool_spec.data_format != DataFormat::NCHW
        || conv.pool_spec.kernel_shape.len() != 2
    {
        return Ok(None);
    }
    let input = model.outlet_fact(node.inputs[0])?;
    if input.datum_type != f32::datum_type() || input.shape.len() != 4 {
        return Ok(None);
    }
    let Some(weight) = model.outlet_fact(node.inputs[1])?.konst.as_ref() else {
        return Ok(None);
    };
    let weights = weight
        .to_array_view::<f32>()?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let bias = match node.inputs.get(2) {
        Some(outlet) => {
            let Some(bias) = model.outlet_fact(*outlet)?.konst.as_ref() else {
                return Ok(None);
            };
            bias.to_array_view::<f32>()?
                .iter()
                .copied()
                .collect::<Vec<_>>()
        }
        None => Vec::new(),
    };
    let shape = input.shape.to_vec();
    let padding = conv
        .pool_spec
        .computed_padding(&[shape[2].clone(), shape[3].clone()]);
    if padding[0].pad_before != padding[0].pad_after
        || padding[1].pad_before != padding[1].pad_after
    {
        return Ok(None);
    }
    let (Ok(pad_h), Ok(pad_w)) = (
        padding[0].pad_before.to_usize(),
        padding[1].pad_before.to_usize(),
    ) else {
        return Ok(None);
    };
    let (Ok(output_h), Ok(output_w)) = (
        padding[0].convoluted.to_usize(),
        padding[1].convoluted.to_usize(),
    ) else {
        return Ok(None);
    };
    let kernel = conv.pool_spec.kernel_shape.clone();
    Ok(Some(ConvParams {
        weights,
        bias,
        kernel_h: kernel[0],
        kernel_w: kernel[1],
        stride_h: conv.pool_spec.stride(0),
        stride_w: conv.pool_spec.stride(1),
        pad_h,
        pad_w,
        dilation_h: conv.pool_spec.dilation(0),
        dilation_w: conv.pool_spec.dilation(1),
        input_channels: conv.input_channels(),
        output_channels: conv.output_channels(),
        output_h,
        output_w,
    }))
}

impl KernConv {
    /// Numeric core of the op: row-tiled im2col into a `[reduction, rows]`
    /// panel, one NoTrans GEMM per tile whose result lands directly in the
    /// NCHW output, then bias over the whole activation. When
    /// [`Self::gather_is_a_copy`] the panel is skipped and the input goes
    /// straight to the GEMM.
    ///
    /// Kept separate from [`EvalOp::eval`] so tests can drive it without
    /// building a tract graph.
    #[cfg(test)]
    fn run(&self, input: &[f32], batch: usize, in_h: usize, in_w: usize) -> Vec<f32> {
        self.run_tiled(input, batch, in_h, in_w, self.tile_rows)
    }

    /// [`Self::run`] with the tile height chosen by the caller, so tests can
    /// walk the tile boundaries a shape actually has.
    #[cfg(test)]
    fn run_tiled(
        &self,
        input: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        rows: usize,
    ) -> Vec<f32> {
        // Zeroed here rather than in `run_into`: the kernel overwrites every
        // element, so a production caller can hand it uninitialised memory,
        // while a test looking for unwritten elements needs the zeros.
        let mut output = vec![0.0f32; batch * self.output_channels * self.output_h * self.output_w];
        self.run_into(input, &mut output, batch, in_h, in_w, rows);
        output
    }

    /// Bias plus SiLU, in place, over the column range the last GEMM wrote.
    /// Only reached when [`Self::apply_silu`] is set; otherwise the bias is
    /// added once over the whole activation after every tile is done.
    ///
    /// The two steps are one sweep rather than two: a separate `+= bias` pass
    /// would read and rewrite every value the GEMM just produced before the
    /// activation touched it, and `silu_bias_in_place` needs the biased value
    /// twice anyway.
    fn activate_tile(&self, out: &mut [f32], spatial: usize, offset: usize, columns: usize) {
        for (channel, row) in out.chunks_exact_mut(spatial).enumerate() {
            let bias = self.bias.get(channel).copied().unwrap_or(0.0);
            let tile = &mut row[offset..offset + columns];
            if bias == 0.0 {
                silu_in_place(tile);
            } else {
                silu_bias_in_place(tile, bias);
            }
        }
    }

    /// [`Self::run_tiled`] against a caller-owned buffer holding at least
    /// `batch * filters * output_h * output_w` values, which nothing reads
    /// first: the GEMM writes with `beta = 0` across every row and column and
    /// the bias pass only adds to what the GEMM wrote. That is what lets
    /// [`EvalOp::eval`] skip both the zero-fill and the copy
    /// `Tensor::from(Array)` makes from a scratch vector.
    fn run_into(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        rows: usize,
    ) {
        let rows = rows.clamp(1, self.output_h.max(1));
        let spatial = self.output_h * self.output_w;
        let reduction = self.input_channels * self.kernel_h * self.kernel_w;
        let filters = self.output_channels;
        let frame = self.input_channels * in_h * in_w;
        debug_assert!(output.len() >= batch * filters * spatial);

        for b in 0..batch {
            let input = &input[b * frame..(b + 1) * frame];
            if self.fast_im2col && self.gather_is_a_copy() {
                // The panel and the tensor are the same `[cin, hw]` buffer, so
                // the gather would rewrite the input byte for byte. Hand the
                // input to the GEMM instead — one call per batch, so the
                // weights are packed once rather than once per row band.
                debug_assert_eq!(input.len(), reduction * spatial);
                let started = timer_start();
                gemm_nn_into(
                    &self.weights,
                    input,
                    &mut output[b * filters * spatial..],
                    spatial,
                    filters,
                    reduction,
                    spatial,
                );
                timer_stop(started, &GEMM_NS);
                if started.is_some() {
                    GEMM_FLOPS.fetch_add(
                        (2 * filters * reduction * spatial) as u64,
                        Ordering::Relaxed,
                    );
                }
                if self.apply_silu {
                    let started = timer_start();
                    self.activate_tile(
                        &mut output[b * filters * spatial..(b + 1) * filters * spatial],
                        spatial,
                        0,
                        spatial,
                    );
                    timer_stop(started, &SILU_NS);
                    if started.is_some() {
                        SILU_BYTES.fetch_add(
                            (2 * filters * spatial * size_of::<f32>()) as u64,
                            Ordering::Relaxed,
                        );
                    }
                }
            } else {
                with_im2col_panel(reduction * rows * self.output_w, |panel| {
                    for oy0 in (0..self.output_h).step_by(rows) {
                        let oy1 = (oy0 + rows).min(self.output_h);
                        let columns = (oy1 - oy0) * self.output_w;
                        let offset = oy0 * self.output_w;
                        let started = timer_start();
                        self.im2col(
                            input,
                            in_h,
                            in_w,
                            oy0,
                            oy1,
                            &mut panel[..reduction * columns],
                        );
                        timer_stop(started, &IM2COL_NS);
                        if started.is_some() {
                            IM2COL_BYTES.fetch_add(
                                (reduction * columns * size_of::<f32>()) as u64,
                                Ordering::Relaxed,
                            );
                        }
                        let started = timer_start();
                        gemm_nn_into(
                            &self.weights,
                            &panel[..reduction * columns],
                            &mut output[b * filters * spatial + offset..],
                            spatial,
                            filters,
                            reduction,
                            columns,
                        );
                        timer_stop(started, &GEMM_NS);
                        if started.is_some() {
                            GEMM_FLOPS.fetch_add(
                                (2 * filters * reduction * columns) as u64,
                                Ordering::Relaxed,
                            );
                        }
                        if self.apply_silu {
                            let started = timer_start();
                            self.activate_tile(
                                &mut output[b * filters * spatial..(b + 1) * filters * spatial],
                                spatial,
                                offset,
                                columns,
                            );
                            timer_stop(started, &SILU_NS);
                            if started.is_some() {
                                SILU_BYTES.fetch_add(
                                    (2 * filters * columns * size_of::<f32>()) as u64,
                                    Ordering::Relaxed,
                                );
                            }
                        }
                    }
                });
            }
            if !self.apply_silu {
                let started = timer_start();
                let out = &mut output[b * filters * spatial..(b + 1) * filters * spatial];
                for (channel, row) in out.chunks_exact_mut(spatial).enumerate() {
                    let bias = self.bias.get(channel).copied().unwrap_or(0.0);
                    for value in row {
                        *value += bias;
                    }
                }
                timer_stop(started, &BIAS_NS);
                if started.is_some() {
                    // The loop reads and rewrites the whole tensor, so count both.
                    BIAS_BYTES
                        .fetch_add((2 * out.len() * size_of::<f32>()) as u64, Ordering::Relaxed);
                }
            }
        }
    }

    /// True when [`Self::im2col`] would reproduce its input exactly: a 1x1
    /// kernel with stride 1, no padding and no dilation shifts nothing, so
    /// `[cin, hw]` is both the input and the panel. That is 55 of the 102
    /// convolutions in this model, and gathering them costs a full copy of the
    /// tensor for no change of layout.
    fn gather_is_a_copy(&self) -> bool {
        self.kernel_h == 1
            && self.kernel_w == 1
            && self.stride_h == 1
            && self.stride_w == 1
            && self.pad_h == 0
            && self.pad_w == 0
            && self.dilation_h == 1
            && self.dilation_w == 1
    }

    /// Build the native op from a typed `Conv`, or return `None` for a shape
    /// the kernel does not handle (grouped, quantized, non-OIHW, or a weight
    /// that is not constant).
    fn from_conv(
        model: &TypedModel,
        node: &TypedNode,
        conv: &tract_onnx::tract_core::ops::cnn::conv::Conv,
    ) -> TractResult<Option<Self>> {
        if conv.group != 1 {
            return Ok(None);
        }
        let Some(params) = conv_params(model, node, conv)? else {
            return Ok(None);
        };
        Ok(Some(params.into_conv(fast_im2col())))
    }

    /// Fill `panel` as `[k, (oy1 - oy0) * out_w]`, for output rows
    /// `[oy0, oy1)`, with the reduction index ordered
    /// `c * kernel_h * kernel_w + ky * kernel_w + kx` to match OIHW weights.
    ///
    /// Padding is applied while gathering, so the padded tensor `tract`
    /// materializes as a separate node never exists. Restricting the range to
    /// a few output rows is what keeps the tile small enough to gather from
    /// and consume without leaving cache: the input band those rows need is
    /// only `(rows * stride + kernel) * in_w` wide.
    fn im2col(
        &self,
        frame: &[f32],
        in_h: usize,
        in_w: usize,
        oy0: usize,
        oy1: usize,
        panel: &mut [f32],
    ) {
        let out_w = self.output_w;
        let rows = oy1 - oy0;
        let tile = rows * out_w;
        let plane = in_h * in_w;
        let step_h = self.stride_h as isize;
        let mut p = 0usize;
        for c in 0..self.input_channels {
            let base = c * plane;
            for ky in 0..self.kernel_h {
                // `iy(oy) = oy * step_h + iy0`; everything below is a range of
                // `oy` where that is inside `[0, in_h)`, which is the same for
                // every `kx` and every channel.
                let iy0 = (ky * self.dilation_h) as isize - self.pad_h as isize;
                let oy_lo = ceil_div(-iy0, step_h);
                let oy_hi = ceil_div(in_h as isize - iy0, step_h);
                for kx in 0..self.kernel_w {
                    let dst = &mut panel[p * tile..(p + 1) * tile];
                    // Rows outside `[oy_lo, oy_hi)` are pure padding: the whole
                    // output row is zero, so it is two fills instead of one per
                    // row.
                    let in_lo = oy_lo.max(oy0).min(oy1);
                    let in_hi = oy_hi.max(oy0).min(oy1).max(in_lo);
                    if in_lo > oy0 {
                        dst[..(in_lo - oy0) * out_w].fill(0.0);
                    }
                    if in_hi < oy1 {
                        dst[(in_hi - oy0) * out_w..].fill(0.0);
                    }
                    let shift = (kx * self.dilation_w) as isize - self.pad_w as isize;
                    let src_rows = &mut dst[(in_lo - oy0) * out_w..(in_hi - oy0) * out_w];
                    if self.stride_w == 1 {
                        // `ix(ox)` steps by one, so the in-bounds window is one
                        // contiguous run. `lead`, `take` and `window` depend only
                        // on `kx`, so they move out of the row loop; when there
                        // is no horizontal padding at all the row is a single
                        // copy. This is every stride-1 layer, which is nearly all
                        // of them.
                        let lead = (if shift < 0 { -shift } else { 0 }).min(out_w as isize);
                        let lead = lead as usize;
                        let avail = (in_w as isize - shift).max(0) as usize;
                        let take = avail.saturating_sub(lead).min(out_w - lead);
                        let window = (shift + lead as isize) as usize;
                        let tail = out_w - lead - take;
                        let n = in_hi - in_lo;
                        // `step_h == 1` puts consecutive `oy` on consecutive
                        // input rows and `in_w == out_w` gives those rows the
                        // same stride as the tile, so every row of the tile
                        // reads the same run of the frame and the tile is one
                        // copy instead of one per row. What a single copy cannot
                        // carry is the padding: `lead` columns at the front of
                        // every row and `tail` at the back, which is a store per
                        // row. Here `lead` and `tail` are 0 or 1, so it is one
                        // store — `fill` on a one-element slice is a `memset`
                        // call, and there is one of those per row per tap.
                        //
                        // `want - span <= tail` is the bound check: a span cut
                        // short by the end of the frame may leave only the last
                        // row unwritten, and only as far as that row's padding,
                        // because that is exactly what the zeroing writes.
                        let mut spanned = false;
                        if step_h == 1 && in_w == out_w && n > 0 {
                            // `in_lo >= oy_lo >= -iy0`, so this is never negative.
                            let src = base + (in_lo as isize + iy0) as usize * in_w + window;
                            let want = n * out_w - lead;
                            let span = want.min(frame.len().saturating_sub(src));
                            if want - span <= tail {
                                spanned = true;
                                if span > 0 {
                                    src_rows[lead..lead + span]
                                        .copy_from_slice(&frame[src..src + span]);
                                }
                                if lead == 1 {
                                    for r in 0..n {
                                        src_rows[r * out_w] = 0.0;
                                    }
                                } else if lead > 0 {
                                    for r in 0..n {
                                        src_rows[r * out_w..r * out_w + lead].fill(0.0);
                                    }
                                }
                                if tail == 1 {
                                    for r in 0..n {
                                        src_rows[r * out_w + out_w - 1] = 0.0;
                                    }
                                } else if tail > 0 {
                                    for r in 0..n {
                                        src_rows[r * out_w + take..(r + 1) * out_w].fill(0.0);
                                    }
                                }
                            }
                        }
                        if !spanned {
                            if lead == 0 && take == out_w {
                                for (index, oy) in (in_lo..in_hi).enumerate() {
                                    let iy = (oy as isize * step_h + iy0) as usize;
                                    let src = base + iy * in_w + window;
                                    src_rows[index * out_w..(index + 1) * out_w]
                                        .copy_from_slice(&frame[src..src + out_w]);
                                }
                            } else if lead == 0 {
                                for (index, oy) in (in_lo..in_hi).enumerate() {
                                    let iy = (oy as isize * step_h + iy0) as usize;
                                    let src = base + iy * in_w;
                                    let row = &mut src_rows[index * out_w..(index + 1) * out_w];
                                    row[..take]
                                        .copy_from_slice(&frame[src + window..src + window + take]);
                                    if tail == 1 {
                                        row[take] = 0.0;
                                    } else if tail > 0 {
                                        row[take..].fill(0.0);
                                    }
                                }
                            } else {
                                for (index, oy) in (in_lo..in_hi).enumerate() {
                                    let iy = (oy as isize * step_h + iy0) as usize;
                                    let src = base + iy * in_w;
                                    let row = &mut src_rows[index * out_w..(index + 1) * out_w];
                                    if lead == 1 {
                                        row[0] = 0.0;
                                        row[1..lead + take].copy_from_slice(
                                            &frame[src + window..src + window + take],
                                        );
                                    } else {
                                        row[..lead].fill(0.0);
                                        row[lead..lead + take].copy_from_slice(
                                            &frame[src + window..src + window + take],
                                        );
                                    }
                                    if tail == 1 {
                                        row[lead + take] = 0.0;
                                    } else if tail > 0 {
                                        row[lead + take..].fill(0.0);
                                    }
                                }
                            }
                        }
                    } else {
                        // Strided: `ix(ox) = ox * stride_w + shift` is still
                        // monotonic, so the in-bounds column range is one span
                        // per row and only that span needs a per-element load.
                        let ox_lo = ceil_div(-shift, self.stride_w as isize).min(out_w);
                        let ox_hi = ceil_div(in_w as isize - shift, self.stride_w as isize)
                            .min(out_w)
                            .max(ox_lo);
                        for (index, oy) in (in_lo..in_hi).enumerate() {
                            let iy = (oy as isize * step_h + iy0) as usize;
                            let src = base + iy * in_w;
                            let row = &mut src_rows[index * out_w..(index + 1) * out_w];
                            row[..ox_lo].fill(0.0);
                            for (offset, ox) in (ox_lo..ox_hi).enumerate() {
                                row[ox_lo + offset] = frame
                                    [src + (ox as isize * self.stride_w as isize + shift) as usize];
                            }
                            row[ox_hi..].fill(0.0);
                        }
                    }
                    p += 1;
                }
            }
        }
    }
}

/// Smallest `n >= 0` with `n * den >= num`, or `0` when `num <= 0`. Used to
/// turn the two inequalities that put an input row inside the frame into `oy`
/// bounds, so the branch they represent is evaluated once per `(ky, kx)`
/// instead of once per output row.
fn ceil_div(num: isize, den: isize) -> usize {
    debug_assert!(den > 0);
    if num <= 0 {
        0
    } else {
        ((num + den - 1) / den) as usize
    }
}

impl Op for KernConv {
    fn name(&self) -> StaticName {
        "KernRTConv".into()
    }

    tract_onnx::tract_core::op_as_typed_op!();
}

impl EvalOp for KernConv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_array_view::<f32>()?;
        ensure!(input.ndim() == 4);
        let batch = input.shape()[0];
        let in_h = input.shape()[2];
        let in_w = input.shape()[3];
        let input = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("KernRTConv requires a contiguous input"))?;

        // Writing straight into the output tensor skips both a zero-filled
        // scratch vector and the copy `Tensor::from(Array)` makes from one.
        // SAFETY: `run_into` stores every element of its output argument — the
        // GEMM writes with `beta = 0` over all `filters * spatial` values of
        // each batch and the bias pass reads only what the GEMM wrote — so the
        // uninitialised memory is fully overwritten before the tensor can be
        // observed, and nothing else reads it in between.
        let shape = vec![batch, self.output_channels, self.output_h, self.output_w];
        let mut tensor = unsafe { Tensor::uninitialized_dt(f32::datum_type(), &shape)? };
        let output = tensor.as_slice_mut::<f32>()?;
        self.run_into(input, output, batch, in_h, in_w, self.tile_rows);
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for KernConv {
    tract_onnx::tract_core::as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let batch = inputs
            .first()
            .and_then(|fact| fact.shape.iter().next().cloned())
            .and_then(|dim| dim.to_usize().ok())
            .unwrap_or(1);
        Ok(tvec!(f32::fact(vec![
            batch,
            self.output_channels,
            self.output_h,
            self.output_w
        ])))
    }
}

/// A depthwise NCHW convolution: one independent kernel per channel, so there
/// is no reduction axis to GEMM over.
///
/// tract's `DepthWise` walks a `Zone` visitor whose f32 inner loop is
/// `inner_loop_generic` — it yields one `(kernel index, input offset)` pair
/// per tap for every output element. On this model that node is about 13% of
/// the profile for 13 MFLOP of arithmetic, so the cost is entirely the
/// per-element iterator.
///
/// The structure here is the one a depthwise actually has: each of the
/// `kernel_h * kernel_w` taps contributes `weight * <input row>` to one output
/// row. With stride 1 across x that contribution is a contiguous shifted
/// [`axpy`], so an output row is touched nine times (once per tap) instead of
/// nine iterator steps per element — and those touches land on a row that
/// never leaves L1, because the next output row overlaps it.
#[derive(Debug, Clone)]
struct KernDepthWise {
    weights: Vec<f32>,
    bias: Vec<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    channels: usize,
    output_h: usize,
    output_w: usize,
}

/// `dst[i] += weight * src[i]` over equal-length contiguous slices.
#[cfg(not(target_arch = "aarch64"))]
fn axpy(dst: &mut [f32], weight: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += weight * *s;
    }
}

/// [`axpy`] on AArch64: four elements per instruction, scalar tail.
///
/// Hand-written instead of left to the auto-vectorizer, matching
/// [`crate::ops`]' stance on the micro-kernels: the elements are independent,
/// so the only thing to get right is that every vector access stays inside
/// both slices.
#[cfg(target_arch = "aarch64")]
#[inline]
fn axpy(dst: &mut [f32], weight: f32, src: &[f32]) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0usize;
    if n >= 4 {
        // SAFETY: `n >= 4` makes the first load legal and the loop condition
        // `i + 4 <= n` keeps every subsequent 16-byte access inside both
        // slices. `dst` and `src` are distinct borrows, so reading `dst` while
        // writing it back cannot observe a partial vector. AArch64 permits
        // unaligned accesses from ordinary memory.
        unsafe {
            let w = vdupq_n_f32(weight);
            let full = n - n % 4;
            while i < full {
                let acc = vld1q_f32(dst.as_ptr().add(i));
                let v = vld1q_f32(src.as_ptr().add(i));
                vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(acc, w, v));
                i += 4;
            }
        }
    }
    for (d, s) in dst[i..].iter_mut().zip(&src[i..]) {
        *d += weight * *s;
    }
}

impl KernDepthWise {
    /// Numeric core of the op, kept separate from [`EvalOp::eval`] so tests
    /// can drive it without building a tract graph.
    fn run(&self, input: &[f32], batch: usize, in_h: usize, in_w: usize) -> Vec<f32> {
        let plane = in_h * in_w;
        let out_plane = self.output_h * self.output_w;
        let mut output = vec![0.0f32; batch * self.channels * out_plane];
        for b in 0..batch {
            let src = &input[b * self.channels * plane..(b + 1) * self.channels * plane];
            let dst =
                &mut output[b * self.channels * out_plane..(b + 1) * self.channels * out_plane];
            for channel in 0..self.channels {
                let taps = self.kernel_h * self.kernel_w;
                let start = channel * taps;
                self.run_channel(
                    &src[channel * plane..(channel + 1) * plane],
                    in_h,
                    in_w,
                    &mut dst[channel * out_plane..(channel + 1) * out_plane],
                    &self.weights[start..start + taps],
                    self.bias.get(channel).copied().unwrap_or(0.0),
                );
            }
        }
        output
    }

    /// Convolve one channel plane into `dst`, which is filled with `bias`
    /// first and then accumulates every tap.
    ///
    /// Every output row is accumulated in place, one shifted slice per tap.
    /// The y coordinate is resolved per tap rather than per element, so an
    /// input row that falls outside the frame skips all `kernel_w` taps at
    /// once.
    fn run_channel(
        &self,
        src: &[f32],
        in_h: usize,
        in_w: usize,
        dst: &mut [f32],
        kernel: &[f32],
        bias: f32,
    ) {
        let out_w = self.output_w;
        for oy in 0..self.output_h {
            let out = &mut dst[oy * out_w..(oy + 1) * out_w];
            out.fill(bias);
            for ky in 0..self.kernel_h {
                let iy = (oy * self.stride_h + ky * self.dilation_h) as isize - self.pad_h as isize;
                if !(0..in_h as isize).contains(&iy) {
                    continue;
                }
                let row = &src[iy as usize * in_w..iy as usize * in_w + in_w];
                for kx in 0..self.kernel_w {
                    let weight = kernel[ky * self.kernel_w + kx];
                    let shift = (kx * self.dilation_w) as isize - self.pad_w as isize;
                    if self.stride_w == 1 {
                        // `ix(ox) = ox + shift` steps by one, so the in-bounds
                        // output columns are a single range and their inputs
                        // are the same range slid by `shift`.
                        let lo = (-shift).max(0);
                        let hi = (in_w as isize - shift).min(out_w as isize).max(0);
                        if lo >= hi {
                            continue;
                        }
                        let lo = lo as usize;
                        let hi = hi as usize;
                        let from = (lo as isize + shift) as usize;
                        axpy(&mut out[lo..hi], weight, &row[from..from + (hi - lo)]);
                    } else {
                        for (ox, value) in out.iter_mut().enumerate() {
                            let ix = (ox * self.stride_w + kx * self.dilation_w) as isize
                                - self.pad_w as isize;
                            if (0..in_w as isize).contains(&ix) {
                                *value += weight * row[ix as usize];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Build the native op from a typed `Conv`, or return `None` unless the
    /// grouping really is depthwise: one kernel per channel, with as many
    /// output channels as input ones.
    fn from_conv(
        model: &TypedModel,
        node: &TypedNode,
        conv: &tract_onnx::tract_core::ops::cnn::conv::Conv,
    ) -> TractResult<Option<Self>> {
        let grouped = conv.group != 1;
        let channels = conv.input_channels();
        if !grouped || conv.group != channels || conv.output_channels() != channels {
            return Ok(None);
        }
        let Some(params) = conv_params(model, node, conv)? else {
            return Ok(None);
        };
        Ok(params.into_depthwise())
    }
}

impl Op for KernDepthWise {
    fn name(&self) -> StaticName {
        "KernRTDepthWise".into()
    }

    tract_onnx::tract_core::op_as_typed_op!();
}

impl EvalOp for KernDepthWise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_array_view::<f32>()?;
        ensure!(input.ndim() == 4);
        let batch = input.shape()[0];
        let in_h = input.shape()[2];
        let in_w = input.shape()[3];
        let input = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("KernRTDepthWise requires a contiguous input"))?;

        let output = self.run(input, batch, in_h, in_w);
        Ok(tvec!(
            tract_ndarray::Array4::from_shape_vec(
                (batch, self.channels, self.output_h, self.output_w),
                output,
            )?
            .into_tensor()
            .into_tvalue()
        ))
    }
}

impl TypedOp for KernDepthWise {
    tract_onnx::tract_core::as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let batch = inputs
            .first()
            .and_then(|fact| fact.shape.iter().next().cloned())
            .and_then(|dim| dim.to_usize().ok())
            .unwrap_or(1);
        Ok(tvec!(f32::fact(vec![
            batch,
            self.channels,
            self.output_h,
            self.output_w
        ])))
    }
}

/// A stride-1 pool whose window is centred: `2 * pad + 1` wide with the same
/// padding on every side, so the output has the input's spatial shape.
///
/// Max pooling is separable and the intermediate is never wanted, so the
/// kernel runs it as two passes — a sliding window along each row of the
/// input, then a sliding window down the rows of that result. Padding
/// contributes nothing on either pass, which is how ONNX treats a `MaxPool`
/// pad.
#[derive(Debug, Clone)]
struct KernMaxPool {
    channels: usize,
    in_h: usize,
    in_w: usize,
    pad: usize,
}

impl KernMaxPool {
    /// Numeric core of the op, kept separate from [`EvalOp::eval`] so tests
    /// can drive it without building a tract graph.
    fn run(&self, input: &[f32], batch: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), batch * self.channels * self.in_h * self.in_w);
        let plane = self.in_h * self.in_w;
        let mut output = vec![0.0f32; batch * self.channels * plane];
        // One plane at a time, so the horizontal pass only ever needs a
        // single plane of scratch: pooling does not change the shape.
        let mut rows = vec![0.0f32; plane];
        for (dst, src) in output
            .chunks_exact_mut(plane)
            .zip(input.chunks_exact(plane))
        {
            sliding_max(src, &mut rows, self.in_h, self.in_w, self.pad);
            fold_rows(&rows, dst, self.in_h, self.in_w, self.pad);
        }
        output
    }

    /// Build the native op from a typed `MaxPool`, or `None` unless the pool
    /// is a centred, stride-1, undilated reduce of a contiguous NCHW f32
    /// tensor.
    fn from_maxpool(model: &TypedModel, node: &TypedNode) -> Option<KernMaxPool> {
        use tract_onnx::tract_core::ops::cnn::{MaxPool, PaddingSpec};
        use tract_onnx::tract_core::ops::nn::DataFormat;

        let pool = node.op_as::<MaxPool>()?;
        if pool.with_index_outputs.is_some() {
            return None;
        }
        let spec = &pool.pool_spec;
        if spec.data_format != DataFormat::NCHW
            || spec.rank() != 2
            || spec.stride(0) != 1
            || spec.stride(1) != 1
            || spec.dilation(0) != 1
            || spec.dilation(1) != 1
        {
            return None;
        }
        let kernel = spec.kernel_shape[0];
        if kernel != spec.kernel_shape[1] || kernel % 2 == 0 {
            return None;
        }
        let pad = (kernel - 1) / 2;
        let centred = |before: &[usize], after: &[usize]| {
            before
                .iter()
                .chain(after.iter())
                .all(|&extent| extent == pad)
        };
        match &spec.padding {
            PaddingSpec::Explicit(before, after) if centred(before, after) => {}
            PaddingSpec::ExplicitOnnxPool(before, after, ceil_mode) => {
                if *ceil_mode || !centred(before, after) {
                    return None;
                }
            }
            _ => return None,
        }

        let input = model.outlet_fact(node.inputs[0]).ok()?;
        let output = &node.outputs[0].fact;
        if input.datum_type != f32::datum_type()
            || output.datum_type != f32::datum_type()
            || input.shape.len() != 4
            || output.shape.len() != 4
        {
            return None;
        }
        let (in_h, in_w) = (
            input.shape[2].to_usize().ok()?,
            input.shape[3].to_usize().ok()?,
        );
        let channels = input.shape[1].to_usize().ok()?;
        if output.shape[2].to_usize().ok()? != in_h
            || output.shape[3].to_usize().ok()? != in_w
            || output.shape[1].to_usize().ok()? != channels
        {
            return None;
        }
        Some(KernMaxPool {
            channels,
            in_h,
            in_w,
            pad,
        })
    }
}

/// The horizontal pass: `dst` row `oy`, column `ox`, is the largest `src`
/// value inside the `2 * pad + 1` window centred on that column, ignoring
/// whatever the window reads past either edge — a padded index contributes
/// nothing to a maximum.
fn sliding_max(src: &[f32], dst: &mut [f32], rows: usize, width: usize, pad: usize) {
    debug_assert_eq!(src.len(), rows * width);
    debug_assert_eq!(dst.len(), rows * width);
    for oy in 0..rows {
        let source = &src[oy * width..(oy + 1) * width];
        let out = &mut dst[oy * width..(oy + 1) * width];
        for (i, value) in out.iter_mut().enumerate() {
            let lo = i.saturating_sub(pad);
            let hi = (i + pad + 1).min(width);
            *value = source[lo..hi]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
        }
    }
}

/// The vertical pass: output row `oy` holds, per column, the largest value
/// across input rows `oy - pad ..= oy + pad`, edges again contributing
/// nothing.
fn fold_rows(src: &[f32], dst: &mut [f32], rows: usize, width: usize, pad: usize) {
    debug_assert_eq!(src.len(), rows * width);
    debug_assert_eq!(dst.len(), rows * width);
    for oy in 0..rows {
        let lo = oy.saturating_sub(pad);
        let hi = (oy + pad + 1).min(rows);
        let out = &mut dst[oy * width..(oy + 1) * width];
        out.copy_from_slice(&src[lo * width..(lo + 1) * width]);
        for iy in lo + 1..hi {
            for (o, v) in out.iter_mut().zip(&src[iy * width..(iy + 1) * width]) {
                *o = o.max(*v);
            }
        }
    }
}

impl Op for KernMaxPool {
    fn name(&self) -> StaticName {
        "KernRTMaxPool".into()
    }

    tract_onnx::tract_core::op_as_typed_op!();
}

impl EvalOp for KernMaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_array_view::<f32>()?;
        ensure!(input.ndim() == 4);
        ensure!(
            input.shape()[2] == self.in_h && input.shape()[3] == self.in_w,
            "KernRTMaxPool input geometry changed"
        );
        let batch = input.shape()[0];
        let input = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("KernRTMaxPool requires a contiguous input"))?;

        let started = timer_start();
        let output = self.run(input, batch);
        timer_stop(started, &MAXPOOL_NS);
        Ok(tvec!(
            tract_ndarray::Array4::from_shape_vec(
                (batch, self.channels, self.in_h, self.in_w),
                output,
            )?
            .into_tensor()
            .into_tvalue()
        ))
    }
}

impl TypedOp for KernMaxPool {
    tract_onnx::tract_core::as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let batch = inputs
            .first()
            .and_then(|fact| fact.shape.iter().next().cloned())
            .and_then(|dim| dim.to_usize().ok())
            .unwrap_or(1);
        Ok(tvec!(f32::fact(vec![
            batch,
            self.channels,
            self.in_h,
            self.in_w
        ])))
    }
}

/// The per-anchor maximum of the class scores: `Reduce<Max>` over the last
/// axis of a contiguous f32 tensor, `[1, 8400, 80] -> [1, 8400, 1]`.
///
/// tract's `Reducer::reduce` walks it through `ndarray`, which for this shape
/// means a scalar fold over 672 K values — 1.6 ms of the run. The reduced axis
/// is the innermost one, so a NEON `vmax` pass over the same memory is a
/// straight vector reduction with no stride to chase.
#[derive(Debug, Clone)]
struct KernReduceMax {
    rows: usize,
    cols: usize,
}

impl KernReduceMax {
    /// Numeric core of the op, kept separate from [`EvalOp::eval`] so tests
    /// can drive it without building a tract graph.
    fn run(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.rows * self.cols);
        input.chunks_exact(self.cols).map(row_max).collect()
    }

    /// Build the native op from a typed `Reduce`, or `None` unless it reduces
    /// f32 over the last axis of a fully concrete shape and leaves the axis in
    /// place with extent 1.
    fn from_reduce(model: &TypedModel, node: &TypedNode) -> Option<KernReduceMax> {
        use tract_onnx::tract_core::ops::nn::{Reduce, Reducer};

        let reduce = node.op_as::<Reduce>()?;
        if reduce.reducer != Reducer::Max {
            return None;
        }
        let input = model.outlet_fact(node.inputs[0]).ok()?;
        let output = &node.outputs[0].fact;
        if input.datum_type != f32::datum_type() || output.datum_type != f32::datum_type() {
            return None;
        }
        let dims: Vec<usize> = input
            .shape
            .iter()
            .map(|dim| dim.to_usize())
            .collect::<Result<_, _>>()
            .ok()?;
        let rank = dims.len();
        if rank == 0 || reduce.axes.as_slice() != [rank - 1] {
            return None;
        }
        let out_dims: Vec<usize> = output
            .shape
            .iter()
            .map(|dim| dim.to_usize())
            .collect::<Result<_, _>>()
            .ok()?;
        if out_dims.len() != rank
            || out_dims[..rank - 1] != dims[..rank - 1]
            || out_dims[rank - 1] != 1
        {
            return None;
        }
        Some(KernReduceMax {
            rows: dims[..rank - 1].iter().product(),
            cols: dims[rank - 1],
        })
    }
}

/// Largest value in a run: a vector reduction over the whole vectors plus a
/// scalar tail.
///
/// The vector lane uses NEON's `FMAX`, which answers with the quiet NaN when
/// one operand is NaN, while the scalar tail uses `f32::max`, which skips it.
/// That only disagrees on input that already contains a NaN, and the values
/// this node reduces are sigmoid outputs.
#[cfg(target_arch = "aarch64")]
fn row_max(row: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let full = row.len() - row.len() % 4;
    let mut acc;
    unsafe {
        acc = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0;
        while i < full {
            acc = vmaxq_f32(acc, vld1q_f32(row.as_ptr().add(i)));
            i += 4;
        }
    }
    let mut best = unsafe { vmaxvq_f32(acc) };
    for &value in &row[full..] {
        best = best.max(value);
    }
    best
}

#[cfg(not(target_arch = "aarch64"))]
fn row_max(row: &[f32]) -> f32 {
    row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

impl Op for KernReduceMax {
    fn name(&self) -> StaticName {
        "KernRTReduceMax".into()
    }

    tract_onnx::tract_core::op_as_typed_op!();
}

impl EvalOp for KernReduceMax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_array_view::<f32>()?;
        let rank = input.ndim();
        ensure!(rank >= 1);
        let mut shape: Vec<usize> = input.shape().to_vec();
        ensure!(
            shape[rank - 1] == self.cols
                && shape[..rank - 1].iter().product::<usize>() == self.rows,
            "KernRTReduceMax input shape changed"
        );
        let input = input
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("KernRTReduceMax requires a contiguous input"))?;

        let started = timer_start();
        let output = self.run(input);
        timer_stop(started, &REDUCE_NS);
        shape[rank - 1] = 1;
        Ok(tvec!(
            tract_ndarray::Array::from_shape_vec(tract_ndarray::IxDyn(&shape), output)?
                .into_tensor()
                .into_tvalue()
        ))
    }
}

impl TypedOp for KernReduceMax {
    tract_onnx::tract_core::as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(!inputs[0].shape.is_empty());
        let mut shape: Vec<_> = inputs[0].shape.to_vec();
        let last = shape.len() - 1;
        shape[last] = 1.to_dim();
        Ok(tvec!(f32::fact(shape)))
    }
}

/// The im2col panel is the largest transient buffer in the graph — about 15 MB
/// for the first convolution — and every conv node would otherwise allocate it
/// on every inference. `EvalOp::eval` only hands out `&self`, so the panel
/// lives in a thread-local instead of the op.
fn with_im2col_panel<T>(len: usize, run: impl FnOnce(&mut [f32]) -> T) -> T {
    thread_local! {
        static PANEL: std::cell::RefCell<Vec<f32>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    PANEL.with(|cell| {
        let mut panel = cell.borrow_mut();
        panel.resize(len, 0.0);
        run(&mut panel)
    })
}

/// Swap every `Conv` node the builder accepts for the native op it returns.
///
/// Weights are usually still behind a folding candidate at this stage, so the
/// caller runs `prop_consts` first; `declutter` is not an option because it
/// also decomposes `Conv` into the very `Pad + LazyIm2col + OptMatMul` shape
/// the im2col rewrite exists to avoid.
///
/// The two-pass shape is not decoration: a node's `op` cannot be replaced
/// while the graph's node list is being iterated, so the rewrites are
/// collected first and applied after. Each builder decides which convolutions
/// it handles by looking at `Conv::group` alone, so the two can run in
/// either order without a node being claimed twice.
fn rewrite_native_conv_nodes<T>(
    model: &mut TypedModel,
    build: impl Fn(
        &TypedModel,
        &TypedNode,
        &tract_onnx::tract_core::ops::cnn::conv::Conv,
    ) -> TractResult<Option<T>>,
) -> TractResult<usize>
where
    T: TypedOp + 'static,
{
    let mut replacements = Vec::new();
    for node in &model.nodes {
        let Some(conv) = node.op_as::<tract_onnx::tract_core::ops::cnn::conv::Conv>() else {
            continue;
        };
        if let Some(native) = build(model, node, conv)? {
            replacements.push((node.id, native));
        }
    }
    let count = replacements.len();
    for (id, native) in replacements {
        model.nodes[id].op = Box::new(native);
    }
    Ok(count)
}

/// Replace every `Conv` the im2col kernel handles with [`KernConv`].
fn rewrite_native_conv(model: &mut TypedModel) -> TractResult<usize> {
    rewrite_native_conv_nodes(model, KernConv::from_conv)
}

/// Replace every depthwise `Conv` with [`KernDepthWise`].
///
/// Runs before `into_optimized`, because that is where tract's own declutter
/// pass turns these nodes into `DepthWise`.
fn rewrite_native_depthwise(model: &mut TypedModel) -> TractResult<usize> {
    rewrite_native_conv_nodes(model, KernDepthWise::from_conv)
}

/// Fold `Mul(X, Sigmoid(X))` — in either operand order — into [`KernSiLU`].
///
/// The `Sigmoid` is rewritten to tract's `Identity` at the same time. It has
/// to stop running, or the graph pays for a sigmoid nobody reads any more,
/// and once it is an identity both of the `Mul`'s inputs hold `X`, so
/// [`KernSiLU`] can read `inputs[0]` without being told which side it was.
fn rewrite_silu(model: &mut TypedModel) -> TractResult<usize> {
    use std::collections::HashSet;

    let mut candidates = Vec::new();
    for node in &model.nodes {
        let name = node.op.name();
        if (name != "Mul" && name != "OptMulUnicast") || node.inputs.len() != 2 {
            continue;
        }
        let (Ok(first), Ok(second)) = (
            model.outlet_fact(node.inputs[0]),
            model.outlet_fact(node.inputs[1]),
        ) else {
            continue;
        };
        if first.datum_type != f32::datum_type()
            || second.datum_type != f32::datum_type()
            || first.shape != second.shape
            || first.konst.is_some()
            || second.konst.is_some()
        {
            continue;
        }
        let mut matched = None;
        for (input, other) in [
            (node.inputs[0], node.inputs[1]),
            (node.inputs[1], node.inputs[0]),
        ] {
            let producer = &model.node(input.node);
            if producer.op.name() == "Sigmoid"
                && producer.inputs.len() == 1
                && producer.inputs[0] == other
            {
                // `other` is both the Sigmoid's input and this multiply's, so
                // it is the tensor the fused node must read directly.
                matched = Some((input.node, other));
                break;
            }
        }
        if let Some((sigmoid, source)) = matched {
            candidates.push((node.id, sigmoid, source));
        }
    }

    // Neutralising a Sigmoid changes every consumer, so only ones whose
    // consumers are all being folded away may be touched.
    let fused: HashSet<usize> = candidates.iter().map(|(mul, _, _)| *mul).collect();
    let mut folded = Vec::new();
    let mut identities = Vec::new();
    for (mul, sigmoid, source) in candidates {
        let reachable = model
            .node(sigmoid)
            .outputs
            .iter()
            .flat_map(|output| &output.successors)
            .all(|inlet| fused.contains(&inlet.node));
        if reachable {
            identities.push(sigmoid);
            folded.push((mul, source));
        }
    }

    let count = folded.len();
    for id in identities {
        model.nodes[id].op = Box::new(tract_onnx::tract_core::ops::identity::Identity);
    }
    for (id, source) in folded {
        // Reading the source directly rather than the neutralised Sigmoid's
        // identity hop is what lets [`fuse_conv_silu`] see the convolution as
        // this node's producer.
        model.nodes[id].inputs = vec![source];
        model.nodes[id].op = Box::new(KernSiLU);
    }
    Ok(count)
}

/// Fold a [`KernSiLU`] into the [`KernConv`] that produces its input, so the
/// bias and the activation run over a tile the GEMM just wrote instead of over
/// a tensor that has since been written back to and re-read from DRAM.
///
/// Two conditions have to hold before the reader may disappear: the producer
/// must be a native convolution, and it must have exactly one consumer — this
/// node — with this node's own output unused elsewhere. Both outputs being
/// model outputs would also change what the model returns, so those are left
/// alone. Everything else keeps the separate `KernSiLU` node and the separate
/// bias pass.
///
/// The consumer edges are rewired onto the convolution's output rather than
/// pointed at an `Identity`: [`Identity::eval`] is a pass-through, but only
/// if it is evaluated at all, and a dead node is cheaper still.
fn fuse_conv_silu(model: &mut TypedModel) -> TractResult<usize> {
    use tract_onnx::tract_core::model::OutletId;

    // Nodes still on a path to a model output. `rewrite_silu` leaves the
    // neutralised Sigmoid behind as a dead `Identity` that still lists the
    // convolution as its input, so counting successors raw would report two
    // readers when only one of them can ever run.
    let mut live = vec![false; model.nodes.len()];
    let mut stack: Vec<usize> = model.outputs.iter().map(|outlet| outlet.node).collect();
    while let Some(id) = stack.pop() {
        if live[id] {
            continue;
        }
        live[id] = true;
        stack.extend(model.nodes[id].inputs.iter().map(|input| input.node));
    }

    let mut folds = Vec::new();
    for node in &model.nodes {
        if node.op.name() != "KernRTSiLU" || node.inputs.len() != 1 {
            continue;
        }
        let source = node.inputs[0];
        let reader = OutletId::new(node.id, 0);
        if model.outputs.contains(&source) || model.outputs.contains(&reader) {
            continue;
        }
        let producer = &model.node(source.node);
        if producer.op.name() != "KernRTConv" || source.slot != 0 {
            continue;
        }
        let mut readers = producer
            .outputs
            .iter()
            .flat_map(|output| &output.successors)
            .filter(|inlet| live[inlet.node])
            .map(|inlet| inlet.node)
            .collect::<Vec<_>>();
        readers.sort_unstable();
        readers.dedup();
        if readers == [node.id] && producer.op_as::<KernConv>().is_some() {
            folds.push((source, reader, node.id));
        }
    }

    let count = folds.len();
    for (source, reader, silu) in folds {
        model.nodes[source.node]
            .op_as_mut::<KernConv>()
            .expect("checked while collecting")
            .apply_silu = true;
        for consumer in &mut model.nodes {
            for input in consumer.inputs.iter_mut() {
                if *input == reader {
                    *input = source;
                }
            }
        }
        model.nodes[silu].op = Box::new(tract_onnx::tract_core::ops::identity::Identity);
    }
    Ok(count)
}

/// Print the per-layer work behind every native convolution: the im2col panel
/// bytes the gather has to write, the GEMM's FLOPs, and the tile the row
/// tiling picked. Both totals are per inference, so dividing the panel total by
/// a measured im2col time gives the gather's effective bandwidth and dividing
/// the FLOP total by the GEMM time gives achieved GFLOPS — which decides
/// whether the fix is a better gather or a different algorithm.
fn report_conv_work(model: &TypedModel, label: &str) {
    struct Row {
        panel: usize,
        out_channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        copy: bool,
        tile_rows: usize,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut panel_bytes = 0usize;
    let mut gemm_flops = 0usize;
    for node in &model.nodes {
        let Some(conv) = node.op_as::<KernConv>() else {
            continue;
        };
        let reduction = conv.input_channels * conv.kernel_h * conv.kernel_w;
        let out_hw = conv.output_h * conv.output_w;
        let panel = reduction * out_hw * size_of::<f32>();
        panel_bytes += panel;
        gemm_flops += 2 * conv.output_channels * reduction * out_hw;
        let in_h = (conv.output_h - 1) * conv.stride_h + (conv.kernel_h - 1) * conv.dilation_h + 1
            - 2 * conv.pad_h;
        let in_w = (conv.output_w - 1) * conv.stride_w + (conv.kernel_w - 1) * conv.dilation_w + 1
            - 2 * conv.pad_w;
        rows.push(Row {
            panel,
            out_channels: conv.output_channels,
            in_h,
            in_w,
            out_h: conv.output_h,
            out_w: conv.output_w,
            copy: conv.gather_is_a_copy(),
            tile_rows: conv.tile_rows,
        });
    }
    rows.sort_by(|a, b| b.panel.cmp(&a.panel));
    eprintln!("kernrt: {label}: {} native convolutions", rows.len());
    for r in &rows {
        eprintln!(
            "kernrt:   panel {:>8} KB  in {:>3}x{:<4} out {:>3}x{:<4} c {:>3}  tile {} rows{}",
            r.panel / 1024,
            r.in_h,
            r.in_w,
            r.out_h,
            r.out_w,
            r.out_channels,
            r.tile_rows,
            if r.copy { " 1x1-copy" } else { "" },
        );
    }
    eprintln!(
        "kernrt:   total panel {} MiB, GEMM {:.2} GFLOP per inference",
        panel_bytes as f64 / (1024.0 * 1024.0),
        gemm_flops as f64 / 1e9,
    );
}

/// Replace every `MaxPool` the native kernel handles with [`KernMaxPool`].
///
/// tract's pool walks a zone visitor for each of its taps, which costs far
/// more than the 0.6 MB the three pools in the neck actually move.
fn rewrite_native_maxpool(model: &mut TypedModel) -> TractResult<usize> {
    let mut replacements = Vec::new();
    for node in &model.nodes {
        if let Some(native) = KernMaxPool::from_maxpool(model, node) {
            replacements.push((node.id, native));
        }
    }
    let count = replacements.len();
    for (id, native) in replacements {
        model.nodes[id].op = Box::new(native);
    }
    Ok(count)
}

/// Swap every `Reduce<Max>` over the last axis for [`KernReduceMax`].
fn rewrite_native_reduce_max(model: &mut TypedModel) -> TractResult<usize> {
    let mut replacements = Vec::new();
    for node in &model.nodes {
        if let Some(native) = KernReduceMax::from_reduce(model, node) {
            replacements.push((node.id, native));
        }
    }
    let count = replacements.len();
    for (id, native) in replacements {
        model.nodes[id].op = Box::new(native);
    }
    Ok(count)
}

fn rewrite_native_upsample(model: &mut TypedModel) -> TractResult<usize> {
    let mut replacements = Vec::new();
    for node in &model.nodes {
        if node.op.name() != "Resize" || node.inputs.is_empty() {
            continue;
        }
        let input = model.outlet_fact(node.inputs[0])?;
        let output = &node.outputs[0].fact;
        if input.datum_type != f32::datum_type()
            || input.shape.len() != 4
            || output.datum_type != f32::datum_type()
            || output.shape.len() != 4
        {
            continue;
        }
        let input_shape = input.shape.to_vec();
        let output_shape = output.shape.to_vec();
        let (Ok(in_h), Ok(in_w), Ok(out_h), Ok(out_w)) = (
            input_shape[2].to_usize(),
            input_shape[3].to_usize(),
            output_shape[2].to_usize(),
            output_shape[3].to_usize(),
        ) else {
            continue;
        };
        if out_h != in_h * 2 || out_w != in_w * 2 {
            continue;
        }
        let shape = output_shape
            .into_iter()
            .map(|dim| dim.to_usize())
            .collect::<TractResult<Vec<_>>>()?;
        replacements.push((
            node.id,
            KernUpsample2x {
                output_shape: shape,
            },
        ));
    }
    let count = replacements.len();
    for (id, native) in replacements {
        model.nodes[id].op = Box::new(native);
    }
    Ok(count)
}

impl Yolo26 {
    /// Load and optimize a fixed-shape YOLO26 ONNX export.
    pub fn from_onnx(path: impl AsRef<Path>, width: usize, height: usize) -> TractResult<Self> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1usize, 3, height, width)),
            )?
            .into_typed()?;
        if std::env::var_os("KERNRT_VERBOSE").is_some() {
            report_nodes(&model, "after into_typed");
            report_convs(&model, "after into_typed");
        }
        let native_conv = std::env::var_os("KERNRT_NATIVE_CONV").is_none_or(|value| value != "0");
        let native_depthwise =
            std::env::var_os("KERNRT_NATIVE_DEPTHWISE").is_none_or(|value| value != "0");
        if native_conv || native_depthwise {
            // `prop_consts` folds the weight chain into facts without the
            // declutter pass that would destroy the Conv nodes these rewrites
            // are looking for. Reported count should match the conv breakdown
            // above.
            model.prop_consts()?;
        }
        if native_conv {
            let replaced = rewrite_native_conv(&mut model)?;
            if std::env::var_os("KERNRT_VERBOSE").is_some() {
                eprintln!("kernrt: native convolutions={replaced}");
            }
        }
        if native_depthwise {
            let replaced = rewrite_native_depthwise(&mut model)?;
            if std::env::var_os("KERNRT_VERBOSE").is_some() {
                eprintln!("kernrt: native depthwise={replaced}");
            }
        }
        if std::env::var_os("KERNRT_NATIVE_SILU").is_none_or(|value| value != "0") {
            let replaced = rewrite_silu(&mut model)?;
            if std::env::var_os("KERNRT_VERBOSE").is_some() {
                eprintln!("kernrt: native SiLU={replaced}");
            }
            if native_conv {
                // Needs both halves of the pattern to be native, and to run
                // after `rewrite_silu` so the reader exists to be folded.
                let folded = fuse_conv_silu(&mut model)?;
                if std::env::var_os("KERNRT_VERBOSE").is_some() {
                    eprintln!("kernrt: SiLU folded into conv={folded}");
                }
            }
        }
        if std::env::var_os("KERNRT_NATIVE_MAXPOOL").is_none_or(|value| value != "0") {
            let replaced = rewrite_native_maxpool(&mut model)?;
            if std::env::var_os("KERNRT_VERBOSE").is_some() {
                eprintln!("kernrt: native maxpool={replaced}");
            }
        }
        if std::env::var_os("KERNRT_NATIVE_REDUCE").is_none_or(|value| value != "0") {
            let replaced = rewrite_native_reduce_max(&mut model)?;
            if std::env::var_os("KERNRT_VERBOSE").is_some() {
                eprintln!("kernrt: native reduce_max={replaced}");
            }
        }
        if std::env::var_os("KERNRT_NATIVE_UPSAMPLE").is_none_or(|value| value != "0") {
            let replaced = rewrite_native_upsample(&mut model)?;
            if std::env::var_os("KERNRT_VERBOSE").is_some() {
                eprintln!("kernrt: native 2x upsample={replaced}");
            }
        }
        if std::env::var_os("KERNRT_VERBOSE").is_some() {
            report_conv_work(&model, "native conv work");
        }
        let model = model.into_optimized()?;
        if std::env::var_os("KERNRT_VERBOSE").is_some() {
            report_nodes(&model, "optimized");
            report_stragglers(&model);
            report_head_ops(&model);
        }
        let model = model.into_runnable()?;
        Ok(Self {
            model,
            width,
            height,
        })
    }

    pub fn input_width(&self) -> usize {
        self.width
    }

    pub fn input_height(&self) -> usize {
        self.height
    }

    /// Run the graph on normalized NCHW f32 input and return the first output
    /// flattened. The method intentionally exposes the raw tensor too, so
    /// future YOLO26 task heads can share the loader without guessing output
    /// semantics.
    pub fn infer_nchw(&self, input: &[f32]) -> TractResult<Vec<f32>> {
        let expected = 3 * self.width * self.height;
        ensure!(
            input.len() == expected,
            "expected {expected} NCHW values, got {}",
            input.len()
        );
        let tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, self.height, self.width), input.to_vec())?
                .into_tensor();
        let started = timer_start();
        let outputs = self.model.run(tvec!(tensor.into_tvalue()))?;
        timer_stop(started, &RUN_NS);
        timer_record_inference(50);
        Ok(outputs[0].to_array_view::<f32>()?.iter().copied().collect())
    }

    /// Convert an RGB image to normalized NCHW with nearest-neighbor resize.
    /// The HTTP server uses this path; benchmark callers should prebuild NCHW
    /// once and call [`infer_nchw`] to measure model execution only.
    pub fn preprocess_rgb(&self, rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
        assert_eq!(rgb.len(), width * height * 3, "RGB buffer shape mismatch");
        let plane = self.width * self.height;
        let mut out = vec![0.0; plane * 3];
        for y in 0..self.height {
            let sy = y * height / self.height;
            for x in 0..self.width {
                let sx = x * width / self.width;
                let src = (sy * width + sx) * 3;
                let dst = y * self.width + x;
                out[dst] = f32::from(rgb[src]) / 255.0;
                out[plane + dst] = f32::from(rgb[src + 1]) / 255.0;
                out[2 * plane + dst] = f32::from(rgb[src + 2]) / 255.0;
            }
        }
        out
    }

    /// Decode the standard NMS-free `[N,300,6]` output. For a one-to-many
    /// export, return an empty vector rather than silently applying the wrong
    /// NMS convention; callers can inspect [`infer_nchw`] for that head.
    pub fn detections(&self, output: &[f32]) -> Vec<Detection> {
        if output.len() % 6 != 0 {
            return Vec::new();
        }
        output
            .chunks_exact(6)
            .filter_map(|v| {
                if v[4] <= 0.0 || !v[4].is_finite() {
                    return None;
                }
                Some(Detection {
                    x1: v[0],
                    y1: v[1],
                    x2: v[2],
                    y2: v[3],
                    score: v[4],
                    class_id: v[5].max(0.0) as usize,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
