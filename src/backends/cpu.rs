//! The tuned CPU backend: hand-written NEON kernels, Q8_0 storage, and f32
//! copies of the weights for the batched GEMM.
//!
//! This is the path every number in `PLAN.md` was measured on. It is unchanged
//! arithmetic — moved behind [`Backend`] rather than reimplemented — so the
//! move itself must not shift a single result.
//!
//! Buffers are plain `Vec<f32>`. There is no device to keep them on, so the
//! handle indirection costs an index and nothing else.

use std::sync::Arc;

use super::{AttnSpec, Backend, Buf, KvSpec, ModelWeights, Proj, RopeSpec, Window};
use crate::ops::Matrix;
use crate::ops::{
    Q8_0_BLOCK, Q8Vec, dot, gemm_nt_into, gemm_q8_0x4_batched_into, gemv_into, gemv_q8_0_into,
    softmax_in_place,
};

/// Default ceiling on the f32 copy of the weights, in bytes.
///
/// Above this the copy costs more than it saves: at 1.7B it is 6.4 GB against
/// 3.4 GB of bf16 weights and a 3.2 GB KV cache, which on a 17 GB machine
/// swaps and makes the batched path *slower* than the decode loop.
const DEFAULT_WIDEN_BUDGET: usize = 2 << 30;

/// f32 copies of every projection, for the batched GEMMs, indexed
/// `[layer][Proj::index()]`.
type Widened = Vec<[Vec<f32>; 9]>;

pub struct Cpu {
    weights: Option<Arc<ModelWeights>>,
    storage: &'static str,
    /// Activation memory, addressed by [`Buf`].
    bufs: Vec<Vec<f32>>,
    /// Scratch for the quantized activation of the current input row.
    q8: Q8Vec,
    /// Row-major quantized activations for the native batched Q8_0x4 GEMM.
    batch_quants: Vec<i8>,
    batch_scales_f16: Vec<u16>,
    /// Whether batched work should run on an f32 copy of the weights.
    prefill_f32: bool,
    /// `Some` when a caller forced the policy.
    prefill_f32_override: Option<bool>,
    widened: Option<Widened>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    kv: KvSpec,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrow two distinct elements of a slice mutably.
fn two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert_ne!(i, j, "buffers must be distinct");
    if i < j {
        let (a, b) = v.split_at_mut(j);
        (&mut a[i], &mut b[0])
    } else {
        let (a, b) = v.split_at_mut(i);
        (&mut b[0], &mut a[j])
    }
}

/// Borrow three distinct elements of a slice mutably.
fn three_mut<T>(v: &mut [T], i: usize, j: usize, k: usize) -> (&mut T, &mut T, &mut T) {
    assert!(
        i != j && j != k && i != k,
        "buffers must be pairwise distinct"
    );
    // Order the indices so two splits isolate all three slots.
    let mut ord = [i, j, k];
    ord.sort_unstable();
    let (lo, hi) = v.split_at_mut(ord[2]);
    let (lo2, mid) = lo.split_at_mut(ord[1]);
    // SAFETY: `split_at_mut` guarantees the three slots are disjoint, and the
    // indices are pairwise distinct, so the three pointers never alias.
    unsafe {
        let mut p = |n: usize| {
            if n == ord[2] {
                hi.as_mut_ptr()
            } else if n == ord[1] {
                mid.as_mut_ptr()
            } else {
                debug_assert_eq!(n, ord[0]);
                lo2.as_mut_ptr().add(n)
            }
        };
        (&mut *p(i), &mut *p(j), &mut *p(k))
    }
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            weights: None,
            storage: "f32",
            bufs: Vec::new(),
            q8: Q8Vec {
                scales: Vec::new(),
                scales_f16: Vec::new(),
                quants: Vec::new(),
            },
            batch_quants: Vec::new(),
            batch_scales_f16: Vec::new(),
            prefill_f32: true,
            prefill_f32_override: None,
            widened: None,
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            kv: KvSpec {
                n_layers: 0,
                kv_dim: 0,
                seq_len: 0,
            },
        }
    }

    /// Force the batched f32 copy on or off, overriding both the size estimate
    /// and `KERNRT_PREFILL_F32`. For benchmarks and tests that need both sides.
    pub fn with_prefill_f32(mut self, enabled: bool) -> Self {
        self.prefill_f32 = enabled;
        self.prefill_f32_override = Some(enabled);
        self
    }

    fn weights(&self) -> &ModelWeights {
        self.weights
            .as_deref()
            .expect("prepare() runs before any compute")
    }

    fn proj_matrix(&self, layer: usize, proj: Proj) -> &Matrix {
        proj.matrix(&self.weights().layers[layer])
    }

    /// Quantize `rows` activation rows into the contiguous form the batched
    /// Q8_0x4 kernel consumes.
    fn quantize_rows(&mut self, input: &[f32], width: usize, rows: usize) {
        debug_assert_eq!(width % Q8_0_BLOCK, 0);
        let blocks = width / Q8_0_BLOCK;
        self.batch_quants.clear();
        self.batch_quants.resize(rows * width, 0);
        self.batch_scales_f16.clear();
        self.batch_scales_f16.resize(rows * blocks, 0);
        for row in 0..rows {
            self.q8
                .quantize_into(&input[row * width..(row + 1) * width]);
            self.batch_quants[row * width..(row + 1) * width]
                .copy_from_slice(&self.q8.quants[..width]);
            self.batch_scales_f16[row * blocks..(row + 1) * blocks]
                .copy_from_slice(&self.q8.scales_f16[..blocks]);
        }
    }

    /// Build the f32 copies the batched path needs. Layers only widen the
    /// projections they carry: attention layers skip the conv pair, conv
    /// layers skip the attention set, and either side leaves a zero-length
    /// placeholder the driver never dispatches.
    fn ensure_widened(&mut self) {
        if self.widened.is_some() {
            return;
        }
        let weights = self.weights();
        let layers = weights
            .layers
            .iter()
            .map(|layer| {
                std::array::from_fn(|i| match Proj::ALL[i].matrix_opt(layer) {
                    Some(m) => m.to_f32_tensor().into_data(),
                    None => Vec::new(),
                })
            })
            .collect();
        self.widened = Some(layers);
    }
}

impl Backend for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn storage(&self) -> &'static str {
        self.storage
    }

    fn prepare(&mut self, weights: Arc<ModelWeights>, kv: KvSpec) {
        // The embedding is widened with the same storage as every projection,
        // so it reports the model width without depending on layer 0 being an
        // attention layer (it is not, on hybrid models).
        self.storage = weights.token_embedding.kind();
        self.prefill_f32 = self
            .prefill_f32_override
            .unwrap_or_else(|| prefill_f32_policy(&weights));
        self.kv = kv;
        self.key_cache = vec![0.0; kv.n_layers * kv.seq_len * kv.kv_dim];
        self.value_cache = vec![0.0; kv.n_layers * kv.seq_len * kv.kv_dim];
        self.weights = Some(weights);
    }

    fn alloc(&mut self, len: usize) -> Buf {
        self.bufs.push(vec![0.0; len]);
        Buf(self.bufs.len() - 1)
    }

    fn upload(&mut self, data: &[f32]) -> Buf {
        self.bufs.push(data.to_vec());
        Buf(self.bufs.len() - 1)
    }

    fn read(&mut self, src: Buf, out: &mut [f32]) {
        out.copy_from_slice(&self.bufs[src.0][..out.len()]);
    }

    fn embed(&mut self, token: usize, out: Buf, offset: usize) {
        let Self { weights, bufs, .. } = self;
        let weights = weights.as_deref().expect("prepare() runs before compute");
        let dim = weights.token_embedding.in_dim();
        weights
            .token_embedding
            .decode_row(token, &mut bufs[out.0][offset..offset + dim]);
    }

    fn copy(&mut self, src: Buf, src_offset: usize, dst: Buf, dst_offset: usize, len: usize) {
        if src.0 == dst.0 {
            self.bufs[dst.0].copy_within(src_offset..src_offset + len, dst_offset);
            return;
        }
        let (s, d) = two_mut(&mut self.bufs, src.0, dst.0);
        d[dst_offset..dst_offset + len].copy_from_slice(&s[src_offset..src_offset + len]);
    }

    fn kv_write(&mut self, layer: usize, slots: &[usize], k: Buf, v: Buf, rows: usize) {
        let width = self.kv.kv_dim;
        for (row, &slot) in slots.iter().enumerate().take(rows) {
            let at = (layer * self.kv.seq_len + slot) * width;
            let src = row * width;
            self.key_cache[at..at + width].copy_from_slice(&self.bufs[k.0][src..src + width]);
            self.value_cache[at..at + width].copy_from_slice(&self.bufs[v.0][src..src + width]);
        }
    }

    fn rms_norm(&mut self, out: Buf, x: Buf, weight: Buf, rows: usize, dim: usize, eps: f32) {
        let w = self.bufs[weight.0].clone();
        let (o, xi) = two_mut(&mut self.bufs, out.0, x.0);
        for row in 0..rows {
            let at = row * dim;
            rms_norm_row(&mut o[at..at + dim], &xi[at..at + dim], &w, eps);
        }
    }

    fn qk_norm(
        &mut self,
        v: Buf,
        rows: usize,
        width: usize,
        head_size: usize,
        weight: Buf,
        eps: f32,
    ) {
        let w = self.bufs[weight.0].clone();
        let data = &mut self.bufs[v.0];
        for row in 0..rows {
            let at = row * width;
            qk_norm_row(&mut data[at..at + width], &w, head_size, eps);
        }
    }

    fn rope(
        &mut self,
        q: Buf,
        k: Buf,
        q_width: usize,
        kv_width: usize,
        rows: usize,
        positions: &[usize],
        spec: RopeSpec,
    ) {
        let (qv, kv) = two_mut(&mut self.bufs, q.0, k.0);
        for (row, &pos) in positions.iter().enumerate().take(rows) {
            let qo = row * q_width;
            let ko = row * kv_width;
            if spec.interleaved {
                rope_interleaved(
                    &mut qv[qo..qo + q_width],
                    &mut kv[ko..ko + kv_width],
                    pos,
                    spec.head_size,
                    kv_width,
                    spec.theta,
                );
            } else {
                rope_half(&mut qv[qo..qo + q_width], pos, spec.head_size, spec.theta);
                rope_half(&mut kv[ko..ko + kv_width], pos, spec.head_size, spec.theta);
            }
        }
    }

    fn project(&mut self, layer: usize, proj: Proj, x: Buf, y: Buf, rows: usize) {
        let (in_dim, out_dim, native) = {
            let w = self.proj_matrix(layer, proj);
            (w.in_dim(), w.out_dim(), matches!(w, Matrix::Q8_0x4(_)))
        };
        let use_native = native && !self.prefill_f32;

        if use_native {
            let xv = self.bufs[x.0].clone();
            self.quantize_rows(&xv, in_dim, rows);
            let Self {
                weights,
                bufs,
                batch_quants,
                batch_scales_f16,
                ..
            } = self;
            let weights = weights.as_deref().expect("prepare() runs before compute");
            let Matrix::Q8_0x4(packed) = proj.matrix(&weights.layers[layer]) else {
                unreachable!("guarded by `native` above")
            };
            let blocks = in_dim / Q8_0_BLOCK;
            gemm_q8_0x4_batched_into(
                packed,
                &batch_quants[..rows * in_dim],
                &batch_scales_f16[..rows * blocks],
                rows,
                &mut bufs[y.0][..rows * out_dim],
            );
            return;
        }

        if self.prefill_f32 {
            self.ensure_widened();
            let Self { bufs, widened, .. } = self;
            let (xv, yv) = two_mut(bufs, x.0, y.0);
            let copy = &widened.as_ref().expect("built above")[layer][proj.index()];
            gemm_nt_into(
                &xv[..rows * in_dim],
                &copy[..out_dim * in_dim],
                &mut yv[..rows * out_dim],
                rows,
                in_dim,
                out_dim,
            );
            return;
        }

        // No f32 copy and no native kernel: run the quantized decode kernel
        // once per row, reading the stored weights instead of a widened copy.
        let Self {
            weights, bufs, q8, ..
        } = self;
        let weights = weights.as_deref().expect("prepare() runs before compute");
        let w = proj.matrix(&weights.layers[layer]);
        let (xv, yv) = two_mut(bufs, x.0, y.0);
        let quantized = matches!(w, Matrix::Q8_0 { .. } | Matrix::Q8_0x4(_));
        for row in 0..rows {
            let src = &xv[row * in_dim..(row + 1) * in_dim];
            let dst = &mut yv[row * out_dim..(row + 1) * out_dim];
            if quantized {
                q8.quantize_into(src);
                gemv_q8_0_into(w, q8, dst).expect("shapes are fixed at load");
            } else {
                gemv_into(w, src, dst).expect("shapes are fixed at load");
            }
        }
    }

    fn attention(
        &mut self,
        q: Buf,
        out: Buf,
        att: Buf,
        layer: usize,
        windows: &[Window],
        spec: AttnSpec,
    ) {
        let qv = self.bufs[q.0].clone();
        let loff = layer * spec.seq_len * spec.kv_dim;
        let key_cache = &self.key_cache;
        let value_cache = &self.value_cache;
        let (ov, av) = two_mut(&mut self.bufs, out.0, att.0);
        let width = spec.n_heads * spec.head_size;
        for (row, w) in windows.iter().enumerate() {
            let scores = row * spec.n_heads * spec.seq_len;
            attention_rows(
                &qv[row * width..(row + 1) * width],
                key_cache,
                value_cache,
                &mut ov[row * width..(row + 1) * width],
                &mut av[scores..scores + spec.n_heads * spec.seq_len],
                loff,
                w.att_start,
                w.slot,
                w.shared,
                spec,
            );
        }
    }

    fn add_in_place(&mut self, dst: Buf, src: Buf, len: usize) {
        let (d, s) = two_mut(&mut self.bufs, dst.0, src.0);
        for (x, &add) in d[..len].iter_mut().zip(&s[..len]) {
            *x += add;
        }
    }

    fn mul(
        &mut self,
        out: Buf,
        out_off: usize,
        a: Buf,
        a_off: usize,
        b: Buf,
        b_off: usize,
        len: usize,
    ) {
        // Same-index elementwise, so any aliasing is benign; the cases below
        // only exist to satisfy the borrow checker.
        if out.0 == a.0 && out.0 == b.0 {
            let v = &mut self.bufs[out.0];
            for i in 0..len {
                let x = v[a_off + i];
                let y = v[b_off + i];
                v[out_off + i] = x * y;
            }
            return;
        }
        if out.0 == a.0 {
            let (o, bv) = two_mut(&mut self.bufs, out.0, b.0);
            for i in 0..len {
                let x = o[a_off + i];
                o[out_off + i] = x * bv[b_off + i];
            }
            return;
        }
        if out.0 == b.0 {
            let (o, av) = two_mut(&mut self.bufs, out.0, a.0);
            for i in 0..len {
                let y = o[b_off + i];
                o[out_off + i] = av[a_off + i] * y;
            }
            return;
        }
        if a.0 == b.0 {
            let (ab, o) = two_mut(&mut self.bufs, a.0, out.0);
            for i in 0..len {
                o[out_off + i] = ab[a_off + i] * ab[b_off + i];
            }
            return;
        }
        let (o, av, bv) = three_mut(&mut self.bufs, out.0, a.0, b.0);
        for i in 0..len {
            o[out_off + i] = av[a_off + i] * bv[b_off + i];
        }
    }

    fn conv_prefill(&mut self, out: Buf, x: Buf, kernel: Buf, rows: usize, dim: usize, k: usize) {
        let xv = self.bufs[x.0].clone();
        let kv = self.bufs[kernel.0].clone();
        let ov = &mut self.bufs[out.0];
        conv_rows(&xv, &kv, ov, rows, dim, k);
    }

    fn conv_decode(&mut self, out: Buf, x: Buf, kernel: Buf, state: Buf, dim: usize, k: usize) {
        // History is state (oldest first) followed by the current row.
        let sv = self.bufs[state.0].clone();
        let xv = self.bufs[x.0].clone();
        let kv = self.bufs[kernel.0].clone();
        let hist_len = k - 1;
        let mut hist = Vec::with_capacity(k * dim);
        hist.extend_from_slice(&sv[..hist_len * dim]);
        hist.extend_from_slice(&xv[..dim]);
        // The history is itself a k-row series; the wanted output is its
        // last row (the first k-1 rows are the zero-padded prefix).
        let ov = &mut self.bufs[out.0];
        let mut full = vec![0.0f32; k * dim];
        conv_rows(&hist, &kv, &mut full, k, dim, k);
        ov[..dim].copy_from_slice(&full[(k - 1) * dim..]);
        // Drop the oldest row, gain the current one.
        if hist_len > 0 {
            let sv = &mut self.bufs[state.0];
            if hist_len > 1 {
                sv.copy_within(dim..hist_len * dim, 0);
            }
            sv[(hist_len - 1) * dim..hist_len * dim].copy_from_slice(&xv[..dim]);
        }
    }

    fn silu_mul(&mut self, gate: Buf, up: Buf, len: usize) {
        let (g, u) = two_mut(&mut self.bufs, gate.0, up.0);
        for (x, &v) in g[..len].iter_mut().zip(&u[..len]) {
            *x = silu(*x) * v;
        }
    }

    fn logits(&mut self, x: Buf, y: Buf) {
        let Self {
            weights, bufs, q8, ..
        } = self;
        let weights = weights.as_deref().expect("prepare() runs before compute");
        let w = weights.classifier();
        let (xv, yv) = two_mut(bufs, x.0, y.0);
        if matches!(w, Matrix::Q8_0 { .. } | Matrix::Q8_0x4(_)) {
            q8.quantize_into(xv);
            gemv_q8_0_into(w, q8, yv).expect("shapes are fixed at load");
        } else {
            gemv_into(w, xv, yv).expect("shapes are fixed at load");
        }
    }
}

/// Whether to materialise an f32 copy of the weights for batched work.
///
/// `KERNRT_PREFILL_F32=0` forces it off and `=1` forces it on regardless of
/// storage or size; unset means "on, except where it would cost exactness".
///
/// The exception is `Q8_0x4`. Its native batched kernel reproduces the decode
/// path's activation quantization block for block, so prefill and decode agree
/// to f32 rounding. Running the same model through an f32 copy is about 2.1x
/// faster (1213 against 571 tok/s at 135M, prefill 128) but it is a *different
/// arithmetic*: exact weights and exact activations instead of int8 for both.
/// On SmolLM2 the two differ by up to 0.52 in logits, which is the more
/// accurate side but not the decode path's.
fn prefill_f32_policy(weights: &ModelWeights) -> bool {
    match std::env::var("KERNRT_PREFILL_F32").as_deref() {
        Ok("0") | Ok("false") => return false,
        Ok("1") | Ok("true") => return true,
        _ => {}
    }
    let quantized_packed = matches!(weights.token_embedding, Matrix::Q8_0x4(_));
    if quantized_packed {
        return false;
    }
    let bytes: usize = weights
        .layers
        .iter()
        .flat_map(|layer| {
            Proj::ALL.iter().filter_map(move |proj| {
                proj.matrix_opt(layer).map(|m| m.out_dim() * m.in_dim() * 4)
            })
        })
        .sum();
    bytes <= DEFAULT_WIDEN_BUDGET
}

/// Causal depthwise convolution over `rows` rows, shared by prefill and the
/// single-row decode path.
///
/// Output row `r` is `sum_j kernel[i*k+j] * hist[r-k+1+j][i]`, where `hist`
/// is `x` zero-padded before row 0. `k` is tiny (3 on LFM2), so this is a
/// direct loop rather than anything blocked.
fn conv_rows(x: &[f32], kernel: &[f32], out: &mut [f32], rows: usize, dim: usize, k: usize) {
    for r in 0..rows {
        for i in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..k {
                // History index `r - (k-1) + j`, zero-padded below zero.
                let h = r as isize - (k as isize - 1) + j as isize;
                if h >= 0 {
                    acc += kernel[i * k + j] * x[h as usize * dim + i];
                }
            }
            out[r * dim + i] = acc;
        }
    }
}

/// Multi-head causal attention for one query row, writing into `out`.
#[allow(clippy::too_many_arguments)]
fn attention_rows(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    out: &mut [f32],
    att: &mut [f32],
    loff: usize,
    att_start: usize,
    slot: usize,
    shared: (usize, usize),
    spec: AttnSpec,
) {
    let head_size = spec.head_size;
    let kv_dim = spec.kv_dim;
    let kv_mul = spec.kv_mul;
    let seq_len = spec.seq_len;
    let scale = 1.0f32 / (head_size as f32).sqrt();
    let (shared_start, shared_end) = shared;
    let n_shared = shared_end - shared_start;
    let count = n_shared + slot - att_start + 1;

    for h in 0..spec.n_heads {
        let qh = &q[h * head_size..h * head_size + head_size];
        let att = &mut att[h * seq_len..h * seq_len + seq_len];

        // The shared range is scored first and the row's own range second, so a
        // suffix row sees the prefix in exactly the order a leaf that spelled
        // the prefix out would have.
        for (i, t) in (shared_start..shared_end).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            att[i] = dot(qh, &key_cache[at..at + head_size]) * scale;
        }
        for (i, t) in (att_start..=slot).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            att[n_shared + i] = dot(qh, &key_cache[at..at + head_size]) * scale;
        }
        softmax_in_place(&mut att[..count]);

        let out = &mut out[h * head_size..h * head_size + head_size];
        out.fill(0.0);
        for (i, t) in (shared_start..shared_end).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            let a = att[i];
            for (o, &v) in out.iter_mut().zip(&value_cache[at..at + head_size]) {
                *o += a * v;
            }
        }
        for (i, t) in (att_start..=slot).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            let a = att[n_shared + i];
            for (o, &v) in out.iter_mut().zip(&value_cache[at..at + head_size]) {
                *o += a * v;
            }
        }
    }
}

/// Root-mean-square norm, `out = weight * x / sqrt(mean(x^2) + eps)`.
fn rms_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += eps;
    let scale = 1.0 / ss.sqrt();
    for ((o, &xv), &wv) in out.iter_mut().zip(x).zip(weight) {
        *o = wv * (scale * xv);
    }
}

/// Qwen3's per-head norm: one `head_size` slice at a time.
fn qk_norm_row(vec: &mut [f32], weight: &[f32], head_size: usize, eps: f32) {
    debug_assert_eq!(weight.len(), head_size);
    for head in vec.chunks_exact_mut(head_size) {
        let mut ss = 0.0f32;
        for &x in head.iter() {
            ss += x * x;
        }
        ss /= head_size as f32;
        ss += eps;
        let scale = 1.0 / ss.sqrt();
        for (o, &w) in head.iter_mut().zip(weight) {
            *o = w * (scale * *o);
        }
    }
}

/// RoPE that rotates adjacent pairs, as in llama2.c.
fn rope_interleaved(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_size: usize,
    kv_dim: usize,
    theta: f32,
) {
    let mut i = 0;
    while i + 1 < q.len() {
        let head_dim = i % head_size;
        let freq = 1.0f32 / theta.powf(head_dim as f32 / head_size as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();

        let (q0, q1) = (q[i], q[i + 1]);
        q[i] = q0 * cos - q1 * sin;
        q[i + 1] = q0 * sin + q1 * cos;

        if i < kv_dim {
            let (k0, k1) = (k[i], k[i + 1]);
            k[i] = k0 * cos - k1 * sin;
            k[i + 1] = k0 * sin + k1 * cos;
        }
        i += 2;
    }
}

/// RoPE that rotates the two halves of each head, as in HuggingFace Llama.
fn rope_half(vec: &mut [f32], pos: usize, head_size: usize, theta: f32) {
    let half = head_size / 2;
    for head in vec.chunks_exact_mut(head_size) {
        for i in 0..half {
            let freq = 1.0f32 / theta.powf((2 * i) as f32 / head_size as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let (lo, hi) = (head[i], head[i + half]);
            head[i] = lo * cos - hi * sin;
            head[i + half] = lo * sin + hi * cos;
        }
    }
}

/// SiLU activation: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
