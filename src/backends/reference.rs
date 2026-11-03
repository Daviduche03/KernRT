//! A scalar reference backend: the oracle.
//!
//! Deliberately unoptimized. No SIMD, no packed weights, no activation
//! quantization, no blocked GEMM — every reduction is a plain left-to-right f32
//! loop. That makes it roughly an order of magnitude slower than [`cpu`], and
//! that is the point: it exists so a new backend can be checked against
//! something that shares none of its structure.
//!
//! [`cpu`]: super::cpu

use std::sync::Arc;

use super::{AttnSpec, Backend, Buf, KvSpec, ModelWeights, Proj, RopeSpec, Window};

pub struct Reference {
    weights: Option<Arc<ModelWeights>>,
    storage: &'static str,
    bufs: Vec<Vec<f32>>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    kv: KvSpec,
}

impl Default for Reference {
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

/// Scalar dot product. A plain loop on purpose: `ops::dot` is the NEON kernel,
/// and reusing it would make the oracle share what it is meant to check.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Reference {
    pub fn new() -> Self {
        Reference {
            weights: None,
            storage: "f32",
            bufs: Vec::new(),
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            kv: KvSpec {
                n_layers: 0,
                kv_dim: 0,
                seq_len: 0,
            },
        }
    }
}

impl Backend for Reference {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn storage(&self) -> &'static str {
        self.storage
    }

    fn prepare(&mut self, weights: Arc<ModelWeights>, kv: KvSpec) {
        self.storage = weights
            .layers
            .first()
            .map_or("f32", |layer| layer.wq.kind());
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
            let (dst, src) = (&mut o[at..at + dim], &xi[at..at + dim]);
            let mut ss = 0.0f32;
            for &v in src {
                ss += v * v;
            }
            ss /= dim as f32;
            ss += eps;
            let scale = 1.0 / ss.sqrt();
            for i in 0..dim {
                dst[i] = w[i] * (scale * src[i]);
            }
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
            let base = row * width;
            for head in 0..width / head_size {
                let at = base + head * head_size;
                let mut ss = 0.0f32;
                for i in 0..head_size {
                    ss += data[at + i] * data[at + i];
                }
                ss /= head_size as f32;
                ss += eps;
                let scale = 1.0 / ss.sqrt();
                for i in 0..head_size {
                    data[at + i] = w[i] * (scale * data[at + i]);
                }
            }
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
                let mut i = 0;
                while i + 1 < q_width {
                    let head_dim = i % spec.head_size;
                    let freq = 1.0f32 / spec.theta.powf(head_dim as f32 / spec.head_size as f32);
                    let (sin, cos) = (pos as f32 * freq).sin_cos();
                    let (q0, q1) = (qv[qo + i], qv[qo + i + 1]);
                    qv[qo + i] = q0 * cos - q1 * sin;
                    qv[qo + i + 1] = q0 * sin + q1 * cos;
                    if i < kv_width {
                        let (k0, k1) = (kv[ko + i], kv[ko + i + 1]);
                        kv[ko + i] = k0 * cos - k1 * sin;
                        kv[ko + i + 1] = k0 * sin + k1 * cos;
                    }
                    i += 2;
                }
            } else {
                let half = spec.head_size / 2;
                for (vec, base, width) in [(&mut qv[..], qo, q_width), (&mut kv[..], ko, kv_width)]
                {
                    for head in 0..width / spec.head_size {
                        let at = base + head * spec.head_size;
                        for i in 0..half {
                            let freq =
                                1.0f32 / spec.theta.powf((2 * i) as f32 / spec.head_size as f32);
                            let (sin, cos) = (pos as f32 * freq).sin_cos();
                            let (lo, hi) = (vec[at + i], vec[at + i + half]);
                            vec[at + i] = lo * cos - hi * sin;
                            vec[at + i + half] = lo * sin + hi * cos;
                        }
                    }
                }
            }
        }
    }

    fn project(&mut self, layer: usize, proj: Proj, x: Buf, y: Buf, rows: usize) {
        let Self { weights, bufs, .. } = self;
        let weights = weights.as_deref().expect("prepare() runs before compute");
        let w = proj.matrix(&weights.layers[layer]);
        let (in_dim, out_dim) = (w.in_dim(), w.out_dim());
        let (xv, yv) = two_mut(bufs, x.0, y.0);
        let mut row = vec![0.0f32; in_dim];
        for r in 0..rows {
            let src = &xv[r * in_dim..(r + 1) * in_dim];
            let dst = &mut yv[r * out_dim..(r + 1) * out_dim];
            for (o, slot) in dst.iter_mut().enumerate().take(out_dim) {
                w.decode_row(o, &mut row);
                *slot = dot(&row, src);
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
        for i in 0..len {
            d[i] += s[i];
        }
    }

    fn silu_mul(&mut self, gate: Buf, up: Buf, len: usize) {
        let (g, u) = two_mut(&mut self.bufs, gate.0, up.0);
        for i in 0..len {
            g[i] = silu(g[i]) * u[i];
        }
    }

    fn logits(&mut self, x: Buf, y: Buf) {
        let Self { weights, bufs, .. } = self;
        let weights = weights.as_deref().expect("prepare() runs before compute");
        let w = weights.classifier();
        let (in_dim, out_dim) = (w.in_dim(), w.out_dim());
        let (xv, yv) = two_mut(bufs, x.0, y.0);
        let src = &xv[..in_dim];
        let mut row = vec![0.0f32; in_dim];
        for (o, slot) in yv.iter_mut().enumerate().take(out_dim) {
            w.decode_row(o, &mut row);
            *slot = dot(&row, src);
        }
    }
}

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
    let (shared_start, shared_end) = shared;
    let n_shared = shared_end - shared_start;
    let count = n_shared + slot - att_start + 1;
    let scale = 1.0f32 / (spec.head_size as f32).sqrt();

    for h in 0..spec.n_heads {
        let qh = &q[h * spec.head_size..(h + 1) * spec.head_size];
        let scores = &mut att[h * spec.seq_len..h * spec.seq_len + spec.seq_len];
        let kv = (h / spec.kv_mul) * spec.head_size;

        for (i, t) in (shared_start..shared_end).enumerate() {
            let at = loff + t * spec.kv_dim + kv;
            scores[i] = dot(qh, &key_cache[at..at + spec.head_size]) * scale;
        }
        for (i, t) in (att_start..=slot).enumerate() {
            let at = loff + t * spec.kv_dim + kv;
            scores[n_shared + i] = dot(qh, &key_cache[at..at + spec.head_size]) * scale;
        }

        let scores = &mut scores[..count];
        let mut max = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max {
                max = s;
            }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }

        let out = &mut out[h * spec.head_size..(h + 1) * spec.head_size];
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for (i, t) in (shared_start..shared_end).enumerate() {
            let at = loff + t * spec.kv_dim + kv;
            let a = scores[i];
            for o in 0..spec.head_size {
                out[o] += a * value_cache[at + o];
            }
        }
        for (i, t) in (att_start..=slot).enumerate() {
            let at = loff + t * spec.kv_dim + kv;
            let a = scores[n_shared + i];
            for o in 0..spec.head_size {
                out[o] += a * value_cache[at + o];
            }
        }
    }
}
