//! The Metal backend: weights, activations and the key/value cache all live in
//! device memory.
//!
//! # Why the whole step, not one projection at a time
//!
//! An earlier version handed activations over as host slices, one synchronised
//! dispatch per projection. That is 210 round trips per token, and each one
//! costs about 250us — an order of magnitude more than the arithmetic. Handles
//! fix it: every op encodes into one command buffer, and
//! [`read`](Backend::read) commits and waits. A decode step is one submission.
//!
//! # What runs where
//!
//! Everything runs on the GPU, because everything now depends on data the GPU
//! produced. Two exceptions are deliberate: the token embedding is a row gather
//! the host copies straight into a shared buffer, and `read` is the only place
//! data comes back. Both are safe because nothing is committed until `read`, so
//! host writes always land before the GPU starts.

use std::sync::Arc;

use metal::{
    Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

use super::{AttnSpec, Backend, Buf, KvSpec, ModelWeights, Proj, RopeSpec, Window};

/// Output tile per threadgroup for the batched path: 16x16 threads, 4x4 each.
const BLOCK_M: u64 = 64;
const BLOCK_N: u64 = 64;

const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// bf16 is the top half of an f32, so widening is a shift.
inline float bf16_to_f32(ushort b) { return as_type<float>((uint)b << 16); }

constant uint TILE  = 16;
constant uint TM    = 4;
constant uint TN    = 4;
constant uint BM    = TILE * TM;
constant uint BN    = TILE * TN;
constant uint BK    = 16;

// c[m, n] = a[m, k] * b[n, k]^T. Both operands are row-major and contiguous
// along k, which is the layout the projection weights already have.
kernel void gemm_nt(device const float* a   [[buffer(0)]],
                    device const ushort* b  [[buffer(1)]],
                    device float* out       [[buffer(2)]],
                    constant uint& m        [[buffer(3)]],
                    constant uint& n        [[buffer(4)]],
                    constant uint& k        [[buffer(5)]],
                    uint2 tid [[thread_position_in_threadgroup]],
                    uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float ash[BM][BK];
    threadgroup float bsh[BN][BK];

    uint row0 = bid.y * BM + tid.y * TM;
    uint col0 = bid.x * BN + tid.x * TN;

    float acc[TM][TN];
    for (uint r = 0; r < TM; ++r)
        for (uint c = 0; c < TN; ++c) acc[r][c] = 0.0f;

    uint full = k / BK;
    for (uint t = 0; t < full; ++t) {
        for (uint s = 0; s < 4; ++s) {
            uint idx = tid.y * TILE + tid.x + s * 256;
            uint r = idx >> 4;
            uint kk = idx & (BK - 1);
            uint gr = min(bid.y * BM + r, m - 1);
            uint gc = min(bid.x * BN + r, n - 1);
            ash[r][kk] = a[gr * k + t * BK + kk];
            bsh[r][kk] = bf16_to_f32(b[gc * k + t * BK + kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < BK; ++i) {
            float av[TM], bv[TN];
            for (uint r = 0; r < TM; ++r) av[r] = ash[tid.y * TM + r][i];
            for (uint c = 0; c < TN; ++c) bv[c] = bsh[tid.x * TN + c][i];
            for (uint r = 0; r < TM; ++r)
                for (uint c = 0; c < TN; ++c)
                    acc[r][c] = fma(av[r], bv[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = full * BK; i < k; ++i) {
        for (uint r = 0; r < TM; ++r) {
            float av = a[min(row0 + r, m - 1) * k + i];
            for (uint c = 0; c < TN; ++c)
                acc[r][c] = fma(av, bf16_to_f32(b[min(col0 + c, n - 1) * k + i]), acc[r][c]);
        }
    }

    for (uint r = 0; r < TM; ++r) {
        if (row0 + r >= m) break;
        for (uint c = 0; c < TN; ++c) {
            if (col0 + c >= n) break;
            out[(row0 + r) * n + col0 + c] = acc[r][c];
        }
    }
}

// Decode: one row in, so the tile above would discard 63 of every 64 rows.
kernel void gemv(device const ushort* w  [[buffer(0)]],
                 device const float* x   [[buffer(1)]],
                 device float* out       [[buffer(2)]],
                 constant uint& n        [[buffer(3)]],
                 constant uint& k        [[buffer(4)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    device const ushort* wp = w + gid * k;
    uint k4 = k >> 2;
    device const float4* x4 = (device const float4*)x;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < k4; ++i) {
        uint o = i << 2;
        float4 xv = x4[i];
        acc.x = fma(bf16_to_f32(wp[o]), xv.x, acc.x);
        acc.y = fma(bf16_to_f32(wp[o + 1]), xv.y, acc.y);
        acc.z = fma(bf16_to_f32(wp[o + 2]), xv.z, acc.z);
        acc.w = fma(bf16_to_f32(wp[o + 3]), xv.w, acc.w);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    for (uint i = k4 << 2; i < k; ++i) sum = fma(bf16_to_f32(wp[i]), x[i], sum);
    out[gid] = sum;
}

kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device float* out     [[buffer(2)]],
                     constant uint& dim    [[buffer(3)]],
                     constant float& eps   [[buffer(4)]],
                     uint row [[thread_position_in_grid]]) {
    uint base = row * dim;
    float ss = 0.0f;
    for (uint i = 0; i < dim; ++i) { float v = x[base + i]; ss += v * v; }
    ss = ss / float(dim) + eps;
    float scale = rsqrt(ss);
    for (uint i = 0; i < dim; ++i) out[base + i] = w[i] * (scale * x[base + i]);
}

// One thread per head slice.
kernel void qk_norm(device float* v        [[buffer(0)]],
                    device const float* w  [[buffer(1)]],
                    constant uint& head    [[buffer(2)]],
                    constant float& eps    [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    uint base = gid * head;
    float ss = 0.0f;
    for (uint i = 0; i < head; ++i) { float x = v[base + i]; ss += x * x; }
    ss = ss / float(head) + eps;
    float scale = rsqrt(ss);
    for (uint i = 0; i < head; ++i) v[base + i] = w[i] * (scale * v[base + i]);
}

// Half-split RoPE: pair (i, i + head/2) inside each head.
// grid = (rows, heads * head/2)
kernel void rope_half(device float* v            [[buffer(0)]],
                      device const uint* pos     [[buffer(1)]],
                      constant uint& width       [[buffer(2)]],
                      constant uint& head        [[buffer(3)]],
                      constant float& theta      [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    // `half` is a type name in MSL, hence `hd2`.
    uint hd2 = head >> 1;
    uint row = gid.x;
    uint pair = gid.y;
    uint h = pair / hd2;
    uint i = pair % hd2;
    float freq = precise::pow(theta, -2.0f * float(i) / float(head));
    float angle = float(pos[row]) * freq;
    float c = precise::cos(angle);
    float s = precise::sin(angle);
    uint at = row * width + h * head + i;
    float lo = v[at];
    float hi = v[at + hd2];
    v[at] = lo * c - hi * s;
    v[at + hd2] = lo * s + hi * c;
}

// Adjacent-pair RoPE (llama2.c): for even i, the pair (i, i+1).
// grid = (rows, width/2)
kernel void rope_pair(device float* v            [[buffer(0)]],
                      device const uint* pos     [[buffer(1)]],
                      constant uint& width       [[buffer(2)]],
                      constant uint& head        [[buffer(3)]],
                      constant float& theta      [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint i = gid.y * 2;
    uint head_dim = i % head;
    float freq = precise::pow(theta, -float(head_dim) / float(head));
    float angle = float(pos[row]) * freq;
    float c = precise::cos(angle);
    float s = precise::sin(angle);
    uint at = row * width + i;
    float x0 = v[at];
    float x1 = v[at + 1];
    v[at] = x0 * c - x1 * s;
    v[at + 1] = x0 * s + x1 * c;
}

// One thread per (row, head): score, softmax, leave the weights in `scores`.
kernel void attn_scores(device const float* q         [[buffer(0)]],
                        device const float* kc        [[buffer(1)]],
                        device float* scores          [[buffer(2)]],
                        constant uint& n_heads        [[buffer(3)]],
                        constant uint& head           [[buffer(4)]],
                        constant uint& kv_dim         [[buffer(5)]],
                        constant uint& kv_mul         [[buffer(6)]],
                        constant uint& seq_len        [[buffer(7)]],
                        constant uint& loff           [[buffer(8)]],
                        device const uint4* windows   [[buffer(9)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint h = gid.x;
    uint row = gid.y;
    if (h >= n_heads) return;
    uint4 w = windows[row];
    uint att_start = w.x, slot = w.y, shared_start = w.z, shared_end = w.w;
    uint n_shared = shared_end - shared_start;
    uint count = n_shared + slot - att_start + 1;

    uint kv = (h / kv_mul) * head;
    device const float* qh = q + (row * n_heads + h) * head;
    device float* sc = scores + (row * n_heads + h) * seq_len;
    float scale = rsqrt(float(head));

    uint i = 0;
    for (uint t = shared_start; t < shared_end; ++t, ++i) {
        device const float* kh = kc + loff + t * kv_dim + kv;
        float s = 0.0f;
        for (uint d = 0; d < head; ++d) s = fma(qh[d], kh[d], s);
        sc[i] = s * scale;
    }
    for (uint t = att_start; t <= slot; ++t, ++i) {
        device const float* kh = kc + loff + t * kv_dim + kv;
        float s = 0.0f;
        for (uint d = 0; d < head; ++d) s = fma(qh[d], kh[d], s);
        sc[i] = s * scale;
    }

    float mx = -INFINITY;
    for (uint j = 0; j < count; ++j) mx = max(mx, sc[j]);
    float sum = 0.0f;
    for (uint j = 0; j < count; ++j) { float e = exp(sc[j] - mx); sc[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (uint j = 0; j < count; ++j) sc[j] *= inv;
}

// grid = (head_size, n_heads, rows)
kernel void attn_apply(device const float* scores     [[buffer(0)]],
                       device const float* vc        [[buffer(1)]],
                       device float* out             [[buffer(2)]],
                       constant uint& n_heads        [[buffer(3)]],
                       constant uint& head           [[buffer(4)]],
                       constant uint& kv_dim         [[buffer(5)]],
                       constant uint& kv_mul         [[buffer(6)]],
                       constant uint& seq_len        [[buffer(7)]],
                       constant uint& loff           [[buffer(8)]],
                       device const uint4* windows   [[buffer(9)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint o = gid.x;
    uint h = gid.y;
    uint row = gid.z;
    if (o >= head) return;
    uint4 w = windows[row];
    uint att_start = w.x, slot = w.y, shared_start = w.z, shared_end = w.w;
    uint n_shared = shared_end - shared_start;

    uint kv = (h / kv_mul) * head;
    device const float* sc = scores + (row * n_heads + h) * seq_len;
    float acc = 0.0f;
    uint i = 0;
    for (uint t = shared_start; t < shared_end; ++t, ++i) {
        acc = fma(sc[i], vc[loff + t * kv_dim + kv + o], acc);
    }
    for (uint t = att_start; t <= slot; ++t, ++i) {
        acc = fma(sc[n_shared + i], vc[loff + t * kv_dim + kv + o], acc);
    }
    out[(row * n_heads + h) * head + o] = acc;
}

kernel void add_in_place(device float* d       [[buffer(0)]],
                         device const float* s [[buffer(1)]],
                         constant uint& n      [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) d[i] += s[i];
}

// The cache write has to be a dispatch, not a host memcpy: the k/v it reads
// were produced by a projection that has only been *encoded* so far.
// One thread per row: the copy is `kv_dim` elements, too small to split.
kernel void kv_write(device const float* k    [[buffer(0)]],
                     device const float* v    [[buffer(1)]],
                     device float* kc         [[buffer(2)]],
                     device float* vc         [[buffer(3)]],
                     device const uint* slots [[buffer(4)]],
                     constant uint& width     [[buffer(5)]],
                     constant uint& seq_len   [[buffer(6)]],
                     constant uint& layer     [[buffer(7)]],
                     uint row [[thread_position_in_grid]]) {
    uint at = (layer * seq_len + slots[row]) * width;
    uint src = row * width;
    for (uint i = 0; i < width; ++i) {
        kc[at + i] = k[src + i];
        vc[at + i] = v[src + i];
    }
}

kernel void copy_f32(device const float* s   [[buffer(0)]],
                     device float* d         [[buffer(1)]],
                     constant uint& n        [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) d[i] = s[i];
}

kernel void silu_mul(device float* g         [[buffer(0)]],
                     device const float* u   [[buffer(1)]],
                     constant uint& n        [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) g[i] = (g[i] / (1.0f + exp(-g[i]))) * u[i];
}
"#;

#[derive(Clone, Copy, Debug)]
struct Layout {
    offset: usize,
    out_dim: usize,
    in_dim: usize,
}

pub struct Metal {
    device: Device,
    queue: CommandQueue,
    gemm_pipeline: ComputePipelineState,
    gemv_pipeline: ComputePipelineState,
    rms_pipeline: ComputePipelineState,
    qk_pipeline: ComputePipelineState,
    rope_half_pipeline: ComputePipelineState,
    rope_pair_pipeline: ComputePipelineState,
    attn_scores_pipeline: ComputePipelineState,
    attn_apply_pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
    silu_pipeline: ComputePipelineState,
    copy_pipeline: ComputePipelineState,
    kv_write_pipeline: ComputePipelineState,
    /// Every projection of every layer plus the classifier, concatenated.
    weights: Buffer,
    layers: Vec<[Layout; 7]>,
    classifier: Layout,
    weights_host: Option<Arc<ModelWeights>>,
    bufs: Vec<Buffer>,
    key_cache: Buffer,
    value_cache: Buffer,
    /// Per-row RoPE positions, rewritten before each layer (same content).
    positions: Buffer,
    positions_host: Vec<u32>,
    windows: Buffer,
    windows_host: Vec<u32>,
    kv: KvSpec,
    /// The open command buffer. Committed and waited in `read`.
    command: Option<CommandBuffer>,
}

fn pipeline(
    device: &Device,
    library: &metal::Library,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|error| format!("cannot find kernel {name}: {error}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("cannot build pipeline {name}: {error}"))
}

impl Metal {
    /// Open the system GPU. Returns the error text rather than panicking: on a
    /// machine without one the caller should fall back, not abort.
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device on this machine")?;
        let library = device
            .new_library_with_source(SHADERS, &CompileOptions::new())
            .map_err(|error| format!("cannot compile Metal shaders: {error}"))?;
        let placeholder = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        Ok(Metal {
            gemm_pipeline: pipeline(&device, &library, "gemm_nt")?,
            gemv_pipeline: pipeline(&device, &library, "gemv")?,
            rms_pipeline: pipeline(&device, &library, "rms_norm")?,
            qk_pipeline: pipeline(&device, &library, "qk_norm")?,
            rope_half_pipeline: pipeline(&device, &library, "rope_half")?,
            rope_pair_pipeline: pipeline(&device, &library, "rope_pair")?,
            attn_scores_pipeline: pipeline(&device, &library, "attn_scores")?,
            attn_apply_pipeline: pipeline(&device, &library, "attn_apply")?,
            add_pipeline: pipeline(&device, &library, "add_in_place")?,
            silu_pipeline: pipeline(&device, &library, "silu_mul")?,
            copy_pipeline: pipeline(&device, &library, "copy_f32")?,
            kv_write_pipeline: pipeline(&device, &library, "kv_write")?,
            queue: device.new_command_queue(),
            weights: placeholder.clone(),
            key_cache: placeholder.clone(),
            value_cache: placeholder.clone(),
            positions: placeholder.clone(),
            positions_host: Vec::new(),
            windows: placeholder,
            windows_host: Vec::new(),
            kv: KvSpec {
                n_layers: 0,
                kv_dim: 0,
                seq_len: 0,
            },
            device,
            layers: Vec::new(),
            classifier: Layout {
                offset: 0,
                out_dim: 0,
                in_dim: 0,
            },
            weights_host: None,
            bufs: Vec::new(),
            command: None,
        })
    }

    fn weights(&self) -> &ModelWeights {
        self.weights_host
            .as_deref()
            .expect("prepare() runs before compute")
    }

    /// Encode one dispatch into the open command buffer.
    fn encode(&mut self, f: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
        if self.command.is_none() {
            self.command = Some(self.queue.new_command_buffer().to_owned());
        }
        let command = self.command.as_ref().expect("just created");
        let encoder = command.new_compute_command_encoder();
        f(encoder);
        encoder.end_encoding();
    }

    /// Submit everything encoded so far and wait for it.
    fn flush(&mut self) {
        if let Some(command) = self.command.take() {
            command.commit();
            command.wait_until_completed();
        }
    }

    fn buf(&self, b: Buf) -> Buffer {
        self.bufs[b.0].clone()
    }

    fn upload_weights(&mut self, weights: &ModelWeights) {
        let mut flat: Vec<u16> = Vec::new();
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let mut layouts = [Layout {
                offset: 0,
                out_dim: 0,
                in_dim: 0,
            }; 7];
            for proj in Proj::ALL {
                let matrix = proj.matrix(layer);
                let offset = flat.len();
                flat.extend_from_slice(&matrix.to_bf16_bits());
                layouts[proj.index()] = Layout {
                    offset,
                    out_dim: matrix.out_dim(),
                    in_dim: matrix.in_dim(),
                };
            }
            layers.push(layouts);
        }
        let classifier = weights.classifier();
        let offset = flat.len();
        flat.extend_from_slice(&classifier.to_bf16_bits());
        let classifier_layout = Layout {
            offset,
            out_dim: classifier.out_dim(),
            in_dim: classifier.in_dim(),
        };

        self.weights = self.device.new_buffer_with_data(
            flat.as_ptr() as *const _,
            (flat.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.layers = layers;
        self.classifier = classifier_layout;
    }

    /// Encode `y[rows, out] = x[rows, in] * w[out, in]^T`.
    fn dispatch(&mut self, layout: Layout, x: Buf, y: Buf, rows: usize) {
        let a = self.buf(x);
        let c = self.buf(y);
        let weights = self.weights.clone();
        let offset = layout.offset as u64 * 2;
        let n = layout.out_dim as u32;
        let k = layout.in_dim as u32;
        let m = rows as u32;
        let vector = rows == 1 && layout.in_dim % 4 == 0;
        let gemm = self.gemm_pipeline.clone();
        let gemv = self.gemv_pipeline.clone();

        self.encode(move |enc| {
            if vector {
                enc.set_compute_pipeline_state(&gemv);
                enc.set_buffer(0, Some(&weights), offset);
                enc.set_buffer(1, Some(&a), 0);
                enc.set_buffer(2, Some(&c), 0);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
                enc.set_bytes(4, 4, &k as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(layout.out_dim as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&gemm);
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&weights), offset);
                enc.set_buffer(2, Some(&c), 0);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (layout.out_dim as u64).div_ceil(BLOCK_N),
                        (rows as u64).div_ceil(BLOCK_M),
                        1,
                    ),
                    MTLSize::new(16, 16, 1),
                );
            }
        });
    }
}

impl Backend for Metal {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn storage(&self) -> &'static str {
        self.weights()
            .layers
            .first()
            .map_or("f32", |layer| layer.wq.kind())
    }

    fn prepare(&mut self, weights: Arc<ModelWeights>, kv: KvSpec) {
        self.upload_weights(&weights);
        self.kv = kv;
        let bytes = |n: usize| (n.max(4) * 4) as u64;
        let shared = MTLResourceOptions::StorageModeShared;
        self.key_cache = self
            .device
            .new_buffer(bytes(kv.n_layers * kv.seq_len * kv.kv_dim), shared);
        self.value_cache = self
            .device
            .new_buffer(bytes(kv.n_layers * kv.seq_len * kv.kv_dim), shared);
        self.positions = self.device.new_buffer(bytes(kv.seq_len), shared);
        self.positions_host = vec![0; kv.seq_len];
        self.weights_host = Some(weights);
    }

    fn alloc(&mut self, len: usize) -> Buf {
        let buffer = self.device.new_buffer(
            (len.max(4) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.bufs.push(buffer);
        Buf(self.bufs.len() - 1)
    }

    fn upload(&mut self, data: &[f32]) -> Buf {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len().max(1) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.bufs.push(buffer);
        Buf(self.bufs.len() - 1)
    }

    fn read(&mut self, src: Buf, out: &mut [f32]) {
        self.flush();
        let buffer = &self.bufs[src.0];
        // SAFETY: `StorageModeShared` memory is host-mapped, and the copy is
        // bounded by the caller's slice.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents() as *const f32,
                out.as_mut_ptr(),
                out.len(),
            );
        }
    }

    fn embed(&mut self, token: usize, out: Buf, offset: usize) {
        // A row gather, not a product. The host writes directly into shared
        // device memory; nothing is committed yet, so there is no race.
        let Self {
            weights_host, bufs, ..
        } = self;
        let weights = weights_host
            .as_ref()
            .expect("prepare() runs before compute");
        let dim = weights.token_embedding.in_dim();
        let mut row = vec![0.0f32; dim];
        weights.token_embedding.decode_row(token, &mut row);
        // SAFETY: the destination was allocated with at least `offset + dim`
        // elements by the driver, and shared memory is host-mapped.
        unsafe {
            std::ptr::copy_nonoverlapping(
                row.as_ptr(),
                (bufs[out.0].contents() as *mut f32).add(offset),
                dim,
            );
        }
    }

    fn copy(&mut self, src: Buf, src_offset: usize, dst: Buf, dst_offset: usize, len: usize) {
        let s = self.buf(src);
        let d = self.buf(dst);
        let n = len as u32;
        let pipeline = self.copy_pipeline.clone();
        let scale = 4u64;
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&s), src_offset as u64 * scale);
            enc.set_buffer(1, Some(&d), dst_offset as u64 * scale);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    fn kv_write(&mut self, layer: usize, slots: &[usize], k: Buf, v: Buf, rows: usize) {
        let width = self.kv.kv_dim;
        let kb = self.buf(k);
        let vb = self.buf(v);
        let kc = self.key_cache.clone();
        let vc = self.value_cache.clone();
        let slot_values: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
        let slot_buf = self.device.new_buffer_with_data(
            slot_values.as_ptr() as *const _,
            (rows.max(1) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w = width as u32;
        let seq_len = self.kv.seq_len as u32;
        let layer = layer as u32;
        let pipeline = self.kv_write_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&kb), 0);
            enc.set_buffer(1, Some(&vb), 0);
            enc.set_buffer(2, Some(&kc), 0);
            enc.set_buffer(3, Some(&vc), 0);
            enc.set_buffer(4, Some(&slot_buf), 0);
            enc.set_bytes(5, 4, &w as *const u32 as *const _);
            enc.set_bytes(6, 4, &seq_len as *const u32 as *const _);
            enc.set_bytes(7, 4, &layer as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    fn rms_norm(&mut self, out: Buf, x: Buf, weight: Buf, rows: usize, dim: usize, eps: f32) {
        let xr = self.buf(x);
        let wr = self.buf(weight);
        let or = self.buf(out);
        let d = dim as u32;
        let pipeline = self.rms_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&xr), 0);
            enc.set_buffer(1, Some(&wr), 0);
            enc.set_buffer(2, Some(&or), 0);
            enc.set_bytes(3, 4, &d as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
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
        let vbuf = self.buf(v);
        let wbuf = self.buf(weight);
        let head = head_size as u32;
        let pipeline = self.qk_pipeline.clone();
        let count = (rows * width / head_size) as u64;
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&vbuf), 0);
            enc.set_buffer(1, Some(&wbuf), 0);
            enc.set_bytes(2, 4, &head as *const u32 as *const _);
            enc.set_bytes(3, 4, &eps as *const f32 as *const _);
            enc.dispatch_threads(MTLSize::new(count, 1, 1), MTLSize::new(64, 1, 1));
        });
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
        if self.positions_host.len() < rows {
            self.positions_host.resize(rows, 0);
            self.positions = self.device.new_buffer(
                (rows.max(4) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
        }
        for (i, &p) in positions.iter().enumerate() {
            self.positions_host[i] = p as u32;
        }
        // SAFETY: `positions_host` holds at least `rows` entries and is the
        // mapping for the shared device buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.positions_host.as_ptr(),
                self.positions.contents() as *mut u32,
                rows,
            );
        }
        let pos = self.positions.clone();
        let (qb, kb) = (self.buf(q), self.buf(k));
        let head = spec.head_size as u32;
        let theta = spec.theta;
        let (pipeline, half) = if spec.interleaved {
            (self.rope_pair_pipeline.clone(), false)
        } else {
            (self.rope_half_pipeline.clone(), true)
        };
        let qw = q_width as u32;
        let kw = kv_width as u32;
        let q_heads = (q_width / spec.head_size) as u64;
        let k_heads = (kv_width / spec.head_size) as u64;
        let q_threads = if half {
            q_heads * (spec.head_size / 2) as u64
        } else {
            q_width as u64 / 2
        };
        let k_threads = if half {
            k_heads * (spec.head_size / 2) as u64
        } else {
            kv_width as u64 / 2
        };
        let pipe_q = pipeline.clone();
        let pipe_k = pipeline;
        let pos2 = pos.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipe_q);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&pos), 0);
            enc.set_bytes(2, 4, &qw as *const u32 as *const _);
            enc.set_bytes(3, 4, &head as *const u32 as *const _);
            enc.set_bytes(4, 4, &theta as *const f32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(rows as u64, q_threads, 1),
                MTLSize::new(8, 8, 1),
            );
        });
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipe_k);
            enc.set_buffer(0, Some(&kb), 0);
            enc.set_buffer(1, Some(&pos2), 0);
            enc.set_bytes(2, 4, &kw as *const u32 as *const _);
            enc.set_bytes(3, 4, &head as *const u32 as *const _);
            enc.set_bytes(4, 4, &theta as *const f32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(rows as u64, k_threads, 1),
                MTLSize::new(8, 8, 1),
            );
        });
    }

    fn project(&mut self, layer: usize, proj: Proj, x: Buf, y: Buf, rows: usize) {
        let layout = self.layers[layer][proj.index()];
        self.dispatch(layout, x, y, rows);
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
        let rows = windows.len();
        let needed = rows * 4;
        if self.windows_host.len() < needed {
            self.windows_host.resize(needed, 0);
            self.windows = self.device.new_buffer(
                (needed.max(4) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
        }
        for (i, w) in windows.iter().enumerate() {
            self.windows_host[i * 4] = w.att_start as u32;
            self.windows_host[i * 4 + 1] = w.slot as u32;
            self.windows_host[i * 4 + 2] = w.shared.0 as u32;
            self.windows_host[i * 4 + 3] = w.shared.1 as u32;
        }
        // SAFETY: `windows_host` holds at least `needed` entries and is the
        // mapping for the shared device buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.windows_host.as_ptr(),
                self.windows.contents() as *mut u32,
                needed,
            );
        }

        let qb = self.buf(q);
        let ob = self.buf(out);
        let scores = self.buf(att);
        let kc = self.key_cache.clone();
        let vc = self.value_cache.clone();
        let win = self.windows.clone();
        let win2 = win.clone();
        let loff = (layer * spec.seq_len * spec.kv_dim) as u32;
        let (n_heads, head, kv_dim, kv_mul, seq_len) = (
            spec.n_heads as u32,
            spec.head_size as u32,
            spec.kv_dim as u32,
            spec.kv_mul as u32,
            spec.seq_len as u32,
        );
        let scores_pipeline = self.attn_scores_pipeline.clone();
        let apply_pipeline = self.attn_apply_pipeline.clone();
        let scores2 = scores.clone();

        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&scores_pipeline);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kc), 0);
            enc.set_buffer(2, Some(&scores), 0);
            for (i, v) in [n_heads, head, kv_dim, kv_mul, seq_len, loff]
                .iter()
                .enumerate()
            {
                enc.set_bytes(3 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.set_buffer(9, Some(&win), 0);
            enc.dispatch_threads(
                MTLSize::new(spec.n_heads as u64, rows as u64, 1),
                MTLSize::new(32, 8, 1),
            );
        });
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&apply_pipeline);
            enc.set_buffer(0, Some(&scores2), 0);
            enc.set_buffer(1, Some(&vc), 0);
            enc.set_buffer(2, Some(&ob), 0);
            for (i, v) in [n_heads, head, kv_dim, kv_mul, seq_len, loff]
                .iter()
                .enumerate()
            {
                enc.set_bytes(3 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.set_buffer(9, Some(&win2), 0);
            enc.dispatch_threads(
                MTLSize::new(spec.head_size as u64, spec.n_heads as u64, rows as u64),
                MTLSize::new(32, 8, 1),
            );
        });
    }

    fn add_in_place(&mut self, dst: Buf, src: Buf, len: usize) {
        let d = self.buf(dst);
        let s = self.buf(src);
        let n = len as u32;
        let pipeline = self.add_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&d), 0);
            enc.set_buffer(1, Some(&s), 0);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    fn silu_mul(&mut self, gate: Buf, up: Buf, len: usize) {
        let g = self.buf(gate);
        let u = self.buf(up);
        let n = len as u32;
        let pipeline = self.silu_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&g), 0);
            enc.set_buffer(1, Some(&u), 0);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    fn logits(&mut self, x: Buf, y: Buf) {
        let layout = self.classifier;
        self.dispatch(layout, x, y, 1);
    }
}

impl std::fmt::Debug for Metal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metal")
            .field("device", &self.device.name())
            .finish()
    }
}

impl Metal {
    /// The device name, for a benchmark table.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each kernel against the arithmetic it replaced. `project` needs a
    /// prepared model and is covered through the integration tests; these do
    /// not, so a wrong shader fails here first and points at the op.
    #[test]
    fn kernels_match_their_reference() {
        let Ok(mut m) = Metal::new() else {
            eprintln!("skipping: no Metal device");
            return;
        };

        // rms_norm
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32, 1.0, 1.0, 1.0];
        let xb = m.upload(&x);
        let wb = m.upload(&w);
        let out = m.alloc(4);
        m.rms_norm(out, xb, wb, 1, 4, 0.0);
        let mut got = vec![0.0; 4];
        m.read(out, &mut got);
        let scale = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        for i in 0..4 {
            assert!(
                (got[i] - x[i] * scale).abs() < 1e-5,
                "rms[{i}] {} vs {}",
                got[i],
                x[i] * scale
            );
        }

        // qk_norm, two heads of size 2
        let v = vec![3.0f32, 4.0, 0.0, 2.0];
        let wq = vec![1.0f32, 1.0];
        let vb = m.upload(&v);
        let wq = m.upload(&wq);
        m.qk_norm(vb, 1, 4, 2, wq, 0.0);
        let mut got = vec![0.0; 4];
        m.read(vb, &mut got);
        // [3,4]: RMS sqrt(12.5); [0,2]: RMS sqrt(2)
        assert!(
            (got[0] - 3.0 / 12.5f32.sqrt()).abs() < 1e-5,
            "qk0 {}",
            got[0]
        );
        assert!(
            (got[3] - 2.0 / 2.0f32.sqrt()).abs() < 1e-5,
            "qk3 {}",
            got[3]
        );

        // silu_mul
        let g = vec![1.0f32, -1.0];
        let u = vec![2.0f32, 2.0];
        let gb = m.upload(&g);
        let ub = m.upload(&u);
        m.silu_mul(gb, ub, 2);
        let mut got = vec![0.0; 2];
        m.read(gb, &mut got);
        let silu = |x: f32| x / (1.0 + (-x).exp());
        assert!((got[0] - silu(1.0) * 2.0).abs() < 1e-5, "silu {}", got[0]);

        // copy
        let a = m.upload(&[7.0f32, 8.0, 9.0]);
        let b = m.alloc(3);
        m.copy(a, 1, b, 0, 2);
        let mut got = vec![0.0; 2];
        m.read(b, &mut got);
        assert_eq!(got, vec![8.0, 9.0], "copy");

        // rope (half-split), head_size 4, pos 3
        let q = vec![0.3f32, -0.7, 1.1, 0.2];
        let k = vec![0.0f32; 4];
        let qbuf = m.upload(&q);
        let kbuf = m.upload(&k);
        m.rope(
            qbuf,
            kbuf,
            4,
            4,
            1,
            &[3],
            RopeSpec {
                head_size: 4,
                theta: 10_000.0,
                interleaved: false,
            },
        );
        let mut got = vec![0.0; 4];
        m.read(qbuf, &mut got);
        let half = 2;
        for i in 0..half {
            let freq = 1.0f32 / 10_000f32.powf((2 * i) as f32 / 4.0);
            let (s, c) = (3.0f32 * freq).sin_cos();
            let lo = q[i];
            let hi = q[i + half];
            assert!(
                (got[i] - (lo * c - hi * s)).abs() < 1e-5,
                "rope lo {i}: {}",
                got[i]
            );
            assert!(
                (got[i + half] - (lo * s + hi * c)).abs() < 1e-5,
                "rope hi {i}"
            );
        }

        // add_in_place
        let d = m.upload(&[1.0f32, 2.0]);
        let s2 = m.upload(&[10.0f32, 20.0]);
        m.add_in_place(d, s2, 2);
        let mut got = vec![0.0; 2];
        m.read(d, &mut got);
        assert_eq!(got, vec![11.0, 22.0], "add");
    }
}
