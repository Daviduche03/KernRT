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
use crate::formats::llama2c::LayerWeights;

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

// y = W x with row-major Q8_0 blocks: 32 int8 + one f32 scale per block.
// Activations stay f32 — only the weights are quantized, so unlike the CPU
// integer path there is no activation quantization error. One thread per row.
kernel void gemv_q8(device const char* w    [[buffer(0)]],
                    device const float* s   [[buffer(1)]],
                    device const float* x   [[buffer(2)]],
                    device float* out       [[buffer(3)]],
                    constant uint& n        [[buffer(4)]],
                    constant uint& k        [[buffer(5)]],
                    constant uint& nb       [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    device const char* wp = w + gid * k;
    device const float* sp = s + gid * nb;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float sum = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            sum = fma(float(wp[b * 32 + i]), x[b * 32 + i], sum);
        }
        acc = fma(sp[b], sum, acc);
    }
    // Rows are block-aligned on every quantized model the loader accepts, so
    // there is no tail: k is a whole number of blocks by construction.
    out[gid] = acc;
}

// Batched Q8_0 GEMM: c[m, n] = a[m, k] * w[n, k]^T with the same row-major
// blocks. Same 64x64 output tile and 4x4 register block as `gemm_nt`, but
// the k step is one 16-wide slice of a 32-wide block: tile t of output row r
// reads exactly one scale, s[r * nb + t / 2]. Activations stay f32.
kernel void gemm_q8(device const float* a   [[buffer(0)]],
                    device const char* b    [[buffer(1)]],
                    device const float* s   [[buffer(2)]],
                    device float* out       [[buffer(3)]],
                    constant uint& m        [[buffer(4)]],
                    constant uint& n        [[buffer(5)]],
                    constant uint& k        [[buffer(6)]],
                    constant uint& nb       [[buffer(7)]],
                    uint2 tid [[thread_position_in_threadgroup]],
                    uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float ash[BM][BK];
    threadgroup char bsh[BN][BK];

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
            bsh[r][kk] = b[gc * k + t * BK + kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bacc[TM][TN];
        for (uint r = 0; r < TM; ++r)
            for (uint c = 0; c < TN; ++c) bacc[r][c] = 0.0f;
        for (uint i = 0; i < BK; ++i) {
            float av[TM], bv[TN];
            for (uint r = 0; r < TM; ++r) av[r] = ash[tid.y * TM + r][i];
            for (uint c = 0; c < TN; ++c) bv[c] = float(bsh[tid.x * TN + c][i]);
            for (uint r = 0; r < TM; ++r)
                for (uint c = 0; c < TN; ++c)
                    bacc[r][c] = fma(av[r], bv[c], bacc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One scale per output column for this tile: tile t covers k
        // [t*16, t*16+16), which lies inside block t/2. The scale belongs to
        // the weight row, i.e. the output column — not the output row.
        for (uint c = 0; c < TN; ++c) {
            uint gcol = col0 + c;
            float sc = (gcol < n) ? s[gcol * nb + t / 2] : 0.0f;
            for (uint r = 0; r < TM; ++r) acc[r][c] = fma(sc, bacc[r][c], acc[r][c]);
        }
    }

    for (uint i = full * BK; i < k; ++i) {
        uint blk = i >> 5;
        for (uint r = 0; r < TM; ++r) {
            uint grow = min(row0 + r, m - 1);
            float av = a[grow * k + i];
            for (uint c = 0; c < TN; ++c) {
                uint gcol = min(col0 + c, n - 1);
                float sc = (row0 + r < m && col0 + c < n && blk < nb) ? s[gcol * nb + blk] : 0.0f;
                acc[r][c] = fma(sc * av, float(b[gcol * k + i]), acc[r][c]);
            }
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

// Fused QKV / GateUp decode: one thread per fused output row, routed to
// one of three output buffers by row range. The slices are [n0 | n1 | n2]
// rows wide (GateUp leaves n2 at zero). Same arithmetic as three GEMVs.
kernel void gemv_qkv(device const ushort* w   [[buffer(0)]],
                     device float* o0         [[buffer(1)]],
                     device float* o1         [[buffer(2)]],
                     device float* o2         [[buffer(3)]],
                     device const float* x    [[buffer(4)]],
                     constant uint& n0        [[buffer(5)]],
                     constant uint& n1        [[buffer(6)]],
                     constant uint& n2        [[buffer(7)]],
                     constant uint& k         [[buffer(8)]],
                     uint gid [[thread_position_in_grid]]) {
    uint n = n0 + n1 + n2;
    if (gid >= n) return;
    uint row = gid;
    uint base = 0;
    device float* out = o0;
    if (gid >= n0 + n1) {
        out = o2;
        row = gid - n0 - n1;
        base = n0 + n1;
    } else if (gid >= n0) {
        out = o1;
        row = gid - n0;
        base = n0;
    }
    device const ushort* wp = w + (base + row) * k;
    uint k4 = k >> 2;
    device const float4* x4 = (device const float4*)x;
    device const ushort4* w4 = (device const ushort4*)wp;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < k4; ++i) {
        float4 xv = x4[i];
        ushort4 wv = w4[i];
        acc.x = fma(bf16_to_f32(wv.x), xv.x, acc.x);
        acc.y = fma(bf16_to_f32(wv.y), xv.y, acc.y);
        acc.z = fma(bf16_to_f32(wv.z), xv.z, acc.z);
        acc.w = fma(bf16_to_f32(wv.w), xv.w, acc.w);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    for (uint i = k4 << 2; i < k; ++i) sum = fma(bf16_to_f32(wp[i]), x[i], sum);
    out[row] = sum;
}

// Fused Q8 QKV / GateUp decode: same routing, int8 weights widened on the
// fly against the per-row f32 scales. Activations stay f32.
kernel void gemv_q8_qkv(device const char* w    [[buffer(0)]],
                        device const float* s   [[buffer(1)]],
                        device float* o0        [[buffer(2)]],
                        device float* o1        [[buffer(3)]],
                        device float* o2        [[buffer(4)]],
                        device const float* x   [[buffer(5)]],
                        constant uint& n0       [[buffer(6)]],
                        constant uint& n1       [[buffer(7)]],
                        constant uint& n2       [[buffer(8)]],
                        constant uint& k        [[buffer(9)]],
                        constant uint& nb       [[buffer(10)]],
                        uint gid [[thread_position_in_grid]]) {
    uint n = n0 + n1 + n2;
    if (gid >= n) return;
    uint row = gid;
    uint base = 0;
    device float* out = o0;
    if (gid >= n0 + n1) {
        out = o2;
        row = gid - n0 - n1;
        base = n0 + n1;
    } else if (gid >= n0) {
        out = o1;
        row = gid - n0;
        base = n0;
    }
    device const char* wp = w + (base + row) * k;
    device const float* sp = s + (base + row) * nb;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float sum = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            sum = fma(float(wp[b * 32 + i]), x[b * 32 + i], sum);
        }
        acc = fma(sp[b], sum, acc);
    }
    out[row] = acc;
}

// Fused QKV / GateUp prefill: the tiled GEMM over the fused row range,
// routing each output column to its slice's buffer. Same tile geometry as
// `gemm_nt`; the slice test is uniform inside a 64-wide tile whenever the
// slice widths are multiples of 64, which holds for every model here.
kernel void gemm_qkv(device const float* a   [[buffer(0)]],
                     device const ushort* b  [[buffer(1)]],
                     device float* o0        [[buffer(2)]],
                     device float* o1        [[buffer(3)]],
                     device float* o2        [[buffer(4)]],
                     constant uint& m        [[buffer(5)]],
                     constant uint& k        [[buffer(6)]],
                     constant uint& n0       [[buffer(7)]],
                     constant uint& n1       [[buffer(8)]],
                     constant uint& n2       [[buffer(9)]],
                     uint2 tid [[thread_position_in_threadgroup]],
                     uint2 bid [[threadgroup_position_in_grid]]) {
    uint nrows = n0 + n1 + n2;
    uint row0 = bid.y * BM + tid.y * TM;
    uint col0 = bid.x * BN + tid.x * TN;

    threadgroup float ash[BM][BK];
    threadgroup float bsh[BN][BK];

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
            uint gc = min(bid.x * BN + r, nrows - 1);
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
                acc[r][c] = fma(av, bf16_to_f32(b[min(col0 + c, nrows - 1) * k + i]), acc[r][c]);
        }
    }

    for (uint c = 0; c < TN; ++c) {
        uint col = col0 + c;
        if (col >= nrows) break;
        device float* out;
        uint o;
        uint w;
        if (col < n0) { out = o0; o = col; w = n0; }
        else if (col < n0 + n1) { out = o1; o = col - n0; w = n1; }
        else { out = o2; o = col - n0 - n1; w = n2; }
        for (uint r = 0; r < TM; ++r) {
            if (row0 + r >= m) break;
            out[(row0 + r) * w + o] = acc[r][c];
        }
    }
}

// Fused Q8 QKV / GateUp prefill: tiled GEMM with per-(weight-row,
// block) scales. Each output column belongs to exactly one weight row, which
// selects both the output buffer and the scale row; the activation row never
// does, which is the bug a first draft of this kernel had.
kernel void gemm_q8_qkv(device const float* a   [[buffer(0)]],
                        device const char* b    [[buffer(1)]],
                        device const float* s   [[buffer(2)]],
                        device float* o0        [[buffer(3)]],
                        device float* o1        [[buffer(4)]],
                        device float* o2        [[buffer(5)]],
                        constant uint& m        [[buffer(6)]],
                        constant uint& k        [[buffer(7)]],
                        constant uint& nb       [[buffer(8)]],
                        constant uint& n0       [[buffer(9)]],
                        constant uint& n1       [[buffer(10)]],
                        constant uint& n2       [[buffer(11)]],
                        uint2 tid [[thread_position_in_threadgroup]],
                        uint2 bid [[threadgroup_position_in_grid]]) {
    uint nrows = n0 + n1 + n2;
    uint row0 = bid.y * BM + tid.y * TM;
    uint col0 = bid.x * BN + tid.x * TN;

    threadgroup float ash[BM][BK];
    threadgroup char bsh[BN][BK];

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
            uint gc = min(bid.x * BN + r, nrows - 1);
            ash[r][kk] = a[gr * k + t * BK + kk];
            bsh[r][kk] = b[gc * k + t * BK + kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bacc[TM][TN];
        for (uint r = 0; r < TM; ++r)
            for (uint c = 0; c < TN; ++c) bacc[r][c] = 0.0f;
        for (uint i = 0; i < BK; ++i) {
            float av[TM], bv[TN];
            for (uint r = 0; r < TM; ++r) av[r] = ash[tid.y * TM + r][i];
            for (uint c = 0; c < TN; ++c) bv[c] = float(bsh[tid.x * TN + c][i]);
            for (uint r = 0; r < TM; ++r)
                for (uint c = 0; c < TN; ++c)
                    bacc[r][c] = fma(av[r], bv[c], bacc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < TN; ++c) {
            uint col = col0 + c;
            if (col >= nrows) break;
            float sc = s[col * nb + t / 2];
            for (uint r = 0; r < TM; ++r) {
                if (row0 + r >= m) continue;
                acc[r][c] = fma(sc, bacc[r][c], acc[r][c]);
            }
        }
    }

    for (uint i = full * BK; i < k; ++i) {
        uint blk = i >> 5;
        for (uint r = 0; r < TM; ++r) {
            uint grow = min(row0 + r, m - 1);
            float av = a[grow * k + i];
            for (uint c = 0; c < TN; ++c) {
                uint gcol = min(col0 + c, nrows - 1);
                float sc = (row0 + r < m && col0 + c < nrows && blk < nb) ? s[gcol * nb + blk] : 0.0f;
                acc[r][c] = fma(sc * av, float(b[gcol * k + i]), acc[r][c]);
            }
        }
    }

    for (uint c = 0; c < TN; ++c) {
        uint col = col0 + c;
        if (col >= nrows) break;
        device float* out;
        uint o;
        uint w;
        if (col < n0) { out = o0; o = col; w = n0; }
        else if (col < n0 + n1) { out = o1; o = col - n0; w = n1; }
        else { out = o2; o = col - n0 - n1; w = n2; }
        for (uint r = 0; r < TM; ++r) {
            if (row0 + r >= m) break;
            out[(row0 + r) * w + o] = acc[r][c];
        }
    }
}
// out[i] = a[a_off + i] * b[b_off + i]. Offsets are element counts.// out[i] = a[a_off + i] * b[b_off + i]. Offsets are element counts.
kernel void mul_offsets(device float* out       [[buffer(0)]],
                        device const float* a    [[buffer(1)]],
                        device const float* b    [[buffer(2)]],
                        constant uint& out_off   [[buffer(3)]],
                        constant uint& a_off     [[buffer(4)]],
                        constant uint& b_off     [[buffer(5)]],
                        constant uint& n         [[buffer(6)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < n) out[out_off + i] = a[a_off + i] * b[b_off + i];
}

// Batched causal depthwise convolution over rows of `x`.
// grid = (dim, rows). Row r sees rows max(0, r-k+1)..=r of x.
kernel void conv_prefill(device float* out        [[buffer(0)]],
                         device const float* x    [[buffer(1)]],
                         device const float* kw   [[buffer(2)]],
                         constant uint& dim       [[buffer(3)]],
                         constant uint& rows      [[buffer(4)]],
                         constant uint& k         [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint r = gid.y;
    if (i >= dim || r >= rows) return;
    float acc = 0.0f;
    for (uint j = 0; j < k; ++j) {
        int h = int(r) - int(k) + 1 + int(j);
        if (h >= 0) acc = fma(kw[i * k + j], x[uint(h) * dim + i], acc);
    }
    out[r * dim + i] = acc;
}

// Single-row decode through the state ring: history is state (oldest first)
// followed by x, then state drops its oldest row and gains x.
kernel void conv_decode(device float* out       [[buffer(0)]],
                        device const float* x    [[buffer(1)]],
                        device const float* kw   [[buffer(2)]],
                        device float* state      [[buffer(3)]],
                        constant uint& dim       [[buffer(4)]],
                        constant uint& k         [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= dim) return;
    float acc = 0.0f;
    for (uint j = 0; j < k - 1; ++j) acc = fma(kw[i * k + j], state[j * dim + i], acc);
    acc = fma(kw[i * k + k - 1], x[i], acc);
    out[i] = acc;
    for (uint j = 0; j + 1 < k - 1; ++j) state[j * dim + i] = state[(j + 1) * dim + i];
    if (k > 1) state[(k - 2) * dim + i] = x[i];
}
"#;

#[derive(Clone, Copy, Debug)]
struct Layout {
    offset: usize,
    out_dim: usize,
    in_dim: usize,
    /// Byte offset into the packed int8 buffer, or `usize::MAX` when this
    /// projection is not quantized on the device.
    q_offset: usize,
    /// Element offset into the f32 scale buffer (one scale per 32-wide block
    /// per row); meaningless when `q_offset` is `usize::MAX`.
    s_offset: usize,
}

impl Layout {
    fn dense(offset: usize, out_dim: usize, in_dim: usize) -> Self {
        Layout {
            offset,
            out_dim,
            in_dim,
            q_offset: usize::MAX,
            s_offset: 0,
        }
    }

    fn quantized(q_offset: usize, s_offset: usize, out_dim: usize, in_dim: usize) -> Self {
        Layout {
            // The bf16 copy is not built for quantized projections; keep the
            // offset at zero rather than a dangling index into it.
            offset: 0,
            out_dim,
            in_dim,
            q_offset,
            s_offset,
        }
    }

    fn is_q8(&self) -> bool {
        self.q_offset != usize::MAX
    }
}

/// One fused QKV or GateUp projection: the row blocks each output owns.
///
/// A fused QKV covers `[q | k | v]` rows; a fused GateUp covers `[gate | up]`
/// with an empty third slice. The weights live in one concatenated buffer —
/// bf16 in [`Metal::weights`], or int8 plus f32 scales in
/// [`Metal::qweights`]/[`Metal::qscales`] — and the kernel routes each output
/// row to its slice. Keeping one kernel shape for both fusions halves the
/// shader count.
#[derive(Clone, Copy, Debug)]
struct FusedLayout {
    /// Element offset into the fused weight buffer (u16 for bf16, i8 for Q8).
    w_offset: usize,
    /// Element offset into the fused f32 scale buffer (Q8 only).
    s_offset: usize,
    /// Row counts of the three slices; the third is zero for GateUp.
    n0: usize,
    n1: usize,
    n2: usize,
    in_dim: usize,
    is_q8: bool,
}

impl FusedLayout {
    fn absent() -> Self {
        FusedLayout {
            w_offset: 0,
            s_offset: 0,
            n0: 0,
            n1: 0,
            n2: 0,
            in_dim: 0,
            is_q8: false,
        }
    }
}

pub struct Metal {
    device: Device,
    queue: CommandQueue,
    gemm_pipeline: ComputePipelineState,
    gemv_pipeline: ComputePipelineState,
    gemm_q8_pipeline: ComputePipelineState,
    gemv_q8_pipeline: ComputePipelineState,
    gemm_qkv_pipeline: ComputePipelineState,
    gemv_qkv_pipeline: ComputePipelineState,
    gemm_q8_qkv_pipeline: ComputePipelineState,
    gemv_q8_qkv_pipeline: ComputePipelineState,
    rms_pipeline: ComputePipelineState,
    qk_pipeline: ComputePipelineState,
    rope_half_pipeline: ComputePipelineState,
    rope_pair_pipeline: ComputePipelineState,
    attn_scores_pipeline: ComputePipelineState,
    attn_apply_pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
    silu_pipeline: ComputePipelineState,
    mul_pipeline: ComputePipelineState,
    conv_prefill_pipeline: ComputePipelineState,
    conv_decode_pipeline: ComputePipelineState,
    copy_pipeline: ComputePipelineState,
    kv_write_pipeline: ComputePipelineState,
    /// Every projection of every layer plus the classifier, concatenated.
    weights: Buffer,
    /// Packed int8 weights for quantized projections, row-major 32-wide
    /// blocks, concatenated in the same order.
    qweights: Buffer,
    /// One f32 scale per block, concatenated in the same order.
    qscales: Buffer,
    /// Fused Q and GateUp weights: `[Q|K|V]` and `[gate|up]` rows concatenated
    /// per layer, in `weights` (bf16) or `qweights`/`qscales` (Q8), with one
    /// layout each. Attention layers carry both; convolution layers carry only
    /// the GateUp half.
    qkv: Vec<FusedLayout>,
    gate_up: Vec<FusedLayout>,
    layers: Vec<[Layout; 9]>,
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
            gemm_q8_pipeline: pipeline(&device, &library, "gemm_q8")?,
            gemv_q8_pipeline: pipeline(&device, &library, "gemv_q8")?,
            gemm_qkv_pipeline: pipeline(&device, &library, "gemm_qkv")?,
            gemv_qkv_pipeline: pipeline(&device, &library, "gemv_qkv")?,
            gemm_q8_qkv_pipeline: pipeline(&device, &library, "gemm_q8_qkv")?,
            gemv_q8_qkv_pipeline: pipeline(&device, &library, "gemv_q8_qkv")?,
            rms_pipeline: pipeline(&device, &library, "rms_norm")?,
            qk_pipeline: pipeline(&device, &library, "qk_norm")?,
            rope_half_pipeline: pipeline(&device, &library, "rope_half")?,
            rope_pair_pipeline: pipeline(&device, &library, "rope_pair")?,
            attn_scores_pipeline: pipeline(&device, &library, "attn_scores")?,
            attn_apply_pipeline: pipeline(&device, &library, "attn_apply")?,
            add_pipeline: pipeline(&device, &library, "add_in_place")?,
            silu_pipeline: pipeline(&device, &library, "silu_mul")?,
            mul_pipeline: pipeline(&device, &library, "mul_offsets")?,
            conv_prefill_pipeline: pipeline(&device, &library, "conv_prefill")?,
            conv_decode_pipeline: pipeline(&device, &library, "conv_decode")?,
            copy_pipeline: pipeline(&device, &library, "copy_f32")?,
            kv_write_pipeline: pipeline(&device, &library, "kv_write")?,
            queue: device.new_command_queue(),
            weights: placeholder.clone(),
            qweights: placeholder.clone(),
            qscales: placeholder.clone(),
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
            classifier: Layout::dense(0, 0, 0),
            qkv: Vec::new(),
            gate_up: Vec::new(),
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

    /// Concatenate several projections' rows into the fused stores, in order.
    ///
    /// Returns the layout the fused kernels bind: bf16 rows appended to
    /// `flat`, or int8 rows plus f32 scales appended to `qflat`/`sflat` when
    /// the model is quantized. `FusedLayout::absent` when any piece is
    /// missing (a convolution layer has no QKV triple); the driver never
    /// dispatches those.
    fn fuse_projs(
        layer: &LayerWeights,
        projs: &[Proj],
        flat: &mut Vec<u16>,
        qflat: &mut Vec<i8>,
        sflat: &mut Vec<f32>,
    ) -> FusedLayout {
        let mut mats = Vec::with_capacity(projs.len());
        for &proj in projs {
            match proj.matrix_opt(layer) {
                Some(m) => mats.push(m),
                None => return FusedLayout::absent(),
            }
        }
        let in_dim = mats[0].in_dim();
        debug_assert!(mats.iter().all(|m| m.in_dim() == in_dim));
        let mut rows = [0usize; 3];
        for (i, m) in mats.iter().enumerate() {
            rows[i] = m.out_dim();
        }
        if mats[0].to_q8_row_major().is_some() {
            let w_offset = qflat.len();
            let s_offset = sflat.len();
            for m in mats {
                let (quants, scales) = m.to_q8_row_major().expect("storage is uniform");
                qflat.extend_from_slice(&quants);
                sflat.extend_from_slice(&scales);
            }
            FusedLayout {
                w_offset,
                s_offset,
                n0: rows[0],
                n1: rows[1],
                n2: rows[2],
                in_dim,
                is_q8: true,
            }
        } else {
            let w_offset = flat.len();
            for m in mats {
                flat.extend_from_slice(&m.to_bf16_bits());
            }
            FusedLayout {
                w_offset,
                s_offset: 0,
                n0: rows[0],
                n1: rows[1],
                n2: rows[2],
                in_dim,
                is_q8: false,
            }
        }
    }

    fn upload_weights(&mut self, weights: &ModelWeights) {
        let mut flat: Vec<u16> = Vec::new();
        let mut qflat: Vec<i8> = Vec::new();
        let mut sflat: Vec<f32> = Vec::new();
        let mut layers = Vec::with_capacity(weights.layers.len());
        let mut qkv = Vec::with_capacity(weights.layers.len());
        let mut gate_up = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let mut layouts = [Layout::dense(0, 0, 0); 9];
            // Q/K/V and Gate/Up fuse into single dispatches; their rows live
            // in the fused stores below, not in per-projection slots.
            const FUSED: [Proj; 5] = [Proj::Q, Proj::K, Proj::V, Proj::Gate, Proj::Up];
            for proj in Proj::ALL {
                if FUSED.contains(&proj) {
                    continue;
                }
                // Layers only widen what they carry: attention layers skip the
                // conv pair, conv layers skip the attention set.
                let Some(matrix) = proj.matrix_opt(layer) else {
                    continue;
                };
                // Quantized projections keep no bf16 copy: the device reads
                // the canonical row-major int8 blocks instead.
                if let Some((quants, scales)) = matrix.to_q8_row_major() {
                    let q_offset = qflat.len();
                    let s_offset = sflat.len();
                    qflat.extend_from_slice(&quants);
                    sflat.extend_from_slice(&scales);
                    layouts[proj.index()] =
                        Layout::quantized(q_offset, s_offset, matrix.out_dim(), matrix.in_dim());
                    continue;
                }
                let offset = flat.len();
                flat.extend_from_slice(&matrix.to_bf16_bits());
                layouts[proj.index()] = Layout::dense(offset, matrix.out_dim(), matrix.in_dim());
            }
            layers.push(layouts);

            // Fused QKV: [Q rows | K rows | V rows], attention layers only.
            qkv.push(Self::fuse_projs(
                layer,
                &[Proj::Q, Proj::K, Proj::V],
                &mut flat,
                &mut qflat,
                &mut sflat,
            ));
            // Fused GateUp: [gate rows | up rows]; every layer carries an MLP.
            gate_up.push(Self::fuse_projs(
                layer,
                &[Proj::Gate, Proj::Up],
                &mut flat,
                &mut qflat,
                &mut sflat,
            ));
        }
        let classifier = weights.classifier();
        let classifier_layout = if let Some((quants, scales)) = classifier.to_q8_row_major() {
            let layout = Layout::quantized(
                qflat.len(),
                sflat.len(),
                classifier.out_dim(),
                classifier.in_dim(),
            );
            qflat.extend_from_slice(&quants);
            sflat.extend_from_slice(&scales);
            layout
        } else {
            let offset = flat.len();
            flat.extend_from_slice(&classifier.to_bf16_bits());
            Layout::dense(offset, classifier.out_dim(), classifier.in_dim())
        };

        // A zero-length `Vec::as_ptr` is dangling, and Metal hands back nil
        // for a zero-length buffer — which panics when dropped. Pad every
        // store so the pointer is always backed, even when a storage has no
        // projections (a fully quantized model uploads no bf16 at all).
        if flat.is_empty() {
            flat.push(0);
        }
        if qflat.is_empty() {
            qflat.push(0);
        }
        if sflat.is_empty() {
            sflat.push(0.0);
        }
        self.weights = self.device.new_buffer_with_data(
            flat.as_ptr() as *const _,
            (flat.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.qweights = self.device.new_buffer_with_data(
            qflat.as_ptr() as *const _,
            qflat.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.qscales = self.device.new_buffer_with_data(
            sflat.as_ptr() as *const _,
            (sflat.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.layers = layers;
        self.classifier = classifier_layout;
        self.qkv = qkv;
        self.gate_up = gate_up;
    }

    /// Encode `y[rows, out] = x[rows, in] * w[out, in]^T` over row-major
    /// Q8_0 blocks: int8 weights plus one f32 scale per 32-wide block.
    fn dispatch_q8(&mut self, layout: Layout, x: Buf, y: Buf, rows: usize) {
        let a = self.buf(x);
        let c = self.buf(y);
        let qw = self.qweights.clone();
        let qs = self.qscales.clone();
        let qo = layout.q_offset as u64;
        let so = (layout.s_offset * 4) as u64;
        let n = layout.out_dim as u32;
        let k = layout.in_dim as u32;
        let m = rows as u32;
        let nb = (layout.in_dim / crate::ops::Q8_0_BLOCK) as u32;
        let vector = rows == 1;
        let gemm = self.gemm_q8_pipeline.clone();
        let gemv = self.gemv_q8_pipeline.clone();

        self.encode(move |enc| {
            if vector {
                enc.set_compute_pipeline_state(&gemv);
                enc.set_buffer(0, Some(&qw), qo);
                enc.set_buffer(1, Some(&qs), so);
                enc.set_buffer(2, Some(&a), 0);
                enc.set_buffer(3, Some(&c), 0);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                enc.set_bytes(6, 4, &nb as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(layout.out_dim as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&gemm);
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&qw), qo);
                enc.set_buffer(2, Some(&qs), so);
                enc.set_buffer(3, Some(&c), 0);
                enc.set_bytes(4, 4, &m as *const u32 as *const _);
                enc.set_bytes(5, 4, &n as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &nb as *const u32 as *const _);
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

    /// Encode `y[rows, out] = x[rows, in] * w[out, in]^T`.
    fn dispatch(&mut self, layout: Layout, x: Buf, y: Buf, rows: usize) {
        if layout.is_q8() {
            self.dispatch_q8(layout, x, y, rows);
            return;
        }
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

    /// Test-only Q8 dispatch over explicit weight buffers, without a
    /// prepared model. The integration tests cover dispatch through `project`;
    /// these cover the kernels against a host oracle on synthetic shapes.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn gemv_q8_dispatch(
        &mut self,
        qw: &Buffer,
        qs: &Buffer,
        x: Buf,
        y: Buf,
        out_dim: usize,
        in_dim: usize,
    ) {
        let a = self.buf(x);
        let c = self.buf(y);
        let (qw, qs) = (qw.clone(), qs.clone());
        let n = out_dim as u32;
        let k = in_dim as u32;
        let nb = (in_dim / crate::ops::Q8_0_BLOCK) as u32;
        let pipeline = self.gemv_q8_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&qw), 0);
            enc.set_buffer(1, Some(&qs), 0);
            enc.set_buffer(2, Some(&a), 0);
            enc.set_buffer(3, Some(&c), 0);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            enc.set_bytes(6, 4, &nb as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    /// Test-only batched Q8 dispatch; see [`Metal::gemv_q8_dispatch`].
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn gemm_q8_dispatch(
        &mut self,
        qw: &Buffer,
        qs: &Buffer,
        x: Buf,
        y: Buf,
        out_dim: usize,
        in_dim: usize,
        rows: usize,
    ) {
        let a = self.buf(x);
        let c = self.buf(y);
        let (qw, qs) = (qw.clone(), qs.clone());
        let m = rows as u32;
        let n = out_dim as u32;
        let k = in_dim as u32;
        let nb = (in_dim / crate::ops::Q8_0_BLOCK) as u32;
        let pipeline = self.gemm_q8_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&qw), 0);
            enc.set_buffer(2, Some(&qs), 0);
            enc.set_buffer(3, Some(&c), 0);
            enc.set_bytes(4, 4, &m as *const u32 as *const _);
            enc.set_bytes(5, 4, &n as *const u32 as *const _);
            enc.set_bytes(6, 4, &k as *const u32 as *const _);
            enc.set_bytes(7, 4, &nb as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (out_dim as u64).div_ceil(BLOCK_N),
                    (rows as u64).div_ceil(BLOCK_M),
                    1,
                ),
                MTLSize::new(16, 16, 1),
            );
        });
    }
    /// One dispatch for a fused [n0 | n1 | n2] projection triple.
    ///
    /// `o2` is written only when `n2 > 0`; the GateUp path passes its own
    /// `gate` buffer twice so the extra binding is harmless but never stored.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_fused(
        &mut self,
        fused: FusedLayout,
        x: Buf,
        o0: Buf,
        o1: Buf,
        o2: Buf,
        rows: usize,
    ) {
        let a = self.buf(x);
        let (b0, b1, b2) = (self.buf(o0), self.buf(o1), self.buf(o2));
        let vector = rows == 1;
        let (n0, n1, n2, k) = (
            fused.n0 as u32,
            fused.n1 as u32,
            fused.n2 as u32,
            fused.in_dim as u32,
        );
        let m = rows as u32;
        if fused.is_q8 {
            let qw = self.qweights.clone();
            let qs = self.qscales.clone();
            let qo = fused.w_offset as u64;
            let so = (fused.s_offset * 4) as u64;
            let nb = (fused.in_dim / crate::ops::Q8_0_BLOCK) as u32;
            let (gemm, gemv) = (
                self.gemm_q8_qkv_pipeline.clone(),
                self.gemv_q8_qkv_pipeline.clone(),
            );
            self.encode(move |enc| {
                if vector {
                    enc.set_compute_pipeline_state(&gemv);
                    enc.set_buffer(0, Some(&qw), qo);
                    enc.set_buffer(1, Some(&qs), so);
                    enc.set_buffer(2, Some(&b0), 0);
                    enc.set_buffer(3, Some(&b1), 0);
                    enc.set_buffer(4, Some(&b2), 0);
                    enc.set_buffer(5, Some(&a), 0);
                    enc.set_bytes(6, 4, &n0 as *const u32 as *const _);
                    enc.set_bytes(7, 4, &n1 as *const u32 as *const _);
                    enc.set_bytes(8, 4, &n2 as *const u32 as *const _);
                    enc.set_bytes(9, 4, &k as *const u32 as *const _);
                    enc.set_bytes(10, 4, &nb as *const u32 as *const _);
                    enc.dispatch_threads(
                        MTLSize::new((n0 + n1 + n2) as u64, 1, 1),
                        MTLSize::new(64, 1, 1),
                    );
                } else {
                    enc.set_compute_pipeline_state(&gemm);
                    enc.set_buffer(0, Some(&a), 0);
                    enc.set_buffer(1, Some(&qw), qo);
                    enc.set_buffer(2, Some(&qs), so);
                    enc.set_buffer(3, Some(&b0), 0);
                    enc.set_buffer(4, Some(&b1), 0);
                    enc.set_buffer(5, Some(&b2), 0);
                    enc.set_bytes(6, 4, &m as *const u32 as *const _);
                    enc.set_bytes(7, 4, &k as *const u32 as *const _);
                    enc.set_bytes(8, 4, &nb as *const u32 as *const _);
                    enc.set_bytes(9, 4, &n0 as *const u32 as *const _);
                    enc.set_bytes(10, 4, &n1 as *const u32 as *const _);
                    enc.set_bytes(11, 4, &n2 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(
                            ((n0 + n1 + n2) as u64).div_ceil(BLOCK_N),
                            (rows as u64).div_ceil(BLOCK_M),
                            1,
                        ),
                        MTLSize::new(16, 16, 1),
                    );
                }
            });
            return;
        }
        let weights = self.weights.clone();
        let wo = fused.w_offset as u64 * 2;
        let (gemm, gemv) = (
            self.gemm_qkv_pipeline.clone(),
            self.gemv_qkv_pipeline.clone(),
        );
        self.encode(move |enc| {
            if vector {
                enc.set_compute_pipeline_state(&gemv);
                enc.set_buffer(0, Some(&weights), wo);
                enc.set_buffer(1, Some(&b0), 0);
                enc.set_buffer(2, Some(&b1), 0);
                enc.set_buffer(3, Some(&b2), 0);
                enc.set_buffer(4, Some(&a), 0);
                enc.set_bytes(5, 4, &n0 as *const u32 as *const _);
                enc.set_bytes(6, 4, &n1 as *const u32 as *const _);
                enc.set_bytes(7, 4, &n2 as *const u32 as *const _);
                enc.set_bytes(8, 4, &k as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new((n0 + n1 + n2) as u64, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&gemm);
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&weights), wo);
                enc.set_buffer(2, Some(&b0), 0);
                enc.set_buffer(3, Some(&b1), 0);
                enc.set_buffer(4, Some(&b2), 0);
                enc.set_bytes(5, 4, &m as *const u32 as *const _);
                enc.set_bytes(6, 4, &k as *const u32 as *const _);
                enc.set_bytes(7, 4, &n0 as *const u32 as *const _);
                enc.set_bytes(8, 4, &n1 as *const u32 as *const _);
                enc.set_bytes(9, 4, &n2 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        ((n0 + n1 + n2) as u64).div_ceil(BLOCK_N),
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
        self.weights().token_embedding.kind()
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
        let o = self.buf(out);
        let ab = self.buf(a);
        let bb = self.buf(b);
        let (oo, ao, bo, n) = (out_off as u32, a_off as u32, b_off as u32, len as u32);
        let pipeline = self.mul_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&o), 0);
            enc.set_buffer(1, Some(&ab), 0);
            enc.set_buffer(2, Some(&bb), 0);
            enc.set_bytes(3, 4, &oo as *const u32 as *const _);
            enc.set_bytes(4, 4, &ao as *const u32 as *const _);
            enc.set_bytes(5, 4, &bo as *const u32 as *const _);
            enc.set_bytes(6, 4, &n as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    fn conv_prefill(&mut self, out: Buf, x: Buf, kernel: Buf, rows: usize, dim: usize, k: usize) {
        let o = self.buf(out);
        let xb = self.buf(x);
        let kb = self.buf(kernel);
        let (d, r, kk) = (dim as u32, rows as u32, k as u32);
        let pipeline = self.conv_prefill_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&o), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&kb), 0);
            enc.set_bytes(3, 4, &d as *const u32 as *const _);
            enc.set_bytes(4, 4, &r as *const u32 as *const _);
            enc.set_bytes(5, 4, &kk as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(dim as u64, rows as u64, 1),
                MTLSize::new(32, 8, 1),
            );
        });
    }

    fn conv_decode(&mut self, out: Buf, x: Buf, kernel: Buf, state: Buf, dim: usize, k: usize) {
        let o = self.buf(out);
        let xb = self.buf(x);
        let kb = self.buf(kernel);
        let st = self.buf(state);
        let (d, kk) = (dim as u32, k as u32);
        let pipeline = self.conv_decode_pipeline.clone();
        self.encode(move |enc| {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&o), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&kb), 0);
            enc.set_buffer(3, Some(&st), 0);
            enc.set_bytes(4, 4, &d as *const u32 as *const _);
            enc.set_bytes(5, 4, &kk as *const u32 as *const _);
            enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    fn logits(&mut self, x: Buf, y: Buf) {
        let layout = self.classifier;
        self.dispatch(layout, x, y, 1);
    }

    fn project_qkv(&mut self, layer: usize, x: Buf, q: Buf, k: Buf, v: Buf, rows: usize) {
        let fused = self.qkv[layer];
        assert!(
            fused.n0 > 0,
            "project_qkv on a layer without attention weights"
        );
        self.dispatch_fused(fused, x, q, k, v, rows);
    }

    fn project_gate_up(&mut self, layer: usize, x: Buf, gate: Buf, up: Buf, rows: usize) {
        let fused = self.gate_up[layer];
        assert!(fused.n0 > 0, "project_gate_up on a layer without an MLP");
        // A GateUp fusion is a QKV fusion whose third slice is empty.
        self.dispatch_fused(fused, x, gate, up, gate, rows);
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

        // Q8 GEMV and GEMM against a host dot product over the dequantized
        // rows: the kernels must read the same logical weights the loader
        // wrote, whatever the host layout was.
        for &(out_dim, in_dim) in &[(64usize, 64usize), (8, 64), (16, 128)] {
            let data: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| ((i * 17 % 251) as f32 - 125.0) / 137.0)
                .collect();
            let matrix = crate::ops::Matrix::from_f32_q8_0(
                crate::tensor::Tensor::from_vec(data, &[out_dim, in_dim]).unwrap(),
            )
            .unwrap();
            let (quants, scales) = matrix.to_q8_row_major().unwrap();
            let qw = m.device.new_buffer_with_data(
                quants.as_ptr() as *const _,
                quants.len().max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let sw = m.device.new_buffer_with_data(
                scales.as_ptr() as *const _,
                (scales.len().max(1) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.37).sin()).collect();
            let xb = m.upload(&x);
            // one row: GEMV
            {
                let yb = m.alloc(out_dim);
                m.gemv_q8_dispatch(&qw, &sw, xb, yb, out_dim, in_dim);
                let mut got = vec![0.0; out_dim];
                m.read(yb, &mut got);
                let mut row = vec![0.0f32; in_dim];
                for (r, g) in got.iter().enumerate() {
                    matrix.decode_row(r, &mut row);
                    let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
                    assert!(
                        (g - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "q8 gemv row {r}: {g} vs {want}"
                    );
                }
            }
            // ones scales: with every scale at 1.0, any scale-indexing
            // bug disappears and only the int8 load/dot path is exercised.
            {
                let (hq, _) = matrix.to_q8_row_major().unwrap();
                let sw1 = vec![1.0f32; scales.len()];
                let qwb = m.device.new_buffer_with_data(
                    hq.as_ptr() as *const _,
                    hq.len().max(1) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let swb = m.device.new_buffer_with_data(
                    sw1.as_ptr() as *const _,
                    (sw1.len().max(1) * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let xb1 = m.upload(&x);
                let yb1 = m.alloc(out_dim);
                m.gemm_q8_dispatch(&qwb, &swb, xb1, yb1, out_dim, in_dim, 1);
                let mut got1 = vec![0.0; out_dim];
                m.read(yb1, &mut got1);
                // reference without scales: pure int8 dot
                for (o, g) in got1.iter().enumerate() {
                    let want: f32 = hq[o * in_dim..(o + 1) * in_dim]
                        .iter()
                        .zip(&x)
                        .map(|(&q, &xv)| q as f32 * xv)
                        .sum();
                    assert!(
                        (g - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "ones-scale out {o}: {g} vs {want}"
                    );
                }
            }
            // one row through the tiled GEMM path (vs GEMV above)
            {
                let xb1 = m.upload(&x);
                let yb1 = m.alloc(out_dim);
                m.gemm_q8_dispatch(&qw, &sw, xb1, yb1, out_dim, in_dim, 1);
                let mut got1 = vec![0.0; out_dim];
                m.read(yb1, &mut got1);
                let mut row = vec![0.0f32; in_dim];
                for (o, g) in got1.iter().enumerate() {
                    matrix.decode_row(o, &mut row);
                    let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
                    let tol = 1e-4 * (1.0 + want.abs());
                    assert!(
                        (g - want).abs() <= tol,
                        "q8 gemm-1row out {o}: {g} vs {want}"
                    );
                }
            }
            // full-width rows: every output row of a multi-row batch.
            {
                let rows = 8usize;
                let mut xx = Vec::with_capacity(rows * in_dim);
                for r in 0..rows {
                    xx.extend(x.iter().map(|v| v * (r as f32 + 1.0)));
                }
                let xb = m.upload(&xx);
                let yb = m.alloc(rows * out_dim);
                m.gemm_q8_dispatch(&qw, &sw, xb, yb, out_dim, in_dim, rows);
                let mut got = vec![0.0; rows * out_dim];
                m.read(yb, &mut got);
                let mut row = vec![0.0f32; in_dim];
                for r in 0..rows {
                    for (o, g) in got[r * out_dim..(r + 1) * out_dim].iter().enumerate() {
                        matrix.decode_row(o, &mut row);
                        let want: f32 = row
                            .iter()
                            .zip(&xx[r * in_dim..(r + 1) * in_dim])
                            .map(|(a, b)| a * b)
                            .sum();
                        // Tiled accumulation sums in a different order than the
                        // reference loop; tolerance is relative to the scale.
                        let tol = 2e-5 * (1.0 + want.abs());
                        assert!(
                            (g - want).abs() <= tol,
                            "q8 gemm row {r} out {o}: {g} vs {want}"
                        );
                    }
                }
            }
            // row-uniform scales: every block in a row shares one scale,
            // so only the row index matters, not the tile index.
            {
                let rows = 4usize;
                let mut xx = Vec::with_capacity(rows * in_dim);
                for r in 0..rows {
                    xx.extend(x.iter().map(|v| v * (r as f32 + 1.0)));
                }
                // build scales[row] = 1.0 + 0.5 * row
                let (hq, _) = matrix.to_q8_row_major().unwrap();
                let nbb = in_dim / crate::ops::Q8_0_BLOCK;
                let mut hs = vec![0.0f32; out_dim * nbb];
                for r in 0..out_dim {
                    for b in 0..nbb {
                        hs[r * nbb + b] = 1.0 + 0.5 * (r as f32);
                    }
                }
                let qwb = m.device.new_buffer_with_data(
                    hq.as_ptr() as *const _,
                    hq.len().max(1) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let swb = m.device.new_buffer_with_data(
                    hs.as_ptr() as *const _,
                    (hs.len().max(1) * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let xb = m.upload(&xx);
                let yb = m.alloc(rows * out_dim);
                m.gemm_q8_dispatch(&qwb, &swb, xb, yb, out_dim, in_dim, rows);
                let mut got = vec![0.0; rows * out_dim];
                m.read(yb, &mut got);
                for r in 0..rows {
                    for (o, g) in got[r * out_dim..(r + 1) * out_dim].iter().enumerate() {
                        // reference with row-uniform scales
                        let want: f32 = matrix.to_q8_row_major().unwrap().0
                            [o * in_dim..(o + 1) * in_dim]
                            .iter()
                            .zip(&xx[r * in_dim..(r + 1) * in_dim])
                            .map(|(&q, &xv)| q as f32 * xv)
                            .sum::<f32>()
                            * (1.0 + 0.5 * (o as f32));
                        let tol = 1e-3 * (1.0 + want.abs());
                        assert!(
                            (g - want).abs() <= tol,
                            "q8 gemm rowuni r={r} o={o}: {g} vs {want}"
                        );
                    }
                }
            }
            // three rows: tiled GEMM
            {
                let rows = 3usize;
                let mut xx = Vec::with_capacity(rows * in_dim);
                for r in 0..rows {
                    xx.extend(x.iter().map(|v| v * (r as f32 + 1.0)));
                }
                let xb = m.upload(&xx);
                let yb = m.alloc(rows * out_dim);
                m.gemm_q8_dispatch(&qw, &sw, xb, yb, out_dim, in_dim, rows);
                let mut got = vec![0.0; rows * out_dim];
                m.read(yb, &mut got);
                let mut row = vec![0.0f32; in_dim];
                for r in 0..rows {
                    for (o, g) in got[r * out_dim..(r + 1) * out_dim].iter().enumerate() {
                        matrix.decode_row(o, &mut row);
                        let want: f32 = row
                            .iter()
                            .zip(&xx[r * in_dim..(r + 1) * in_dim])
                            .map(|(a, b)| a * b)
                            .sum();
                        assert!(
                            (g - want).abs() <= 1e-3 * (1.0 + want.abs()),
                            "q8 gemm row {r} out {o}: {g} vs {want}"
                        );
                    }
                }
            }
        }

        // mul with offsets, including same-buffer operands
        let ab = m.upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let o = m.alloc(2);
        m.mul(o, 0, ab, 1, ab, 3, 2);
        let mut got = vec![0.0; 2];
        m.read(o, &mut got);
        assert_eq!(got, vec![2.0 * 4.0, 3.0 * 5.0], "mul offsets");

        // conv_prefill against a hand computation, dim 4, k 3, rows 4.
        // Row r sees max(0, r-2)..=r, zero-padded below row 0.
        let x: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let kw: Vec<f32> = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, //
        ];
        let xb = m.upload(&x);
        let kb = m.upload(&kw);
        let co = m.alloc(16);
        m.conv_prefill(co, xb, kb, 4, 4, 3);
        let mut got = vec![0.0; 16];
        m.read(co, &mut got);
        // Lane 0 keeps tap 0: rows see x[r-2], x[r-1]... vs zero pad.
        assert_eq!(got[0], 0.0, "conv pad row 0");
        assert_eq!(got[4], 0.0, "conv pad row 1 lane 0");
        assert_eq!(got[8], x[0], "conv row 2 lane 0");
        assert_eq!(got[12], x[4], "conv row 3 lane 0");
        // Lane 3 sums all three taps.
        assert_eq!(got[3], x[3] + 0.0 + 0.0, "conv row 0 lane 3");
        assert_eq!(got[15], x[7] + x[11] + x[15], "conv row 3 lane 3");

        // conv_decode: history [r0, r1] plus x, then the ring advances.
        let st = m.upload(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        let xr = m.upload(&[1.0f32, 1.0, 1.0, 1.0]);
        let orow = m.alloc(4);
        m.conv_decode(orow, xr, kb, st, 4, 3);
        let mut got = vec![0.0; 4];
        m.read(orow, &mut got);
        // Lane 0 (tap 0): state row 0 lane 0 (10). Lane 3 (all taps):
        // state row 0 lane 3 (40) + state row 1 lane 3 (80) + x lane 3 (1).
        assert_eq!(got[0], 10.0, "decode lane 0");
        assert_eq!(got[3], 40.0 + 80.0 + 1.0, "decode lane 3");
        let mut ring = vec![0.0; 8];
        m.read(st, &mut ring);
        assert_eq!(
            ring,
            vec![50.0, 60.0, 70.0, 80.0, 1.0, 1.0, 1.0, 1.0],
            "ring advances"
        );
    }
}
