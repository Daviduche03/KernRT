//! LFM2 hybrid decoder (LFM2-350M, LFM2-700M; 1.2B/2.6B share the code).
//!
//! LFM2 alternates two layer kinds under one residual stream. Attention layers
//! are Llama-shaped (GQA with per-head q/k norms, half-split RoPE, SwiGLU
//! MLP); convolution layers replace the attention block with a short causal
//! depthwise conv over a gated projection:
//!
//! ```text
//! h = B * x,  conv_out = causal_dw_conv(h),  y = C * conv_out,  out = W_o y
//! ```
//!
//! where `[B, C, x] = in_proj(xb)` is one `[3*dim, dim]` projection split
//! three ways. The decode path keeps the previous `k-1` rows of `h` in a ring
//! (`k = conv_L_cache = 3`), so a step costs one depthwise pass over `dim`
//! taps rather than a re-scan of the context.
//!
//! The heavy projections go through [`Backend`](crate::backends::Backend)
//! exactly like [`Llama`](super::llama::Llama), including the batched GEMM
//! prefill path; the conv arithmetic is small enough to stay scalar on CPU
//! and one thread per element on Metal.

use super::LanguageModel;
use super::hf::{
    LoadError, Storage, matrix_from_raw, parse_lfm2_config, take_f32, take_matrix_raw,
    take_optional_f32, take_optional_raw,
};
use super::llama::Config;
use crate::backends::{AttnSpec, Backend, Buf, KvSpec, ModelWeights, Proj, RopeSpec, Window};
use crate::formats::llama2c::LayerWeights;
use crate::formats::safetensors::{self, RawTensor};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// One decoded token's worth of scratch, plus the conv ring.
struct State {
    x: Buf,
    xb: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    o: Buf,
    /// `[3*dim]` input-projection output, split into B/C/x thirds.
    bcx: Buf,
    /// `B * x`, the conv input; also the ring's newest row on decode.
    h: Buf,
    /// Causal depthwise output.
    co: Buf,
    /// `C * conv_out`, feeding `conv_out_proj`.
    cy: Buf,
    gate: Buf,
    up: Buf,
    mlp: Buf,
    hidden: Buf,
    att: Buf,
    logits: Buf,
    logits_host: Vec<f32>,
    prefill: Option<PrefillBuffers>,
}

/// Row-major `[n_tok, width]` scratch for a batched forward pass.
#[derive(Clone, Copy)]
struct PrefillBuffers {
    n_tok: usize,
    x: Buf,
    xb: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    o: Buf,
    bcx: Buf,
    h: Buf,
    co: Buf,
    cy: Buf,
    gate: Buf,
    up: Buf,
    att: Buf,
}

impl PrefillBuffers {
    fn new(backend: &mut dyn Backend, n_tok: usize, config: &Config) -> Self {
        let dim = config.dim;
        let kv_dim = config.kv_dim();
        let q_dim = config.q_dim();
        let hidden = config.hidden_dim;
        Self {
            n_tok,
            x: backend.alloc(n_tok * dim),
            xb: backend.alloc(n_tok * dim),
            q: backend.alloc(n_tok * q_dim),
            k: backend.alloc(n_tok * kv_dim),
            v: backend.alloc(n_tok * kv_dim),
            attn: backend.alloc(n_tok * q_dim),
            o: backend.alloc(n_tok * dim),
            bcx: backend.alloc(n_tok * 3 * dim),
            h: backend.alloc(n_tok * dim),
            co: backend.alloc(n_tok * dim),
            cy: backend.alloc(n_tok * dim),
            gate: backend.alloc(n_tok * hidden),
            up: backend.alloc(n_tok * hidden),
            att: backend.alloc(n_tok * config.n_heads * config.seq_len),
        }
    }
}

/// The norm weights of one layer, resident in backend memory.
struct Lfm2Norms {
    op: Buf,
    ffn: Buf,
    q: Option<Buf>,
    k: Option<Buf>,
}

/// A loaded LFM2 transformer.
pub struct Lfm2 {
    config: Config,
    /// Per layer: attention or short conv.
    is_attention: Vec<bool>,
    /// Depthwise kernel width per layer (`conv_L_cache`, 3 on every LFM2).
    conv_k: Vec<usize>,
    backend: Box<dyn Backend>,
    norms: Vec<Lfm2Norms>,
    /// Per-layer depthwise kernels, `[dim, k]` f32.
    conv_kernels: Vec<Option<Buf>>,
    /// Per-layer conv rings, `(k-1)*dim` f32, oldest first.
    conv_state: Vec<Option<Buf>>,
    rms_final: Buf,
    state: State,
}

impl Lfm2 {
    /// Load a HuggingFace-format LFM2 directory: `config.json` plus
    /// `model.safetensors`.
    pub fn from_hf(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        Self::from_hf_with(dir, crate::backends::from_env())
    }

    /// [`from_hf`](Self::from_hf) over an explicit backend.
    pub fn from_hf_with(
        dir: impl AsRef<Path>,
        backend: Box<dyn Backend>,
    ) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let (config, bf16, is_attention, _conv_k) = parse_lfm2_config(&dir.join("config.json"))?;
        let storage = if std::env::var_os("KERNRT_Q8_0X4").is_some() {
            Storage::Q8_0x4
        } else if std::env::var_os("KERNRT_Q8_0").is_some() {
            Storage::Q8_0
        } else if bf16 && std::env::var_os("KERNRT_ALL_F32").is_none() {
            Storage::Bf16
        } else {
            Storage::F32
        };
        let tensors = safetensors::load_raw(dir.join("model.safetensors"))?;
        Self::from_raw_tensors_with(config, is_attention, storage, tensors, backend)
    }

    /// Build a model from an already-parsed config and tensor map.
    pub(crate) fn from_raw_tensors_with(
        config: Config,
        is_attention: Vec<bool>,
        storage: Storage,
        mut tensors: HashMap<String, RawTensor>,
        backend: Box<dyn Backend>,
    ) -> Result<Self, LoadError> {
        assert_eq!(
            is_attention.len(),
            config.n_layers,
            "layer mask must cover every layer"
        );
        let token_embedding = take_matrix_raw(&mut tensors, "model.embed_tokens.weight", storage)?;
        let rms_final = take_f32(&mut tensors, "model.embedding_norm.weight")?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for (l, &attn) in is_attention.iter().enumerate() {
            let p = format!("model.layers.{l}");
            let (wq, wk, wv, wo, rms_q, rms_k) = if attn {
                (
                    Some(take_matrix_raw(
                        &mut tensors,
                        &format!("{p}.self_attn.q_proj.weight"),
                        storage,
                    )?),
                    Some(take_matrix_raw(
                        &mut tensors,
                        &format!("{p}.self_attn.k_proj.weight"),
                        storage,
                    )?),
                    Some(take_matrix_raw(
                        &mut tensors,
                        &format!("{p}.self_attn.v_proj.weight"),
                        storage,
                    )?),
                    // LFM2 names it `out_proj` where Llama says `o_proj`.
                    Some(take_matrix_raw(
                        &mut tensors,
                        &format!("{p}.self_attn.out_proj.weight"),
                        storage,
                    )?),
                    take_optional_f32(&mut tensors, &format!("{p}.self_attn.q_layernorm.weight"))?,
                    take_optional_f32(&mut tensors, &format!("{p}.self_attn.k_layernorm.weight"))?,
                )
            } else {
                (None, None, None, None, None, None)
            };
            let (conv_in, conv_out, conv_kernel) = if attn {
                (None, None, None)
            } else {
                let conv_in =
                    take_matrix_raw(&mut tensors, &format!("{p}.conv.in_proj.weight"), storage)?;
                let conv_out =
                    take_matrix_raw(&mut tensors, &format!("{p}.conv.out_proj.weight"), storage)?;
                // Stored `[dim, 1, k]`; the trailing singleton makes it `[dim, k]`.
                let raw = take_f32(&mut tensors, &format!("{p}.conv.conv.weight"))?;
                let shape = raw.shape().to_vec();
                if shape.len() != 3 || shape[1] != 1 {
                    return Err(LoadError::BadConfig(format!(
                        "{p}.conv.conv.weight has shape {shape:?}, expected [dim, 1, k]"
                    )));
                }
                let kernel = Tensor::from_vec(raw.into_data(), &[shape[0], shape[2]])
                    .map_err(LoadError::Tensor)?;
                (Some(conv_in), Some(conv_out), Some(kernel))
            };
            layers.push(LayerWeights {
                rms_att: take_f32(&mut tensors, &format!("{p}.operator_norm.weight"))?,
                wq,
                wk,
                wv,
                wo,
                rms_ffn: take_f32(&mut tensors, &format!("{p}.ffn_norm.weight"))?,
                w1: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.feed_forward.w1.weight"),
                    storage,
                )?,
                w2: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.feed_forward.w2.weight"),
                    storage,
                )?,
                w3: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.feed_forward.w3.weight"),
                    storage,
                )?,
                rms_q,
                rms_k,
                conv_in,
                conv_out,
                conv_kernel,
                is_attention: attn,
            });
        }
        // Tied embeddings store no head; tolerate an untied one anyway.
        let wcls = match take_optional_raw(&mut tensors, "lm_head.weight")? {
            Some(raw) => Some(matrix_from_raw(raw, storage)?),
            None => None,
        };
        Ok(Self::with_backend(
            config,
            is_attention,
            ModelWeights {
                token_embedding,
                layers,
                rms_final,
                wcls,
            },
            backend,
        ))
    }

    /// Build a model over an explicit backend.
    pub fn with_backend(
        config: Config,
        is_attention: Vec<bool>,
        weights: ModelWeights,
        backend: Box<dyn Backend>,
    ) -> Self {
        let weights = Arc::new(weights);
        let mut backend = backend;
        backend.prepare(
            Arc::clone(&weights),
            KvSpec {
                n_layers: config.n_layers,
                kv_dim: config.kv_dim(),
                seq_len: config.seq_len,
            },
        );

        let mut norms = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            norms.push(Lfm2Norms {
                op: backend.upload(layer.rms_att.data()),
                ffn: backend.upload(layer.rms_ffn.data()),
                q: layer.rms_q.as_ref().map(|t| backend.upload(t.data())),
                k: layer.rms_k.as_ref().map(|t| backend.upload(t.data())),
            });
        }
        let rms_final = backend.upload(weights.rms_final.data());

        let dim = config.dim;
        // One kernel and one ring per conv layer; attention layers hold None.
        let mut conv_kernels = Vec::with_capacity(weights.layers.len());
        let mut conv_state = Vec::with_capacity(weights.layers.len());
        let mut conv_k = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            match &layer.conv_kernel {
                Some(kernel) => {
                    let k = kernel.shape()[1];
                    conv_kernels.push(Some(backend.upload(kernel.data())));
                    conv_k.push(k);
                    // The ring must start at zero: a cold decode reads it as
                    // history before anything has been written. `alloc` is
                    // zeroed on the CPU backends but uninitialised on Metal,
                    // so seed it explicitly rather than depending on either.
                    let ring = backend.alloc(dim * (k - 1).max(1));
                    let zeros = backend.upload(&vec![0.0f32; dim * (k - 1).max(1)]);
                    backend.copy(zeros, 0, ring, 0, dim * (k - 1).max(1));
                    conv_state.push(Some(ring));
                }
                None => {
                    conv_kernels.push(None);
                    conv_state.push(None);
                    conv_k.push(0);
                }
            }
        }

        let kv_dim = config.kv_dim();
        let q_dim = config.q_dim();
        let state = State {
            x: backend.alloc(dim),
            xb: backend.alloc(dim),
            q: backend.alloc(q_dim),
            k: backend.alloc(kv_dim),
            v: backend.alloc(kv_dim),
            attn: backend.alloc(q_dim),
            o: backend.alloc(dim),
            bcx: backend.alloc(3 * dim),
            h: backend.alloc(dim),
            co: backend.alloc(dim),
            cy: backend.alloc(dim),
            gate: backend.alloc(config.hidden_dim),
            up: backend.alloc(config.hidden_dim),
            mlp: backend.alloc(dim),
            hidden: backend.alloc(dim),
            att: backend.alloc(config.n_heads * config.seq_len),
            logits: backend.alloc(config.vocab_size),
            logits_host: vec![0.0; config.vocab_size],
            prefill: None,
        };
        Lfm2 {
            config,
            is_attention,
            conv_k,
            backend,
            norms,
            conv_kernels,
            conv_state,
            rms_final,
            state,
        }
    }

    /// The storage width the weights were loaded at.
    pub fn storage(&self) -> &'static str {
        self.backend.storage()
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Process one token at absolute position `pos` and return the logits for
    /// the next token.
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        let config = self.config;
        let dim = config.dim;
        let hidden = config.hidden_dim;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let head_size = config.head_size();
        let eps = config.rms_eps;
        let attn_spec = AttnSpec {
            n_heads: config.n_heads,
            kv_mul: config.kv_mul(),
            head_size,
            kv_dim,
            seq_len: config.seq_len,
        };
        let rope_spec = RopeSpec {
            head_size,
            theta: config.rope_theta,
            interleaved: config.rope_interleaved,
        };
        let positions = [pos];

        let Self {
            backend,
            norms,
            conv_kernels,
            conv_state,
            rms_final,
            state,
            is_attention,
            conv_k,
            ..
        } = self;

        backend.embed(token, state.x, 0);

        for (l, (n, attn)) in norms
            .iter()
            .zip(is_attention.iter())
            .enumerate()
            .take(config.n_layers)
        {
            backend.rms_norm(state.xb, state.x, n.op, 1, dim, eps);

            if *attn {
                backend.project(l, Proj::Q, state.xb, state.q, 1);
                backend.project(l, Proj::K, state.xb, state.k, 1);
                backend.project(l, Proj::V, state.xb, state.v, 1);

                // LFM2 normalises q and k per head on every attention layer.
                if let Some(w) = n.q {
                    backend.qk_norm(state.q, 1, q_dim, head_size, w, eps);
                }
                if let Some(w) = n.k {
                    backend.qk_norm(state.k, 1, kv_dim, head_size, w, eps);
                }
                backend.rope(state.q, state.k, q_dim, kv_dim, 1, &positions, rope_spec);
                backend.kv_write(l, &[pos], state.k, state.v, 1);

                backend.attention(
                    state.q,
                    state.attn,
                    state.att,
                    l,
                    &[Window {
                        att_start: 0,
                        slot: pos,
                        shared: (0, 0),
                    }],
                    attn_spec,
                );

                backend.project(l, Proj::O, state.attn, state.o, 1);
                backend.add_in_place(state.x, state.o, dim);
            } else {
                backend.project(l, Proj::ConvIn, state.xb, state.bcx, 1);
                // h = B * x over the first and last thirds.
                backend.mul(state.h, 0, state.bcx, 0, state.bcx, 2 * dim, dim);
                let kernel = conv_kernels[l].expect("conv layer carries a kernel");
                let ring = conv_state[l].expect("conv layer carries a ring");
                backend.conv_decode(state.co, state.h, kernel, ring, dim, conv_k[l]);
                // y = C * conv_out over the middle third.
                backend.mul(state.cy, 0, state.bcx, dim, state.co, 0, dim);
                backend.project(l, Proj::ConvOut, state.cy, state.o, 1);
                backend.add_in_place(state.x, state.o, dim);
            }

            // Feed-forward RMSNorm, then the gated MLP.
            backend.rms_norm(state.xb, state.x, n.ffn, 1, dim, eps);
            backend.project(l, Proj::Gate, state.xb, state.gate, 1);
            backend.project(l, Proj::Up, state.xb, state.up, 1);
            backend.silu_mul(state.gate, state.up, hidden);
            backend.project(l, Proj::Down, state.gate, state.mlp, 1);
            backend.add_in_place(state.x, state.mlp, dim);
        }

        backend.rms_norm(state.xb, state.x, *rms_final, 1, dim, eps);
        backend.logits(state.xb, state.logits);
        backend.read(state.logits, &mut state.logits_host);
        &state.logits_host
    }

    /// Process `tokens` as one batch starting at absolute position `start`,
    /// filling the caches, and return the logits for the token after the last one.
    pub fn prefill(&mut self, tokens: &[usize], start: usize) -> &[f32] {
        let n_tok = tokens.len();
        assert!(
            start + n_tok <= self.config.seq_len,
            "prefill runs past the context length"
        );
        if n_tok == 0 {
            return &self.state.logits_host;
        }
        if !matches!(self.storage(), "f32" | "bf16" | "q8_0x4") {
            for (offset, &token) in tokens.iter().enumerate() {
                self.forward(token, start + offset);
            }
            return &self.state.logits_host;
        }
        self.prefill_batched(tokens, start);
        &self.state.logits_host
    }

    /// One batched forward over a whole prompt.
    fn prefill_batched(&mut self, tokens: &[usize], start: usize) {
        let n_tok = tokens.len();
        if n_tok == 0 {
            return;
        }
        // Every row attends from position 0; slots march with the rows.
        let slots: Vec<usize> = (0..n_tok).map(|i| start + i).collect();
        let positions = slots.clone();
        let windows: Vec<Window> = slots
            .iter()
            .map(|&slot| Window {
                att_start: 0,
                slot,
                shared: (0, 0),
            })
            .collect();

        let config = self.config;
        let dim = config.dim;
        let hidden = config.hidden_dim;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let head_size = config.head_size();
        let eps = config.rms_eps;
        let attn_spec = AttnSpec {
            n_heads: config.n_heads,
            kv_mul: config.kv_mul(),
            head_size,
            kv_dim,
            seq_len: config.seq_len,
        };
        let rope_spec = RopeSpec {
            head_size,
            theta: config.rope_theta,
            interleaved: config.rope_interleaved,
        };

        let Self {
            backend,
            norms,
            conv_kernels,
            conv_state,
            rms_final,
            state,
            is_attention,
            conv_k,
            ..
        } = self;

        if state.prefill.as_ref().is_none_or(|p| p.n_tok < n_tok) {
            state.prefill = Some(PrefillBuffers::new(backend.as_mut(), n_tok, &config));
        }
        let pb = *state.prefill.as_ref().expect("built above");

        for (row, &token) in tokens.iter().enumerate() {
            backend.embed(token, pb.x, row * dim);
        }

        for (l, (n, attn)) in norms
            .iter()
            .zip(is_attention.iter())
            .enumerate()
            .take(config.n_layers)
        {
            backend.rms_norm(pb.xb, pb.x, n.op, n_tok, dim, eps);

            if *attn {
                backend.project(l, Proj::Q, pb.xb, pb.q, n_tok);
                backend.project(l, Proj::K, pb.xb, pb.k, n_tok);
                backend.project(l, Proj::V, pb.xb, pb.v, n_tok);

                if let Some(w) = n.q {
                    backend.qk_norm(pb.q, n_tok, q_dim, head_size, w, eps);
                }
                if let Some(w) = n.k {
                    backend.qk_norm(pb.k, n_tok, kv_dim, head_size, w, eps);
                }
                backend.rope(pb.q, pb.k, q_dim, kv_dim, n_tok, &positions, rope_spec);
                backend.kv_write(l, &slots, pb.k, pb.v, n_tok);

                backend.attention(pb.q, pb.attn, pb.att, l, &windows, attn_spec);

                backend.project(l, Proj::O, pb.attn, pb.o, n_tok);
                backend.add_in_place(pb.x, pb.o, n_tok * dim);
            } else {
                backend.project(l, Proj::ConvIn, pb.xb, pb.bcx, n_tok);
                // h = B * x, row by row over the thirds.
                for row in 0..n_tok {
                    backend.mul(
                        pb.h,
                        row * dim,
                        pb.bcx,
                        row * 3 * dim,
                        pb.bcx,
                        row * 3 * dim + 2 * dim,
                        dim,
                    );
                }
                let kernel = conv_kernels[l].expect("conv layer carries a kernel");
                let k = conv_k[l];
                backend.conv_prefill(pb.co, pb.h, kernel, n_tok, dim, k);
                // Seed the decode ring with the last k-1 rows of h.
                if let Some(ring) = conv_state[l] {
                    let take = n_tok.min(k - 1);
                    backend.copy(
                        pb.h,
                        (n_tok - take) * dim,
                        ring,
                        (k - 1 - take) * dim,
                        take * dim,
                    );
                }
                // y = C * conv_out, row by row.
                for row in 0..n_tok {
                    backend.mul(
                        pb.cy,
                        row * dim,
                        pb.bcx,
                        row * 3 * dim + dim,
                        pb.co,
                        row * dim,
                        dim,
                    );
                }
                backend.project(l, Proj::ConvOut, pb.cy, pb.o, n_tok);
                backend.add_in_place(pb.x, pb.o, n_tok * dim);
            }

            backend.rms_norm(pb.xb, pb.x, n.ffn, n_tok, dim, eps);
            backend.project(l, Proj::Gate, pb.xb, pb.gate, n_tok);
            backend.project(l, Proj::Up, pb.xb, pb.up, n_tok);
            backend.silu_mul(pb.gate, pb.up, n_tok * hidden);
            backend.project(l, Proj::Down, pb.gate, pb.o, n_tok);
            backend.add_in_place(pb.x, pb.o, n_tok * dim);
        }

        // Logits for the final prompt row only.
        let last = (n_tok - 1) * dim;
        backend.copy(pb.x, last, state.hidden, 0, dim);
        backend.rms_norm(state.xb, state.hidden, *rms_final, 1, dim, config.rms_eps);
        backend.logits(state.xb, state.logits);
        backend.read(state.logits, &mut state.logits_host);
    }
}

impl LanguageModel for Lfm2 {
    fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        Lfm2::forward(self, token, pos)
    }

    fn context_len(&self) -> usize {
        self.config.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::ModelWeights;
    use crate::formats::llama2c::LayerWeights;
    use crate::ops::Matrix;

    fn test_config() -> Config {
        Config {
            dim: 32,
            hidden_dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 32,
            seq_len: 64,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            rope_interleaved: false,
            head_dim: 8,
        }
    }

    fn fill(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
        Matrix::from_f32(Tensor::from_vec(fill(rows * cols, seed), &[rows, cols]).unwrap())
    }

    fn vector(len: usize, seed: u64) -> Tensor {
        Tensor::from_vec(fill(len, seed), &[len]).unwrap()
    }

    /// Layer kinds from the caller, with every shape distinct enough to catch
    /// a transposition.
    fn weights(config: &Config, mask: &[bool]) -> ModelWeights {
        weights_at(config, mask, "f32")
    }

    /// The same logical model in a chosen storage. The Metal backend holds
    /// weights as bf16, so a cross-backend check must compare bf16 against
    /// bf16 rather than measuring the format.
    fn weights_at(config: &Config, mask: &[bool], storage: &str) -> ModelWeights {
        let cast = |m: Matrix| match storage {
            "bf16" => Matrix::from_bf16_bits(m.to_bf16_bits(), m.out_dim(), m.in_dim()),
            _ => m,
        };
        let (dim, hidden, vocab) = (config.dim, config.hidden_dim, config.vocab_size);
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let build_layer = |l: usize, attn: bool| {
            let s = 100 * (l as u64 + 1);
            let (wq, wk, wv, wo, rq, rk, ci, co, ck) = if attn {
                (
                    Some(cast(matrix(q_dim, dim, s + 2))),
                    Some(cast(matrix(kv_dim, dim, s + 3))),
                    Some(cast(matrix(kv_dim, dim, s + 4))),
                    Some(cast(matrix(dim, q_dim, s + 5))),
                    Some(vector(config.head_dim, s + 10)),
                    Some(vector(config.head_dim, s + 11)),
                    None,
                    None,
                    None,
                )
            } else {
                (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(cast(matrix(3 * dim, dim, s + 12))),
                    Some(cast(matrix(dim, dim, s + 13))),
                    // Depthwise kernel [dim, 3], small non-trivial values.
                    Some(Tensor::from_vec(fill(dim * 3, s + 14), &[dim, 3]).unwrap()),
                )
            };
            LayerWeights {
                rms_att: vector(dim, s + 1),
                wq,
                wk,
                wv,
                wo,
                rms_ffn: vector(dim, s + 6),
                w1: cast(matrix(hidden, dim, s + 7)),
                w2: cast(matrix(dim, hidden, s + 8)),
                w3: cast(matrix(hidden, dim, s + 9)),
                rms_q: rq,
                rms_k: rk,
                conv_in: ci,
                conv_out: co,
                conv_kernel: ck,
                is_attention: attn,
            }
        };
        let layers = vec![build_layer(0, mask[0]), build_layer(1, mask[1])];
        ModelWeights {
            token_embedding: cast(matrix(vocab, dim, 1)),
            layers,
            rms_final: vector(dim, 2),
            wcls: None,
        }
    }

    fn model_at(name: &str, storage: &str) -> Lfm2 {
        let config = test_config();
        let mask = vec![false, true];
        let weights = weights_at(&config, &mask, storage);
        let backend = crate::backends::open(name).unwrap();
        Lfm2::with_backend(config, mask, weights, backend)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Prefill must match decode with each layer kind in isolation, so a
    /// failure points at the conv path or the attention path, not both.
    #[test]
    fn prefill_matches_decode_per_layer_kind() {
        let prompt: Vec<usize> = vec![5, 11, 2, 30];
        for (label, mask) in [
            ("attn-only", vec![true, true]),
            ("conv-only", vec![false, false]),
        ] {
            let config = test_config();
            let w = weights(&config, &mask);
            let mut batched = Lfm2::with_backend(
                config,
                mask.clone(),
                w,
                crate::backends::open("cpu").unwrap(),
            );
            let got = batched.prefill(&prompt, 0).to_vec();
            let w2 = weights(&test_config(), &mask);
            let mut sequential = Lfm2::with_backend(
                test_config(),
                mask,
                w2,
                crate::backends::open("cpu").unwrap(),
            );
            let mut want = Vec::new();
            for (pos, &token) in prompt.iter().enumerate() {
                want = sequential.forward(token, pos).to_vec();
            }
            let diff = max_diff(&got, &want);
            eprintln!("{label}: diff {diff}");
            assert!(diff <= 1e-3, "{label}: prefill differs by {diff}");
        }
    }

    /// The two conv paths must agree with each other: the batched prefill
    /// against stepped single-row decodes through a zero ring. This caught a
    /// real bug where the decode path read only the oldest history row.
    #[test]
    fn conv_prefill_matches_stepped_decode() {
        use crate::backends::Backend;
        use crate::backends::cpu::Cpu;
        let mut b = Cpu::new();
        let dim = 8;
        let k = 3;
        let rows = 5;
        // deterministic h and kernel
        let h: Vec<f32> = (0..rows * dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let kw: Vec<f32> = (0..dim * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let hb = b.upload(&h);
        let kb = b.upload(&kw);
        let out = b.alloc(rows * dim);
        b.conv_prefill(out, hb, kb, rows, dim, k);
        let mut got = vec![0.0; rows * dim];
        b.read(out, &mut got);

        // stepped: state starts zero, one conv_decode per row
        let state = b.alloc(dim * (k - 1));
        let mut want = vec![0.0; rows * dim];
        for r in 0..rows {
            let xr = b.upload(&h[r * dim..(r + 1) * dim]);
            let orow = b.alloc(dim);
            b.conv_decode(orow, xr, kb, state, dim, k);
            let mut one = vec![0.0; dim];
            b.read(orow, &mut one);
            want[r * dim..(r + 1) * dim].copy_from_slice(&one);
        }
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        eprintln!("conv prefill-vs-stepped diff {diff}");
        assert!(diff <= 1e-5, "conv paths disagree by {diff}");
    }

    /// The GPU backend, when one is compiled in and present, must reproduce
    /// the scalar oracle through the conv ring as well as attention.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_the_scalar_oracle() {
        let opened = crate::backends::open("metal");
        let Ok(backend) = opened else {
            let error = match opened {
                Err(error) => error,
                Ok(_) => unreachable!(),
            };
            assert!(
                error.to_string().contains("no Metal device"),
                "Metal is present but failed to start: {error}"
            );
            eprintln!("skipping: no Metal device available");
            return;
        };
        drop(backend);

        // Same prompt through each layer kind in isolation, so a failure
        // points at the conv kernels or the shared attention path.
        for (label, mask) in [
            ("attn-only", vec![true, true]),
            ("conv-only", vec![false, false]),
        ] {
            let config = test_config();
            let weights = weights_at(&config, &mask, "bf16");
            let mut gpu = Lfm2::with_backend(
                config,
                mask.clone(),
                weights,
                crate::backends::open("metal").unwrap(),
            );
            let weights = weights_at(&test_config(), &mask, "bf16");
            let mut oracle = Lfm2::with_backend(
                test_config(),
                mask,
                weights,
                crate::backends::open("reference").unwrap(),
            );
            let prompt: Vec<usize> = vec![5, 11, 2, 30];
            for (pos, &token) in prompt.iter().enumerate() {
                let a = gpu.forward(token, pos).to_vec();
                let b = oracle.forward(token, pos).to_vec();
                let diff = max_diff(&a, &b);
                let scale = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!("metal {label} pos {pos}: diff {diff} scale {scale}");
                assert!(diff <= 1e-3, "metal {label} differs at {pos} by {diff}");
            }
        }

        let build = |name: &str| model_at(name, "bf16");
        let prompt: Vec<usize> = vec![5, 11, 2, 30, 7, 7, 19, 1, 24];
        let mut gpu = build("metal");
        let mut oracle = build("reference");
        let mut want = Vec::new();
        for (pos, &token) in prompt.iter().enumerate() {
            let a = gpu.forward(token, pos).to_vec();
            want = oracle.forward(token, pos).to_vec();
            let diff = max_diff(&a, &want);
            assert!(diff <= 1e-3, "decode differs at {pos} by {diff}");
            assert_eq!(argmax(&a), argmax(&want), "different token at {pos}");
        }

        let got = gpu.prefill(&prompt, 0).to_vec();
        let diff = max_diff(&got, &want);
        assert!(diff <= 1e-3, "prefill differs from decode by {diff}");
        assert_eq!(
            argmax(&got),
            argmax(&want),
            "prefill picked a different token"
        );
    }
}
