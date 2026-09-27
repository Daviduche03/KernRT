//! Llama-2 style transformer forward pass, covering both the llama2.c
//! TinyStories checkpoints and HuggingFace Llama-format models such as SmolLM2.
//!
//! One token at a time with a key/value cache, mirroring `run.c` step for step
//! so the two can be compared numerically. The heavy projections go through
//! [`crate::ops::gemv`]; the small element-wise pieces (RMSNorm, RoPE, softmax,
//! SiLU gating) live here because they are transformer-specific and operate on
//! the state buffers in place.

use crate::formats::llama2c::{Checkpoint, LayerWeights, Llama2cError};
use crate::formats::safetensors::{self, RawTensor, SafeTensorsError};
use crate::ops::{
    Matrix, Q8_0_BLOCK, Q8Vec, dot, gemm_nt_into, gemm_q8_0x4_batched_into, gemv_into,
    gemv_q8_0_into, softmax_in_place,
};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

/// Model hyperparameters, independent of which file format supplied them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// `true` rotates adjacent pairs (llama2.c); `false` rotates the two halves
    /// of each head (HuggingFace Llama, and therefore SmolLM2).
    pub rope_interleaved: bool,
}

impl Config {
    /// Elements per attention head.
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Width of one key/value vector (smaller than `dim` under grouped-query
    /// attention).
    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }

    /// How many query heads share each key/value head.
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

/// Errors from loading a HuggingFace-format model.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    SafeTensors(SafeTensorsError),
    MissingTensor(String),
    BadConfig(String),
    /// A tensor's shape or element count was inconsistent.
    Tensor(crate::tensor::TensorError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "i/o error: {e}"),
            LoadError::Json(e) => write!(f, "invalid config.json: {e}"),
            LoadError::SafeTensors(e) => write!(f, "{e}"),
            LoadError::MissingTensor(name) => write!(f, "weight file is missing {name:?}"),
            LoadError::BadConfig(msg) => write!(f, "invalid config: {msg}"),
            LoadError::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io(e) => Some(e),
            LoadError::Json(e) => Some(e),
            LoadError::SafeTensors(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        LoadError::Json(e)
    }
}

impl From<SafeTensorsError> for LoadError {
    fn from(e: SafeTensorsError) -> Self {
        LoadError::SafeTensors(e)
    }
}

impl From<crate::tensor::TensorError> for LoadError {
    fn from(e: crate::tensor::TensorError) -> Self {
        LoadError::Tensor(e)
    }
}

/// Buffers reused across tokens so the forward pass does not allocate per step.
struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    /// Projection outputs, one buffer per shape so the decode loop stops
    /// allocating 211 vectors per token. `q`/`k`/`v` fit `dim` and `kv_dim`,
    /// the gated MLP's two branches fit `hidden_dim`, and the MLP result fits
    /// `dim`.
    buf_q: Vec<f32>,
    buf_k: Vec<f32>,
    buf_v: Vec<f32>,
    buf_o: Vec<f32>,
    buf_gate: Vec<f32>,
    buf_up: Vec<f32>,
    buf_mlp: Vec<f32>,
    /// Scratch for the quantized form of the current activation (Q8_0 only).
    q8: Q8Vec,
    /// Batched scratch, sized on first use because the prompt length is only
    /// known then.
    prefill: Option<PrefillBuffers>,
    /// f32 copies of every projection, built once on first prefill. Built
    /// lazily because most runs decode one token at a time and never pay for
    /// them, and kept beside `prefill` so a single `&mut self.state` borrow
    /// covers both while the layers are read through a separate immutable one.
    prefill_weights: Option<PrefillWeights>,
}

/// Row-major `[n_tok, width]` scratch for a batched forward pass.
///
/// Sized on the first prompt and reused until a longer one arrives. The decode
/// loop keeps its own buffers in [`State`], so filling these cannot disturb a
/// decode run.
struct PrefillBuffers {
    /// Rows the buffers were sized for; requests up to this length reuse them.
    n_tok: usize,
    /// `[n_tok, dim]` residual stream.
    x: Vec<f32>,
    /// `[n_tok, dim]` norm output, and later the attention output.
    xb: Vec<f32>,
    /// `[n_tok, dim]` query.
    q: Vec<f32>,
    /// `[n_tok, kv_dim]` rotated keys.
    k: Vec<f32>,
    /// `[n_tok, kv_dim]` values.
    v: Vec<f32>,
    /// `[n_tok, dim]` projection output.
    o: Vec<f32>,
    /// `[n_tok, hidden_dim]` gate, then the gated product in place.
    gate: Vec<f32>,
    /// `[n_tok, hidden_dim]` up projection.
    up: Vec<f32>,
    /// Row-major Q8 activations for the native batched Q8_0x4 GEMM. The
    /// quantizer is kept separately so filling a prompt is allocation-free.
    q8_quants: Vec<i8>,
    q8_scales_f16: Vec<u16>,
    q8_batch: Q8Vec,
    /// `[n_heads, seq_len]` attention scores for the row being processed.
    att: Vec<f32>,
}

/// f32 copies of one layer's projections, for the batched GEMMs.
///
/// Prefill only ever runs on short prompts, and a prompt's activations are far
/// too small to matter on their own — what matters is that each of the seven
/// projections reads its weights once instead of once per token. Converting
/// to f32 also means one kernel serves every storage the model was loaded at,
/// so the batched path does not grow a second copy of itself per storage.
struct PrefillLayer {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

/// The f32 projection copies for every layer, built once on first prefill.
///
/// The shapes are not repeated here: every call site already holds a
/// [`Config`], and one copy of the truth is less to keep in sync than two.
struct PrefillWeights {
    layers: Vec<PrefillLayer>,
}

impl PrefillBuffers {
    /// Scratch for `n_tok` rows at the given model shape.
    fn new(n_tok: usize, config: &Config) -> Self {
        let dim = config.dim;
        let kv_dim = config.kv_dim();
        let hidden = config.hidden_dim;
        Self {
            n_tok,
            x: vec![0.0; n_tok * dim],
            xb: vec![0.0; n_tok * dim],
            q: vec![0.0; n_tok * dim],
            k: vec![0.0; n_tok * kv_dim],
            v: vec![0.0; n_tok * kv_dim],
            o: vec![0.0; n_tok * dim],
            gate: vec![0.0; n_tok * hidden],
            up: vec![0.0; n_tok * hidden],
            q8_quants: vec![0; n_tok * hidden],
            q8_scales_f16: vec![0; n_tok * hidden.div_ceil(Q8_0_BLOCK)],
            q8_batch: Q8Vec::quantize(&[]),
            att: vec![0.0; config.n_heads * config.seq_len],
        }
    }
}

impl PrefillWeights {
    /// Copy every projection of every layer to f32.
    ///
    /// One conversion per weight for the whole session, not one per layer per
    /// prompt: the batched GEMM reads each matrix as a contiguous `[out, in]`
    /// f32 block, which works regardless of which storage it came from.
    fn build(layers: &[LayerWeights]) -> Self {
        let copy = |w: &Matrix| w.to_f32_tensor().into_data();
        Self {
            layers: layers
                .iter()
                .map(|layer| PrefillLayer {
                    wq: copy(&layer.wq),
                    wk: copy(&layer.wk),
                    wv: copy(&layer.wv),
                    wo: copy(&layer.wo),
                    w1: copy(&layer.w1),
                    w2: copy(&layer.w2),
                    w3: copy(&layer.w3),
                })
                .collect(),
        }
    }
}

/// A loaded transformer.
pub struct Llama {
    config: Config,
    token_embedding: Matrix,
    layers: Vec<LayerWeights>,
    rms_final: Tensor,
    /// `None` when the classifier is tied to the token embedding.
    wcls: Option<Matrix>,
    state: State,
}

impl Llama {
    /// Load a llama2.c checkpoint (the TinyStories format).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Llama2cError> {
        Ok(Self::from_checkpoint(crate::formats::llama2c::load(path)?))
    }

    /// Adopt a parsed llama2.c checkpoint. That format predates configurable
    /// RoPE, so its settings are fixed.
    pub fn from_checkpoint(checkpoint: Checkpoint) -> Self {
        let c = checkpoint.config;
        let config = Config {
            dim: c.dim,
            hidden_dim: c.hidden_dim,
            n_layers: c.n_layers,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            vocab_size: c.vocab_size,
            seq_len: c.seq_len,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_interleaved: true,
        };
        Self::new(
            config,
            checkpoint.token_embedding,
            checkpoint.layers,
            checkpoint.rms_final,
            checkpoint.wcls,
        )
    }

    /// Load a HuggingFace-format model directory: `config.json` plus
    /// `model.safetensors`.
    ///
    /// The tensor names are the standard `model.layers.N.self_attn.q_proj` set;
    /// their shapes already match this runtime's `[out, in]` layout, including
    /// the narrower `k_proj`/`v_proj` of grouped-query attention.
    pub fn from_hf(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let (config, bf16) = parse_hf_config(&dir.join("config.json"))?;
        // Measurement aids, so one binary can A/B storage widths:
        //   KERNRT_Q8_0=1     quantize the weights to int8 (row-major Q8_0)
        //   KERNRT_Q8_0X4=1   four-row interleaved layout and lane-SDOT kernel
        //   KERNRT_ALL_F32=1  keep f32 instead of bf16
        let storage = if std::env::var_os("KERNRT_Q8_0X4").is_some() {
            Storage::Q8_0x4
        } else if std::env::var_os("KERNRT_Q8_0").is_some() {
            Storage::Q8_0
        } else if bf16 && std::env::var_os("KERNRT_ALL_F32").is_none() {
            Storage::Bf16
        } else {
            Storage::F32
        };
        // The raw reader keeps bf16 weights in their stored form, so a bf16
        // model never round-trips through `f32`.
        let mut tensors = safetensors::load_raw(dir.join("model.safetensors"))?;

        let token_embedding = take_matrix_raw(&mut tensors, "model.embed_tokens.weight", storage)?;
        let rms_final = take_f32(&mut tensors, "model.norm.weight")?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for l in 0..config.n_layers {
            let p = format!("model.layers.{l}");
            layers.push(LayerWeights {
                rms_att: take_f32(&mut tensors, &format!("{p}.input_layernorm.weight"))?,
                wq: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.self_attn.q_proj.weight"),
                    storage,
                )?,
                wk: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.self_attn.k_proj.weight"),
                    storage,
                )?,
                wv: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.self_attn.v_proj.weight"),
                    storage,
                )?,
                wo: take_matrix_raw(
                    &mut tensors,
                    &format!("{p}.self_attn.o_proj.weight"),
                    storage,
                )?,
                rms_ffn: take_f32(
                    &mut tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                w1: take_matrix_raw(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), storage)?,
                w2: take_matrix_raw(&mut tensors, &format!("{p}.mlp.down_proj.weight"), storage)?,
                w3: take_matrix_raw(&mut tensors, &format!("{p}.mlp.up_proj.weight"), storage)?,
            });
        }
        // Tied models store no lm_head.
        let wcls = match take_optional_raw(&mut tensors, "lm_head.weight")? {
            Some(raw) => Some(matrix_from_raw(raw, storage)?),
            None => None,
        };
        Ok(Self::new(config, token_embedding, layers, rms_final, wcls))
    }

    fn new(
        config: Config,
        token_embedding: Matrix,
        layers: Vec<LayerWeights>,
        rms_final: Tensor,
        wcls: Option<Matrix>,
    ) -> Self {
        let kv_dim = config.kv_dim();
        let state = State {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            buf_q: vec![0.0; config.dim],
            buf_k: vec![0.0; kv_dim],
            buf_v: vec![0.0; kv_dim],
            buf_o: vec![0.0; config.dim],
            buf_gate: vec![0.0; config.hidden_dim],
            buf_up: vec![0.0; config.hidden_dim],
            buf_mlp: vec![0.0; config.dim],
            q8: Q8Vec {
                scales: Vec::new(),
                scales_f16: Vec::new(),
                quants: Vec::new(),
            },
            prefill: None,
            prefill_weights: None,
        };
        Llama {
            config,
            token_embedding,
            layers,
            rms_final,
            wcls,
            state,
        }
    }

    /// Process `tokens` as one batch starting at absolute position `start`,
    /// filling the key/value cache, and return the logits for the token after
    /// the last one.
    ///
    /// [`forward`](Self::forward) is a single token wide, so a prompt otherwise
    /// costs one full pass per token. This runs the same transformer with the
    /// batch dimension `n_tok` in the outer loop: projections become
    /// `[n_tok, in] x [in, out]` matrix multiplies through the packed GEMM, and
    /// attention is causal over the whole batch at once.
    ///
    /// The result is the same as stepping [`forward`](Self::forward) over the
    /// prompt, up to floating-point summation order, which the tests pin down.
    ///
    /// `start + tokens.len()` must not exceed the context length.
    pub fn prefill(&mut self, tokens: &[usize], start: usize) -> &[f32] {
        let n_tok = tokens.len();
        assert!(
            start + n_tok <= self.config.seq_len,
            "prefill runs past the context length"
        );
        if n_tok == 0 {
            return &self.state.logits;
        }
        // Q8_0x4 has an exact batched integer path. Plain Q8_0 remains on the
        // decode loop until it gains the same block-interleaved kernel.
        if !matches!(self.storage(), "f32" | "bf16" | "q8_0x4") {
            for (offset, &token) in tokens.iter().enumerate() {
                self.forward(token, start + offset);
            }
            return &self.state.logits;
        }
        self.prefill_batched(tokens, start);
        &self.state.logits
    }

    /// One batched forward over a whole prompt.
    ///
    /// The same arithmetic as [`forward`](Self::forward) stepped token by
    /// token, except that each projection multiplies every row at once instead
    /// of one row at a time, so each weight matrix is read once rather than
    /// once per token. Attention stays per row: its scores are causal, so row
    /// `i` attends to `start..=start + i` and there is nothing to batch.
    fn prefill_batched(&mut self, tokens: &[usize], start: usize) {
        let config = self.config;
        let n_tok = tokens.len();
        let dim = config.dim;
        let hidden = config.hidden_dim;
        let kv_dim = config.kv_dim();
        let head_size = config.head_size();
        let seq_len = config.seq_len;

        let token_embedding = &self.token_embedding;
        let rms_final = &self.rms_final;
        let classifier = self.wcls.as_ref().unwrap_or(&self.token_embedding);
        let layers = &self.layers;
        let native_q8x4 = matches!(
            layers.first().map(|layer| &layer.wq),
            Some(Matrix::Q8_0x4(_))
        );
        let state = &mut self.state;

        if !native_q8x4 && state.prefill_weights.is_none() {
            state.prefill_weights = Some(PrefillWeights::build(layers));
        }
        if state.prefill.as_ref().is_none_or(|p| p.n_tok < n_tok) {
            state.prefill = Some(PrefillBuffers::new(n_tok, &config));
        }
        let weights = state.prefill_weights.as_ref();
        let pf = state.prefill.as_mut().expect("built above");
        // Split out so attention can take this row's query and that row's
        // output at the same time, which a second borrow of `pf` would not
        // allow.
        let PrefillBuffers {
            x,
            xb,
            q,
            k,
            v,
            o,
            gate,
            up,
            q8_quants,
            q8_scales_f16,
            q8_batch,
            att,
            ..
        } = pf;

        for (row, &token) in tokens.iter().enumerate() {
            let embedding = token_embedding.row(token);
            x[row * dim..(row + 1) * dim].copy_from_slice(&embedding);
        }

        for (l, layer) in layers.iter().take(config.n_layers).enumerate() {
            let w = weights.map(|weights| &weights.layers[l]);
            let loff = l * seq_len * kv_dim;

            // Attention RMSNorm, one row at a time: it is element-wise, so
            // there is no product to batch and nothing to gain from stacking
            // the rows.
            for row in 0..n_tok {
                let at = row * dim;
                rms_norm(
                    &mut xb[at..at + dim],
                    &x[at..at + dim],
                    layer.rms_att.data(),
                    config.rms_eps,
                );
            }

            if native_q8x4 {
                quantize_rows_q8(q8_batch, xb, q8_quants, q8_scales_f16, dim);
            }
            batched_project(
                &layer.wq,
                w.map(|w| w.wq.as_slice()),
                &*xb,
                q8_quants,
                q8_scales_f16,
                n_tok,
                q,
            );
            batched_project(
                &layer.wk,
                w.map(|w| w.wk.as_slice()),
                &*xb,
                q8_quants,
                q8_scales_f16,
                n_tok,
                k,
            );
            batched_project(
                &layer.wv,
                w.map(|w| w.wv.as_slice()),
                &*xb,
                q8_quants,
                q8_scales_f16,
                n_tok,
                v,
            );

            // RoPE rotates q and k before k is cached, matching both
            // references, so it runs over the batched rows before any row
            // reads the cache.
            for row in 0..n_tok {
                let pos = start + row;
                let qr = row * dim;
                let kr = row * kv_dim;
                if config.rope_interleaved {
                    rope_interleaved(
                        &mut q[qr..qr + dim],
                        &mut k[kr..kr + kv_dim],
                        pos,
                        head_size,
                        kv_dim,
                        config.rope_theta,
                    );
                } else {
                    rope_half(&mut q[qr..qr + dim], pos, head_size, config.rope_theta);
                    rope_half(&mut k[kr..kr + kv_dim], pos, head_size, config.rope_theta);
                }
                let slot = loff + pos * kv_dim;
                state.key_cache[slot..slot + kv_dim].copy_from_slice(&k[kr..kr + kv_dim]);
                state.value_cache[slot..slot + kv_dim].copy_from_slice(&v[kr..kr + kv_dim]);
            }

            // Causal, so row `row` sees positions up to `start + row` and the
            // rows do not interact. The score buffer is reused: only one row
            // is in flight at a time.
            for row in 0..n_tok {
                let pos = start + row;
                let qr = row * dim;
                attention(
                    AttentionInput {
                        q: &q[qr..qr + dim],
                        key_cache: &state.key_cache,
                        value_cache: &state.value_cache,
                    },
                    att,
                    &mut xb[qr..qr + dim],
                    loff,
                    pos,
                    &config,
                );
            }

            if native_q8x4 {
                quantize_rows_q8(q8_batch, xb, q8_quants, q8_scales_f16, dim);
            }
            batched_project(
                &layer.wo,
                w.map(|w| w.wo.as_slice()),
                &*xb,
                q8_quants,
                q8_scales_f16,
                n_tok,
                o,
            );
            for (residual, &add) in x.iter_mut().zip(o.iter()) {
                *residual += add;
            }

            // Feed-forward RMSNorm, then the gated MLP.
            for row in 0..n_tok {
                let at = row * dim;
                rms_norm(
                    &mut xb[at..at + dim],
                    &x[at..at + dim],
                    layer.rms_ffn.data(),
                    config.rms_eps,
                );
            }
            if native_q8x4 {
                quantize_rows_q8(q8_batch, xb, q8_quants, q8_scales_f16, dim);
            }
            batched_project(
                &layer.w1,
                w.map(|w| w.w1.as_slice()),
                &*xb,
                q8_quants,
                q8_scales_f16,
                n_tok,
                gate,
            );
            batched_project(
                &layer.w3,
                w.map(|w| w.w3.as_slice()),
                &*xb,
                q8_quants,
                q8_scales_f16,
                n_tok,
                up,
            );
            for (g, &u) in gate.iter_mut().zip(up.iter()) {
                *g = silu(*g) * u;
            }
            if native_q8x4 {
                quantize_rows_q8(q8_batch, gate, q8_quants, q8_scales_f16, hidden);
            }
            batched_project(
                &layer.w2,
                w.map(|w| w.w2.as_slice()),
                &*gate,
                q8_quants,
                q8_scales_f16,
                n_tok,
                o,
            );
            for (residual, &add) in x.iter_mut().zip(o.iter()) {
                *residual += add;
            }
        }

        // Only the logits after the final prompt token are wanted, so the
        // classifier runs on that one row through the ordinary path. Sending
        // all `n_tok` rows through it would compute 128 copies of the result
        // and hand 127 away, and this way it stays a matvec instead of a
        // matrix product.
        let last = (n_tok - 1) * dim;
        rms_norm(
            &mut xb[last..last + dim],
            &x[last..last + dim],
            rms_final.data(),
            config.rms_eps,
        );
        project(
            classifier,
            &mut state.q8,
            &xb[last..last + dim],
            &mut state.logits,
        );
    }

    /// The storage width the weights were loaded at.
    pub fn storage(&self) -> &'static str {
        self.layers.first().map_or("f32", |layer| layer.wq.kind())
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Process one token at absolute position `pos` and return the logits for
    /// the next token. Positions must be fed in order, since attention reads
    /// the key/value cache written by earlier calls.
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        let config = self.config;
        let kv_dim = config.kv_dim();
        let head_size = config.head_size();
        let seq_len = config.seq_len;

        let token_embedding = &self.token_embedding;
        let rms_final = &self.rms_final;
        let classifier = self.wcls.as_ref().unwrap_or(&self.token_embedding);
        let layers = &self.layers;
        let state = &mut self.state;

        let row = token_embedding.row(token);
        state.x.copy_from_slice(&row);

        for (l, layer) in layers.iter().take(config.n_layers).enumerate() {
            // Attention RMSNorm.
            rms_norm(
                &mut state.xb,
                &state.x,
                layer.rms_att.data(),
                config.rms_eps,
            );

            let loff = l * seq_len * kv_dim;
            let row = loff + pos * kv_dim;

            // q, k and v all read the same normalized activation, so it is
            // quantized once and reused across the three of them.
            quantize_for(&mut state.q8, &layer.wq, &state.xb);
            project_with(&layer.wq, &state.q8, &state.xb, &mut state.buf_q);
            project_with(&layer.wk, &state.q8, &state.xb, &mut state.buf_k);
            project_with(&layer.wv, &state.q8, &state.xb, &mut state.buf_v);

            // RoPE rotates q and k *before* k is cached, matching both
            // references.
            if config.rope_interleaved {
                rope_interleaved(
                    &mut state.buf_q,
                    &mut state.buf_k,
                    pos,
                    head_size,
                    kv_dim,
                    config.rope_theta,
                );
            } else {
                rope_half(&mut state.buf_q, pos, head_size, config.rope_theta);
                rope_half(&mut state.buf_k, pos, head_size, config.rope_theta);
            }
            state.key_cache[row..row + kv_dim].copy_from_slice(&state.buf_k);
            state.value_cache[row..row + kv_dim].copy_from_slice(&state.buf_v);

            // `attention` reads q while writing the attention scores and the
            // output, so the four buffers are split out of the state for the
            // call rather than borrowed through it.
            attention(
                AttentionInput {
                    q: &state.buf_q,
                    key_cache: &state.key_cache,
                    value_cache: &state.value_cache,
                },
                &mut state.att,
                &mut state.xb,
                loff,
                pos,
                &config,
            );

            project(&layer.wo, &mut state.q8, &state.xb, &mut state.buf_o);
            for (x, &add) in state.x.iter_mut().zip(&state.buf_o) {
                *x += add;
            }

            // Feed-forward RMSNorm, then the gated MLP.
            rms_norm(
                &mut state.xb,
                &state.x,
                layer.rms_ffn.data(),
                config.rms_eps,
            );

            quantize_for(&mut state.q8, &layer.w1, &state.xb);
            project_with(&layer.w1, &state.q8, &state.xb, &mut state.buf_gate);
            project_with(&layer.w3, &state.q8, &state.xb, &mut state.buf_up);
            for (g, &u) in state.buf_gate.iter_mut().zip(&state.buf_up) {
                *g = silu(*g) * u;
            }

            project(
                &layer.w2,
                &mut state.q8,
                &state.buf_gate,
                &mut state.buf_mlp,
            );
            for (x, &add) in state.x.iter_mut().zip(&state.buf_mlp) {
                *x += add;
            }
        }

        // Final norm and classifier.
        rms_norm(&mut state.xb, &state.x, rms_final.data(), config.rms_eps);
        project(classifier, &mut state.q8, &state.xb, &mut state.logits);
        &state.logits
    }
}

/// Run `w * x` into `y`, quantizing the activation when the weights are Q8_0.
/// This is the only path the decode loop uses, so a Q8_0 projection always gets
/// a correctly quantized input.
fn project(w: &Matrix, q8: &mut Q8Vec, x: &[f32], y: &mut [f32]) {
    quantize_for(q8, w, x);
    project_with(w, q8, x, y);
}

/// Run `w * x` using an activation already quantized by [`quantize_for`]. Call
/// this for the second and later projection of a shared input.
fn project_with(w: &Matrix, q8: &Q8Vec, x: &[f32], y: &mut [f32]) {
    if matches!(w, Matrix::Q8_0 { .. } | Matrix::Q8_0x4(_)) {
        gemv_q8_0_into(w, q8, y).expect("shapes are fixed at load");
    } else {
        gemv_into(w, x, y).expect("shapes are fixed at load");
    }
}

/// Quantize `x` for `w` when `w` is Q8_0; a no-op for wider storages.
fn quantize_for(q8: &mut Q8Vec, w: &Matrix, x: &[f32]) {
    if matches!(w, Matrix::Q8_0 { .. } | Matrix::Q8_0x4(_)) {
        q8.quantize_into(x);
    }
}

/// Quantize a prompt's activation rows into the contiguous form consumed by
/// the batched Q8_0x4 kernel. Transformer projection widths are block aligned;
/// keeping that invariant here makes every kernel inner loop branch-free.
fn quantize_rows_q8(
    q8: &mut Q8Vec,
    input: &[f32],
    quants: &mut [i8],
    scales_f16: &mut [u16],
    width: usize,
) {
    debug_assert_eq!(width % Q8_0_BLOCK, 0);
    debug_assert_eq!(input.len() % width, 0);
    let blocks = width / Q8_0_BLOCK;
    let rows = input.len() / width;
    for row in 0..rows {
        q8.quantize_into(&input[row * width..(row + 1) * width]);
        quants[row * width..(row + 1) * width].copy_from_slice(&q8.quants[..width]);
        scales_f16[row * blocks..(row + 1) * blocks].copy_from_slice(&q8.scales_f16[..blocks]);
    }
}

/// Dispatch one projection during prefill. The Q8_0x4 branch preserves decode
/// arithmetic exactly while reusing each packed weight block across all prompt
/// rows; wider storages use their pre-expanded GEMM weights.
fn batched_project(
    matrix: &Matrix,
    widened: Option<&[f32]>,
    input: &[f32],
    quants: &[i8],
    scales_f16: &[u16],
    rows: usize,
    out: &mut [f32],
) {
    match matrix {
        Matrix::Q8_0x4(packed) => {
            let in_dim = packed.in_dim();
            let blocks = in_dim / Q8_0_BLOCK;
            gemm_q8_0x4_batched_into(
                packed,
                &quants[..rows * in_dim],
                &scales_f16[..rows * blocks],
                rows,
                out,
            );
        }
        _ => gemm_nt_into(
            input,
            widened.expect("non-Q8 prefill requires widened weights"),
            out,
            rows,
            matrix.in_dim(),
            matrix.out_dim(),
        ),
    }
}

/// How weight matrices are held in memory.
#[derive(Clone, Copy)]
enum Storage {
    F32,
    Bf16,
    Q8_0,
    Q8_0x4,
}

/// Take a raw tensor by name and widen it to f32. Used for the small tensors
/// (norms) that every path consumes as `f32`.
fn take_f32(tensors: &mut HashMap<String, RawTensor>, name: &str) -> Result<Tensor, LoadError> {
    Ok(match take_raw(tensors, name)? {
        RawTensor::F32(t) | RawTensor::F16(t) => t,
        RawTensor::Bf16 { bits, shape } => Tensor::from_vec(
            bits.iter()
                .map(|&b| f32::from_bits(u32::from(b) << 16))
                .collect(),
            &shape,
        )?,
    })
}

/// Take an optional raw tensor by name, for values a tied model omits.
fn take_optional_raw(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
) -> Result<Option<RawTensor>, LoadError> {
    Ok(tensors.remove(name))
}

/// Take a raw tensor by name without widening it.
fn take_raw(tensors: &mut HashMap<String, RawTensor>, name: &str) -> Result<RawTensor, LoadError> {
    tensors
        .remove(name)
        .ok_or_else(|| LoadError::MissingTensor(name.to_string()))
}

/// Take a weight matrix, honouring its stored form so a bf16 file never passes
/// through `f32`.
fn take_matrix_raw(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
    storage: Storage,
) -> Result<Matrix, LoadError> {
    matrix_from_raw(take_raw(tensors, name)?, storage)
}

/// Wrap a raw tensor in the requested storage, widening only when the storage
/// is wider than what the file holds.
fn matrix_from_raw(t: RawTensor, storage: Storage) -> Result<Matrix, LoadError> {
    let shape = t.shape().to_vec();
    if shape.len() != 2 {
        return Err(LoadError::BadConfig(
            "a weight matrix was not rank 2".into(),
        ));
    }
    match (t, storage) {
        // The file is already bf16 and the model wants bf16: hand over the bits
        // directly. This is the case the widening loader made expensive.
        (RawTensor::Bf16 { bits, .. }, Storage::Bf16) => {
            Ok(Matrix::from_bf16_bits(bits, shape[0], shape[1]))
        }
        (RawTensor::Bf16 { bits, .. }, Storage::Q8_0) if shape[1] % Q8_0_BLOCK == 0 => {
            let wide: Vec<f32> = bits
                .iter()
                .map(|&b| f32::from_bits(u32::from(b) << 16))
                .collect();
            let tensor = Tensor::from_vec(wide, &shape)?;
            Ok(Matrix::from_f32_q8_0(tensor).expect("row length is block-aligned"))
        }
        (RawTensor::Bf16 { bits, .. }, Storage::Q8_0x4)
            if shape[1] % Q8_0_BLOCK == 0 && shape[0] % crate::ops::Q8_0X4_ROWS == 0 =>
        {
            let wide: Vec<f32> = bits
                .iter()
                .map(|&b| f32::from_bits(u32::from(b) << 16))
                .collect();
            let tensor = Tensor::from_vec(wide, &shape)?;
            Ok(Matrix::from_f32_q8_0x4(tensor).expect("shape is row and group aligned"))
        }
        (raw, storage) => {
            let tensor = match raw {
                RawTensor::F32(t) | RawTensor::F16(t) => t,
                RawTensor::Bf16 { bits, .. } => Tensor::from_vec(
                    bits.iter()
                        .map(|&b| f32::from_bits(u32::from(b) << 16))
                        .collect(),
                    &shape,
                )?,
            };
            match storage {
                Storage::F32 => Ok(Matrix::from_f32(tensor)),
                Storage::Q8_0x4
                    if shape[1] % Q8_0_BLOCK == 0 && shape[0] % crate::ops::Q8_0X4_ROWS == 0 =>
                {
                    Ok(Matrix::from_f32_q8_0x4(tensor).expect("shape is row and group aligned"))
                }
                // A bf16 model asking for Q8_0 on a non-block-aligned row:
                // narrowing keeps the bytes down rather than widening to f32.
                Storage::Q8_0 | Storage::Q8_0x4 | Storage::Bf16 => {
                    let bits: Vec<u16> = tensor
                        .data()
                        .iter()
                        .map(|v| (v.to_bits() >> 16) as u16)
                        .collect();
                    Ok(Matrix::from_bf16_bits(bits, shape[0], shape[1]))
                }
            }
        }
    }
}

/// Parse a HuggingFace `config.json` into a [`Config`], plus whether the stored
/// weights are bf16.
fn parse_hf_config(path: &Path) -> Result<(Config, bool), LoadError> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let get = |key: &str| -> Result<usize, LoadError> {
        value
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| LoadError::BadConfig(format!("missing integer {key:?}")))
    };
    let n_heads = get("num_attention_heads")?;
    let n_kv_heads = value
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(n_heads);
    let config = Config {
        dim: get("hidden_size")?,
        hidden_dim: get("intermediate_size")?,
        n_layers: get("num_hidden_layers")?,
        n_heads,
        n_kv_heads,
        vocab_size: get("vocab_size")?,
        seq_len: get("max_position_embeddings")?,
        rms_eps: value
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32,
        rope_theta: value
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000.0) as f32,
        rope_interleaved: value
            .get("rope_interleaved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };
    if config.dim % config.n_heads != 0 || config.n_heads % config.n_kv_heads != 0 {
        return Err(LoadError::BadConfig(
            "head counts do not divide the model dimension".into(),
        ));
    }
    let bf16 = value.get("torch_dtype").and_then(|v| v.as_str()) == Some("bfloat16");
    Ok((config, bf16))
}

/// Multi-head causal attention for one position, writing into `state.xb`.
/// The read-only side of attention: the query and the key/value cache history
/// for the current layer.
struct AttentionInput<'a> {
    q: &'a [f32],
    key_cache: &'a [f32],
    value_cache: &'a [f32],
}

fn attention(
    input: AttentionInput<'_>,
    att: &mut [f32],
    xb: &mut [f32],
    loff: usize,
    pos: usize,
    config: &Config,
) {
    let AttentionInput {
        q,
        key_cache,
        value_cache,
    } = input;
    let head_size = config.head_size();
    let kv_dim = config.kv_dim();
    let kv_mul = config.kv_mul();
    let seq_len = config.seq_len;
    let scale = 1.0f32 / (head_size as f32).sqrt();

    for h in 0..config.n_heads {
        let qh = &q[h * head_size..h * head_size + head_size];
        let att = &mut att[h * seq_len..h * seq_len + seq_len];

        for (t, slot) in att.iter_mut().enumerate().take(pos + 1) {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            *slot = dot(qh, &key_cache[at..at + head_size]) * scale;
        }
        softmax_in_place(&mut att[..pos + 1]);

        let xb = &mut xb[h * head_size..h * head_size + head_size];
        xb.fill(0.0);
        for (t, &a) in att.iter().enumerate().take(pos + 1) {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            let vh = &value_cache[at..at + head_size];
            for (o, &v) in xb.iter_mut().zip(vh) {
                *o += a * v;
            }
        }
    }
}

/// Root-mean-square norm, `out = weight * x / sqrt(mean(x^2) + eps)`.
fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
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

/// RoPE that rotates adjacent pairs, as in llama2.c: for even index `i`, the
/// pair `(x[i], x[i+1])` whose frequency depends on `i % head_size`.
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

/// RoPE that rotates the two halves of each head, as in HuggingFace Llama: for
/// `i < head_size/2`, the pair `(x[i], x[i + head_size/2])` with frequency
/// `theta^(-2i/head_size)`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_typical_hf_config() {
        let json = r#"{
            "hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30,
            "num_attention_heads": 9, "num_key_value_heads": 3, "vocab_size": 49152,
            "max_position_embeddings": 8192, "rms_norm_eps": 1e-05,
            "rope_theta": 100000.0, "rope_interleaved": false, "torch_dtype": "bfloat16"
        }"#;
        let dir = std::env::temp_dir().join(format!("kernrt-hf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, json).unwrap();
        let (config, bf16) = parse_hf_config(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(config.dim, 576);
        assert_eq!(config.n_layers, 30);
        assert_eq!(config.n_heads, 9);
        assert_eq!(config.n_kv_heads, 3);
        assert_eq!(config.head_size(), 64);
        assert_eq!(config.kv_dim(), 192);
        assert_eq!(config.kv_mul(), 3);
        assert_eq!(config.rope_theta, 100_000.0);
        assert!(!config.rope_interleaved);
        assert!(bf16);
    }

    /// Both RoPE styles must be rotations: they preserve the norm of the vector.
    #[test]
    fn rope_styles_preserve_norms() {
        let q = vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.9, 0.5, -0.1];
        let sum = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();

        let mut interleaved = q.clone();
        let mut key = vec![0.0; interleaved.len()];
        let len = interleaved.len();
        rope_interleaved(&mut interleaved, &mut key, 3, 4, len, 10_000.0);
        assert!((sum(&interleaved) - sum(&q)).abs() < 1e-5);

        let mut half = q.clone();
        rope_half(&mut half, 3, 4, 10_000.0);
        assert!((sum(&half) - sum(&q)).abs() < 1e-5);
    }
}
