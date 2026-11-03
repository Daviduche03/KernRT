//! Llama-2 style transformer forward pass, covering both the llama2.c
//! TinyStories checkpoints and HuggingFace Llama-format models such as SmolLM2.
//!
//! One token at a time with a key/value cache, mirroring `run.c` step for step
//! so the two can be compared numerically. The heavy projections go through
//! [`crate::ops::gemv`]; the small element-wise pieces (RMSNorm, RoPE, softmax,
//! SiLU gating) live here because they are transformer-specific and operate on
//! the state buffers in place.

use super::LanguageModel;
use super::hf::{
    LoadError, Storage, matrix_from_raw, parse_hf_config, take_f32, take_matrix_raw,
    take_optional_f32, take_optional_raw,
};
use crate::backends::{AttnSpec, Backend, Buf, KvSpec, ModelWeights, Proj, RopeSpec, Window};
use crate::formats::llama2c::{Checkpoint, LayerWeights, Llama2cError};
use crate::formats::safetensors::{self, RawTensor};
use crate::ops::Matrix;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    /// Elements per attention head. Stored explicitly rather than derived from
    /// `dim`, because some architectures (Qwen3) set it independently: 0.6B has
    /// `dim = 1024`, 16 heads, but a head width of 128, so `q_proj` is twice as
    /// wide as `dim`.
    pub head_dim: usize,
}

impl Config {
    /// Elements per attention head.
    pub fn head_size(&self) -> usize {
        self.head_dim
    }

    /// Width of the query projection, `n_heads * head_dim`.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Width of one key/value vector, `n_kv_heads * head_dim` (smaller than
    /// `q_dim` under grouped-query attention).
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// How many query heads share each key/value head.
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

struct State {
    x: Buf,
    xb: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    o: Buf,
    gate: Buf,
    up: Buf,
    mlp: Buf,
    /// One row of scratch for the final norm and the classifier.
    hidden: Buf,
    att: Buf,
    logits: Buf,
    /// The only host copies the driver keeps: the last hidden row and the
    /// logits. Everything else stays in backend memory.
    hidden_host: Vec<f32>,
    logits_host: Vec<f32>,
    /// Batched scratch, sized on first use because the prompt length is only
    /// known then.
    prefill: Option<PrefillBuffers>,
}

/// One token in a batched forward, decoupled from the sequence it belongs to.
///
/// A single standalone prompt has `rope_pos == slot == start + i` and
/// `att_start == 0`. A batch of independent sequences keeps RoPE positions
/// restarting per sequence while `slot`/`att_start` carve out disjoint cache
/// ranges, so one pass can score all of them without cross-talk.
///
/// Rows continuing a *shared* prefix also carry `att_shared`: a second, disjoint
/// cache range holding the prefix all of them read ([`Llama::prefill_shared`]).
struct PrefillRow {
    token: usize,
    /// RoPE position, restarting at zero per sequence.
    rope_pos: usize,
    /// Key/value cache slot for this row's key and value.
    slot: usize,
    /// First cache slot of this row's sequence; attention starts here.
    att_start: usize,
    /// Extra cache range also attended, `(start, end)` exclusive. `(0, 0)` for
    /// rows with nothing shared.
    att_shared: (usize, usize),
}

/// Row-major `[n_tok, width]` scratch for a batched forward pass.
///
/// Sized on the first prompt and reused until a longer one arrives. The decode
/// loop keeps its own buffers in [`State`], so filling these cannot disturb a
/// decode run.
#[derive(Clone, Copy)]
struct PrefillBuffers {
    /// Rows the buffers were sized for; requests up to this length reuse them.
    n_tok: usize,
    /// `[n_tok, dim]` residual stream.
    x: Buf,
    /// `[n_tok, dim]` norm output.
    xb: Buf,
    /// `[n_tok, q_dim]` query.
    q: Buf,
    /// `[n_tok, kv_dim]` keys.
    k: Buf,
    /// `[n_tok, kv_dim]` values.
    v: Buf,
    /// `[n_tok, q_dim]` attention output, feeding `o_proj`.
    attn: Buf,
    /// `[n_tok, dim]` projection output.
    o: Buf,
    /// `[n_tok, hidden_dim]` gate, then the gated product in place.
    gate: Buf,
    /// `[n_tok, hidden_dim]` up projection.
    up: Buf,
    /// `[n_heads, seq_len]` attention scores for the row being processed.
    att: Buf,
}

impl PrefillBuffers {
    /// Scratch for `n_tok` rows at the given model shape.
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
            gate: backend.alloc(n_tok * hidden),
            up: backend.alloc(n_tok * hidden),
            att: backend.alloc(n_tok * config.n_heads * config.seq_len),
        }
    }
}

/// A loaded transformer.
pub struct Llama {
    config: Config,
    /// Kernels, memory and per-model prepared weight material.
    backend: Box<dyn Backend>,
    /// The small per-layer norm weights, uploaded once.
    norms: Vec<LayerNorms>,
    rms_final: Buf,
    state: State,
}

/// The norm weights of one layer, resident in backend memory.
struct LayerNorms {
    att: Buf,
    ffn: Buf,
    /// Qwen3's per-head q/k norms; `None` for Llama and SmolLM2.
    q: Option<Buf>,
    k: Option<Buf>,
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
            head_dim: c.dim / c.n_heads,
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
        Self::from_hf_with(dir, crate::backends::from_env())
    }

    /// Load a HuggingFace-format model over an explicit backend.
    ///
    /// The plain [`from_hf`](Self::from_hf) reads `KERNRT_BACKEND`; this is for
    /// a caller that has already resolved one — a manifest entry that pins a
    /// backend per model, or a benchmark that sweeps several.
    pub fn from_hf_with(
        dir: impl AsRef<Path>,
        backend: Box<dyn Backend>,
    ) -> Result<Self, LoadError> {
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
        let tensors = safetensors::load_raw(dir.join("model.safetensors"))?;
        Self::from_raw_tensors_with(config, storage, tensors, backend)
    }

    /// Build a model from an already-parsed config and tensor map.
    ///
    /// The tensor names are the standard `model.layers.N...` set. Checkpoints
    /// that nest the backbone under another prefix (NanoJev's `backbone.`)
    /// rename their keys before calling this.
    pub(crate) fn from_raw_tensors(
        config: Config,
        storage: Storage,
        tensors: HashMap<String, RawTensor>,
    ) -> Result<Self, LoadError> {
        Self::from_raw_tensors_with(config, storage, tensors, crate::backends::from_env())
    }

    /// [`from_raw_tensors`](Self::from_raw_tensors) over an explicit backend.
    pub(crate) fn from_raw_tensors_with(
        config: Config,
        storage: Storage,
        mut tensors: HashMap<String, RawTensor>,
        backend: Box<dyn Backend>,
    ) -> Result<Self, LoadError> {
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
                rms_q: take_optional_f32(&mut tensors, &format!("{p}.self_attn.q_norm.weight"))?,
                rms_k: take_optional_f32(&mut tensors, &format!("{p}.self_attn.k_norm.weight"))?,
            });
        }
        // Tied models store no lm_head.
        let wcls = match take_optional_raw(&mut tensors, "lm_head.weight")? {
            Some(raw) => Some(matrix_from_raw(raw, storage)?),
            None => None,
        };
        Ok(Self::with_backend(
            config,
            ModelWeights {
                token_embedding,
                layers,
                rms_final,
                wcls,
            },
            backend,
        ))
    }

    fn new(
        config: Config,
        token_embedding: Matrix,
        layers: Vec<LayerWeights>,
        rms_final: Tensor,
        wcls: Option<Matrix>,
    ) -> Self {
        Self::with_backend(
            config,
            ModelWeights {
                token_embedding,
                layers,
                rms_final,
                wcls,
            },
            crate::backends::from_env(),
        )
    }

    /// Build a model over an explicit backend.
    ///
    /// The default constructors pick one from `KERNRT_BACKEND`; this is the
    /// entry point for a caller that has already chosen, and for tests that
    /// compare two backends on one checkpoint.
    pub fn with_backend(config: Config, weights: ModelWeights, backend: Box<dyn Backend>) -> Self {
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
            norms.push(LayerNorms {
                att: backend.upload(layer.rms_att.data()),
                ffn: backend.upload(layer.rms_ffn.data()),
                q: layer.rms_q.as_ref().map(|t| backend.upload(t.data())),
                k: layer.rms_k.as_ref().map(|t| backend.upload(t.data())),
            });
        }
        let rms_final = backend.upload(weights.rms_final.data());

        let dim = config.dim;
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
            gate: backend.alloc(config.hidden_dim),
            up: backend.alloc(config.hidden_dim),
            mlp: backend.alloc(dim),
            hidden: backend.alloc(dim),
            att: backend.alloc(config.n_heads * config.seq_len),
            logits: backend.alloc(config.vocab_size),
            hidden_host: vec![0.0; dim],
            logits_host: vec![0.0; config.vocab_size],
            prefill: None,
        };
        Llama {
            config,
            backend,
            norms,
            rms_final,
            state,
        }
    }

    /// Process `tokens` as one batch starting at absolute position `start`,
    /// and return the final-normed hidden state of the last token.
    ///
    /// Same forward pass as [`prefill`](Self::prefill), but exposes the
    /// backbone's last hidden row instead of the next-token logits. Decision
    /// heads (NanoJev) score the end-of-sequence representation, so they need
    /// this rather than logits.
    ///
    /// The slice borrows the prefill scratch (or the decode state on the
    /// fallback path): copy it out before the next call.
    pub fn prefill_hidden(&mut self, tokens: &[usize], start: usize) -> &[f32] {
        let n_tok = tokens.len();
        assert!(
            start + n_tok <= self.config.seq_len,
            "prefill runs past the context length"
        );
        assert!(n_tok > 0, "prefill_hidden needs at least one token");
        let config = self.config;
        let dim = config.dim;
        if !matches!(self.storage(), "f32" | "bf16" | "q8_0x4") {
            for (offset, &token) in tokens.iter().enumerate() {
                self.forward(token, start + offset);
            }
            let state = &mut self.state;
            self.backend.read(state.xb, &mut state.hidden_host[..dim]);
            return &self.state.hidden_host;
        }
        self.prefill_batched(tokens, start);
        let last = (n_tok - 1) * dim;
        let Self {
            backend,
            rms_final,
            state,
            ..
        } = self;
        let pb = *state.prefill.as_ref().expect("prefill_batched builds them");
        backend.copy(pb.x, last, state.hidden, 0, dim);
        backend.rms_norm(state.xb, state.hidden, *rms_final, 1, dim, config.rms_eps);
        backend.read(state.xb, &mut state.hidden_host[..dim]);
        &state.hidden_host
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
            return &self.state.logits_host;
        }
        // Q8_0x4 has an exact batched integer path. Plain Q8_0 remains on the
        // decode loop until it gains the same block-interleaved kernel.
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
    ///
    /// The same arithmetic as [`forward`](Self::forward) stepped token by
    /// token, except that each projection multiplies every row at once instead
    /// of one row at a time, so each weight matrix is read once rather than
    /// once per token. Attention stays per row: its scores are causal, so row
    /// `i` attends to `start..=start + i` and there is nothing to batch.
    fn prefill_batched(&mut self, tokens: &[usize], start: usize) {
        let n_tok = tokens.len();
        if n_tok == 0 {
            return;
        }
        let rows: Vec<PrefillRow> = tokens
            .iter()
            .enumerate()
            .map(|(i, &token)| PrefillRow {
                token,
                rope_pos: start + i,
                slot: start + i,
                att_start: 0,
                att_shared: (0, 0),
            })
            .collect();
        self.prefill_rows(&rows);

        // Only the logits after the final prompt token are wanted, so the
        // classifier runs on that one row: sending all `n_tok` through it would
        // compute `n_tok` copies and hand all but one away.
        let config = self.config;
        let dim = config.dim;
        let last = (n_tok - 1) * dim;
        let Self {
            backend,
            rms_final,
            state,
            ..
        } = self;
        let pb = *state.prefill.as_ref().expect("prefill_rows builds them");
        backend.copy(pb.x, last, state.hidden, 0, dim);
        backend.rms_norm(state.xb, state.hidden, *rms_final, 1, dim, config.rms_eps);
        backend.logits(state.xb, state.logits);
        backend.read(state.logits, &mut state.logits_host);
    }

    /// A batched forward over several independent sequences at once, returning
    /// the final-normed hidden row of each sequence's last token.
    ///
    /// Each sequence gets a disjoint range of key/value cache slots so its
    /// queries cannot see another's keys, while every projection runs as one
    /// `[total_rows, in] x [in, out]` product, so each weight is read once for
    /// the whole batch. This is the shape a decision model needs (many
    /// candidate prompts scored in one pass) and the shape multi-request
    /// serving needs in general.
    ///
    /// RoPE positions restart at zero per sequence; cache slots do not, so
    /// sequence `b` attends only over its own slots.
    ///
    /// `sum(sequences[i].len())` must not exceed the context length.
    pub fn prefill_multi(&mut self, sequences: &[&[usize]]) -> Vec<Vec<f32>> {
        let config = self.config;
        let mut rows = Vec::new();
        let mut last_rows = Vec::with_capacity(sequences.len());
        let mut offset = 0;
        for seq in sequences {
            if seq.is_empty() {
                last_rows.push(None);
                continue;
            }
            assert!(
                offset + seq.len() <= config.seq_len,
                "prefill_multi runs past the context length"
            );
            for (i, &token) in seq.iter().enumerate() {
                rows.push(PrefillRow {
                    token,
                    rope_pos: i,
                    slot: offset + i,
                    att_start: offset,
                    att_shared: (0, 0),
                });
            }
            last_rows.push(Some(rows.len() - 1));
            offset += seq.len();
        }
        self.prefill_rows(&rows);
        self.final_hiddens(&last_rows)
    }

    /// A batched forward over one shared prefix plus a continuation per
    /// sequence, returning each continuation's final-normed hidden row.
    ///
    /// `suffixes[i]` picks up where `prefix` leaves off, so the prefix's rows
    /// are embedded, projected and cached **once** — `prefix.len()` rows in
    /// slots `0..prefix.len()` — and every suffix row reads them back through
    /// its `att_shared` window instead of spelling out its own copy. A request
    /// that runs `sum(prefix.len() + suffixes[i].len())` rows through
    /// [`prefill_multi`](Self::prefill_multi) runs `prefix.len() +
    /// sum(suffixes[i].len())` here instead, and the cache holds the prefix
    /// once rather than once per sequence. For a decision model — where the
    /// state and question dominate and the candidates differ only in their last
    /// few tokens — that is close to half the work.
    ///
    /// The results match `prefill_multi` over the concatenated leaves: prefix
    /// rows compute identically wherever they are stored, and a suffix row
    /// accumulates its scores shared-range first, which is the order a leaf
    /// that repeated the prefix would have produced.
    ///
    /// `prefix.len() + sum(suffixes[i].len())` must not exceed the context
    /// length.
    pub fn prefill_shared(&mut self, prefix: &[usize], suffixes: &[&[usize]]) -> Vec<Vec<f32>> {
        let config = self.config;
        assert!(
            prefix.len() <= config.seq_len,
            "prefill_shared prefix runs past the context length"
        );
        let mut rows: Vec<PrefillRow> = prefix
            .iter()
            .enumerate()
            .map(|(i, &token)| PrefillRow {
                token,
                rope_pos: i,
                slot: i,
                att_start: 0,
                att_shared: (0, 0),
            })
            .collect();
        let mut last_rows = Vec::with_capacity(suffixes.len());
        let mut base = prefix.len();
        for suffix in suffixes {
            if suffix.is_empty() {
                last_rows.push(None);
                continue;
            }
            // `base` accumulates, so the final suffix's check covers the whole
            // request's rows and cache slots.
            assert!(
                base + suffix.len() <= config.seq_len,
                "prefill_shared runs past the context length"
            );
            for (i, &token) in suffix.iter().enumerate() {
                rows.push(PrefillRow {
                    token,
                    rope_pos: prefix.len() + i,
                    slot: base + i,
                    att_start: base,
                    att_shared: (0, prefix.len()),
                });
            }
            last_rows.push(Some(rows.len() - 1));
            base += suffix.len();
        }
        self.prefill_rows(&rows);
        self.final_hiddens(&last_rows)
    }

    /// The final-normed hidden row for each row index a
    /// [`prefill_rows`](Self::prefill_rows) call handed back (`None` yields
    /// zeros, for an empty sequence).
    fn final_hiddens(&mut self, rows: &[Option<usize>]) -> Vec<Vec<f32>> {
        let config = self.config;
        let dim = config.dim;
        let eps = config.rms_eps;
        let pb = *self
            .state
            .prefill
            .as_ref()
            .expect("prefill_rows builds them");
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            match row {
                None => out.push(vec![0.0; dim]),
                Some(row) => {
                    self.backend
                        .copy(pb.x, row * dim, self.state.hidden, 0, dim);
                    self.backend.rms_norm(
                        self.state.xb,
                        self.state.hidden,
                        self.rms_final,
                        1,
                        dim,
                        eps,
                    );
                    let mut hidden = vec![0.0; dim];
                    self.backend.read(self.state.xb, &mut hidden);
                    out.push(hidden);
                }
            }
        }
        out
    }

    /// Run the transformer once over an explicit list of rows.
    ///
    /// Each row carries its token, its RoPE position (restarting per sequence)
    /// and its key/value cache slot (disjoint per sequence), so one pass scores
    /// many independent sequences without them seeing each other.
    fn prefill_rows(&mut self, rows: &[PrefillRow]) {
        let config = self.config;
        let n_tok = rows.len();
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
        let positions: Vec<usize> = rows.iter().map(|r| r.rope_pos).collect();
        let slots: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        let windows: Vec<Window> = rows
            .iter()
            .map(|r| Window {
                att_start: r.att_start,
                slot: r.slot,
                shared: r.att_shared,
            })
            .collect();

        let Self {
            backend,
            norms,
            state,
            ..
        } = self;

        if state.prefill.as_ref().is_none_or(|p| p.n_tok < n_tok) {
            state.prefill = Some(PrefillBuffers::new(backend.as_mut(), n_tok, &config));
        }
        let pb = *state.prefill.as_ref().expect("built above");

        for (row, r) in rows.iter().enumerate() {
            backend.embed(r.token, pb.x, row * dim);
        }

        for (l, n) in norms.iter().enumerate().take(config.n_layers) {
            // Attention RMSNorm. Element-wise, so it is one pass over the rows.
            backend.rms_norm(pb.xb, pb.x, n.att, n_tok, dim, eps);

            backend.project(l, Proj::Q, pb.xb, pb.q, n_tok);
            backend.project(l, Proj::K, pb.xb, pb.k, n_tok);
            backend.project(l, Proj::V, pb.xb, pb.v, n_tok);

            if let Some(w) = n.q {
                backend.qk_norm(pb.q, n_tok, q_dim, head_size, w, eps);
            }
            if let Some(w) = n.k {
                backend.qk_norm(pb.k, n_tok, kv_dim, head_size, w, eps);
            }

            // RoPE rotates q and k before k is cached, matching both
            // references, so it runs over the batched rows before any row
            // reads the cache.
            backend.rope(pb.q, pb.k, q_dim, kv_dim, n_tok, &positions, rope_spec);
            backend.kv_write(l, &slots, pb.k, pb.v, n_tok);

            // Causal within each sequence: row `row` sees its own sequence's
            // slots up to its own position and no other sequence's.
            backend.attention(pb.q, pb.attn, pb.att, l, &windows, attn_spec);

            backend.project(l, Proj::O, pb.attn, pb.o, n_tok);
            backend.add_in_place(pb.x, pb.o, n_tok * dim);

            // Feed-forward RMSNorm, then the gated MLP.
            backend.rms_norm(pb.xb, pb.x, n.ffn, n_tok, dim, eps);
            backend.project(l, Proj::Gate, pb.xb, pb.gate, n_tok);
            backend.project(l, Proj::Up, pb.xb, pb.up, n_tok);
            backend.silu_mul(pb.gate, pb.up, n_tok * hidden);
            backend.project(l, Proj::Down, pb.gate, pb.o, n_tok);
            backend.add_in_place(pb.x, pb.o, n_tok * dim);
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
    /// the next token. Positions must be fed in order, since attention reads
    /// the key/value cache written by earlier calls.
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
            rms_final,
            state,
            ..
        } = self;

        backend.embed(token, state.x, 0);

        for (l, n) in norms.iter().enumerate().take(config.n_layers) {
            backend.rms_norm(state.xb, state.x, n.att, 1, dim, eps);

            backend.project(l, Proj::Q, state.xb, state.q, 1);
            backend.project(l, Proj::K, state.xb, state.k, 1);
            backend.project(l, Proj::V, state.xb, state.v, 1);

            // RoPE rotates q and k *before* k is cached, matching both
            // references.
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
}

impl LanguageModel for Llama {
    fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        Llama::forward(self, token, pos)
    }

    fn context_len(&self) -> usize {
        self.config.seq_len
    }
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

    /// Qwen3 sets `head_dim` independently of `hidden_size / n_heads`, so the
    /// query width is not the model dimension and the explicit value wins.
    #[test]
    fn parses_a_config_with_an_independent_head_dim() {
        let json = r#"{
            "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 151936,
            "max_position_embeddings": 40960, "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0, "head_dim": 128, "torch_dtype": "bfloat16"
        }"#;
        let dir = std::env::temp_dir().join(format!("kernrt-qwen3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, json).unwrap();
        let (config, bf16) = parse_hf_config(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(config.head_size(), 128);
        assert_eq!(config.q_dim(), 2048);
        assert_eq!(config.kv_dim(), 1024);
        assert_eq!(config.kv_mul(), 2);
        assert!(bf16);
    }

    /// Without `head_dim`, the head width still comes from the model dimension
    /// so SmolLM2 and Llama keep their old geometry.
    #[test]
    fn defaults_head_dim_to_the_model_dimension() {
        let json = r#"{
            "hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30,
            "num_attention_heads": 9, "num_key_value_heads": 3, "vocab_size": 49152,
            "max_position_embeddings": 8192
        }"#;
        let dir = std::env::temp_dir().join(format!("kernrt-headdim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, json).unwrap();
        let (config, _) = parse_hf_config(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(config.head_size(), 64);
        assert_eq!(config.q_dim(), 576);
        assert_eq!(config.kv_dim(), 192);
    }

    /// The Qwen3 per-head norm rescales each head to unit RMS times the weight,
    /// independently of the other heads.
    #[test]
    fn qk_norm_normalizes_each_head_separately() {
        use crate::backends::Backend;
        use crate::backends::cpu::Cpu;

        let head_size = 4;
        // Two heads with very different magnitudes.
        let data = vec![3.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1];
        let weight = vec![1.0; head_size];
        let mut backend = Cpu::new();
        let v = backend.upload(&data);
        let w = backend.upload(&weight);
        backend.qk_norm(v, 1, data.len(), head_size, w, 0.0);

        let mut out = vec![0.0; data.len()];
        backend.read(v, &mut out);
        let rms = |h: &[f32]| (h.iter().map(|x| x * x).sum::<f32>() / h.len() as f32).sqrt();
        assert!((rms(&out[..4]) - 1.0).abs() < 1e-5);
        assert!((rms(&out[4..]) - 1.0).abs() < 1e-5);
        // Direction is preserved: `[3, 0, 0, 0]` has RMS 1.5, so index 0
        // becomes 2.0 while the rest stay zero. A whole-vector norm would have
        // mixed the two heads instead.
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!(out[1..4].iter().all(|&x| x == 0.0));
    }

    /// Both RoPE styles must be rotations: they preserve the norm of the vector.
    #[test]
    fn rope_styles_preserve_norms() {
        use crate::backends::Backend;
        use crate::backends::RopeSpec;
        use crate::backends::cpu::Cpu;

        let q = vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.9, 0.5, -0.1];
        let zero = vec![0.0f32; q.len()];
        let sum = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();

        for interleaved in [true, false] {
            let mut backend = Cpu::new();
            let qb = backend.upload(&q);
            let kb = backend.upload(&zero);
            let width = q.len();
            backend.rope(
                qb,
                kb,
                width,
                width,
                1,
                &[3],
                RopeSpec {
                    head_size: 4,
                    theta: 10_000.0,
                    interleaved,
                },
            );
            let mut out = vec![0.0; width];
            backend.read(qb, &mut out);
            assert!(
                (sum(&out) - sum(&q)).abs() < 1e-5,
                "interleaved={interleaved}: norm not preserved"
            );
        }
    }
}
