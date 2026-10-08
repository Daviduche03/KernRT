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
use crate::formats::llama2c::{Checkpoint, LayerWeights, Llama2cError};
use crate::formats::safetensors::{self, RawTensor};
use crate::ops::{
    Matrix, Q8_0_BLOCK, Q8Vec, dot, gemm_nt_into, gemm_q8_0x4_batched_into, gemv_into,
    gemv_q8_0_into, softmax_in_place,
};
use crate::tensor::Tensor;
use std::collections::HashMap;
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
    /// Attention output, `q_dim` wide (wider than `dim` when `head_dim` is set
    /// independently, as in Qwen3). The `o_proj` then maps it back to `dim`.
    buf_attn: Vec<f32>,
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
struct PrefillBuffers {
    /// Rows the buffers were sized for; requests up to this length reuse them.
    n_tok: usize,
    /// `[n_tok, dim]` residual stream.
    x: Vec<f32>,
    /// `[n_tok, dim]` norm output.
    xb: Vec<f32>,
    /// `[n_tok, q_dim]` query (wider than `x` when `head_dim` is independent).
    q: Vec<f32>,
    /// `[n_tok, kv_dim]` rotated keys.
    k: Vec<f32>,
    /// `[n_tok, kv_dim]` values.
    v: Vec<f32>,
    /// `[n_tok, q_dim]` attention output, feeding `o_proj`.
    attn: Vec<f32>,
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
        let q_dim = config.q_dim();
        let hidden = config.hidden_dim;
        Self {
            n_tok,
            x: vec![0.0; n_tok * dim],
            xb: vec![0.0; n_tok * dim],
            q: vec![0.0; n_tok * q_dim],
            k: vec![0.0; n_tok * kv_dim],
            v: vec![0.0; n_tok * kv_dim],
            attn: vec![0.0; n_tok * q_dim],
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
        Self::from_raw_tensors(config, storage, tensors)
    }

    /// Build a model from an already-parsed config and tensor map.
    ///
    /// The tensor names are the standard `model.layers.N...` set. Checkpoints
    /// that nest the backbone under another prefix (NanoJev's `backbone.`)
    /// rename their keys before calling this.
    pub(crate) fn from_raw_tensors(
        config: Config,
        storage: Storage,
        mut tensors: HashMap<String, RawTensor>,
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
            buf_q: vec![0.0; config.q_dim()],
            buf_k: vec![0.0; kv_dim],
            buf_v: vec![0.0; kv_dim],
            buf_attn: vec![0.0; config.q_dim()],
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
        if !matches!(self.storage(), "f32" | "bf16" | "q8_0x4") {
            for (offset, &token) in tokens.iter().enumerate() {
                self.forward(token, start + offset);
            }
            return &self.state.xb;
        }
        self.prefill_batched(tokens, start);
        let dim = self.config.dim;
        let last = (n_tok - 1) * dim;
        let pf = self
            .state
            .prefill
            .as_ref()
            .expect("prefill buffers are built by prefill_batched");
        &pf.xb[last..last + dim]
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
        let n_tok = tokens.len();
        if n_tok == 0 {
            return;
        }
        // A single sequence: every row attends from position 0.
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
        // classifier runs on that one row through the ordinary path. Sending
        // all `n_tok` rows through it would compute `n_tok` copies of the
        // result and hand all but one away, and this way it stays a matvec
        // instead of a matrix product.
        let config = self.config;
        let dim = config.dim;
        let last = (n_tok - 1) * dim;
        let pf = self
            .state
            .prefill
            .as_ref()
            .expect("prefill_rows built the buffers");
        let mut hidden = vec![0.0f32; dim];
        rms_norm(
            &mut hidden,
            &pf.x[last..last + dim],
            self.rms_final.data(),
            config.rms_eps,
        );
        let classifier = self.wcls.as_ref().unwrap_or(&self.token_embedding);
        project(
            classifier,
            &mut self.state.q8,
            &hidden,
            &mut self.state.logits,
        );
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
    fn final_hiddens(&self, rows: &[Option<usize>]) -> Vec<Vec<f32>> {
        let config = self.config;
        let dim = config.dim;
        let pf = self
            .state
            .prefill
            .as_ref()
            .expect("prefill_rows built the buffers");
        rows.iter()
            .copied()
            .map(|row| match row {
                None => vec![0.0; dim],
                Some(row) => {
                    let at = row * dim;
                    let mut hidden = vec![0.0f32; dim];
                    rms_norm(
                        &mut hidden,
                        &pf.x[at..at + dim],
                        self.rms_final.data(),
                        config.rms_eps,
                    );
                    hidden
                }
            })
            .collect()
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
        let seq_len = config.seq_len;

        let token_embedding = &self.token_embedding;
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
            attn,
            o,
            gate,
            up,
            q8_quants,
            q8_scales_f16,
            q8_batch,
            att,
            ..
        } = pf;

        for (row, r) in rows.iter().enumerate() {
            let embedding = token_embedding.row(r.token);
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
            for (row, r) in rows.iter().enumerate() {
                let qr = row * q_dim;
                let kr = row * kv_dim;
                if let Some(rms_q) = &layer.rms_q {
                    qk_norm_per_head(
                        &mut q[qr..qr + q_dim],
                        rms_q.data(),
                        head_size,
                        config.rms_eps,
                    );
                }
                if let Some(rms_k) = &layer.rms_k {
                    qk_norm_per_head(
                        &mut k[kr..kr + kv_dim],
                        rms_k.data(),
                        head_size,
                        config.rms_eps,
                    );
                }
                if config.rope_interleaved {
                    rope_interleaved(
                        &mut q[qr..qr + q_dim],
                        &mut k[kr..kr + kv_dim],
                        r.rope_pos,
                        head_size,
                        kv_dim,
                        config.rope_theta,
                    );
                } else {
                    rope_half(
                        &mut q[qr..qr + q_dim],
                        r.rope_pos,
                        head_size,
                        config.rope_theta,
                    );
                    rope_half(
                        &mut k[kr..kr + kv_dim],
                        r.rope_pos,
                        head_size,
                        config.rope_theta,
                    );
                }
                let at = loff + r.slot * kv_dim;
                state.key_cache[at..at + kv_dim].copy_from_slice(&k[kr..kr + kv_dim]);
                state.value_cache[at..at + kv_dim].copy_from_slice(&v[kr..kr + kv_dim]);
            }

            // Causal within each sequence: row `row` sees its own sequence's
            // slots up to its own position and no other sequence's. The score
            // buffer is reused: only one row is in flight at a time.
            for (row, r) in rows.iter().enumerate() {
                let qr = row * q_dim;
                attention(
                    AttentionInput {
                        q: &q[qr..qr + q_dim],
                        key_cache: &state.key_cache,
                        value_cache: &state.value_cache,
                        att_start: r.att_start,
                        slot: r.slot,
                        att_shared: r.att_shared,
                    },
                    att,
                    &mut attn[qr..qr + q_dim],
                    loff,
                    &config,
                );
            }

            if native_q8x4 {
                quantize_rows_q8(q8_batch, attn, q8_quants, q8_scales_f16, q_dim);
            }
            batched_project(
                &layer.wo,
                w.map(|w| w.wo.as_slice()),
                &*attn,
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
            if let Some(rms_q) = &layer.rms_q {
                qk_norm_per_head(&mut state.buf_q, rms_q.data(), head_size, config.rms_eps);
            }
            if let Some(rms_k) = &layer.rms_k {
                qk_norm_per_head(&mut state.buf_k, rms_k.data(), head_size, config.rms_eps);
            }
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
                    att_start: 0,
                    slot: pos,
                    att_shared: (0, 0),
                },
                &mut state.att,
                &mut state.buf_attn,
                loff,
                &config,
            );

            project(&layer.wo, &mut state.q8, &state.buf_attn, &mut state.buf_o);
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
    let in_dim = matrix.in_dim();
    let out_dim = matrix.out_dim();
    // The scratch buffers are sized for the longest batch seen so far and
    // reused for shorter ones, so only the first `rows` rows are live.
    let out = &mut out[..rows * out_dim];
    match matrix {
        Matrix::Q8_0x4(packed) => {
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
            &input[..rows * in_dim],
            widened.expect("non-Q8 prefill requires widened weights"),
            out,
            rows,
            in_dim,
            out_dim,
        ),
    }
}

/// The read-only side of attention: the query, the key/value cache history for
/// the current layer, and the window of that cache the query may see.
///
/// The window is the row's own slots `att_start..=slot` (both inclusive) plus,
/// for a row continuing a shared prefix, the extra range `att_shared.0..
/// att_shared.1` (exclusive; `(0, 0)` when there is none). A single standalone
/// sequence uses `att_start = 0`; a batched multi-sequence forward gives each
/// sequence a disjoint slot range so its queries cannot see another sequence's
/// keys, and [`Llama::prefill_shared`] keeps the shared prefix in one range of
/// its own.
struct AttentionInput<'a> {
    q: &'a [f32],
    key_cache: &'a [f32],
    value_cache: &'a [f32],
    /// First slot of this row's own range.
    att_start: usize,
    /// Last slot of this row's own range, inclusive.
    slot: usize,
    /// Extra disjoint range also attended, `(start, end)` exclusive.
    att_shared: (usize, usize),
}

/// Multi-head causal attention for one query position, writing into `xb`.
fn attention(
    input: AttentionInput<'_>,
    att: &mut [f32],
    xb: &mut [f32],
    loff: usize,
    config: &Config,
) {
    let AttentionInput {
        q,
        key_cache,
        value_cache,
        att_start,
        slot,
        att_shared,
    } = input;
    let head_size = config.head_size();
    let kv_dim = config.kv_dim();
    let kv_mul = config.kv_mul();
    let seq_len = config.seq_len;
    let scale = 1.0f32 / (head_size as f32).sqrt();
    let (shared_start, shared_end) = att_shared;
    let n_shared = shared_end - shared_start;
    let count = n_shared + slot - att_start + 1;

    for h in 0..config.n_heads {
        let qh = &q[h * head_size..h * head_size + head_size];
        let att = &mut att[h * seq_len..h * seq_len + seq_len];

        // The shared range is scored first and the row's own range second, so
        // a suffix row sees the prefix in exactly the order a leaf that spelled
        // the prefix out would have: sharing changes how often the prefix is
        // computed, not what anyone attends to.
        for (i, t) in (shared_start..shared_end).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            att[i] = dot(qh, &key_cache[at..at + head_size]) * scale;
        }
        for (i, t) in (att_start..=slot).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            att[n_shared + i] = dot(qh, &key_cache[at..at + head_size]) * scale;
        }
        softmax_in_place(&mut att[..count]);

        let xb = &mut xb[h * head_size..h * head_size + head_size];
        xb.fill(0.0);
        for (i, t) in (shared_start..shared_end).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            let a = att[i];
            for (o, &v) in xb.iter_mut().zip(&value_cache[at..at + head_size]) {
                *o += a * v;
            }
        }
        for (i, t) in (att_start..=slot).enumerate() {
            let at = loff + t * kv_dim + (h / kv_mul) * head_size;
            let a = att[n_shared + i];
            for (o, &v) in xb.iter_mut().zip(&value_cache[at..at + head_size]) {
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

/// Qwen3 normalizes every attention head independently, so the norm spans one
/// `head_size` slice and the weight is applied within that slice rather than
/// across the whole `dim` vector.
fn qk_norm_per_head(vec: &mut [f32], weight: &[f32], head_size: usize, eps: f32) {
    debug_assert_eq!(weight.len(), head_size);
    for head in vec.chunks_exact_mut(head_size) {
        let mut ss = 0.0f32;
        for &v in head.iter() {
            ss += v * v;
        }
        ss /= head_size as f32;
        ss += eps;
        let scale = 1.0 / ss.sqrt();
        for (o, &w) in head.iter_mut().zip(weight) {
            *o = w * (scale * *o);
        }
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

impl LanguageModel for Llama {
    fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        Llama::forward(self, token, pos)
    }

    fn context_len(&self) -> usize {
        self.config.seq_len
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
        let head_size = 4;
        // Two heads with very different magnitudes.
        let mut v = vec![3.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1];
        let weight = vec![1.0; head_size];
        qk_norm_per_head(&mut v, &weight, head_size, 0.0);

        let rms = |h: &[f32]| (h.iter().map(|x| x * x).sum::<f32>() / h.len() as f32).sqrt();
        assert!((rms(&v[..4]) - 1.0).abs() < 1e-5);
        assert!((rms(&v[4..]) - 1.0).abs() < 1e-5);
        // Direction is preserved: `[3, 0, 0, 0]` has RMS 1.5, so index 0
        // becomes 2.0 while the rest stay zero. A whole-vector norm would have
        // mixed the two heads instead.
        assert!((v[0] - 2.0).abs() < 1e-5);
        assert!(v[1..4].iter().all(|&x| x == 0.0));
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
