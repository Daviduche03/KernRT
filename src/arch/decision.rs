//! NanoJev-style decision models: one backbone forward per candidate, zero
//! autoregressive decoding.
//!
//! A question plus its candidates are formatted exactly like the reference
//! `predict_toy_decisions.py`: each candidate becomes a leaf sequence
//! `prefix + "Candidate:\n{text}\nDecision:" + [eos]`, the backbone runs over
//! every leaf, and the head scores the end-of-sequence (EOS position) hidden
//! state. The prefix is identical for all candidates, so it is computed once
//! and each leaf continues from it ([`Llama::prefill_shared`]).
//! `boolean` questions use the scalar path only (`[0, z]`), `score`
//! questions use the scalar path with one leaf per level, and `choice`
//! questions add the attention-based set head over the candidate rows.

use super::llm::hf::{LoadError, Storage, parse_hf_config, take_f32};
use super::llm::llama::Llama;
use crate::formats::safetensors;
use crate::ops::softmax_in_place;
use crate::tokenizer::BpeTokenizer;
use std::collections::HashMap;
use std::path::Path;

/// Width of the set-head token space (`set_project` output).
const SET_DIM: usize = 128;
/// Attention heads in the set head (`MultiheadAttention(128, 4)`).
const SET_HEADS: usize = 4;

/// The question kind, matching the reference `type` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Boolean,
    Choice,
    Score,
}

impl QuestionType {
    fn label(self) -> &'static str {
        match self {
            QuestionType::Boolean => "boolean",
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
        }
    }
}

/// The trained head: LayerNorm + scalar path, plus the optional set path.
struct Head {
    hidden: usize,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    scalar_w: Vec<f32>,
    scalar_b: f32,
    set_head: bool,
    /// `[SET_DIM, hidden + 1]` row-major (`set_project.weight`).
    set_project_w: Vec<f32>,
    set_project_b: Vec<f32>,
    /// `[3 * SET_DIM, SET_DIM]` row-major q/k/v blocks (`in_proj_weight`).
    in_proj_w: Vec<f32>,
    in_proj_b: Vec<f32>,
    /// `[SET_DIM, SET_DIM]` row-major (`out_proj.weight`).
    out_proj_w: Vec<f32>,
    out_proj_b: Vec<f32>,
    set_output_w: Vec<f32>,
    set_output_b: f32,
}

/// A loaded NanoJev checkpoint: Qwen3 backbone plus decision head.
pub struct DecisionModel {
    backbone: Llama,
    head: Head,
    eos: usize,
}

impl DecisionModel {
    /// Load `backbone_config/config.json` + `best.safetensors` from `dir`.
    ///
    /// `seq_len` caps the context (and therefore the key/value cache); the
    /// checkpoint declares 40960 positions, whose full cache would be ~9 GB.
    /// Decision leaves are a few hundred tokens, so a few thousand is plenty.
    /// Weight storage honours the same `KERNRT_Q8_0*` variables as
    /// [`Llama::from_hf`]; the default keeps the checkpoint's f32.
    pub fn from_checkpoint(dir: impl AsRef<Path>, seq_len: usize) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        if seq_len == 0 {
            return Err(LoadError::BadConfig("seq_len must be positive".into()));
        }
        let (mut config, _bf16) = parse_hf_config(&dir.join("backbone_config/config.json"))?;
        config.seq_len = config.seq_len.min(seq_len);
        let storage = if std::env::var_os("KERNRT_Q8_0X4").is_some() {
            Storage::Q8_0x4
        } else if std::env::var_os("KERNRT_Q8_0").is_some() {
            Storage::Q8_0
        } else {
            Storage::F32
        };

        // The backbone tensors live under `backbone.` instead of `model.`;
        // everything else is the decision head.
        let mut backbone_tensors = HashMap::new();
        let mut head_tensors = HashMap::new();
        for (name, tensor) in safetensors::load_raw(dir.join("best.safetensors"))? {
            if let Some(rest) = name.strip_prefix("backbone.") {
                backbone_tensors.insert(format!("model.{rest}"), tensor);
            } else {
                head_tensors.insert(name, tensor);
            }
        }
        let backbone = Llama::from_raw_tensors(config, storage, backbone_tensors)?;
        let head = Head::load(&mut head_tensors, config.dim)?;
        if !head_tensors.is_empty() {
            let mut leftover: Vec<_> = head_tensors.keys().cloned().collect();
            leftover.sort();
            return Err(LoadError::BadConfig(format!(
                "unconsumed head tensors: {}",
                leftover.join(", ")
            )));
        }

        let eos = read_eos(&dir.join("backbone_config/config.json"))?;
        Ok(Self {
            backbone,
            head,
            eos,
        })
    }

    /// End-of-sequence id the leaves are terminated with.
    pub fn eos(&self) -> usize {
        self.eos
    }

    /// Access the underlying backbone, for profiling and verification.
    pub fn backbone_mut(&mut self) -> &mut Llama {
        &mut self.backbone
    }

    /// The backbone's final-normed hidden state at the last token of `leaf`.
    ///
    /// Exposed for verification against the reference implementation; the
    /// decision path calls this internally.
    pub fn leaf_hidden(&mut self, leaf: &[usize], _tok: &BpeTokenizer) -> Vec<f32> {
        self.backbone.prefill_hidden(leaf, 0).to_vec()
    }

    /// Head internals for debugging: the scalar path and the set correction
    /// for a batch of hidden rows.
    pub fn head_debug(&self, hidden: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let head = &self.head;
        let normed: Vec<Vec<f32>> = hidden
            .iter()
            .map(|h| layer_norm(h, &head.norm_w, &head.norm_b))
            .collect();
        let z: Vec<f32> = normed
            .iter()
            .map(|h| dot(&head.scalar_w, h) + head.scalar_b)
            .collect();
        let delta = head.set_delta(&normed);
        (z, delta)
    }

    /// Score one question and return a probability per candidate.
    ///
    /// `criteria` is `(id, description)` pairs in order: choice candidates
    /// format as `"{id}: {description}"`, score levels pass their labels as
    /// descriptions, and boolean questions optionally carry
    /// `("false"|"true", description)` rows for the prompt. Mirrors the
    /// reference `prepare_examples` text exactly.
    pub fn decide(
        &mut self,
        state: &str,
        qtype: QuestionType,
        instructions: &str,
        criteria: &[(&str, &str)],
        tok: &BpeTokenizer,
    ) -> Result<Vec<f32>, LoadError> {
        let texts: Vec<String> = match qtype {
            QuestionType::Boolean => vec!["The proposition is true.".to_string()],
            QuestionType::Choice => criteria
                .iter()
                .map(|(id, desc)| format!("{id}: {desc}"))
                .collect(),
            QuestionType::Score => criteria.iter().map(|(_, desc)| desc.to_string()).collect(),
        };
        if texts.len() < 2 || texts.len() > 255 {
            return Err(LoadError::BadConfig(format!(
                "need 2-255 candidates, have {}",
                texts.len()
            )));
        }
        let mut prompt = format!(
            "Question type: {}\nQuestion:\n{instructions}\n",
            qtype.label()
        );
        if qtype == QuestionType::Boolean {
            for (key, label) in [("false", "False"), ("true", "True")] {
                if let Some((_, desc)) = criteria.iter().find(|(id, _)| *id == key) {
                    prompt.push_str(&format!("{label} criterion: {desc}\n"));
                }
            }
        }
        let mut prefix = tok.encode(&format!("State:\n{state}\n"), false);
        prefix.extend(tok.encode(&prompt, false));

        // All leaves are scored in one batched forward; the EOS position of
        // each carries the verdict. Every weight matrix is read once for the
        // whole candidate set, and the state+question prefix — byte-identical
        // across candidates — is prefilled once rather than once per leaf,
        // each candidate's short suffix continuing from the cached rows.
        let mut suffixes = Vec::with_capacity(texts.len());
        for text in &texts {
            let mut suffix = tok.encode(&format!("Candidate:\n{text}\nDecision:"), false);
            suffix.push(self.eos);
            suffixes.push(suffix);
        }
        let total: usize = prefix.len() + suffixes.iter().map(Vec::len).sum::<usize>();
        if total > self.backbone.config().seq_len {
            return Err(LoadError::BadConfig(format!(
                "{total} leaf tokens exceed the {} context",
                self.backbone.config().seq_len
            )));
        }
        let refs: Vec<&[usize]> = suffixes.iter().map(Vec::as_slice).collect();
        let hidden = self.backbone.prefill_shared(&prefix, &refs);

        let head = &self.head;
        let mut z: Vec<f32> = hidden
            .iter()
            .map(|h| {
                dot(&head.scalar_w, &layer_norm(h, &head.norm_w, &head.norm_b)) + head.scalar_b
            })
            .collect();
        // Booleans never touch the set head: `[0, z]` straight away.
        if qtype == QuestionType::Boolean {
            z.truncate(1);
            let mut logits = vec![0.0, z[0]];
            softmax_in_place(&mut logits);
            return Ok(logits);
        }
        if head.set_head && qtype == QuestionType::Choice {
            let delta = head.set_delta(
                &hidden
                    .iter()
                    .map(|h| layer_norm(h, &head.norm_w, &head.norm_b))
                    .collect::<Vec<_>>(),
            );
            for (z, d) in z.iter_mut().zip(&delta) {
                *z += *d;
            }
        }
        softmax_in_place(&mut z);
        Ok(z)
    }
}

/// Read `eos_token_id` from a model config: it is the authority, since
/// `tokenizer.json` carries no top-level `eos_token` field.
fn read_eos(path: &Path) -> Result<usize, LoadError> {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path)?).map_err(LoadError::Json)?;
    value
        .get("eos_token_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| LoadError::BadConfig("config lacks eos_token_id".into()))
}

impl Head {
    fn load(
        tensors: &mut HashMap<String, crate::formats::safetensors::RawTensor>,
        hidden: usize,
    ) -> Result<Self, LoadError> {
        let norm_w = take_vec(tensors, "norm.weight", &[hidden])?;
        let norm_b = take_vec(tensors, "norm.bias", &[hidden])?;
        let scalar_w = take_vec(tensors, "scalar.weight", &[1, hidden])?;
        let scalar_b = take_vec(tensors, "scalar.bias", &[1])?[0];
        let set_head = tensors.contains_key("set_project.weight");
        let (
            set_project_w,
            set_project_b,
            in_proj_w,
            in_proj_b,
            out_proj_w,
            out_proj_b,
            set_output_w,
            set_output_b,
        ) = if set_head {
            (
                take_vec(tensors, "set_project.weight", &[SET_DIM, hidden + 1])?,
                take_vec(tensors, "set_project.bias", &[SET_DIM])?,
                take_vec(
                    tensors,
                    "set_attention.in_proj_weight",
                    &[3 * SET_DIM, SET_DIM],
                )?,
                take_vec(tensors, "set_attention.in_proj_bias", &[3 * SET_DIM])?,
                take_vec(
                    tensors,
                    "set_attention.out_proj.weight",
                    &[SET_DIM, SET_DIM],
                )?,
                take_vec(tensors, "set_attention.out_proj.bias", &[SET_DIM])?,
                take_vec(tensors, "set_output.weight", &[1, SET_DIM])?,
                take_vec(tensors, "set_output.bias", &[1])?[0],
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                0.0,
            )
        };
        Ok(Self {
            hidden,
            norm_w,
            norm_b,
            scalar_w,
            scalar_b,
            set_head,
            set_project_w,
            set_project_b,
            in_proj_w,
            in_proj_b,
            out_proj_w,
            out_proj_b,
            set_output_w,
            set_output_b,
        })
    }

    /// The set-path correction: project each candidate row to 128-wide,
    /// self-attend over the candidates, and project back to one scalar each.
    /// Mirrors `DecisionModel.forward`'s `set_project`/`set_attention`/
    /// `set_output` block with all rows valid (one question at a time).
    fn set_delta(&self, h: &[Vec<f32>]) -> Vec<f32> {
        let k = h.len();
        let log_k = (k as f32).ln();
        // u = set_project([h, log k]).
        let mut u = vec![0.0f32; k * SET_DIM];
        for (i, row) in h.iter().enumerate() {
            for o in 0..SET_DIM {
                let base = o * (self.hidden + 1);
                let mut acc =
                    self.set_project_b[o] + self.set_project_w[base + self.hidden] * log_k;
                for (j, &x) in row.iter().enumerate() {
                    acc += self.set_project_w[base + j] * x;
                }
                u[i * SET_DIM + o] = acc;
            }
        }
        // q/k/v from the packed in_proj rows, then 4-head attention.
        let (qw, kw, vw) = (
            &self.in_proj_w[..SET_DIM * SET_DIM],
            &self.in_proj_w[SET_DIM * SET_DIM..2 * SET_DIM * SET_DIM],
            &self.in_proj_w[2 * SET_DIM * SET_DIM..],
        );
        let (qb, kb, vb) = (
            &self.in_proj_b[..SET_DIM],
            &self.in_proj_b[SET_DIM..2 * SET_DIM],
            &self.in_proj_b[2 * SET_DIM..],
        );
        let mut q = vec![0.0f32; k * SET_DIM];
        let mut kk = vec![0.0f32; k * SET_DIM];
        let mut v = vec![0.0f32; k * SET_DIM];
        for i in 0..k {
            matvec_add(
                &mut q[i * SET_DIM..(i + 1) * SET_DIM],
                qw,
                &u[i * SET_DIM..(i + 1) * SET_DIM],
                qb,
            );
            matvec_add(
                &mut kk[i * SET_DIM..(i + 1) * SET_DIM],
                kw,
                &u[i * SET_DIM..(i + 1) * SET_DIM],
                kb,
            );
            matvec_add(
                &mut v[i * SET_DIM..(i + 1) * SET_DIM],
                vw,
                &u[i * SET_DIM..(i + 1) * SET_DIM],
                vb,
            );
        }
        let head_dim = SET_DIM / SET_HEADS;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; k * SET_DIM];
        let mut scores = vec![0.0f32; k];
        for head in 0..SET_HEADS {
            for i in 0..k {
                for (j, s) in scores.iter_mut().enumerate().take(k) {
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q[i * SET_DIM + head * head_dim + d]
                            * kk[j * SET_DIM + head * head_dim + d];
                    }
                    *s = acc * scale;
                }
                softmax_in_place(&mut scores[..k]);
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate().take(k) {
                        acc += p * v[j * SET_DIM + head * head_dim + d];
                    }
                    out[i * SET_DIM + head * head_dim + d] = acc;
                }
            }
        }
        // delta = set_output(tanh(u + out_proj(out))).
        let mut delta = vec![0.0f32; k];
        let mut mixed = vec![0.0f32; SET_DIM];
        for i in 0..k {
            matvec_add(
                &mut mixed,
                &self.out_proj_w,
                &out[i * SET_DIM..(i + 1) * SET_DIM],
                &self.out_proj_b,
            );
            let mut acc = self.set_output_b;
            for d in 0..SET_DIM {
                acc += self.set_output_w[d] * (u[i * SET_DIM + d] + mixed[d]).tanh();
            }
            delta[i] = acc;
        }
        delta
    }
}

/// Take a head tensor by name, checking its shape, as owned f32.
fn take_vec(
    tensors: &mut HashMap<String, crate::formats::safetensors::RawTensor>,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, LoadError> {
    let tensor = take_f32(tensors, name)?;
    if tensor.shape() != expected {
        return Err(LoadError::BadConfig(format!(
            "{name:?} has shape {:?}, expected {expected:?}",
            tensor.shape()
        )));
    }
    Ok(tensor.data().to_vec())
}

/// `nn.LayerNorm` with bias, eps 1e-5 (the PyTorch default).
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .zip(weight)
        .zip(bias)
        .map(|((v, w), b)| (v - mean) * inv * w + b)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y = W x + b` for row-major `[y.len(), x.len()]` `w`.
fn matvec_add(y: &mut [f32], w: &[f32], x: &[f32], b: &[f32]) {
    for (i, (y, b)) in y.iter_mut().zip(b).enumerate() {
        *y = dot(&w[i * x.len()..(i + 1) * x.len()], x) + b;
    }
}
