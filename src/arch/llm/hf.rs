//! Loading HuggingFace-format checkpoints: the `config.json` parser, the
//! weight-storage policy, and the helpers that pull tensors out of a
//! safetensors map.
//!
//! Split out of [`super::llama`] so the loader can be shared by every model
//! that reads an HF checkpoint — the Llama-format decoders and NanoJev's
//! decision head alike — without one model depending on another's internals.

use super::llama::Config;
use crate::formats::safetensors::{RawTensor, SafeTensorsError};
use crate::ops::{Matrix, Q8_0_BLOCK};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

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

/// How weight matrices are held in memory.
#[derive(Clone, Copy)]
pub(crate) enum Storage {
    F32,
    Bf16,
    Q8_0,
    Q8_0x4,
}

/// Take a raw tensor by name and widen it to f32. Used for the small tensors
/// (norms) that every path consumes as `f32`.
pub(crate) fn take_f32(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
) -> Result<Tensor, LoadError> {
    Ok(widen(take_raw(tensors, name)?)?)
}

/// Widen a raw tensor to f32.
fn widen(raw: RawTensor) -> Result<Tensor, crate::tensor::TensorError> {
    Ok(match raw {
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
pub(crate) fn take_optional_raw(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
) -> Result<Option<RawTensor>, LoadError> {
    Ok(tensors.remove(name))
}

/// Take an optional f32 tensor, for the norms only some architectures carry
/// (Qwen3's per-head `q_norm`/`k_norm`).
pub(crate) fn take_optional_f32(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
) -> Result<Option<Tensor>, LoadError> {
    match tensors.remove(name) {
        None => Ok(None),
        Some(raw) => Ok(Some(widen(raw)?)),
    }
}

/// Take a raw tensor by name without widening it.
pub(crate) fn take_raw(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
) -> Result<RawTensor, LoadError> {
    tensors
        .remove(name)
        .ok_or_else(|| LoadError::MissingTensor(name.to_string()))
}

/// Take a weight matrix, honouring its stored form so a bf16 file never passes
/// through `f32`.
pub(crate) fn take_matrix_raw(
    tensors: &mut HashMap<String, RawTensor>,
    name: &str,
    storage: Storage,
) -> Result<Matrix, LoadError> {
    matrix_from_raw(take_raw(tensors, name)?, storage)
}

/// Wrap a raw tensor in the requested storage, widening only when the storage
/// is wider than what the file holds.
pub(crate) fn matrix_from_raw(t: RawTensor, storage: Storage) -> Result<Matrix, LoadError> {
    let shape = t.shape().to_vec();
    if shape.len() != 2 {
        return Err(LoadError::BadConfig(
            "a weight matrix was not rank 2".into(),
        ));
    }
    let aligned_q8x4 = shape[1] % Q8_0_BLOCK == 0 && shape[0] % crate::ops::Q8_0X4_ROWS == 0;
    match (t, storage) {
        // The file is already bf16 and the model wants bf16: hand over the bits
        // directly. This is the case the widening loader made expensive.
        (RawTensor::Bf16 { bits, .. }, Storage::Bf16) => {
            Ok(Matrix::from_bf16_bits(bits, shape[0], shape[1]))
        }
        (RawTensor::Bf16 { bits, .. }, Storage::Q8_0) if shape[1] % Q8_0_BLOCK == 0 => {
            let tensor = widen(RawTensor::Bf16 { bits, shape })?;
            Ok(Matrix::from_f32_q8_0(tensor).expect("row length is block-aligned"))
        }
        (RawTensor::Bf16 { bits, .. }, Storage::Q8_0x4) if aligned_q8x4 => {
            let tensor = widen(RawTensor::Bf16 { bits, shape })?;
            Ok(Matrix::from_f32_q8_0x4(tensor).expect("shape is row and group aligned"))
        }
        (raw, storage) => {
            let tensor = widen(raw)?;
            match storage {
                Storage::F32 => Ok(Matrix::from_f32(tensor)),
                Storage::Q8_0x4 if aligned_q8x4 => {
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
///
/// Qwen3-family configs nest the rotary base under `rope_parameters`
/// (`{"rope_theta": ..., "rope_type": ...}`) with no top-level `rope_theta`,
/// so that object is the fallback.
pub(crate) fn parse_hf_config(path: &Path) -> Result<(Config, bool), LoadError> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let get = |key: &str| -> Result<usize, LoadError> {
        value
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| LoadError::BadConfig(format!("missing integer {key:?}")))
    };
    let n_heads = get("num_attention_heads")?;
    let dim = get("hidden_size")?;
    let n_kv_heads = value
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(n_heads);
    let config = Config {
        dim,
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
            .or_else(|| {
                value
                    .get("rope_parameters")
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(10_000.0) as f32,
        rope_interleaved: value
            .get("rope_interleaved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        head_dim: value
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads),
    };
    if config.head_dim == 0 || config.n_heads % config.n_kv_heads != 0 {
        return Err(LoadError::BadConfig(
            "head counts do not divide the model dimension".into(),
        ));
    }
    let bf16 = value.get("torch_dtype").and_then(|v| v.as_str()) == Some("bfloat16");
    Ok((config, bf16))
}

/// LFM2's `intermediate_size`, derived from the block hyperparameters.
///
/// The checkpoint stores `block_ff_dim`, and the MLP reshapes it exactly the
/// way the reference does: an optional `2/3` shrink with a multiplier, then
/// round up to a multiple of `block_multiple_of`. The reference applies this
/// in `Lfm2MLP.__init__`; a loader that skipped it would size every `w1/w2/w3`
/// wrong.
pub(crate) fn lfm2_ff_dim(
    block_ff_dim: usize,
    auto_adjust: bool,
    multiplier: Option<f64>,
    multiple_of: usize,
) -> usize {
    let mut dim = block_ff_dim;
    if auto_adjust {
        dim = dim * 2 / 3;
        if let Some(m) = multiplier {
            dim = (m * dim as f64) as usize;
        }
        dim = multiple_of * dim.div_ceil(multiple_of);
    }
    dim
}

/// Parse an LFM2 `config.json` into a [`Config`](super::llama::Config).
///
/// LFM2 files have no `intermediate_size`: the MLP width comes from
/// [`lfm2_ff_dim`]. There is no `num_key_value_heads`-style fallback problem —
/// the field is always present — but `max_position_embeddings` is 128k, which
/// would make the KV cache gigabytes on its own. It is capped at 8192, the
/// SmolLM2 context length, until the cache pages.
///
/// Returns the config, whether the weights are bf16, the per-layer kind
/// (`true` for attention) and the convolution kernel width.
pub(crate) fn parse_lfm2_config(
    path: &Path,
) -> Result<(Config, bool, Vec<bool>, usize), LoadError> {
    use super::llama::Config;
    let value: serde_json::Value = serde_json::from_reader(fs::File::open(path)?)?;
    let get = |key: &str| -> Result<usize, LoadError> {
        value
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| LoadError::BadConfig(format!("missing integer {key:?}")))
    };
    let dim = get("hidden_size")?;
    let n_heads = get("num_attention_heads")?;
    let n_kv_heads = get("num_key_value_heads")?;
    let n_layers = get("num_hidden_layers")?;
    let full_attn: Vec<usize> = value
        .get("full_attn_idxs")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect()
        })
        .unwrap_or_else(|| (0..n_layers).collect());
    let is_attn: Vec<bool> = (0..n_layers).map(|l| full_attn.contains(&l)).collect();
    // Every layer carries the same SwiGLU MLP, so the width is model-wide.
    let hidden_dim = lfm2_ff_dim(
        get("block_ff_dim")?,
        value
            .get("block_auto_adjust_ff_dim")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        value
            .get("block_ffn_dim_multiplier")
            .and_then(|v| v.as_f64()),
        value
            .get("block_multiple_of")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1),
    );
    let conv_k = value
        .get("conv_L_cache")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(3);
    let config = Config {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size: get("vocab_size")?,
        seq_len: get("max_position_embeddings")?.min(8192),
        rms_eps: value
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32,
        rope_theta: value
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1_000_000.0) as f32,
        // LFM2 rotates the two halves of each head, like HuggingFace Llama.
        rope_interleaved: false,
        head_dim: dim / n_heads,
    };
    if config.head_dim == 0 || config.n_heads % config.n_kv_heads != 0 {
        return Err(LoadError::BadConfig(
            "head counts do not divide the model dimension".into(),
        ));
    }
    let bf16 = value.get("torch_dtype").and_then(|v| v.as_str()) == Some("bfloat16");
    Ok((config, bf16, is_attn, conv_k))
}

#[cfg(test)]
mod lfm2_tests {
    use super::*;

    #[test]
    fn ff_dim_matches_the_checkpoint_shapes() {
        // 350M: block_ff_dim 6656 -> w1 [4608, dim].
        assert_eq!(lfm2_ff_dim(6656, true, Some(1.0), 256), 4608);
        // 700M: block_ff_dim 10240 -> w1 [6912, dim].
        assert_eq!(lfm2_ff_dim(10240, true, Some(1.0), 256), 6912);
        // 1.2B: block_ff_dim 12288 -> w1 [8192, dim].
        assert_eq!(lfm2_ff_dim(12288, true, Some(1.0), 256), 8192);
        // No adjustment passes the width through.
        assert_eq!(lfm2_ff_dim(4096, false, None, 256), 4096);
    }

    #[test]
    fn parses_an_lfm2_config_with_a_layer_mask_and_a_cap() {
        let json = r#"{
            "hidden_size": 1024, "block_ff_dim": 6656,
            "block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256, "num_hidden_layers": 16,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "vocab_size": 65536, "max_position_embeddings": 128000,
            "norm_eps": 1e-05, "rope_theta": 1000000.0,
            "conv_L_cache": 3, "torch_dtype": "bfloat16",
            "full_attn_idxs": [2, 5, 8, 10, 12, 14]
        }"#;
        let dir = std::env::temp_dir().join(format!("kernrt-lfm2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, json).unwrap();
        let (config, bf16, is_attn, conv_k) = parse_lfm2_config(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(config.dim, 1024);
        assert_eq!(config.hidden_dim, 4608);
        assert_eq!(config.n_layers, 16);
        // 128k contexts would make the KV cache gigabytes; the loader caps it.
        assert_eq!(config.seq_len, 8192);
        assert_eq!(config.head_size(), 64);
        assert_eq!(config.kv_dim(), 512);
        assert!(!config.rope_interleaved);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert!(bf16);
        assert_eq!(conv_k, 3);
        assert_eq!(is_attn.len(), 16);
        for l in [2, 5, 8, 10, 12, 14] {
            assert!(is_attn[l], "layer {l} should attend");
        }
        assert!(!is_attn[0], "layer 0 should convolve");
        assert_eq!(is_attn.iter().filter(|&&a| a).count(), 6);
    }
}
