//! Reader for the `llama2.c` checkpoint format used by the TinyStories models.
//!
//! Layout: seven little-endian `i32`s (dim, hidden_dim, n_layers, n_heads,
//! n_kv_heads, vocab_size, seq_len), then every weight as contiguous `f32` in a
//! fixed order. A negative `vocab_size` is the format's signal that the
//! classifier weight is untied from the token embedding (otherwise they are the
//! same tensor, stored once). Between the final norm and the classifier the
//! format keeps two legacy RoPE tables that current inference ignores; they are
//! skipped here but still counted, which is why the expected length lines up
//! exactly with the file.

use crate::ops::Matrix;
use crate::tensor::{Tensor, TensorError};
use std::fmt;
use std::fs;
use std::path::Path;

/// Model hyperparameters, matching `run.c`'s `Config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
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

/// Per-layer weights, each in the row-major `[out, in]` layout `run.c` uses.
pub struct LayerWeights {
    pub rms_att: Tensor,
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub rms_ffn: Tensor,
    pub w1: Matrix,
    pub w2: Matrix,
    pub w3: Matrix,
}

/// A parsed checkpoint.
pub struct Checkpoint {
    pub config: Config,
    pub token_embedding: Matrix,
    pub layers: Vec<LayerWeights>,
    pub rms_final: Tensor,
    /// `Some` only when the classifier is untied; otherwise it is the token
    /// embedding.
    pub wcls: Option<Matrix>,
}

/// Everything that can go wrong reading a checkpoint.
#[derive(Debug)]
pub enum Llama2cError {
    Io(std::io::Error),
    /// The file was shorter than the header, or than its shape implies.
    TooShort {
        needed: u64,
        available: u64,
    },
    /// The weight bytes were not a whole number of `f32`.
    Misaligned,
    /// The header described an impossible model.
    BadConfig(String),
    /// Bytes remained after the last tensor the shape implies.
    Trailing {
        extra: usize,
    },
    Tensor(TensorError),
}

impl fmt::Display for Llama2cError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Llama2cError::Io(e) => write!(f, "i/o error: {e}"),
            Llama2cError::TooShort { needed, available } => {
                write!(f, "checkpoint needs {needed} bytes but has {available}")
            }
            Llama2cError::Misaligned => write!(f, "weight section is not a whole number of f32"),
            Llama2cError::BadConfig(msg) => write!(f, "invalid config: {msg}"),
            Llama2cError::Trailing { extra } => {
                write!(f, "{extra} bytes remained after the last tensor")
            }
            Llama2cError::Tensor(e) => write!(f, "invalid tensor: {e}"),
        }
    }
}

impl std::error::Error for Llama2cError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Llama2cError::Io(e) => Some(e),
            Llama2cError::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Llama2cError {
    fn from(e: std::io::Error) -> Self {
        Llama2cError::Io(e)
    }
}

impl From<TensorError> for Llama2cError {
    fn from(e: TensorError) -> Self {
        Llama2cError::Tensor(e)
    }
}

/// Read a checkpoint from disk.
pub fn load(path: impl AsRef<Path>) -> Result<Checkpoint, Llama2cError> {
    let bytes = fs::read(path)?;
    from_bytes(&bytes)
}

fn from_bytes(bytes: &[u8]) -> Result<Checkpoint, Llama2cError> {
    const HEADER: usize = 7 * 4;
    if bytes.len() < HEADER {
        return Err(Llama2cError::TooShort {
            needed: HEADER as u64,
            available: bytes.len() as u64,
        });
    }
    let mut raw = [0i32; 7];
    for (i, slot) in raw.iter_mut().enumerate() {
        let at = i * 4;
        *slot = i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    }
    let shared = raw[5] > 0;
    let config = Config {
        dim: positive(raw[0], "dim")?,
        hidden_dim: positive(raw[1], "hidden_dim")?,
        n_layers: positive(raw[2], "n_layers")?,
        n_heads: positive(raw[3], "n_heads")?,
        n_kv_heads: positive(raw[4], "n_kv_heads")?,
        vocab_size: positive(raw[5].abs(), "vocab_size")?,
        seq_len: positive(raw[6], "seq_len")?,
    };
    if config.dim % config.n_heads != 0 {
        return Err(Llama2cError::BadConfig(
            "dim is not a multiple of n_heads".into(),
        ));
    }
    if config.n_heads % config.n_kv_heads != 0 {
        return Err(Llama2cError::BadConfig(
            "n_heads is not a multiple of n_kv_heads".into(),
        ));
    }

    let body = &bytes[HEADER..];
    if body.len() % 4 != 0 {
        return Err(Llama2cError::Misaligned);
    }
    let floats: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let d = config.dim;
    let h = config.hidden_dim;
    let l = config.n_layers;
    let v = config.vocab_size;
    let kv = config.kv_dim();
    let mut cursor = Cursor {
        data: &floats,
        at: 0,
    };

    let token_embedding = Matrix::from_f32(cursor.tensor(&[v, d])?);
    // The blocks below are grouped by tensor across all layers, not layer by
    // layer, so each is read whole and then split.
    let rms_att = cursor.block(l * d)?.to_vec();
    let wq = cursor.block(l * d * d)?.to_vec();
    let wk = cursor.block(l * d * kv)?.to_vec();
    let wv = cursor.block(l * d * kv)?.to_vec();
    let wo = cursor.block(l * d * d)?.to_vec();
    let rms_ffn = cursor.block(l * d)?.to_vec();
    let w1 = cursor.block(l * d * h)?.to_vec();
    let w2 = cursor.block(l * h * d)?.to_vec();
    let w3 = cursor.block(l * d * h)?.to_vec();
    let rms_final = cursor.tensor(&[d])?;

    let head_size = config.head_size();
    let rope = config.seq_len * head_size / 2;
    cursor.skip(rope)?; // legacy freq_cis_real
    cursor.skip(rope)?; // legacy freq_cis_imag

    let wcls = if shared {
        None
    } else {
        Some(Matrix::from_f32(cursor.tensor(&[v, d])?))
    };

    if cursor.at != floats.len() {
        return Err(Llama2cError::Trailing {
            extra: (floats.len() - cursor.at) * 4,
        });
    }

    let slice = |src: &[f32], index: usize, per: usize, shape: &[usize]| {
        Tensor::from_vec(src[index * per..(index + 1) * per].to_vec(), shape)
    };
    let mut layers = Vec::with_capacity(l);
    for i in 0..l {
        layers.push(LayerWeights {
            rms_att: slice(&rms_att, i, d, &[d])?,
            wq: Matrix::from_f32(slice(&wq, i, d * d, &[d, d])?),
            wk: Matrix::from_f32(slice(&wk, i, kv * d, &[kv, d])?),
            wv: Matrix::from_f32(slice(&wv, i, kv * d, &[kv, d])?),
            wo: Matrix::from_f32(slice(&wo, i, d * d, &[d, d])?),
            rms_ffn: slice(&rms_ffn, i, d, &[d])?,
            w1: Matrix::from_f32(slice(&w1, i, h * d, &[h, d])?),
            w2: Matrix::from_f32(slice(&w2, i, d * h, &[d, h])?),
            w3: Matrix::from_f32(slice(&w3, i, h * d, &[h, d])?),
        });
    }

    Ok(Checkpoint {
        config,
        token_embedding,
        layers,
        rms_final,
        wcls,
    })
}

fn positive(value: i32, what: &str) -> Result<usize, Llama2cError> {
    if value <= 0 {
        return Err(Llama2cError::BadConfig(format!("{what} must be positive")));
    }
    Ok(value as usize)
}

struct Cursor<'a> {
    data: &'a [f32],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn block(&mut self, count: usize) -> Result<&'a [f32], Llama2cError> {
        let end = self.at.checked_add(count).ok_or(Llama2cError::TooShort {
            needed: u64::MAX,
            available: self.data.len() as u64,
        })?;
        if end > self.data.len() {
            return Err(Llama2cError::TooShort {
                needed: end as u64 * 4,
                available: self.data.len() as u64 * 4,
            });
        }
        let slice = &self.data[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn skip(&mut self, count: usize) -> Result<(), Llama2cError> {
        self.block(count).map(|_| ())
    }

    fn tensor(&mut self, shape: &[usize]) -> Result<Tensor, Llama2cError> {
        let count: usize = shape.iter().product();
        let data = self.block(count)?;
        Ok(Tensor::from_vec(data.to_vec(), shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 16,
            seq_len: 8,
        }
    }

    /// Encode a checkpoint with sequential floats, in the exact format order.
    fn encode(config: &Config, shared: bool) -> Vec<u8> {
        let d = config.dim;
        let h = config.hidden_dim;
        let l = config.n_layers;
        let v = config.vocab_size;
        let kv = config.kv_dim();
        let rope = config.seq_len * config.head_size() / 2;
        let total = v * d
            + l * d
            + l * d * d
            + l * d * kv
            + l * d * kv
            + l * d * d
            + l * d
            + l * d * h
            + l * h * d
            + l * d * h
            + d
            + 2 * rope
            + if shared { 0 } else { v * d };

        let mut out = Vec::new();
        let vocab_field = if shared {
            config.vocab_size as i32
        } else {
            -(config.vocab_size as i32)
        };
        for value in [
            config.dim as i32,
            config.hidden_dim as i32,
            config.n_layers as i32,
            config.n_heads as i32,
            config.n_kv_heads as i32,
            vocab_field,
            config.seq_len as i32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for i in 0..total {
            out.extend_from_slice(&(i as f32).to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_shapes_and_values_in_order() {
        let config = config();
        let checkpoint = from_bytes(&encode(&config, true)).unwrap();
        assert_eq!(checkpoint.config, config);
        assert_eq!(checkpoint.token_embedding.out_dim(), 16);
        assert_eq!(checkpoint.token_embedding.in_dim(), 4);
        assert_eq!(checkpoint.token_embedding.row(0), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(checkpoint.layers.len(), 2);
        assert_eq!(checkpoint.layers[0].wq.out_dim(), 4);
        assert_eq!(checkpoint.layers[0].wq.in_dim(), 4);
        assert_eq!(checkpoint.layers[0].wk.out_dim(), 2);
        assert_eq!(checkpoint.layers[0].wk.in_dim(), 4);
        assert_eq!(checkpoint.layers[0].w1.out_dim(), 8);
        assert_eq!(checkpoint.layers[0].w2.in_dim(), 8);
        assert_eq!(checkpoint.layers[0].w3.out_dim(), 8);
        assert_eq!(checkpoint.rms_final.shape(), &[4]);
    }

    #[test]
    fn shared_classifier_is_absent() {
        assert!(from_bytes(&encode(&config(), true)).unwrap().wcls.is_none());
    }

    #[test]
    fn untied_classifier_is_read() {
        let checkpoint = from_bytes(&encode(&config(), false)).unwrap();
        let wcls = checkpoint.wcls.unwrap();
        assert_eq!(wcls.out_dim(), 16);
        assert_eq!(wcls.in_dim(), 4);
    }

    #[test]
    fn rejects_short_and_trailing_files() {
        let mut bytes = encode(&config(), true);
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(
            from_bytes(&bytes),
            Err(Llama2cError::TooShort { .. })
        ));

        let mut bytes = encode(&config(), true);
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(matches!(
            from_bytes(&bytes),
            Err(Llama2cError::Trailing { .. })
        ));
    }

    #[test]
    fn rejects_impossible_heads() {
        let mut bad = config();
        bad.n_heads = 3; // dim=4 is not divisible by 3
        assert!(matches!(
            from_bytes(&encode(&bad, true)),
            Err(Llama2cError::BadConfig(_))
        ));
    }
}
